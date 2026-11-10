"""Heterogen transformerni o'qitish (JAX), bazaviy modellar bilan solishtirish va vaznlarni eksport qilish.

Vaqt bo'yicha bo'lish: o'qitish 2020-2023, validatsiya 2024, test 2025.
Natijalar: model/model.npz, model/model_meta.json, model/natijalar.json
"""
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "app"))
import engine as E  # noqa: E402

np.random.seed(0)
D, HEADS, LAYERS, FF, DROP = 64, 4, 2, 128, 0.2


def init_params(vocab_size, key):
    ks = iter(jax.random.split(key, 64))
    p = {}

    def lin(name, i, o):
        lim = 1.0 / np.sqrt(i)
        p[name + ".weight"] = jax.random.uniform(next(ks), (o, i), minval=-lim, maxval=lim)
        p[name + ".bias"] = jnp.zeros((o,))

    def ln(name):
        p[name + ".weight"] = jnp.ones((D,)); p[name + ".bias"] = jnp.zeros((D,))

    p["cls"] = jax.random.normal(next(ks), (1, 1, D)) * 0.02
    lin("static_proj", E.STATIC_DIM, D); lin("week_proj", E.WEEK_DIM, D)
    p["tok_emb.weight"] = jax.random.normal(next(ks), (vocab_size, D)) * 0.1
    p["week_pos.weight"] = jax.random.normal(next(ks), (E.WEEKS, D)) * 0.1
    p["text_pos.weight"] = jax.random.normal(next(ks), (E.MAX_TEXT, D)) * 0.1
    p["type_emb.weight"] = jax.random.normal(next(ks), (4, D)) * 0.1
    for i in range(LAYERS):
        pre = f"layers.{i}."
        ln(pre + "ln1"); ln(pre + "ln2")
        for n in ("q", "k", "v", "o"):
            lin(pre + n, D, D)
        lin(pre + "ff1", D, FF); lin(pre + "ff2", FF, D)
    ln("ln_f")
    lin("risk_head", D, 1); lin("score_head", D, 1)
    lin("topic_head", D, len(E.TOPICS)); lin("sent_head", D, 3)
    return p


def _lin(p, x, name):
    return x @ p[name + ".weight"].T + p[name + ".bias"]


def _ln(p, x, name):
    mu = x.mean(-1, keepdims=True); var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / jnp.sqrt(var + 1e-5) * p[name + ".weight"] + p[name + ".bias"]


def _drop(x, key, train):
    if not train:
        return x
    keep = jax.random.bernoulli(key, 1 - DROP, x.shape)
    return jnp.where(keep, x / (1 - DROP), 0.0)


def forward(p, st, wk, wm, ti, tm, key, train):
    n = st.shape[0]
    te = p["type_emb.weight"]
    cls = jnp.broadcast_to(p["cls"], (n, 1, D)) + te[0]
    s = _lin(p, st, "static_proj")[:, None] + te[1]
    w = _lin(p, wk, "week_proj") + p["week_pos.weight"][None] + te[2]
    t = p["tok_emb.weight"][ti] + p["text_pos.weight"][None] + te[3]
    x = jnp.concatenate([cls, s, w, t], 1)
    mask = jnp.concatenate([jnp.ones((n, 2)), wm, tm], 1)
    bias = (1 - mask)[:, None, None, :] * -1e9
    L = x.shape[1]; dh = D // HEADS
    keys = jax.random.split(key, LAYERS * 3)
    for i in range(LAYERS):
        pre = f"layers.{i}."
        y = _ln(p, x, pre + "ln1")
        q = _lin(p, y, pre + "q").reshape(n, L, HEADS, dh).transpose(0, 2, 1, 3)
        k = _lin(p, y, pre + "k").reshape(n, L, HEADS, dh).transpose(0, 2, 1, 3)
        v = _lin(p, y, pre + "v").reshape(n, L, HEADS, dh).transpose(0, 2, 1, 3)
        a = jax.nn.softmax(q @ k.transpose(0, 1, 3, 2) / np.sqrt(dh) + bias, -1)
        a = _drop(a, keys[3 * i], train)
        o = (a @ v).transpose(0, 2, 1, 3).reshape(n, L, D)
        x = x + _drop(_lin(p, o, pre + "o"), keys[3 * i + 1], train)
        y = _ln(p, x, pre + "ln2")
        h = jax.nn.gelu(_lin(p, y, pre + "ff1"), approximate=True)
        x = x + _drop(_lin(p, h, pre + "ff2"), keys[3 * i + 2], train)
    x = _ln(p, x, "ln_f")
    c = x[:, 0]
    tx = x[:, 2 + E.WEEKS:]
    pooled = (tx * tm[..., None]).sum(1) / jnp.maximum(tm.sum(1, keepdims=True), 1.0)
    return (_lin(p, c, "risk_head")[:, 0], _lin(p, c, "score_head")[:, 0],
            _lin(p, pooled, "topic_head"), _lin(p, pooled, "sent_head"))


def loss_fn(p, batch, key):
    st, wk, wm, ti, tm, y, sc, tp, se = batch
    r, s_, t_, sn = forward(p, st, wk, wm, ti, tm, key, True)
    ht = (tm.sum(1) > 0).astype(jnp.float32)
    l_risk = optax.sigmoid_binary_cross_entropy(r, y).mean()
    l_score = ((s_ - sc) ** 2).mean()
    l_topic = (optax.sigmoid_binary_cross_entropy(t_, tp).mean(1) * ht).sum() / jnp.maximum(ht.sum(), 1)
    l_sent = (optax.softmax_cross_entropy_with_integer_labels(sn, se) * ht).sum() / jnp.maximum(ht.sum(), 1)
    return l_risk + 0.5 * l_score + 0.5 * l_topic + 0.5 * l_sent


def metrics(y, p, thr=0.5):
    yh = (p >= thr).astype(int)
    return dict(accuracy=accuracy_score(y, yh), precision=precision_score(y, yh, zero_division=0),
                recall=recall_score(y, yh), f1=f1_score(y, yh), auc=roc_auc_score(y, p))


def main():
    st = pd.read_csv(os.path.join(ROOT, "data", "talabalar_full.csv"))
    wk = pd.read_csv(os.path.join(ROOT, "data", "haftalik_full.csv"))
    st["fikr"] = st["fikr"].fillna("")
    tr = st[st.yil <= 2023].reset_index(drop=True)
    va = st[st.yil == 2024].reset_index(drop=True)
    te = st[st.yil == 2025].reset_index(drop=True)
    vocab = E.Vocab.build(tr["fikr"])
    print("Lug'at hajmi:", len(vocab.itos), "| o'qitish/val/test:", len(tr), len(va), len(te))

    def labels(df):
        y = df.past_ozlashtirish.values.astype(np.float32)
        sc = (df.yakuniy_ball.values / 100).astype(np.float32)
        tp = np.zeros((len(df), len(E.TOPICS)), np.float32)
        for i, s in enumerate(df.mavzular.fillna("")):
            for t in str(s).split(";"):
                if t in E.TOPICS:
                    tp[i, E.TOPICS.index(t)] = 1
        se = df.kayfiyat.map({k: i for i, k in enumerate(E.SENTIMENTS)}).values.astype(np.int64)
        return y, sc, tp, se

    Xtr = E.build_arrays(tr, wk, vocab); Ytr = labels(tr)
    va_full = [jnp.asarray(x) for x in E.build_arrays(va, wk, vocab)]

    params = init_params(len(vocab.itos), jax.random.PRNGKey(0))
    n = len(tr); bs = 128; epochs = 40
    steps = epochs * (n // bs)
    sched = optax.warmup_cosine_decay_schedule(0.0, 2e-3, steps // 10, steps, 1e-5)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=1e-2))
    opt_state = opt.init(params)

    @jax.jit
    def step(params, opt_state, batch, key):
        loss, g = jax.value_and_grad(loss_fn)(params, batch, key)
        upd, opt_state = opt.update(g, opt_state, params)
        return optax.apply_updates(params, upd), opt_state, loss

    @jax.jit
    def predict_risk(params, st, wk_, wm, ti, tm):
        return jax.nn.sigmoid(forward(params, st, wk_, wm, ti, tm, jax.random.PRNGKey(0), False)[0])

    @jax.jit
    def predict_topics(params, st, wk_, wm, ti, tm):
        return jax.nn.sigmoid(forward(params, st, wk_, wm, ti, tm, jax.random.PRNGKey(0), False)[2])

    _, _, va_tp, _ = labels(va)
    va_ht = va_full[4].sum(1) > 0

    rng = np.random.default_rng(0)
    key = jax.random.PRNGKey(1)
    best, best_params = -1, None
    t0 = time.time()
    for ep in range(epochs):
        perm = rng.permutation(n)
        tot = 0
        for s in range(0, n - bs + 1, bs):
            idx = perm[s:s + bs]
            st_, wk_, wm_, ti_, tm_ = [x[idx].copy() for x in Xtr]
            # erta prognoz uchun: yarim hollarda semestrning faqat birinchi k haftasi ko'rsatiladi
            cut = rng.integers(3, E.WEEKS + 1, len(idx))
            trunc = rng.random(len(idx)) < 0.5
            keepw = (np.arange(E.WEEKS)[None] < cut[:, None]).astype(np.float32)
            wm_ = np.where(trunc[:, None], wm_ * keepw, wm_).astype(np.float32)
            wk_ = wk_ * wm_[..., None]
            # matn dropout: model matnsiz ham ishlashi uchun
            drop_t = (rng.random(len(idx)) < 0.15)[:, None]
            tm_ = np.where(drop_t, 0.0, tm_).astype(np.float32)
            ti_ = (ti_ * tm_).astype(np.int32)
            y, sc, tp, se = [x[idx] for x in Ytr]
            key, sub = jax.random.split(key)
            params, opt_state, loss = step(params, opt_state, (st_, wk_, wm_, ti_, tm_, y, sc, tp, se.astype(np.int32)), sub)
            tot += float(loss) * len(idx)
        pv = np.asarray(predict_risk(params, *va_full))
        auc = roc_auc_score(va.past_ozlashtirish, pv)
        tpv = (np.asarray(predict_topics(params, *va_full)) >= 0.5).astype(int)
        ht = np.asarray(va_ht)
        tf1 = f1_score(va_tp[ht], tpv[ht], average="macro", zero_division=0)
        crit = auc + 0.5 * tf1   # ikkala vazifa: xavf prognozi va matn tahlili
        if crit > best:
            best, best_auc, best_params = crit, auc, jax.tree_util.tree_map(lambda x: np.asarray(x), params)
        print(f"davr {ep+1:2d}  loss {tot/n:.4f}  val AUC {auc:.4f}  mavzu F1 {tf1:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    params = best_params

    # ------------------------------------------------------------- eksport
    os.makedirs(os.path.join(ROOT, "model"), exist_ok=True)
    np.savez_compressed(os.path.join(ROOT, "model", "model.npz"), **{k: np.asarray(v, np.float32) for k, v in params.items()})
    meta = dict(d=D, heads=HEADS, layers=LAYERS, ff=FF, vocab=vocab.itos[2:],
                trained_on="sintetik ma'lumotlar, 2020-2023 o'quv yillari", val_auc=best_auc)
    with open(os.path.join(ROOT, "model", "model_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)

    # ------------------------------------------------------------- numpy nusxa bilan tekshirish
    npm = E.HeteroTransformer(os.path.join(ROOT, "model", "model.npz"))
    te_arr = E.build_arrays(te, wk, vocab)
    pj = np.asarray(predict_risk(params, *[jnp.asarray(x) for x in te_arr]))
    pn = npm.predict(*te_arr)["risk"]
    print("JAX va numpy farqi (max):", float(np.abs(pj - pn).max()))

    # ------------------------------------------------------------- baholash
    yte = te.past_ozlashtirish.values
    res = {"transformer": {}, "baseline": {}, "ablation": {}, "text": {}}

    def flat(df, upto):
        s, wk_, wm, _, _ = E.build_arrays(df, wk, vocab, upto)
        agg = []
        for j in range(E.WEEK_DIM):
            v = wk_[..., j]; m = wm
            mean = (v * m).sum(1) / np.maximum(m.sum(1), 1)
            last = np.array([v[i, int(m[i].sum()) - 1] if m[i].sum() else 0 for i in range(len(df))])
            agg += [mean, last]
        return np.hstack([s, wk_.reshape(len(df), -1), np.stack(agg, 1)])

    tfidf = TfidfVectorizer(tokenizer=E.tokenize, lowercase=False, token_pattern=None, min_df=2)
    Ttr = tfidf.fit_transform(tr.fikr); Tte = tfidf.transform(te.fikr)
    for upto in (4, 8, 12, 16):
        arr = E.build_arrays(te, wk, vocab, upto)
        p = npm.predict(*arr)["risk"]
        res["transformer"][upto] = metrics(yte, p)
        Ftr, Fte = flat(tr, upto), flat(te, upto)
        Atr = hstack([csr_matrix(Ftr), Ttr]).tocsr(); Ate = hstack([csr_matrix(Fte), Tte]).tocsr()
        bl = {}
        lr = LogisticRegression(max_iter=3000, C=1.0).fit(Atr, tr.past_ozlashtirish)
        bl["Logistik regressiya"] = metrics(yte, lr.predict_proba(Ate)[:, 1])
        rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0).fit(Atr, tr.past_ozlashtirish)
        bl["Tasodifiy o'rmon (RF)"] = metrics(yte, rf.predict_proba(Ate)[:, 1])
        gb = GradientBoostingClassifier(random_state=0).fit(Atr.toarray(), tr.past_ozlashtirish)
        bl["Gradient busting (GB)"] = metrics(yte, gb.predict_proba(Ate.toarray())[:, 1])
        res["baseline"][upto] = bl
        print(f"\n== {upto}-hafta ==")
        print("  Transformer         ", {k: round(v, 3) for k, v in res["transformer"][upto].items()})
        for k, v in bl.items():
            print(f"  {k:20s}", {kk: round(vv, 3) for kk, vv in v.items()})

    st_, wk_, wm_, ti_, tm_ = E.build_arrays(te, wk, vocab)
    res["ablation"] = {
        "To'liq model (barcha ma'lumotlar)": metrics(yte, npm.predict(st_, wk_, wm_, ti_, tm_)["risk"]),
        "Matnsiz": metrics(yte, npm.predict(st_, wk_, wm_, ti_, np.zeros_like(tm_))["risk"]),
        "Haftalik ma'lumotlarsiz": metrics(yte, npm.predict(st_, wk_, np.zeros_like(wm_), ti_, tm_)["risk"]),
        "Shaxsiy ma'lumotlarsiz": metrics(yte, npm.predict(np.zeros_like(st_), wk_, wm_, ti_, tm_)["risk"]),
    }
    print("\nAblatsiya:")
    for k, v in res["ablation"].items():
        print(f"  {k:35s}", {kk: round(vv, 3) for kk, vv in v.items()})

    out = npm.predict(st_, wk_, wm_, ti_, tm_)
    ht = out["has_text"]
    _, _, tp, se = labels(te)
    tpp = (out["topics"] >= 0.5).astype(int)
    res["text"] = {
        "mavzu_f1_makro": float(f1_score(tp[ht], tpp[ht], average="macro", zero_division=0)),
        "kayfiyat_aniqlik": float(accuracy_score(se[ht], out["sentiment"][ht].argmax(1))),
    }
    print("\nMatn tahlili:", res["text"])
    res["info"] = dict(train=len(tr), val=len(va), test=len(te), vocab=len(vocab.itos),
                       params=int(sum(np.asarray(v).size for v in params.values())), val_auc=best_auc)
    with open(os.path.join(ROOT, "model", "natijalar.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=float)
    print("\nParametrlar soni:", res["info"]["params"])


if __name__ == "__main__":
    main()
