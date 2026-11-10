"""TalabaTahlil hisoblash yadrosi.

Faqat numpy va pandas ishlatadi: o'qitilgan transformer vaznlari (model.npz) shu yerda
ishga tushiriladi, shuning uchun dastur PyTorch'siz ishlaydi.

Model: heterogen transformer. Kirish ketma-ketligi turli xil ma'lumotlardan iborat tokenlar:
    [CLS] [STATIK] [H1] ... [H16] [so'z1] ... [so'zN]
  * STATIK  - talabaning o'zgarmas belgilari (fakultet, kurs, kirish bali, ...)
  * H1..H16 - har bir o'quv haftasi (davomat, baho, LMS faollik, topshiriq)
  * so'zlar - talaba fikrining so'zlari
Barcha tokenlar bitta umumiy fazoga proyeksiya qilinadi va self-attention orqali o'zaro
ta'sirlashadi. CLS chiqishi xavf ehtimolini, matn tokenlari esa matn mavzusi va kayfiyatini beradi.
"""
import json
import math
import os
import re

import numpy as np
import pandas as pd

WEEKS = 16
MAX_TEXT = 32
FAC_CODES = ["AF", "IQ", "PD", "FL", "TF"]
TOPICS = ["moliyaviy", "ish", "sogliq", "oilaviy", "motivatsiya", "murakkablik", "dars_sifati"]
TOPIC_NAMES = {
    "moliyaviy": "Moliyaviy qiyinchilik", "ish": "Ish bilan bandlik", "sogliq": "Sog'liq",
    "oilaviy": "Oilaviy sharoit", "motivatsiya": "Motivatsiya pasayishi",
    "murakkablik": "Fan murakkabligi", "dars_sifati": "Dars sifati",
}
SENTIMENTS = ["salbiy", "neytral", "ijobiy"]
SENT_NAMES = {"salbiy": "Salbiy", "neytral": "Neytral", "ijobiy": "Ijobiy"}

STATIC_DIM = 1 + 4 + len(FAC_CODES) + 3   # kirish bali, kurs(4), fakultet(5), kontrakt/yotoqxona/ishlaydi
WEEK_DIM = 4

WEEKLY_COLS = ["talaba_id", "hafta", "davomat", "baho", "lms_daqiqa", "topshiriq"]
STUDENT_COLS = ["talaba_id", "fish", "fakultet", "fakultet_kodi", "kurs", "guruh", "kirish_bali",
                "talim_shakli", "yotoqxona", "ishlaydi", "fikr"]

_APOS = re.compile(r"[ʻʼ’‘`´]")


def tokenize(text):
    if not isinstance(text, str):
        return []
    t = _APOS.sub("'", text.lower())
    return re.findall(r"[a-z']+", t)


# --------------------------------------------------------------------------- ma'lumot tayyorlash

def fac_code_of(row):
    code = str(row.get("fakultet_kodi", "") or "").strip().upper()
    if code in FAC_CODES:
        return code
    g = str(row.get("guruh", "") or "")[:2].upper()
    return g if g in FAC_CODES else ""


def static_vector(row):
    v = np.zeros(STATIC_DIM, dtype=np.float32)
    try:
        v[0] = (float(row.get("kirish_bali", 120)) - 120.0) / 30.0
    except (TypeError, ValueError):
        v[0] = 0.0
    try:
        k = int(row.get("kurs", 1))
    except (TypeError, ValueError):
        k = 1
    v[1 + min(max(k, 1), 4) - 1] = 1.0
    code = fac_code_of(row)
    if code:
        v[5 + FAC_CODES.index(code)] = 1.0
    v[10] = 1.0 if str(row.get("talim_shakli", "")).strip().lower().startswith("kontrakt") else 0.0
    v[11] = float(row.get("yotoqxona", 0) or 0)
    v[12] = float(row.get("ishlaydi", 0) or 0)
    return v


def week_features(davomat, baho, lms, topshiriq):
    return np.array([
        float(davomat),
        float(baho) / 100.0,
        math.log1p(max(float(lms), 0.0)) / 6.0,
        float(topshiriq),
    ], dtype=np.float32)


class Vocab:
    def __init__(self, words):
        self.itos = ["<pad>", "<unk>"] + list(words)
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def encode(self, text):
        ids = [self.stoi.get(w, 1) for w in tokenize(text)][:MAX_TEXT]
        return ids

    @staticmethod
    def build(texts, min_count=2):
        cnt = {}
        for t in texts:
            for w in tokenize(t):
                cnt[w] = cnt.get(w, 0) + 1
        words = sorted(w for w, c in cnt.items() if c >= min_count)
        return Vocab(words)


def build_arrays(students, weekly, vocab, upto_week=WEEKS):
    """DataFrame'lardan model kirishlarini yasaydi.

    Qaytaradi: static (N,13), weeks (N,16,4), week_mask (N,16), text_ids (N,32), text_mask (N,32)
    """
    n = len(students)
    static = np.stack([static_vector(r) for r in students.to_dict("records")]) if n else np.zeros((0, STATIC_DIM), np.float32)
    weeks = np.zeros((n, WEEKS, WEEK_DIM), dtype=np.float32)
    wmask = np.zeros((n, WEEKS), dtype=np.float32)
    idx = {sid: i for i, sid in enumerate(students["talaba_id"].astype(str))}
    if weekly is not None and len(weekly):
        w = weekly[weekly["hafta"].between(1, upto_week)]
        for sid, h, d, b, l, t in w[["talaba_id", "hafta", "davomat", "baho", "lms_daqiqa", "topshiriq"]].itertuples(index=False):
            i = idx.get(str(sid))
            if i is None:
                continue
            h = int(h) - 1
            weeks[i, h] = week_features(d, b, l, t)
            wmask[i, h] = 1.0
    tids = np.zeros((n, MAX_TEXT), dtype=np.int64)
    tmask = np.zeros((n, MAX_TEXT), dtype=np.float32)
    for i, txt in enumerate(students.get("fikr", pd.Series([""] * n)).fillna("").astype(str)):
        ids = vocab.encode(txt)
        tids[i, :len(ids)] = ids
        tmask[i, :len(ids)] = 1.0
    return static, weeks, wmask, tids, tmask


# --------------------------------------------------------------------------- numpy transformer

def _layernorm(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g + b


def _gelu(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))


def _softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return np.log(p / (1 - p))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class HeteroTransformer:
    """model.npz vaznlari bilan ishlaydigan numpy transformer (PyTorch modelining aynan nusxasi)."""

    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.w = {k: z[k].astype(np.float32) for k in z.files}
        meta_path = os.path.join(os.path.dirname(path), "model_meta.json")
        with open(meta_path, encoding="utf-8") as f:
            self.meta = json.load(f)
        self.heads = self.meta["heads"]
        self.layers = self.meta["layers"]
        self.vocab = Vocab(self.meta["vocab"])

    def _lin(self, x, name):
        return x @ self.w[name + ".weight"].T + self.w[name + ".bias"]

    def embed(self, static, weeks, wmask, tids, tmask):
        w = self.w
        n = static.shape[0]
        d = w["cls"].shape[-1]
        cls = np.broadcast_to(w["cls"].reshape(1, 1, d), (n, 1, d))
        st = self._lin(static, "static_proj")[:, None, :] + w["type_emb.weight"][1]
        wk = self._lin(weeks, "week_proj") + w["week_pos.weight"][None, :WEEKS] + w["type_emb.weight"][2]
        tx = w["tok_emb.weight"][tids] + w["text_pos.weight"][None, :MAX_TEXT] + w["type_emb.weight"][3]
        cls = cls + w["type_emb.weight"][0]
        x = np.concatenate([cls, st, wk, tx], axis=1)
        mask = np.concatenate([np.ones((n, 2), np.float32), wmask, tmask], axis=1)
        return x, mask

    def forward(self, static, weeks, wmask, tids, tmask, return_attn=False):
        x, mask = self.embed(static, weeks, wmask, tids, tmask)
        n, L, d = x.shape
        h = self.heads
        dh = d // h
        bias = (1.0 - mask)[:, None, None, :] * -1e9
        attn_last = None
        for i in range(self.layers):
            p = f"layers.{i}."
            y = _layernorm(x, self.w[p + "ln1.weight"], self.w[p + "ln1.bias"])
            q = self._lin(y, p + "q").reshape(n, L, h, dh).transpose(0, 2, 1, 3)
            k = self._lin(y, p + "k").reshape(n, L, h, dh).transpose(0, 2, 1, 3)
            v = self._lin(y, p + "v").reshape(n, L, h, dh).transpose(0, 2, 1, 3)
            a = _softmax(q @ k.transpose(0, 1, 3, 2) / math.sqrt(dh) + bias)
            if i == self.layers - 1:
                attn_last = a
            o = (a @ v).transpose(0, 2, 1, 3).reshape(n, L, d)
            x = x + self._lin(o, p + "o")
            y = _layernorm(x, self.w[p + "ln2.weight"], self.w[p + "ln2.bias"])
            x = x + self._lin(_gelu(self._lin(y, p + "ff1")), p + "ff2")
        x = _layernorm(x, self.w["ln_f.weight"], self.w["ln_f.bias"])
        cls = x[:, 0]
        risk = _sigmoid(self._lin(cls, "risk_head")[:, 0])
        score = np.clip(self._lin(cls, "score_head")[:, 0], 0, 1.2) * 100
        tm = mask[:, 2 + WEEKS:]
        tx = x[:, 2 + WEEKS:]
        denom = np.maximum(tm.sum(1, keepdims=True), 1.0)
        pooled = (tx * tm[..., None]).sum(1) / denom
        topics = _sigmoid(self._lin(pooled, "topic_head"))
        sent = _softmax(self._lin(pooled, "sent_head"))
        has_text = tm.sum(1) > 0
        topics[~has_text] = 0.0
        out = dict(risk=risk, score=score, topics=topics, sentiment=sent, has_text=has_text)
        if return_attn:
            out["attn"] = attn_last.mean(1)[:, 0, :]   # CLS -> barcha tokenlar, boshlar bo'yicha o'rtacha
        return out

    def predict(self, static, weeks, wmask, tids, tmask, batch=512, return_attn=False):
        outs = []
        for s in range(0, static.shape[0], batch):
            sl = slice(s, s + batch)
            outs.append(self.forward(static[sl], weeks[sl], wmask[sl], tids[sl], tmask[sl], return_attn))
        if not outs:
            return {}
        return {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}

    def contributions(self, static, weeks, wmask, tids, tmask, ref):
        """Ma'lumot turlarining hissasi (logit fazosida): har bir tur "o'rtacha talaba" qiymati bilan
        almashtirilganda xavf qanchaga o'zgarishi. ref = reference_profile(...) natijasi."""
        st_ref, wk_ref = ref
        r0 = logit(self.predict(static, weeks, wmask, tids, tmask)["risk"])
        r_st = logit(self.predict(np.broadcast_to(st_ref, static.shape).copy(), weeks, wmask, tids, tmask)["risk"])
        wk_avg = np.broadcast_to(wk_ref, weeks.shape) * wmask[..., None]
        r_wk = logit(self.predict(static, wk_avg.astype(np.float32), wmask, tids, tmask)["risk"])
        r_tx = logit(self.predict(static, weeks, wmask, tids, np.zeros_like(tmask))["risk"])
        return r0 - r_st, r0 - r_wk, r0 - r_tx

    def explain(self, static, weeks, wmask, tids, tmask, ref):
        """Bitta talaba uchun izoh: attention vaznlari va har bir ma'lumot turining hissasi."""
        base = self.forward(static, weeks, wmask, tids, tmask, return_attn=True)
        r0 = float(base["risk"][0])
        c_st, c_wk, c_tx = self.contributions(static, weeks, wmask, tids, tmask, ref)
        raw = {
            "Shaxsiy ma'lumotlar": float(c_st[0]),
            "Haftalik o'zlashtirish": float(c_wk[0]),
            "Talaba fikri (matn)": float(c_tx[0]),
        }
        tot = sum(abs(v) for v in raw.values()) or 1.0
        contrib = {k: v / tot for k, v in raw.items()}   # nisbiy ulush
        a = base["attn"][0]
        week_att = a[2:2 + WEEKS] * wmask[0]
        text_att = a[2 + WEEKS:] * tmask[0]
        words = [self.vocab.itos[i] for i in tids[0] if i > 0]
        return dict(risk=r0, contrib=contrib, week_attn=week_att,
                    text_attn=text_att[:len(words)], words=words, static_attn=float(a[1]),
                    topics=base["topics"][0], sentiment=base["sentiment"][0])


def reference_profile(static, weeks, wmask):
    """"O'rtacha talaba": statik belgilarning o'rtachasi va har bir hafta bo'yicha o'rtacha ko'rsatkichlar."""
    st_ref = static.mean(0)
    cnt = np.maximum(wmask.sum(0), 1)[:, None]
    wk_ref = (weeks * wmask[..., None]).sum(0) / cnt
    return st_ref.astype(np.float32), wk_ref.astype(np.float32)


# --------------------------------------------------------------------------- tavsiyalar

def risk_level(p):
    if p >= 0.6:
        return "Yuqori"
    if p >= 0.3:
        return "O'rta"
    return "Past"


def recommendations(student, wk_df, risk, topics, upto_week=WEEKS):
    """Qaror qabul qilishni qo'llab-quvvatlash: talaba holatiga qarab aniq tavsiyalar."""
    rec = []
    w = wk_df[wk_df["hafta"] <= upto_week].sort_values("hafta") if wk_df is not None and len(wk_df) else None
    if w is not None and len(w):
        att = w["davomat"].mean()
        last = w.tail(4)
        first = w.head(max(len(w) - 4, 1))
        g_drop = first["baho"].mean() - last["baho"].mean()
        sub = w["topshiriq"].mean()
        lms = w["lms_daqiqa"].mean()
        if att < 0.75:
            rec.append(("Davomat", f"Davomat {att*100:.0f}% - guruh murabbiyi talaba bilan suhbat o'tkazsin, "
                                     "sababli/sababsiz qoldirilgan darslar aniqlansin."))
        if g_drop > 8:
            rec.append(("Baholar pasaymoqda", f"So'nggi haftalarda o'rtacha baho {g_drop:.0f} ballga tushgan - "
                                               "fan o'qituvchisi qo'shimcha konsultatsiya belgilasin."))
        if sub < 0.6:
            rec.append(("Topshiriqlar", f"Topshiriqlarning faqat {sub*100:.0f}% topshirilgan - "
                                        "qarzdorlik jadvali tuzilib, muddatlar belgilansin."))
        if lms < 30:
            rec.append(("LMS faollik", f"LMS'da haftasiga o'rtacha {lms:.0f} daqiqa - elektron "
                                       "materiallar bilan ishlash nazorat qilinsin."))
    tmap = {
        "moliyaviy": ("Moliyaviy holat", "Ijtimoiy himoya bo'limiga yo'naltirilsin: kontraktni bo'lib to'lash "
                                         "yoki moddiy yordam imkoniyati ko'rib chiqilsin."),
        "ish": ("Ish bilan bandlik", "Individual o'quv jadvali yoki masofaviy topshiriqlar imkoniyati ko'rib chiqilsin."),
        "sogliq": ("Sog'liq", "Tibbiy punkt bilan bog'lanilsin; kasallik sababli qoldirilgan mashg'ulotlarni "
                              "qayta topshirish tartibi tushuntirilsin."),
        "oilaviy": ("Oilaviy sharoit", "Psixolog va tyutor bilan suhbat tashkil etilsin, zarur bo'lsa ota-ona bilan bog'lanilsin."),
        "motivatsiya": ("Motivatsiya", "Tyutor/psixolog bilan individual suhbat, ilmiy to'garak yoki loyihaga jalb qilish tavsiya etiladi."),
        "murakkablik": ("Fan murakkabligi", "Iqtidorli talaba bilan juftlik (mentorlik) va qo'shimcha amaliy mashg'ulotlar tavsiya etiladi."),
        "dars_sifati": ("Dars sifati", "Talaba fikri kafedra mudiriga yetkazilsin; dars metodikasi muhokama qilinsin."),
    }
    for i, t in enumerate(TOPICS):
        if topics is not None and topics[i] >= 0.5:
            rec.append(tmap[t])
    if risk >= 0.6 and not rec:
        rec.append(("Umumiy", "Talaba bilan individual suhbat o'tkazilib, qiyinchilik sabablari aniqlansin."))
    if risk < 0.3 and not rec:
        rec.append(("Holat yaxshi", "Alohida chora talab etilmaydi; talabani olimpiada va ilmiy ishlarga jalb qilish mumkin."))
    return rec


# --------------------------------------------------------------------------- yuklash

def load_dataset(path):
    """Excel (ikki varaq: talabalar, haftalik) yoki papka/CSV juftini o'qiydi."""
    if path.lower().endswith((".xlsx", ".xls")):
        x = pd.read_excel(path, sheet_name=None)
        names = {k.lower(): k for k in x}
        st = x[names.get("talabalar", list(x)[0])]
        wk = x[names.get("haftalik", list(x)[1] if len(x) > 1 else list(x)[0])]
    else:
        st = pd.read_csv(path)
        base = os.path.dirname(path)
        cand = [f for f in os.listdir(base) if "haftalik" in f.lower() and f.endswith(".csv")]
        wk = pd.read_csv(os.path.join(base, cand[0])) if cand else pd.DataFrame(columns=WEEKLY_COLS)
    st.columns = [c.strip().lower() for c in st.columns]
    wk.columns = [c.strip().lower() for c in wk.columns]
    miss = [c for c in ["talaba_id"] if c not in st.columns] + [c for c in WEEKLY_COLS if c not in wk.columns]
    if miss:
        raise ValueError("Faylda quyidagi ustunlar topilmadi: " + ", ".join(miss))
    for c in STUDENT_COLS:
        if c not in st.columns:
            st[c] = "" if c in ("fish", "fakultet", "fakultet_kodi", "guruh", "talim_shakli", "fikr") else 0
    st["talaba_id"] = st["talaba_id"].astype(str)
    wk["talaba_id"] = wk["talaba_id"].astype(str)
    st["fikr"] = st["fikr"].fillna("")
    return st.reset_index(drop=True), wk.reset_index(drop=True)
