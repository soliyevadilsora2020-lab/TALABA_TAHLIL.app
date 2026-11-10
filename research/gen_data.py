"""Sintetik ta'lim ma'lumotlari generatori (2020-2025 o'quv yillari).

Har bir talaba uchun:
  * statik belgilar  : fakultet, kurs, kirish bali, ta'lim shakli, yotoqxona, ishlaydi, o'quv yili
  * haftalik belgilar: 16 hafta uchun davomat, joriy baho, LMS faollik (daqiqa), topshiriq topshirgan
  * matn             : talabaning so'rovnomadagi fikri (o'zbek tilida)
  * belgilar (label) : yakuniy ball, past o'zlashtirish (yakuniy < 55), matn mavzulari va kayfiyati

Yashirin omillar: qobiliyat (a), motivatsiya (m), semestr o'rtasidagi muammo (shock).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

SEED = 42
N_STUDENTS = 6000
WEEKS = 16
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "data")

FACULTIES = [
    ("AF", "Aniq fanlar va intellektual muhandislik"),
    ("IQ", "Iqtisodiyot va turizm"),
    ("PD", "Pedagogika"),
    ("FL", "Filologiya"),
    ("TF", "Tabiiy fanlar"),
]
YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
TOPICS = ["moliyaviy", "ish", "sogliq", "oilaviy", "motivatsiya", "murakkablik", "dars_sifati"]
SENTIMENTS = ["salbiy", "neytral", "ijobiy"]

FIRST = ["Aziz", "Bekzod", "Dilnoza", "Gulnora", "Jasur", "Kamola", "Laziz", "Madina", "Nodir",
         "Oybek", "Parvina", "Rustam", "Sardor", "Shahnoza", "Temur", "Umida", "Xurshid", "Yulduz",
         "Zarina", "Akmal", "Barno", "Dilshod", "Feruza", "Habiba", "Islom", "Javohir", "Komil",
         "Lola", "Mohira", "Nilufar", "Otabek", "Sevara", "Ulug'bek", "Shoxrux", "Mavluda", "Asal"]
LAST = ["Karimov", "Rahimov", "Toshev", "Qodirov", "Ergashev", "Nazarov", "Saidov", "Yusupov",
        "Hamidov", "Jo'rayev", "Murodov", "Sobirov", "Xolmatov", "Norov", "Rasulov", "Aminov",
        "Fayziyev", "Boboyev", "Ismoilov", "Usmonov"]
FEMALE = {"Dilnoza", "Gulnora", "Kamola", "Madina", "Parvina", "Shahnoza", "Umida", "Yulduz",
          "Zarina", "Barno", "Feruza", "Habiba", "Lola", "Mohira", "Nilufar", "Sevara", "Mavluda", "Asal"}

PHRASES = {
    "moliyaviy": ["kontrakt to'lovi uchun pul topishga qiynalyapman",
                  "oilamizning moddiy ahvoli og'ir",
                  "yo'l kira va ijara uchun pul yetmayapti",
                  "kontraktni o'z vaqtida to'lay olmayapman"],
    "ish": ["kechqurunlari ishlayman darslarga vaqt yetmayapti",
            "ishdan charchab darsga tayyorlana olmayapman",
            "ish jadvali dars jadvali bilan to'g'ri kelmaydi",
            "oilani boqish uchun ishlashga majburman"],
    "sogliq": ["so'nggi haftalarda tez-tez kasal bo'ldim",
               "sog'lig'im sababli bir necha darsni qoldirdim",
               "shifoxonada davolandim va ortda qoldim",
               "boshim tez-tez og'riydi diqqatimni jamlay olmayman"],
    "oilaviy": ["oilaviy sharoit sabab uyda ko'p vaqt o'tkazyapman",
                "oilamda muammolar bor diqqatimni jamlay olmayapman",
                "uyda kichik ukalarimga qarashim kerak",
                "oilaviy tadbirlar sabab darslarni qoldirdim"],
    "motivatsiya": ["o'qishga qiziqishim kamaydi",
                    "yo'nalishni noto'g'ri tanlaganga o'xshayman",
                    "darslar menga zerikarli tuyulyapti",
                    "nima uchun o'qiyotganimni bilmay qoldim"],
    "murakkablik": ["fan juda murakkab mavzularni tushunmayapman",
                    "formulalar va masalalar menga qiyin",
                    "uy vazifalari juda ko'p va qiyin",
                    "oldingi mavzularni bilmaganim uchun yangilarini tushunmayapman"],
    "dars_sifati_neg": ["darslar tushunarsiz o'tmoqda",
                        "o'qituvchi savollarga vaqt ajratmaydi",
                        "amaliy mashg'ulotlar yetarli emas"],
    "dars_sifati_pos": ["o'qituvchilar mavzuni tushunarli tushuntiradi",
                        "darslar qiziqarli o'tmoqda",
                        "amaliy mashg'ulotlar juda foydali"],
    "ijobiy": ["o'qishim yaxshi ketyapti",
               "barcha topshiriqlarni o'z vaqtida bajaryapman",
               "kutubxonada ko'p shug'ullanaman",
               "olimpiadaga tayyorlanyapman",
               "guruhdoshlarim bilan birga o'qiymiz",
               "yangi bilimlar olishdan xursandman"],
    "neytral": ["hammasi odatdagidek",
                "ba'zi fanlar qiyin lekin uddalayapman",
                "o'qish odatiy davom etmoqda",
                "takliflarim yo'q"],
}

SHOCK_TYPES = ["moliyaviy", "ish", "sogliq", "oilaviy"]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def make_text(rng, m, shock_type, risk_hint):
    """Talaba fikrini yashirin holatga qarab tuzadi. Matn mavzulari va kayfiyatni qaytaradi."""
    if rng.random() < 0.15:
        return "", [], "neytral"  # fikr qoldirmagan
    parts, topics = [], set()
    if shock_type and rng.random() < 0.75:
        parts.append(rng.choice(PHRASES[shock_type]))
        topics.add(shock_type)
    if m < -0.6 and rng.random() < 0.7:
        parts.append(rng.choice(PHRASES["motivatsiya"]))
        topics.add("motivatsiya")
    if risk_hint > 0.5 and rng.random() < 0.4:
        parts.append(rng.choice(PHRASES["murakkablik"]))
        topics.add("murakkablik")
    r = rng.random()
    if r < 0.18:
        if risk_hint > 0.45 and rng.random() < 0.6:
            parts.append(rng.choice(PHRASES["dars_sifati_neg"]))
        else:
            parts.append(rng.choice(PHRASES["dars_sifati_pos"]))
        topics.add("dars_sifati")
    # tasodifiy "shovqin": yaxshi talaba ham ba'zan muammoni aytadi
    if not shock_type and rng.random() < 0.06:
        t = rng.choice(SHOCK_TYPES)
        parts.append(rng.choice(PHRASES[t]))
        topics.add(t)
    if not parts:
        parts.append(rng.choice(PHRASES["ijobiy"] if m + rng.normal(0, 0.6) > 0 else PHRASES["neytral"]))
    elif rng.random() < 0.25:
        parts.append(rng.choice(PHRASES["neytral"]))
    rng.shuffle(parts)
    text = ". ".join(p[0].upper() + p[1:] for p in parts) + "."

    neg = len([t for t in topics if t != "dars_sifati"]) + (
        1 if any(p in PHRASES["dars_sifati_neg"] for p in parts) else 0)
    pos = sum(p in PHRASES["ijobiy"] or p in PHRASES["dars_sifati_pos"] for p in parts)
    sentiment = "salbiy" if neg > pos else ("ijobiy" if pos > neg else "neytral")
    return text, sorted(topics), sentiment


def main():
    rng = np.random.default_rng(SEED)
    rows, weekly = [], []
    group_counter = {}

    for i in range(N_STUDENTS):
        year = int(rng.choice(YEARS, p=[0.12, 0.15, 0.17, 0.18, 0.19, 0.19]))
        fi = int(rng.integers(len(FACULTIES)))
        fac_code, fac_name = FACULTIES[fi]
        course = int(rng.integers(1, 5))
        key = (fac_code, year, course)
        group_counter[key] = group_counter.get(key, 0) + 1
        group_no = (group_counter[key] - 1) // 25 + 1
        group = f"{fac_code}-{course}{str(year)[-2:]}-{group_no:02d}"

        a = rng.normal(0, 1)                      # qobiliyat
        m = rng.normal(0, 1)                      # motivatsiya
        fac_diff = [0.25, -0.05, -0.15, -0.1, 0.1][fi]  # fakultet murakkabligi
        works = int(rng.random() < sigmoid(-1.3 - 0.3 * m + 0.25 * (course - 2)))
        contract = int(rng.random() < 0.72)
        dorm = int(rng.random() < 0.3)
        entrance = float(np.clip(120 + 20 * a + rng.normal(0, 10), 60, 189))

        shock_type, shock_week = None, None
        p_shock = 0.18 + 0.12 * contract * (a < -0.3) + 0.15 * works
        if rng.random() < p_shock:
            shock_type = str(rng.choice(SHOCK_TYPES, p=[0.3, 0.25, 0.25, 0.2]))
            if works and rng.random() < 0.5:
                shock_type = "ish"
            shock_week = int(rng.integers(3, 13))
        shock_sev = rng.uniform(0.6, 1.4) if shock_type else 0.0

        grades, atts, lmss, subs = [], [], [], []
        for w in range(1, WEEKS + 1):
            s = shock_sev if (shock_type and w >= shock_week) else 0.0
            drift = -0.05 * (w - 8) * (m < -0.8)  # motivatsiyasi pastlarda asta pasayish
            att_p = sigmoid(2.0 + 0.8 * m - 1.6 * s - 0.6 * works + drift)
            att = rng.binomial(4, att_p) / 4.0
            g = 74 + 11 * a + 6 * m - 18 * s - 25 * fac_diff + 8 * drift + rng.normal(0, 9)
            if att == 0:
                g -= 15
            g = float(np.clip(g, 0, 100))
            lms = float(np.clip(np.exp(np.log(55) + 0.45 * m + 0.2 * a - 0.8 * s + rng.normal(0, 0.45)), 0, 400))
            sub = int(rng.random() < sigmoid(1.2 + 1.0 * m + 0.5 * a - 1.3 * s))
            grades.append(g); atts.append(att); lmss.append(lms); subs.append(sub)
            weekly.append((i, w, round(att, 2), round(g, 1), round(lms, 0), sub))

        final = (0.45 * np.mean(grades[8:]) + 0.15 * np.mean(grades[:8]) + 18 * np.mean(atts)
                 + 10 * np.mean(subs) + 0.04 * (entrance - 120) + rng.normal(0, 5))
        final = float(np.clip(final, 0, 100))
        risk = int(final < 55)

        text, topics, sentiment = make_text(rng, m, shock_type, risk_hint=float(a < -0.4) * 0.7 + rng.random() * 0.3)
        fn = str(rng.choice(FIRST)); ln = str(rng.choice(LAST))
        if fn in FEMALE:
            ln = ln + "a"
        rows.append(dict(
            talaba_id=f"T{year}-{i:05d}", fish=f"{ln} {fn}", yil=year, fakultet=fac_name,
            fakultet_kodi=fac_code, kurs=course, guruh=group, kirish_bali=round(entrance, 1),
            talim_shakli="kontrakt" if contract else "grant", yotoqxona=dorm, ishlaydi=works,
            fikr=text, yakuniy_ball=round(final, 1), past_ozlashtirish=risk,
            mavzular=";".join(topics), kayfiyat=sentiment,
        ))

    os.makedirs(OUT, exist_ok=True)
    st = pd.DataFrame(rows)
    wk = pd.DataFrame(weekly, columns=["idx", "hafta", "davomat", "baho", "lms_daqiqa", "topshiriq"])
    wk.insert(0, "talaba_id", st["talaba_id"].values[wk["idx"].values])
    wk = wk.drop(columns="idx")
    st.to_csv(os.path.join(OUT, "talabalar_full.csv"), index=False)
    wk.to_csv(os.path.join(OUT, "haftalik_full.csv"), index=False)
    print("talabalar:", len(st), "| past o'zlashtiruvchi ulushi:", round(st.past_ozlashtirish.mean(), 3))
    print(st.groupby("yil").past_ozlashtirish.mean().round(3).to_dict())
    print("kayfiyat:", st.kayfiyat.value_counts().to_dict())
    print(st[["fish", "guruh", "fikr", "past_ozlashtirish"]].head(8).to_string())


if __name__ == "__main__":
    main()
