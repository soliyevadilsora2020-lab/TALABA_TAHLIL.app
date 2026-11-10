# TalabaTahlil

**Transformer asosida heterogen axborotni intellektual qayta ishlash va qaror qabul qilishni qo'llab-quvvatlash tizimi (ta'lim sohasi misolida)** — PhD tadqiqoti doirasidagi dasturiy prototip.

## Dastur nima qiladi

Dastur turli xil ma'lumotlarni bitta modelda birgalikda qayta ishlaydi:

| Ma'lumot turi | Misol |
|---|---|
| Shaxsiy (statik) | fakultet, kurs, kirish bali, ta'lim shakli, yotoqxona, ish bilan bandlik |
| Haftalik (vaqt qatori) | 16 hafta davomida davomat, joriy baho, LMS faollik, topshiriq |
| Matn | talabaning so'rovnomadagi fikri (o'zbek tilida) |

Natijada dastur quyidagilarni beradi:
1. Har bir talabaning **past o'zlashtirish ehtimoli** va **yakuniy ball prognozi**. Prognozni semestr boshidan, masalan 4-haftadan, olish mumkin.
2. **Izoh**: qaysi ma'lumot turi va qaysi haftalar qarorga ko'proq ta'sir qilgani, talaba fikridagi qaysi so'zlarga model e'tibor bergani.
3. **Matn tahlili**: fikrlarning mavzusi (moliyaviy, ish, sog'liq, oilaviy, motivatsiya, fan murakkabligi, dars sifati) va kayfiyati.
4. **Tavsiyalar**: har bir talaba uchun aniq choralar (murabbiy bilan suhbat, konsultatsiya, ijtimoiy himoya bo'limi va boshqalar).
5. **Hisobot**: Excel fayl, unda talabalar, tavsiyalar va guruhlar kesimidagi statistika bo'ladi.

## Ma'lumotlar

Dastur hozircha **sintetik (sun'iy) ma'lumotlarda** o'qitilgan: 2020–2025 o'quv yillariga oid 6000 talaba. Ma'lumotlar `research/gen_data.py` skripti orqali yaratilgan. Vaqt bo'yicha bo'linishi:
- 2020–2023 yillar — o'qitish;
- 2024 yil — validatsiya;
- 2025 yil — test.

Haqiqiy ma'lumotlar (masalan, HEMIS'dan olingan) xuddi shu formatdagi Excel faylda yuklanadi.

**Excel fayl formati** — ikki varaqdan iborat:
- `talabalar`: `talaba_id, fish, fakultet, fakultet_kodi, kurs, guruh, kirish_bali, talim_shakli, yotoqxona, ishlaydi, fikr`
- `haftalik`: `talaba_id, hafta, davomat (0–1), baho (0–100), lms_daqiqa, topshiriq (0/1)`

Namuna sifatida `data/namuna_2025.xlsx` faylini ko'ring.

## Windows dasturini olish

1. Shu papkadagi barcha fayllarni GitHub omborga yuklang. `.github/workflows/windows.yml` fayli ham yuklangan bo'lishi kerak.
2. **Actions** bo'limida yig'ish o'zi ishga tushadi va taxminan 5–8 daqiqa davom etadi.
3. Yig'ish tugagach, **Releases** bo'limidan `TalabaTahlil-windows.zip` faylini yuklab oling.
4. Arxivni oching va `TalabaTahlil.exe` faylini ishga tushiring. Windows "Windows protected your PC" deb ogohlantirsa, **More info → Run anyway** ni bosing.

## Python orqali ishga tushirish

```
pip install -r requirements.txt
python app/main.py --demo
```

## Modelni qayta o'qitish

Model JAX kutubxonasida o'qitiladi. Dastur esa uni faqat numpy yordamida ishga tushiradi.

```
pip install jax optax scikit-learn
python research/gen_data.py
python research/train.py
```

`train.py` skripti modelni o'qitadi va natijalarni bazaviy modellar (logistik regressiya, tasodifiy o'rmon, gradient busting) bilan solishtiradi. Ablatsiya tahlilini ham bajaradi. Natijalar `model/natijalar.json` fayliga yoziladi.
