"""TalabaTahlil - transformer asosida heterogen ta'lim ma'lumotlarini intellektual qayta ishlash
va qaror qabul qilishni qo'llab-quvvatlash tizimi (PhD tadqiqoti prototipi).
"""
import json
import os
import sys
import traceback

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex, QSortFilterProxyModel
from PySide6.QtGui import QAction, QColor, QFont, QIcon
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout, QHeaderView, QLabel,
    QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QPushButton, QScrollArea,
    QSpinBox, QSplitter, QStackedWidget, QTableView, QTableWidget, QTableWidgetItem, QTextBrowser,
    QToolBar, QVBoxLayout, QWidget,
)
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

import engine as E  # noqa: E402

APP_NAME = "TalabaTahlil"
BASE = getattr(sys, "_MEIPASS", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

C = {
    "bg": "#F4F6FA", "card": "#FFFFFF", "ink": "#1B2437", "muted": "#6B7489", "line": "#E2E6EE",
    "primary": "#1F4E8C", "accent": "#2A9D8F",
    "high": "#D64545", "mid": "#E9A23B", "low": "#2E9E6A",
}
LEVEL_COLOR = {"Yuqori": C["high"], "O'rta": C["mid"], "Past": C["low"]}

QSS = f"""
QMainWindow, QWidget#page {{ background: {C['bg']}; }}
QWidget {{ color: {C['ink']}; font-size: 10pt; }}
QListWidget#nav {{ background: {C['primary']}; border: none; padding-top: 8px; }}
QListWidget#nav::item {{ color: #DCE6F5; padding: 12px 16px; border-radius: 6px; margin: 2px 8px; }}
QListWidget#nav::item:selected {{ background: rgba(255,255,255,0.16); color: white; font-weight: 600; }}
QFrame#card {{ background: {C['card']}; border: 1px solid {C['line']}; border-radius: 10px; }}
QLabel#h1 {{ font-size: 16pt; font-weight: 700; }}
QLabel#h2 {{ font-size: 12pt; font-weight: 600; }}
QLabel#muted {{ color: {C['muted']}; }}
QLabel#kpi {{ font-size: 22pt; font-weight: 700; }}
QPushButton {{ background: {C['primary']}; color: white; border: none; border-radius: 6px; padding: 7px 14px; font-weight: 600; }}
QPushButton:hover {{ background: #173D6F; }}
QPushButton#ghost {{ background: white; color: {C['primary']}; border: 1px solid {C['primary']}; }}
QToolBar {{ background: white; border-bottom: 1px solid {C['line']}; spacing: 8px; padding: 6px; }}
QTableView, QTableWidget {{ background: white; border: 1px solid {C['line']}; border-radius: 8px; gridline-color: {C['line']};
    selection-background-color: #DCE8F7; selection-color: {C['ink']}; }}
QHeaderView::section {{ background: #EEF2F8; border: none; border-bottom: 1px solid {C['line']}; padding: 6px; font-weight: 600; }}
QLineEdit, QComboBox, QSpinBox {{ background: white; border: 1px solid {C['line']}; border-radius: 6px; padding: 5px 8px; }}
QTextBrowser {{ background: white; border: 1px solid {C['line']}; border-radius: 8px; padding: 6px; }}
"""


def card(*widgets, margins=14, spacing=8):
    f = QFrame(); f.setObjectName("card")
    lay = QVBoxLayout(f); lay.setContentsMargins(margins, margins, margins, margins); lay.setSpacing(spacing)
    for w in widgets:
        if isinstance(w, QWidget):
            lay.addWidget(w)
        else:
            lay.addLayout(w)
    return f


def label(text, obj=None, wrap=False):
    l = QLabel(text)
    if obj:
        l.setObjectName(obj)
    l.setWordWrap(wrap)
    return l


class Chart(FigureCanvasQTAgg):
    def __init__(self, w=5, h=3):
        self.fig = Figure(figsize=(w, h), dpi=100, facecolor="white")
        super().__init__(self.fig)
        self.setMinimumHeight(int(h * 80))

    def clear(self):
        self.fig.clear()
        return self.fig

    @staticmethod
    def style(ax):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.spines["left"].set_color("#B8C0CE"); ax.spines["bottom"].set_color("#B8C0CE")
        ax.tick_params(colors="#4A5468", labelsize=8)
        ax.title.set_fontsize(10); ax.title.set_fontweight("bold")


class DFModel(QAbstractTableModel):
    def __init__(self, df=None, color_col=None, pct_cols=()):
        super().__init__()
        self.df = df if df is not None else pd.DataFrame()
        self.color_col = color_col
        self.pct_cols = set(pct_cols)

    def set(self, df):
        self.beginResetModel(); self.df = df.reset_index(drop=True); self.endResetModel()

    def rowCount(self, p=QModelIndex()):
        return len(self.df)

    def columnCount(self, p=QModelIndex()):
        return len(self.df.columns)

    def data(self, idx, role=Qt.DisplayRole):
        if not idx.isValid():
            return None
        v = self.df.iat[idx.row(), idx.column()]
        col = self.df.columns[idx.column()]
        if role == Qt.DisplayRole:
            if col in self.pct_cols:
                return f"{v*100:.0f}%"
            if isinstance(v, float):
                return f"{v:.1f}"
            return str(v)
        if role == Qt.UserRole:
            return v
        if role == Qt.ForegroundRole and self.color_col and col == self.color_col:
            return QColor(LEVEL_COLOR.get(v, C["ink"]))
        if role == Qt.FontRole and self.color_col and col == self.color_col:
            f = QFont(); f.setBold(True); return f
        return None

    HEADER = {"Xavf": "Ehtimol", "Daraja": "Ehtimol darajasi"}

    def headerData(self, s, o, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and o == Qt.Horizontal:
            c = str(self.df.columns[s])
            return self.HEADER.get(c, c)
        return None


# ============================================================================ sahifalar

class OverviewPage(QWidget):
    def __init__(self):
        super().__init__(); self.setObjectName("page")
        lay = QVBoxLayout(self); lay.setContentsMargins(20, 18, 20, 18); lay.setSpacing(14)
        lay.addWidget(label("Umumiy ko'rinish", "h1"))
        self.sub = label("Ma'lumot yuklang va «Tahlil qilish» tugmasini bosing.", "muted")
        lay.addWidget(self.sub)
        row = QHBoxLayout(); row.setSpacing(12)
        self.kpis = {}
        for key, title, color in [("n", "Talabalar", C["primary"]), ("hi", "Ehtimoli yuqori", C["high"]),
                                  ("mid", "Ehtimoli o'rta", C["mid"]), ("lo", "Ehtimoli past", C["low"]),
                                  ("score", "O'rtacha prognoz ball", C["accent"])]:
            v = label("—", "kpi"); v.setStyleSheet(f"color:{color}")
            row.addWidget(card(label(title, "muted"), v))
            self.kpis[key] = v
        lay.addLayout(row)
        grid = QGridLayout(); grid.setSpacing(12)
        self.c_fac = Chart(5, 3); self.c_hist = Chart(5, 3); self.c_topic = Chart(5, 3); self.c_course = Chart(5, 3)
        grid.addWidget(card(self.c_fac), 0, 0); grid.addWidget(card(self.c_hist), 0, 1)
        grid.addWidget(card(self.c_course), 1, 0); grid.addWidget(card(self.c_topic), 1, 1)
        lay.addLayout(grid, 1)

    def update_view(self, R, week):
        n = len(R)
        lv = R["Daraja"].value_counts()
        self.kpis["n"].setText(str(n))
        for k, name in (("hi", "Yuqori"), ("mid", "O'rta"), ("lo", "Past")):
            c = int(lv.get(name, 0))
            self.kpis[k].setText(f"{c}  ({c/max(n,1)*100:.0f}%)")
        self.kpis["score"].setText(f"{R['Prognoz ball'].mean():.1f}")
        self.sub.setText(f"Tahlil semestrning {week}-haftasigacha bo'lgan ma'lumotlar asosida bajarildi.")

        f = self.c_fac.clear(); ax = f.add_subplot(111); Chart.style(ax)
        t = pd.crosstab(R["Fakultet kodi"], R["Daraja"]).reindex(columns=["Yuqori", "O'rta", "Past"], fill_value=0)
        left = np.zeros(len(t))
        for lvname in ["Yuqori", "O'rta", "Past"]:
            ax.barh(t.index, t[lvname], left=left, color=LEVEL_COLOR[lvname], label=lvname, height=0.6)
            left += t[lvname].values
        ax.set_title("Fakultetlar kesimida past o'zlashtirish ehtimoli", pad=18)
        ax.legend(fontsize=7, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.98))
        f.tight_layout(); self.c_fac.draw()

        f = self.c_hist.clear(); ax = f.add_subplot(111); Chart.style(ax)
        bins = np.linspace(0, 1, 21)
        ax.hist(R["Xavf"], bins=bins, color=C["primary"], alpha=0.85, edgecolor="white")
        ax.axvline(0.3, color=C["mid"], ls="--", lw=1); ax.axvline(0.6, color=C["high"], ls="--", lw=1)
        ax.set_title("Past o'zlashtirish ehtimolining taqsimoti"); ax.set_xlabel("ehtimol", fontsize=8); ax.set_ylabel("talabalar", fontsize=8)
        f.tight_layout(); self.c_hist.draw()

        f = self.c_course.clear(); ax = f.add_subplot(111); Chart.style(ax)
        g = R.groupby("Kurs")["Xavf"].apply(lambda s: (s >= 0.6).mean() * 100)
        ax.bar([f"{k}-kurs" for k in g.index], g.values, color=C["accent"], width=0.55)
        for i, v in enumerate(g.values):
            ax.text(i, v + 0.5, f"{v:.0f}%", ha="center", fontsize=8)
        ax.set_title("Kurslar bo'yicha ehtimoli yuqori talabalar ulushi")
        f.tight_layout(); self.c_course.draw()

        f = self.c_topic.clear(); ax = f.add_subplot(111); Chart.style(ax)
        counts = {E.TOPIC_NAMES[t]: int(R[f"_t_{t}"].sum()) for t in E.TOPICS}
        s = pd.Series(counts).sort_values()
        ax.barh(s.index, s.values, color=C["primary"], height=0.6)
        ax.set_title("Fikrlarda aniqlangan muammolar")
        f.tight_layout(); self.c_topic.draw()


class StudentsPage(QWidget):
    def __init__(self, on_open):
        super().__init__(); self.setObjectName("page")
        self.on_open = on_open
        lay = QVBoxLayout(self); lay.setContentsMargins(20, 18, 20, 18); lay.setSpacing(10)
        lay.addWidget(label("Talabalar ro'yxati", "h1"))
        lay.addWidget(label("Ustun sarlavhasini bosib saralang. Talaba ustiga ikki marta bosing - kartasi ochiladi.", "muted"))
        f = QHBoxLayout(); f.setSpacing(8)
        self.q = QLineEdit(); self.q.setPlaceholderText("Ism yoki ID bo'yicha qidirish...")
        self.fac = QComboBox(); self.grp = QComboBox(); self.lvl = QComboBox()
        self.lvl.addItems(["Barcha darajalar", "Yuqori", "O'rta", "Past"])
        for w in (self.q, self.fac, self.grp, self.lvl):
            f.addWidget(w)
        f.setStretch(0, 2)
        lay.addLayout(f)
        self.model = DFModel(color_col="Daraja", pct_cols=("Xavf",))
        self.view = QTableView(); self.view.setModel(self.model); self.view.setSortingEnabled(False)
        self.view.setSelectionBehavior(QTableView.SelectRows); self.view.verticalHeader().setVisible(False)
        self.view.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.view.horizontalHeader().setStretchLastSection(True)
        self.view.horizontalHeader().sectionClicked.connect(self._sort)
        self.view.doubleClicked.connect(self._open)
        lay.addWidget(self.view, 1)
        self.count = label("", "muted"); lay.addWidget(self.count)
        self.R = None; self.sort_col = "Xavf"; self.sort_asc = False
        self.q.textChanged.connect(self.refresh); self.fac.currentIndexChanged.connect(self._fac_changed)
        self.grp.currentIndexChanged.connect(self.refresh); self.lvl.currentIndexChanged.connect(self.refresh)

    COLS = ["ID", "F.I.Sh.", "Guruh", "Fakultet kodi", "Kurs", "Xavf", "Daraja", "Prognoz ball", "Asosiy sabab", "Kayfiyat"]

    def set_results(self, R):
        self.R = R
        self.fac.blockSignals(True); self.fac.clear()
        self.fac.addItems(["Barcha fakultetlar"] + sorted(R["Fakultet"].astype(str).unique()))
        self.fac.blockSignals(False)
        self._fac_changed()

    def _fac_changed(self):
        if self.R is None:
            return
        R = self.R if self.fac.currentIndex() <= 0 else self.R[self.R["Fakultet"] == self.fac.currentText()]
        self.grp.blockSignals(True); self.grp.clear()
        self.grp.addItems(["Barcha guruhlar"] + sorted(R["Guruh"].astype(str).unique()))
        self.grp.blockSignals(False)
        self.refresh()

    def _sort(self, i):
        col = self.COLS[i]
        self.sort_asc = (not self.sort_asc) if col == self.sort_col else (col not in ("Xavf", "Prognoz ball"))
        self.sort_col = col
        self.refresh()

    def refresh(self):
        if self.R is None:
            return
        R = self.R
        if self.fac.currentIndex() > 0:
            R = R[R["Fakultet"] == self.fac.currentText()]
        if self.grp.currentIndex() > 0:
            R = R[R["Guruh"] == self.grp.currentText()]
        if self.lvl.currentIndex() > 0:
            R = R[R["Daraja"] == self.lvl.currentText()]
        q = self.q.text().strip().lower()
        if q:
            R = R[R["F.I.Sh."].str.lower().str.contains(q, regex=False) | R["ID"].str.lower().str.contains(q, regex=False)]
        R = R.sort_values(self.sort_col, ascending=self.sort_asc)
        self.model.set(R[self.COLS])
        widths = [110, 180, 95, 95, 45, 70, 125, 95, 220]
        for i, w in enumerate(widths):
            self.view.setColumnWidth(i, w)
        self.count.setText(f"Ko'rsatilmoqda: {len(R)} ta talaba")

    def _open(self, idx):
        sid = self.model.df.iloc[idx.row()]["ID"]
        self.on_open(sid)


class StudentCardPage(QWidget):
    def __init__(self):
        super().__init__(); self.setObjectName("page")
        outer = QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QFrame.NoFrame)
        inner = QWidget(); inner.setObjectName("page"); scroll.setWidget(inner); outer.addWidget(scroll)
        lay = QVBoxLayout(inner); lay.setContentsMargins(20, 18, 20, 18); lay.setSpacing(12)
        top = QHBoxLayout()
        self.title = label("Talaba kartasi", "h1"); top.addWidget(self.title, 1)
        self.pick = QComboBox(); self.pick.setMinimumWidth(320); self.pick.setEditable(True)
        self.pick.setInsertPolicy(QComboBox.NoInsert)
        top.addWidget(label("Talaba:", "muted")); top.addWidget(self.pick)
        lay.addLayout(top)
        self.info = label("Ro'yxatdan talabani tanlang.", "muted", wrap=True); lay.addWidget(self.info)

        row = QHBoxLayout(); row.setSpacing(12)
        self.risk_lbl = label("—", "kpi"); self.level_lbl = label("", "h2")
        self.score_lbl = label("—", "kpi"); self.score_lbl.setStyleSheet(f"color:{C['accent']}")
        row.addWidget(card(label("Past o'zlashtirish ehtimoli", "muted"), self.risk_lbl, self.level_lbl))
        row.addWidget(card(label("Prognoz qilingan yakuniy ball", "muted"), self.score_lbl))
        self.c_contrib = Chart(4.2, 2.2)
        row.addWidget(card(label("Qaysi ma'lumot qarorga ko'proq ta'sir qildi", "h2"), self.c_contrib), 2)
        lay.addLayout(row)

        self.c_weeks = Chart(9, 3.2)
        lay.addWidget(card(label("Haftalik o'zlashtirish va modelning e'tibori (attention)", "h2"), self.c_weeks))

        row2 = QHBoxLayout(); row2.setSpacing(12)
        self.text = QTextBrowser(); self.text.setMinimumHeight(220)
        row2.addWidget(card(label("Talaba fikri: transformer e'tibor bergan so'zlar", "h2"), self.text), 1)
        self.recs = QTextBrowser(); self.recs.setMinimumHeight(220)
        row2.addWidget(card(label("Tavsiya etiladigan choralar", "h2"), self.recs), 1)
        lay.addLayout(row2)
        lay.addStretch(1)
        self.app = None
        self.pick.activated.connect(lambda i: self.app and self.app.open_student(self.pick.itemData(i)))

    def set_students(self, R):
        self.pick.blockSignals(True); self.pick.clear()
        for sid, name, g, r in R.sort_values("Xavf", ascending=False)[["ID", "F.I.Sh.", "Guruh", "Xavf"]].itertuples(index=False):
            self.pick.addItem(f"{name} — {g} ({r*100:.0f}%)", sid)
        self.pick.blockSignals(False)

    def show_student(self, row, wk, ex, recs, week):
        i = self.pick.findData(row["ID"])
        if i >= 0:
            self.pick.setCurrentIndex(i)
        self.title.setText(row["F.I.Sh."])
        self.info.setText(f"{row['ID']}  •  {row['Fakultet']}  •  {row['Kurs']}-kurs  •  guruh {row['Guruh']}  •  "
                          f"{row['_shakl']}  •  {'ishlaydi' if row['_ish'] else 'ishlamaydi'}  •  "
                          f"tahlil: 1–{week}-haftalar")
        lvl = E.risk_level(ex["risk"])
        self.risk_lbl.setText(f"{ex['risk']*100:.0f}%"); self.risk_lbl.setStyleSheet(f"color:{LEVEL_COLOR[lvl]}")
        self.level_lbl.setText(f"Ehtimol darajasi: {lvl}"); self.level_lbl.setStyleSheet(f"color:{LEVEL_COLOR[lvl]}")
        self.score_lbl.setText(f"{row['Prognoz ball']:.0f}")

        f = self.c_contrib.clear(); ax = f.add_subplot(111); Chart.style(ax)
        items = list(ex["contrib"].items())
        vals = [v * 100 for _, v in items]
        ax.barh([k for k, _ in items], vals, color=[C["high"] if v > 0 else C["low"] for v in vals], height=0.55)
        ax.axvline(0, color="#999", lw=0.8)
        for y, v in enumerate(vals):
            ax.text(v + (2 if v >= 0 else -2), y, f"{v:+.0f}%", va="center", ha="left" if v >= 0 else "right", fontsize=8)
        ax.set_xlabel("xulosadagi nisbiy ulush, % (+ ehtimolni oshiradi, − kamaytiradi)", fontsize=8)
        lim = max(20, max(abs(v) for v in vals) * 1.35); ax.set_xlim(-lim, lim)
        f.tight_layout(); self.c_contrib.draw()

        f = self.c_weeks.clear()
        ax = f.add_subplot(111); Chart.style(ax)
        w = wk[wk["hafta"] <= week].sort_values("hafta")
        x = w["hafta"].values
        ax.bar(x, w["davomat"].values * 100, color="#C9D7EC", width=0.6, label="Davomat, %")
        ax.plot(x, w["baho"].values, color=C["primary"], marker="o", ms=4, lw=2, label="Joriy baho")
        ax.plot(x, w["topshiriq"].values * 100, ls="none", marker="s", ms=4, color=C["accent"], label="Topshiriq (100 = topshirgan)")
        ax.axvspan(week + 0.5, 16.5, color="#EEEEEE", alpha=0.7, lw=0)
        if week < 16:
            ax.text((week + 16.5) / 2 + 0.25, 50, "semestrning qolgan haftalari\n(hali ma'lum emas)", ha="center", fontsize=8, color="#888")
        ax.set_xlim(0.5, 16.5); ax.set_ylim(0, 112); ax.set_xticks(range(1, 17))
        ax.set_xlabel("o'quv haftasi", fontsize=8)
        ax2 = ax.twinx()
        att = ex["week_attn"][:16].copy(); att[week:] = 0
        ax2.fill_between(range(1, 17), 0, att / max(att.max(), 1e-6), step="mid", color=C["high"], alpha=0.12, label="Model e'tibori")
        ax2.set_ylim(0, 3); ax2.set_yticks([]); ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
        f.tight_layout(); self.c_weeks.draw()

        words, att = ex["words"], ex["text_attn"]
        if words:
            lo, hi = float(att.min()), float(att.max())
            spans = []
            for wd, a in zip(words, att):
                alpha = 0.15 + 0.85 * (float(a) - lo) / max(hi - lo, 1e-6)
                spans.append(f"<span style='background: rgba(214,69,69,{alpha*0.55:.2f}); padding:2px; border-radius:3px'>{wd}</span>")
            topics = [E.TOPIC_NAMES[t] for t, p in zip(E.TOPICS, ex["topics"]) if p >= 0.5]
            sent = E.SENTIMENTS[int(np.argmax(ex["sentiment"]))]
            html = (f"<p style='color:#6B7489'>Asl matn:</p><p><i>{row['_fikr']}</i></p>"
                    f"<p style='color:#6B7489'>So'zlar (qizil - model ko'proq e'tibor bergan):</p>"
                    f"<p style='font-size:11pt; line-height:170%'>{' '.join(spans)}</p>"
                    f"<p><b>Kayfiyat:</b> {E.SENT_NAMES[sent]} &nbsp;&nbsp; <b>Mavzular:</b> {', '.join(topics) or 'aniqlanmadi'}</p>")
        else:
            html = "<p style='color:#6B7489'>Talaba fikr qoldirmagan. Model faqat raqamli ma'lumotlar asosida xulosa qildi.</p>"
        self.text.setHtml(html)

        items = "".join(f"<li style='margin-bottom:8px'><b>{t}.</b> {d}</li>" for t, d in recs)
        self.recs.setHtml(f"<ol style='margin-left:-14px'>{items}</ol>")


class TextPage(QWidget):
    def __init__(self):
        super().__init__(); self.setObjectName("page")
        lay = QVBoxLayout(self); lay.setContentsMargins(20, 18, 20, 18); lay.setSpacing(12)
        lay.addWidget(label("Matn tahlili", "h1"))
        lay.addWidget(label("Talabalar so'rovnomadagi fikrlari transformer yordamida mavzu va kayfiyat bo'yicha tasniflandi.", "muted"))
        row = QHBoxLayout(); row.setSpacing(12)
        self.c_sent = Chart(4, 2.8); self.c_risk = Chart(6, 2.8)
        row.addWidget(card(self.c_sent), 2); row.addWidget(card(self.c_risk), 3)
        lay.addLayout(row)
        self.filter = QComboBox(); lay.addWidget(self.filter)
        self.model = DFModel(); self.view = QTableView(); self.view.setModel(self.model)
        self.view.verticalHeader().setVisible(False); self.view.horizontalHeader().setStretchLastSection(True)
        self.view.setWordWrap(True)
        lay.addWidget(self.view, 1)
        self.R = None
        self.filter.currentIndexChanged.connect(self.refresh)

    def update_view(self, R):
        self.R = R
        T = R[R["_fikr"].str.len() > 0]
        f = self.c_sent.clear(); ax = f.add_subplot(111)
        vc = T["Kayfiyat"].value_counts().reindex(["Salbiy", "Neytral", "Ijobiy"], fill_value=0)
        ax.pie(vc.values, labels=[f"{k}\n{v}" for k, v in vc.items()], colors=[C["high"], "#B8C0CE", C["low"]],
               startangle=90, wedgeprops=dict(width=0.42, edgecolor="white"), textprops=dict(fontsize=8))
        ax.set_title("Fikrlar kayfiyati", fontsize=10, fontweight="bold")
        f.tight_layout(); self.c_sent.draw()

        f = self.c_risk.clear(); ax = f.add_subplot(111); Chart.style(ax)
        names, vals, cnts = [], [], []
        for t in E.TOPICS:
            m = R[f"_t_{t}"] == 1
            if m.sum() >= 3:
                names.append(E.TOPIC_NAMES[t]); vals.append(R.loc[m, "Xavf"].mean() * 100); cnts.append(int(m.sum()))
        order = np.argsort(vals)
        ax.barh([names[i] for i in order], [vals[i] for i in order], color=C["mid"], height=0.6)
        for y, i in enumerate(order):
            ax.text(vals[i] + 1, y, f"{vals[i]:.0f}%  (n={cnts[i]})", va="center", fontsize=7)
        ax.set_xlim(0, 115)
        ax.set_title("Muammo turi bo'yicha o'rtacha past o'zlashtirish ehtimoli")
        f.tight_layout(); self.c_risk.draw()

        self.filter.blockSignals(True); self.filter.clear()
        self.filter.addItem("Barcha fikrlar", None)
        for t in E.TOPICS:
            self.filter.addItem(E.TOPIC_NAMES[t], t)
        self.filter.blockSignals(False)
        self.refresh()

    def refresh(self):
        if self.R is None:
            return
        T = self.R[self.R["_fikr"].str.len() > 0]
        t = self.filter.currentData()
        if t:
            T = T[T[f"_t_{t}"] == 1]
        D = T[["F.I.Sh.", "Guruh", "Kayfiyat", "Mavzular", "_fikr"]].rename(columns={"_fikr": "Fikr matni"})
        self.model.set(D)
        for i, w in enumerate([170, 95, 75, 230]):
            self.view.setColumnWidth(i, w)


class ModelPage(QWidget):
    def __init__(self, meta, res):
        super().__init__(); self.setObjectName("page")
        outer = QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QFrame.NoFrame)
        inner = QWidget(); inner.setObjectName("page"); scroll.setWidget(inner); outer.addWidget(scroll)
        lay = QVBoxLayout(inner); lay.setContentsMargins(20, 18, 20, 18); lay.setSpacing(12)
        lay.addWidget(label("Model haqida", "h1"))
        info = res.get("info", {})
        params = f"{info['params']:,}".replace(",", " ") if "params" in info else "—"
        arch = QTextBrowser(); arch.setMinimumHeight(330)
        arch.setHtml(f"""
<p><b>Heterogen transformer.</b> Turli tabiatli ma'lumotlar bitta ketma-ketlikka tokenlar sifatida joylanadi:</p>
<p style='font-family:Consolas,monospace; background:#F4F6FA; padding:6px'>[CLS] [STATIK] [H1] … [H16] [so'z1] … [so'zN]</p>
<ul>
<li><b>STATIK</b> — fakultet, kurs, kirish bali, ta'lim shakli, yotoqxona, ish bilan bandlik;</li>
<li><b>H1…H16</b> — har bir o'quv haftasi: davomat, joriy baho, LMS faollik, topshiriq;</li>
<li><b>so'zlar</b> — talabaning so'rovnomadagi fikri.</li>
</ul>
<p>Har bir token turi alohida chiziqli proyeksiya orqali {meta.get('d',64)} o'lchamli umumiy fazoga o'tkaziladi va tur hamda
pozitsiya embeddinglari qo'shiladi. So'ngra {meta.get('layers',2)} qatlamli, {meta.get('heads',4)} boshli self-attention
bloklari barcha tokenlarni o'zaro bog'laydi. Chiqishlar: CLS tokenidan — <b>past o'zlashtirish ehtimoli</b> va
<b>yakuniy ball prognozi</b>; matn tokenlaridan — <b>mavzular</b> (7 ta) va <b>kayfiyat</b> (3 sinf).
Model ko'p vazifali (multi-task) usulda o'qitilgan; o'qitishda semestr tasodifiy haftada kesilgani uchun
model semestr boshidanoq erta prognoz bera oladi.</p>
<p><b>Qaror qabul qilishni qo'llab-quvvatlash:</b> har bir talaba uchun okklyuziya usulida ma'lumot turlarining
hissasi hisoblanadi, attention vaznlari esa muhim haftalar va so'zlarni ko'rsatadi; shular asosida tavsiyalar shakllantiriladi.</p>
<p style='color:#6B7489'>O'qitish: {meta.get('trained_on','')}. Validatsiya 2024, test 2025-yil ma'lumotlari.
Parametrlar soni: {params}. Lug'at: {info.get('vocab','—')} so'z.</p>""")
        lay.addWidget(card(label("Arxitektura", "h2"), arch))

        self.c_f1 = Chart(9, 3)
        lay.addWidget(card(label("Erta prognoz: F1 o'lchovi semestr haftalari bo'yicha (test: 2025-yil)", "h2"), self.c_f1))
        tbl = QTableWidget(); tbl.setMinimumHeight(190)
        lay.addWidget(card(label("16-hafta holatida modellarni solishtirish (test to'plami)", "h2"), tbl))
        abl = QTableWidget(); abl.setMinimumHeight(170)
        lay.addWidget(card(label("Ablatsiya tahlili: har bir ma'lumot turining ahamiyati", "h2"), abl))
        lay.addStretch(1)
        if res:
            self._fill(res, tbl, abl)

    @staticmethod
    def _table(tbl, rows, cols):
        tbl.setColumnCount(len(cols)); tbl.setRowCount(len(rows)); tbl.setHorizontalHeaderLabels(cols)
        tbl.verticalHeader().setVisible(False)
        for r, vals in enumerate(rows):
            for c, v in enumerate(vals):
                it = QTableWidgetItem(v if isinstance(v, str) else f"{v:.3f}")
                if r == 0:
                    f = QFont(); f.setBold(True); it.setFont(f)
                tbl.setItem(r, c, it)
        tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        tbl.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)

    def _fill(self, res, tbl, abl):
        keys = ["accuracy", "precision", "recall", "f1", "auc"]
        cols = ["Model", "Aniqlik (Accuracy)", "Precision", "Recall", "F1", "ROC-AUC"]
        tr = res["transformer"]; bl = res["baseline"]
        wks = sorted(int(k) for k in tr)
        last = str(wks[-1])
        rows = [["Heterogen transformer (taklif etilgan)"] + [tr[last][k] for k in keys]]
        for name, m in bl[last].items():
            rows.append([name] + [m[k] for k in keys])
        self._table(tbl, rows, cols)
        rows = [[name] + [m[k] for k in keys] for name, m in res["ablation"].items()]
        self._table(abl, rows, ["Variant"] + cols[1:])

        f = self.c_f1.clear(); ax = f.add_subplot(111); Chart.style(ax)
        ax.plot(wks, [tr[str(w)]["f1"] for w in wks], marker="o", lw=2.5, color=C["high"], label="Heterogen transformer")
        palette = [C["primary"], C["accent"], C["mid"]]
        for (name, _), col in zip(bl[last].items(), palette):
            ax.plot(wks, [bl[str(w)][name]["f1"] for w in wks], marker="o", lw=1.5, color=col, label=name)
        ax.set_xticks(wks); ax.set_xticklabels([f"{w}-hafta" for w in wks])
        ax.set_ylabel("F1", fontsize=8); ax.legend(fontsize=8, frameon=False)
        f.tight_layout(); self.c_f1.draw()


# ============================================================================ asosiy oyna

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} — ta'limda qaror qabul qilishni qo'llab-quvvatlash tizimi")
        icon = os.path.join(BASE, "app", "icon.png")
        if os.path.exists(icon):
            self.setWindowIcon(QIcon(icon))
        self.model = E.HeteroTransformer(os.path.join(BASE, "model", "model.npz"))
        try:
            with open(os.path.join(BASE, "model", "natijalar.json"), encoding="utf-8") as f:
                self.res = json.load(f)
        except OSError:
            self.res = {}
        self.st = self.wk = self.R = None
        self.arrays = None

        tb = QToolBar(); tb.setMovable(False); self.addToolBar(tb)
        title = label(f"  {APP_NAME}", "h2"); title.setStyleSheet(f"color:{C['primary']}; font-size:13pt"); tb.addWidget(title)
        tb.addSeparator()
        b_open = QPushButton("Faylni ochish"); b_open.setObjectName("ghost"); b_open.clicked.connect(self.open_file)
        b_demo = QPushButton("Namuna ma'lumotlar"); b_demo.setObjectName("ghost"); b_demo.clicked.connect(self.load_demo)
        tb.addWidget(b_open); tb.addWidget(b_demo); tb.addSeparator()
        tb.addWidget(label(" Tahlil haftasi: ", "muted"))
        self.week = QSpinBox(); self.week.setRange(1, 16); self.week.setValue(8); self.week.setSuffix(" / 16")
        tb.addWidget(self.week)
        b_run = QPushButton("Tahlil qilish"); b_run.clicked.connect(self.run); tb.addWidget(b_run)
        spacer = QWidget(); spacer.setSizePolicy(spacer.sizePolicy().horizontalPolicy().Expanding, spacer.sizePolicy().verticalPolicy())
        tb.addWidget(spacer)
        b_rep = QPushButton("Hisobotni Excelga saqlash"); b_rep.setObjectName("ghost"); b_rep.clicked.connect(self.export)
        tb.addWidget(b_rep)
        self.status = label("Ma'lumot yuklanmagan", "muted")
        self.statusBar().addWidget(self.status)

        self.nav = QListWidget(); self.nav.setObjectName("nav"); self.nav.setFixedWidth(210)
        for t in ["Umumiy ko'rinish", "Talabalar ro'yxati", "Talaba kartasi", "Matn tahlili", "Model haqida"]:
            self.nav.addItem(QListWidgetItem(t))
        self.pages = QStackedWidget()
        self.p_over = OverviewPage(); self.p_list = StudentsPage(self.open_student)
        self.p_card = StudentCardPage(); self.p_card.app = self
        self.p_text = TextPage(); self.p_model = ModelPage(self.model.meta, self.res)
        for p in (self.p_over, self.p_list, self.p_card, self.p_text, self.p_model):
            self.pages.addWidget(p)
        self.nav.currentRowChanged.connect(self.pages.setCurrentIndex)
        self.nav.setCurrentRow(0)
        body = QWidget(); h = QHBoxLayout(body); h.setContentsMargins(0, 0, 0, 0); h.setSpacing(0)
        h.addWidget(self.nav); h.addWidget(self.pages, 1)
        self.setCentralWidget(body)
        self.resize(1360, 860)

    # ---------------------------------------------------------------- ma'lumot
    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Ma'lumot faylini tanlang", "", "Excel yoki CSV (*.xlsx *.xls *.csv)")
        if path:
            self._load(path)

    def load_demo(self):
        self._load(os.path.join(BASE, "data", "namuna_2025.xlsx"))

    def _load(self, path):
        try:
            self.st, self.wk = E.load_dataset(path)
        except Exception as e:
            QMessageBox.critical(self, "Xatolik", f"Faylni o'qib bo'lmadi:\n{e}")
            return
        mx = int(self.wk["hafta"].max()) if len(self.wk) else 16
        self.week.setValue(min(max(mx, 1), 16) if mx < 16 else 8)
        self.status.setText(f"Yuklandi: {os.path.basename(path)} — {len(self.st)} talaba, {len(self.wk)} haftalik yozuv")
        self.run()

    # ---------------------------------------------------------------- tahlil
    def run(self):
        if self.st is None:
            QMessageBox.information(self, APP_NAME, "Avval ma'lumot faylini oching yoki «Namuna ma'lumotlar»ni tanlang.")
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            wkn = int(self.week.value())
            arr = E.build_arrays(self.st, self.wk, self.model.vocab, wkn)
            self.arrays = arr
            out = self.model.predict(*arr)
            st_, wk_, wm_, ti_, tm_ = arr
            self.ref = E.reference_profile(st_, wk_, wm_)
            c_st, c_wk, c_tx = self.model.contributions(*arr, self.ref)
            self.R = self._results(out, c_st, c_wk, c_tx, wkn)
            self.p_over.update_view(self.R, wkn)
            self.p_list.set_results(self.R)
            self.p_card.set_students(self.R)
            self.p_text.update_view(self.R)
            self.status.setText(self.status.text().split("  |")[0] + f"  |  Tahlil: 1–{wkn}-haftalar")
        except Exception:
            QMessageBox.critical(self, "Xatolik", traceback.format_exc())
        finally:
            QApplication.restoreOverrideCursor()

    def _results(self, out, c_st, c_wk, c_tx, wkn):
        st = self.st
        wk = self.wk[self.wk["hafta"] <= wkn]
        agg = wk.groupby("talaba_id").agg(att=("davomat", "mean"), sub=("topshiriq", "mean"), lms=("lms_daqiqa", "mean"))
        reasons = []
        for i, sid in enumerate(st["talaba_id"]):
            parts = {"st": c_st[i], "wk": c_wk[i], "tx": c_tx[i]}
            key = max(parts, key=parts.get)
            r = out["risk"][i]
            if r < 0.3:
                reasons.append("—"); continue
            if key == "tx" and out["has_text"][i]:
                tp = out["topics"][i]
                j = int(np.argmax(tp))
                reasons.append(f"Fikr: {E.TOPIC_NAMES[E.TOPICS[j]].lower()}" if tp[j] >= 0.5 else "Fikrdagi salbiy holat")
            elif key == "st":
                reasons.append("Shaxsiy omillar (ish, ta'lim shakli, kirish bali)")
            else:
                a = agg.loc[sid] if sid in agg.index else None
                if a is None:
                    reasons.append("Haftalik ma'lumot yo'q")
                elif a["att"] < 0.75:
                    reasons.append(f"Davomat past ({a['att']*100:.0f}%)")
                elif a["sub"] < 0.6:
                    reasons.append(f"Topshiriqlar topshirilmagan ({a['sub']*100:.0f}%)")
                elif a["lms"] < 30:
                    reasons.append("LMS faollik past")
                else:
                    reasons.append("Joriy baholar past")
        sent_idx = out["sentiment"].argmax(1)
        R = pd.DataFrame({
            "ID": st["talaba_id"].astype(str), "F.I.Sh.": st["fish"].astype(str), "Guruh": st["guruh"].astype(str),
            "Fakultet": st["fakultet"].astype(str),
            "Fakultet kodi": [E.fac_code_of(r) or "—" for r in st.to_dict("records")],
            "Kurs": st["kurs"].astype(int), "Xavf": out["risk"].astype(float),
            "Daraja": [E.risk_level(p) for p in out["risk"]], "Prognoz ball": out["score"].astype(float),
            "Asosiy sabab": reasons,
            "Kayfiyat": [E.SENT_NAMES[E.SENTIMENTS[k]] if h else "—" for k, h in zip(sent_idx, out["has_text"])],
            "Mavzular": [", ".join(E.TOPIC_NAMES[t] for t, p in zip(E.TOPICS, tp) if p >= 0.5) for tp in out["topics"]],
            "_fikr": st["fikr"].astype(str), "_shakl": st["talim_shakli"].astype(str), "_ish": st["ishlaydi"].astype(int),
        })
        for j, t in enumerate(E.TOPICS):
            R[f"_t_{t}"] = (out["topics"][:, j] >= 0.5).astype(int)
        return R

    def open_student(self, sid):
        if self.R is None or sid is None:
            return
        i = int(np.where(self.R["ID"].values == sid)[0][0])
        row = self.R.iloc[i]
        sl = slice(i, i + 1)
        ex = self.model.explain(*[a[sl] for a in self.arrays], self.ref)
        wkn = int(self.week.value())
        wk_all = self.wk[self.wk["talaba_id"] == sid]
        recs = E.recommendations(row, wk_all, ex["risk"], ex["topics"], wkn)
        self.p_card.show_student(row, wk_all[wk_all["hafta"] <= 16], ex, recs, wkn)
        self.nav.setCurrentRow(2)

    # ---------------------------------------------------------------- hisobot
    def export(self):
        if self.R is None:
            QMessageBox.information(self, APP_NAME, "Avval tahlil qiling.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Hisobotni saqlash", "talabatahlil_hisobot.xlsx", "Excel (*.xlsx)")
        if not path:
            return
        try:
            self.save_report(path)
            QMessageBox.information(self, APP_NAME, f"Hisobot saqlandi:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Xatolik", str(e))

    def save_report(self, path):
        wkn = int(self.week.value())
        R = self.R.sort_values("Xavf", ascending=False)
        main_cols = ["ID", "F.I.Sh.", "Fakultet", "Guruh", "Kurs", "Xavf", "Daraja", "Prognoz ball", "Asosiy sabab", "Kayfiyat", "Mavzular"]
        T = R[main_cols].copy(); T["Xavf"] = (T["Xavf"] * 100).round(1); T["Prognoz ball"] = T["Prognoz ball"].round(1)
        T = T.rename(columns={"Xavf": "Past o'zlashtirish ehtimoli, %", "Daraja": "Ehtimol darajasi"})
        recs = []
        for _, row in R[R["Xavf"] >= 0.3].iterrows():
            i = int(np.where(self.R["ID"].values == row["ID"])[0][0])
            tp = np.array([row[f"_t_{t}"] for t in E.TOPICS], float)
            for t, d in E.recommendations(row, self.wk[self.wk["talaba_id"] == row["ID"]], row["Xavf"], tp, wkn):
                recs.append(dict(ID=row["ID"], **{"F.I.Sh.": row["F.I.Sh."], "Guruh": row["Guruh"], "Ehtimol darajasi": row["Daraja"],
                                                  "Yo'nalish": t, "Tavsiya": d}))
        S = R.groupby(["Fakultet", "Guruh"]).agg(**{
            "Talabalar": ("ID", "count"),
            "Ehtimoli yuqori": ("Daraja", lambda s: int((s == "Yuqori").sum())),
            "Ehtimoli o'rta": ("Daraja", lambda s: int((s == "O'rta").sum())),
            "O'rtacha ehtimol, %": ("Xavf", lambda s: round(s.mean() * 100, 1)),
            "O'rtacha prognoz ball": ("Prognoz ball", lambda s: round(s.mean(), 1)),
        }).reset_index()
        with pd.ExcelWriter(path, engine="openpyxl") as xw:
            T.to_excel(xw, sheet_name="Talabalar", index=False)
            pd.DataFrame(recs).to_excel(xw, sheet_name="Tavsiyalar", index=False)
            S.to_excel(xw, sheet_name="Guruhlar", index=False)
            for ws in xw.book.worksheets:
                for col in ws.columns:
                    width = max(len(str(c.value or "")) for c in col[:200])
                    ws.column_dimensions[col[0].column_letter].width = min(max(10, width + 2), 70)
                ws.freeze_panes = "A2"


def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setStyleSheet(QSS)
    f = app.font(); f.setPointSize(10); app.setFont(f)
    w = MainWindow()
    w.show()
    if "--demo" in sys.argv:
        w.load_demo()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
