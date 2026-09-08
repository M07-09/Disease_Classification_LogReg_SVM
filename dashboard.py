"""
Disease Classification Using Logistic Regression and SVM — interactive dashboard.
One page per assignment part (1-6), explained for readers with no AI background.

Run:  streamlit run dashboard.py
"""

from pathlib import Path
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_recall_fscore_support,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "breast_cancer_dataset.csv.xlsx"
TEST_SIZE, SEED = 0.30, 42

LR, SVM = "Logistic Regression", "SVM"
# Colour is reserved for meaning: which model, and which diagnosis.
MODEL_COLOR = {LR: "#2a78d6", SVM: "#eb6834"}
CLASS_COLOR = {0: "#199e70", 1: "#e34948"}
CLASS_DEEP = {0: "#0f766e", 1: "#b91c1c"}  # darker shades, used behind white text
CLASS_NAME = {0: "Healthy (0)", 1: "Diseased (1)"}
CLASS_LONG = {0: "Healthy (0) — benign tumour", 1: "Diseased (1) — malignant tumour"}
CLASS_SHORT = {0: "Healthy", 1: "Diseased"}

BG, SURFACE = "#f9f9f7", "#fcfcfb"
BORDER = "rgba(11,11,11,.10)"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"

# One calm blue family for structural fills that carry no clinical meaning.
BLUE_PALE, BLUE_SOFT, BLUE_MID = "#e8f1fd", "#cde2fb", "#9ec5f4"
DIVERGING = [[0, "#2a78d6"], [0.5, "#f0efec"], [1, "#e34948"]]
# Confusion-matrix tints: calm fills, dark text, meaning carried by the labels too.
CM_FILL = {"correct": "#d7ece2", "false_alarm": "#fbe6cb", "missed": "#f7d6d6"}

MEASUREMENTS = {
    "radius": "how large the cell nucleus is (distance from its centre to its edge)",
    "texture": "how uneven the shading of the nucleus looks",
    "perimeter": "the length of the border around the nucleus",
    "area": "the surface area of the nucleus",
    "smoothness": "how smooth or bumpy the edge of the nucleus is",
    "compactness": "how tightly packed the shape is",
    "concavity": "how deep the inward dents in the outline are",
    "concave points": "how many inward dents the outline has",
    "symmetry": "how symmetric the two halves of the nucleus are",
    "fractal dimension": "how jagged and complex the outline is",
}
SUFFIX = {
    "mean": "average across all cells in the sample",
    "error": "how much it varies from cell to cell (standard error)",
    "worst": "the largest value seen in the sample",
}

EXAMPLE_HEALTHY = [
    12.0, 15.0, 78.0, 450.0, 0.09, 0.08, 0.03, 0.02, 0.17, 0.06,
    0.3, 1.0, 2.0, 25.0, 0.006, 0.02, 0.02, 0.01, 0.02, 0.003,
    13.5, 20.0, 87.0, 550.0, 0.13, 0.18, 0.12, 0.07, 0.28, 0.08,
]
EXAMPLE_DISEASED = [
    20.5, 21.0, 135.0, 1300.0, 0.11, 0.22, 0.25, 0.15, 0.20, 0.06,
    0.8, 1.2, 5.5, 120.0, 0.007, 0.03, 0.04, 0.02, 0.02, 0.004,
    25.0, 28.0, 165.0, 1900.0, 0.15, 0.45, 0.5, 0.22, 0.32, 0.09,
]


# ------------------------------------------------------------------
# Data and models
# ------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    df = pd.read_excel(DATA_FILE)
    if df.isnull().sum().sum() > 0:
        df = df.fillna(df.mean(numeric_only=True))
    return df


@st.cache_resource(show_spinner=False)
def train() -> dict:
    df = load_data()
    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    scaler = StandardScaler().fit(X_train)
    X_tr, X_te = scaler.transform(X_train), scaler.transform(X_test)

    models = {
        LR: LogisticRegression(max_iter=5000, random_state=SEED).fit(X_tr, y_train),
        SVM: CalibratedClassifierCV(SVC(kernel="rbf", random_state=SEED), ensemble=False).fit(X_tr, y_train),
    }

    results = {}
    for name, model in models.items():
        pred = model.predict(X_te)
        prob = model.predict_proba(X_te)[:, 1]
        p, r, f, s = precision_recall_fscore_support(y_test, pred, labels=[0, 1], zero_division=0)
        fpr, tpr, _ = roc_curve(y_test, prob)
        results[name] = dict(
            acc=accuracy_score(y_test, pred), cm=confusion_matrix(y_test, pred, labels=[0, 1]),
            precision=p, recall=r, f1=f, support=s, auc=roc_auc_score(y_test, prob), fpr=fpr, tpr=tpr,
        )

    # 2-D projection used only to draw the decision boundaries (illustrative)
    X_all_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=SEED)
    X_pca = pca.fit_transform(X_all_scaled)
    xx, yy = np.meshgrid(
        np.linspace(X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1, 220),
        np.linspace(X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1, 220),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_z = {
        LR: LogisticRegression(max_iter=5000).fit(X_pca, y).predict(grid).reshape(xx.shape),
        SVM: SVC(kernel="rbf").fit(X_pca, y).predict(grid).reshape(xx.shape),
    }

    scaled_stats = pd.DataFrame({
        "Before (train mean)": X_train.mean().round(3),
        "Before (train std)": X_train.std().round(3),
        "After (train mean)": pd.Series(X_tr.mean(axis=0), index=X.columns).round(3),
        "After (train std)": pd.Series(X_tr.std(axis=0), index=X.columns).round(3),
    })

    return dict(
        df=df, scaler=scaler, models=models, results=results, feature_names=list(X.columns),
        n_train=len(X_train), n_test=len(X_test), y_train=y_train, y_test=y_test,
        scaled_stats=scaled_stats, X_pca=X_pca, y_all=y.to_numpy(), grid_x=xx, grid_y=yy, grid_z=grid_z,
    )


def pack() -> dict:
    return st.session_state["pack"]


# ------------------------------------------------------------------
# Styling helpers
# ------------------------------------------------------------------
def inject_css():
    # Animations are scoped to the main area only; the sidebar is never animated.
    main = '[data-testid="stMain"]'
    animated = ['[data-testid="stPlotlyChart"]', '[data-testid="stMetric"]', '[data-testid="stExpander"]',
                '[data-testid="stDataFrame"]', '.card', '.explain', '.hero', '.verdict', '.part-head']
    animated_sel = ", ".join(f"{main} {s}" for s in animated)
    stagger = "".join(
        ", ".join(f'{main} [data-testid="stColumn"]:nth-child({n}) {s}' for s in animated)
        + f" {{ animation-delay: {(n - 1) * 90}ms; }}"
        for n in range(2, 7)
    )
    st.markdown(f"""
<style>
@keyframes fadeUp {{ from {{ opacity: 0; transform: translateY(14px); }} to {{ opacity: 1; transform: none; }} }}
@keyframes pop {{ 0% {{ opacity: 0; transform: scale(.96); }} 60% {{ transform: scale(1.015); }} 100% {{ opacity: 1; transform: none; }} }}
{animated_sel} {{ animation: fadeUp .6s cubic-bezier(.2,.7,.2,1) both; }}
{main} .verdict {{ animation-name: pop; animation-duration: .55s; }}
{stagger}
{main} [data-testid="stPlotlyChart"] {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 14px; padding: 10px 8px 4px; }}
[data-testid="stMetric"] {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 14px; padding: 14px 18px; }}
.hero {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 16px; padding: 26px 28px; }}
.hero h1 {{ font-size: 31px; margin: 0 0 10px 0; letter-spacing: -.01em; color: {INK}; }}
.hero p {{ font-size: 16px; color: {INK2}; margin: 0; max-width: 960px; line-height: 1.6; }}
.hero .eyebrow {{ display: block; font-size: 11.5px; font-weight: 700; letter-spacing: .1em; text-transform: uppercase;
  color: {MUTED}; margin-bottom: 10px; }}
.part-head {{ margin: 2px 0 16px 0; }}
.part-head .eyebrow {{ display: block; font-size: 11.5px; font-weight: 700; letter-spacing: .1em; text-transform: uppercase;
  color: {MUTED}; margin-bottom: 4px; }}
.part-head h2 {{ margin: 0; font-size: 28px; color: {INK}; letter-spacing: -.01em; }}
.part-head p {{ margin: 8px 0 0 0; color: {INK2}; font-size: 15.5px; line-height: 1.6; max-width: 980px; }}
.card {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 14px; padding: 18px 20px; margin-bottom: 12px; height: 100%; }}
.card h4 {{ margin: 0 0 8px 0; font-size: 15.5px; color: {INK}; }}
.card p {{ margin: 0; color: {INK2}; font-size: 14.2px; line-height: 1.6; }}
.card code {{ background: #f0efec; padding: 1px 6px; border-radius: 5px; color: #1f2a37; }}
.explain {{ background: #eef4fc; border-left: 4px solid {MODEL_COLOR[LR]}; border-radius: 10px; padding: 12px 16px;
  margin: 6px 0 18px 0; color: #1f2a37; font-size: 14.5px; line-height: 1.65; }}
.explain .tag {{ display: block; font-size: 11.5px; font-weight: 700; letter-spacing: .07em; text-transform: uppercase;
  color: {MODEL_COLOR[LR]}; margin-bottom: 4px; }}
.verdict {{ border-radius: 14px; padding: 22px 24px; color: #fff; margin: 8px 0 14px 0; background: var(--bg, {MODEL_COLOR[LR]}); }}
.verdict h3 {{ margin: 0 0 6px 0; font-size: 23px; color: #fff; }}
.verdict p {{ margin: 0; font-size: 15px; line-height: 1.55; color: rgba(255,255,255,.95); }}
footer {{ visibility: hidden; }}
@media (prefers-reduced-motion: reduce) {{ *, *::before, *::after {{ animation: none !important; transition: none !important; }} }}
</style>""", unsafe_allow_html=True)


def part_header(n: int, title: str, subtitle: str = ""):
    st.markdown(f'<div class="part-head"><span class="eyebrow">Part {n}</span><h2>{title}</h2>'
                f'{f"<p>{subtitle}</p>" if subtitle else ""}</div>', unsafe_allow_html=True)


def explain(text: str, tag: str = "In plain words"):
    st.markdown(f'<div class="explain"><span class="tag">{tag}</span>{text}</div>', unsafe_allow_html=True)


def card(title: str, body: str):
    st.markdown(f'<div class="card"><h4>{title}</h4><p>{body}</p></div>', unsafe_allow_html=True)


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def style(fig: go.Figure, height: int = 380, legend: bool = True) -> go.Figure:
    fig.update_layout(
        template="plotly_white", height=height, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family=FONT, color=INK2, size=13), title_font=dict(size=16, color=INK),
        margin=dict(l=50, r=24, t=62, b=50), showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=GRID, font=dict(family=FONT, size=13, color=INK), align="left"),
    )
    axis = dict(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, tickfont=dict(color=MUTED), title_font=dict(color=INK2))
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


def chart(fig: go.Figure):
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def pct(v: float) -> str:
    return f"{v:.1%}"


# ------------------------------------------------------------------
# Animated HTML components (count-up tiles, stepper, rings, gauges)
# ------------------------------------------------------------------
BASE_HTML_CSS = f"""
html, body {{ margin: 0; background: transparent; font-family: {FONT}; color: {INK}; }}
@keyframes up {{ from {{ opacity: 0; transform: translateY(12px); }} to {{ opacity: 1; transform: none; }} }}
"""
COUNT_JS = """
document.querySelectorAll('.num').forEach(el => {
  const target = +el.dataset.v, d = +el.dataset.d, t0 = performance.now(), dur = 1400;
  const ease = x => 1 - Math.pow(1 - x, 3);
  const tick = now => {
    const p = Math.min(1, (now - t0) / dur);
    el.textContent = (target * ease(p)).toLocaleString('en-US', {minimumFractionDigits: d, maximumFractionDigits: d});
    if (p < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
});"""


def kpi_row(items, height: int = 150):
    cards = "".join(
        f'<div class="k" style="--c:{it["color"]}; animation-delay:{i * 90}ms">'
        f'<div class="lbl">{it["label"]}</div>'
        f'<div class="val"><span class="num" data-v="{it["value"]}" data-d="{it.get("decimals", 0)}">0</span>{it.get("suffix", "")}</div>'
        f'<div class="sub">{it.get("sub", "")}</div></div>'
        for i, it in enumerate(items)
    )
    html = f"""<style>{BASE_HTML_CSS}
.row {{ display: grid; grid-template-columns: repeat({len(items)}, minmax(0, 1fr)); gap: 14px; padding: 2px 2px 10px; }}
.k {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 14px; padding: 16px 18px;
      position: relative; overflow: hidden; animation: up .7s cubic-bezier(.2,.7,.2,1) both; }}
.k::before {{ content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px; background: var(--c); }}
.lbl {{ font-size: 12px; color: {INK2}; text-transform: uppercase; letter-spacing: .06em; font-weight: 600; }}
.val {{ font-size: 30px; font-weight: 700; margin-top: 4px; color: {INK}; line-height: 1.15; }}
.sub {{ font-size: 12.5px; color: {MUTED}; margin-top: 4px; }}
</style>
<div class="row">{cards}</div>
<script>{COUNT_JS}</script>"""
    st.iframe(html, height=height)


def parts_stepper():
    steps = [
        ("Part 1", "Dataset exploration"), ("Part 2", "Data preprocessing"), ("Part 3", "Model development"),
        ("Part 4", "New patient prediction"), ("Part 5", "Visualization"), ("Part 6", "Comparison & discussion"),
    ]
    items = "".join(
        f'<div class="step" style="--d:{0.35 + i * 0.4}s"><div class="dot">{i + 1}</div>'
        f'<div class="t">{t}</div><div class="s">{s}</div></div>'
        for i, (t, s) in enumerate(steps)
    )
    html = f"""<style>{BASE_HTML_CSS}
.wrap {{ position: relative; padding-top: 6px; }}
.track {{ position: absolute; top: 24px; left: 8.3%; right: 8.3%; height: 3px; background: {GRID}; border-radius: 2px; }}
.fill {{ height: 100%; width: 0; background: {MODEL_COLOR[LR]}; border-radius: 2px; animation: grow 2.4s .3s ease-out forwards; }}
@keyframes grow {{ to {{ width: 100%; }} }}
.steps {{ position: relative; display: grid; grid-template-columns: repeat(6, 1fr); gap: 10px; }}
.step {{ text-align: center; }}
.dot {{ width: 38px; height: 38px; border-radius: 50%; background: {SURFACE}; border: 2px solid {AXIS}; margin: 0 auto;
        display: grid; place-items: center; font-weight: 700; color: {MUTED}; animation: light .45s var(--d) forwards; }}
@keyframes light {{ to {{ background: {MODEL_COLOR[LR]}; border-color: {MODEL_COLOR[LR]}; color: #fff; transform: scale(1.06); }} }}
.t {{ margin-top: 10px; font-weight: 700; font-size: 14px; opacity: 0; animation: up .5s var(--d) forwards; color: {INK}; }}
.s {{ margin-top: 3px; font-size: 12.5px; color: {INK2}; line-height: 1.4; opacity: 0; animation: up .5s calc(var(--d) + .1s) forwards; }}
</style>
<div class="wrap"><div class="track"><div class="fill"></div></div><div class="steps">{items}</div></div>"""
    st.iframe(html, height=130)


def ring(percent: float, label: str, sub: str, color: str, height: int = 190):
    circ = 2 * np.pi * 54
    html = f"""<style>{BASE_HTML_CSS}
.wrap {{ display: flex; align-items: center; gap: 18px; background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 14px;
         padding: 14px 18px; animation: up .6s both; }}
.arc {{ transition: stroke-dashoffset 1.4s cubic-bezier(.2,.7,.2,1); }}
.big {{ font-size: 28px; font-weight: 700; color: {INK}; }}
.lbl {{ font-size: 12px; color: {INK2}; text-transform: uppercase; letter-spacing: .06em; font-weight: 600; }}
.sub {{ font-size: 13px; color: {MUTED}; margin-top: 4px; line-height: 1.45; }}
</style>
<div class="wrap">
  <svg viewBox="0 0 130 130" width="126" height="126">
    <circle cx="65" cy="65" r="54" fill="none" stroke="{GRID}" stroke-width="12"/>
    <circle class="arc" cx="65" cy="65" r="54" fill="none" stroke="{color}" stroke-width="12" stroke-linecap="round"
            stroke-dasharray="{circ:.1f}" stroke-dashoffset="{circ:.1f}" transform="rotate(-90 65 65)" data-p="{percent}"/>
    <text x="65" y="72" text-anchor="middle" font-size="23" font-weight="700" fill="{INK}"><tspan class="num" data-v="{percent:.1f}" data-d="1">0</tspan>%</text>
  </svg>
  <div><div class="lbl">{label}</div><div class="big">{percent:.0f}%</div><div class="sub">{sub}</div></div>
</div>
<script>
requestAnimationFrame(() => requestAnimationFrame(() => {{
  document.querySelectorAll('.arc').forEach(el => {{ el.style.strokeDashoffset = {circ:.1f} * (1 - (+el.dataset.p) / 100); }});
}}));
{COUNT_JS}
</script>"""
    st.iframe(html, height=height)


def gauges(probs: dict):
    arc = 251.3  # length of the semicircle path below (radius 80)
    cards = ""
    for i, (name, p) in enumerate(probs.items()):
        cls = int(p >= 0.5)
        cards += f"""
<div class="g" style="animation-delay:{i * 120}ms; --c:{CLASS_COLOR[cls]}">
  <div><span class="chip" style="background:{MODEL_COLOR[name]}">{name}</span></div>
  <svg viewBox="0 0 200 118" width="100%">
    <path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="{GRID}" stroke-width="14" stroke-linecap="round"/>
    <path class="arc" d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="{CLASS_COLOR[cls]}" stroke-width="14"
          stroke-linecap="round" stroke-dasharray="{arc}" stroke-dashoffset="{arc}" data-p="{p}"/>
    <line x1="100" y1="12" x2="100" y2="28" stroke="{MUTED}" stroke-width="2"/>
    <text x="100" y="92" text-anchor="middle" font-size="29" font-weight="700" fill="{INK}"><tspan class="num" data-v="{100 * p:.1f}" data-d="1">0</tspan>%</text>
    <text x="100" y="112" text-anchor="middle" font-size="11" fill="{MUTED}">probability of being diseased</text>
  </svg>
  <div class="verdict">{CLASS_NAME[cls]}</div>
</div>"""
    html = f"""<style>{BASE_HTML_CSS}
.grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; padding: 2px 2px 10px; }}
.g {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 14px; padding: 14px 18px 12px; text-align: center;
      animation: up .6s cubic-bezier(.2,.7,.2,1) both; }}
.chip {{ display: inline-block; padding: 4px 12px; border-radius: 999px; font-size: 12.5px; font-weight: 600; color: #fff; }}
.arc {{ transition: stroke-dashoffset 1.5s cubic-bezier(.2,.7,.2,1); }}
.verdict {{ font-weight: 700; font-size: 17px; margin-top: 2px; color: var(--c); }}
</style>
<div class="grid">{cards}</div>
<script>
requestAnimationFrame(() => requestAnimationFrame(() => {{
  document.querySelectorAll('.arc').forEach(el => {{ el.style.strokeDashoffset = {arc} * (1 - +el.dataset.p); }});
}}));
{COUNT_JS}
</script>"""
    st.iframe(html, height=300)


# ------------------------------------------------------------------
# Charts
# ------------------------------------------------------------------
def best_model_for_class(results: dict, cls: int) -> str:
    a, b = results[LR]["recall"][cls], results[SVM]["recall"][cls]
    if abs(a - b) < 1e-9:
        return f"both models, {pct(a)} of them recognised"
    best = LR if a > b else SVM
    return f"{best} ({pct(max(a, b))} of them recognised)"


def fig_donut(counts: pd.Series, results: dict) -> go.Figure:
    n = int(counts.sum())
    values = [int(counts[0]), int(counts[1])]
    custom = [[values[c], values[c] / n, best_model_for_class(results, c),
               "benign — not cancer" if c == 0 else "malignant — cancer"] for c in (0, 1)]
    fig = go.Figure(go.Pie(
        labels=[CLASS_NAME[0], CLASS_NAME[1]], values=values, hole=0.62, sort=False, direction="clockwise", rotation=-90,
        marker=dict(colors=[CLASS_COLOR[0], CLASS_COLOR[1]], line=dict(color=SURFACE, width=3)),
        textinfo="label+percent", textfont=dict(size=14, color="#fff"), insidetextorientation="horizontal",
        customdata=custom,
        hovertemplate="<b>%{label}</b> · %{customdata[3]}<br>%{customdata[0]} patients — %{customdata[1]:.0%} of the dataset"
                      "<br>Best at recognising this group: %{customdata[2]}<extra></extra>",
    ))
    fig.add_annotation(text=f"<b>{n}</b><br><span style='font-size:12px;color:{MUTED}'>patients</span>", x=0.5, y=0.5,
                       showarrow=False, font=dict(size=28, color=INK))
    style(fig, 400, legend=False)
    fig.update_layout(title="1. Class distribution — hover a slice", margin=dict(l=20, r=20, t=62, b=20))
    return fig


def fig_feature_treemap(names: list) -> go.Figure:
    groups = {"mean": "Mean values", "error": "Error values", "worst": "Worst values"}
    ids, labels, parents, colors, meanings = ["all"], ["30 features"], [""], [SURFACE], ["Hover any tile to read what it measures"]
    for g, glabel in groups.items():
        ids.append(g); labels.append(glabel); parents.append("all"); colors.append(BLUE_MID)
        meanings.append(f"10 measurements — {SUFFIX[g]}")
    for n in names:
        g = "mean" if n.startswith("mean ") else "worst" if n.startswith("worst ") else "error"
        ids.append(n); labels.append(n.replace(f"{g} ", "") if g != "error" else n[: -len(" error")])
        parents.append(g); colors.append(BLUE_PALE); meanings.append(feature_meaning(n))
    fig = go.Figure(go.Treemap(
        ids=ids, labels=labels, parents=parents, customdata=meanings, branchvalues="remainder",
        marker=dict(colors=colors, line=dict(color=SURFACE, width=2), pad=dict(t=28, l=6, r=6, b=6)),
        textfont=dict(size=13, color=INK), tiling=dict(packing="squarify"),
        hovertemplate="<b>%{label}</b><br>%{customdata}<extra></extra>", maxdepth=3,
    ))
    style(fig, 460, legend=False)
    fig.update_layout(title="The 30 features — 10 measurements × 3 versions (hover to read the meaning)",
                      margin=dict(l=10, r=10, t=62, b=10))
    return fig


def fig_scaling_dumbbell(stats: pd.DataFrame) -> go.Figure:
    before = stats["Before (train std)"].sort_values()
    after = stats.loc[before.index, "After (train std)"]
    fig = go.Figure()
    for feat in before.index:
        fig.add_trace(go.Scatter(x=[before[feat], after[feat]], y=[feat, feat], mode="lines",
                                 line=dict(color=AXIS, width=1.5), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=before.values, y=before.index, mode="markers", name="Before scaling",
                             marker=dict(color=MODEL_COLOR[SVM], size=9, line=dict(color=SURFACE, width=1.5)),
                             hovertemplate="<b>%{y}</b><br>spread (std) before scaling: <b>%{x:.3g}</b><extra></extra>"))
    fig.add_trace(go.Scatter(x=after.values, y=after.index, mode="markers", name="After scaling",
                             marker=dict(color=MODEL_COLOR[LR], size=9, line=dict(color=SURFACE, width=1.5)),
                             hovertemplate="<b>%{y}</b><br>spread after scaling: <b>%{x:.2f}</b> — same for every feature<extra></extra>"))
    style(fig, 640)
    fig.update_layout(title="Spread of each feature before → after standardisation (log scale)",
                      xaxis=dict(type="log", title="Standard deviation"), margin=dict(l=190),
                      yaxis=dict(tickfont=dict(size=11)))
    return fig


def fig_split_sunburst(y_train: pd.Series, y_test: pd.Series) -> go.Figure:
    n = len(y_train) + len(y_test)
    ids = ["all", "train", "test", "train0", "train1", "test0", "test1"]
    labels = ["All patients", "Training 70%", "Testing 30%", CLASS_SHORT[0], CLASS_SHORT[1], CLASS_SHORT[0], CLASS_SHORT[1]]
    parents = ["", "all", "all", "train", "train", "test", "test"]
    values = [n, len(y_train), len(y_test), int((y_train == 0).sum()), int((y_train == 1).sum()),
              int((y_test == 0).sum()), int((y_test == 1).sum())]
    colors = [BLUE_PALE, BLUE_MID, BLUE_SOFT, CLASS_COLOR[0], CLASS_COLOR[1], CLASS_COLOR[0], CLASS_COLOR[1]]
    notes = ["the whole dataset", "used to teach the models", "locked away for the exam",
             "healthy patients in the training set", "diseased patients in the training set",
             "healthy patients in the test set", "diseased patients in the test set"]
    fig = go.Figure(go.Sunburst(
        ids=ids, labels=labels, parents=parents, values=values, branchvalues="total",
        marker=dict(colors=colors, line=dict(color=SURFACE, width=2)), customdata=notes,
        textfont=dict(size=13, color=INK), insidetextorientation="radial",
        hovertemplate="<b>%{label}</b>: %{value} patients<br>%{percentRoot:.0%} of all patients · %{customdata}<extra></extra>",
    ))
    style(fig, 440, legend=False)
    fig.update_layout(title="The 70 / 30 split keeps the same healthy-diseased ratio in both sets",
                      margin=dict(l=10, r=10, t=62, b=10))
    return fig


def fig_radar(results: dict) -> go.Figure:
    axes = ["Accuracy", "Precision (Diseased)", "Recall (Diseased)", "F1 (Diseased)", "Precision (Healthy)", "Recall (Healthy)"]
    fig = go.Figure()
    for name, r in results.items():
        vals = [r["acc"], r["precision"][1], r["recall"][1], r["f1"][1], r["precision"][0], r["recall"][0]]
        fig.add_trace(go.Scatterpolar(
            r=vals + vals[:1], theta=axes + axes[:1], name=name, fill="toself",
            line=dict(color=MODEL_COLOR[name], width=2), fillcolor=rgba(MODEL_COLOR[name], 0.14),
            marker=dict(size=6, color=MODEL_COLOR[name]),
            hovertemplate="<b>%{theta}</b>: %{r:.1%}<extra>" + name + "</extra>",
        ))
    style(fig, 460)
    fig.update_layout(
        title="Six scores at once — the bigger the shape, the better the model",
        polar=dict(bgcolor=SURFACE,
                   radialaxis=dict(range=[0.8, 1.0], tickformat=".0%", gridcolor=GRID, linecolor=AXIS, tickfont=dict(color=MUTED)),
                   angularaxis=dict(gridcolor=GRID, linecolor=AXIS, tickfont=dict(color=INK2, size=12))),
        margin=dict(l=70, r=70, t=62, b=40),
    )
    return fig


def fig_corr_heatmap(df: pd.DataFrame) -> go.Figure:
    cols = [c for c in df.columns if c.startswith("mean")] + ["diagnosis"]
    corr = df[cols].corr()
    labels = [c.replace("mean ", "") for c in cols]
    strength = np.vectorize(lambda v: "strong" if abs(v) >= 0.7 else "moderate" if abs(v) >= 0.4 else "weak")(corr.values)
    direction = np.vectorize(lambda v: "positive — they rise together" if v > 0 else "negative — one rises as the other falls")(corr.values)
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=labels, y=labels, colorscale=DIVERGING, zmin=-1, zmax=1, zmid=0,
        text=np.round(corr.values, 2), texttemplate="%{text}", textfont=dict(size=10),
        xgap=2, ygap=2, colorbar=dict(title="corr.", thickness=12, outlinewidth=0, tickfont=dict(color=MUTED)),
        customdata=np.dstack([strength, direction]),
        hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>correlation %{z:.2f} — %{customdata[0]} %{customdata[1]}<extra></extra>",
    ))
    style(fig, 560, legend=False)
    fig.update_layout(title="2. Correlation heatmap (mean-value features)", margin=dict(l=120, b=120))
    fig.update_xaxes(tickangle=-40, showgrid=False)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return fig


def fig_confusion(cm: np.ndarray, name: str) -> go.Figure:
    # z encodes the kind of cell, not the count, so correct answers and mistakes read differently.
    z = [[0, 1], [2, 3]]
    c = CM_FILL
    scale = [[0, c["correct"]], [0.25, c["correct"]], [0.25, c["false_alarm"]], [0.5, c["false_alarm"]],
             [0.5, c["missed"]], [0.75, c["missed"]], [0.75, c["correct"]], [1, c["correct"]]]
    labels = [["Correctly cleared", "False alarm"], ["Missed cancer", "Correctly caught"]]
    notes = [["healthy patients the model correctly called healthy", "healthy patients wrongly flagged as diseased — worrying, but not dangerous"],
             ["diseased patients wrongly cleared as healthy — the dangerous mistake", "diseased patients the model correctly caught"]]
    text = [[f"<b style='font-size:21px'>{cm[i][j]}</b><br>{labels[i][j]}" for j in range(2)] for i in range(2)]
    fig = go.Figure(go.Heatmap(
        z=z, x=["Model said: Healthy", "Model said: Diseased"], y=["Actually Healthy", "Actually Diseased"],
        colorscale=scale, zmin=0, zmax=3, showscale=False, text=text, texttemplate="%{text}",
        textfont=dict(size=13, color=INK), xgap=4, ygap=4, customdata=np.dstack([cm, np.array(notes)]),
        hovertemplate="<b>%{customdata[0]} patients</b><br>%{customdata[1]}<extra></extra>",
    ))
    style(fig, 340, legend=False)
    fig.update_layout(title=f"3. Confusion matrix — {name}", margin=dict(l=110, b=60))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return fig


def fig_roc(results: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guess",
                             line=dict(color=AXIS, width=1.5, dash="dot"), hoverinfo="skip"))
    curves = {}
    for name, r in results.items():
        curves[name] = (r["fpr"], r["tpr"])
        fig.add_trace(go.Scatter(
            x=r["fpr"], y=r["tpr"], mode="lines", name=f"{name}  (AUC {r['auc']:.3f})",
            line=dict(color=MODEL_COLOR[name], width=2.5, shape="hv"),
            hovertemplate="At this setting the model catches <b>%{y:.0%}</b> of cancers<br>"
                          "while wrongly flagging <b>%{x:.0%}</b> of healthy patients<extra>" + name + "</extra>",
        ))
    n = max(len(c[0]) for c in curves.values())
    fig.frames = [
        go.Frame(data=[go.Scatter(x=fpr[:min(k, len(fpr))], y=tpr[:min(k, len(tpr))]) for fpr, tpr in curves.values()],
                 traces=[1, 2])
        for k in np.unique(np.linspace(1, n, 45).astype(int))
    ]
    style(fig, 460)
    fig.update_layout(
        title="4. ROC curves — press Draw to watch them form",
        xaxis=dict(title="False positive rate (healthy flagged as diseased)", tickformat=".0%", range=[-0.01, 1.01]),
        yaxis=dict(title="True positive rate (cancers caught)", tickformat=".0%", range=[0, 1.03]),
        updatemenus=[dict(type="buttons", showactive=False, x=1, y=1.14, xanchor="right", yanchor="bottom",
                          bgcolor=SURFACE, bordercolor=AXIS, font=dict(color=INK2, size=12), pad=dict(l=8, r=8, t=4, b=4),
                          buttons=[dict(label="Draw the curves", method="animate",
                                        args=[None, dict(frame=dict(duration=45, redraw=False), transition=dict(duration=0),
                                                         fromcurrent=False, mode="immediate")])])],
    )
    return fig


def fig_accuracy(results: dict, n_test: int, baseline: float) -> go.Figure:
    fig = go.Figure()
    for name, r in results.items():
        fig.add_trace(go.Bar(
            x=[r["acc"]], y=[name], orientation="h", name=name, marker=dict(color=MODEL_COLOR[name], cornerradius=6),
            text=[f"{pct(r['acc'])}  ·  {round(r['acc'] * n_test)} of {n_test} correct"], textposition="inside",
            insidetextanchor="end", textfont=dict(color="#fff", size=13),
            hovertemplate="<b>" + name + "</b><br>%{x:.2%} of the test patients classified correctly<extra></extra>",
        ))
    fig.add_vline(x=baseline, line=dict(color=MUTED, width=2, dash="dash"),
                  annotation_text=f"Always answering 'healthy' would score {baseline:.0%}", annotation_position="top left",
                  annotation_font=dict(color=INK2, size=12))
    style(fig, 300, legend=False)
    fig.update_layout(title="5. Model accuracy comparison", bargap=0.35,
                      xaxis=dict(range=[0, 1.0], tickformat=".0%", title="Accuracy"),
                      yaxis=dict(autorange="reversed"), margin=dict(l=150))
    return fig


def fig_boundary(p: dict) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=(LR, SVM), horizontal_spacing=0.08)
    stepped = [[0, CLASS_COLOR[0]], [0.5, CLASS_COLOR[0]], [0.5, CLASS_COLOR[1]], [1, CLASS_COLOR[1]]]
    for col, name in enumerate((LR, SVM), start=1):
        fig.add_trace(go.Contour(
            x=p["grid_x"][0], y=p["grid_y"][:, 0], z=p["grid_z"][name], zmin=0, zmax=1,
            colorscale=stepped, opacity=0.16, showscale=False, hoverinfo="skip",
            contours=dict(start=0.5, end=0.5, size=1, coloring="fill"), line=dict(color=INK, width=1.2),
        ), row=1, col=col)
        for cls in (0, 1):
            m = p["y_all"] == cls
            fig.add_trace(go.Scatter(
                x=p["X_pca"][m, 0], y=p["X_pca"][m, 1], mode="markers", name=CLASS_NAME[cls],
                showlegend=(col == 1), marker=dict(color=CLASS_COLOR[cls], size=7, line=dict(color=SURFACE, width=1)),
                hovertemplate="A real " + CLASS_SHORT[cls].lower() + " patient<br>PC1 %{x:.2f} · PC2 %{y:.2f}<extra>"
                              + name + "</extra>",
            ), row=1, col=col)
    style(fig, 470)
    fig.update_layout(title="6. Decision boundaries (PCA 2-D projection, illustrative only)")
    fig.update_annotations(font=dict(color=INK, size=14))
    fig.update_xaxes(title_text="PC1")
    fig.update_yaxes(title_text="PC2")
    return fig


def fig_metric_dumbbell(results: dict) -> go.Figure:
    metrics = [("Accuracy", lambda r: r["acc"]), ("Precision (Diseased)", lambda r: r["precision"][1]),
               ("Recall (Diseased)", lambda r: r["recall"][1]), ("F1-score (Diseased)", lambda r: r["f1"][1]),
               ("AUC", lambda r: r["auc"])]
    fig = go.Figure()
    for label, fn in metrics:
        a, b = fn(results[LR]), fn(results[SVM])
        fig.add_trace(go.Scatter(x=[a, b], y=[label, label], mode="lines", line=dict(color=AXIS, width=2.5),
                                 hoverinfo="skip", showlegend=False))
    for name in (LR, SVM):
        xs = [fn(results[name]) for _, fn in metrics]
        fig.add_trace(go.Scatter(
            x=xs, y=[m for m, _ in metrics], mode="markers+text", name=name,
            text=[f"{v:.1%}" if v <= 1 else f"{v:.3f}" for v in xs],
            textposition="top center" if name == LR else "bottom center", textfont=dict(color=INK2, size=11),
            marker=dict(color=MODEL_COLOR[name], size=13, line=dict(color=SURFACE, width=2)),
            hovertemplate="<b>%{y}</b>: %{x:.3f}<extra>" + name + "</extra>",
        ))
    style(fig, 380)
    fig.update_layout(title="Every score side by side — the dots sit almost on top of each other",
                      xaxis=dict(range=[0.9, 1.01], tickformat=".0%"), yaxis=dict(autorange="reversed"), margin=dict(l=160))
    return fig


# ------------------------------------------------------------------
# Text helpers
# ------------------------------------------------------------------
def feature_meaning(name: str) -> str:
    if name.startswith("mean "):
        kind, base = "mean", name[5:]
    elif name.startswith("worst "):
        kind, base = "worst", name[6:]
    else:
        kind, base = "error", name[: -len(" error")]
    return f"{MEASUREMENTS[base].capitalize()} — {SUFFIX[kind]}."


def report_table(r: dict) -> pd.DataFrame:
    return pd.DataFrame({
        "Class": [CLASS_NAME[0], CLASS_NAME[1]],
        "Precision": [pct(r["precision"][0]), pct(r["precision"][1])],
        "Recall": [pct(r["recall"][0]), pct(r["recall"][1])],
        "F1-score": [pct(r["f1"][0]), pct(r["f1"][1])],
        "Support (patients)": [int(r["support"][0]), int(r["support"][1])],
    })


def cm_sentence(cm: np.ndarray, name: str) -> str:
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    return (f"<b>{name}</b> saw <b>{cm.sum()}</b> test patients. It correctly cleared <b>{tn}</b> healthy people "
            f"and correctly caught <b>{tp}</b> cancers. It raised <b>{fp}</b> false alarm{'s' if fp != 1 else ''} "
            f"(healthy flagged as diseased) and missed <b>{fn}</b> cancer{'s' if fn != 1 else ''} — "
            f"the mistake that matters most in medicine.")


def winner(results: dict) -> str:
    a, b = results[LR]["acc"], results[SVM]["acc"]
    return LR if a > b else SVM if b > a else "tie"


# ------------------------------------------------------------------
# Pages — one per assignment part
# ------------------------------------------------------------------
def page_overview():
    p = pack()
    res, df = p["results"], p["df"]

    st.markdown("""<div class="hero"><span class="eyebrow">Machine learning · medical diagnosis</span>
<h1>Disease Classification Using Logistic Regression and SVM</h1>
<p><b>Objective:</b> apply machine learning to classify whether a patient is <b>Healthy (0)</b> or <b>Diseased (1)</b>
using real-world medical data. Dataset: <b>Breast Cancer Wisconsin (Diagnostic)</b> from the UCI Machine Learning
Repository — 569 tumour biopsies, each described by 30 measurements and a confirmed diagnosis.</p></div>""",
                unsafe_allow_html=True)
    st.write("")
    kpi_row([
        dict(label="Samples", value=len(df), color=MODEL_COLOR[LR], sub="patients in the dataset"),
        dict(label="Features", value=len(p["feature_names"]), color=MODEL_COLOR[LR], sub="measurements per patient"),
        dict(label="Training", value=p["n_train"], color=MODEL_COLOR[LR], sub="70% used to learn"),
        dict(label="Testing", value=p["n_test"], color=MODEL_COLOR[LR], sub="30% kept for testing"),
        dict(label="LR accuracy", value=round(res[LR]["acc"] * 100, 1), decimals=1, suffix="%",
             color=MODEL_COLOR[LR], sub="on the test patients"),
        dict(label="SVM accuracy", value=round(res[SVM]["acc"] * 100, 1), decimals=1, suffix="%",
             color=MODEL_COLOR[SVM], sub="on the test patients"),
    ])
    st.subheader("The six parts of the assignment")
    parts_stepper()
    explain("Think of it like a student and an exam. The two models <b>study</b> 70% of the patients (answers included), "
            "then sit an <b>exam</b> on the remaining 30% they have never seen. Every score in this dashboard is an exam score — "
            "that is what tells us how the models would behave on new patients. Use the menu on the left to walk through the six "
            "parts, and hover over any chart for a plain-language explanation.")


def page_part1():
    p = pack()
    df, names = p["df"], p["feature_names"]
    counts = df["diagnosis"].value_counts()

    part_header(1, "Dataset Exploration",
                "Breast Cancer Wisconsin (Diagnostic) — a public dataset from the UCI Machine Learning Repository. Each row is one "
                "patient: a sample of the tumour was photographed under a microscope and 30 numbers were measured from the cell "
                "nuclei. A pathologist's final diagnosis is the answer key.")

    kpi_row([
        dict(label="Number of samples", value=len(df), color=MODEL_COLOR[LR], sub="one row per patient"),
        dict(label="Number of features", value=len(names), color=MODEL_COLOR[LR], sub="+ 1 target column: diagnosis"),
        dict(label=CLASS_NAME[0], value=int(counts[0]), color=CLASS_COLOR[0], sub=f"{counts[0] / len(df):.0%} — benign tumour"),
        dict(label=CLASS_NAME[1], value=int(counts[1]), color=CLASS_COLOR[1], sub=f"{counts[1] / len(df):.0%} — malignant tumour"),
    ])

    st.subheader("Target variable and class labels")
    c1, c2 = st.columns([1.1, 1.4])
    with c1:
        st.dataframe(pd.DataFrame({
            "Label": ["0", "1"], "Class": [CLASS_LONG[0], CLASS_LONG[1]],
            "Patients": [int(counts[0]), int(counts[1])],
            "Share": [f"{counts[0] / len(df):.0%}", f"{counts[1] / len(df):.0%}"],
        }), hide_index=True, width="stretch")
    with c2:
        explain("The target column is <b>diagnosis</b>. About <b>63%</b> of the patients are healthy and <b>37%</b> are diseased. "
                "This matters: a lazy model that always answers 'healthy' would already be right 63% of the time, "
                "so any useful model has to score well above that.")

    st.subheader("Meaning of each feature")
    chart(fig_feature_treemap(names))
    explain("There are really only <b>10 things</b> being measured on the cell nuclei — size, texture, shape, dents, symmetry. "
            "Each one is recorded three ways: the <b>mean</b> across cells, the <b>error</b> (how much it varies between cells) "
            "and the <b>worst</b> (largest) value — giving 30 features. Hover any tile to read what it measures.")
    with st.expander("Full feature table and basic statistics"):
        st.dataframe(pd.DataFrame({"Feature": names, "Meaning": [feature_meaning(n) for n in names]}),
                     hide_index=True, width="stretch", height=420)
        st.dataframe(df[names].describe().T.round(4), width="stretch")
        st.caption("count = patients, mean = average, std = spread, min/max = smallest and largest value seen.")


def page_part2():
    p = pack()
    df = p["df"]

    part_header(2, "Data Preprocessing", "Three steps prepare the raw measurements for the models: check for gaps, put every "
                                         "feature on the same scale, and set aside a fair test set.")

    st.subheader("1. Handle missing values")
    c1, c2 = st.columns([1, 1.6])
    with c1:
        complete = 100 * (1 - df.isnull().sum().sum() / df.size)
        ring(complete, "Data completeness", f"{int(df.isnull().sum().sum())} missing values across {df.size:,} cells",
             MODEL_COLOR[LR])
    with c2:
        explain("Every patient has all 30 measurements filled in, so nothing needs to be imputed. "
                "(The code still guards against it: if any value were missing, it would be filled with the column mean.)")

    st.subheader("2. Standardise the input features")
    st.markdown("The measurements live on very different scales (an *area* can be 2,000 while a *smoothness* is 0.1). "
                "`StandardScaler` rescales every feature to **mean 0 and standard deviation 1**, fitted on the training data only.")
    chart(fig_scaling_dumbbell(p["scaled_stats"]))
    explain("Each row is one feature. The <b>orange dot</b> is how widely its raw values are spread — from 0.003 up to more than "
            "500 — and the <b>blue dot</b> is the spread after scaling: exactly 1 for every feature. Without this step the "
            "big-number features (area, perimeter) would drown out the small ones.")
    with st.expander("Before / after table"):
        st.dataframe(p["scaled_stats"], width="stretch")

    st.subheader("3. Split into training and testing data")
    c1, c2 = st.columns([1.3, 1])
    with c1:
        chart(fig_split_sunburst(p["y_train"], p["y_test"]))
    with c2:
        y_train, y_test = p["y_train"], p["y_test"]
        st.dataframe(pd.DataFrame({
            "Set": ["Training (70%)", "Testing (30%)"],
            "Patients": [len(y_train), len(y_test)],
            CLASS_SHORT[0]: [int((y_train == 0).sum()), int((y_test == 0).sum())],
            CLASS_SHORT[1]: [int((y_train == 1).sum()), int((y_test == 1).sum())],
            "Diseased share": [f"{(y_train == 1).mean():.0%}", f"{(y_test == 1).mean():.0%}"],
        }), hide_index=True, width="stretch")
        explain(f"<b>{len(y_train)}</b> patients teach the models and <b>{len(y_test)}</b> are locked away for the exam "
                f"(<code>train_test_split</code>, test_size=0.30, random_state=42). The split is <b>stratified</b>, so both "
                "rings keep the same healthy/diseased ratio — the exam is a fair sample of the whole dataset.")


def page_part3():
    p = pack()
    res, n_test = p["results"], p["n_test"]

    part_header(3, "Model Development", f"Two classifiers were trained on the {p['n_train']} standardised training patients "
                                        f"and then evaluated on the {n_test} unseen test patients.")

    c1, c2 = st.columns(2)
    with c1:
        card(LR, "Works like a <b>weighted checklist</b>: every measurement gets a weight, the weights are added up, and the total "
                 "is turned into a probability of disease. Geometrically it draws one <b>straight</b> dividing line between the two "
                 "groups.<br><code>LogisticRegression(max_iter=5000)</code>")
    with c2:
        card(f"{SVM} (Support Vector Machine)",
             "Finds the dividing line with the <b>widest safety margin</b> between the groups, and with the RBF kernel that line "
             "may be <b>curved</b>. It is wrapped in <code>CalibratedClassifierCV</code> so it can also output a probability.<br>"
             "<code>SVC(kernel=\"rbf\")</code>")

    st.subheader("Evaluation on the test data")
    c1, c2 = st.columns([1.15, 1])
    with c1:
        chart(fig_radar(res))
    with c2:
        for name in (LR, SVM):
            st.metric(f"{name} — accuracy", pct(res[name]["acc"]), help="Share of the test patients classified correctly.")
            st.dataframe(report_table(res[name]), hide_index=True, width="stretch")
    explain("<b>Accuracy</b> is the share of test patients the model got right. <b>Precision</b> answers 'when the model says "
            "diseased, how often is it right?', <b>recall</b> answers 'of the patients who really have cancer, how many did it "
            "catch?', and <b>F1-score</b> balances the two. The radar shows all six scores at once: the two shapes almost overlap, "
            "which is the whole story — both models are excellent and very close.")


def page_part4():
    p = pack()
    df, names = p["df"], p["feature_names"]

    part_header(4, "New Patient Prediction", "Enter the 30 medical measurements of a new patient and both models will predict "
                                             "whether the patient is Healthy (0) or Diseased (1). Start from an example or type "
                                             "your own values.")

    presets = ["Example A — healthy-like measurements", "Example B — diseased-like measurements", "Enter my own values"]
    choice = st.pills("Start from", presets, default=presets[0], key="preset") or presets[0]
    if choice == presets[0]:
        defaults, pk = EXAMPLE_HEALTHY, "A"
    elif choice == presets[1]:
        defaults, pk = EXAMPLE_DISEASED, "B"
    else:
        defaults, pk = df[names].mean().tolist(), "M"
        st.caption("Fields start at the dataset average — replace them with the new patient's measurements.")

    maxes = df[names].max()

    def fmt(feat):
        m = maxes[feat]
        return ("%.4f", 0.001) if m < 1 else ("%.2f", 0.1) if m < 100 else ("%.1f", 1.0)

    groups = [
        ("Mean values — 10 measurements", [n for n in names if n.startswith("mean ")], True),
        ("Error values — 10 measurements", [n for n in names if n.endswith(" error")], False),
        ("Worst values — 10 measurements", [n for n in names if n.startswith("worst ")], False),
    ]
    with st.form("patient_form", border=False):
        for label, feats, opened in groups:
            with st.expander(label, expanded=opened):
                cols = st.columns(5)
                for i, feat in enumerate(feats):
                    f, step = fmt(feat)
                    with cols[i % 5]:
                        st.number_input(feat, value=float(defaults[names.index(feat)]), step=step, format=f,
                                        key=f"in_{pk}_{names.index(feat)}", min_value=0.0)
        submitted = st.form_submit_button("Predict with both models", type="primary", width="stretch")

    if not submitted:
        explain("Press <b>Predict</b> to run the patient through Logistic Regression and SVM. "
                "Each measurement is explained on the Part 1 page.")
        return

    values = [st.session_state[f"in_{pk}_{i}"] for i in range(len(names))]
    render_prediction(p, values)


def render_prediction(p: dict, values: list):
    names = p["feature_names"]
    st.code("new_patient = [[" + ", ".join(f"{v:g}" for v in values) + "]]", language="python")
    with st.status("Predicting...", expanded=True) as s:
        st.write("Standardising the 30 measurements with the scaler fitted on the training data")
        time.sleep(0.45)
        x_scaled = p["scaler"].transform(pd.DataFrame([values], columns=names))
        st.write("Running Logistic Regression")
        time.sleep(0.4)
        st.write("Running SVM")
        time.sleep(0.4)
        probs = {name: float(m.predict_proba(x_scaled)[0, 1]) for name, m in p["models"].items()}
        s.update(label="Done", state="complete", expanded=False)

    gauges(probs)

    v_lr, v_svm = int(probs[LR] >= 0.5), int(probs[SVM] >= 0.5)
    if v_lr == v_svm:
        conf = (probs[LR] + probs[SVM]) / 2
        conf = conf if v_lr == 1 else 1 - conf
        st.markdown(f'<div class="verdict" style="--bg:{CLASS_DEEP[v_lr]}"><h3>Both models predict: {CLASS_LONG[v_lr]}</h3>'
                    f'<p>Average confidence {conf:.0%}. Two very different methods reaching the same answer makes the result '
                    f'more trustworthy.</p></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="verdict" style="--bg:{INK2}"><h3>The models disagree</h3>'
                    f'<p>{LR} predicts <b>{CLASS_NAME[v_lr]}</b> ({probs[LR]:.0%} diseased); {SVM} predicts '
                    f'<b>{CLASS_NAME[v_svm]}</b> ({probs[SVM]:.0%} diseased). '
                    f'This patient sits close to the dividing line — a case for a specialist.</p></div>',
                    unsafe_allow_html=True)
    explain("The percentage is each model's probability that the tumour is malignant. At <b>50% or more</b> the model outputs "
            "<b>Diseased (1)</b>; below that, <b>Healthy (0)</b>. This is an educational project, not a medical device — "
            "a real diagnosis always comes from a doctor.")


def page_part5():
    p = pack()
    df, res, n_test = p["df"], p["results"], p["n_test"]
    counts = df["diagnosis"].value_counts()

    part_header(5, "Visualization", "Six plots — two describe the data, four show the classification results. "
                                    "Hover over anything for a plain-language explanation.")

    c1, c2 = st.columns([1, 1.25])
    with c1:
        chart(fig_donut(counts, res))
    with c2:
        st.write("")
        explain("The ring shows how the 569 patients split between the two classes: <b>63%</b> healthy and <b>37%</b> diseased. "
                "Hover a slice to see the exact count and which model was best at recognising that group.")
        explain(f"<b>Healthy patients:</b> best recognised by {best_model_for_class(res, 0)}.<br>"
                f"<b>Diseased patients:</b> best recognised by {best_model_for_class(res, 1)}.", tag="Who spots whom")

    chart(fig_corr_heatmap(df))
    explain("Blue = the two measurements rise together, red = one rises while the other falls, grey = unrelated. "
            "The last row shows how strongly each mean measurement is linked to the diagnosis: bigger, more dented nuclei "
            "(<i>concave points</i>, <i>perimeter</i>, <i>radius</i>, <i>area</i>) go with malignancy.")

    c1, c2 = st.columns(2)
    with c1:
        chart(fig_confusion(res[LR]["cm"], LR))
    with c2:
        chart(fig_confusion(res[SVM]["cm"], SVM))
    explain("Each tile counts test patients. The two <b>green</b> tiles are correct answers. The <b>amber</b> tile is a false "
            "alarm (healthy flagged as diseased); the <b>red</b> tile is the dangerous one — a real cancer cleared as healthy.",
            tag="How to read this")
    st.markdown(f'<div class="card"><p>{cm_sentence(res[LR]["cm"], LR)}</p><br><p>{cm_sentence(res[SVM]["cm"], SVM)}</p></div>',
                unsafe_allow_html=True)

    chart(fig_roc(res))
    explain("Every model has a dial: turn it towards 'suspicious' and it catches more cancers but raises more false alarms. "
            "The ROC curve shows every setting of that dial at once; the closer it hugs the <b>top-left corner</b>, the better. "
            "The dotted diagonal is random guessing. <b>AUC</b> (area under the curve) summarises it: 1.000 would be perfect.")

    chart(fig_accuracy(res, n_test, counts[0] / len(df)))
    explain(f"Accuracy of each model on the same {n_test} test patients. The difference is "
            f"{abs(res[LR]['acc'] - res[SVM]['acc']):.1%} — about "
            f"{round(abs(res[LR]['acc'] - res[SVM]['acc']) * n_test)} patient(s). "
            "The dashed line is the score of a model that never learns anything and just says 'healthy' every time.")

    chart(fig_boundary(p))
    explain("To draw this, the 30 measurements were squeezed into 2 summary numbers per patient (PCA) and each model was refitted "
            "on those 2 numbers just for the picture. Logistic Regression draws a <b>straight</b> dividing line; SVM draws a "
            "<b>curved</b> one. The real models use all 30 features — this view is illustrative only.")


def page_part6():
    p = pack()
    res, n_test = p["results"], p["n_test"]
    w = winner(res)
    diff = abs(res[LR]["acc"] - res[SVM]["acc"])
    n_diff = round(diff * n_test)
    lr, sv = res[LR], res[SVM]

    part_header(6, "Comparison and Discussion", "The three questions of the assignment, answered with the numbers from the test set.")

    st.subheader("Q1. Which model achieved higher accuracy?")
    if w == "tie":
        head = "It is a tie"
        body = f"Both models classified exactly the same number of the {n_test} test patients correctly."
        bg = INK2
    else:
        other = SVM if w == LR else LR
        head = f"{w}: {pct(res[w]['acc'])} vs {pct(res[other]['acc'])} for {other}"
        body = (f"{w} answered {n_diff} more patient{'s' if n_diff != 1 else ''} correctly out of {n_test} — a difference of "
                f"{diff:.1%}. On a test this small, a single patient can flip the ranking, so in raw accuracy the two models are "
                "essentially tied.")
        bg = MODEL_COLOR[w]
    st.markdown(f'<div class="verdict" style="--bg:{bg}"><h3>{head}</h3><p>{body}</p></div>', unsafe_allow_html=True)
    chart(fig_metric_dumbbell(res))

    st.subheader("Q2. Why might one model perform better than the other for this dataset?")
    st.markdown(f"""
- **The two classes are almost linearly separable** once the features are standardised (see the decision-boundary plot in Part 5).
  Measurements such as *mean concave points* and *worst radius* rise steadily with malignancy, so one straight dividing line —
  exactly what Logistic Regression draws — already separates the groups very well. That is why a simple linear model reaches {pct(lr["acc"])}.
- **SVM with an RBF kernel can bend its boundary.** Where the two groups overlap, a curved line can wrap around a few borderline
  patients that a straight line cannot, and the maximum-margin rule makes it slightly more robust to outliers. Here that bought
  {n_diff} extra correct patient{'s' if n_diff != 1 else ''}: SVM raised {int(sv["cm"][0, 1])} false alarm{'s' if int(sv["cm"][0, 1]) != 1 else ''}
  versus {int(lr["cm"][0, 1])} for Logistic Regression, while both missed the same {int(lr["cm"][1, 0])} cancers (recall {pct(lr["recall"][1])} for both).
- **Standardisation matters for both.** Without it, features with large numbers (area, perimeter) would dominate the distance
  calculations SVM relies on and slow down Logistic Regression's optimisation. After scaling, both models see every measurement on an equal footing.
- **The remaining difference is within noise.** With {n_test} test patients, one misclassification moves accuracy by about {1 / n_test:.1%};
  the gap between the models is of that order, so the honest conclusion is that both fit this dataset very well.
""")

    st.subheader("Q3. Would you recommend Logistic Regression or SVM for this medical problem?")
    c1, c2 = st.columns(2)
    with c1:
        card(f"Recommended: {LR}",
             "<b>1. Interpretable.</b> Its weights show a doctor <i>which</i> measurements drove a prediction, and its output is a "
             "genuine probability. <b>2. Auditable.</b> A simple model is easier to validate, explain to patients and defend to "
             "regulators — essential for medical and legal accountability. <b>3. Cheap.</b> It trains and predicts in milliseconds "
             "and needs no probability calibration. It gives up only a fraction of a percent of accuracy for all of that.")
    with c2:
        card(f"When to prefer {SVM}",
             "If the goal is to squeeze out the last bit of accuracy and nobody needs to read the model's reasoning — for example an "
             "internal automated triage step — SVM's slightly higher score and perfect precision on this split make it a strong "
             "choice. In this project it works best as a <b>second opinion</b> that flags cases where the two models disagree.")
    explain(f"<b>Verdict:</b> for a clinical decision-support tool, <b>Logistic Regression</b> is the recommended model. "
            f"It matches SVM on the number of cancers caught ({pct(lr['recall'][1])}), trails it by only "
            f"{n_diff} patient{'s' if n_diff != 1 else ''} in accuracy, and — unlike SVM — can explain every decision it makes. "
            f"SVM remains a valuable secondary check.", tag="Recommendation")


# ------------------------------------------------------------------
# App
# ------------------------------------------------------------------
def boot() -> dict:
    if st.session_state.get("booted"):
        return train()
    with st.status("Preparing the models — first run only", expanded=True) as s:
        st.write("Reading the patient records")
        time.sleep(0.4)
        st.write("Splitting them 70% / 30% and standardising the features")
        time.sleep(0.4)
        st.write("Training Logistic Regression and SVM")
        result = train()
        st.write("Evaluating both models on the test patients")
        time.sleep(0.4)
        s.update(label="Models are ready", state="complete", expanded=False)
    st.session_state["booted"] = True
    return result


def main():
    st.set_page_config(page_title="Disease Classification — LR vs SVM", page_icon=":material/monitor_heart:",
                       layout="wide", initial_sidebar_state="expanded")
    inject_css()

    nav = st.navigation([
        st.Page(page_overview, title="Overview", icon=":material/home:", default=True),
        st.Page(page_part1, title="Part 1: Dataset Exploration", icon=":material/dataset:"),
        st.Page(page_part2, title="Part 2: Data Preprocessing", icon=":material/tune:"),
        st.Page(page_part3, title="Part 3: Model Development", icon=":material/model_training:"),
        st.Page(page_part4, title="Part 4: New Patient Prediction", icon=":material/stethoscope:"),
        st.Page(page_part5, title="Part 5: Visualization", icon=":material/analytics:"),
        st.Page(page_part6, title="Part 6: Comparison & Discussion", icon=":material/gavel:"),
    ])
    with st.sidebar:
        st.caption("Breast Cancer Wisconsin (Diagnostic) · Logistic Regression vs SVM · educational use only")

    st.session_state["pack"] = boot()
    nav.run()


if __name__ == "__main__":
    main()
