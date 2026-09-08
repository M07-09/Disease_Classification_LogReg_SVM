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
MODEL_COLOR = {LR: "#2a78d6", SVM: "#eb6834"}
CLASS_COLOR = {0: "#199e70", 1: "#e34948"}
CLASS_NAME = {0: "Healthy (0)", 1: "Diseased (1)"}
CLASS_LONG = {0: "Healthy (0) — benign tumour", 1: "Diseased (1) — malignant tumour"}
CLASS_SHORT = {0: "Healthy", 1: "Diseased"}

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"
BLUES = [[0, "#cde2fb"], [0.5, "#3987e5"], [1, "#0d366b"]]
DIVERGING = [[0, "#2a78d6"], [0.5, "#f0efec"], [1, "#e34948"]]

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
                '[data-testid="stDataFrame"]', '.card', '.explain', '.hero', '.verdict']
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
[data-testid="stMetric"] {{ background: {SURFACE}; border: 1px solid rgba(11,11,11,.08); border-radius: 14px; padding: 14px 18px; }}
.hero h1 {{ font-size: 32px; margin: 0 0 8px 0; letter-spacing: -.01em; color: {INK}; }}
.hero p {{ font-size: 16.5px; color: {INK2}; margin: 0; max-width: 940px; line-height: 1.55; }}
.card {{ background: {SURFACE}; border: 1px solid rgba(11,11,11,.08); border-radius: 14px; padding: 18px 20px; margin-bottom: 12px; height: 100%; }}
.card h4 {{ margin: 0 0 6px 0; font-size: 15.5px; color: {INK}; }}
.card p {{ margin: 0; color: {INK2}; font-size: 14px; line-height: 1.55; }}
.explain {{ background: #eef4fc; border-left: 4px solid {MODEL_COLOR[LR]}; border-radius: 10px; padding: 12px 16px;
           margin: 6px 0 16px 0; color: #1f2a37; font-size: 14.5px; line-height: 1.6; }}
.explain .tag {{ display: block; font-size: 11.5px; font-weight: 700; letter-spacing: .06em; text-transform: uppercase;
                color: {MODEL_COLOR[LR]}; margin-bottom: 4px; }}
.verdict {{ border-radius: 16px; padding: 22px 24px; color: #fff; margin: 8px 0 14px 0; }}
.verdict h3 {{ margin: 0 0 4px 0; font-size: 24px; color: #fff; }}
.verdict p {{ margin: 0; opacity: .95; font-size: 15px; line-height: 1.5; }}
h2, h3 {{ letter-spacing: -.01em; }}
footer {{ visibility: hidden; }}
@media (prefers-reduced-motion: reduce) {{ *, *::before, *::after {{ animation: none !important; transition: none !important; }} }}
</style>""", unsafe_allow_html=True)


def explain(text: str, tag: str = "In plain words"):
    st.markdown(f'<div class="explain"><span class="tag">{tag}</span>{text}</div>', unsafe_allow_html=True)


def card(title: str, body: str):
    st.markdown(f'<div class="card"><h4>{title}</h4><p>{body}</p></div>', unsafe_allow_html=True)


def style(fig: go.Figure, height: int = 380, legend: bool = True) -> go.Figure:
    fig.update_layout(
        template="plotly_white", height=height, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family=FONT, color=INK2, size=13), title_font=dict(size=16, color=INK),
        margin=dict(l=50, r=24, t=60, b=50), showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=GRID, font=dict(family=FONT, size=13, color=INK)),
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
# Animated HTML components (count-up tiles, stepper, gauges)
# ------------------------------------------------------------------
BASE_HTML_CSS = f"""
html, body {{ margin: 0; background: transparent; font-family: {FONT}; color: {INK}; }}
@keyframes up {{ from {{ opacity: 0; transform: translateY(12px); }} to {{ opacity: 1; transform: none; }} }}
"""


def kpi_row(items, height: int = 150):
    cards = "".join(
        f'<div class="k" style="--c:{it["color"]}; animation-delay:{i * 90}ms">'
        f'<div class="lbl">{it["label"]}</div>'
        f'<div class="val"><span class="num" data-v="{it["value"]}" data-d="{it.get("decimals", 0)}">0</span>{it.get("suffix", "")}</div>'
        f'<div class="sub">{it.get("sub", "")}</div></div>'
        for i, it in enumerate(items)
    )
    html = f"""<style>{BASE_HTML_CSS}
.row {{ display: grid; grid-template-columns: repeat({len(items)}, minmax(0, 1fr)); gap: 14px; }}
.k {{ background: {SURFACE}; border: 1px solid rgba(11,11,11,.08); border-radius: 14px; padding: 16px 18px;
      position: relative; overflow: hidden; animation: up .7s cubic-bezier(.2,.7,.2,1) both; }}
.k::before {{ content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px; background: var(--c); }}
.lbl {{ font-size: 12.5px; color: {INK2}; text-transform: uppercase; letter-spacing: .05em; font-weight: 600; }}
.val {{ font-size: 30px; font-weight: 700; margin-top: 4px; color: {INK}; line-height: 1.15; }}
.sub {{ font-size: 12.5px; color: {MUTED}; margin-top: 4px; }}
</style>
<div class="row">{cards}</div>
<script>
document.querySelectorAll('.num').forEach(el => {{
  const target = +el.dataset.v, d = +el.dataset.d, t0 = performance.now(), dur = 1400;
  const ease = x => 1 - Math.pow(1 - x, 3);
  const tick = now => {{
    const p = Math.min(1, (now - t0) / dur);
    el.textContent = (target * ease(p)).toLocaleString('en-US', {{minimumFractionDigits: d, maximumFractionDigits: d}});
    if (p < 1) requestAnimationFrame(tick);
  }};
  requestAnimationFrame(tick);
}});
</script>"""
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
@keyframes light {{ to {{ background: {MODEL_COLOR[LR]}; border-color: {MODEL_COLOR[LR]}; color: #fff; transform: scale(1.08); }} }}
.t {{ margin-top: 10px; font-weight: 700; font-size: 14px; opacity: 0; animation: up .5s var(--d) forwards; }}
.s {{ margin-top: 3px; font-size: 12.5px; color: {INK2}; line-height: 1.4; opacity: 0; animation: up .5s calc(var(--d) + .1s) forwards; }}
</style>
<div class="wrap"><div class="track"><div class="fill"></div></div><div class="steps">{items}</div></div>"""
    st.iframe(html, height=130)


def gauges(probs: dict):
    arc = 251.3  # length of the semicircle path below (radius 80)
    cards = ""
    for i, (name, p) in enumerate(probs.items()):
        cls = int(p >= 0.5)
        color = CLASS_COLOR[cls]
        cards += f"""
<div class="g" style="animation-delay:{i * 120}ms">
  <div><span class="chip" style="background:{MODEL_COLOR[name]}">{name}</span></div>
  <svg viewBox="0 0 200 118" width="100%">
    <path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="{GRID}" stroke-width="14" stroke-linecap="round"/>
    <path class="arc" d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="{color}" stroke-width="14"
          stroke-linecap="round" stroke-dasharray="{arc}" stroke-dashoffset="{arc}" data-p="{p}"/>
    <line x1="100" y1="12" x2="100" y2="28" stroke="{MUTED}" stroke-width="2"/>
    <text x="100" y="92" text-anchor="middle" font-size="30" font-weight="700" fill="{INK}"><tspan class="num" data-p="{p}">0</tspan>%</text>
    <text x="100" y="112" text-anchor="middle" font-size="11" fill="{MUTED}">probability of being diseased</text>
  </svg>
  <div class="verdict" style="color:{color}">{CLASS_NAME[cls]}</div>
</div>"""
    html = f"""<style>{BASE_HTML_CSS}
.grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
.g {{ background: {SURFACE}; border: 1px solid rgba(11,11,11,.08); border-radius: 14px; padding: 14px 18px 12px; text-align: center;
      animation: up .6s cubic-bezier(.2,.7,.2,1) both; }}
.chip {{ display: inline-block; padding: 3px 12px; border-radius: 999px; font-size: 12.5px; font-weight: 600; color: #fff; }}
.arc {{ transition: stroke-dashoffset 1.5s cubic-bezier(.2,.7,.2,1); }}
.verdict {{ font-weight: 700; font-size: 18px; margin-top: 2px; }}
</style>
<div class="grid">{cards}</div>
<script>
requestAnimationFrame(() => requestAnimationFrame(() => {{
  document.querySelectorAll('.arc').forEach(el => {{ el.style.strokeDashoffset = {arc} * (1 - +el.dataset.p); }});
}}));
document.querySelectorAll('.num').forEach(el => {{
  const target = 100 * +el.dataset.p, t0 = performance.now(), dur = 1500;
  const ease = x => 1 - Math.pow(1 - x, 3);
  const tick = now => {{
    const p = Math.min(1, (now - t0) / dur);
    el.textContent = (target * ease(p)).toFixed(1);
    if (p < 1) requestAnimationFrame(tick);
  }};
  requestAnimationFrame(tick);
}});
</script>"""
    st.iframe(html, height=300)


# ------------------------------------------------------------------
# Charts (the six plots of Part 5)
# ------------------------------------------------------------------
def fig_class_distribution(counts: pd.Series) -> go.Figure:
    n = int(counts.sum())
    fig = go.Figure()
    for cls in (0, 1):
        c = int(counts[cls])
        fig.add_trace(go.Bar(
            y=["All patients"], x=[c], orientation="h", name=CLASS_NAME[cls],
            marker=dict(color=CLASS_COLOR[cls], line=dict(color=SURFACE, width=2)),
            text=f"{CLASS_NAME[cls]}: {c} patients ({c / n:.0%})", textposition="inside",
            insidetextanchor="middle", textfont=dict(color="#fff", size=14),
            hovertemplate=f"{CLASS_NAME[cls]}<br><b>%{{x}} patients</b><extra></extra>",
        ))
    style(fig, 190, legend=False)
    fig.update_layout(barmode="stack", title="1. Class distribution", bargap=0.3,
                      xaxis=dict(visible=False), yaxis=dict(visible=False), margin=dict(l=10, r=10, t=70, b=10))
    return fig


def fig_corr_heatmap(df: pd.DataFrame) -> go.Figure:
    cols = [c for c in df.columns if c.startswith("mean")] + ["diagnosis"]
    corr = df[cols].corr()
    labels = [c.replace("mean ", "") for c in cols]
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=labels, y=labels, colorscale=DIVERGING, zmin=-1, zmax=1, zmid=0,
        text=np.round(corr.values, 2), texttemplate="%{text}", textfont=dict(size=10),
        xgap=2, ygap=2, colorbar=dict(title="corr.", thickness=12, outlinewidth=0),
        hovertemplate="%{y} vs %{x}<br>correlation: <b>%{z:.2f}</b><extra></extra>",
    ))
    style(fig, 560, legend=False)
    fig.update_layout(title="2. Correlation heatmap (mean-value features)", margin=dict(l=120, b=120))
    fig.update_xaxes(tickangle=-40, showgrid=False)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return fig


def fig_confusion(cm: np.ndarray, name: str) -> go.Figure:
    labels = [["Correctly cleared", "False alarm"], ["Missed cancer", "Correctly caught"]]
    text = [[f"<b>{cm[i][j]}</b><br>{labels[i][j]}" for j in range(2)] for i in range(2)]
    fig = go.Figure(go.Heatmap(
        z=cm, x=["Predicted: Healthy", "Predicted: Diseased"], y=["Actual: Healthy", "Actual: Diseased"],
        colorscale=BLUES, zmin=0, showscale=False, text=text, texttemplate="%{text}", textfont=dict(size=14),
        xgap=4, ygap=4, hovertemplate="%{y}<br>%{x}<br><b>%{z} patients</b><extra></extra>",
    ))
    style(fig, 340, legend=False)
    fig.update_layout(title=f"3. Confusion matrix — {name}", margin=dict(l=110, b=60))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return fig


def fig_roc(results: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guess",
                             line=dict(color=AXIS, width=1.5), hoverinfo="skip"))
    for name, r in results.items():
        fig.add_trace(go.Scatter(
            x=r["fpr"], y=r["tpr"], mode="lines", name=f"{name}  (AUC {r['auc']:.3f})",
            line=dict(color=MODEL_COLOR[name], width=2.5, shape="hv"),
            hovertemplate="false-positive rate: %{x:.0%}<br>true-positive rate: %{y:.0%}<extra>" + name + "</extra>",
        ))
    style(fig, 440)
    fig.update_layout(title="4. ROC curves",
                      xaxis=dict(title="False positive rate (healthy flagged as diseased)", tickformat=".0%", range=[-0.01, 1.01]),
                      yaxis=dict(title="True positive rate (cancers caught)", tickformat=".0%", range=[0, 1.03]))
    return fig


def fig_accuracy(results: dict, n_test: int) -> go.Figure:
    fig = go.Figure()
    for name, r in results.items():
        fig.add_trace(go.Bar(
            name=name, x=[name], y=[r["acc"]], marker=dict(color=MODEL_COLOR[name], cornerradius=4),
            text=[pct(r["acc"])], textposition="outside", textfont=dict(color=INK2, size=13),
            hovertemplate="accuracy: <b>%{y:.2%}</b><extra>" + name + "</extra>",
        ))
    style(fig, 380, legend=False)
    fig.update_layout(title=f"5. Model accuracy comparison ({n_test} test patients)", bargap=0.45,
                      yaxis=dict(range=[0, 1.12], tickformat=".0%", title="Accuracy"))
    return fig


def fig_boundary(p: dict) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=(LR, SVM), horizontal_spacing=0.08)
    stepped = [[0, CLASS_COLOR[0]], [0.5, CLASS_COLOR[0]], [0.5, CLASS_COLOR[1]], [1, CLASS_COLOR[1]]]
    for col, name in enumerate((LR, SVM), start=1):
        fig.add_trace(go.Contour(
            x=p["grid_x"][0], y=p["grid_y"][:, 0], z=p["grid_z"][name], zmin=0, zmax=1,
            colorscale=stepped, opacity=0.2, showscale=False, hoverinfo="skip",
            contours=dict(start=0.5, end=0.5, size=1, coloring="fill"), line=dict(color=INK, width=1.2),
        ), row=1, col=col)
        for cls in (0, 1):
            m = p["y_all"] == cls
            fig.add_trace(go.Scatter(
                x=p["X_pca"][m, 0], y=p["X_pca"][m, 1], mode="markers", name=CLASS_NAME[cls],
                showlegend=(col == 1), marker=dict(color=CLASS_COLOR[cls], size=7, line=dict(color=SURFACE, width=1)),
                hovertemplate="PC1 %{x:.2f}<br>PC2 %{y:.2f}<extra>" + CLASS_SHORT[cls] + "</extra>",
            ), row=1, col=col)
    style(fig, 470)
    fig.update_layout(title="6. Decision boundaries (PCA 2-D projection, illustrative only)")
    fig.update_xaxes(title_text="PC1")
    fig.update_yaxes(title_text="PC2")
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

    st.markdown("""<div class="hero"><h1>Disease Classification Using Logistic Regression and SVM</h1>
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
        dict(label="LR accuracy", value=round(res[LR]["acc"] * 100, 1), decimals=1, suffix="%", color=MODEL_COLOR[LR], sub="on the test patients"),
        dict(label="SVM accuracy", value=round(res[SVM]["acc"] * 100, 1), decimals=1, suffix="%", color=MODEL_COLOR[SVM], sub="on the test patients"),
    ])
    st.subheader("The six parts of the assignment")
    parts_stepper()
    explain("Think of it like a student and an exam. The two models <b>study</b> 70% of the patients (answers included), "
            "then sit an <b>exam</b> on the remaining 30% they have never seen. Every score in this dashboard is an exam score — "
            "that is what tells us how the models would behave on new patients. Use the menu on the left to walk through the six parts.")


def page_part1():
    p = pack()
    df, names = p["df"], p["feature_names"]
    counts = df["diagnosis"].value_counts()

    st.header("Part 1: Dataset Exploration")
    st.markdown("**Dataset:** Breast Cancer Wisconsin (Diagnostic) — a public dataset from the UCI Machine Learning Repository "
                "(`breast_cancer_dataset.csv.xlsx`). Each row is one patient: a small sample of the tumour was photographed under a "
                "microscope and 30 numbers were measured from the cell nuclei in that image. A pathologist's final diagnosis is the answer key.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Number of samples", f"{len(df)}", help="One row per patient.")
    c2.metric("Number of features", f"{len(names)}", help="Plus 1 target column, 'diagnosis'.")
    c3.metric(CLASS_NAME[0], f"{int(counts[0])}", help=f"{counts[0] / len(df):.0%} of patients — benign tumour")
    c4.metric(CLASS_NAME[1], f"{int(counts[1])}", help=f"{counts[1] / len(df):.0%} of patients — malignant tumour")

    st.subheader("Target variable and class labels")
    st.markdown("The target column is **`diagnosis`**, with two class labels:")
    st.dataframe(pd.DataFrame({
        "Label": [0, 1], "Class": [CLASS_LONG[0], CLASS_LONG[1]],
        "Patients": [int(counts[0]), int(counts[1])],
        "Share": [f"{counts[0] / len(df):.0%}", f"{counts[1] / len(df):.0%}"],
    }), hide_index=True, width="content")
    explain("About 63% of the patients are healthy and 37% are diseased. This matters: a lazy model that always answers "
            "'healthy' would already be right 63% of the time, so any useful model has to score well above that.")

    st.subheader("Meaning of each feature")
    st.markdown("There are really only **10 things** being measured on the cell nuclei. Each one is recorded three ways — "
                "the **mean** across cells, the **error** (how much it varies between cells) and the **worst** (largest) value — "
                "giving 30 features in total.")
    st.dataframe(pd.DataFrame({"Feature": names, "Meaning": [feature_meaning(n) for n in names]}),
                 hide_index=True, width="stretch", height=420)

    with st.expander("Basic statistics of the features"):
        st.dataframe(df[names].describe().T.round(4), width="stretch")
        st.caption("count = patients, mean = average, std = spread, min/max = smallest and largest value seen.")


def page_part2():
    p = pack()
    df, names = p["df"], p["feature_names"]

    st.header("Part 2: Data Preprocessing")

    st.subheader("1. Handle missing values")
    missing = df.isnull().sum()
    c1, c2 = st.columns([1, 2])
    c1.metric("Missing values found", f"{int(missing.sum())}")
    with c2:
        explain("Every patient has all 30 measurements filled in, so nothing needs to be imputed. "
                "(The code still guards against it: if any value were missing, it would be filled with the column mean.)")

    st.subheader("2. Standardise the input features")
    st.markdown("The measurements live on very different scales (an *area* can be 2,000 while a *smoothness* is 0.1). "
                "`StandardScaler` rescales every feature to **mean 0 and standard deviation 1**, so no single measurement "
                "dominates just because its numbers are bigger. The scaler is fitted on the training data only.")
    st.dataframe(p["scaled_stats"].head(6), width="stretch")
    st.caption("First six features, before and after standardisation (training set). After scaling every mean is ~0 and every std is ~1.")

    st.subheader("3. Split into training and testing data")
    y_train, y_test = p["y_train"], p["y_test"]
    split = pd.DataFrame({
        "Set": ["Training (70%)", "Testing (30%)"],
        "Patients": [len(y_train), len(y_test)],
        CLASS_NAME[0]: [int((y_train == 0).sum()), int((y_test == 0).sum())],
        CLASS_NAME[1]: [int((y_train == 1).sum()), int((y_test == 1).sum())],
        "Diseased share": [f"{(y_train == 1).mean():.0%}", f"{(y_test == 1).mean():.0%}"],
    })
    st.dataframe(split, hide_index=True, width="stretch")
    explain(f"<b>{len(y_train)}</b> patients are used to teach the models and <b>{len(y_test)}</b> are locked away for the exam "
            f"(`train_test_split`, test_size=0.30, random_state=42). The split is <b>stratified</b>, so both sets keep the same "
            "healthy/diseased ratio — the exam is a fair sample of the whole dataset.")


def page_part3():
    p = pack()
    res, n_test = p["results"], p["n_test"]

    st.header("Part 3: Model Development")
    st.markdown(f"Two classifiers were trained on the {p['n_train']} standardised training patients and then evaluated "
                f"on the {n_test} unseen test patients.")

    c1, c2 = st.columns(2)
    with c1:
        card(f"{LR}",
             "Works like a <b>weighted checklist</b>: every measurement gets a weight, the weights are added up, and the total is "
             "turned into a probability of disease. Geometrically it draws one <b>straight</b> dividing line between the two groups.<br>"
             "<code>LogisticRegression(max_iter=5000)</code>")
    with c2:
        card(f"{SVM} (Support Vector Machine)",
             "Finds the dividing line with the <b>widest safety margin</b> between the groups, and with the RBF kernel that line "
             "may be <b>curved</b>. It is wrapped in <code>CalibratedClassifierCV</code> so it can also output a probability.<br>"
             "<code>SVC(kernel=\"rbf\")</code>")

    st.subheader("Evaluation on the test data")
    c1, c2 = st.columns(2)
    for col, name in zip((c1, c2), (LR, SVM)):
        with col:
            st.metric(f"{name} — accuracy", pct(res[name]["acc"]), help="Share of the test patients classified correctly.")
            st.dataframe(report_table(res[name]), hide_index=True, width="stretch")
    explain("<b>Accuracy</b> is the share of test patients the model got right. In the tables, <b>precision</b> answers "
            "'when the model says diseased, how often is it right?', <b>recall</b> answers 'of the patients who really have cancer, "
            "how many did it catch?', and <b>F1-score</b> balances the two. <b>Support</b> is how many test patients are in each class.")


def page_part4():
    p = pack()
    df, names = p["df"], p["feature_names"]

    st.header("Part 4: New Patient Prediction")
    st.markdown("Enter the 30 medical measurements of a new patient and both models will predict whether the patient is "
                "**Healthy (0)** or **Diseased (1)**. You can start from one of the two example patients or type your own values.")

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
        st.markdown(f'<div class="verdict" style="background:{CLASS_COLOR[v_lr]}"><h3>Both models predict: {CLASS_LONG[v_lr]}</h3>'
                    f'<p>Average confidence {conf:.0%}. Two very different methods reaching the same answer makes the result more trustworthy.</p></div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="verdict" style="background:{INK2}"><h3>The models disagree</h3>'
                    f'<p>{LR} predicts <b>{CLASS_NAME[v_lr]}</b> ({probs[LR]:.0%} diseased); {SVM} predicts <b>{CLASS_NAME[v_svm]}</b> '
                    f'({probs[SVM]:.0%} diseased). This patient sits close to the dividing line — a case for a specialist.</p></div>',
                    unsafe_allow_html=True)
    explain("The percentage is each model's probability that the tumour is malignant. At <b>50% or more</b> the model outputs "
            "<b>Diseased (1)</b>; below that, <b>Healthy (0)</b>. This is an educational project, not a medical device — "
            "a real diagnosis always comes from a doctor.")


def page_part5():
    p = pack()
    df, res, n_test = p["df"], p["results"], p["n_test"]

    st.header("Part 5: Visualization")
    st.markdown("Six plots: two describe the data, four show the classification results of both models.")

    chart(fig_class_distribution(df["diagnosis"].value_counts()))
    explain("How many patients fall in each class. The dataset is moderately imbalanced (63% healthy / 37% diseased).")

    chart(fig_corr_heatmap(df))
    explain("Blue = the two measurements rise together, red = one rises while the other falls, grey = unrelated. "
            "The last row shows how strongly each mean measurement is linked to the diagnosis: bigger, more dented nuclei "
            "(<i>concave points</i>, <i>perimeter</i>, <i>radius</i>, <i>area</i>) go with malignancy.")

    c1, c2 = st.columns(2)
    with c1:
        chart(fig_confusion(res[LR]["cm"], LR))
    with c2:
        chart(fig_confusion(res[SVM]["cm"], SVM))
    explain("Each square counts test patients. The two <b>diagonal</b> squares are correct answers. <b>Top-right</b> is a false "
            "alarm (healthy flagged as diseased); <b>bottom-left</b> is the dangerous one — a real cancer cleared as healthy.",
            tag="How to read this")
    st.markdown(f'<div class="card"><p>{cm_sentence(res[LR]["cm"], LR)}</p><br><p>{cm_sentence(res[SVM]["cm"], SVM)}</p></div>',
                unsafe_allow_html=True)

    chart(fig_roc(res))
    explain("Every model has a dial: turn it towards 'suspicious' and it catches more cancers but raises more false alarms. "
            "The ROC curve shows every setting of that dial at once; the closer it hugs the <b>top-left corner</b>, the better. "
            "The grey diagonal is random guessing. <b>AUC</b> (area under the curve) summarises it: 1.000 would be perfect.")

    chart(fig_accuracy(res, n_test))
    explain(f"Accuracy of each model on the same {n_test} test patients. The difference is "
            f"{abs(res[LR]['acc'] - res[SVM]['acc']):.1%} — about {round(abs(res[LR]['acc'] - res[SVM]['acc']) * n_test)} patient(s).")

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

    st.header("Part 6: Comparison and Discussion")

    st.dataframe(pd.DataFrame({
        "Metric": ["Accuracy", "Precision (Diseased)", "Recall (Diseased)", "F1-score (Diseased)", "AUC", "Cancers missed", "False alarms"],
        LR: [pct(lr["acc"]), pct(lr["precision"][1]), pct(lr["recall"][1]), pct(lr["f1"][1]), f"{lr['auc']:.3f}", str(lr["cm"][1, 0]), str(lr["cm"][0, 1])],
        SVM: [pct(sv["acc"]), pct(sv["precision"][1]), pct(sv["recall"][1]), pct(sv["f1"][1]), f"{sv['auc']:.3f}", str(sv["cm"][1, 0]), str(sv["cm"][0, 1])],
    }), hide_index=True, width="stretch")

    st.subheader("Q1. Which model achieved higher accuracy?")
    if w == "tie":
        head, body = "It is a tie", f"Both models classified exactly the same number of the {n_test} test patients correctly."
    else:
        other = SVM if w == LR else LR
        head = f"{w}: {pct(res[w]['acc'])} vs {pct(res[other]['acc'])} for {other}"
        body = (f"{w} answered {n_diff} more patient{'s' if n_diff != 1 else ''} correctly out of {n_test} — a difference of {diff:.1%}. "
                "On a test this small, a single patient can flip the ranking, so in raw accuracy the two models are essentially tied.")
    st.markdown(f'<div class="verdict" style="background:{MODEL_COLOR.get(w, INK2)}"><h3>{head}</h3><p>{body}</p></div>',
                unsafe_allow_html=True)

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
            f"It matches SVM on the number of cancers caught ({pct(lr['recall'][1])}), trails it by only {n_diff} patient{'s' if n_diff != 1 else ''} "
            f"in accuracy, and — unlike SVM — can explain every decision it makes. SVM remains a valuable secondary check.",
            tag="Recommendation")


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
