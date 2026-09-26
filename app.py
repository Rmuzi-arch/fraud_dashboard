"""Fraud analysis dashboard — hacker-terminal styled Streamlit app."""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).parent / "credit_card_fraud_10k.csv"
TARGET = "is_fraud"

GREEN, CYAN, RED, AMBER = "#00ff41", "#00e5ff", "#ff0055", "#ffb000"
BG, PANEL, TEXT, MUTED = "#0a0e0a", "#0f1a12", "#b8ffc8", "#4a7a55"

st.set_page_config(page_title="FRAUD_ANALYZER", page_icon="💀", layout="wide")

# ---------- Plotly hacker template ----------
pio.templates["hacker"] = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor=PANEL,
        plot_bgcolor=BG,
        font=dict(family="JetBrains Mono, Fira Code, monospace", color=TEXT, size=12),
        colorway=[GREEN, RED, CYAN, AMBER, "#b967ff"],
        xaxis=dict(gridcolor="#143d1e", zerolinecolor="#1f5c2c", linecolor="#1f5c2c"),
        yaxis=dict(gridcolor="#143d1e", zerolinecolor="#1f5c2c", linecolor="#1f5c2c"),
        margin=dict(l=40, r=20, t=50, b=40),
        title=dict(font=dict(color=GREEN, size=15)),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
)
pio.templates.default = "plotly_dark+hacker"
FRAUD_COLORS = {"LEGIT": GREEN, "FRAUD": RED}

# ---------- CSS ----------
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, button, input, select, textarea {{
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
}}
.stApp {{
    background: {BG};
    background-image: repeating-linear-gradient(0deg, rgba(0,255,65,.025) 0 1px, transparent 1px 3px);
}}
section[data-testid="stSidebar"] {{ background: {PANEL}; border-right: 1px solid #1f5c2c; }}
h1, h2, h3 {{ color: {GREEN} !important; text-shadow: 0 0 8px rgba(0,255,65,.55); letter-spacing: 1px; }}
.term-title {{ font-size: 2.1rem; font-weight: 700; color: {GREEN};
    text-shadow: 0 0 10px rgba(0,255,65,.7); margin-bottom: 0; }}
.term-sub {{ color: {MUTED}; margin-top: 2px; }}
.cursor {{ animation: blink 1s steps(1) infinite; }}
@keyframes blink {{ 50% {{ opacity: 0; }} }}
.kpi {{ background: {PANEL}; border: 1px solid #1f5c2c; border-radius: 6px; padding: 14px 16px;
    box-shadow: 0 0 12px rgba(0,255,65,.12), inset 0 0 18px rgba(0,255,65,.04); }}
.kpi .label {{ color: {MUTED}; font-size: .78rem; text-transform: uppercase; letter-spacing: 1.5px; }}
.kpi .value {{ font-size: 1.8rem; font-weight: 700; }}
.kpi .delta {{ font-size: .8rem; color: {MUTED}; }}
.stTabs [data-baseweb="tab"] {{ color: {MUTED}; }}
.stTabs [aria-selected="true"] {{ color: {GREEN} !important; text-shadow: 0 0 6px rgba(0,255,65,.6); }}
.stTabs [data-baseweb="tab-highlight"] {{ background-color: {GREEN}; }}
.insight {{ border-left: 3px solid {CYAN}; background: rgba(0,229,255,.05); padding: 8px 12px;
    margin: 6px 0; color: {TEXT}; }}
</style>
""",
    unsafe_allow_html=True,
)


def kpi(col, label, value, color=GREEN, delta=""):
    col.markdown(
        f"""<div class="kpi"><div class="label">{label}</div>
        <div class="value" style="color:{color};text-shadow:0 0 8px {color}88">{value}</div>
        <div class="delta">{delta}</div></div>""",
        unsafe_allow_html=True,
    )


# ---------- Data ----------
@st.cache_data
def load_data(source) -> pd.DataFrame:
    df = pd.read_csv(source)
    df["label"] = np.where(df[TARGET] == 1, "FRAUD", "LEGIT")
    return df


st.markdown(
    '<p class="term-title">&gt; FRAUD_ANALYZER v1.0<span class="cursor">_</span></p>'
    '<p class="term-sub">// credit card transaction threat intelligence console</p>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### `$ ./filters`")
    upload = st.file_uploader("load other CSV (optional)", type="csv")

df_all = load_data(upload if upload is not None else DATA_PATH)
if TARGET not in df_all.columns:
    st.error(f"CSV must contain a `{TARGET}` column.")
    st.stop()

ID_COLS = {"transaction_id", TARGET, "label"}
num_cols = [c for c in df_all.select_dtypes("number").columns if c not in ID_COLS]
binary_cols = [c for c in num_cols if set(df_all[c].dropna().unique()) <= {0, 1}]
cont_cols = [c for c in num_cols if c not in binary_cols]
cat_cols = [c for c in df_all.select_dtypes(exclude="number").columns if c not in ID_COLS]

# ---------- Sidebar filters ----------
with st.sidebar:
    fraud_mode = st.radio("fraud status", ["ALL", "FRAUD only", "LEGIT only"], horizontal=True)
    mask = pd.Series(True, index=df_all.index)

    for c in cat_cols:
        opts = sorted(df_all[c].dropna().unique())
        sel = st.multiselect(c, opts, default=opts)
        mask &= df_all[c].isin(sel)

    for c in cont_cols:
        lo, hi = df_all[c].min(), df_all[c].max()
        is_int = pd.api.types.is_integer_dtype(df_all[c])
        lo, hi = (int(lo), int(hi)) if is_int else (float(np.floor(lo)), float(np.ceil(hi)))
        if lo == hi:
            continue
        r = st.slider(c, lo, hi, (lo, hi))
        mask &= df_all[c].between(*r)

    for c in binary_cols:
        v = st.radio(c, ["ALL", "YES", "NO"], horizontal=True)
        if v != "ALL":
            mask &= df_all[c] == (1 if v == "YES" else 0)

    if fraud_mode == "FRAUD only":
        mask &= df_all[TARGET] == 1
    elif fraud_mode == "LEGIT only":
        mask &= df_all[TARGET] == 0

df = df_all[mask]
base_rate = df_all[TARGET].mean()

if df.empty:
    st.warning("// no records match current filters")
    st.stop()

# ---------- KPIs ----------
n, n_fraud = len(df), int(df[TARGET].sum())
rate = n_fraud / n
c1, c2, c3, c4 = st.columns(4)
kpi(c1, "transactions", f"{n:,}", GREEN, f"of {len(df_all):,} total")
kpi(c2, "fraud cases", f"{n_fraud:,}", RED)
kpi(c3, "fraud rate", f"{rate:.2%}", RED if rate > base_rate else GREEN,
    f"baseline {base_rate:.2%} · lift ×{rate / base_rate:.1f}" if base_rate else "")
kpi(c4, "fraud $ volume", f"${df.loc[df[TARGET] == 1, 'amount'].sum():,.0f}" if "amount" in df else "—", AMBER)
st.write("")

tabs = st.tabs(["[01] OVERVIEW", "[02] FEATURE_EXPLORER", "[03] PREDICTORS",
                "[04] SEGMENTS", "[05] DATA_DUMP"])


def fraud_rate_by(frame, col, bins=None):
    key = pd.cut(frame[col], bins=bins) if bins else frame[col]
    g = frame.groupby(key, observed=True)[TARGET].agg(["mean", "size", "sum"]).reset_index()
    g.columns = [col, "fraud_rate", "count", "fraud"]
    g[col] = g[col].astype(str)
    return g


# ---------- Tab 1: Overview ----------
with tabs[0]:
    a, b = st.columns(2)
    if "transaction_hour" in df:
        g = fraud_rate_by(df, "transaction_hour")
        g["transaction_hour"] = g["transaction_hour"].astype(int)
        fig = px.bar(g, x="transaction_hour", y="fraud_rate", title="fraud rate by hour",
                     color="fraud_rate", color_continuous_scale=[GREEN, AMBER, RED],
                     hover_data=["count", "fraud"])
        fig.add_hline(y=base_rate, line_dash="dot", line_color=CYAN, annotation_text="baseline")
        fig.update_layout(yaxis_tickformat=".1%", coloraxis_showscale=False)
        a.plotly_chart(fig, width="stretch")
    if cat_cols:
        g = fraud_rate_by(df, cat_cols[0]).sort_values("fraud_rate")
        fig = px.bar(g, y=cat_cols[0], x="fraud_rate", orientation="h",
                     title=f"fraud rate by {cat_cols[0]}", hover_data=["count", "fraud"],
                     color="fraud_rate", color_continuous_scale=[GREEN, AMBER, RED])
        fig.add_vline(x=base_rate, line_dash="dot", line_color=CYAN)
        fig.update_layout(xaxis_tickformat=".1%", coloraxis_showscale=False)
        b.plotly_chart(fig, width="stretch")

    a, b = st.columns(2)
    pie = df["label"].value_counts().reset_index()
    fig = px.pie(pie, names="label", values="count", hole=.6, title="class balance",
                 color="label", color_discrete_map=FRAUD_COLORS)
    a.plotly_chart(fig, width="stretch")
    if binary_cols:
        rows = [{"flag": c, "value": "YES" if v else "NO", "fraud_rate": df.loc[df[c] == v, TARGET].mean()}
                for c in binary_cols for v in (1, 0)]
        fig = px.bar(pd.DataFrame(rows), x="flag", y="fraud_rate", color="value", barmode="group",
                     title="fraud rate by risk flag", color_discrete_map={"YES": RED, "NO": GREEN})
        fig.update_layout(yaxis_tickformat=".1%")
        b.plotly_chart(fig, width="stretch")

# ---------- Tab 2: Feature explorer ----------
with tabs[1]:
    feat = st.selectbox("select feature", cont_cols + binary_cols + cat_cols)
    a, b = st.columns(2)
    if feat in cont_cols:
        fig = px.histogram(df, x=feat, color="label", barmode="overlay", histnorm="probability density",
                           nbins=40, opacity=.65, title=f"{feat} distribution: fraud vs legit",
                           color_discrete_map=FRAUD_COLORS)
        a.plotly_chart(fig, width="stretch")
        fig = px.box(df, x="label", y=feat, color="label", points="outliers",
                     title=f"{feat} spread", color_discrete_map=FRAUD_COLORS)
        b.plotly_chart(fig, width="stretch")
        nb = st.slider("bins for fraud-rate curve", 3, 20, 10)
        edges = np.unique(np.quantile(df[feat], np.linspace(0, 1, nb + 1)))
        g = fraud_rate_by(df, feat, bins=list(edges)) if len(edges) > 2 else fraud_rate_by(df, feat)
    else:
        g = fraud_rate_by(df, feat)
    fig = px.bar(g, x=feat, y="fraud_rate", hover_data=["count", "fraud"], text="count",
                 title=f"fraud rate per {feat} bucket  (bar label = #transactions)",
                 color="fraud_rate", color_continuous_scale=[GREEN, AMBER, RED])
    fig.add_hline(y=base_rate, line_dash="dot", line_color=CYAN, annotation_text="baseline")
    fig.update_layout(yaxis_tickformat=".1%", coloraxis_showscale=False)
    st.plotly_chart(fig, width="stretch")

    if feat in cont_cols:
        s = df.groupby("label")[feat].describe().T
        st.dataframe(s.style.format("{:.2f}"), width="stretch")

# ---------- Tab 3: Predictive features ----------
@st.cache_resource
def feature_power(frame: pd.DataFrame):
    X = pd.get_dummies(frame[cont_cols + binary_cols + cat_cols], columns=cat_cols, dtype=int)
    y = frame[TARGET]
    corr = X.corrwith(y).fillna(0)
    mi = pd.Series(mutual_info_classif(X, y, random_state=0,
                                       discrete_features=[c not in cont_cols for c in X.columns]),
                   index=X.columns)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, stratify=y, random_state=0)
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced",
                                n_jobs=-1, random_state=0).fit(Xtr, ytr)
    auc = roc_auc_score(yte, rf.predict_proba(Xte)[:, 1])
    res = pd.DataFrame({"correlation": corr, "abs_corr": corr.abs(), "mutual_info": mi,
                        "rf_importance": rf.feature_importances_})
    norm = res[["abs_corr", "mutual_info", "rf_importance"]]
    res["score"] = (norm / norm.max().replace(0, 1)).mean(axis=1)
    return res.sort_values("score", ascending=False), auc


with tabs[2]:
    if fraud_mode != "ALL" or df[TARGET].nunique() < 2 or df[TARGET].sum() < 10:
        st.info("// need both classes (≥10 fraud cases) — set fraud status to ALL / widen filters")
    else:
        res, auc = feature_power(df.drop(columns="label"))
        a, b, c = st.columns(3)
        kpi(a, "random forest roc-auc", f"{auc:.3f}", CYAN, "on 25% hold-out")
        kpi(b, "top predictor", res.index[0], RED)
        kpi(c, "features scored", str(len(res)), GREEN)
        st.write("")
        metric = st.radio("rank by", ["score", "rf_importance", "mutual_info", "abs_corr"], horizontal=True,
                          help="score = mean of the three normalized metrics")
        r = res.sort_values(metric).reset_index(names="feature")
        fig = px.bar(r, x=metric, y="feature", orientation="h", title=f"feature ranking by {metric}",
                     color=metric, color_continuous_scale=[MUTED, GREEN, CYAN], height=max(350, 28 * len(r)))
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")

        fig = px.bar(res.sort_values("correlation").reset_index(names="feature"), x="correlation", y="feature",
                     orientation="h", title="direction: correlation with fraud (red = raises risk)",
                     color=res.sort_values("correlation")["correlation"].gt(0).map({True: "raises", False: "lowers"}).values,
                     color_discrete_map={"raises": RED, "lowers": GREEN}, height=max(350, 28 * len(res)))
        st.plotly_chart(fig, width="stretch")
        st.dataframe(res.style.format("{:.4f}").background_gradient(cmap="Greens", subset=["score"]),
                     width="stretch")

# ---------- Tab 4: Segments ----------
def bucket(frame, col, q=5):
    if col in cont_cols and frame[col].nunique() > 8:
        return pd.qcut(frame[col], q=q, duplicates="drop").astype(str)
    return frame[col].astype(str)


with tabs[3]:
    dims = cont_cols + binary_cols + cat_cols
    a, b, c = st.columns(3)
    fx = a.selectbox("X axis", dims, index=dims.index("transaction_hour") if "transaction_hour" in dims else 0)
    fy = b.selectbox("Y axis", dims, index=dims.index("device_trust_score") if "device_trust_score" in dims else 1)
    q = c.slider("quantile buckets", 3, 10, 5)
    t = df.assign(_x=bucket(df, fx, q), _y=bucket(df, fy, q))
    piv = t.pivot_table(index="_y", columns="_x", values=TARGET, aggfunc="mean", observed=True)
    cnt = t.pivot_table(index="_y", columns="_x", values=TARGET, aggfunc="size", observed=True)
    fig = go.Figure(go.Heatmap(z=piv.values, x=piv.columns, y=piv.index, customdata=cnt.values,
                               colorscale=[[0, BG], [.3, "#0d5c1f"], [.6, AMBER], [1, RED]],
                               hovertemplate="x=%{x}<br>y=%{y}<br>fraud rate=%{z:.2%}<br>n=%{customdata}<extra></extra>",
                               text=np.vectorize(lambda v: "" if np.isnan(v) else f"{v:.0%}")(piv.values),
                               texttemplate="%{text}"))
    fig.update_layout(title=f"fraud rate heatmap: {fy} × {fx}", xaxis_title=fx, yaxis_title=fy, height=480)
    st.plotly_chart(fig, width="stretch")

    st.markdown("### `$ top_risky_segments`")
    a, b = st.columns(2)
    seg_dims = a.multiselect("group by", dims, default=[d for d in ["merchant_category", "foreign_transaction",
                                                                   "location_mismatch"] if d in dims] or dims[:2])
    min_n = b.number_input("min transactions per segment", 5, 5000, 30, step=5)
    if seg_dims:
        t = df.assign(**{d: bucket(df, d, q) for d in seg_dims})
        seg = (t.groupby(seg_dims, observed=True)[TARGET].agg(transactions="size", fraud="sum", fraud_rate="mean")
               .reset_index())
        seg["lift"] = seg["fraud_rate"] / base_rate
        seg = seg[seg["transactions"] >= min_n].sort_values(["fraud_rate", "transactions"], ascending=False)
        st.dataframe(seg.style.format({"fraud_rate": "{:.2%}", "lift": "×{:.1f}"})
                     .background_gradient(cmap="Reds", subset=["fraud_rate"]),
                     width="stretch", hide_index=True)
        for _, r in seg.head(3).iterrows():
            if r["lift"] > 1.5:
                desc = " & ".join(f"{d}={r[d]}" for d in seg_dims)
                st.markdown(f'<div class="insight">⚠ <b>{desc}</b> → fraud rate {r.fraud_rate:.1%} '
                            f'(×{r.lift:.1f} baseline, n={r.transactions})</div>', unsafe_allow_html=True)

# ---------- Tab 5: Data table ----------
with tabs[4]:
    cols = [c for c in df.columns if c != "label"]
    a, b, c = st.columns([2, 1, 2])
    sort_col = a.selectbox("sort by", cols, index=cols.index("amount") if "amount" in cols else 0)
    asc = b.radio("order", ["DESC", "ASC"], horizontal=True) == "ASC"
    search = c.text_input("grep (transaction_id)", "")
    view = df[cols].sort_values(sort_col, ascending=asc)
    if search.strip() and "transaction_id" in view:
        view = view[view["transaction_id"].astype(str).str.contains(search.strip())]
    st.caption(f"// {len(view):,} rows")
    show = view.head(5000)
    st.dataframe(show.style.apply(
        lambda r: [f"background-color: rgba(255,0,85,.22); color: {RED}" if r[TARGET] == 1 else ""] * len(r),
        axis=1), width="stretch", hide_index=True, height=520)
    st.download_button("⬇ export filtered CSV", view.to_csv(index=False), "fraud_filtered.csv", "text/csv")
