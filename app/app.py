"""
Sri Lanka Food Price Spike Early-Warning — Streamlit front end for the Explainable Boosting Machine.
Run from the project root:   streamlit run app/app.py
"""
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
st.set_page_config(page_title="SL Food Price Spike Alert", page_icon="🌾", layout="wide")


@st.cache_resource
def load_bundle():
    with open(ROOT / "models" / "spike_model.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_reference():
    hist = pd.read_csv(ROOT / "data" / "food_price_spike_dataset.csv", parse_dates=["date"])
    latest = pd.read_csv(ROOT / "data" / "food_price_spike_latest_month.csv", parse_dates=["date", "target_month"])
    hist = hist[hist.is_vegetable_fruit == 0]
    latest = latest[latest.is_vegetable_fruit == 0]
    counts = hist.commodity.value_counts()
    hist = hist[hist.commodity.isin(counts[counts >= 100].index)]      # drop items with almost no history
    latest = latest[latest.commodity.isin(hist.commodity.unique())]
    markets = hist.groupby("market")[["province", "latitude", "longitude"]].first()
    items = hist.groupby("commodity")[["category", "unit"]].first()
    return hist, latest, markets, items


B = load_bundle()
MODEL, FEATURES, THR = B["model"], B["features"], B["threshold"]
hist, latest, MARKETS, ITEMS = load_reference()

NICE = {"national_change_1m_pct": "National price change this month (%)", "price_vs_national_pct": "Price vs national median (%)",
        "month": "Month of current price", "commodity": "Commodity", "market": "Market", "price": "Current price (Rs)",
        "change_1m_pct": "Change vs last month (%)", "change_3m_pct": "Change over 3 months (%)",
        "price_vs_3m_avg_pct": "Price vs 3-month average (%)", "volatility_3m": "Recent volatility (3 months)",
        "national_median_price": "National median price (Rs)", "markets_reporting": "Markets reporting nationally",
        "change_prev_1m_pct": "Change one month earlier (%)", "change_prev_2m_pct": "Change two months earlier (%)",
        "category": "Food group", "province": "Province", "season": "Season (Maha/Yala)", "unit": "Unit",
        "festive_month": "Festive month (Apr/Dec)"}
nice = lambda t: " × ".join(NICE.get(p.strip(), p.strip()) for p in t.split("&"))


def pct(a, b):
    if a is None or b is None or pd.isna(a) or pd.isna(b) or b == 0:
        return np.nan
    return (a / b - 1) * 100


def build_features(market, commodity, month, price, lag1, lag2, lag3, nat_price, nat_change, n_markets):
    """Re-create the engineered features exactly as in src/prepare_food_price_dataset.py."""
    lag2 = lag2 if lag2 and lag2 > 0 else np.nan
    lag3 = lag3 if lag3 and lag3 > 0 else np.nan
    ch = [pct(price, lag1), pct(lag1, lag2), pct(lag2, lag3)]
    window = [v for v in (price, lag1, lag2) if not pd.isna(v)]
    vol_vals = [v for v in ch if not pd.isna(v)]
    row = {
        "province": MARKETS.loc[market, "province"], "market": market,
        "category": ITEMS.loc[commodity, "category"], "commodity": commodity, "unit": ITEMS.loc[commodity, "unit"],
        "season": "Maha" if month in (10, 11, 12, 1, 2, 3) else "Yala",
        "price": price, "price_lag1": lag1, "price_lag2": lag2, "price_lag3": lag3, "national_median_price": nat_price,
        "latitude": MARKETS.loc[market, "latitude"], "longitude": MARKETS.loc[market, "longitude"],
        "month": month, "festive_month": int(month in (4, 12)),
        "change_1m_pct": ch[0], "change_prev_1m_pct": ch[1], "change_prev_2m_pct": ch[2], "change_3m_pct": pct(price, lag3),
        "price_vs_3m_avg_pct": pct(price, np.mean(window)) if len(window) >= 2 else np.nan,
        "volatility_3m": float(np.std(vol_vals, ddof=1)) if len(vol_vals) >= 2 else np.nan,
        "national_change_1m_pct": nat_change, "price_vs_national_pct": pct(price, nat_price),
        "markets_reporting": n_markets,
    }
    return pd.DataFrame([row])[FEATURES]


def explain(X):
    d = MODEL.explain_local(X).data(0)
    out = pd.DataFrame({"factor": [nice(n) for n in d["names"]],
                        "value": [("" if isinstance(v, str) and v == "" else (f"{v:,.2f}" if isinstance(v, (int, float, np.floating)) else str(v)))
                                  for v in d["values"]],
                        "contribution": d["scores"]})
    return out.reindex(out.contribution.abs().sort_values(ascending=False).index), d["extra"]["scores"][0]


# ---------------------------------------------------------------- UI
st.title("🌾 Sri Lanka Food Price Spike Early-Warning")
st.caption(f"Predicts whether a retail price will rise by more than {B['spike_pct']:.0f}% next month. "
           f"Model: {B['model_name']} (glass-box) trained on WFP market data ({B['trained_on']}). Alert threshold = {THR:.2f}.")

tab1, tab2, tab3, tab4 = st.tabs(["🔎 Check one item", f"🗺️ Market scan ({latest.target_month.max():%B %Y})",
                                  "🧠 How the model thinks", "ℹ️ About"])

with tab1:
    c1, c2 = st.columns(2)
    items = sorted(ITEMS.index); mkts = sorted(MARKETS.index)
    commodity = c1.selectbox("Commodity", items, index=items.index("Rice (white)") if "Rice (white)" in items else 0)
    market = c2.selectbox("Market", mkts, index=mkts.index("Colombo City") if "Colombo City" in mkts else 0)

    ref = latest[(latest.commodity == commodity) & (latest.market == market)]
    if not ref.empty:
        r = ref.iloc[0]
        st.info(f"Fields pre-filled with the latest WFP data for this market ({r.date:%B %Y}). Edit any value to run a what-if.")
        d = dict(month=int(r.month), price=r.price, lag1=r.price_lag1, lag2=r.price_lag2, lag3=r.price_lag3,
                 nat_price=r.national_median_price, nat_change=r.national_change_1m_pct, n=int(r.markets_reporting))
    else:
        h = hist[hist.commodity == commodity]
        st.warning("No recent WFP price for this pair — fields pre-filled with national typical values. Please enter local prices.")
        med = float(h.national_median_price.iloc[-1]) if len(h) else 100.0
        d = dict(month=int(h.month.iloc[-1]) if len(h) else 9, price=med, lag1=med, lag2=med, lag3=med,
                 nat_price=med, nat_change=0.0, n=int(h.markets_reporting.iloc[-1]) if len(h) else 30)

    fv = lambda v, default: float(default if pd.isna(v) else v)
    with st.form("inputs"):
        a, b, c, e = st.columns(4)
        price = a.number_input(f"Current price (Rs / {ITEMS.loc[commodity, 'unit']})", min_value=0.01, value=fv(d["price"], 100))
        lag1 = b.number_input("Price 1 month ago", min_value=0.01, value=fv(d["lag1"], d["price"]))
        lag2 = c.number_input("Price 2 months ago (0 = unknown)", min_value=0.0, value=fv(d["lag2"], 0))
        lag3 = e.number_input("Price 3 months ago (0 = unknown)", min_value=0.0, value=fv(d["lag3"], 0))
        a, b, c, e = st.columns(4)
        nat_price = a.number_input("National median price this month", min_value=0.01, value=fv(d["nat_price"], d["price"]))
        nat_change = b.number_input("National price change this month (%)", value=fv(d["nat_change"], 0))
        month = c.selectbox("Month of the current price", list(range(1, 13)), index=d["month"] - 1,
                            format_func=lambda m: pd.Timestamp(2025, m, 1).strftime("%B"))
        n_markets = e.number_input("Markets reporting nationally", min_value=1, max_value=60, value=d["n"])
        st.form_submit_button("Predict next month", type="primary")

    # prediction is shown for the pre-filled values and refreshed on every submit
    X = build_features(market, commodity, month, price, lag1, lag2, lag3, nat_price, nat_change, n_markets)
    p = float(MODEL.predict_proba(X)[0, 1])
    nxt = pd.Timestamp(2025, month, 1) + pd.offsets.MonthBegin(1)
    m1, m2 = st.columns([1, 2])
    with m1:
        st.metric(f"Spike probability for {nxt:%B}", f"{p:.0%}")
        st.progress(min(p, 1.0))
        if p >= THR:
            st.error(f"⚠️ SPIKE ALERT — {commodity} in {market} is likely to rise by more than 5% next month.")
        else:
            st.success(f"✅ No alert — a >5% rise is not expected for {commodity} in {market}.")
    with m2:
        ex, intercept = explain(X)
        top = ex.head(8).iloc[::-1]
        fig, ax = plt.subplots(figsize=(7, 3.6))
        labels = [f"{f} = {v}" if v else f for f, v in zip(top.factor, top.value)]
        ax.barh(labels, top.contribution, color=np.where(top.contribution > 0, "#D62728", "#1F77B4"))
        ax.axvline(0, color="k", lw=.7); ax.set_xlabel("contribution to spike risk (log-odds)")
        ax.tick_params(labelsize=8); ax.set_title("Why? Exact EBM contributions (red raises, blue lowers risk)", fontsize=10)
        st.pyplot(fig, clear_figure=True)
    with st.expander("All contributions and engineered features"):
        st.caption(f"Prediction = intercept ({intercept:.2f}) + sum of contributions, converted to a probability.")
        st.dataframe(ex.round(3), hide_index=True, width="stretch")

with tab2:
    scan = latest[["province", "market", "category", "commodity", "price", "change_1m_pct"]].copy()
    scan["spike_probability"] = MODEL.predict_proba(latest[FEATURES])[:, 1]
    scan["alert"] = scan.spike_probability >= THR
    f1, f2 = st.columns(2)
    prov = f1.multiselect("Province", sorted(scan.province.unique()))
    cats = f2.multiselect("Food group", sorted(scan.category.unique()))
    keep = pd.Series(True, index=scan.index)
    if prov: keep &= scan.province.isin(prov)
    if cats: keep &= scan.category.isin(cats)
    s = scan[keep]
    k1, k2, k3 = st.columns(3)
    k1.metric("Market–commodity pairs", len(s))
    k2.metric("Spike alerts", int(s.alert.sum()))
    k3.metric("Share flagged", f"{s.alert.mean():.0%}" if len(s) else "–")
    st.markdown("**Share of markets flagged, by commodity**")
    st.bar_chart(s.groupby("commodity").alert.mean().sort_values(ascending=False).head(15))
    st.markdown("**Highest-risk items**")
    st.dataframe(s.sort_values("spike_probability", ascending=False).round(2), hide_index=True, width="stretch")

with tab3:
    st.markdown("The EBM is a **glass-box model**: its prediction is the sum of one learned curve per feature (plus a few pairwise "
                "interactions). These curves *are* the model — nothing is approximated.")
    imp = pd.Series(MODEL.term_importances(), index=MODEL.term_names_).sort_values(ascending=False)
    cA, cB = st.columns([1, 1.3])
    with cA:
        st.markdown("**Overall importance of each term**")
        fig, ax = plt.subplots(figsize=(5.5, 6))
        top = imp.head(15).iloc[::-1]
        ax.barh([nice(t) for t in top.index], top.values, color=["#D62728" if "&" in t else "#1F77B4" for t in top.index])
        ax.tick_params(labelsize=8); ax.set_xlabel("mean |contribution|"); ax.set_title("red = interaction", fontsize=9)
        st.pyplot(fig, clear_figure=True)
    with cB:
        mains = [t for t in imp.index if "&" not in t]
        term = st.selectbox("Show the learned shape function for", mains, format_func=nice)
        dd = MODEL.explain_global().data(MODEL.term_names_.index(term))
        sc = np.asarray(dd["scores"]); lo = np.asarray(dd.get("lower_bounds", sc)); hi = np.asarray(dd.get("upper_bounds", sc))
        fig, ax = plt.subplots(figsize=(7, 4.6))
        if MODEL.feature_types_in_[MODEL.feature_names_in_.index(term)] == "nominal":
            names = [str(n) for n in dd["names"]]
            o = np.arange(len(sc)) if term == "month" else np.argsort(sc)
            ax.barh([names[k] for k in o], sc[o], color=np.where(sc[o] > 0, "#D62728", "#1F77B4"))
            ax.axvline(0, color="k", lw=.7); ax.set_xlabel("contribution (log-odds)"); ax.tick_params(labelsize=8)
        else:
            x = np.asarray(dd["names"], dtype=float)[:-1]
            ax.step(x, sc, where="post", color="#D62728", lw=2)
            ax.fill_between(x, lo, hi, step="post", color="#D62728", alpha=.15)
            ax.axhline(0, color="k", lw=.7); ax.set_ylabel("contribution (log-odds)"); ax.set_xlabel(nice(term))
            q = hist[term].quantile([.02, .98]).values if term in hist else None
            if q is not None and np.isfinite(q).all(): ax.set_xlim(*q)
        ax.set_title(nice(term)); st.pyplot(fig, clear_figure=True)
        st.caption("Above zero → raises the spike risk; below zero → lowers it. Shaded band = 95% range across bagged models.")

with tab4:
    st.markdown(f"""
**Problem.** Early warning of food price spikes (> 5 % month-on-month rise) in Sri Lankan retail markets.

**Data.** WFP Global Food Prices — Sri Lanka (Humanitarian Data Exchange), retail prices Oct 2022 – Sep 2025; staples, pulses,
fish, eggs, chicken and dried chilli (vegetables & fruits excluded: weather-driven). Aggregated market prices — no personal data.

**Model.** {B['model_name']} — a generalised additive model with pairwise interactions learned by cyclic gradient boosting
(InterpretML). Hyperparameters: `{B['params']}`.

**Test performance (Jan–Sep 2025, unseen):**
""")
    st.json(B["test_metrics"])
    st.markdown("**Comparison with standard and lecture models (same split):**")
    st.dataframe(pd.DataFrame(B["comparison"]).set_index("model"), width="stretch")
    st.caption("Decision-support tool — alerts are probabilities, not guarantees, and should be reviewed by a person.")
