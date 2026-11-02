#!/usr/bin/env python
# coding: utf-8

# # Early Warning of Food Price Spikes in Sri Lankan Markets with an Explainable Boosting Machine
# **MSc in AI — Machine Learning Assignment** · Prasadini A P N (269170A)
# 
# This notebook contains the complete pipeline: data collection and preprocessing, the new algorithm (Explainable Boosting Machine),
# training and evaluation against standard and lecture models, explainability (EBM shape functions, feature importance, PDP, LIME),
# a critical discussion (limitations, data quality, bias, ethics) and the model export used by the Streamlit front end.
# 
# | Section | Marking-scheme item |
# |---|---|
# | 1–3 | Problem definition & dataset collection (source, features, target, size, preprocessing, ethics) |
# | 4 | Selection of a new ML algorithm — EBM |
# | 5–6 | Model training & evaluation (split, hyperparameters, metrics, results) |
# | 7 | Explainability & interpretation |
# | 8 | Critical discussion |
# | 9 | Model export & single-sample inference (front-end integration) |

# In[1]:


import json, pickle, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit, ParameterSampler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score, precision_score, recall_score,
                             accuracy_score, roc_curve, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay)
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from lime.lime_tabular import LimeTabularExplainer

SEED = 42
np.random.seed(SEED)
ROOT = Path("..")
DATA, FIG, MODELS = ROOT / "data", ROOT / "figures", ROOT / "models"
FIG.mkdir(exist_ok=True); MODELS.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 130
def savefig(name):
    plt.tight_layout(); plt.savefig(FIG / f"{name}.png", bbox_inches="tight", dpi=160); plt.show()


# ## 1. Problem definition and relevance
# Since the 2022 economic crisis (foreign-exchange shortage, fuel crisis, import restrictions and rupee depreciation), retail food prices
# in Sri Lanka roughly doubled and still move sharply and unevenly between markets. Food is the largest share of spending for
# low-income households, so a sudden rise in rice, dhal, eggs or fish directly affects food security.
# 
# **Goal.** An early-warning tool that tells households, traders and agencies (Divisional Secretariats, the Department of Census &
# Statistics, Consumer Affairs Authority, WFP) **which commodity–market pairs are likely to see a sharp price rise next month**, and
# *why*, so that they can plan purchases, stock or interventions.
# 
# **ML task — binary classification.** For market *m*, commodity *c* and month *t*: will the retail price rise by **more than 5 %
# in month t + 1**? (`spike_next_month` = 1/0)

# ## 2. Dataset collection
# * **Source:** World Food Programme (WFP) *Global Food Prices — Sri Lanka*, downloaded as CSV from the Humanitarian Data Exchange
#   (HDX). WFP compiles monthly market prices from its own market monitoring and from national sources (e.g. Department of Census &
#   Statistics / HARTI price bulletins).
# * **Content:** monthly retail prices in LKR per commodity and market, with province, district, coordinates and food group.
# * **Ethics:** the data are aggregated market prices published openly for humanitarian use. They contain **no personal or sensitive
#   information** about individuals, so no consent issues arise. The source is credited in the report and app.

# In[2]:


raw = pd.read_csv(DATA / "wfp_food_prices_lka.csv", skiprows=[1])   # row 1 holds HXL tags (#date, #adm1 ...)
raw["date"] = pd.to_datetime(raw["date"])
print("Raw shape:", raw.shape)
print("Period:", raw.date.min().date(), "to", raw.date.max().date())
print("Provinces:", raw.admin1.nunique(), "| Markets:", raw.market.nunique(), "| Commodities:", raw.commodity.nunique())
print("\nPrice types:\n", raw.pricetype.value_counts().to_string())
print("\nMissing values:\n", raw.isna().sum()[lambda s: s > 0].to_string())
print("Duplicate rows:", raw.duplicated().sum())
raw.head()


# In[3]:


fig, ax = plt.subplots(1, 2, figsize=(13, 4))
raw.groupby(raw.date.dt.year).size().plot.bar(ax=ax[0], color="#4C72B0")
ax[0].set_title("Price observations per year"); ax[0].set_xlabel(""); ax[0].set_ylabel("rows")
raw.category.value_counts().sort_values().plot.barh(ax=ax[1], color="#55A868")
ax[1].set_title("Observations by food group")
savefig("01_coverage")


# In[4]:


key_items = ["Rice (white)", "Rice (red nadu)", "Wheat flour", "Lentils", "Coconut", "Sugar"]
nat = (raw[(raw.pricetype == "Retail") & raw.commodity.isin(key_items)]
       .groupby([pd.Grouper(key="date", freq="QS"), "commodity"]).price.median().unstack())
ax = nat.plot(figsize=(12, 4.5), marker=".", lw=1.2)
ax.axvspan(pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31"), color="red", alpha=.08)
ax.text(pd.Timestamp("2022-02-01"), ax.get_ylim()[1]*.92, "2022 economic crisis", color="darkred")
ax.set_title("National median retail price of staple items (LKR, quarterly)"); ax.set_ylabel("LKR"); ax.set_xlabel("")
ax.legend(ncol=3, fontsize=8)
savefig("02_staple_prices")


# **Data quality findings.** Coverage is very uneven (2018–2019 almost empty, ~40 markets reported monthly only from 2023), wholesale
# prices exist only for 2020–21, and 194 rows lack location fields. We therefore model **retail prices from Oct 2022 to Sep 2025**,
# where the panel is dense, and back-fill missing locations from other rows of the same market.

# ## 3. Dataset construction, features, target and preprocessing
# `src/prepare_food_price_dataset.py` turns the raw file into a supervised dataset:
# * **Cleaning:** retail only; one price per market × commodity × month; locations back-filled; a **regular monthly grid** so that
#   "last month" always means the previous calendar month (missed reports become blanks instead of silently shifting lags).
# * **Feature engineering** (all known at month *t*): price lags (1–3 months), % changes, 3-month volatility, deviation from the
#   3-month average, national median price and national % change of the same commodity, deviation from the national median, markets
#   reporting, location, food group, month, cultivation season (Maha Oct–Mar / Yala Apr–Sep), festive month (April Avurudu, December).
# * **Target:** `spike_next_month` = 1 if the price in *t + 1* is > 5 % above the price in *t*.
# 
# *Why classification and not the exact price?* A feasibility test showed no model beat the naive forecast "next month = this month"
# (MAPE 11.7 % vs 12.5 %): monthly price levels behave like a random walk, whereas the **risk of a sharp rise** is learnable.

# In[5]:


df = pd.read_csv(DATA / "food_price_spike_dataset.csv", parse_dates=["date", "target_month"])
latest = pd.read_csv(DATA / "food_price_spike_latest_month.csv", parse_dates=["date", "target_month"])
print("Engineered dataset:", df.shape, "| unlabelled latest-month rows:", latest.shape[0])

fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
sns.histplot(df.next_change_pct.clip(-40, 60), bins=80, ax=ax[0], color="#4C72B0")
ax[0].axvline(5, color="red", ls="--"); ax[0].set_title("Next-month price change (%)\n(red: 5 % spike threshold)")
(df.groupby("category").spike_next_month.mean().sort_values()*100).plot.barh(ax=ax[1], color="#C44E52")
ax[1].set_title("Spike rate by food group (%)"); ax[1].set_ylabel("")
(df.groupby(df.date.dt.month).spike_next_month.mean()*100).plot.bar(ax=ax[2], color="#8172B2")
ax[2].set_title("Spike rate by month of current price (%)"); ax[2].set_xlabel("month")
savefig("03_target_eda")
print(f"Overall spike rate: {df.spike_next_month.mean():.1%}")


# ### 3.1 Scope decision
# Vegetables & fruits spike most often, but their prices are driven by weather and harvests that are not in the data. A quick
# check with a fixed-parameter EBM on the time-based split shows how learnable each part is:

# In[6]:


CAT = ["province", "market", "category", "commodity", "unit", "season"]
PRICE = ["price", "price_lag1", "price_lag2", "price_lag3", "national_median_price"]
NUM = ["latitude", "longitude", "month", "festive_month",
       "change_1m_pct", "change_prev_1m_pct", "change_prev_2m_pct", "change_3m_pct",
       "price_vs_3m_avg_pct", "volatility_3m", "national_change_1m_pct", "price_vs_national_pct", "markets_reporting"]
FEATURES = CAT + PRICE + NUM
NOMINAL = set(CAT + ["month"])          # month treated as a category -> one score per calendar month
TARGET = "spike_next_month"

def make_ebm(feats, **params):
    return ExplainableBoostingClassifier(feature_names=feats, feature_types=["nominal" if f in NOMINAL else "continuous" for f in feats],
                                         random_state=SEED, n_jobs=-1, **params)

def quick_auc(mask):
    tr, te = mask & (df.split == "train"), mask & (df.split == "test")
    m = make_ebm(FEATURES, interactions=0, outer_bags=2, max_bins=32, learning_rate=0.05).fit(df.loc[tr, FEATURES], df.loc[tr, TARGET])
    return roc_auc_score(df.loc[te, TARGET], m.predict_proba(df.loc[te, FEATURES])[:, 1])

veg = df.is_vegetable_fruit == 1
scope = pd.Series({"All items": quick_auc(df.index == df.index), "Vegetables & fruits only": quick_auc(veg),
                   "Excluding vegetables & fruits": quick_auc(~veg)}, name="test ROC-AUC").round(3)
scope.to_frame()


# In[7]:


data = df[~veg].reset_index(drop=True)
latest_nv = latest[latest.is_vegetable_fruit == 0].reset_index(drop=True)
print("Final modelling dataset:", data.shape[0], "rows |", data.commodity.nunique(), "commodities |", data.market.nunique(), "markets |",
      data.province.nunique(), "provinces |", len(FEATURES), "candidate features")
print(data.groupby("split")[TARGET].agg(["size", "mean"]).rename(columns={"mean": "spike_rate"}))
nulls = data[FEATURES].isna().mean().mul(100).round(2)
nulls[nulls > 0].sort_values(ascending=False).to_frame("% missing (null detection)")


# ### 3.2 Preprocessing
# * **Missing values** come only from months a market did not report. For the comparison models they are **median-imputed with a
#   missing-indicator flag**; the **EBM handles missing values natively** (a dedicated "missing" bin with its own learned score), so
#   no artificial values are injected.
# * **Encoding:** one-hot encoding for the comparison models (unknown-category safe); the EBM treats categoricals as nominal features.
# * **Transformation & normalisation:** log transform of skewed price levels and standard scaling for the distance/linear models
#   (k-NN, logistic regression). The EBM bins each feature into quantile bins, so it is invariant to monotone transformations and needs
#   no scaling.
# * All preprocessing is fitted on training data only (inside a scikit-learn `Pipeline`).

# In[8]:


def make_preprocessor():
    price_pipe = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                           ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")), ("scale", StandardScaler())])
    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())])
    cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                         ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False))])
    return ColumnTransformer([("price", price_pipe, PRICE), ("num", num_pipe, NUM), ("cat", cat_pipe, CAT)])


# ## 4. Selection of a new algorithm — Explainable Boosting Machine (EBM)
# **What it is.** An EBM (Lou, Caruana et al.; implemented in Microsoft's InterpretML) is a **Generalised Additive Model with
# pairwise interactions (GA²M)**:
# 
# $$\text{logit}\,P(\text{spike}) = \beta_0 + \sum_j f_j(x_j) + \sum_{(i,j)} f_{ij}(x_i, x_j)$$
# 
# Each shape function $f_j$ is learned by **cyclic gradient boosting**: in every round the algorithm boosts a tiny tree on **one feature
# at a time** with a very low learning rate, cycling through all features thousands of times, then repeats this on bagged samples
# ("outer bags") and averages. Finally it detects the most important pairwise interactions (FAST algorithm) and learns them the same way.
# 
# **Why selected for this problem**
# 1. **Trust and accountability.** An early-warning tool that may influence purchases, stock or policy must say *why* it raises an alert.
#    The EBM is a **glass-box (intrinsically interpretable) model**: every prediction is an exact sum of per-feature contributions, so
#    explanations are faithful, not post-hoc approximations.
# 2. **Captures non-linear market behaviour** (thresholds in price changes, seasonality by month, mean reversion) that logistic
#    regression cannot, while staying additive and readable.
# 3. **Native handling of categoricals and missing values** — important with 21 commodities, 9 provinces and irregular reporting.
# 4. Not a deep-learning model and not covered in the lectures.
# 
# **How it differs from standard models**
# 
# | | Decision tree | Logistic regression | k-NN | Random forest / XGBoost | **EBM** |
# |---|---|---|---|---|---|
# | Model form | one tree of if-then splits | linear in features | majority vote of nearest neighbours | hundreds of deep trees mixing all features | **sum of one-feature (and few two-feature) shape functions** |
# | Non-linearity | yes, but unstable | no (unless engineered) | yes, local | yes | **yes, per feature** |
# | Interactions | implicit, all | none | implicit | implicit, high-order | **explicit, only selected pairs** |
# | Interpretability | readable only if shallow | coefficients | none (instance-based) | needs post-hoc SHAP/LIME | **exact global & local explanations built in** |
# | Scaling / missing values | not needed / needs imputation | needs scaling & imputation | needs scaling & imputation | XGB handles NaN | **not needed / native missing bin** |
# | Training | fast | fast | no training (slow prediction) | fast | **slower (many small boosting rounds), fast prediction** |

# ## 5. Training and evaluation set-up
# ### 5.1 Train / validation / test split (time-based)
# Shuffling would leak future information, so the split follows time:
# * **Training-fit:** target months Dec 2022 – Jun 2024
# * **Validation:** Jul – Dec 2024 — used for hyperparameter choice, feature-set choice and the alert threshold
# * **Test (unseen):** Jan – Sep 2025 — used once, for the final evaluation
# 
# After tuning, each model is refitted on training-fit + validation and evaluated on the test set.

# In[9]:


train = data[data.split == "train"].sort_values("target_month").reset_index(drop=True)
test = data[data.split == "test"].reset_index(drop=True)
is_val = (train.target_month >= "2024-07-01").values
y_train, y_test = train[TARGET].astype(int).values, test[TARGET].astype(int).values
ps = PredefinedSplit(np.where(is_val, 0, -1))
split_tbl = pd.DataFrame({"rows": [(~is_val).sum(), is_val.sum(), len(test)],
                          "spike rate": [y_train[~is_val].mean(), y_train[is_val].mean(), y_test.mean()]},
                         index=["train-fit (Dec 2022–Jun 2024)", "validation (Jul–Dec 2024)", "test (Jan–Sep 2025)"]).round(3)
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
split_tbl


# ### 5.2 Metrics
# The classes are imbalanced (~21 % spikes), so accuracy alone is misleading. We report **ROC-AUC** (ranking quality, threshold-free),
# **PR-AUC** (precision–recall trade-off; random = spike rate), and **F1, precision, recall** at an alert threshold chosen on the
# validation months (maximising F1). For an early-warning tool, recall (spikes caught) and precision (alerts that are real) both matter.

# In[10]:


results, probas, fitted, val_auc = {}, {}, {}, {}
def evaluate(name, p_test, thr=0.5):
    pred = (p_test >= thr).astype(int)
    results[name] = {"ROC-AUC": roc_auc_score(y_test, p_test), "PR-AUC": average_precision_score(y_test, p_test),
                     "F1": f1_score(y_test, pred), "Precision": precision_score(y_test, pred, zero_division=0),
                     "Recall": recall_score(y_test, pred), "Accuracy": accuracy_score(y_test, pred), "threshold": thr}
    probas[name] = p_test

def best_threshold(y_true, p):
    pr, rc, t = precision_recall_curve(y_true, p)
    f1 = 2 * pr * rc / (pr + rc + 1e-9)
    return float(t[np.argmax(f1[:-1])])


# ## 6. Model training
# ### 6.1 EBM — feature set and hyperparameters (chosen on validation)
# First we check on the validation months whether all candidate features help the EBM. Absolute price levels drift with inflation
# (a model trained on 2023 prices sees different levels in 2025), and market identity / coordinates add many high-variance terms.

# In[11]:


DROP_FOR_EBM = ["market", "latitude", "longitude"] + PRICE
FS = {"all features": FEATURES,
      "without market identity, coordinates & absolute price levels": [f for f in FEATURES if f not in DROP_FOR_EBM]}
base_params = dict(interactions=10, outer_bags=4, max_bins=32, learning_rate=0.02, min_samples_leaf=30)
fs_scores = {}
for name, feats in FS.items():
    t0 = time.time()
    m = make_ebm(feats, **base_params).fit(train.loc[~is_val, feats], y_train[~is_val])
    fs_scores[name] = roc_auc_score(y_train[is_val], m.predict_proba(train.loc[is_val, feats])[:, 1])
    print(f"{name}: validation AUC = {fs_scores[name]:.3f} ({time.time()-t0:.0f}s)")
EBM_FEATURES = FS[max(fs_scores, key=fs_scores.get)]
print("\nSelected EBM features:", EBM_FEATURES)


# **Hyperparameters.** No defaults are used; the search space follows from how the EBM works:
# 
# | Hyperparameter | Values tried | Reasoning |
# |---|---|---|
# | `learning_rate` | 0.01, 0.02, 0.05 | small steps per feature make cyclic boosting order-independent and smooth |
# | `max_bins` | 32, 64 | fewer bins = smoother shape functions, less over-fitting on noisy monthly changes |
# | `max_leaves` | 2, 3 | size of the tiny per-feature trees (2 = stumps) |
# | `min_samples_leaf` | 30, 60 | minimum evidence for each bin's score |
# | `interactions` | 5, 10, 20 | number of pairwise terms — more capacity vs. readability |
# | `smoothing_rounds` | 0, 200 | initial rounds with random splits to smooth shape functions |
# | `outer_bags` | 4 during search, 8 final | bagging for stable shapes and confidence intervals |

# In[12]:


ebm_space = {"learning_rate": [0.01, 0.02, 0.05], "max_bins": [32, 64], "max_leaves": [2, 3], "min_samples_leaf": [30, 60],
             "interactions": [5, 10, 20], "smoothing_rounds": [0, 200]}
trials = []
t0 = time.time()
for p in ParameterSampler(ebm_space, n_iter=8, random_state=SEED):
    m = make_ebm(EBM_FEATURES, outer_bags=4, **p).fit(train.loc[~is_val, EBM_FEATURES], y_train[~is_val])
    pv = m.predict_proba(train.loc[is_val, EBM_FEATURES])[:, 1]
    trials.append({**p, "val_AUC": roc_auc_score(y_train[is_val], pv), "_p": pv})
tr_df = pd.DataFrame(trials).sort_values("val_AUC", ascending=False)
best = tr_df.iloc[0]
EBM_PARAMS = {k: (int(best[k]) if k != "learning_rate" else float(best[k])) for k in ebm_space}
ebm_thr = best_threshold(y_train[is_val], best["_p"])
print(f"search took {time.time()-t0:.0f}s | best params: {EBM_PARAMS} | validation AUC {best.val_AUC:.3f} | alert threshold {ebm_thr:.3f}")
tr_df.drop(columns="_p").round(3)


# In[13]:


t0 = time.time()
ebm = make_ebm(EBM_FEATURES, outer_bags=8, **EBM_PARAMS).fit(train[EBM_FEATURES], y_train)
fitted["EBM (new algorithm)"] = ebm; val_auc["EBM (new algorithm)"] = best.val_AUC
evaluate("EBM (new algorithm)", ebm.predict_proba(test[EBM_FEATURES])[:, 1], ebm_thr)
print(f"final EBM fitted in {time.time()-t0:.0f}s with {len(ebm.term_names_)} terms "
      f"({sum('&' in t for t in ebm.term_names_)} pairwise interactions)")
pd.Series(results["EBM (new algorithm)"]).round(3).to_frame("EBM test")


# ### 6.2 Comparison models
# * **Rule-based "existing approaches":** momentum ("rose > 5 % this month → will rise again") and national trend.
# * **Standard models:** logistic regression, decision tree, k-NN.
# * **Lecture ensembles (reference):** Random Forest, XGBoost, CatBoost.
# 
# Each is tuned on the same validation months, its threshold chosen on validation, then refitted on the full training period.

# In[14]:


def rule_score(x):
    return (1 / (1 + np.exp(-(x.fillna(0) - 5) / 3))).values
evaluate("Rule: momentum (last change > 5%)", rule_score(test.change_1m_pct))
evaluate("Rule: national trend > 5%", rule_score(test.national_change_1m_pct))

def tune(name, model, space, n_iter):
    t0 = time.time()
    pipe = Pipeline([("prep", make_preprocessor()), ("model", model)])
    s = RandomizedSearchCV(pipe, space, n_iter=n_iter, cv=ps, scoring="roc_auc", random_state=SEED, n_jobs=1, refit=False)
    s.fit(train[FEATURES], y_train)
    p = s.best_params_
    m_fit = pipe.set_params(**p).fit(train.loc[~is_val, FEATURES], y_train[~is_val])          # threshold on validation
    thr = best_threshold(y_train[is_val], m_fit.predict_proba(train.loc[is_val, FEATURES])[:, 1])
    final = Pipeline([("prep", make_preprocessor()), ("model", model)]).set_params(**p).fit(train[FEATURES], y_train)
    fitted[name], val_auc[name] = final, s.best_score_
    evaluate(name, final.predict_proba(test[FEATURES])[:, 1], thr)
    pretty = ", ".join(f"{k.split('__')[1]}={v}" for k, v in p.items())
    print(f"{name:<20} val AUC {s.best_score_:.3f} | {time.time()-t0:4.0f}s | {pretty}")

tune("Logistic Regression", LogisticRegression(max_iter=3000, class_weight="balanced"),
     {"model__C": [0.003, 0.01, 0.03, 0.1, 0.3, 1]}, 6)
tune("Decision Tree", DecisionTreeClassifier(class_weight="balanced", random_state=SEED),
     {"model__max_depth": [3, 4, 5, 6, 8], "model__min_samples_leaf": [20, 50, 100, 200]}, 10)
tune("k-NN", KNeighborsClassifier(n_jobs=-1),
     {"model__n_neighbors": [25, 50, 100, 200], "model__weights": ["uniform", "distance"]}, 8)
tune("Random Forest", RandomForestClassifier(class_weight="balanced_subsample", n_jobs=-1, random_state=SEED),
     {"model__n_estimators": [300, 500], "model__max_depth": [8, 12, 16], "model__min_samples_leaf": [5, 10, 20],
      "model__max_features": ["sqrt", 0.5]}, 8)
tune("XGBoost", XGBClassifier(n_jobs=-1, random_state=SEED, eval_metric="auc", scale_pos_weight=pos_weight),
     {"model__n_estimators": [300, 500, 800], "model__learning_rate": [0.02, 0.05], "model__max_depth": [3, 4, 6],
      "model__subsample": [0.6, 0.8], "model__colsample_bytree": [0.6, 0.8], "model__min_child_weight": [1, 5],
      "model__reg_lambda": [1, 10]}, 10)

# CatBoost on raw categories (cat_features is not compatible with sklearn cloning, so its search is written out)
Xc_tr, Xc_te = train[FEATURES].copy(), test[FEATURES].copy()
Xc_tr[CAT] = Xc_tr[CAT].astype(str); Xc_te[CAT] = Xc_te[CAT].astype(str)
def make_cb(**p): return CatBoostClassifier(**p, random_seed=SEED, verbose=0, auto_class_weights="Balanced", cat_features=CAT)
t0, cb_trials = time.time(), []
for p in ParameterSampler({"iterations": [500, 800], "learning_rate": [0.03, 0.05], "depth": [4, 6, 8], "l2_leaf_reg": [3, 10]},
                          n_iter=6, random_state=SEED):
    m = make_cb(**p).fit(Xc_tr[~is_val], y_train[~is_val]); pv = m.predict_proba(Xc_tr[is_val])[:, 1]
    cb_trials.append((roc_auc_score(y_train[is_val], pv), p, pv))
a, p, pv = max(cb_trials, key=lambda t: t[0])
fitted["CatBoost"], val_auc["CatBoost"] = make_cb(**p).fit(Xc_tr, y_train), a
evaluate("CatBoost", fitted["CatBoost"].predict_proba(Xc_te)[:, 1], best_threshold(y_train[is_val], pv))
print(f"{'CatBoost':<20} val AUC {a:.3f} | {time.time()-t0:4.0f}s | {p}")


# ### 6.3 Results

# In[15]:


order = ["EBM (new algorithm)", "Logistic Regression", "Decision Tree", "k-NN", "Random Forest", "XGBoost", "CatBoost",
         "Rule: national trend > 5%", "Rule: momentum (last change > 5%)"]
res = pd.DataFrame(results).T.loc[order, ["ROC-AUC", "PR-AUC", "F1", "Precision", "Recall", "Accuracy", "threshold"]]
res.insert(0, "val ROC-AUC", pd.Series(val_auc))
group = {"EBM (new algorithm)": "new (glass-box)", "Logistic Regression": "standard", "Decision Tree": "standard", "k-NN": "standard",
         "Random Forest": "lecture ensemble", "XGBoost": "lecture ensemble", "CatBoost": "lecture ensemble"}
res.insert(0, "type", [group.get(i, "rule baseline") for i in res.index])
res.round(3)


# **What the results indicate (test set, Jan–Sep 2025)**
# 
# * **Rules fail.** The momentum rule scores ROC-AUC **0.50** (a coin flip) and the national-trend rule 0.58: after a jump, local prices
#   tend to *fall back*, so naive alerts would mislead.
# * **The EBM (0.711 ROC-AUC, 0.380 PR-AUC) beats every standard model** — logistic regression 0.697, decision tree 0.624, k-NN 0.654.
#   k-NN suffers because distances across ~100 mixed one-hot and numeric columns are not meaningful; a single tree is unstable on noisy
#   monthly changes.
# * **At its validation-chosen threshold the EBM gives the most precise alerts of all models** (precision 0.403, accuracy 0.727): about
#   40 % of its alerts are real spikes against a 22 % base rate (1.8× better than chance), and it catches about half of all spikes (recall 0.475).
# * **Lecture ensembles are only slightly more accurate** (ROC-AUC 0.732–0.741, +0.02–0.03) and reach higher recall at their own
#   thresholds, but they are black boxes that need post-hoc approximations to be explained. The EBM's validation (0.725) and test
#   (0.711) scores are consistent, so it is not over-fitted.
# * **Interpretation:** the model is a useful **screening / watch-list tool**, not a certainty — it narrows attention to a short list of
#   high-risk market–commodity pairs, and the small accuracy cost buys exact, auditable explanations.

# In[16]:


fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
palette = dict(zip(order, ["#D62728", "#1F77B4", "#8C564B", "#E377C2", "#2CA02C", "#FF7F0E", "#9467BD", "#7F7F7F", "#BCBD22"]))
for n in order:
    lw = 3 if n.startswith("EBM") else 1.3
    fpr, tpr, _ = roc_curve(y_test, probas[n]); ax[0].plot(fpr, tpr, lw=lw, color=palette[n], label=f"{n} ({results[n]['ROC-AUC']:.3f})")
    p_, r_, _ = precision_recall_curve(y_test, probas[n]); ax[1].plot(r_, p_, lw=lw, color=palette[n], label=f"{n} ({results[n]['PR-AUC']:.3f})")
ax[0].plot([0, 1], [0, 1], "k--", lw=.8); ax[0].set(title="ROC curves — test 2025", xlabel="False positive rate", ylabel="True positive rate")
ax[1].axhline(y_test.mean(), color="k", ls="--", lw=.8); ax[1].set(title="Precision–recall curves — test 2025", xlabel="Recall", ylabel="Precision", ylim=(0, 1))
ax[0].legend(fontsize=7.5, loc="lower right"); ax[1].legend(fontsize=7.5, loc="upper right")
savefig("04_roc_pr")


# In[17]:


p_ebm = probas["EBM (new algorithm)"]; pred_ebm = (p_ebm >= ebm_thr).astype(int)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ConfusionMatrixDisplay(confusion_matrix(y_test, pred_ebm), display_labels=["no spike", "spike"]).plot(ax=ax[0], cmap="Blues", colorbar=False)
ax[0].set_title(f"EBM — confusion matrix (test, threshold {ebm_thr:.2f})")
m_plot = res.drop(index=[i for i in res.index if i.startswith("Rule")])[["ROC-AUC", "PR-AUC", "F1"]].sort_values("ROC-AUC")
m_plot.plot.barh(ax=ax[1], width=.8); ax[1].set_xlim(.3, .8); ax[1].set_title("Test performance by model")
for c in ax[1].containers: ax[1].bar_label(c, fmt="%.3f", fontsize=7, padding=2)
savefig("05_confusion_comparison")


# ## 7. Explainability and interpretation
# Because the EBM is additive, its own shape functions **are** the explanation (no approximation). We complement them with
# (a) **feature-importance analysis** (EBM term importance and model-agnostic permutation importance on the test set),
# (b) **Partial Dependence Plots** as an independent check, and (c) **LIME** for a single prediction, compared with the EBM's exact
# local contributions.
# 
# ### 7.1 Global feature importance

# In[18]:


imp = pd.Series(ebm.term_importances(), index=ebm.term_names_).sort_values()
perm = permutation_importance(ebm, test[EBM_FEATURES], y_test, scoring="roc_auc", n_repeats=5, random_state=SEED, n_jobs=1)
perm_s = pd.Series(perm.importances_mean, index=EBM_FEATURES).sort_values()
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
imp.tail(15).plot.barh(ax=ax[0], color=["#D62728" if "&" in t else "#1F77B4" for t in imp.tail(15).index])
ax[0].set_title("EBM term importance (mean |score|)\nred = pairwise interaction")
perm_s.plot.barh(ax=ax[1], color="#2CA02C"); ax[1].set_title("Permutation importance on test set\n(drop in ROC-AUC when shuffled)")
savefig("06_importance")
imp.sort_values(ascending=False).head(10).round(3).to_frame("EBM importance")


# **Most influential features.** Both importance measures agree on the top three: **commodity**, **national price change this
# month** and **month** (seasonality), followed by the market's **price relative to the national median**, recent changes and
# **volatility**. The strongest single term in the EBM is the **commodity × month interaction** — seasonality differs by item (e.g. rice
# vs beans). Permutation importance is slightly negative for `festive_month`, `season` and `unit`: their information is already contained
# in `month` and `commodity`, so they are redundant (candidates for removal in a simpler model).

# ### 7.2 What the model has learned — shape functions
# Each panel shows the learned contribution (log-odds of a spike) of one feature, with the 95 % band from the outer bags.
# Above zero = raises the spike risk; below zero = lowers it.

# In[19]:


glob = ebm.explain_global()
def plot_term(ax, name, clip=None):
    i = ebm.term_names_.index(name); d = glob.data(i)
    sc, lo, hi = np.asarray(d["scores"]), np.asarray(d.get("lower_bounds", d["scores"])), np.asarray(d.get("upper_bounds", d["scores"]))
    if name in NOMINAL:
        names = [str(n) for n in d["names"]]
        o = np.argsort(sc) if name != "month" else np.arange(len(sc))
        ax.barh([names[k] for k in o], sc[o], xerr=[sc[o]-lo[o], hi[o]-sc[o]], color=np.where(sc[o] > 0, "#D62728", "#1F77B4"), alpha=.8)
        ax.axvline(0, color="k", lw=.7); ax.set_xlabel("contribution (log-odds)")
    else:
        edges = np.asarray(d["names"], dtype=float); x = edges[:-1]
        ax.step(x, sc, where="post", color="#D62728", lw=2); ax.fill_between(x, lo, hi, step="post", color="#D62728", alpha=.15)
        ax.axhline(0, color="k", lw=.7); ax.set_ylabel("contribution (log-odds)")
        if clip: ax.set_xlim(*clip)
    ax.set_title(name)

main_terms = [t for t in imp.sort_values(ascending=False).index if "&" not in t][:6]
fig, axes = plt.subplots(2, 3, figsize=(17, 9))
for ax, t in zip(axes.ravel(), main_terms):
    clip = tuple(test[t].quantile([.02, .98])) if t not in NOMINAL else None
    plot_term(ax, t, clip)
savefig("07_shape_functions")
print("Top main-effect terms:", main_terms)


# **What the model has learned (shape functions) and whether it matches domain knowledge**
# 
# | Learned pattern | Domain explanation | Aligned? |
# |---|---|---|
# | **Beans, fish (sardinella, skipjack) and potatoes** carry higher baseline risk; **rice, dried chilli and lentils** lower | perishable / weather-dependent supply (monsoon at sea, upcountry harvests) vs storable staples traded in large volumes and, for rice, subject to government price interventions | ✔ |
# | **National price rising** (> ~+2 %) → strongly higher risk; national fall → lower risk | price shocks spread between markets through traders with a lag (spatial price transmission) | ✔ |
# | **Market cheaper than the national median** → higher risk; more expensive → lower risk | spatial arbitrage: cheap markets catch up, expensive markets fall back (mean reversion) | ✔ |
# | **Large rise two months earlier** → lower risk | after a jump, prices usually stabilise — why the momentum rule fails | ✔ |
# | **Higher recent volatility** → higher risk (levels off above ~15) | unstable supply chains keep producing large moves | ✔ |
# | **Prices observed in November** → highest risk of a December rise; **January, July and October** → low | December festive/year-end demand and north-east-monsoon disruption; January is followed by the Maha harvest | mostly ✔ (July/October effects are less clear and may reflect only 2–3 years of data) |

# In[20]:


inter = [t for t in imp.sort_values(ascending=False).index if "&" in t]
if inter:
    t = inter[0]; d = glob.data(ebm.term_names_.index(t))
    ln, rn = d["left_names"], d["right_names"]
    sc = np.asarray(d["scores"])
    fig, ax = plt.subplots(figsize=(10, 6))
    yl = [str(v) if not isinstance(v, float) else f"{v:.1f}" for v in ln][:sc.shape[0]]
    xl = [str(v) if not isinstance(v, float) else f"{v:.1f}" for v in rn][:sc.shape[1]]
    sns.heatmap(sc, cmap="vlag", center=0, ax=ax, yticklabels=yl, xticklabels=xl, cbar_kws={"label": "log-odds"})
    ax.set_title(f"Strongest pairwise interaction: {t}")
    savefig("08_interaction")
print("Pairwise interactions learned:", inter)


# **Strongest interaction — commodity × month.** Seasonality is item-specific: **rice observed in January is strongly negative** (the Maha
# paddy harvest arrives in February–March and lowers rice prices), while **beans observed in September–December** and **mung beans observed in
# January–March** carry extra risk. A purely additive model without this term would force one seasonal pattern on all items.

# ### 7.3 Partial Dependence Plots (independent check)
# PDPs average the model's predicted probability over the test data while varying one feature. For an additive model they should
# follow the same shape as the EBM's own functions — a sanity check that the explanation reflects the model's real behaviour.

# In[21]:


pdp_feats = [t for t in main_terms if t not in NOMINAL][:3]
fig, ax = plt.subplots(1, len(pdp_feats), figsize=(5.5 * len(pdp_feats), 4))
X_pdp = test[EBM_FEATURES].dropna(subset=pdp_feats)   # PDP grid needs observed (non-missing) values
PartialDependenceDisplay.from_estimator(ebm, X_pdp, pdp_feats, ax=ax, percentiles=(0.03, 0.97), grid_resolution=40,
                                        line_kw={"color": "#D62728", "lw": 2})
fig.suptitle("Partial dependence of P(spike) on the top continuous features (test set)")
savefig("09_pdp")


# **PDP check.** The partial-dependence curves have the same shape as the EBM's own functions (risk jumps when the national change turns
# positive, and falls as a market becomes more expensive than the national median), confirming that the explanations describe the
# model's actual behaviour on the test data.

# ### 7.4 Local explanation of one alert — EBM exact contributions vs LIME

# In[22]:


tp = np.where((pred_ebm == 1) & (y_test == 1))[0]
idx = int(tp[np.argmax(p_ebm[tp])])
case = test.iloc[idx]
print(f"Case: {case.commodity} in {case.market} ({case.province}); features from {case.date:%b %Y} -> predicting {case.target_month:%b %Y}")
print(f"price Rs {case.price:.0f} -> Rs {case.next_price:.0f} ({case.next_change_pct:+.1f} %), actual spike = {int(case[TARGET])}; "
      f"EBM P(spike) = {p_ebm[idx]:.2f}")
loc = ebm.explain_local(test.loc[[idx], EBM_FEATURES], y_test[[idx]]).data(0)
contrib = pd.DataFrame({"term": loc["names"], "value": [str(v)[:22] for v in loc["values"]], "contribution": loc["scores"]})
contrib = contrib.reindex(contrib.contribution.abs().sort_values(ascending=False).index).head(10)
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
c = contrib.iloc[::-1]
ax[0].barh([f"{t} = {v}" if v else t for t, v in zip(c.term, c.value)], c.contribution, color=np.where(c.contribution > 0, "#D62728", "#1F77B4"))
ax[0].axvline(0, color="k", lw=.7); ax[0].set_title(f"EBM exact contributions (intercept {loc['extra']['scores'][0]:.2f})")

codes = {c_: sorted(train[c_].astype(str).unique()) for c_ in EBM_FEATURES if c_ in NOMINAL and c_ != "month"}
def enc(X):
    Z = X.copy()
    for c_, vals in codes.items(): Z[c_] = Z[c_].astype(str).map({v: i for i, v in enumerate(vals)}).fillna(0)
    return Z.astype(float).values
def dec_predict(Z):
    X = pd.DataFrame(Z, columns=EBM_FEATURES)
    for c_, vals in codes.items(): X[c_] = [vals[int(round(v))] for v in X[c_].clip(0, len(vals) - 1)]
    return ebm.predict_proba(X)
med = train[EBM_FEATURES].median(numeric_only=True)
lime = LimeTabularExplainer(enc(train[EBM_FEATURES].fillna(med)), feature_names=EBM_FEATURES, class_names=["no spike", "spike"],
                            categorical_features=[EBM_FEATURES.index(c_) for c_ in codes],
                            categorical_names={EBM_FEATURES.index(c_): v for c_, v in codes.items()}, random_state=SEED)
le = lime.explain_instance(enc(test.loc[[idx], EBM_FEATURES].fillna(med))[0], dec_predict, num_features=10, num_samples=3000)
lw_ = pd.DataFrame(le.as_list(), columns=["condition", "weight"]).iloc[::-1]
ax[1].barh(lw_.condition, lw_.weight, color=np.where(lw_.weight > 0, "#D62728", "#1F77B4")); ax[1].axvline(0, color="k", lw=.7)
ax[1].set_title("LIME local surrogate (towards 'spike')")
savefig("10_local_explanation")
contrib.round(3)


# **Local explanation.** For trenched sardinella in Monaragala (Jun → Jul 2025) the EBM's exact breakdown and LIME agree on the main
# reasons for the alert: the **national fish price was already rising** (+4.1 %), **Monaragala was ~20 % cheaper than the national median
# and ~14 % below its own 3-month average** (room to catch up), and the commodity itself carries high risk. The price did rise by
# 12.9 %. The EBM contributions are exact and additive (intercept + contributions = predicted log-odds); LIME's weights are approximate
# and depend on random sampling, which is why the glass-box explanation is preferred in the app.

# ## 8. Critical discussion
# ### 8.1 Performance by group — risk of bias or unfairness
# An alert system could treat regions unequally (e.g. more false alarms in the North and East, where reporting is less regular).
# We check alert rates and error rates per province and per food group on the test set.

# In[23]:


def group_report(col):
    g = test.assign(p=p_ebm, alert=pred_ebm, y=y_test).groupby(col)
    out = g.apply(lambda s: pd.Series({
        "rows": len(s), "markets": s.market.nunique(), "actual spike rate": s.y.mean(), "alert rate": s.alert.mean(),
        "ROC-AUC": roc_auc_score(s.y, s.p) if s.y.nunique() > 1 else np.nan,
        "recall": recall_score(s.y, s.alert, zero_division=0), "precision": precision_score(s.y, s.alert, zero_division=0),
        "missing lag-3 (%)": s.price_lag3.isna().mean() * 100}))
    return out.sort_values("ROC-AUC")
prov = group_report("province"); food = group_report("category")
fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
prov[["ROC-AUC", "recall", "precision"]].plot.barh(ax=ax[0], width=.8); ax[0].set_title("EBM test performance by province"); ax[0].set_xlim(0, 1)
prov[["actual spike rate", "alert rate"]].plot.barh(ax=ax[1], width=.8, color=["#7F7F7F", "#D62728"]); ax[1].set_title("Actual spike rate vs alert rate by province")
savefig("11_fairness_province")
prov.round(3)


# **Bias / unfairness check.** Performance is fairly even across seven provinces (ROC-AUC 0.71–0.74), but **weaker in the Northern
# province (AUC 0.61, precision 0.25, recall 0.33) and the Western province (AUC 0.67)**. In the North the model raises alerts for 25 %
# of cases while only 19 % spike — more false alarms. Plausible causes: few markets (5), more missing reports (≈12 % missing lag-3), and
# different supply chains (Jaffna peninsula and Vanni markets supplied over long distances). If deployed, alerts for the North should be
# flagged as lower-confidence and the model monitored per region.

# In[24]:


food.round(3)


# **By food group:** cereals and dried chilli are ranked best (AUC ≈ 0.81) but cereals get many false alarms (precision 0.29), while
# fish/meat/eggs are hardest (AUC 0.67) because catches depend on weather at sea.
# 
# ### 8.2 Limitations of the model
# * **Moderate accuracy (ROC-AUC 0.71):** price history alone cannot foresee shocks — exchange-rate moves, fuel-price revisions, import
#   bans or levies, weather. Vegetables & fruits were excluded for this reason.
# * **Short history:** ~3 years of dense data, so each calendar month is seen only 2–3 times; seasonal shapes may change.
# * **Concept drift:** trained on a post-crisis stabilisation period; a new crisis would change behaviour → monthly retraining and drift
#   monitoring are needed.
# * **Binary 5 % threshold** treats a 5.1 % and a 40 % rise the same; validation used a single time window.
# 
# ### 8.3 Data-quality issues
# * Very uneven coverage before 2023; wholesale prices only 2020–21; 194 rows without location.
# * Up to 15 % missing lag values from months a market did not report; irregular reporters are concentrated in some provinces.
# * One price snapshot per month (mid-month), mixed units (KG, unit, litre), and a few items with almost no history (e.g. fresh chicken,
#   coconut oil — 5–6 rows) whose learned effects are unreliable (wide bands in the shape plot).
# 
# ### 8.4 Real-world impact and ethical considerations
# * **Positive impact:** earlier warning for households, traders and agencies (e.g. targeted buffer-stock releases, cash-transfer timing).
# * **Self-fulfilling prophecy / hoarding:** public alerts could trigger panic buying or speculative stock-holding that *causes* the spike.
#   Alerts should go first to decision-makers, with probabilities and explanations, not as public headlines.
# * **Unequal reliability** across provinces (see above) — users in the North must not be given false confidence.
# * **Human oversight and accountability:** the tool supports, not replaces, analysts; explanations make each alert auditable.
# * **Privacy:** only aggregated, openly published market prices are used — no personal data.

# ## 9. Model export and single-sample inference (front-end integration)
# The trained EBM, its feature list, the alert threshold and metadata are saved as one pickle used by the Streamlit app.
# Inference is demonstrated on **genuinely new data**: September 2025 observations, whose October 2025 outcome is not in the dataset.

# In[25]:


bundle = {"model": ebm, "model_name": "Explainable Boosting Machine", "features": EBM_FEATURES,
          "all_features": FEATURES, "cat_features": [f for f in EBM_FEATURES if f in NOMINAL and f != "month"],
          "threshold": ebm_thr, "spike_pct": 5.0, "params": EBM_PARAMS,
          "test_metrics": {k: round(float(v), 4) for k, v in results["EBM (new algorithm)"].items()},
          "comparison": res.drop(columns="type").round(3).reset_index().rename(columns={"index": "model"}).to_dict("records"),
          "trained_on": f"{train.target_month.min():%Y-%m} to {train.target_month.max():%Y-%m}",
          "commodities": sorted(data.commodity.unique()), "markets": sorted(data.market.unique())}
with open(MODELS / "spike_model.pkl", "wb") as f:
    pickle.dump(bundle, f)
json.dump({k: v for k, v in bundle.items() if k != "model"}, open(MODELS / "model_meta.json", "w"), indent=2, default=str)
res.round(3).to_csv(MODELS / "model_comparison.csv")
print("saved", MODELS / "spike_model.pkl", f"({(MODELS / 'spike_model.pkl').stat().st_size/1e6:.1f} MB)")


# In[26]:


with open(MODELS / "spike_model.pkl", "rb") as f:
    loaded = pickle.load(f)
one = latest_nv[(latest_nv.commodity == "Rice (white)") & (latest_nv.market == "Colombo City")]
prob = loaded["model"].predict_proba(one[loaded["features"]])[0, 1]
r = one.iloc[0]
print(f"New sample: {r.commodity} @ {r.market}, Rs {r.price:.0f} in {r.date:%b %Y}")
print(f"P(price rises > 5 % in {r.target_month:%b %Y}) = {prob:.2f} -> {'SPIKE ALERT' if prob >= loaded['threshold'] else 'no alert'} "
      f"(threshold {loaded['threshold']:.2f})")
scan = latest_nv[["market", "province", "commodity", "price"]].assign(spike_probability=ebm.predict_proba(latest_nv[EBM_FEATURES])[:, 1])
scan["alert"] = scan.spike_probability >= ebm_thr
print(f"\nOctober 2025 outlook: {scan.alert.sum()} of {len(scan)} market–commodity pairs flagged ({scan.alert.mean():.0%})")
scan.sort_values("spike_probability", ascending=False).head(10).round(2)


# ## 10. Conclusion
# * A **new glass-box algorithm — the Explainable Boosting Machine —** predicts > 5 % monthly price spikes for staples, pulses, fish,
#   eggs, chicken and chilli across 38 Sri Lankan markets with **ROC-AUC 0.711 and the most precise alerts of all models** on unseen
#   2025 data, beating rules and all standard models and coming within 0.02–0.03 AUC of black-box ensembles.
# * Its learned shape functions match market economics (spatial price transmission, catch-up of cheap markets, mean reversion after
#   jumps, item-specific seasonality), so its alerts are explainable to non-experts.
# * Applied to September 2025 data, it flags **96 of 576 market–commodity pairs (17 %)** for October 2025, led by beans across Uva,
#   Central and other provinces.
# * **Future work:** add exchange-rate, fuel-price and rainfall data (to include vegetables), predict the size of the rise, and deploy
#   with per-region monitoring and monthly retraining.
