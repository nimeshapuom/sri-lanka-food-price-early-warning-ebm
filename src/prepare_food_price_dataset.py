"""
Prepare the WFP Sri Lanka food price data for a next-month price-spike model.

Usage:  python prepare_food_price_dataset.py wfp_food_prices_lka.csv
Outputs (same folder as this script is run from):
  food_price_spike_dataset.csv      - labelled rows for training/testing (split column)
  food_price_spike_latest_month.csv - newest month, target unknown -> real inference samples
"""
import sys
import numpy as np
import pandas as pd

SRC = sys.argv[1] if len(sys.argv) > 1 else "wfp_food_prices_lka.csv"
START = "2022-10-01"     # dense, consistent coverage starts here (2023+ has ~1,200 series)
TEST_FROM = "2025-01-01"  # target months from here on are the test set
SPIKE_PCT = 5.0           # "spike" = price rises more than 5% next month

# ---------- 1. Load & clean ----------
raw = pd.read_csv(SRC, skiprows=[1])                     # row 1 is the HXL tag row (#date, #adm1...)
raw["date"] = pd.to_datetime(raw["date"]).dt.to_period("M").dt.to_timestamp()
raw = raw[(raw.pricetype == "Retail") & (raw.date >= START)]  # wholesale only exists 2020-21

# fill the few missing locations from other rows of the same market
for col in ["admin1", "admin2", "latitude", "longitude"]:
    raw[col] = raw.groupby("market")[col].transform(lambda s: s.ffill().bfill())

# one price per market x commodity x month
df = raw.groupby(["market", "commodity", "date"], as_index=False).agg(
    price=("price", "mean"), category=("category", "first"), unit=("unit", "first"),
    province=("admin1", "first"), district=("admin2", "first"),
    latitude=("latitude", "first"), longitude=("longitude", "first"))

# ---------- 2. Regular monthly grid (so "last month" really means last calendar month) ----------
months = pd.date_range(df.date.min(), df.date.max(), freq="MS")
parts = []
for (m, c), g in df.groupby(["market", "commodity"]):
    g = g.set_index("date").reindex(months)
    g["market"], g["commodity"] = m, c
    for col in ["category", "unit", "province", "district", "latitude", "longitude"]:
        g[col] = g[col].ffill().bfill()
    parts.append(g)
d = pd.concat(parts).rename_axis("date").reset_index().sort_values(["market", "commodity", "date"])
key = ["market", "commodity"]
p = d.groupby(key)["price"]

def pct(a, b):
    return (a / b - 1) * 100

# ---------- 3. Features (all use information available at month t) ----------
d["price_lag1"] = p.shift(1)
d["price_lag2"] = p.shift(2)
d["price_lag3"] = p.shift(3)
d["change_1m_pct"] = pct(d.price, d.price_lag1)
d["change_prev_1m_pct"] = pct(d.price_lag1, d.price_lag2)
d["change_prev_2m_pct"] = pct(d.price_lag2, d.price_lag3)
d["change_3m_pct"] = pct(d.price, d.price_lag3)
d["rolling3_mean"] = p.transform(lambda s: s.rolling(3, min_periods=2).mean())
d["price_vs_3m_avg_pct"] = pct(d.price, d.rolling3_mean)
d["volatility_3m"] = d.groupby(key)["change_1m_pct"].transform(lambda s: s.rolling(3, min_periods=2).std())

nat = d.groupby(["commodity", "date"]).agg(
    national_median_price=("price", "median"),
    national_change_1m_pct=("change_1m_pct", "median"),
    markets_reporting=("price", "count")).reset_index()
d = d.merge(nat, on=["commodity", "date"], how="left")
d["price_vs_national_pct"] = pct(d.price, d.national_median_price)

d["month"] = d.date.dt.month
d["year"] = d.date.dt.year
# Sri Lankan seasonal context
d["season"] = np.where(d.month.isin([10, 11, 12, 1, 2, 3]), "Maha", "Yala")
d["festive_month"] = d.month.isin([4, 12]).astype(int)   # Avurudu (Apr) and Christmas (Dec)
d["is_vegetable_fruit"] = (d.category == "vegetables and fruits").astype(int)

# ---------- 4. Targets ----------
d["next_price"] = d.groupby(key)["price"].shift(-1)
d["next_change_pct"] = pct(d.next_price, d.price)
d["spike_next_month"] = (d.next_change_pct > SPIKE_PCT).astype("Int64")
d["target_month"] = d.date + pd.offsets.MonthBegin(1)

# keep rows with an observed price this month and at least one month of history
d = d[d.price.notna() & d.change_1m_pct.notna()]

cols = ["date", "target_month", "province", "district", "market", "latitude", "longitude",
        "category", "commodity", "unit", "is_vegetable_fruit",
        "month", "year", "season", "festive_month",
        "price", "price_lag1", "price_lag2", "price_lag3",
        "change_1m_pct", "change_prev_1m_pct", "change_prev_2m_pct", "change_3m_pct",
        "price_vs_3m_avg_pct", "volatility_3m",
        "national_median_price", "national_change_1m_pct", "price_vs_national_pct", "markets_reporting",
        "next_price", "next_change_pct", "spike_next_month"]
num = d[cols].select_dtypes("number").columns
d[num] = d[num].round(3)

last = d.date.max()
labelled = d[d.next_price.notna()].copy()
labelled["split"] = np.where(labelled.target_month >= TEST_FROM, "test", "train")
latest = d[d.date == last].drop(columns=["next_price", "next_change_pct", "spike_next_month"])

labelled[cols + ["split"]].to_csv("food_price_spike_dataset.csv", index=False, date_format="%Y-%m-%d")
latest[[c for c in cols if c in latest]].to_csv("food_price_spike_latest_month.csv", index=False, date_format="%Y-%m-%d")

print(f"labelled rows: {len(labelled)}  (train {sum(labelled.split=='train')}, test {sum(labelled.split=='test')})")
print(f"spike rate: train {labelled[labelled.split=='train'].spike_next_month.mean():.3f}, "
      f"test {labelled[labelled.split=='test'].spike_next_month.mean():.3f}")
print(f"latest-month rows for inference: {len(latest)}  (features from {last:%Y-%m}, predicting {last + pd.offsets.MonthBegin(1):%Y-%m})")
print("missing values per column (labelled):")
print(labelled[cols].isna().sum()[lambda s: s > 0].to_string())
