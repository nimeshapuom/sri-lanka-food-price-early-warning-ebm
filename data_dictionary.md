# Sri Lanka Food Price Spike Dataset — Data Dictionary

**Source:** WFP Global Food Prices — Sri Lanka (HDX), retail prices in LKR.
**Built by:** `prepare_food_price_dataset.py` (run: `python prepare_food_price_dataset.py wfp_food_prices_lka.csv`).

**Problem:** Will the retail price of a commodity in a given Sri Lankan market rise by **more than 5% next month**? (binary classification — an early-warning signal for food price spikes)

## Files

| File | Rows | Purpose |
|---|---|---|
| `food_price_spike_dataset.csv` | 24,489 | Labelled rows. `split` = `train` (target months Dec 2022–Dec 2024, 16,362 rows) or `test` (Jan–Sep 2025, 8,127 rows). |
| `food_price_spike_latest_month.csv` | 1,000 | Sep 2025 features with **no label** — genuinely new data points for single-sample inference (predicting Oct 2025). |

## Columns

| Column | Type | Description |
|---|---|---|
| `date` | date | Month the features describe (t). |
| `target_month` | date | Month being predicted (t + 1). |
| `province`, `district`, `market` | categorical | Location (9 provinces, 44 markets). |
| `latitude`, `longitude` | numeric | Market coordinates. |
| `category` | categorical | WFP food group (cereals and tubers, pulses and nuts, meat/fish/eggs, vegetables and fruits, …). |
| `commodity` | categorical | Item, e.g. Rice (white), Dhal, Coconut. |
| `unit` | categorical | KG, Unit, L, … |
| `is_vegetable_fruit` | 0/1 | 1 for vegetables & fruits — filter on this to compare with/without these weather-driven items. |
| `month`, `year` | int | Calendar month/year of `date`. |
| `season` | categorical | Maha (Oct–Mar) or Yala (Apr–Sep) cultivation season. |
| `festive_month` | 0/1 | April (Sinhala & Tamil New Year) or December. |
| `price` | numeric | Retail price this month (LKR). |
| `price_lag1/2/3` | numeric | Price 1, 2, 3 months earlier (blank if not reported). |
| `change_1m_pct` | numeric | % change from last month to this month. |
| `change_prev_1m_pct`, `change_prev_2m_pct` | numeric | The two monthly % changes before that. |
| `change_3m_pct` | numeric | % change over the last 3 months. |
| `price_vs_3m_avg_pct` | numeric | How far this month's price is above/below its 3-month average. |
| `volatility_3m` | numeric | Std. dev. of recent monthly % changes. |
| `national_median_price` | numeric | Median price of this commodity across all markets this month. |
| `national_change_1m_pct` | numeric | Median 1-month % change of this commodity across all markets. |
| `price_vs_national_pct` | numeric | How much dearer/cheaper this market is than the national median. |
| `markets_reporting` | int | Number of markets reporting this commodity this month. |
| `next_price` | numeric | **Leakage — do not use as a feature.** Next month's price. |
| `next_change_pct` | numeric | **Leakage — do not use as a feature.** Next month's % change (regression target). |
| `spike_next_month` | 0/1 | **Target.** 1 if `next_change_pct` > 5%. ~30% of train rows, ~27% of test rows. |
| `split` | text | `train` / `test` (time-based — never shuffle). |

## Notes for modelling

- **Missing values are intentional.** Lag and change columns are blank where a market did not report in an earlier month (`price_lag3` ~14% blank). Handle them in your preprocessing step (imputation or native NaN handling in XGBoost/LightGBM/CatBoost) and discuss the choice.
- **Do not use** `date`, `target_month`, `next_price`, `next_change_pct` as features.
- **Scaling:** tree models don't need it, but apply it inside a pipeline if you add a distance- or gradient-based model to the comparison.
- **Class imbalance:** use `class_weight` / `scale_pos_weight` / `auto_class_weights`, and report AUC, precision, recall and F1, not just accuracy.
- **Sanity check (CatBoost, test set):** AUC ≈ 0.64 on all items, ≈ 0.74 without vegetables & fruits. Baseline rule "last month rose >5%" ≈ 0.53.
