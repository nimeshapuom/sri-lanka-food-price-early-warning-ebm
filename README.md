# Early Warning of Food Price Spikes in Sri Lankan Markets - Explainable Boosting Machine

**MSc in AI · Machine Learning Assignment** · Prasadini A P N (269170A)

Predicts whether the **retail price of a food item in a Sri Lankan market will rise by more than 5 % next month**, using WFP market
price data (Oct 2022 – Sep 2025; 14,239 rows, 21 commodities, 38 markets, 9 provinces). The main model is an algorithm not covered
in the lectures — the **Explainable Boosting Machine (EBM)**, a glass-box generalised additive model with pairwise interactions
(InterpretML). It is compared with rule baselines, standard models (logistic regression, decision tree, k-NN) and lecture ensembles
(Random Forest, XGBoost, CatBoost), explained with EBM shape functions, feature importance, PDP and LIME, and served in a Streamlit app.

## Results (unseen test period Jan–Sep 2025)

| Model | Type | Val AUC | Test ROC-AUC | PR-AUC | Precision | Recall |
|---|---|---|---|---|---|---|
| **EBM (new algorithm)** | glass-box | 0.725 | **0.711** | 0.380 | **0.403** | 0.475 |
| Logistic Regression | standard | 0.700 | 0.697 | 0.365 | 0.353 | 0.549 |
| Decision Tree | standard | 0.686 | 0.624 | 0.285 | 0.275 | 0.404 |
| k-NN | standard | 0.657 | 0.654 | 0.296 | 0.315 | 0.568 |
| Random Forest | lecture ensemble | 0.718 | 0.732 | 0.381 | 0.383 | 0.651 |
| XGBoost | lecture ensemble | 0.729 | 0.741 | 0.397 | 0.357 | 0.743 |
| CatBoost | lecture ensemble | 0.740 | 0.736 | 0.383 | 0.372 | 0.676 |
| Rule: national trend > 5 % | baseline | – | 0.578 | 0.299 | 0.343 | 0.281 |
| Rule: momentum | baseline | – | 0.500 | 0.244 | 0.277 | 0.298 |

## Project structure

```
report/Food_Price_Spike_EBM_Report_269170A.pdf   written report
data/wfp_food_prices_lka.csv                      raw WFP data (HDX, public)
data/food_price_spike_dataset.csv                 engineered, labelled dataset (train/test split column)
data/food_price_spike_latest_month.csv            Sep 2025 rows without labels -> new samples for inference
data_dictionary.md                                column descriptions
src/prepare_food_price_dataset.py                 cleaning & feature engineering
src/analysis_pipeline.py                          training, evaluation & explainability (script export of the notebook)
notebooks/food_price_spike_analysis.ipynb         full executed analysis (also as .html)
models/spike_model.pkl                            final EBM + feature list + threshold
models/model_meta.json, models/model_comparison.csv
app/app.py                                        Streamlit front end
figures/                                          all charts
```

## How to run

```bash
pip install -r requirements.txt
cd data && python ../src/prepare_food_price_dataset.py wfp_food_prices_lka.csv && cd ..   # optional: rebuild dataset
jupyter notebook notebooks/food_price_spike_analysis.ipynb                                 # ~15 min
streamlit run app/app.py
```

The app has four tabs: **Check one item** (pre-filled latest prices, editable what-if inputs, spike probability, exact EBM
contributions), **Market scan** (risk ranking of all market–commodity pairs, filters), **How the model thinks** (term importance
and learned shape functions) and **About** (data, model, metrics, comparison).

## Data source & ethics

World Food Programme — *Sri Lanka: Food Prices*, Humanitarian Data Exchange (HDX). Aggregated, openly published market prices;
no personal or sensitive data.
