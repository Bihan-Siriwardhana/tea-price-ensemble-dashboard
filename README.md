# Colombo Tea Auction: Ensemble Learning Dashboard

Dataset: `hesandism/colombo-tea-auction-prices` (Hugging Face), Sri Lankan tea auction prices with USD/LKR rate and weather severity.

## Folder layout
```
Tea/
├── download_dataset.py
├── colombo_tea_auction.csv
├── Tea_Ensemble_Learning.ipynb   (submit this)
├── app.py                        (dashboard)
├── requirements.txt
└── README.md
```

## How to run
1. Install packages (once):
   `pip install pandas numpy scikit-learn xgboost lightgbm catboost statsmodels matplotlib seaborn joblib plotly streamlit jupyter ipykernel shap`
2. Download data (once): `python download_dataset.py`
3. Open `Tea_Ensemble_Learning.ipynb` in VS Code, select the Python kernel, click **Run All**.
4. Dashboard: `streamlit run app.py`
   (the first load trains the models, about 1-2 minutes; after that it is instant)

## What is inside
- Bagging (Bagging, Random Forest), Boosting (Gradient Boosting, XGBoost, LightGBM, CatBoost), Stacking (RF+XGB+LGBM with Ridge)
- Random Forest vs XGBoost with tuning and feature importance
- Hybrids: RF + ARIMA (time series) and RF + ANN (tabular regression)
- Time-based train/test split (oldest 80% train, newest 20% test) to avoid leakage
- Innovation: "Fair-Price Meter" (price range plus overpriced flag) and "Best time to sell" advice

## Host free (optional)
Upload `app.py`, `requirements.txt` and `colombo_tea_auction.csv` to a GitHub repo, then deploy at share.streamlit.io.

## 3-minute video script (voice-over is enough)
- 0:00-0:30  Problem: tea is a key Sri Lankan export and auction prices swing with the exchange rate and weather.
- 0:30-1:30  Dashboard demo: the 5 tabs (Data, Model comparison, RF vs XGBoost, Fair-Price Meter, RF+ARIMA).
- 1:30-2:30  Results: bagging vs boosting vs stacking, RF vs XGBoost winner and why, hybrid findings.
- 2:30-3:00  Innovation: Fair-Price Meter for exporters and small growers. State that AI (Claude) helped generate the code.

## Submission checklist (deadline Monday 23rd, midnight)
- [ ] `.ipynb` run top to bottom with outputs visible, interpretation cells filled in
- [ ] AI-usage statement (already in the notebook)
- [ ] Video of 3 minutes or less
- [ ] Dashboard link (optional)
