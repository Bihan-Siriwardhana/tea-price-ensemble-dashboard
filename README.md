# Colombo Tea Auction: Ensemble Learning Dashboard

Dataset: `hesandism/colombo-tea-auction-prices` (Hugging Face), Sri Lankan tea auction prices with USD/LKR rate and weather severity.

## Folder layout
```
Tea/
├── app.py
├── colombo_tea_auction.csv
├── download_dataset.py
├── README.md
├── requirements.txt
├── Tea_Ensemble_Learning.ipynb
└── models/                 
    ├── best_model.joblib
    ├── preprocessor.joblib
    └── model_info.json
```

## How to run
1. Install packages (once):
   `pip install pandas numpy scikit-learn xgboost lightgbm catboost statsmodels matplotlib seaborn joblib plotly streamlit jupyter ipykernel shap`
2. Download data (once): `python download_dataset.py`
3. Open `Tea_Ensemble_Learning.ipynb` in VS Code, select the Python kernel, click **Run All**.
4. Dashboard: `streamlit run app.py`
   

## What is inside
- Bagging (Bagging, Random Forest), Boosting (Gradient Boosting, XGBoost, LightGBM, CatBoost), Stacking (RF+XGB+LGBM with Ridge)
- Random Forest vs XGBoost with tuning and feature importance
- Hybrids: RF + ARIMA (time series) and RF + ANN (tabular regression)
- Time-based train/test split (oldest 80% train, newest 20% test) to avoid leakage
- Innovation: "Fair-Price Meter" (price range plus overpriced flag) and "Best time to sell" advice

## Host free (optional)
Upload `app.py`, `requirements.txt` and `colombo_tea_auction.csv` to a GitHub repo, then deploy at share.streamlit.io.


