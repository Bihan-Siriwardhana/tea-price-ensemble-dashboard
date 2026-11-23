"""Ensemble Learning Dashboard: Colombo Tea Auction Prices.
Run:  streamlit run app.py     (needs colombo_tea_auction.csv in the same folder)
"""
import time, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd, streamlit as st
import plotly.express as px, plotly.graph_objects as go
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import (AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor, GradientBoostingRegressor,
                              RandomForestRegressor, StackingRegressor, VotingRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

CSV_PATH, RS = "colombo_tea_auction.csv", 42
st.set_page_config(page_title="Tea Ensemble Dashboard", page_icon="🍃", layout="wide")


def build_date(d):
    for c in ["sale_date", "date", "auction_date"]:
        if c in d.columns:
            x = pd.to_datetime(d[c], errors="coerce")
            if x.notna().mean() > 0.9:
                return x
    if "sale_date_raw" in d.columns:
        s = d["sale_date_raw"].astype(str)
        x = pd.to_datetime(s.str.extract(r"(\d{1,2})")[0] + " " + s.str.extract(r"([A-Za-z]{3,9})\s+\d{4}")[0]
                           + " " + s.str.extract(r"(\d{4})")[0], errors="coerce")
        if x.notna().mean() > 0.9:
            return x
    m = d["sale_id"].str.extract(r"(\d{4})_(\d+)")
    return pd.to_datetime(m[0] + "-01-01") + pd.to_timedelta((m[1].astype(int) - 1) * 7, unit="D")


@st.cache_data(show_spinner="Loading and preparing data...")
def load_data():
    df = pd.read_csv(CSV_PATH)
    cands = [c for c in df.columns if "price" in c.lower()]
    pref = [c for c in ["price", "price_lkr", "avg_price", "price_per_kg", "price_lkr_per_kg"] if c in df.columns]
    T = (pref or [c for c in cands if "lkr" in c.lower()] or cands)[0]
    df["date"] = build_date(df)
    df[T] = pd.to_numeric(df[T], errors="coerce")
    df = df.dropna(subset=[T, "date"])
    df = df[df[T] > 0].sort_values("date").reset_index(drop=True)
    df["year"], df["month"] = df.date.dt.year, df.date.dt.month
    df["week"] = df.date.dt.isocalendar().week.astype(int)
    keys = [k for k in ["catalogue", "category"] if k in df.columns]
    g = df.groupby(keys + ["date"])[T].mean().reset_index().sort_values(keys + ["date"])
    g["lag1"] = g.groupby(keys)[T].shift(1)
    g["roll4"] = g.groupby(keys)[T].transform(lambda s: s.shift(1).rolling(4, min_periods=1).mean())
    df = df.merge(g[keys + ["date", "lag1", "roll4"]], on=keys + ["date"], how="left")
    df = df.dropna(subset=["lag1"]).sort_values("date").reset_index(drop=True)
    drop = [T, "date", "sale_id", "sale_date_raw"]
    price_like = [c for c in df.columns if "price" in c.lower() and c != T]
    bools = df.select_dtypes(include="bool").columns.tolist()
    df[bools] = df[bools].astype(str)
    cat = [c for c in df.select_dtypes(include=["object", "category"]).columns
           if c not in drop + price_like and df[c].nunique() <= 60]
    num = [c for c in df.select_dtypes(include="number").columns
           if c not in drop + price_like and df[c].notna().any()]
    return df, T, cat, num


df, T, cat_cols, num_cols = load_data()
feat = num_cols + cat_cols
dates = np.sort(df.date.unique())
cutoff = dates[int(len(dates) * 0.8)]
train, test = df[df.date < cutoff], df[df.date >= cutoff]


def make_pre():
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)])


def zoo(fast):
    k = 0.5 if fast else 1.0
    n = lambda v: max(50, int(v * k))
    base = [("rf", RandomForestRegressor(n(120), n_jobs=-1, random_state=RS)),
            ("xgb", XGBRegressor(n_estimators=n(250), learning_rate=.05, n_jobs=-1, random_state=RS)),
            ("lgbm", LGBMRegressor(n_estimators=n(250), learning_rate=.05, verbose=-1, random_state=RS))]
    return {
        "Decision Tree": ("Baseline", DecisionTreeRegressor(max_depth=12, random_state=RS)),
        "Bagged Trees": ("Bagging", BaggingRegressor(DecisionTreeRegressor(), n_estimators=n(80), n_jobs=-1, random_state=RS)),
        "Random Forest": ("Bagging", RandomForestRegressor(n_estimators=n(250), n_jobs=-1, random_state=RS)),
        "Extra Trees": ("Bagging", ExtraTreesRegressor(n_estimators=n(250), n_jobs=-1, random_state=RS)),
        "AdaBoost": ("Boosting", AdaBoostRegressor(DecisionTreeRegressor(max_depth=6), n_estimators=n(100), learning_rate=.5, random_state=RS)),
        "Gradient Boosting": ("Boosting", GradientBoostingRegressor(n_estimators=n(200), random_state=RS)),
        "XGBoost": ("Boosting", XGBRegressor(n_estimators=n(400), learning_rate=.05, max_depth=6, n_jobs=-1, random_state=RS)),
        "LightGBM": ("Boosting", LGBMRegressor(n_estimators=n(400), learning_rate=.05, verbose=-1, random_state=RS)),
        "Voting (avg)": ("Stacking", VotingRegressor(base)),
        "Stacking (RF+XGB+LGBM→Ridge)": ("Stacking", StackingRegressor(base, final_estimator=Ridge(), cv=KFold(3))),
    }


@st.cache_resource(show_spinner="Training ensemble models... (first load only, about 1-2 minutes)")
def train_all(fast):
    pre = make_pre()
    Xtr, Xte = pre.fit_transform(train[feat]), pre.transform(test[feat])
    ytr, yte = train[T].values, test[T].values
    fitted, rows = {}, []

    def score(name, fam, p, sec=np.nan):
        rows.append(dict(Model=name, Family=fam, RMSE=float(np.sqrt(mean_squared_error(yte, p))),
                         MAE=mean_absolute_error(yte, p), R2=r2_score(yte, p), FitTime_s=sec))

    for name, (fam, est) in zoo(fast).items():
        t = time.time(); est.fit(Xtr, ytr); sec = round(time.time() - t, 2)
        fitted[name] = est; score(name, fam, est.predict(Xte), sec)
    ann = TransformedTargetRegressor(
        regressor=MLPRegressor(hidden_layer_sizes=(128, 64, 32), early_stopping=True, max_iter=600, random_state=RS),
        transformer=StandardScaler()).fit(Xtr, ytr)
    fitted["ANN (MLP)"] = ann; score("ANN (MLP)", "ANN", ann.predict(Xte))
    score("Hybrid RF+ANN (60/40)", "Hybrid", 0.6 * fitted["Random Forest"].predict(Xte) + 0.4 * ann.predict(Xte))
    results = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    return pre, fitted, results, Xte, yte


@st.cache_data(show_spinner="Fitting ARIMA + RF hybrid...")
def ts_hybrid():
    FX = "fx_usd" if "fx_usd" in df.columns else None
    WX = "avg_weather_severity" if "avg_weather_severity" in df.columns else None
    aggs = {"price": (T, "mean")}
    if FX: aggs["fx"] = (FX, "mean")
    if WX: aggs["wx"] = (WX, "mean")
    ts = df.groupby("date").agg(**aggs).reset_index().sort_values("date")
    ts["month"], ts["week"] = ts.date.dt.month, ts.date.dt.isocalendar().week.astype(int)
    f = ["month", "week"] + [c for c in ["fx", "wx"] if c in ts.columns]
    tr, te = ts[ts.date < cutoff].reset_index(drop=True), ts[ts.date >= cutoff].reset_index(drop=True)
    ar = ARIMA(tr.price.values, order=(1, 1, 1)).fit()
    ar_pred = np.asarray(ar.append(te.price.values, refit=False).predict(start=len(tr), end=len(tr) + len(te) - 1))
    rfr = RandomForestRegressor(300, min_samples_leaf=3, random_state=RS, n_jobs=-1).fit(tr[f].iloc[1:], np.asarray(ar.resid)[1:])
    hyb = ar_pred + rfr.predict(te[f])
    rf_only = RandomForestRegressor(300, min_samples_leaf=3, random_state=RS, n_jobs=-1).fit(tr[f], tr.price).predict(te[f])
    y = te.price.values
    mm = lambda q: dict(RMSE=float(np.sqrt(mean_squared_error(y, q))), MAE=mean_absolute_error(y, q), R2=r2_score(y, q))
    scores = pd.DataFrame({"ARIMA": mm(ar_pred), "RF (calendar+FX+weather)": mm(rf_only), "Hybrid ARIMA+RF": mm(hyb)}).T
    return te.date, y, ar_pred, hyb, scores


# ------------------------------------------------------------------ UI
st.title("🍃 Colombo Tea Auction prices ")  # shown above every page
st.caption("Bagging · Boosting · Stacking · RF vs XGBoost · RF+ANN · RF+ARIMA  |  Colombo Tea Auction prices (Sri Lanka)")
fast = st.sidebar.toggle("Fast mode (fewer trees)", True)
pre, models, res, Xte, yte = train_all(fast)
st.sidebar.markdown("<p style='letter-spacing:.15em;font-size:.75rem;font-weight:700;opacity:.6;margin:1.2rem 0 .2rem'>WORKSPACE</p>", unsafe_allow_html=True)
page = st.sidebar.radio("Navigation", ["Overview", "Model Comparison", "Feature Analysis", "Prediction", "Hybrid Forecast"],
                        label_visibility="collapsed")
st.header(page)

if page == "Overview":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records", f"{len(df):,}"); c2.metric("Median price", f"Rs. {df[T].median():,.0f}")
    c3.metric("From", str(df.date.min().date())); c4.metric("To", str(df.date.max().date()))
    a, b = st.columns(2)
    monthly = df.groupby(df.date.dt.to_period("M"))[T].mean()
    monthly.index = monthly.index.to_timestamp()
    a.plotly_chart(px.line(monthly, title="Average price over time", labels=dict(value=T, date="")).update_layout(showlegend=False), width="stretch")
    if "catalogue" in df.columns:
        b.plotly_chart(px.box(df, x="catalogue", y=T, color="catalogue", title="Price by elevation (catalogue)"), width="stretch")
    if "category" in df.columns:
        top = df.category.value_counts().index[:12]
        st.plotly_chart(px.box(df[df.category.isin(top)], x="category", y=T, title="Price by tea category (12 most common)"), width="stretch")
    st.dataframe(df[["date", T] + feat].head(50), width="stretch")

elif page == "Model Comparison":
    fam = st.multiselect("Families", res.Family.unique().tolist(), res.Family.unique().tolist())
    r = res[res.Family.isin(fam)]
    st.plotly_chart(px.bar(r, x="RMSE", y="Model", color="Family", orientation="h",
                           title="Test RMSE (Rs/kg), lower is better").update_yaxes(autorange="reversed"), width="stretch")
    st.dataframe(r.round(3), width="stretch")
    best = res.iloc[0]
    st.success(f"Lowest RMSE: **{best.Model}** (RMSE {best.RMSE:,.1f}, R² {best.R2:.3f})")
    st.markdown("**Bagging** cuts variance · **Boosting** cuts bias · **Stacking** learns who to trust · **Hybrid** combines different inductive biases.")

elif page == "Feature Analysis":
    st.caption("Random Forest vs XGBoost: accuracy, actual vs predicted and the features that drive the price.")
    names = ["Random Forest", "XGBoost"]
    cols = st.columns(2)
    fn = pre.get_feature_names_out()
    for col, n in zip(cols, names):
        p = models[n].predict(Xte); row = res[res.Model == n].iloc[0]
        col.subheader(n)
        m1, m2 = col.columns(2); m1.metric("RMSE (Rs/kg)", f"{row.RMSE:,.1f}"); m2.metric("R²", f"{row.R2:.3f}")
        fig = px.scatter(x=yte, y=p, opacity=.4, labels=dict(x="Actual", y="Predicted"))
        lim = [float(min(yte.min(), p.min())), float(max(yte.max(), p.max()))]
        fig.add_shape(type="line", x0=lim[0], y0=lim[0], x1=lim[1], y1=lim[1], line=dict(dash="dash"))
        col.plotly_chart(fig, width="stretch")
        imp = pd.Series(models[n].feature_importances_, index=[s.split("__")[-1] for s in fn]).sort_values().tail(10)
        col.plotly_chart(px.bar(imp, orientation="h", title=f"{n}: top features").update_layout(showlegend=False), width="stretch")

elif page == "Prediction":
    st.subheader("What should this tea sell for?")
    st.caption("Innovation idea: a price meter for exporters and growers. It gives a price range from several ensemble models and flags asking prices that look too high.")
    med = {c: float(df[c].median()) for c in num_cols}
    mode = {c: str(df[c].mode()[0]) for c in cat_cols}
    row = {**med, **mode}
    c = st.columns(4)
    for i, col in enumerate(["catalogue", "category"]):
        if col in cat_cols:
            row[col] = c[i].selectbox(col.title(), sorted(df[col].unique()))
    row["month"] = c[2].slider("Month", 1, 12, 6)
    row["lag1"] = c[3].number_input("Previous sale price for this tea", value=float(med.get("lag1", 0)))
    c2 = st.columns(3)
    if "fx_usd" in med:
        row["fx_usd"] = c2[0].number_input("USD/LKR rate", value=med["fx_usd"])
    if "avg_weather_severity" in med:
        row["avg_weather_severity"] = c2[1].slider("Weather severity", 0.0, 10.0, med["avg_weather_severity"])
    ask = c2[2].number_input("Asking price (Rs/kg), optional", 0.0, 1e7, 0.0)
    row.update({"week": int(row["month"] * 4.345), "year": int(df.year.max()), "roll4": row["lag1"]})
    X1 = pre.transform(pd.DataFrame([row])[feat])
    use = ["Random Forest", "XGBoost", "LightGBM", "Gradient Boosting", "Stacking (RF+XGB+LGBM→Ridge)"]
    preds = {n: float(models[n].predict(X1)[0]) for n in use}
    est = preds["Stacking (RF+XGB+LGBM→Ridge)"]; lo, hi = min(preds.values()), max(preds.values())
    m1, m2 = st.columns(2); m1.metric("Fair price (stacked)", f"Rs. {est:,.1f}"); m2.metric("Ensemble range", f"{lo:,.1f} to {hi:,.1f}")
    if ask > 0:
        d = (ask - est) / est * 100
        (st.error if d > 20 else st.warning if d > 8 else st.success)(
            f"Asking price is {d:+.0f}% vs fair price → " + ("🚩 OVERPRICED" if d > 20 else "slightly high" if d > 8 else "fair / good deal"))
    st.plotly_chart(px.bar(x=list(preds), y=list(preds.values()), labels=dict(x="Model", y="Predicted Rs/kg"),
                           title="Each ensemble's opinion"), width="stretch")
    st.markdown("---")
    st.subheader("Best time to sell")
    d = df
    if "catalogue" in d.columns: d = d[d.catalogue == row["catalogue"]]
    if "category" in d.columns: d = d[d.category == row["category"]]
    if len(d):
        bm = d.groupby(d.date.dt.month)[T].mean()
        st.plotly_chart(px.bar(bm, labels=dict(value="Avg price", date="Month"), title="Average price by month").update_layout(showlegend=False), width="stretch")
        st.info(f"Historically the best month for this tea is **month {int(bm.idxmax())}** (avg {bm.max():,.0f}); the weakest is month {int(bm.idxmin())}.")

elif page == "Hybrid Forecast":
    dts, y, ar, hyb, sc = ts_hybrid()
    f = go.Figure()
    f.add_scatter(x=dts, y=y, name="Actual", line=dict(color="black"))
    f.add_scatter(x=dts, y=ar, name="ARIMA")
    f.add_scatter(x=dts, y=hyb, name="Hybrid ARIMA+RF", line=dict(color="red", width=3))
    f.update_layout(title="Average sale price: one-step-ahead forecasts on the test period", yaxis_title="Rs/kg")
    st.plotly_chart(f, width="stretch")
    st.dataframe(sc.round(3), width="stretch")
    st.caption("ARIMA = linear trend and autocorrelation · RF = non-linear correction of the ARIMA residuals using calendar, FX and weather.")