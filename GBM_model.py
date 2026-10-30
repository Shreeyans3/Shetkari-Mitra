import os
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)

os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------
# Adjust path if your file is inside Data/mandi_prices_simplified.csv
DATA_PATH = "Data/mandi_prices_simplified.csv" if os.path.exists("Data/mandi_prices_simplified.csv") else "mandi_prices_simplified.csv"
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])

KEYS = ["district", "market", "commodity"]
PRICE = "modal_price_rs_per_quintal"
LAST = "last_known_modal_price_rs_per_quintal"
ARR = "arrival_quantity_tonnes"

df = df.sort_values(KEYS + ["date"]).reset_index(drop=True)
df["series"] = df[KEYS].astype(str).agg("|".join, axis=1)

# --------------------------------------------------
# 2. UNIFIED FEATURE ENGINEERING (Known at Origin Date)
# --------------------------------------------------
g = df.groupby(KEYS, sort=False)

for lag in [1, 7, 14, 30, 364]:
    df[f"price_lag_{lag}"] = g[LAST].shift(lag)
    df[f"mom_{lag}"] = df[LAST] / df[f"price_lag_{lag}"] - 1

for w in [7, 14, 30]:
    roll = g[LAST].transform(lambda s: s.rolling(w, min_periods=1).mean())
    df[f"price_rolling_{w}"] = roll
    df[f"last_vs_roll_{w}"] = df[LAST] / roll - 1
    df[f"price_std_{w}"] = g[LAST].transform(lambda s: s.rolling(w, min_periods=2).std())
    df[f"volatility_{w}"] = df[f"price_std_{w}"] / roll

df["price_max_30"] = g[LAST].transform(lambda s: s.rolling(30, min_periods=1).max())
df["price_min_30"] = g[LAST].transform(lambda s: s.rolling(30, min_periods=1).min())
df["pos_in_30d_range"] = (df[LAST] - df["price_min_30"]) / (
    df["price_max_30"] - df["price_min_30"] + 1e-9
)

# Arrivals
df["arr_lag_7"] = g[ARR].shift(7)
df["arr_roll_7"] = g[ARR].transform(lambda s: s.rolling(7, min_periods=1).mean())
df["arr_roll_30"] = g[ARR].transform(lambda s: s.rolling(30, min_periods=1).mean())
df["arr_vs_roll_30"] = df[ARR] / (df["arr_roll_30"] + 1e-9)

# Cross-market state-wide signal
cd = df.groupby(["commodity", "date"])
df["comm_mean_price"] = cd[LAST].transform("mean")
df["price_vs_comm_mean"] = df[LAST] / df["comm_mean_price"]
df["comm_mean_mom_7"] = cd["mom_7"].transform("mean")
df["comm_mean_mom_30"] = cd["mom_30"].transform("mean")

# Calendar at origin
df["day_of_week"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month
df["day_of_month"] = df["date"].dt.day
df["day_of_year"] = df["date"].dt.dayofyear

categorical_features = ["district", "market", "commodity", "series"]
for col in categorical_features:
    df[col] = df[col].astype("category")

numeric_base = [
    ARR, LAST, "days_since_last_report", "report_available",
    *[c for c in df.columns if c.startswith((
        "price_lag_", "mom_", "price_rolling_", "last_vs_roll_",
        "volatility_", "arr_", "comm_mean_", "price_vs_comm"))],
    "pos_in_30d_range",
    "day_of_week", "month", "day_of_month", "day_of_year",
]
numeric_base = list(dict.fromkeys(numeric_base))

# --------------------------------------------------
# 3. DATE WINDOWS & METRIC HELPERS
# --------------------------------------------------
TRAIN_START, TRAIN_END = pd.Timestamp("2024-01-01"), pd.Timestamp("2026-05-31")
VAL_START, VAL_END = pd.Timestamp("2026-06-01"), pd.Timestamp("2026-07-31")
TEST_START, TEST_END = pd.Timestamp("2026-08-01"), pd.Timestamp("2026-10-10")

def metrics(y, p):
    return (
        mean_absolute_error(y, p),
        np.sqrt(mean_squared_error(y, p)),
        mean_absolute_percentage_error(y, p) * 100,
    )

def sample_lgb_params(rng):
    return dict(
        objective=str(rng.choice(["regression", "regression_l1"])),
        learning_rate=float(rng.choice([0.03, 0.05, 0.08])),
        num_leaves=int(rng.choice([15, 31, 63])),
        max_depth=int(rng.choice([-1, 4, 6, 8])),
        min_child_samples=int(rng.choice([10, 20, 50, 100])),
        subsample=float(rng.choice([0.6, 0.7, 0.8, 0.9])),
        subsample_freq=1,
        colsample_bytree=float(rng.choice([0.5, 0.6, 0.8, 1.0])),
        reg_alpha=float(rng.choice([0.0, 0.1, 1.0, 5.0])),
        reg_lambda=float(rng.choice([1.0, 3.0, 10.0, 30.0])),
    )

# --------------------------------------------------
# 4. MULTI-HORIZON TRAINING WITH 10-TRIAL TUNING
# --------------------------------------------------
N_TRIALS = 10
results = []

for horizon in [5, 7, 14]:
    print(f"\n{'=' * 50}\nLIGHTGBM FORECAST HORIZON: {horizon} DAYS\n{'=' * 50}")

    work = df.copy()
    gw = work.groupby(KEYS, sort=False)
    work["target_price"] = gw[PRICE].shift(-horizon)
    work["target_date"] = gw["date"].shift(-horizon)
    work["target_report_available"] = gw["report_available"].shift(-horizon)

    gap_ok = (work["target_date"] - work["date"]).dt.days.eq(horizon)

    td = work["target_date"]
    work["target_month"] = td.dt.month
    work["target_day_of_week"] = td.dt.dayofweek
    work["target_sin"] = np.sin(2 * np.pi * td.dt.dayofyear / 365.25)
    work["target_cos"] = np.cos(2 * np.pi * td.dt.dayofyear / 365.25)

    numeric_features = numeric_base + [
        "target_month", "target_day_of_week", "target_sin", "target_cos"
    ]
    features = categorical_features + numeric_features

    observed = (
        work["target_report_available"].eq(1)
        & work["target_price"].notna()
        & gap_ok
        & work[LAST].gt(0)
    )

    def split(a, b):
        return work[(td >= a) & (td <= b) & observed].copy()

    train, val, test = split(TRAIN_START, TRAIN_END), split(VAL_START, VAL_END), split(TEST_START, TEST_END)
    print(f"Rows train/val/test: {len(train)} / {len(val)} / {len(test)}")

    # Target = log ratio of future price to current last known price
    def tgt(d):
        return np.log(d["target_price"] / d[LAST])

    def back(d, raw_pred):
        return d[LAST].to_numpy() * np.exp(raw_pred)

    Xtr, ytr = train[features], tgt(train)
    Xva, yva = val[features], tgt(val)

    # Naive Baseline on Validation
    b = metrics(val["target_price"], val[LAST])
    print(f"Naive baseline (val)  MAE {b[0]:.2f}  RMSE {b[1]:.2f}  MAPE {b[2]:.2f}%")

    # ---- 10-Trial Random Search on Validation Set ----
    rng = np.random.default_rng(42)
    best = None
    for i in range(N_TRIALS):
        p = sample_lgb_params(rng)
        m = lgb.LGBMRegressor(
            n_estimators=1200,
            random_state=42,
            n_jobs=-1,
            force_col_wise=True,
            verbose=-1,
            **p,
        )
        m.fit(
            Xtr, ytr,
            eval_set=[(Xva, yva)],
            eval_metric="mae",
            callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)],
        )
        pred = back(val, m.predict(Xva))
        mae = mean_absolute_error(val["target_price"], pred)
        if best is None or mae < best["mae"]:
            best = dict(mae=mae, params=p, n_est=m.best_iteration_, pred=pred)
        print(f"  trial {i + 1:02d}  val MAE {mae:.2f}  (best {best['mae']:.2f})")

    vm = metrics(val["target_price"], best["pred"])
    print(f"\nBest validation  MAE {vm[0]:.2f}  RMSE {vm[1]:.2f}  MAPE {vm[2]:.2f}%")
    print("Best params:", best["params"], "| trees:", best["n_est"])

    # ---- Final Fit on Train + Val (5-Seed Ensemble) ----
    final_train = pd.concat([train, val])
    Xf, yf = final_train[features], tgt(final_train)
    Xte = test[features]
    n_final = int(best["n_est"] * 1.15) + 1

    seed_models = []
    raw_preds = []
    for seed in [42, 7, 123, 2024, 99]:
        m = lgb.LGBMRegressor(
            n_estimators=n_final,
            random_state=seed,
            n_jobs=-1,
            force_col_wise=True,
            verbose=-1,
            **best["params"],
        )
        m.fit(Xf, yf)
        seed_models.append(m)
        raw_preds.append(m.predict(Xte))

    predictions = back(test, np.mean(raw_preds, axis=0))

    y_test = test["target_price"].to_numpy()
    base = metrics(y_test, test[LAST])
    mae, rmse, mape = metrics(y_test, predictions)
    print(f"\nNaive baseline (test)  MAE {base[0]:.2f}  RMSE {base[1]:.2f}  MAPE {base[2]:.2f}%")
    print(f"Final LightGBM test    MAE {mae:.2f}  RMSE {rmse:.2f}  MAPE {mape:.2f}%")

    # Save predictions CSV
    out = test[["target_date", "district", "market", "commodity"]].copy()
    out["actual_price"] = y_test
    out["predicted_price"] = predictions
    out["forecast_horizon_days"] = horizon
    out.to_csv(f"reports/lightgbm_{horizon}_day_test_predictions.csv", index=False)

    # Save trained ensemble + metadata for live project inference!
    joblib.dump({
        "models": seed_models,
        "features": features,
        "categorical_features": categorical_features,
        "categories": {col: df[col].cat.categories for col in categorical_features},
        "horizon": horizon
    }, f"models/lgb_bundle_h{horizon}.joblib")

    results.append({
        "horizon_days": horizon,
        "train_rows": len(train), "validation_rows": len(val), "test_rows": len(test),
        "baseline_test_MAE": base[0], "baseline_test_MAPE_percent": base[2],
        "test_MAE": mae, "test_RMSE": rmse, "test_MAPE_percent": mape,
    })

comparison = pd.DataFrame(results)
print("\nFINAL LIGHTGBM MODEL COMPARISON")
print(comparison.to_string(index=False))
comparison.to_csv("reports/lightgbm_model_comparison.csv", index=False)