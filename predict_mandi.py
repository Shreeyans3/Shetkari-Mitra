import os
import json
import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------
# 1. PREPARE LATEST FEATURES FROM DATASET
# ---------------------------------------------------------
def load_and_prepare_features():
    data_path = (
        "Data/mandi_prices_simplified.csv"
        if os.path.exists("Data/mandi_prices_simplified.csv")
        else "mandi_prices_simplified.csv"
    )
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])

    KEYS = ["district", "market", "commodity"]
    LAST = "last_known_modal_price_rs_per_quintal"
    ARR = "arrival_quantity_tonnes"

    df = df.sort_values(KEYS + ["date"]).reset_index(drop=True)
    df["series"] = df[KEYS].astype(str).agg("|".join, axis=1)

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

    df["arr_lag_7"] = g[ARR].shift(7)
    df["arr_roll_7"] = g[ARR].transform(lambda s: s.rolling(7, min_periods=1).mean())
    df["arr_roll_30"] = g[ARR].transform(lambda s: s.rolling(30, min_periods=1).mean())
    df["arr_vs_roll_30"] = df[ARR] / (df["arr_roll_30"] + 1e-9)

    cd = df.groupby(["commodity", "date"])
    df["comm_mean_price"] = cd[LAST].transform("mean")
    df["price_vs_comm_mean"] = df[LAST] / df["comm_mean_price"]
    df["comm_mean_mom_7"] = cd["mom_7"].transform("mean")
    df["comm_mean_mom_30"] = cd["mom_30"].transform("mean")

    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["day_of_month"] = df["date"].dt.day
    df["day_of_year"] = df["date"].dt.dayofyear

    return df


# ---------------------------------------------------------
# 2. PREDICT FUTURE PRICES FOR SELECTED MANDI & COMMODITY
# ---------------------------------------------------------
def predict_for_farmer(district: str, market: str, commodity: str, df_features: pd.DataFrame):
    sub = df_features[
        (df_features["district"] == district)
        & (df_features["market"] == market)
        & (df_features["commodity"] == commodity)
    ].sort_values("date")

    if sub.empty:
        return {"error": f"No data found for {commodity} in {market} ({district})."}

    latest_row = sub.iloc[[-1]].copy()
    origin_date = latest_row["date"].iloc[0]
    current_price = float(latest_row["last_known_modal_price_rs_per_quintal"].iloc[0])
    staleness_days = int(latest_row["days_since_last_report"].iloc[0])

    forecasts = {}
    for horizon in [5, 7, 14]:
        model_path = f"models/lgb_bundle_h{horizon}.joblib"
        if not os.path.exists(model_path):
            return {"error": f"Model file {model_path} not found. Run GBM_model.py first."}

        bundle = joblib.load(model_path)
        target_date = origin_date + pd.Timedelta(days=horizon)

        row_h = latest_row.copy()
        row_h["target_month"] = target_date.month
        row_h["target_day_of_week"] = target_date.dayofweek
        row_h["target_sin"] = np.sin(2 * np.pi * target_date.dayofyear / 365.25)
        row_h["target_cos"] = np.cos(2 * np.pi * target_date.dayofyear / 365.25)

        for col in bundle["categorical_features"]:
            row_h[col] = pd.Categorical(row_h[col], categories=bundle["categories"][col])

        X_input = row_h[bundle["features"]]

        # Predict using the 5-seed LightGBM ensemble
        raw_log_preds = [m.predict(X_input)[0] for m in bundle["models"]]
        pred_price = current_price * float(np.exp(np.mean(raw_log_preds)))

        change_rs = pred_price - current_price
        change_pct = (change_rs / current_price) * 100

        forecasts[f"day_{horizon}"] = {
            "target_date": str(target_date.date()),
            "predicted_price_rs_quintal": round(pred_price, 2),
            "expected_change_rs": round(change_rs, 2),
            "expected_change_pct": round(change_pct, 2),
        }

    # Profit-Aware Advisory Logic (SELL NOW vs WAIT)
    high_perishables = [
        "Tomato", "Bhindi(Ladies Finger)", "Brinjal",
        "Cabbage", "Cauliflower", "Green Chilli"
    ]
    threshold_pct = 8.0 if commodity in high_perishables else 5.0

    pct_7d = forecasts["day_7"]["expected_change_pct"]
    if pct_7d >= threshold_pct:
        decision = "WAIT (HOLD)"
        reason = (
            f"7-day price is projected to rise by +{pct_7d:.2f}%, which beats the "
            f"+{threshold_pct}% storage & spoilage risk threshold for {commodity}."
        )
    else:
        decision = "SELL NOW"
        reason = (
            f"7-day projected change ({pct_7d:+.2f}%) does not beat the +{threshold_pct}% "
            f"perishability & holding cost threshold for {commodity}."
        )

    return {
        "district": district,
        "market": market,
        "commodity": commodity,
        "last_recorded_date": str(origin_date.date()),
        "current_price_rs_quintal": round(current_price, 2),
        "data_staleness_days": staleness_days,
        "staleness_warning": "WARNING: Market report is older than 5 days" if staleness_days > 5 else "Data is fresh",
        "forecasts": forecasts,
        "advisory": {
            "decision": decision,
            "reason": reason,
        },
    }


# ---------------------------------------------------------
# 3. MAIN EXECUTION BLOCK (THIS PRINTS OUTPUT IN TERMINAL!)
# ---------------------------------------------------------
if __name__ == "__main__":
    print("Loading mandi dataset and computing latest market features...")
    df_feat = load_and_prepare_features()

    # Test Case 1: Pune(Pimpri) - Tomato
    sample_queries = [
        ("Pune", "Pune(Pimpri)", "Tomato"),
        ("Ahilyanagar", "APMC Sangamner", "Onion"),
        ("Kolhapur", "APMC Kolhapur", "Green Chilli"),
    ]

    for dist, mkt, comm in sample_queries:
        result = predict_for_farmer(dist, mkt, comm, df_feat)
        print("\n" + "=" * 65)
        print(f" HYPERLOCAL MANDI ADVISORY: {comm.upper()} @ {mkt} ({dist})")
        print("=" * 65)
        if "error" in result:
            print("Error:", result["error"])
            continue

        print(f"Last Recorded Date : {result['last_recorded_date']} ({result['staleness_warning']})")
        print(f"Current Modal Price: ₹{result['current_price_rs_quintal']} / quintal")
        print("-" * 65)
        for h_key, f_val in result["forecasts"].items():
            horizon_label = h_key.replace("day_", "t+") + " Days"
            print(
                f"  {horizon_label:<10} ({f_val['target_date']}) -> "
                f"Predicted: ₹{f_val['predicted_price_rs_quintal']:>7.2f}/q  "
                f"({f_val['expected_change_pct']:+.2f}%)"
            )
        print("-" * 65)
        print(f"RECOMMENDATION     : >>> {result['advisory']['decision']} <<<")
        print(f"REASON             : {result['advisory']['reason']}")
        print("=" * 65)