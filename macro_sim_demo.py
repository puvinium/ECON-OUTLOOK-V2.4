"""
macro_sim_demo.py
-----------------
Standalone simulation of macro_sim.py using real historical BLS/FRED data
embedded directly. Runs the full sklearn ML pipeline identically to the live
version — no network calls required. Use this to verify model behaviour or
demo the app when API keys are unavailable.

Data sources (public records):
  BLS  LNS14000000  — U-3 Unemployment Rate (seasonally adjusted, monthly avg)
  BLS  CUUR0000SA0  — CPI-U All Items (not seasonally adjusted, monthly index)
  FRED A191RL1Q225SBEA — Real GDP % change QoQ annualised
  FRED FEDFUNDS       — Effective Federal Funds Rate (monthly avg)
  FRED PCEPILFE       — Core PCE Price Index (monthly avg)
"""

import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Embedded real historical data (2015 Q1 – 2024 Q4, quarterly averages)
# ---------------------------------------------------------------------------

QUARTERLY_INDEX = pd.period_range("2015Q1", "2024Q4", freq="Q")

# U-3 Unemployment Rate (%)
UNEMPLOYMENT = [
    5.63, 5.40, 5.23, 5.00,   # 2015
    4.93, 4.90, 4.93, 4.73,   # 2016
    4.70, 4.40, 4.33, 4.13,   # 2017
    4.07, 3.87, 3.83, 3.73,   # 2018
    3.80, 3.63, 3.57, 3.50,   # 2019
    3.83,13.03, 8.83, 6.73,   # 2020
    6.23, 5.90, 5.13, 4.23,   # 2021
    3.80, 3.63, 3.57, 3.57,   # 2022
    3.50, 3.47, 3.60, 3.73,   # 2023
    3.83, 4.03, 4.23, 4.23,   # 2024
]

# CPI-U All Items Index (not seasonally adjusted)
CPI_ALL_URBAN = [
    234.76, 237.80, 238.65, 237.34,   # 2015
    237.11, 240.23, 240.85, 241.43,   # 2016
    243.24, 244.73, 245.52, 246.52,   # 2017
    248.01, 251.59, 252.15, 252.04,   # 2018
    253.74, 256.09, 256.57, 257.21,   # 2019
    258.68, 256.39, 260.28, 260.47,   # 2020
    262.47, 267.05, 273.09, 278.80,   # 2021
    283.72, 292.30, 296.11, 297.96,   # 2022
    300.84, 304.13, 307.03, 307.05,   # 2023
    309.69, 313.55, 315.30, 315.55,   # 2024
]

# Real GDP % change from prior quarter, annualised (SAAR)
REAL_GDP_GROWTH = [
     3.16,  3.04,  1.97,  0.42,   # 2015
     1.50,  2.31,  2.90,  2.11,   # 2016
     2.13,  3.06,  3.23,  2.81,   # 2017
     2.45,  4.17,  3.44,  1.10,   # 2018
     3.06,  2.35,  2.08,  2.36,   # 2019
    -4.55,-31.22, 35.26,  4.46,   # 2020
     6.33,  6.99,  2.68,  7.04,   # 2021
    -1.56, -0.58,  3.20,  2.64,   # 2022
     2.22,  2.11,  4.93,  3.36,   # 2023
     1.40,  3.03,  2.77,  2.34,   # 2024
]

# Effective Federal Funds Rate (%)
FED_FUNDS_RATE = [
    0.11, 0.13, 0.14, 0.24,   # 2015
    0.37, 0.38, 0.40, 0.54,   # 2016
    0.66, 1.00, 1.16, 1.33,   # 2017
    1.51, 1.82, 1.95, 2.20,   # 2018
    2.40, 2.38, 2.18, 1.74,   # 2019
    1.58, 0.05, 0.09, 0.09,   # 2020
    0.07, 0.06, 0.07, 0.07,   # 2021
    0.08, 0.77, 2.34, 3.65,   # 2022
    4.57, 5.08, 5.33, 5.33,   # 2023
    5.33, 5.33, 5.20, 4.64,   # 2024
]

# Core PCE Price Index (chained 2017 USD, level index)
CORE_PCE = [
    107.03, 107.62, 108.06, 108.47,   # 2015
    108.75, 109.14, 109.59, 110.00,   # 2016
    110.30, 110.74, 111.12, 111.52,   # 2017
    112.00, 112.74, 113.27, 113.83,   # 2018
    114.35, 114.87, 115.32, 115.79,   # 2019
    116.17, 115.89, 117.06, 117.71,   # 2020
    118.47, 120.06, 122.19, 124.42,   # 2021
    126.42, 129.31, 131.04, 132.27,   # 2022
    133.35, 134.44, 135.47, 136.31,   # 2023
    137.11, 138.09, 138.97, 139.74,   # 2024
]


# ---------------------------------------------------------------------------
# Build DataFrame
# ---------------------------------------------------------------------------

def build_dataframe() -> pd.DataFrame:
    dates = [p.to_timestamp(how="E") for p in QUARTERLY_INDEX]   # quarter-end dates
    df = pd.DataFrame({
        "unemployment_rate": UNEMPLOYMENT,
        "cpi_all_urban":     CPI_ALL_URBAN,
        "real_gdp_growth":   REAL_GDP_GROWTH,
        "fed_funds_rate":    FED_FUNDS_RATE,
        "core_pce":          CORE_PCE,
    }, index=pd.DatetimeIndex(dates))

    # YoY CPI inflation proxy (4-quarter pct change)
    df["inflation_yoy_pct"] = df["cpi_all_urban"].pct_change(4) * 100
    return df


# ---------------------------------------------------------------------------
# ML Pipeline (identical to macro_sim.py)
# ---------------------------------------------------------------------------

def build_lag_features(series: pd.Series, n_lags: int = 4) -> pd.DataFrame:
    frame = pd.DataFrame({"target": series})
    for lag in range(1, n_lags + 1):
        frame[f"lag_{lag}"] = series.shift(lag)
    return frame.dropna()


def train_and_predict(df: pd.DataFrame, target_col: str,
                      n_lags: int = 4, label: str = "") -> dict:
    if target_col not in df.columns:
        return {"label": label or target_col,
                "error": f"Column '{target_col}' not in dataset."}

    lag_df = build_lag_features(df[target_col], n_lags=n_lags)
    if len(lag_df) < n_lags + 2:
        return {"label": label or target_col,
                "error": f"Not enough data ({len(lag_df)} rows)."}

    X = lag_df.drop(columns="target").values
    y = lag_df["target"].values

    split = max(len(X) - 4, int(len(X) * 0.8))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_s, y_train)

    mae = mean_absolute_error(y_test, model.predict(X_test_s)) if len(X_test) else np.nan

    last_lags = df[target_col].iloc[-n_lags:].values[::-1].reshape(1, -1)
    next_val  = model.predict(scaler.transform(last_lags))[0]

    return {
        "label":       label or target_col,
        "last_actual": float(df[target_col].dropna().iloc[-1]),
        "last_date":   str(df[target_col].dropna().index[-1].date()),
        "next_pred":   float(next_val),
        "test_mae":    float(mae) if not np.isnan(mae) else None,
        "n_train":     int(split),
        "n_test":      len(X_test),
    }


PREDICTION_TARGETS = {
    "real_gdp_growth":   "Real GDP Growth (% QoQ, SAAR)",
    "inflation_yoy_pct": "CPI Inflation Rate YoY (%)",
    "unemployment_rate": "Unemployment Rate (%)",
    "fed_funds_rate":    "Federal Funds Rate (%)",
}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_header():
    print()
    print("=" * 66)
    print("  MACROECONOMIC SIMULATION APP  —  Simulated API Execution")
    print("  Data: BLS (LNS14000000, CUUR0000SA0) + FRED (GDP, FFR, PCE)")
    print("  Range: 2015 Q1 → 2024 Q4  |  Model: OLS LinearRegression")
    print("=" * 66)


def print_data_summary(df: pd.DataFrame):
    print()
    print("  [STEP 1/3]  Data loaded  ✓")
    print(f"  Rows   : {len(df)} quarterly observations (2015-Q1 → 2024-Q4)")
    print(f"  Columns: {list(df.columns)}")
    print()
    print("  Latest snapshot (2024 Q4):")
    last = df.iloc[-1]
    snap = {
        "  Unemployment Rate (%)  ": last["unemployment_rate"],
        "  CPI Index              ": last["cpi_all_urban"],
        "  Inflation YoY (%)      ": last["inflation_yoy_pct"],
        "  Real GDP Growth (%)    ": last["real_gdp_growth"],
        "  Fed Funds Rate (%)     ": last["fed_funds_rate"],
        "  Core PCE Index         ": last["core_pce"],
    }
    for k, v in snap.items():
        print(f"  {k}: {v:>7.3f}")
    print()
    print("  [STEP 2/3]  Features built (4 autoregressive lags)  ✓")
    print("  [STEP 3/3]  Models trained + evaluated  ✓")


def print_results(results: list):
    divider = "─" * 66
    print()
    print("=" * 66)
    print("  PREDICTIONS  —  Q1 2025  (next quarter from last observation)")
    print("=" * 66)

    for r in results:
        print()
        if "error" in r:
            print(f"  {r.get('label', 'Unknown')}")
            print(f"  {'':>4}[!] Skipped: {r['error']}")
            print(f"  {divider}")
            continue

        delta  = r["next_pred"] - r["last_actual"]
        arrow  = "▲" if delta >= 0 else "▼"
        sign   = "+" if delta >= 0 else ""
        conf   = "Low" if r["test_mae"] is None else (
                 "High" if r["test_mae"] < 0.5 else
                 "Medium" if r["test_mae"] < 2.0 else "Low")

        print(f"  ┌─ {r['label']}")
        print(f"  │  Last actual  [{r['last_date']}] : {r['last_actual']:>8.3f}")
        print(f"  │  Q1 2025 prediction          : {r['next_pred']:>8.3f}  {arrow}  "
              f"({sign}{delta:.3f})")
        if r["test_mae"] is not None:
            print(f"  │  Model MAE (held-out 4 qtrs) : {r['test_mae']:>8.3f}  "
                  f"[Confidence: {conf}]")
        print(f"  │  Training / test split       :  "
              f"{r['n_train']} train  /  {r['n_test']} test quarters")
        print(f"  └{'─' * 64}")

    print()
    print("=" * 66)
    print("  SUMMARY")
    print("=" * 66)
    ok = [r for r in results if "error" not in r]
    for r in ok:
        delta = r["next_pred"] - r["last_actual"]
        arrow = "▲" if delta >= 0 else "▼"
        print(f"  {arrow}  {r['label']:<38} → {r['next_pred']:.3f}%")
    print()
    print("  NOTE: Predictions are purely autoregressive (lag features).")
    print("  They extrapolate recent trends and carry substantial uncertainty.")
    print("  Do not use for financial or policy decisions.")
    print("=" * 66)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print_header()

    print()
    print("  Simulated fetch: BLS LNS14000000  (unemployment rate) ......... OK")
    print("  Simulated fetch: BLS CUUR0000SA0  (CPI all urban) .............. OK")
    print("  Simulated fetch: FRED A191RL1Q225SBEA (real GDP growth) ........ OK")
    print("  Simulated fetch: FRED FEDFUNDS     (federal funds rate) ......... OK")
    print("  Simulated fetch: FRED PCEPILFE     (core PCE index) ............. OK")

    df = build_dataframe()
    print_data_summary(df)

    results = []
    for col, label in PREDICTION_TARGETS.items():
        results.append(train_and_predict(df, col, n_lags=4, label=label))

    print_results(results)

    df.to_csv("macro_data_demo.csv")
    print("  Quarterly dataset saved → macro_data_demo.csv")
    print()


if __name__ == "__main__":
    main()
