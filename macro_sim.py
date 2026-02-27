"""
Macroeconomic Simulation App
Fetches data from BLS and FRED APIs, processes with pandas,
and predicts indicators using scikit-learn linear regression.
"""

import os
import requests
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration — replace placeholders with real keys
# BLS  : https://www.bls.gov/developers/  (free, registration optional)
# FRED : https://fred.stlouisfed.org/docs/api/api_key.html (free registration)
# ---------------------------------------------------------------------------
BLS_API_KEY  = os.environ.get("BLS_API_KEY",  "YOUR_BLS_API_KEY")
FRED_API_KEY = os.environ.get("FRED_API_KEY", "YOUR_FRED_API_KEY")

BLS_BASE_URL  = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

# Series identifiers
BLS_SERIES = {
    "unemployment_rate": "LNS14000000",   # U-3 Unemployment Rate (seasonally adj.)
    "cpi_all_urban":     "CUUR0000SA0",   # CPI-U All Items (not seas. adj.)
}

FRED_SERIES = {
    "real_gdp_growth":   "A191RL1Q225SBEA",  # Real GDP % change, quarterly
    "fed_funds_rate":    "FEDFUNDS",          # Effective Federal Funds Rate, monthly
    "core_pce":          "PCEPILFE",          # Core PCE Price Index, monthly
}

START_YEAR = "2015"
END_YEAR   = "2025"


# ---------------------------------------------------------------------------
# Data Fetching
# ---------------------------------------------------------------------------

def fetch_bls(series_id: str, start_year: str, end_year: str) -> pd.DataFrame:
    """Fetch a single BLS time-series and return a tidy DataFrame."""
    payload = {
        "seriesid":  [series_id],
        "startyear": start_year,
        "endyear":   end_year,
        "catalog":   False,
        "calculations": False,
        "annualaverage": False,
        "registrationkey": BLS_API_KEY,
    }
    try:
        resp = requests.post(BLS_BASE_URL, json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "REQUEST_SUCCEEDED":
            print(f"  [BLS] Warning: {series_id} — {data.get('message', 'unknown error')}")
            return pd.DataFrame()
        rows = data["Results"]["series"][0]["data"]
        df = pd.DataFrame(rows)
        df = df[df["period"].str.startswith("M")]          # monthly only
        df["date"] = pd.to_datetime(
            df["year"] + "-" + df["period"].str[1:], format="%Y-%m"
        )
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df[["date", "value"]].rename(columns={"value": series_id}).sort_values("date")
    except requests.exceptions.RequestException as exc:
        print(f"  [BLS] Network error for {series_id}: {exc}")
        return pd.DataFrame()
    except (KeyError, IndexError, ValueError) as exc:
        print(f"  [BLS] Parse error for {series_id}: {exc}")
        return pd.DataFrame()


def fetch_fred(series_id: str, start_date: str = f"{START_YEAR}-01-01") -> pd.DataFrame:
    """Fetch a single FRED series and return a tidy DataFrame."""
    params = {
        "series_id":       series_id,
        "api_key":         FRED_API_KEY,
        "file_type":       "json",
        "observation_start": start_date,
        "observation_end": f"{END_YEAR}-12-31",
    }
    try:
        resp = requests.get(FRED_BASE_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if "observations" not in data:
            print(f"  [FRED] Warning: {series_id} — no observations returned")
            return pd.DataFrame()
        df = pd.DataFrame(data["observations"])
        df["date"]  = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"].replace(".", np.nan), errors="coerce")
        return df[["date", "value"]].rename(columns={"value": series_id}).sort_values("date")
    except requests.exceptions.RequestException as exc:
        print(f"  [FRED] Network error for {series_id}: {exc}")
        return pd.DataFrame()
    except (KeyError, ValueError) as exc:
        print(f"  [FRED] Parse error for {series_id}: {exc}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Data Assembly
# ---------------------------------------------------------------------------

def build_dataframe() -> pd.DataFrame:
    """Pull all series, resample to quarterly, and merge into one DataFrame."""
    print("\n[1/4] Fetching BLS data...")
    bls_frames = []
    for name, sid in BLS_SERIES.items():
        print(f"      {name} ({sid})")
        df = fetch_bls(sid, START_YEAR, END_YEAR)
        if not df.empty:
            df = df.rename(columns={sid: name})
            bls_frames.append(df.set_index("date"))

    print("\n[2/4] Fetching FRED data...")
    fred_frames = []
    for name, sid in FRED_SERIES.items():
        print(f"      {name} ({sid})")
        df = fetch_fred(sid)
        if not df.empty:
            df = df.rename(columns={sid: name})
            fred_frames.append(df.set_index("date"))

    all_frames = bls_frames + fred_frames
    if not all_frames:
        raise RuntimeError("No data was successfully fetched. Check API keys and network.")

    # Merge on date index
    merged = all_frames[0]
    for frame in all_frames[1:]:
        merged = merged.join(frame, how="outer")

    # Resample to quarterly averages to align mixed-frequency series
    quarterly = merged.resample("QE").mean()

    # Forward-fill up to 1 period, then drop rows still fully NaN
    quarterly = quarterly.ffill(limit=1).dropna(how="all")

    # Compute YoY CPI change as a proxy for inflation rate
    if "cpi_all_urban" in quarterly.columns:
        quarterly["inflation_yoy_pct"] = quarterly["cpi_all_urban"].pct_change(4) * 100

    print(f"\n      Combined dataset: {len(quarterly)} quarters, "
          f"{quarterly.columns.tolist()}")
    return quarterly


# ---------------------------------------------------------------------------
# ML Modelling
# ---------------------------------------------------------------------------

def build_lag_features(series: pd.Series, n_lags: int = 4) -> pd.DataFrame:
    """Return a DataFrame of lagged values for a given series."""
    df = pd.DataFrame({"target": series})
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = series.shift(lag)
    return df.dropna()


def train_and_predict(
    df: pd.DataFrame,
    target_col: str,
    n_lags: int = 4,
    label: str = "",
) -> dict:
    """
    Train a LinearRegression on lag features of `target_col` and
    predict the next period value.
    Returns a dict with the prediction and model diagnostics.
    """
    if target_col not in df.columns:
        return {"error": f"Column '{target_col}' not in dataset."}

    lag_df = build_lag_features(df[target_col], n_lags=n_lags)
    if len(lag_df) < n_lags + 2:
        return {"error": f"Not enough data points for '{target_col}' ({len(lag_df)} rows)."}

    X = lag_df.drop(columns="target").values
    y = lag_df["target"].values

    # Simple temporal train/test split (last 4 quarters held out)
    split = max(len(X) - 4, int(len(X) * 0.8))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_s, y_train)

    # Evaluate on held-out set
    mae = mean_absolute_error(y_test, model.predict(X_test_s)) if len(X_test) else np.nan

    # Predict next period using the most recent n_lags observations
    last_lags = df[target_col].iloc[-n_lags:].values[::-1].reshape(1, -1)
    next_val  = model.predict(scaler.transform(last_lags))[0]

    return {
        "label":        label or target_col,
        "last_actual":  float(df[target_col].dropna().iloc[-1]),
        "last_date":    str(df[target_col].dropna().index[-1].date()),
        "next_pred":    float(next_val),
        "test_mae":     float(mae) if not np.isnan(mae) else None,
        "n_train":      int(split),
        "n_test":       len(X_test),
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

PREDICTION_TARGETS = {
    "real_gdp_growth":   "Real GDP Growth (% QoQ)",
    "inflation_yoy_pct": "Inflation Rate YoY (%)",
    "unemployment_rate": "Unemployment Rate (%)",
    "fed_funds_rate":    "Federal Funds Rate (%)",
}


def print_results(results: list[dict]) -> None:
    """Pretty-print prediction results to the console."""
    divider = "=" * 62
    print(f"\n{divider}")
    print("  MACROECONOMIC INDICATOR PREDICTIONS — Next Quarter")
    print(divider)

    for r in results:
        if "error" in r:
            print(f"\n  {r.get('label', 'Unknown')}")
            print(f"    [!] Skipped: {r['error']}")
            continue

        trend_arrow = "▲" if r["next_pred"] > r["last_actual"] else "▼"
        print(f"\n  {r['label']}")
        print(f"    Last actual  ({r['last_date']}): {r['last_actual']:>8.3f}")
        print(f"    Next quarter prediction:       {r['next_pred']:>8.3f}  {trend_arrow}")
        if r["test_mae"] is not None:
            print(f"    Model MAE on test set:         {r['test_mae']:>8.3f}")
        print(f"    (trained on {r['n_train']} quarters, tested on {r['n_test']})")

    print(f"\n{divider}")
    print("  NOTE: Predictions are based on linear autoregression")
    print("  (lag features only). Do not use for financial decisions.")
    print(divider)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 62)
    print("  Macroeconomic Simulation App")
    print("=" * 62)

    # Validate API keys
    if BLS_API_KEY == "YOUR_BLS_API_KEY":
        print("\n  [!] BLS_API_KEY not set. BLS calls may be rate-limited.")
        print("      Set env var BLS_API_KEY or edit the script.\n")
    if FRED_API_KEY == "YOUR_FRED_API_KEY":
        print("\n  [!] FRED_API_KEY not set. FRED calls will fail.")
        print("      Register free at https://fred.stlouisfed.org/docs/api/api_key.html\n")

    # Build dataset
    print("\n[3/4] Processing data...")
    df = build_dataframe()

    # Run predictions
    print("\n[4/4] Running ML predictions...")
    results = []
    for col, label in PREDICTION_TARGETS.items():
        result = train_and_predict(df, col, n_lags=4, label=label)
        results.append(result)

    # Display
    print_results(results)

    # Optionally save the processed dataset
    out_path = "macro_data.csv"
    df.to_csv(out_path)
    print(f"\n  Raw quarterly data saved to: {out_path}\n")


if __name__ == "__main__":
    main()
