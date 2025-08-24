# training.py
# Linear-only forecasting with robust cleaning, standardized numerics, OHE categoricals.
# Model = Ridge (linear); alpha chosen by minimizing Poisson deviance via time-aware CV.
# Outputs:
#   - Sheet1 (cleaned input)
#   - Forecast (per-colony predictions + discretized)
#   - Monthly_Totals (continuous total + floored integer)
# Prints test MSE + test Poisson deviance.

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# -----------------------------
# Config
# -----------------------------
INPUT_PATH  = "Birth_prediction_18.08.25.xlsx"
SHEET_NAME  = 0
OUTPUT_XLSX = "birth_forecasts_to_next_Aug226.xlsx"
RANDOM_SEED = 42
EPS = 1e-12  # for Poisson deviance stability

# -----------------------------
# Helpers
# -----------------------------
def build_ohe():
    """Handle both new and old sklearn signatures (sparse_output vs sparse)."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def clip_pos(x):
    try:
        return max(0.0, float(x))
    except Exception:
        return 0.0

def poisson_mean_deviance(y_true, y_pred):
    """Mean Poisson deviance (lower is better)."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    yp = np.maximum(yp, EPS)  # avoid log(0); also make non-negative
    # term y*log(y/mu) with convention 0*log(0/mu)=0
    term = np.zeros_like(yt)
    mask = yt > 0
    term[mask] = yt[mask] * np.log(yt[mask] / yp[mask])
    dev = 2.0 * (term - (yt - yp))
    return float(np.mean(dev))

def threshold_map(x: float) -> int:
    """Discretize per your rule."""
    if x < 0.2:
        return 0
    elif x < 1.2:
        return 1
    elif x < 2.3:
        return 2
    else:
        return 3

def month_sin_cos(df):
    """Add simple seasonality features from month; safe, no leakage."""
    m = (df["datcre"].dt.month if "datcre" in df.columns else df["month"]).astype(float)
    df["month_sin"] = np.sin(2*np.pi*(m/12.0))
    df["month_cos"] = np.cos(2*np.pi*(m/12.0))
    return df

def make_expanding_time_splits(n, n_splits=5, min_train_frac=0.5):
    """
    Expanding-window CV for time series:
    yields (train_idx, val_idx) with growing train windows and contiguous val blocks.
    """
    n = int(n)
    n_splits = max(2, min(n_splits, max(2, n//5)))
    min_train = max(1, int(n * min_train_frac))
    if min_train >= n:
        yield np.arange(0, n-1), np.arange(n-1, n)
        return
    fold_edges = np.linspace(min_train, n-1, num=n_splits, dtype=int)
    for i, end in enumerate(fold_edges):
        next_end = fold_edges[i+1] if (i+1) < len(fold_edges) else n-1
        if end >= next_end:
            continue
        train_idx = np.arange(0, end)
        val_idx   = np.arange(end, next_end)
        if len(val_idx) > 0:
            yield train_idx, val_idx

def safe_to_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# -----------------------------
# Load & CLEAN data
# -----------------------------
df = pd.read_excel(INPUT_PATH, sheet_name=SHEET_NAME)

# Standardize column names
df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

# Trim strings
for c in df.select_dtypes(include=["object"]).columns:
    df[c] = df[c].astype(str).str.strip().replace({"": np.nan, "nan": np.nan})

# Dates
for dcol in ["datcre", "latest_date_of_male_intro"]:
    if dcol in df.columns:
        df[dcol] = pd.to_datetime(df[dcol], errors="coerce")

# Drop rows with no date
if "datcre" not in df.columns:
    raise ValueError("Expected 'datcre' column in input")
df = df[~df["datcre"].isna()].copy()

# Target presence check (keep rows for training if target present)
if "num_births_this_mth" not in df.columns:
    raise ValueError("Expected 'num_births_this_mth' in input")

# Coerce key numeric columns; clip negatives on counts
numeric_maybe = [
    "num_births_this_mth",
    "num_births_prev_mth","num_births_prev_2_mth","num_births_prev_3_mth",
    "num_births_prev_4_mth","num_births_prev_5_mth","num_births_prev_6_mth",
    "current_no_f","avg_age","min_age","max_age",
    "mths_since_col_creation","change_in_num_f_during_mth"
]
df = safe_to_numeric(df, numeric_maybe)
for c in ["num_births_this_mth","num_births_prev_mth","num_births_prev_2_mth",
          "num_births_prev_3_mth","num_births_prev_4_mth","num_births_prev_5_mth",
          "num_births_prev_6_mth","current_no_f","mths_since_col_creation"]:
    if c in df.columns:
        df[c] = df[c].clip(lower=0)

# Calendar/derived features
df["month"] = df["datcre"].dt.month
df["year"]  = df["datcre"].dt.year
if "latest_date_of_male_intro" in df.columns:
    df["days_since_male_intro"] = (df["datcre"] - df["latest_date_of_male_intro"]).dt.days

# Optional label helper
df["any_birth"] = np.where(df["num_births_this_mth"].notna() & (df["num_births_this_mth"] > 0), 1, 0)

# Deduplicate exact duplicate rows (safe)
df = df.drop_duplicates().copy()

# Sort by time (and colony if exists)
sort_cols = ["datcre"] + (["colref"] if "colref" in df.columns else [])
df = df.sort_values(sort_cols).reset_index(drop=True)

# Seasonality
df = month_sin_cos(df)

# -----------------------------
# Feature lists
# -----------------------------
num_cols_canonical = [
    "num_births_prev_mth","num_births_prev_2_mth","num_births_prev_3_mth",
    "num_births_prev_4_mth","num_births_prev_5_mth","num_births_prev_6_mth",
    "current_no_f","avg_age","min_age","max_age",
    "mths_since_col_creation","month","year","days_since_male_intro",
    "change_in_num_f_during_mth",
    "month_sin","month_cos"
]
num_cols = [c for c in num_cols_canonical if c in df.columns]
cat_cols = [c for c in ["site","colony_type"] if c in df.columns]

if len(num_cols) + len(cat_cols) == 0:
    # Minimal fallback
    fallback = [
        "num_births_prev_mth","num_births_prev_2_mth","num_births_prev_3_mth",
        "current_no_f","avg_age","mths_since_col_creation","month","year",
        "month_sin","month_cos"
    ]
    num_cols = [c for c in fallback if c in df.columns]
    cat_cols = []

feat_cols = num_cols + cat_cols
assert len(feat_cols) > 0, "No usable features found"

# -----------------------------
# Train / Test split (time-ordered)
# -----------------------------
model_df = df.dropna(subset=["num_births_this_mth"]).copy()
if len(model_df) < 10:
    raise ValueError("Not enough labeled rows to train/test a model (need >= 10).")

split_idx = int(0.8 * len(model_df))
split_idx = min(max(split_idx, 1), len(model_df)-1)
train_df  = model_df.iloc[:split_idx].copy()
test_df   = model_df.iloc[split_idx:].copy()

X_train = train_df[feat_cols].copy()
y_train = train_df["num_births_this_mth"].astype(float).values
X_test  = test_df[feat_cols].copy()
y_test  = test_df["num_births_this_mth"].astype(float).values


pre_cat = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("ohe", build_ohe())
])
pre_num = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler(with_mean=True, with_std=True))
])
preprocess = ColumnTransformer([
    ("cat", pre_cat, cat_cols),
    ("num", pre_num, num_cols)
])

# -----------------------------
# Select alpha by minimizing Poisson deviance (time-aware CV)
# -----------------------------
alpha_grid = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0]  # 0.0 == OLS
cv_records = []

def fit_and_poisson_cv(alpha):
    fold_devs = []
    for tr_idx, va_idx in make_expanding_time_splits(len(X_train), n_splits=5, min_train_frac=0.5):
        X_tr, y_tr = X_train.iloc[tr_idx], y_train[tr_idx]
        X_va, y_va = X_train.iloc[va_idx], y_train[va_idx]

        pipe = Pipeline([
            ("prep", preprocess),
            ("model", Ridge(alpha=alpha, random_state=RANDOM_SEED))
        ])
        pipe.fit(X_tr, y_tr)
        yhat_va = pipe.predict(X_va)
        yhat_va = np.maximum(yhat_va, 0.0)  # counts
        dev = poisson_mean_deviance(y_va, yhat_va)
        fold_devs.append(dev)
    return fold_devs

for a in alpha_grid:
    devs = fit_and_poisson_cv(a)
    cv_records.append((a, float(np.mean(devs)), float(np.std(devs)), devs))

best_alpha, best_mean_dev, best_std_dev, _ = sorted(cv_records, key=lambda t: (t[1], t[0]))[0]

# Fit final model on full train with best alpha
model = Pipeline([
    ("prep", preprocess),
    ("model", Ridge(alpha=best_alpha, random_state=RANDOM_SEED))
])
model.fit(X_train, y_train)

# -----------------------------
# Baseline & Test evaluation (robust to NaNs)
# -----------------------------
global_mean = float(np.nanmean(y_train))
if "num_births_prev_mth" in X_test.columns:
    prev = pd.to_numeric(X_test["num_births_prev_mth"], errors="coerce")
    baseline_pred = prev.fillna(global_mean).to_numpy(dtype=float)
else:
    baseline_pred = np.full_like(y_test, fill_value=global_mean, dtype=float)

baseline_pred = np.maximum(0.0, baseline_pred)

test_pred  = np.maximum(0.0, model.predict(X_test))
train_pred = np.maximum(0.0, model.predict(X_train))

baseline_mse = mean_squared_error(y_test, baseline_pred)
test_mse     = mean_squared_error(y_test, test_pred)
train_mse    = mean_squared_error(y_train, train_pred)

# Poisson deviance (reporting metric)
test_dev  = poisson_mean_deviance(y_test, test_pred)
train_dev = poisson_mean_deviance(y_train, train_pred)

print("\n=== Linear (Ridge/OLS), alpha selected by Poisson deviance CV ===")
print(f"Best alpha: {best_alpha}  (CV Poisson dev mean={best_mean_dev:.6f} ± {best_std_dev:.6f})")
print(f"Train MSE (clipped): {train_mse:.6f}   | Train Poisson dev: {train_dev:.6f}")
print(f" Test MSE (clipped): {test_mse:.6f}   |  Test Poisson dev: {test_dev:.6f}")
print(f"Baseline MSE (prev-month w/ NA fill): {baseline_mse:.6f}")


last_obs    = pd.to_datetime(df["datcre"]).max()
start_month = (last_obs + DateOffset(months=1)).replace(day=1)
end_month   = pd.Timestamp(year=last_obs.year + 1, month=8, day=1)
months = pd.period_range(start=start_month.to_period("M"),
                         end=end_month.to_period("M"),
                         freq="M")

# Last observed row per colony -> starting state
if "colref" in df.columns:
    last = df.sort_values("datcre").groupby("colref", as_index=False).tail(1).copy()
    id_col = "colref"
else:
    group_cols = [c for c in ["site","colony_type"] if c in df.columns]
    last = df.sort_values("datcre").groupby(group_cols, as_index=False).tail(1).copy()
    id_col = "col_unique_id"
    last[id_col] = np.arange(len(last), dtype=int)

def shift_lags_inplace(state_dict, y_pred):
    """Shift lag features and insert y_pred as prev_1 if present."""
    keys = [
        "num_births_prev_mth",
        "num_births_prev_2_mth",
        "num_births_prev_3_mth",
        "num_births_prev_4_mth",
        "num_births_prev_5_mth",
        "num_births_prev_6_mth",
    ]
    if keys[-1] in state_dict: state_dict[keys[-1]] = state_dict.get(keys[-2], np.nan)
    if keys[-2] in state_dict: state_dict[keys[-2]] = state_dict.get(keys[-3], np.nan)
    if keys[-3] in state_dict: state_dict[keys[-3]] = state_dict.get(keys[-4], np.nan)
    if keys[-4] in state_dict: state_dict[keys[-4]] = state_dict.get(keys[-5], np.nan)
    if keys[-5] in state_dict: state_dict[keys[-5]] = state_dict.get(keys[-6], np.nan if len(keys) >= 6 else np.nan)
    if keys[0] in state_dict:  state_dict[keys[0]] = float(y_pred)

def ensure_scoring_row(state_dict, date_for_month):
    """Build single-row DataFrame with expected features for a calendar month."""
    row = {c: state_dict.get(c, np.nan) for c in feat_cols}
    row["month"] = date_for_month.month
    row["year"]  = date_for_month.year
    # seasonal terms for scoring month
    row["month_sin"] = np.sin(2*np.pi*(row["month"]/12.0)) if "month_sin" in feat_cols else np.nan
    row["month_cos"] = np.cos(2*np.pi*(row["month"]/12.0)) if "month_cos" in feat_cols else np.nan
    # male intro offset
    lmi = state_dict.get("latest_date_of_male_intro", np.nan)
    row["days_since_male_intro"] = np.nan if pd.isna(lmi) else (date_for_month - pd.to_datetime(lmi)).days

    s = pd.DataFrame([row])
    for c in cat_cols:
        if c in s.columns:
            s[c] = s[c].astype("object")
    for c in feat_cols:
        if c not in s.columns:
            s[c] = np.nan
    return s[feat_cols]

def predict_one(s: pd.DataFrame) -> float:
    """Predict with linear model; clip to non-negative for counts."""
    return clip_pos(model.predict(s)[0])

forecast_rows = []
for _, last_row in last.iterrows():
    state = last_row.to_dict()
    current_date = pd.to_datetime(state["datcre"])

    for m in months:
        # Walk up to target month m (recursive, one month at a time)
        while current_date.to_period("M") < m:
            next_date = current_date + DateOffset(months=1)

            # Update maturity/ages if present
            if "mths_since_col_creation" in state:
                try:
                    state["mths_since_col_creation"] = float(state.get("mths_since_col_creation", 0)) + 1.0
                except Exception:
                    state["mths_since_col_creation"] = 1.0
            for a in ["avg_age","min_age","max_age"]:
                if a in state:
                    try:
                        state[a] = float(pd.to_numeric(state.get(a), errors="coerce")) + 1.0
                    except Exception:
                        state[a] = np.nan

            s_step = ensure_scoring_row(state, next_date)
            y_step = predict_one(s_step)
            shift_lags_inplace(state, y_step)

            current_date = next_date
            state["datcre"] = current_date
            state["month"]  = current_date.month
            state["year"]   = current_date.year

        # Score month m and save
        s_final = ensure_scoring_row(state, current_date)
        y_hat   = predict_one(s_final)

        out = {
            "date": pd.Timestamp(current_date.date()),
            id_col: state.get(id_col, ""),
            "site": state.get("site", ""),
            "pred_linear": y_hat,
            "predicted_births": y_hat,
            "model_used": f"ridge(alpha={best_alpha})",
            "births_discrete": int(threshold_map(y_hat)),
        }
        forecast_rows.append(out)

# -----------------------------
# Build outputs & save
# -----------------------------
forecast_df = pd.DataFrame(forecast_rows).sort_values(["date", id_col]).reset_index(drop=True)

monthly_totals = forecast_df.groupby("date", as_index=False).agg(
    total_births=("births_discrete", "sum"),
    total_forecast_continuous=("predicted_births", "sum")
)

monthly_totals["total_forecast_floor"] = np.floor(monthly_totals["total_forecast_continuous"]).astype(int)

metrics_df = pd.DataFrame({
    "metric": [
        "best_alpha",
        "cv_poisson_dev_mean",
        "cv_poisson_dev_std",
        "train_mse_clipped",
        "test_mse_clipped",
        "train_poisson_dev",
        "test_poisson_dev",
        "baseline_mse"
    ],
    "value": [
        best_alpha,
        best_mean_dev,
        best_std_dev,
        mean_squared_error(y_train, train_pred),
        test_mse,
        poisson_mean_deviance(y_train, train_pred),
        test_dev,
        baseline_mse
    ]
})


# === BEGIN: Diagnostics & Visuals (OLS vs Ridge) ===
import matplotlib.pyplot as plt

# 1) Train a "normal" least-squares model (Ridge with alpha=0 == OLS)
ols_model = Pipeline([
    ("prep", preprocess),
    ("model", Ridge(alpha=0.0, random_state=RANDOM_SEED))
])
ols_model.fit(X_train, y_train)

train_pred_ols = np.maximum(0.0, ols_model.predict(X_train))
test_pred_ols  = np.maximum(0.0, ols_model.predict(X_test))

# 2) Compute OLS metrics (so you can compare apples-to-apples)
test_mse_ols = mean_squared_error(y_test, test_pred_ols)
test_dev_ols = poisson_mean_deviance(y_test, test_pred_ols)

print("\n=== 'Normal' OLS (alpha=0) vs Ridge ===")
print(f"OLS   -> Test MSE: {test_mse_ols:.6f} | Test Poisson dev: {test_dev_ols:.6f}")
print(f"Ridge -> Test MSE: {test_mse:.6f} | Test Poisson dev: {test_dev:.6f}")

# Add OLS metrics to the Metrics sheet
extra_metrics = pd.DataFrame({
    "metric": ["test_mse_ols", "test_poisson_dev_ols"],
    "value":  [test_mse_ols,     test_dev_ols]
})
metrics_df = pd.concat([metrics_df, extra_metrics], ignore_index=True)

# 3) Helper for Poisson deviance residuals (to see "funnel" go away)
def poisson_deviance_vec(y, mu, eps=EPS):
    y = np.asarray(y, dtype=float)
    mu = np.maximum(np.asarray(mu, dtype=float), eps)
    term = np.zeros_like(y)
    mask = y > 0
    term[mask] = y[mask] * np.log(y[mask] / mu[mask])
    return 2.0 * (term - (y - mu))

def deviance_residuals(y, mu, eps=EPS):
    d = poisson_deviance_vec(y, mu, eps)
    s = np.sign(y - mu)
    return s * np.sqrt(np.maximum(d, 0.0))

# 4) Make and save plots
PLOTS_DIR = Path("plots"); PLOTS_DIR.mkdir(exist_ok=True)

# (a) Actual vs Pred: OLS
plt.figure(figsize=(10,5))
plt.plot(test_df["datcre"], y_test, marker="o", label="Actual")
plt.plot(test_df["datcre"], test_pred_ols, marker="x", label="Predicted (OLS)")
plt.title("Test: Actual vs Predicted (OLS)")
plt.xlabel("Date"); plt.ylabel("Births"); plt.legend(); plt.grid(True)
plt.tight_layout(); plt.savefig(PLOTS_DIR/"01_actual_vs_pred_OLS.png"); plt.close()

# (b) Actual vs Pred: Ridge
plt.figure(figsize=(10,5))
plt.plot(test_df["datcre"], y_test, marker="o", label="Actual")
plt.plot(test_df["datcre"], test_pred, marker="x", label=f"Predicted (Ridge α={best_alpha})")
plt.title("Test: Actual vs Predicted (Ridge)")
plt.xlabel("Date"); plt.ylabel("Births"); plt.legend(); plt.grid(True)
plt.tight_layout(); plt.savefig(PLOTS_DIR/"02_actual_vs_pred_Ridge.png"); plt.close()

# (c) Residuals vs Fitted: OLS
plt.figure(figsize=(9,4))
plt.axhline(0, linewidth=1)
plt.scatter(test_pred_ols, y_test - test_pred_ols)
plt.title("Residuals vs Fitted (OLS)")
plt.xlabel("Fitted μ"); plt.ylabel("Residual"); plt.grid(True)
plt.tight_layout(); plt.savefig(PLOTS_DIR/"03_residuals_vs_fitted_OLS.png"); plt.close()

# (d) Residuals vs Fitted: Ridge
plt.figure(figsize=(9,4))
plt.axhline(0, linewidth=1)
plt.scatter(test_pred, y_test - test_pred)
plt.title("Residuals vs Fitted (Ridge)")
plt.xlabel("Fitted μ"); plt.ylabel("Residual"); plt.grid(True)
plt.tight_layout(); plt.savefig(PLOTS_DIR/"04_residuals_vs_fitted_Ridge.png"); plt.close()

# (e) Deviance residuals vs fitted: OLS
plt.figure(figsize=(9,4))
plt.axhline(0, linewidth=1)
plt.scatter(test_pred_ols, deviance_residuals(y_test, test_pred_ols))
plt.title("Deviance Residuals vs Fitted (OLS)")
plt.xlabel("Fitted μ"); plt.ylabel("Deviance residual"); plt.grid(True)
plt.tight_layout(); plt.savefig(PLOTS_DIR/"05_dev_resid_vs_fitted_OLS.png"); plt.close()

# (f) Deviance residuals vs fitted: Ridge
plt.figure(figsize=(9,4))
plt.axhline(0, linewidth=1)
plt.scatter(test_pred, deviance_residuals(y_test, test_pred))
plt.title("Deviance Residuals vs Fitted (Ridge)")
plt.xlabel("Fitted μ"); plt.ylabel("Deviance residual"); plt.grid(True)
plt.tight_layout(); plt.savefig(PLOTS_DIR/"06_dev_resid_vs_fitted_Ridge.png"); plt.close()

# (g) CV Poisson deviance vs α (already computed in cv_records)
alphas = [r[0] for r in cv_records]
dev_means = [r[1] for r in cv_records]
plt.figure(figsize=(7,4))
plt.plot(alphas, dev_means, marker="o")
plt.title("CV Mean Poisson Deviance vs α")
plt.xlabel("α"); plt.ylabel("Mean Poisson deviance (lower is better)")
plt.grid(True); plt.tight_layout()
plt.savefig(PLOTS_DIR/"07_cv_poisson_dev_vs_alpha.png"); plt.close()

print(f"Saved diagnostic plots to: {PLOTS_DIR.resolve()}")
# === END: Diagnostics & Visuals (OLS vs Ridge) ===

out_path = Path(OUTPUT_XLSX)
with pd.ExcelWriter(out_path, engine="openpyxl", mode="w") as writer:
    df.to_excel(writer, sheet_name="Sheet1", index=False)
    forecast_df.to_excel(writer, sheet_name="Forecast", index=False)
    monthly_totals.to_excel(writer, sheet_name="Monthly_Totals", index=False)
    metrics_df.to_excel(writer, sheet_name="Metrics", index=False)

print(f"\nSaved forecasts to: {out_path.resolve()}")
print("\nMonthly totals (last 5 rows):")
print(monthly_totals.tail(5).to_string(index=False))

# === Forecast totals plot ===
plt.figure(figsize=(10,5))
plt.plot(monthly_totals["date"], monthly_totals["total_forecast_continuous"], marker="o", label="Continuous total")
plt.plot(monthly_totals["date"], monthly_totals["total_births"], marker="x", label="Discrete total (sum of bins)")
plt.title("Forecasted Monthly Totals")
plt.xlabel("Date"); plt.ylabel("Predicted births"); plt.legend(); plt.grid(True)
plt.tight_layout(); plt.savefig(PLOTS_DIR/"08_forecast_monthly_totals.png"); plt.close()
