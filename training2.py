# training.py
# Trains 4 model families (GLM, GBDT, Hurdle, Linear), picks the best by Poisson deviance,
# rolls forecasts month-by-month until August of the next calendar year after the last
# observation, applies a discrete-threshold rule, and saves:
#   - Sheet1 (original data)
#   - Forecast (per-colony predictions for all models + chosen + discretized)
#   - Monthly_Totals_Thresholded (sum of discretized births per month,
#     and the continuous total from the chosen model for reference)
#
# Input: Birth_prediction_18.08.25.xlsx (first sheet)
# Output: birth_forecasts_to_next_Aug_with_totals.xlsx

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression

# -----------------------------
# Config
# -----------------------------
INPUT_PATH = "Birth_prediction_18.08.25.xlsx"
SHEET_NAME = 0
OUTPUT_XLSX = "birth_forecasts_to_next_Aug_with_totals.xlsx"

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
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.maximum(0.0, np.asarray(y_pred, dtype=float))
    yhat = y_pred.copy()
    yhat[yhat < 1e-12] = 1e-12
    ratio = np.ones_like(y_true)
    mask = y_true > 0
    ratio[mask] = y_true[mask] / yhat[mask]
    term = np.zeros_like(y_true)
    term[mask] = y_true[mask] * np.log(ratio[mask])
    dev = 2.0 * (term - (y_true - yhat))
    return float(np.mean(dev))

def threshold_map(x: float) -> int:
    """Your discretization rule on the chosen model's prediction."""
    if x < 0.2:
        return 0
    elif x < 1.2:
        return 1
    elif x < 2.3:
        return 2
    else:
        return 3

# -----------------------------
# Load data & basic features
# -----------------------------
df = pd.read_excel(INPUT_PATH, sheet_name=SHEET_NAME).copy()
df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

for dcol in ["datcre", "latest_date_of_male_intro"]:
    if dcol in df.columns:
        df[dcol] = pd.to_datetime(df[dcol], errors="coerce")

if "datcre" not in df.columns:
    raise ValueError("Expected 'datcre' column in input")
if "num_births_this_mth" not in df.columns:
    raise ValueError("Expected 'num_births_this_mth' in input")

df["month"] = df["datcre"].dt.month
df["year"]  = df["datcre"].dt.year
if "latest_date_of_male_intro" in df.columns:
    df["days_since_male_intro"] = (df["datcre"] - df["latest_date_of_male_intro"]).dt.days
df["any_birth"] = (df["num_births_this_mth"] > 0).astype(int)

sort_cols = ["datcre"] + (["colref"] if "colref" in df.columns else [])
df = df.sort_values(sort_cols).reset_index(drop=True)

# -----------------------------
# Feature lists
# -----------------------------
num_cols_canonical = [
    "num_births_prev_mth","num_births_prev_2_mth","num_births_prev_3_mth",
    "num_births_prev_4_mth","num_births_prev_5_mth","num_births_prev_6_mth",
    "current_no_f","avg_age","min_age","max_age",
    "mths_since_col_creation","month","year","days_since_male_intro",
    "change_in_num_f_during_mth"
]
num_cols = [c for c in num_cols_canonical if c in df.columns]
cat_cols = [c for c in ["site","colony_type"] if c in df.columns]
if len(num_cols) + len(cat_cols) == 0:
    fallback = [
        "num_births_prev_mth","num_births_prev_2_mth","num_births_prev_3_mth",
        "current_no_f","avg_age","mths_since_col_creation","month","year","days_since_male_intro"
    ]
    num_cols = [c for c in fallback if c in df.columns]
    cat_cols = [c for c in ["site","colony_type"] if c in df.columns]
feat_cols = num_cols + cat_cols
assert len(feat_cols) > 0, "No usable features found"

# -----------------------------
# Train / test split (time-ordered)
# -----------------------------
model_df = df.dropna(subset=["num_births_this_mth"]).copy()
split_idx = int(0.8 * len(model_df))
train_df = model_df.iloc[:split_idx].copy()
test_df  = model_df.iloc[split_idx:].copy()

X_train = train_df[feat_cols].copy()
y_train = train_df["num_births_this_mth"].astype(float).values
X_test  = test_df[feat_cols].copy()
y_test  = test_df["num_births_this_mth"].astype(float).values

# -----------------------------
# Preprocessor
# -----------------------------
pre_cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("ohe", build_ohe())])
pre_num = Pipeline([("impute", SimpleImputer(strategy="median"))])
preprocess = ColumnTransformer([("cat", pre_cat, cat_cols), ("num", pre_num, num_cols)])

# -----------------------------
# Models
# -----------------------------
# (A) GLM Poisson or Tweedie (power=1)
USE_TWEEDIE = False
try:
    from sklearn.linear_model import PoissonRegressor
    glm_model = Pipeline([("prep", preprocess), ("model", PoissonRegressor(alpha=0.01, max_iter=1000))])
except Exception:
    from sklearn.linear_model import TweedieRegressor
    glm_model = Pipeline([("prep", preprocess), ("model", TweedieRegressor(power=1, alpha=0.01, max_iter=1000))])
    USE_TWEEDIE = True
glm_model.fit(X_train, y_train)

# (B) GBDT (Poisson if available, else squared_error)
from sklearn.ensemble import HistGradientBoostingRegressor
try:
    gbdt_model = Pipeline([
        ("prep", preprocess),
        ("model", HistGradientBoostingRegressor(loss="poisson", max_iter=120, learning_rate=0.12, random_state=42))
    ])
    gbdt_model.fit(X_train, y_train)
except Exception:
    gbdt_model = Pipeline([
        ("prep", preprocess),
        ("model", HistGradientBoostingRegressor(loss="squared_error", max_iter=150, learning_rate=0.10, random_state=42))
    ])
    gbdt_model.fit(X_train, y_train)

# (C) Hurdle (Logit + Poisson/Tweedie on positives)
y_train_bin = (y_train > 0).astype(int)
n_pos, n_neg = int(y_train_bin.sum()), int(len(y_train_bin) - y_train_bin.sum())
single_class = (n_pos == 0) or (n_neg == 0)

if single_class:
    p_const = float(n_pos / len(y_train))
    logit_model = None
else:
    logit_model = Pipeline([
        ("prep", preprocess),
        ("model", LogisticRegression(max_iter=400, class_weight="balanced", solver="lbfgs"))
    ])
    logit_model.fit(X_train, y_train_bin)

pos_train = train_df[train_df["num_births_this_mth"] > 0]
if len(pos_train) == 0:
    pos_model = None
else:
    X_train_pos = pos_train[feat_cols].copy()
    y_train_pos = pos_train["num_births_this_mth"].astype(float).values
    if USE_TWEEDIE:
        from sklearn.linear_model import TweedieRegressor
        pos_model = Pipeline([("prep", preprocess), ("model", TweedieRegressor(power=1, alpha=0.01, max_iter=1000))])
    else:
        from sklearn.linear_model import PoissonRegressor
        pos_model = Pipeline([("prep", preprocess), ("model", PoissonRegressor(alpha=0.01, max_iter=1000))])
    pos_model.fit(X_train_pos, y_train_pos)

# (D) Linear Regression
lin_model = Pipeline([("prep", preprocess), ("model", LinearRegression())])
lin_model.fit(X_train, y_train)

# -----------------------------
# Model comparison (test Poisson deviance)
# -----------------------------
pred_glm_test = np.maximum(0.0, glm_model.predict(X_test))
pred_gbdt_test = np.maximum(0.0, gbdt_model.predict(X_test))
if logit_model is None:
    proba_any_test = np.full(len(X_test), float(0.0 if "p_const" not in locals() else p_const), dtype=float)
else:
    proba_any_test = logit_model.predict_proba(X_test)[:, 1]
mu_pos_test = np.zeros(len(X_test), dtype=float) if pos_model is None else pos_model.predict(X_test)
pred_hurdle_test = np.maximum(0.0, proba_any_test * mu_pos_test)
pred_lin_test = np.maximum(0.0, lin_model.predict(X_test))

mpd_glm = poisson_mean_deviance(y_test, pred_glm_test)
mpd_gbdt = poisson_mean_deviance(y_test, pred_gbdt_test)
mpd_h   = poisson_mean_deviance(y_test, pred_hurdle_test)
mpd_lin = poisson_mean_deviance(y_test, pred_lin_test)

poisson_devs = {"glm": mpd_glm, "gbdt": mpd_gbdt, "hurdle": mpd_h, "linear": mpd_lin}
best_key = min(poisson_devs, key=poisson_devs.get)

print("\n=== Time-ordered Test Mean Poisson Deviance (lower=better) ===")
for k, v in poisson_devs.items():
    print(f"{k:>7}: {v:.6f}")
print(f"Best family: {best_key}")

# -----------------------------
# Inference roll to next August
# -----------------------------
# Build month list: from month after last observation → August next year

last_obs = pd.to_datetime(df["datcre"]).max()
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

def predict_one(key: str, s: pd.DataFrame) -> float:
    """Predict a single value with the given family key."""
    if key == "glm":
        return clip_pos(glm_model.predict(s)[0])
    elif key == "gbdt":
        return clip_pos(gbdt_model.predict(s)[0])
    elif key == "hurdle":
        if logit_model is None:
            p = float(0.0 if "p_const" not in globals() else p_const)
        else:
            p = float(logit_model.predict_proba(s)[0, 1])
        if (pos_model is None):
            mu = 0.0
        else:
            mu = clip_pos(pos_model.predict(s)[0])
        return clip_pos(p * mu)
    elif key == "linear":
        return clip_pos(lin_model.predict(s)[0])
    else:
        raise ValueError(f"Unknown family key: {key}")

forecast_rows = []

for _, last_row in last.iterrows():
    state = last_row.to_dict()
    current_date = pd.to_datetime(state["datcre"])

    for m in months:
        # Step forward month-by-month using ONLY the chosen model to advance lags
        while current_date.to_period("M") < m:
            next_date = current_date + DateOffset(months=1)

            # Update maturity/ages
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
            y_step = predict_one(best_key, s_step)   # only chosen model here
            shift_lags_inplace(state, y_step)

            current_date = next_date
            state["datcre"] = current_date
            state["month"]  = current_date.month
            state["year"]   = current_date.year

        # At month m, compute ALL model predictions for saving
        s_final = ensure_scoring_row(state, current_date)
        y_glm  = predict_one("glm", s_final)
        y_gbdt = predict_one("gbdt", s_final)
        y_hurdle = predict_one("hurdle", s_final)
        y_lin  = predict_one("linear", s_final)
        y_chosen = {"glm": y_glm, "gbdt": y_gbdt, "hurdle": y_hurdle, "linear": y_lin}[best_key]

        out = {
            "date": pd.Timestamp(current_date.date()),
            id_col: state.get(id_col, ""),
            "site": state.get("site", ""),
            "pred_glm": y_glm,
            "pred_gbdt": y_gbdt,
            "pred_hurdle": y_hurdle,
            "pred_linear": y_lin,
            "predicted_births": y_chosen,
            "model_used": best_key
        }
        # Apply thresholding rule (discretized count)
        out["births_discrete"] = int(threshold_map(y_chosen))

        forecast_rows.append(out)

# -----------------------------
# Build outputs & save
# -----------------------------

forecast_df = pd.DataFrame(forecast_rows).sort_values(["date", id_col]).reset_index(drop=True)

monthly_totals = forecast_df.groupby("date", as_index=False).agg(
    total_births=("births_discrete", "sum"),
    total_forecast_continuous=("predicted_births", "sum")
)

out_path = Path(OUTPUT_XLSX)
with pd.ExcelWriter(out_path, engine="openpyxl", mode="w") as writer:
    df.to_excel(writer, sheet_name="Sheet1", index=False)
    forecast_df.to_excel(writer, sheet_name="Forecast", index=False)
    monthly_totals.to_excel(writer, sheet_name="Monthly_Totals_Thresholded", index=False)

print(f"\nSaved forecasts to: {out_path.resolve()}")
print("\nMonthly totals (last 5 rows):")
print(monthly_totals.tail(5).to_string(index=False))
