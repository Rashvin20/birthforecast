# ---------------------------------------------
# Birth Count Prediction: 3 Models + Inference
# ---------------------------------------------
# Models:
#   (A) GLM (PoissonRegressor; falls back to Tweedie if needed)
#   (B) Poisson GBDT (falls back to squared_error if Poisson loss not supported)
#   (C) Hurdle: Logistic(any-birth) + Poisson(on positives) with SAFE fallbacks
#
# Manual metrics (NumPy only): RMSE, MAE, Poisson deviance, R^2
# Matplotlib: y vs ŷ scatter; residual hist
# Output Excel: next_month_birth_forecasts.xlsx
#   - Sheet "Forecasts": date, colref/col_unique_id, predicted_births
#   - Sheet "Forecasts_all_models": ... + pred_glm, pred_gbdt, pred_hurdle
#
# Requirements: pandas, numpy, matplotlib, scikit-learn, xlsxwriter (or openpyxl)
# ---------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from pandas.tseries.offsets import DateOffset

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# --- Poisson GLM import with fallback to Tweedie ---
USE_TWEEDIE = False
try:
    from sklearn.linear_model import PoissonRegressor
except Exception:
    from sklearn.linear_model import TweedieRegressor
    USE_TWEEDIE = True

# --- GBDT import ---
from sklearn.ensemble import HistGradientBoostingRegressor

# =========
# 0) LOAD
# =========
INPUT_PATH = "Birth_prediction_18.08.25.xlsx"  # adjust if needed
df = pd.read_excel(INPUT_PATH, sheet_name="Sheet1").copy()
df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

for dcol in ["datcre", "latest_date_of_male_intro"]:
    if dcol in df.columns:
        df[dcol] = pd.to_datetime(df[dcol], errors="coerce")

if "datcre" in df.columns:
    df["month"] = df["datcre"].dt.month
    df["year"]  = df["datcre"].dt.year

if "latest_date_of_male_intro" in df.columns and "datcre" in df.columns:
    df["days_since_male_intro"] = (df["datcre"] - df["latest_date_of_male_intro"]).dt.days

if "num_births_this_mth" not in df.columns:
    raise ValueError("Expected column 'num_births_this_mth' not found.")
df["any_birth"] = (df["num_births_this_mth"] > 0).astype(int)

# Chronological order (avoid leakage)
sort_cols = ["datcre"] + (["colref"] if "colref" in df.columns else [])
df = df.sort_values(sort_cols).reset_index(drop=True)

# ==========
# 1) FEATS
# ==========
ignore_cols = {"colid", "datcre", "current_cage", "latest_date_of_male_intro"}
candidate_num = [
    "mths_since_col_creation","current_no_f","no_f_with_bab",
    "min_age","max_age","avg_age",
    "no_f_prev_mth1","avg_age_prev1","no_f_prev_mth2","avg_age_prev2",
    "change_in_num_f_during_mth",
    "num_births_prev_mth","num_births_prev_2_mth","num_births_prev_3_mth",
    "num_births_prev_4_mth","num_births_prev_5_mth","num_births_prev_6_mth",
    "days_since_male_intro","month","year","is_new_colony"
]
num_cols = [c for c in candidate_num if c in df.columns and c not in ignore_cols]
cat_cols = [c for c in ["site","colony_type"] if c in df.columns and c not in ignore_cols]

# ============================
# 2) TRAIN / TEST (time split)
# ============================
model_df = df.dropna(subset=["num_births_this_mth"]).copy()
split_idx = int(0.8 * len(model_df))
train_df = model_df.iloc[:split_idx].copy()
test_df  = model_df.iloc[split_idx:].copy()

X_train = train_df[num_cols + cat_cols].copy()
y_train = train_df["num_births_this_mth"].astype(float).values
X_test  = test_df[num_cols + cat_cols].copy()
y_test  = test_df["num_births_this_mth"].astype(float).values

# ========================
# 3) PREPROCESSOR (Impute+OHE)
# ========================
pre_cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore"))])
pre_num = Pipeline([("impute", SimpleImputer(strategy="median"))])
preprocess = ColumnTransformer([
    ("cat", pre_cat, cat_cols),
    ("num", pre_num, num_cols),
])

# =======================
# 4) MANUAL METRICS
# =======================
def _safe_pos(a, eps=1e-12):
    a = np.asarray(a, dtype=float)
    a[a < eps] = eps
    return a

def manual_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_pred = np.where(y_pred < 0.0, 0.0, y_pred)

    mse = float(np.mean((y_true - y_pred)**2.0))
    rmse = float(np.sqrt(mse))
    mae  = float(np.mean(np.abs(y_true - y_pred)))

    yhat = _safe_pos(y_pred)
    ratio = np.ones_like(y_true)
    mask = y_true > 0
    ratio[mask] = y_true[mask] / yhat[mask]
    term = np.zeros_like(y_true)
    term[mask] = y_true[mask] * np.log(ratio[mask])
    dev  = 2.0 * (term - (y_true - yhat))
    mpd  = float(np.mean(dev))

    ybar = float(np.mean(y_true))
    sst  = float(np.sum((y_true - ybar)**2.0))
    sse  = float(np.sum((y_true - y_pred)**2.0))
    r2   = float(1.0 - sse/sst) if sst > 0 else float("nan")
    return rmse, mae, mpd, r2

# =======================
# 5) BASELINES (manual)
# =======================
if "num_births_prev_mth" in X_test.columns:
    naive_last = X_test["num_births_prev_mth"].fillna(0).values
else:
    naive_last = np.full_like(y_test, fill_value=np.mean(y_train), dtype=float)

prev_cols = [c for c in ["num_births_prev_mth","num_births_prev_2_mth","num_births_prev_3_mth"] if c in X_test.columns]
if prev_cols:
    naive_3m = X_test[prev_cols].astype(float).mean(axis=1).fillna(0).values
else:
    naive_3m = np.full_like(y_test, fill_value=np.mean(y_train), dtype=float)

print("\n=== Time-ordered Test Metrics ===")
rmse, mae, mpd, r2 = manual_metrics(y_test, naive_last)
print("\nNaive_last:")
print(f"  RMSE: {rmse:.4f}\n  MAE: {mae:.4f}\n  PoissonDev: {mpd:.4f}\n  R2: {r2:.4f}")

rmse, mae, mpd, r2 = manual_metrics(y_test, naive_3m)
print("\nNaive_3m:")
print(f"  RMSE: {rmse:.4f}\n  MAE: {mae:.4f}\n  PoissonDev: {mpd:.4f}\n  R2: {r2:.4f}")

# =======================
# 6A) (A) GLM: Poisson/Tweedie
# =======================
if USE_TWEEDIE:
    from sklearn.linear_model import TweedieRegressor
    glm_model = Pipeline([("prep", preprocess),
                          ("model", TweedieRegressor(power=1, alpha=1.0, max_iter=2000))])
else:
    glm_model = Pipeline([("prep", preprocess),
                          ("model", PoissonRegressor(alpha=1.0, max_iter=2000))])

glm_model.fit(X_train, y_train)
pred_glm = glm_model.predict(X_test)
pred_glm = np.where(pred_glm < 0.0, 0.0, pred_glm)
rmse_glm, mae_glm, mpd_glm, r2_glm = manual_metrics(y_test, pred_glm)
print("\n(A) GLM (Poisson/Tweedie):")
print(f"  RMSE: {rmse_glm:.4f}\n  MAE: {mae_glm:.4f}\n  PoissonDev: {mpd_glm:.4f}\n  R2: {r2_glm:.4f}")

# =======================
# 6B) (B) GBDT: Poisson (fallback)
# =======================
USE_SQUARED = False
try:
    gbdt_model = Pipeline([("prep", preprocess),
                           ("model", HistGradientBoostingRegressor(loss="poisson",
                                                                   max_iter=250,
                                                                   learning_rate=0.12,
                                                                   random_state=42))])
    gbdt_model.fit(X_train, y_train)
    pred_gbdt = gbdt_model.predict(X_test)
except Exception:
    USE_SQUARED = True
    gbdt_model = Pipeline([("prep", preprocess),
                           ("model", HistGradientBoostingRegressor(loss="squared_error",
                                                                   max_iter=300,
                                                                   learning_rate=0.10,
                                                                   random_state=42))])
    gbdt_model.fit(X_train, y_train)
    pred_gbdt = gbdt_model.predict(X_test)

pred_gbdt = np.where(pred_gbdt < 0.0, 0.0, pred_gbdt)
rmse_gbdt, mae_gbdt, mpd_gbdt, r2_gbdt = manual_metrics(y_test, pred_gbdt)
print("\n(B) GBDT:", "Poisson" if not USE_SQUARED else "SquaredError (fallback)")
print(f"  RMSE: {rmse_gbdt:.4f}\n  MAE: {mae_gbdt:.4f}\n  PoissonDev: {mpd_gbdt:.4f}\n  R2: {r2_gbdt:.4f}")

# =======================
# 6C) (C) Hurdle: SAFE logistic + Poisson
# =======================
y_train_bin = (y_train > 0).astype(int)
n_pos = int(y_train_bin.sum())
n_neg = int(len(y_train_bin) - n_pos)

# Stage 1: P(any birth)
single_class = (n_pos == 0) or (n_neg == 0)
if single_class:
    # degenerate case: set constant probability
    p_const = float(n_pos / len(y_train))
    proba_any_test = np.full(len(X_test), p_const, dtype=float)
    logit_model = None
else:
    logit_model = Pipeline([("prep", preprocess),
                            ("model", LogisticRegression(max_iter=2000,
                                                         class_weight="balanced",
                                                         solver="lbfgs"))])
    logit_model.fit(X_train, y_train_bin)
    proba_any_test = logit_model.predict_proba(X_test)[:, 1]

# Stage 2: Poisson on positives
pos_train = train_df[train_df["num_births_this_mth"] > 0]
if len(pos_train) == 0:
    # no positives at all -> predicted positive count = 0
    mu_pos_test = np.zeros(len(X_test), dtype=float)
    pos_model = None
else:
    X_train_pos = pos_train[num_cols + cat_cols].copy()
    y_train_pos = pos_train["num_births_this_mth"].astype(float).values
    if USE_TWEEDIE:
        from sklearn.linear_model import TweedieRegressor
        pos_model = Pipeline([("prep", preprocess),
                              ("model", TweedieRegressor(power=1, alpha=1.0, max_iter=2000))])
    else:
        pos_model = Pipeline([("prep", preprocess),
                              ("model", PoissonRegressor(alpha=1.0, max_iter=2000))])
    pos_model.fit(X_train_pos, y_train_pos)
    mu_pos_test = pos_model.predict(X_test)

mu_pos_test = np.where(mu_pos_test < 0.0, 0.0, mu_pos_test)
pred_hurdle = proba_any_test * mu_pos_test
pred_hurdle = np.where(pred_hurdle < 0.0, 0.0, pred_hurdle)

rmse_h, mae_h, mpd_h, r2_h = manual_metrics(y_test, pred_hurdle)
print("\n(C) Hurdle (Logit + Poisson/Tweedie):")
print(f"  RMSE: {rmse_h:.4f}\n  MAE: {mae_h:.4f}\n  PoissonDev: {mpd_h:.4f}\n  R2: {r2_h:.4f}")
if single_class:
    print("  [Note] Logistic stage had one class in training; used constant probability fallback.")

# =======================
# 7) Matplotlib plots
# =======================
def plot_scatter(y_true, y_pred, title):
    plt.figure()
    plt.scatter(y_true, y_pred, s=12)
    lim = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lim, lim)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_residuals(y_true, y_pred, title):
    plt.figure()
    resid = y_true - y_pred
    plt.hist(resid, bins=30)
    plt.xlabel("Residual (y - ŷ)")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.tight_layout()
    plt.show()

plot_scatter(y_test, pred_glm,    "GLM: Actual vs Predicted")
plot_scatter(y_test, pred_gbdt,   "GBDT: Actual vs Predicted")
plot_scatter(y_test, pred_hurdle, "Hurdle: Actual vs Predicted")

plot_residuals(y_test, pred_glm,    "GLM Residuals")
plot_residuals(y_test, pred_gbdt,   "GBDT Residuals")
plot_residuals(y_test, pred_hurdle, "Hurdle Residuals")

# =======================
# 8) Inference (next month, per colony)
# =======================
df_latest = df.copy()
if "colref" in df_latest.columns:
    last_rows = df_latest.sort_values("datcre").groupby("colref", as_index=False).tail(1).copy()
    id_col = "colref"
else:
    grp_cols = [c for c in ["site","colony_type"] if c in df_latest.columns]
    last_rows = df_latest.sort_values("datcre").groupby(grp_cols, as_index=False).tail(1).copy()
    id_col = None
    last_rows["col_unique_id"] = np.arange(len(last_rows))

# Advance one month
last_rows["date"] = last_rows["datcre"] + DateOffset(months=1)

# Shift lag features to next month
def shift_lags_row(row):
    return pd.Series({
        "num_births_prev_mth":   row.get("num_births_this_mth", np.nan),
        "num_births_prev_2_mth": row.get("num_births_prev_mth", np.nan),
        "num_births_prev_3_mth": row.get("num_births_prev_2_mth", np.nan),
        "num_births_prev_4_mth": row.get("num_births_prev_3_mth", np.nan),
        "num_births_prev_5_mth": row.get("num_births_prev_4_mth", np.nan),
        "num_births_prev_6_mth": row.get("num_births_prev_5_mth", np.nan),
    })

lag_updates = last_rows.apply(shift_lags_row, axis=1)
for c in lag_updates.columns:
    last_rows[c] = lag_updates[c]

# Age/timing updates
if "mths_since_col_creation" in last_rows.columns:
    last_rows["mths_since_col_creation"] = last_rows["mths_since_col_creation"].fillna(0) + 1
for a in ["avg_age","min_age","max_age"]:
    if a in last_rows.columns:
        last_rows[a] = pd.to_numeric(last_rows[a], errors="coerce") + 1.0
last_rows["month"] = last_rows["date"].dt.month
last_rows["year"]  = last_rows["date"].dt.year
if "latest_date_of_male_intro" in last_rows.columns:
    last_rows["days_since_male_intro"] = (last_rows["date"] - last_rows["latest_date_of_male_intro"]).dt.days

# Build scoring frame
score_cols = num_cols + cat_cols
for mc in score_cols:
    if mc not in last_rows.columns:
        last_rows[mc] = 0 if mc in num_cols else ""
X_next = last_rows[score_cols].copy()

# Predict with the three models
next_glm    = glm_model.predict(X_next)
next_glm    = np.where(next_glm < 0.0, 0.0, next_glm)

try:
    next_gbdt = gbdt_model.predict(X_next)
except Exception:
    next_gbdt = np.zeros(len(X_next), dtype=float)
next_gbdt   = np.where(next_gbdt < 0.0, 0.0, next_gbdt)

if logit_model is None:
    # same p_const used at test time
    p_const = float(n_pos / len(y_train))
    next_proba = np.full(len(X_next), p_const, dtype=float)
else:
    next_proba = logit_model.predict_proba(X_next)[:, 1]

if pos_model is None:
    next_mu_pos = np.zeros(len(X_next), dtype=float)
else:
    next_mu_pos = pos_model.predict(X_next)
next_mu_pos = np.where(next_mu_pos < 0.0, 0.0, next_mu_pos)
next_hurdle = np.where(next_proba * next_mu_pos < 0.0, 0.0, next_proba * next_mu_pos)

# Choose best model by lowest Poisson deviance on TEST set
poisson_devs = {
    "glm": mpd_glm,
    "gbdt": mpd_gbdt,
    "hurdle": mpd_h
}
best_key = min(poisson_devs, key=poisson_devs.get)
if best_key == "glm":
    chosen = next_glm
elif best_key == "gbdt":
    chosen = next_gbdt
else:
    chosen = next_hurdle

# ===========
# 9) EXPORT
# ===========
OUT_PATH = Path("next_month_birth_forecasts.xlsx")
# Sheet 1: single chosen prediction
if id_col:
    ids = last_rows[id_col]
    id_name = id_col
else:
    ids = last_rows["col_unique_id"]
    id_name = "col_unique_id"

export_main = pd.DataFrame({
    "date": last_rows["date"].dt.date,
    id_name: ids,
    "predicted_births": chosen
})

# Sheet 2: all three model predictions (for transparency)
export_all = pd.DataFrame({
    "date": last_rows["date"].dt.date,
    id_name: ids,
    "pred_glm": next_glm,
    "pred_gbdt": next_gbdt,
    "pred_hurdle": next_hurdle
})

with pd.ExcelWriter(OUT_PATH, engine="xlsxwriter") as writer:
    export_main.to_excel(writer, index=False, sheet_name="Forecasts")
    export_all.to_excel(writer, index=False, sheet_name="Forecasts_all_models")

print(f"\nSaved next-month forecasts to: {OUT_PATH.resolve()}")
print("\nPreview (Forecasts):")
print(export_main.head(10).to_string(index=False))
