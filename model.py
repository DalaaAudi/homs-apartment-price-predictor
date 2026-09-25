import os
import warnings
import datetime
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV, RepeatedKFold, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import uniform, randint

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[Warning] XGBoost is not installed - RandomForest will be used as fallback.")

DATASET_PATH = "data/dataset.csv"
MODEL_PATH   = "models/property_model.pkl"
TARGET       = "price"

DROP_COLUMNS = [
    "price_per_sqm",
    "age_category",
    "floor_raw",
    "total_floors",
    "balconies_count",
    "near_school",
    "near_transport",
    "kitchen_type",
    "advertiser_type",
    "in_complex",
    "facade_directions",
    "is_main_street",
    "near_mosque",
    "near_park",
    "near_university",
]

NUMERIC_FEATURES = [
    "total_area",
    "net_area",
    "rooms_count",
    "bathrooms_count",
    "halls_count",
    "floor_num",
    "building_age_clean",
    "near_hospital",        
    "near_market",          
]

BINARY_MAPPINGS: dict[str, set[str]] = {
    "has_elevator": {"يوجد مصعد", "يوجد", "1", "true"},
    "has_parking":  {"مفتوح" , "مغلق", "1", "true"},
    "is_furnished": {"مفروش", "1", "true"},
}

CATEGORICAL_FEATURES = [
    "house_condition",  
    "legal_status",     
]

ENGINEERED_FEATURES = [
    "area_per_room",        
    "net_to_total_ratio",   
    "has_heating",          
    "location_target_enc",  
]

CV_N_SPLITS  = 5
CV_N_REPEATS = 5


def ensure_folders():
    for d in ["data", "models", "reports"]:
        os.makedirs(d, exist_ok=True)


def generate_html_report(metrics: dict, model_name: str, report_dir: str = "reports") -> str:
    os.makedirs(report_dir, exist_ok=True)
    
    now = datetime.datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
    file_timestamp = now.strftime("%Y%m%d_%H%M%S")
    
    # اسم نظيف ومحدد يحوي التاريخ والساعة لمنع التداخل مع الاحتفاظ بملف واحد فقط
    report_path = os.path.join(report_dir, f"evaluation_report_{file_timestamp}.html")
    
    r2 = metrics.get("r2_score", 0)
    mae = metrics.get("mae", 0)
    acc = metrics.get("accuracy", 0)
    cv_mean = metrics.get("cv_r2_mean", None)
    cv_std = metrics.get("cv_r2_std", None)

    cv_html = ""
    if cv_mean is not None and cv_std is not None:
        cv_html = f"""
        <div class="metric-card">
            <span class="label">Repeated CV R²</span>
            <span class="value" dir="ltr">{cv_mean:.4f} &plusmn; {cv_std:.4f}</span>
        </div>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>تقرير تقييم نموذج العقارات - حمص</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #f8fafc;
            color: #1e293b;
            margin: 0;
            padding: 40px 20px;
        }}
        .container {{
            max-width: 850px;
            margin: 0 auto;
            background: #ffffff;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
            padding: 32px;
            border: 1px solid #e2e8f0;
        }}
        .header {{
            border-bottom: 2px solid #f1f5f9;
            padding-bottom: 20px;
            margin-bottom: 28px;
        }}
        .header h1 {{
            margin: 0 0 10px 0;
            color: #0f172a;
            font-size: 22px;
        }}
        .meta {{
            color: #64748b;
            font-size: 13px;
            line-height: 1.6;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 16px;
            margin-bottom: 28px;
        }}
        @media (max-width: 640px) {{
            .grid {{ grid-template-columns: 1fr; }}
        }}
        .metric-card {{
            background: #f8fafc;
            padding: 20px;
            border-radius: 8px;
            border-right: 4px solid #2563eb;
            border-top: 1px solid #f1f5f9;
            border-left: 1px solid #f1f5f9;
            border-bottom: 1px solid #f1f5f9;
        }}
        .metric-card .label {{
            display: block;
            color: #475569;
            font-size: 13px;
            font-weight: 500;
            margin-bottom: 8px;
        }}
        .metric-card .value {{
            font-size: 22px;
            font-weight: 700;
            color: #0f172a;
            display: inline-block;
        }}
        .footer {{
            text-align: center;
            font-size: 12px;
            color: #94a3b8;
            margin-top: 24px;
            border-top: 1px dashed #e2e8f0;
            padding-top: 16px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏠 تقرير تقييم نموذج التنبؤ بأسعار العقارات (حمص)</h1>
            <div class="meta">
                <span><strong>نوع النموذج:</strong> <span dir="ltr">{model_name}</span></span> | 
                <span><strong>تاريخ التدريب:</strong> <span dir="ltr">{timestamp_str}</span></span>
            </div>
        </div>

        <div class="grid">
            <div class="metric-card">
                <span class="label">معامل التحديد (R² Score)</span>
                <span class="value" dir="ltr">{r2:.4f} ({r2*100:.1f}%)</span>
            </div>
            <div class="metric-card">
                <span class="label">متوسط الخطأ المطلق (MAE)</span>
                <span class="value" dir="ltr">${int(mae):,}</span>
            </div>
            <div class="metric-card">
                <span class="label">الدقة ضمن مجال (&plusmn;15%)</span>
                <span class="value" dir="ltr">{acc:.1f}%</span>
            </div>
            {cv_html}
        </div>

        <div class="footer">
            تم إنشاؤه تلقائياً بواسطة نظام Homs Real Estate ML Pipeline
        </div>
    </div>
</body>
</html>
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return report_path


def _encode_binary(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, positive_values in BINARY_MAPPINGS.items():
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().isin(positive_values).astype(int)
    return df


def _build_target_encoding(df_train: pd.DataFrame, y_train_log: pd.Series, smoothing_k: int = 8) -> tuple[dict, float]:
    global_log_mean = float(y_train_log.mean())
    tmp = df_train[["location"]].copy()
    tmp["y"] = y_train_log.values
    stats = tmp.groupby("location")["y"].agg(["mean", "count"])
    stats["smoothed"] = (
        (stats["mean"] * stats["count"] + global_log_mean * smoothing_k)
        / (stats["count"] + smoothing_k)
    )
    return stats["smoothed"].to_dict(), global_log_mean


def _engineer_features(df: pd.DataFrame, target_enc_map: dict, global_log_mean: float) -> pd.DataFrame:
    df = df.copy()
    df["area_per_room"] = df["total_area"] / df["rooms_count"].clip(lower=1)

    if "net_area" in df.columns:
        df["net_to_total_ratio"] = df["net_area"] / df["total_area"].clip(lower=1)

    if "heating_type" in df.columns:
        no_heat = {"لا يوجد", "nan", ""}
        df["has_heating"] = (~df["heating_type"].astype(str).str.strip().isin(no_heat)).astype(int)

    if "location" in df.columns:
        df["location_target_enc"] = (
            df["location"].astype(str).str.strip()
            .map(target_enc_map)
            .fillna(global_log_mean)
        )
    return df


def _fit_label_encoders(df: pd.DataFrame) -> tuple[dict, dict]:
    encoders, modes = {}, {}
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            series = df[col].astype(str).str.strip()
            le = LabelEncoder()
            le.fit(series)
            encoders[col] = le
            modes[col] = series.mode().iloc[0] if not series.empty else "غير معروف"
    return encoders, modes


def _apply_label_encoders(df: pd.DataFrame, encoders: dict, modes: dict) -> pd.DataFrame:
    df = df.copy()
    for col, le in encoders.items():
        if col not in df.columns:
            continue
        known = set(le.classes_)
        fallback_value = modes.get(col, le.classes_[0])
        df[col] = (
            df[col].astype(str).str.strip()
            .apply(lambda x: x if x in known else fallback_value)
        )
        df[col] = le.transform(df[col]).astype(float)
    return df


def _get_feature_columns(df: pd.DataFrame) -> list[str]:
    candidates = (
        NUMERIC_FEATURES
        + list(BINARY_MAPPINGS.keys())
        + CATEGORICAL_FEATURES
        + ENGINEERED_FEATURES
    )
    return [c for c in candidates if c in df.columns]


def _random_search_xgb(X_train: pd.DataFrame, y_train: pd.Series, n_iter: int = 20) -> tuple[XGBRegressor, dict]:
    param_dist = {
        "n_estimators":     randint(100, 350),
        "learning_rate":    uniform(0.02, 0.08),      
        "max_depth":        randint(3, 5),             
        "min_child_weight": randint(3, 9),            
        "subsample":        uniform(0.65, 0.3),         
        "colsample_bytree": uniform(0.6, 0.35),         
        "reg_alpha":        uniform(0.3, 1.5),         
        "reg_lambda":       uniform(1.0, 3.0),         
        "gamma":            uniform(0.1, 0.5),
    }

    cv_strategy = RepeatedKFold(n_splits=CV_N_SPLITS, n_repeats=CV_N_REPEATS, random_state=42)
    base = XGBRegressor(random_state=42, verbosity=0, n_jobs=-1)
    
    search = RandomizedSearchCV(
        estimator            = base,
        param_distributions  = param_dist,
        n_iter               = n_iter,
        cv                   = cv_strategy,
        scoring              = "r2",
        random_state         = 42,
        n_jobs               = -1,
        refit                = True,
        verbose              = 0,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def _repeated_cv_summary(best_params: dict, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    cv_strategy = RepeatedKFold(n_splits=CV_N_SPLITS, n_repeats=CV_N_REPEATS, random_state=123)
    eval_model = XGBRegressor(**{**best_params, "random_state": 42, "verbosity": 0})
    scores = cross_val_score(eval_model, X_train, y_train, cv=cv_strategy, scoring="r2", n_jobs=-1)

    return {
        "cv_r2_mean": float(scores.mean()),
        "cv_r2_std":  float(scores.std()),
        "cv_r2_min":  float(scores.min()),
        "cv_r2_max":  float(scores.max()),
        "cv_n_folds": int(len(scores)),
    }


def _refit_with_early_stopping(best_params: dict, X_train: pd.DataFrame, y_train: pd.Series) -> XGBRegressor:
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=7)
    params = {
        **best_params,
        "n_estimators":          1500,
        "verbosity":             0,
        "random_state":          42,
        "early_stopping_rounds": 40,
    }
    final_model = XGBRegressor(**params)
    final_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return final_model


def train(use_random_search: bool = True) -> dict:
    ensure_folders()
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset file not found at: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]

    cols_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)

    df = _encode_binary(df)
    df = df.dropna(subset=[TARGET])
    df = df[df[TARGET] > 0].reset_index(drop=True)

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    df_train = df_train.copy().reset_index(drop=True)
    df_test  = df_test.copy().reset_index(drop=True)

    y_train_raw = df_train[TARGET].astype(float)
    y_test_raw  = df_test[TARGET].astype(float)
    y_train_log = np.log1p(y_train_raw)
    y_test_log  = np.log1p(y_test_raw)

    target_enc_map, global_log_mean = _build_target_encoding(df_train, y_train_log)

    df_train = _engineer_features(df_train, target_enc_map, global_log_mean)
    df_test  = _engineer_features(df_test,  target_enc_map, global_log_mean)

    label_encoders, label_encoder_modes = _fit_label_encoders(df_train)
    df_train = _apply_label_encoders(df_train, label_encoders, label_encoder_modes)
    df_test  = _apply_label_encoders(df_test,  label_encoders, label_encoder_modes)

    feature_cols = _get_feature_columns(df_train)
    X_train = df_train[feature_cols].astype(float)
    X_test  = df_test[feature_cols].astype(float)

    train_medians = X_train.median()
    X_train = X_train.fillna(train_medians)
    X_test  = X_test.fillna(train_medians)

    best_params, cv_summary = {}, {}
    if XGBOOST_AVAILABLE and use_random_search:
        _, best_params = _random_search_xgb(X_train, y_train_log, n_iter=20)
        cv_summary  = _repeated_cv_summary(best_params, X_train, y_train_log)
        best_model  = _refit_with_early_stopping(best_params, X_train, y_train_log)
        model_name  = "XGBoost (RandomSearch + RepeatedKFold)"
    elif XGBOOST_AVAILABLE:
        best_params = dict(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42)
        best_model  = XGBRegressor(**best_params, verbosity=0)
        best_model.fit(X_train, y_train_log)
        model_name  = "XGBoost (Fixed Params, No Search)"
    else:
        best_model  = RandomForestRegressor(n_estimators=200, max_depth=3, min_samples_leaf=4, random_state=42, n_jobs=-1)
        best_model.fit(X_train, y_train_log)
        model_name  = "RandomForest (XGBoost Unavailable)"

    preds_log = best_model.predict(X_test)
    preds_raw = np.clip(np.expm1(preds_log), 0, None)

    r2        = r2_score(y_test_raw, preds_raw)
    mae       = mean_absolute_error(y_test_raw, preds_raw)
    acc_15pct = float(np.mean(np.abs(preds_raw - y_test_raw.values) / (y_test_raw.values + 1e-9) <= 0.15) * 100)

    GREEN  = "\033[92m"
    CYAN   = "\033[96m"
    YELLOW = "\033[93m"
    GRAY   = "\033[90m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"

    line = f"{GRAY}============================================================{RESET}"

    print(f"\n{line}")
    print(f"  {BOLD}{CYAN}MODEL TRAINING REPORT{RESET}")
    print(line)
    print(f"  Model Architecture : {BOLD}{model_name}{RESET}")
    print(f"  R² Score (Test)    : {GREEN}{BOLD}{r2:.4f} ({r2*100:.1f}%){RESET}")
    if cv_summary:
        print(f"  Repeated CV R²     : {YELLOW}{cv_summary['cv_r2_mean']:.4f}{RESET} ± {GRAY}{cv_summary['cv_r2_std']:.4f}{RESET}")
    print(f"  MAE (Price Error)  : {CYAN}${int(mae):,}{RESET}")
    print(f"  Accuracy (±15%)    : {GREEN}{acc_15pct:.1f}%{RESET}")
    print(f"{line}\n")

    metrics = {
        "r2_score": r2,
        "mae":      mae,
        "accuracy": acc_15pct,
        **cv_summary,   
    }

    report_file = generate_html_report(metrics, model_name)
    print(f"  {GRAY}[Report] HTML report saved to: {report_file}{RESET}\n")

    bundle = {
        "model":               best_model,
        "model_name":          model_name,
        "features":            feature_cols,
        "label_encoders":      label_encoders,
        "label_encoder_modes": label_encoder_modes,   
        "target_enc_map":      target_enc_map,
        "global_log_mean":     global_log_mean,
        "train_medians":       train_medians.to_dict(),
        "binary_mappings":     {k: list(v) for k, v in BINARY_MAPPINGS.items()},
        "metrics":             metrics,
    }
    
    joblib.dump(bundle, MODEL_PATH)
    return metrics


def load_model() -> dict | None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'models', 'model_bundle.joblib')
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception:
            return None
            
    fallback_path = os.path.join(base_dir, MODEL_PATH)
    if os.path.exists(fallback_path):
        try:
            return joblib.load(fallback_path)
        except Exception:
            return None

    return None

def predict(bundle: dict, input_data: dict) -> float:
    features        = bundle["features"]
    label_encoders  = bundle["label_encoders"]
    label_enc_modes = bundle.get("label_encoder_modes", {})
    target_enc_map  = bundle["target_enc_map"]
    global_log_mean = bundle["global_log_mean"]
    train_medians   = bundle["train_medians"]

    row = {col: train_medians.get(col, 0.0) for col in features}
    row.update(input_data)
    df_row = pd.DataFrame([row])

    binary_mappings = {k: set(v) for k, v in bundle.get("binary_mappings", {}).items()}
    for col, positive_values in binary_mappings.items():
        if col in df_row.columns:
            raw = str(df_row[col].iloc[0]).strip()
            df_row[col] = int(
                raw in positive_values 
                or raw.lower() in positive_values 
                or raw in {"1", "true", "on"}
            )

    if "area_per_room" in features:
        rooms = max(float(df_row.get("rooms_count", [1]).iloc[0]), 1)
        df_row["area_per_room"] = float(df_row["total_area"].iloc[0]) / rooms

    if "net_to_total_ratio" in features and "net_area" in df_row.columns:
        total = max(float(df_row["total_area"].iloc[0]), 1)
        df_row["net_to_total_ratio"] = float(df_row["net_area"].iloc[0]) / total

    if "has_heating" in features and "heating_type" in input_data:
        df_row["has_heating"] = int(str(input_data["heating_type"]).strip() not in {"لا يوجد", "nan", ""})

    if "location_target_enc" in features:
        loc = str(input_data.get("location", "")).strip()
        df_row["location_target_enc"] = target_enc_map.get(loc, global_log_mean)

    for col, le in label_encoders.items():
        if col not in df_row.columns:
            continue
        val = str(df_row[col].iloc[0]).strip()
        fallback = label_enc_modes.get(col, le.classes_[0])
        val_to_encode = val if val in set(le.classes_) else fallback
        df_row[col] = le.transform([val_to_encode])[0]

    X = df_row[features].astype(float).fillna(pd.Series(train_medians))
    price_log = float(bundle["model"].predict(X)[0])
    return max(float(np.expm1(price_log)), 0.0)


def get_known_locations(bundle: dict) -> list[str]:
    return sorted(bundle.get("target_enc_map", {}).keys())


if __name__ == "__main__":
    ensure_folders()
    train()