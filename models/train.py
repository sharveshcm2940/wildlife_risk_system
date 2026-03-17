"""
Wildlife-Vehicle Collision Risk — Model Training Pipeline
XGBoost + SHAP explainability + model persistence
"""

import numpy as np
import pandas as pd
import joblib
import json
import os
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    confusion_matrix, brier_score_loss
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
import shap

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent
DATA_PATH  = BASE_DIR / "data" / "wildlife_accidents.csv"
MODEL_DIR  = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)


# ── Feature schema ────────────────────────────────────────────────────────────
CATEGORICAL_COLS = ['road_type', 'season', 'species']
TARGET           = 'accident'
DROP_COLS        = ['accident', 'risk_score', 'latitude', 'longitude']

FEATURE_GROUPS = {
    "🌍 Spatial / GIS":       ['ndvi', 'dist_water_km', 'corridor_dist_km', 'protected_dist_km'],
    "🚗 Traffic":             ['speed_limit', 'actual_speed', 'speed_ratio', 'driver_risk'],
    "🕐 Temporal":            ['hour', 'day_of_week', 'night_flag', 'dawn_dusk', 'rush_hour', 'breeding_season'],
    "🌧 Environment":         ['rainfall_mm', 'visibility_m', 'temperature_c', 'humidity_pct'],
    "🏆 Engineered":          ['movement_score', 'kde_density', 'night_light', 'rolling_7day'],
    "🗺 Road Geometry":       ['curvature_deg_km', 'street_lighting', 'road_width_m'],
    "🦁 Wildlife":            ['species_risk'],
    "📉 Historical":          ['past_accidents'],
}


class WildlifeRiskModel:
    """End-to-end training, evaluation, and SHAP explainability pipeline."""

    def __init__(self):
        self.model        = None
        self.encoders     = {}
        self.scaler       = StandardScaler()
        self.feature_cols = []
        self.shap_values  = None
        self.X_test       = None
        self.metrics      = {}

    # ── Pre-processing ────────────────────────────────────────────────────────
    def preprocess(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        df = df.copy()

        # Encode categoricals
        for col in CATEGORICAL_COLS:
            if col in df.columns:
                if fit:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self.encoders[col] = le
                else:
                    le = self.encoders[col]
                    df[col] = le.transform(df[col].astype(str))

        return df

    def get_feature_cols(self, df: pd.DataFrame) -> list[str]:
        return [c for c in df.columns if c not in DROP_COLS + ['latitude', 'longitude']]

    # ── Training ──────────────────────────────────────────────────────────────
    def train(self, df: pd.DataFrame) -> dict:
        df_proc = self.preprocess(df, fit=True)
        self.feature_cols = self.get_feature_cols(df_proc)

        X = df_proc[self.feature_cols]
        y = df_proc[TARGET]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )
        self.X_test  = X_test
        self.y_test  = y_test
        self.X_train = X_train

        # ── XGBoost ───────────────────────────────────────────────────────────
        self.model = xgb.XGBClassifier(
            n_estimators     = 500,
            max_depth        = 6,
            learning_rate    = 0.05,
            subsample        = 0.80,
            colsample_bytree = 0.80,
            min_child_weight = 3,
            gamma            = 0.1,
            reg_alpha        = 0.1,
            reg_lambda       = 1.0,
            scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum(),
            eval_metric      = 'logloss',
            early_stopping_rounds = 30,
            random_state     = 42,
            use_label_encoder= False,
        )

        self.model.fit(
            X_train, y_train,
            eval_set    = [(X_test, y_test)],
            verbose     = False,
        )

        # ── Evaluation ────────────────────────────────────────────────────────
        y_pred      = self.model.predict(X_test)
        y_prob      = self.model.predict_proba(X_test)[:, 1]

        self.metrics = {
            "roc_auc"     : round(roc_auc_score(y_test, y_prob),   4),
            "avg_precision": round(average_precision_score(y_test, y_prob), 4),
            "brier_score" : round(brier_score_loss(y_test, y_prob), 4),
            "accuracy"    : round((y_pred == y_test).mean(),         4),
            "report"      : classification_report(y_test, y_pred, output_dict=True),
            "confusion"   : confusion_matrix(y_test, y_pred).tolist(),
            "n_train"     : len(X_train),
            "n_test"      : len(X_test),
            "best_iteration": self.model.best_iteration,
        }

        # ── SHAP ──────────────────────────────────────────────────────────────
        print("Computing SHAP values …")
        explainer          = shap.TreeExplainer(self.model)
        sample             = X_test.sample(min(500, len(X_test)), random_state=42)
        self.shap_values   = explainer.shap_values(sample)
        self.shap_sample   = sample
        self.shap_expected = float(explainer.expected_value)

        # Feature importance from SHAP
        shap_imp = pd.DataFrame({
            'feature'   : self.feature_cols,
            'shap_mean' : np.abs(self.shap_values).mean(axis=0),
        }).sort_values('shap_mean', ascending=False)

        self.metrics['shap_importance'] = shap_imp.to_dict('records')

        print(f"  ROC-AUC : {self.metrics['roc_auc']:.4f}")
        print(f"  Avg-Prec: {self.metrics['avg_precision']:.4f}")
        print(f"  Accuracy: {self.metrics['accuracy']:.4f}")

        return self.metrics

    # ── Predict ───────────────────────────────────────────────────────────────
    def predict_risk(self, input_df: pd.DataFrame) -> np.ndarray:
        df_proc = self.preprocess(input_df, fit=False)
        X       = df_proc[self.feature_cols]
        return self.model.predict_proba(X)[:, 1]

    def predict_shap(self, input_df: pd.DataFrame):
        df_proc  = self.preprocess(input_df, fit=False)
        X        = df_proc[self.feature_cols]
        explainer = shap.TreeExplainer(self.model)
        sv        = explainer.shap_values(X)
        return sv, X, explainer.expected_value

    # ── Persistence ───────────────────────────────────────────────────────────
    def save(self):
        joblib.dump(self.model,        MODEL_DIR / "xgb_model.pkl")
        joblib.dump(self.encoders,     MODEL_DIR / "encoders.pkl")
        joblib.dump(self.scaler,       MODEL_DIR / "scaler.pkl")
        joblib.dump(self.feature_cols, MODEL_DIR / "feature_cols.pkl")
        joblib.dump(self.shap_values,  MODEL_DIR / "shap_values.pkl")
        joblib.dump(self.shap_sample,  MODEL_DIR / "shap_sample.pkl")
        joblib.dump(self.X_test,       MODEL_DIR / "X_test.pkl")
        joblib.dump(self.y_test,       MODEL_DIR / "y_test.pkl")

        with open(MODEL_DIR / "metrics.json", "w") as f:
            metrics_serializable = {
                k: v for k, v in self.metrics.items()
                if k not in ('report',)
            }
            metrics_serializable['shap_expected'] = self.shap_expected
            json.dump(metrics_serializable, f, indent=2)
        print(f"Model artifacts saved → {MODEL_DIR}")

    def load(self):
        self.model        = joblib.load(MODEL_DIR / "xgb_model.pkl")
        self.encoders     = joblib.load(MODEL_DIR / "encoders.pkl")
        self.scaler       = joblib.load(MODEL_DIR / "scaler.pkl")
        self.feature_cols = joblib.load(MODEL_DIR / "feature_cols.pkl")
        self.shap_values  = joblib.load(MODEL_DIR / "shap_values.pkl")
        self.shap_sample  = joblib.load(MODEL_DIR / "shap_sample.pkl")
        self.X_test       = joblib.load(MODEL_DIR / "X_test.pkl")
        self.y_test       = joblib.load(MODEL_DIR / "y_test.pkl")
        with open(MODEL_DIR / "metrics.json") as f:
            self.metrics  = json.load(f)
        return self


def train_and_save():
    """Full training run — called once on app startup if no saved model exists."""
    from data.generate_data import generate_dataset

    print("Generating dataset …")
    df = generate_dataset(12_000)

    print("Training model …")
    m = WildlifeRiskModel()
    m.train(df)
    m.save()

    # Persist the dataset too (for map / analytics views)
    df.to_parquet(MODEL_DIR / "dataset.parquet", index=False)
    print("Done.")
    return m, df


if __name__ == "__main__":
    train_and_save()
