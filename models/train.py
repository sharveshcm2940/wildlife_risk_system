"""
Wildlife-Vehicle Collision Risk — Model Training Pipeline
XGBoost + Random Forest + SHAP explainability + model persistence
"""

import numpy as np
import pandas as pd
import joblib
import json
import os
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    confusion_matrix, brier_score_loss, precision_recall_curve, roc_curve
)
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import shap

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)


# ── Feature schema ────────────────────────────────────────────────────────────
CATEGORICAL_COLS = ['road_type', 'season', 'species']
TARGET           = 'accident'
DROP_COLS        = ['accident', 'risk_score', 'latitude', 'longitude']

FEATURE_GROUPS = {
    "🌍 Spatial / GIS":  ['ndvi', 'dist_water_km', 'corridor_dist_km', 'protected_dist_km'],
    "🚗 Traffic":        ['speed_limit', 'actual_speed', 'speed_ratio', 'driver_risk'],
    "🕐 Temporal":       ['hour', 'day_of_week', 'night_flag', 'dawn_dusk', 'rush_hour', 'breeding_season'],
    "🌧 Environment":    ['rainfall_mm', 'visibility_m', 'temperature_c', 'humidity_pct'],
    "🏆 Engineered":     ['movement_score', 'kde_density', 'night_light', 'rolling_7day'],
    "🗺 Road Geometry":  ['curvature_deg_km', 'street_lighting', 'road_width_m'],
    "🦁 Wildlife":       ['species_risk'],
    "📉 Historical":     ['past_accidents'],
}


class WildlifeRiskModel:
    """End-to-end training, evaluation, and SHAP explainability pipeline.
    Supports both XGBoost and Random Forest models."""

    def __init__(self):
        self.xgb_model    = None
        self.rf_model     = None
        self.model        = None          # alias → best model
        self.encoders     = {}
        self.feature_cols = []
        self.shap_values  = None
        self.shap_sample  = None
        self.shap_expected = 0.0
        self.X_test       = None
        self.y_test       = None
        self.metrics      = {}
        self.xgb_metrics  = {}
        self.rf_metrics   = {}

    # ── Pre-processing ────────────────────────────────────────────────────────
    def preprocess(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        df = df.copy()
        for col in CATEGORICAL_COLS:
            if col in df.columns:
                if fit:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self.encoders[col] = le
                else:
                    le = self.encoders[col]
                    # Handle unseen labels gracefully
                    known = set(le.classes_)
                    df[col] = df[col].astype(str).apply(
                        lambda x: x if x in known else le.classes_[0]
                    )
                    df[col] = le.transform(df[col])
        return df

    def get_feature_cols(self, df: pd.DataFrame) -> list[str]:
        return [c for c in df.columns if c not in DROP_COLS + ['latitude', 'longitude']]

    # ── Evaluate one model ────────────────────────────────────────────────────
    @staticmethod
    def _evaluate(model, X_test, y_test, model_name: str) -> dict:
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        report = classification_report(y_test, y_pred, output_dict=True)
        cm     = confusion_matrix(y_test, y_pred).tolist()
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        prec, rec, _ = precision_recall_curve(y_test, y_prob)

        return {
            "model_name":    model_name,
            "roc_auc":       round(roc_auc_score(y_test, y_prob), 4),
            "avg_precision": round(average_precision_score(y_test, y_prob), 4),
            "brier_score":   round(brier_score_loss(y_test, y_prob), 4),
            "accuracy":      round(float((y_pred == y_test).mean()), 4),
            "report":        report,
            "confusion":     cm,
            "roc_curve":     {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
            "pr_curve":      {"precision": prec.tolist(), "recall": rec.tolist()},
        }

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

        # ═══════════════════════════════════════════════════════════════════════
        # MODEL 1: XGBoost
        # ═══════════════════════════════════════════════════════════════════════
        print("Training XGBoost …")
        self.xgb_model = xgb.XGBClassifier(
            n_estimators     = 500,
            max_depth        = 6,
            learning_rate    = 0.05,
            subsample        = 0.80,
            colsample_bytree = 0.80,
            min_child_weight = 3,
            gamma            = 0.1,
            reg_alpha        = 0.1,
            reg_lambda       = 1.0,
            scale_pos_weight = float((y_train == 0).sum()) / float((y_train == 1).sum()),
            eval_metric      = 'logloss',
            early_stopping_rounds = 30,
            random_state     = 42,
        )
        self.xgb_model.fit(
            X_train, y_train,
            eval_set = [(X_test, y_test)],
            verbose  = False,
        )
        self.xgb_metrics = self._evaluate(self.xgb_model, X_test, y_test, "XGBoost")
        self.xgb_metrics["n_train"]         = len(X_train)
        self.xgb_metrics["n_test"]          = len(X_test)
        self.xgb_metrics["best_iteration"]  = int(self.xgb_model.best_iteration)
        self.xgb_metrics["n_estimators"]    = 500
        self.xgb_metrics["max_depth"]       = 6
        self.xgb_metrics["learning_rate"]   = 0.05
        print(f"  XGBoost  — ROC-AUC: {self.xgb_metrics['roc_auc']:.4f}  "
              f"Accuracy: {self.xgb_metrics['accuracy']:.4f}")

        # ═══════════════════════════════════════════════════════════════════════
        # MODEL 2: Random Forest
        # ═══════════════════════════════════════════════════════════════════════
        print("Training Random Forest …")
        self.rf_model = RandomForestClassifier(
            n_estimators   = 400,
            max_depth      = 12,
            min_samples_split = 5,
            min_samples_leaf  = 2,
            max_features   = "sqrt",
            class_weight   = "balanced",
            n_jobs         = -1,
            random_state   = 42,
        )
        self.rf_model.fit(X_train, y_train)
        self.rf_metrics = self._evaluate(self.rf_model, X_test, y_test, "Random Forest")
        self.rf_metrics["n_train"]        = len(X_train)
        self.rf_metrics["n_test"]         = len(X_test)
        self.rf_metrics["n_estimators"]   = 400
        self.rf_metrics["max_depth"]      = 12
        print(f"  RF       — ROC-AUC: {self.rf_metrics['roc_auc']:.4f}  "
              f"Accuracy: {self.rf_metrics['accuracy']:.4f}")

        # ═══════════════════════════════════════════════════════════════════════
        # Cross-validation (5-fold) for both models
        # ═══════════════════════════════════════════════════════════════════════
        print("Running 5-fold cross-validation …")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        xgb_cv = cross_val_score(
            xgb.XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                scale_pos_weight=float((y_train == 0).sum()) / float((y_train == 1).sum()),
                eval_metric='logloss', random_state=42,
            ),
            X, y, cv=cv, scoring="roc_auc", n_jobs=-1,
        )
        self.xgb_metrics["cv_roc_auc_mean"] = round(float(xgb_cv.mean()), 4)
        self.xgb_metrics["cv_roc_auc_std"]  = round(float(xgb_cv.std()), 4)

        rf_cv = cross_val_score(
            RandomForestClassifier(
                n_estimators=300, max_depth=12,
                class_weight="balanced", random_state=42, n_jobs=-1,
            ),
            X, y, cv=cv, scoring="roc_auc", n_jobs=-1,
        )
        self.rf_metrics["cv_roc_auc_mean"] = round(float(rf_cv.mean()), 4)
        self.rf_metrics["cv_roc_auc_std"]  = round(float(rf_cv.std()), 4)

        print(f"  XGB CV AUC: {xgb_cv.mean():.4f} ± {xgb_cv.std():.4f}")
        print(f"  RF  CV AUC: {rf_cv.mean():.4f} ± {rf_cv.std():.4f}")

        # Choose best as default
        if self.xgb_metrics["roc_auc"] >= self.rf_metrics["roc_auc"]:
            self.model = self.xgb_model
            best_name  = "XGBoost"
        else:
            self.model = self.rf_model
            best_name  = "Random Forest"

        # ═══════════════════════════════════════════════════════════════════════
        # SHAP (for XGBoost — tree-based)
        # ═══════════════════════════════════════════════════════════════════════
        print("Computing SHAP values …")
        explainer        = shap.TreeExplainer(self.xgb_model)
        sample           = X_test.sample(min(500, len(X_test)), random_state=42)
        self.shap_values = explainer.shap_values(sample)
        self.shap_sample = sample
        self.shap_expected = float(explainer.expected_value)

        shap_imp = pd.DataFrame({
            'feature':   self.feature_cols,
            'shap_mean': np.abs(self.shap_values).mean(axis=0),
        }).sort_values('shap_mean', ascending=False)

        self.xgb_metrics['shap_importance'] = shap_imp.to_dict('records')

        # RF feature importance (Gini)
        rf_imp = pd.DataFrame({
            'feature':    self.feature_cols,
            'importance': self.rf_model.feature_importances_,
        }).sort_values('importance', ascending=False)
        self.rf_metrics['feature_importance'] = rf_imp.to_dict('records')

        # Combined metrics dict
        self.metrics = {
            "best_model":   best_name,
            "xgb":          self.xgb_metrics,
            "rf":           self.rf_metrics,
            "feature_cols": self.feature_cols,
            "shap_expected": self.shap_expected,
        }

        print(f"\nBest model: {best_name}")
        return self.metrics

    # ── Predict ───────────────────────────────────────────────────────────────
    def predict_risk(self, input_df: pd.DataFrame) -> np.ndarray:
        """Predict with best model."""
        df_proc = self.preprocess(input_df, fit=False)
        X = df_proc[self.feature_cols]
        return self.model.predict_proba(X)[:, 1]

    def predict_both(self, input_df: pd.DataFrame) -> dict:
        """Predict with BOTH models and return comparison."""
        df_proc = self.preprocess(input_df, fit=False)
        X = df_proc[self.feature_cols]

        xgb_prob = float(self.xgb_model.predict_proba(X)[:, 1][0])
        rf_prob  = float(self.rf_model.predict_proba(X)[:, 1][0])

        return {
            "xgb_probability":  xgb_prob,
            "rf_probability":   rf_prob,
            "avg_probability":  (xgb_prob + rf_prob) / 2,
            "agreement":        abs(xgb_prob - rf_prob) < 0.15,
            "difference":       abs(xgb_prob - rf_prob),
        }

    def predict_shap(self, input_df: pd.DataFrame):
        df_proc   = self.preprocess(input_df, fit=False)
        X         = df_proc[self.feature_cols]
        explainer = shap.TreeExplainer(self.xgb_model)
        sv        = explainer.shap_values(X)
        return sv, X, explainer.expected_value

    # ── Persistence ───────────────────────────────────────────────────────────
    def save(self):
        joblib.dump(self.xgb_model,    MODEL_DIR / "xgb_model.pkl")
        joblib.dump(self.rf_model,     MODEL_DIR / "rf_model.pkl")
        joblib.dump(self.encoders,     MODEL_DIR / "encoders.pkl")
        joblib.dump(self.feature_cols, MODEL_DIR / "feature_cols.pkl")
        joblib.dump(self.shap_values,  MODEL_DIR / "shap_values.pkl")
        joblib.dump(self.shap_sample,  MODEL_DIR / "shap_sample.pkl")
        joblib.dump(self.X_test,       MODEL_DIR / "X_test.pkl")
        joblib.dump(self.y_test,       MODEL_DIR / "y_test.pkl")

        # Save metrics (full — including report)
        with open(MODEL_DIR / "metrics.json", "w") as f:
            json.dump(self.metrics, f, indent=2, default=str)

        print(f"Model artifacts saved → {MODEL_DIR}")

    def load(self):
        self.xgb_model    = joblib.load(MODEL_DIR / "xgb_model.pkl")
        self.encoders     = joblib.load(MODEL_DIR / "encoders.pkl")
        self.feature_cols = joblib.load(MODEL_DIR / "feature_cols.pkl")
        self.shap_values  = joblib.load(MODEL_DIR / "shap_values.pkl")
        self.shap_sample  = joblib.load(MODEL_DIR / "shap_sample.pkl")
        self.X_test       = joblib.load(MODEL_DIR / "X_test.pkl")
        self.y_test       = joblib.load(MODEL_DIR / "y_test.pkl")

        # Load RF if available
        rf_path = MODEL_DIR / "rf_model.pkl"
        if rf_path.exists():
            self.rf_model = joblib.load(rf_path)
        else:
            self.rf_model = None

        with open(MODEL_DIR / "metrics.json") as f:
            self.metrics = json.load(f)

        # Set best model
        best = self.metrics.get("best_model", "XGBoost")
        if best == "Random Forest" and self.rf_model:
            self.model = self.rf_model
        else:
            self.model = self.xgb_model

        # Convenience accessors
        self.xgb_metrics   = self.metrics.get("xgb", {})
        self.rf_metrics    = self.metrics.get("rf", {})
        self.shap_expected = self.metrics.get("shap_expected", 0.0)

        return self


def train_and_save():
    """Full training run — called once on app startup if no saved model exists."""
    from data.generate_data import generate_dataset

    print("Generating dataset …")
    df = generate_dataset(12_000)

    print("Training both models …")
    m = WildlifeRiskModel()
    m.train(df)
    m.save()

    # Persist the dataset too (for map / analytics views)
    df.to_parquet(MODEL_DIR / "dataset.parquet", index=False)
    print("Done.")
    return m, df


if __name__ == "__main__":
    train_and_save()
