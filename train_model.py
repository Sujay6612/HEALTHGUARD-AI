from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import pickle

import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_CANDIDATES = [
    BASE_DIR / "data" / "heart.csv",
    BASE_DIR / "heart.csv",
    BASE_DIR / "data" / "heart_disease.csv",
    BASE_DIR / "data" / "processed.cleveland.data",
    BASE_DIR / "processed.cleveland.data",
]
MODEL_PATH = BASE_DIR / "model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
METADATA_PATH = BASE_DIR / "model_metadata.json"

APP_FEATURE_COLUMNS = ["age", "sex", "trestbps", "chol", "restecg"]
TARGET_CANDIDATES = ["target", "condition", "output", "label", "num", "heartdisease"]
FEATURE_ALIASES = {
    "age": ["age"],
    "sex": ["sex", "gender"],
    "trestbps": ["trestbps", "resting_bp", "restbp", "restingbloodpressure", "blood_pressure"],
    "chol": ["chol", "cholesterol", "serumcholestoral", "serum_cholesterol"],
    "restecg": ["restecg", "resting_ecg", "ecg", "restecgresults", "restingecg"],
}
UCI_PROCESSED_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]


def find_dataset_path() -> Path:
    for path in DATA_CANDIDATES:
        if path.exists():
            return path
    candidate_list = ", ".join(str(path.relative_to(BASE_DIR)) for path in DATA_CANDIDATES)
    raise FileNotFoundError(
        f"No dataset found. Add a CSV file at one of these paths: {candidate_list}"
    )


def normalize_column_name(column: str) -> str:
    return "".join(character for character in column.lower() if character.isalnum())


def find_target_column(frame: pd.DataFrame) -> str:
    lowered = {normalize_column_name(column): column for column in frame.columns}
    for candidate in TARGET_CANDIDATES:
        normalized_candidate = normalize_column_name(candidate)
        if normalized_candidate in lowered:
            return lowered[normalized_candidate]
    raise ValueError(
        f"Target column not found. Expected one of: {', '.join(TARGET_CANDIDATES)}"
    )


def resolve_feature_columns(frame: pd.DataFrame) -> dict[str, str]:
    normalized_columns = {
        normalize_column_name(column): column for column in frame.columns
    }
    resolved = {}

    for app_column, aliases in FEATURE_ALIASES.items():
        match = None
        for alias in aliases:
            normalized_alias = normalize_column_name(alias)
            if normalized_alias in normalized_columns:
                match = normalized_columns[normalized_alias]
                break
        if match is None:
            alias_list = ", ".join(aliases)
            raise ValueError(
                f"Could not match required feature '{app_column}'. Supported aliases: {alias_list}"
            )
        resolved[app_column] = match

    return resolved


def load_training_frame() -> tuple[pd.DataFrame, str, Path]:
    dataset_path = find_dataset_path()
    if dataset_path.name.lower() == "processed.cleveland.data":
        frame = pd.read_csv(dataset_path, header=None, names=UCI_PROCESSED_COLUMNS, na_values=["?"])
    else:
        frame = pd.read_csv(dataset_path)
        frame.columns = [column.strip() for column in frame.columns]

    target_column = find_target_column(frame)
    feature_mapping = resolve_feature_columns(frame)

    selected = frame[list(feature_mapping.values()) + [target_column]].copy()
    selected = selected.rename(
        columns={source: target for target, source in feature_mapping.items()}
    )
    selected = selected.dropna(subset=[target_column])
    return selected, target_column, dataset_path


def normalize_target(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        normalized = series.astype(str).str.strip().str.lower()
        mapping = {
            "0": 0,
            "1": 1,
            "false": 0,
            "true": 1,
            "no": 0,
            "yes": 1,
            "negative": 0,
            "positive": 1,
            "absence": 0,
            "presence": 1,
            "healthy": 0,
            "disease": 1,
        }
        if normalized.isin(mapping.keys()).all():
            return normalized.map(mapping).astype(int)

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        raise ValueError("Target column contains values that could not be converted to a binary label.")

    # Common heart datasets use 0 for no disease and 1-4 for disease severity.
    return (numeric > 0).astype(int)


def build_candidates():
    common_steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]

    return {
        "logistic_regression": (
            Pipeline(
                common_steps
                + [
                    (
                        "model",
                        LogisticRegression(max_iter=2000, random_state=42),
                    )
                ]
            ),
            {
                "model__C": [0.05, 0.1, 1.0, 10.0, 25.0],
                "model__class_weight": [None, "balanced"],
                "model__solver": ["lbfgs"],
            },
        ),
        "random_forest": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        RandomForestClassifier(random_state=42),
                    ),
                ]
            ),
            {
                "model__n_estimators": [200, 350, 500],
                "model__max_depth": [None, 4, 6, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__class_weight": [None, "balanced"],
                "model__min_samples_split": [2, 4, 8],
            },
        ),
        "gradient_boosting": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", GradientBoostingClassifier(random_state=42)),
                ]
            ),
            {
                "model__n_estimators": [100, 150, 200],
                "model__learning_rate": [0.03, 0.05, 0.08, 0.1],
                "model__max_depth": [1, 2, 3],
                "model__subsample": [0.8, 1.0],
            },
        ),
        "extra_trees": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", ExtraTreesClassifier(random_state=42)),
                ]
            ),
            {
                "model__n_estimators": [250, 400, 600],
                "model__max_depth": [None, 4, 8, 12],
                "model__min_samples_leaf": [1, 2, 4],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "hist_gradient_boosting": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", HistGradientBoostingClassifier(random_state=42)),
                ]
            ),
            {
                "model__learning_rate": [0.03, 0.05, 0.08, 0.1],
                "model__max_depth": [None, 3, 5, 7],
                "model__max_iter": [150, 250, 350],
                "model__min_samples_leaf": [10, 20, 30],
            },
        ),
    }


def evaluate_model(model, features_test, target_test):
    predictions = model.predict(features_test)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features_test)[:, 1]
    else:
        probabilities = predictions

    return {
        "accuracy": round(float(accuracy_score(target_test, predictions)), 4),
        "precision": round(float(precision_score(target_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(target_test, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(target_test, predictions, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(target_test, probabilities)), 4),
    }


def train():
    frame, target_column, dataset_path = load_training_frame()
    features = frame[APP_FEATURE_COLUMNS]
    target = normalize_target(frame[target_column])

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    best_name = None
    best_model = None
    best_metrics = None
    best_cv_score = -1.0
    candidate_results = []

    for model_name, (pipeline, param_grid) in build_candidates().items():
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=5,
            n_jobs=-1,
        )
        search.fit(x_train, y_train)
        metrics = evaluate_model(search.best_estimator_, x_test, y_test)
        candidate_results.append(
            {
                "model_name": model_name,
                "cv_roc_auc": round(float(search.best_score_), 4),
                "test_roc_auc": metrics["roc_auc"],
                "test_accuracy": metrics["accuracy"],
                "test_f1": metrics["f1"],
            }
        )

        if float(search.best_score_) > best_cv_score:
            best_name = model_name
            best_model = search.best_estimator_
            best_metrics = metrics
            best_cv_score = float(search.best_score_)

    with MODEL_PATH.open("wb") as file:
        pickle.dump(best_model, file)

    # Saved only for backward compatibility with older app versions.
    with SCALER_PATH.open("wb") as file:
        pickle.dump(None, file)

    metadata = {
        "status": "evaluated",
        "model_name": best_name,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "data_source": str(dataset_path.relative_to(BASE_DIR)),
        "features": APP_FEATURE_COLUMNS,
        "target_column": target_column,
        "row_count": int(len(frame)),
        "metrics": best_metrics,
        "candidate_results": candidate_results,
        "notes": [
            "The trainer compares multiple tuned models using the same 5 user-facing inputs to improve accuracy without increasing form length.",
            "Metrics are measured on a held-out test split from the provided dataset.",
            "Retrain after updating the dataset to keep the app's reported accuracy current.",
        ],
    }

    with METADATA_PATH.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("Training completed.")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    train()
