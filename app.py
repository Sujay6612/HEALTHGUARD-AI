from datetime import datetime
import csv
from io import StringIO
import json
import os
from pathlib import Path
import pickle
import secrets
import sqlite3

import numpy as np
from flask import Flask, Response, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"
DB_PATH = BASE_DIR / "heartiq_records.db"
ADMIN_ID = os.environ.get("HEARTIQ_ADMIN_ID", "ADMIN")
ADMIN_PASSWORD = os.environ.get("HEARTIQ_ADMIN_PASSWORD", "admin123")
ADMIN_NAME = os.environ.get("HEARTIQ_ADMIN_NAME", "HeartIQ Admin")

FEATURE_RANGES = {
    "age": {"label": "Age", "min": 18, "max": 100, "default": 45},
    "sex": {"label": "Sex", "min": 0, "max": 1, "default": 1},
    "bp": {"label": "Resting Blood Pressure", "min": 80, "max": 220, "default": 120},
    "chol": {"label": "Cholesterol", "min": 100, "max": 400, "default": 190},
    "ecg": {"label": "Resting ECG", "min": 0, "max": 2, "default": 1},
}

TREND_FIELDS = {
    "bp": {
        "label": "Blood pressure",
        "unit": "mmHg",
        "lower_is_better": True,
        "tip": "Focus on low-sodium meals, daily movement, hydration, and regular BP checks.",
    },
    "chol": {
        "label": "Cholesterol",
        "unit": "mg/dL",
        "lower_is_better": True,
        "tip": "Cut back on trans fats, prioritize fiber, and discuss lipid screening with a clinician.",
    },
    "ecg": {
        "label": "Resting ECG",
        "unit": "level",
        "lower_is_better": True,
        "tip": "Repeated ECG worsening deserves professional follow-up, especially with chest discomfort or fatigue.",
    },
}


def load_pickle(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path.name}")
    with path.open("rb") as file:
        return pickle.load(file)


def load_metadata():
    if not METADATA_PATH.exists():
        return {
            "status": "legacy",
            "model_name": "Legacy model",
            "trained_at": None,
            "metrics": {},
            "data_source": "Unknown",
            "notes": [
                "No evaluation metadata found yet.",
                "Retrain with a real dataset to see measured accuracy metrics.",
            ],
            "candidate_results": [],
        }

    with METADATA_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


app = Flask(__name__)
app.secret_key = secrets.token_hex(32)

_artifacts_cache = {
    "model_mtime": None,
    "scaler_mtime": None,
    "metadata_mtime": None,
    "model": None,
    "scaler": None,
    "metadata": None,
}


def get_connection():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS patient_accounts (
                patient_id TEXT PRIMARY KEY,
                patient_name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS patient_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                patient_name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                age INTEGER NOT NULL,
                sex INTEGER NOT NULL,
                bp INTEGER NOT NULL,
                chol INTEGER NOT NULL,
                ecg INTEGER NOT NULL,
                risk_probability REAL NOT NULL,
                confidence REAL NOT NULL,
                prediction_label TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS admin_settings (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                admin_id TEXT NOT NULL,
                admin_name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        existing_admin = connection.execute(
            "SELECT id FROM admin_settings WHERE id = 1"
        ).fetchone()
        if existing_admin is None:
            connection.execute(
                """
                INSERT INTO admin_settings (id, admin_id, admin_name, password_hash, updated_at)
                VALUES (1, ?, ?, ?, ?)
                """,
                (
                    ADMIN_ID.upper(),
                    ADMIN_NAME,
                    generate_password_hash(ADMIN_PASSWORD),
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )


def get_artifacts():
    model_mtime = MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else None
    scaler_mtime = SCALER_PATH.stat().st_mtime if SCALER_PATH.exists() else None
    metadata_mtime = METADATA_PATH.stat().st_mtime if METADATA_PATH.exists() else None

    if _artifacts_cache["model"] is None or _artifacts_cache["model_mtime"] != model_mtime:
        _artifacts_cache["model"] = load_pickle(MODEL_PATH)
        _artifacts_cache["model_mtime"] = model_mtime

    if _artifacts_cache["scaler"] is None or _artifacts_cache["scaler_mtime"] != scaler_mtime:
        _artifacts_cache["scaler"] = load_pickle(SCALER_PATH) if SCALER_PATH.exists() else None
        _artifacts_cache["scaler_mtime"] = scaler_mtime

    if _artifacts_cache["metadata"] is None or _artifacts_cache["metadata_mtime"] != metadata_mtime:
        _artifacts_cache["metadata"] = load_metadata()
        _artifacts_cache["metadata_mtime"] = metadata_mtime

    return (
        _artifacts_cache["model"],
        _artifacts_cache["scaler"],
        _artifacts_cache["metadata"],
    )


def get_default_form_data():
    return {field: config["default"] for field, config in FEATURE_RANGES.items()}


def parse_prediction_form(form):
    values = {}
    errors = []

    for field, config in FEATURE_RANGES.items():
        raw_value = form.get(field, "").strip()
        if raw_value == "":
            values[field] = config["default"]
            errors.append(f"{config['label']} is required.")
            continue

        try:
            value = int(raw_value)
        except ValueError:
            values[field] = config["default"]
            errors.append(f"{config['label']} must be a whole number.")
            continue

        if not config["min"] <= value <= config["max"]:
            errors.append(
                f"{config['label']} must be between {config['min']} and {config['max']}."
            )
        values[field] = value

    return values, errors


def build_prediction(values):
    model, scaler, _ = get_artifacts()
    ordered_values = [
        values["age"],
        values["sex"],
        values["bp"],
        values["chol"],
        values["ecg"],
    ]
    input_array = np.array([ordered_values], dtype=float)
    model_input = scaler.transform(input_array) if scaler is not None else input_array
    predicted_class = int(model.predict(model_input)[0])

    if hasattr(model, "predict_proba"):
        risk_probability = float(model.predict_proba(model_input)[0][1])
    else:
        risk_probability = 1.0 if predicted_class == 1 else 0.0

    probability_percent = round(risk_probability * 100, 1)
    confidence_percent = round(max(risk_probability, 1 - risk_probability) * 100, 1)

    if probability_percent >= 70:
        level = "High attention"
    elif probability_percent >= 40:
        level = "Moderate attention"
    else:
        level = "Lower attention"

    summary = (
        "Potential elevated heart-health risk pattern detected."
        if probability_percent >= 55
        else "Current inputs suggest a lower predicted risk pattern."
    )

    insights = []
    if values["age"] >= 55:
        insights.append("Age is in a range where regular cardiovascular screening becomes more important.")
    if values["bp"] >= 140:
        insights.append("Resting blood pressure looks elevated and may deserve closer monitoring.")
    if values["chol"] >= 240:
        insights.append("Cholesterol is notably high compared with commonly recommended levels.")
    if values["ecg"] == 2:
        insights.append("The ECG input reflects a more concerning resting result.")
    if not insights:
        insights.append("Inputs stay near healthier reference ranges, which helps the overall prediction.")

    guidance = build_personalized_guidance(values, probability_percent, insights)
    suggestions = guidance["priority_actions"]

    return {
        "label": "At Risk" if predicted_class == 1 else "Not at Risk",
        "summary": summary,
        "probability": probability_percent,
        "confidence": confidence_percent,
        "level": level,
        "insights": insights,
        "suggestions": suggestions,
        "guidance": guidance,
    }


def build_personalized_guidance(values, probability_percent, insights):
    diet = []
    exercise = []
    sleep = []
    checkups = []
    focus = []
    wins = []

    if values["bp"] >= 140:
        diet.append("Reduce high-sodium packaged foods and prioritize potassium-rich choices like bananas, spinach, and beans.")
        exercise.append("Aim for steady moderate cardio such as brisk walking for at least 30 minutes on most days.")
        checkups.append("Repeat blood pressure checks over several days instead of relying on one reading.")
        focus.append("Bring resting blood pressure back toward a healthier range.")
    elif values["bp"] <= 125:
        wins.append("Blood pressure is currently in a steadier range.")

    if values["chol"] >= 240:
        diet.append("Increase fiber from oats, fruits, legumes, and vegetables while cutting back on fried and highly processed foods.")
        exercise.append("Pair cardio with 2 or 3 light strength sessions each week to support cholesterol control.")
        checkups.append("Consider lipid profile follow-up if cholesterol stays elevated on repeated checks.")
        focus.append("Lower cholesterol through food quality and consistent activity.")
    elif values["chol"] <= 200:
        wins.append("Cholesterol is staying closer to common healthy targets.")

    if values["ecg"] == 2:
        sleep.append("Protect sleep quality by aiming for a stable sleep window and reducing late-night caffeine or heavy meals.")
        checkups.append("A more concerning ECG category should be reviewed professionally, especially if symptoms are present.")
        focus.append("Monitor ECG-related symptoms and avoid ignoring persistent discomfort, palpitations, or fatigue.")
    elif values["ecg"] == 0:
        wins.append("Resting ECG input looks calmer than the higher-risk category.")

    if values["age"] >= 55:
        checkups.append("Schedule regular cardiovascular screening and keep a simple history of readings over time.")
        sleep.append("Prioritize 7 to 8 hours of sleep and keep stress-recovery habits consistent.")
        focus.append("Consistency matters more now than occasional intense effort.")
    else:
        exercise.append("Build a routine you can keep long-term instead of relying on short bursts of effort.")

    if probability_percent >= 70:
        checkups.append("Because the predicted risk is high, discuss repeated results with a licensed medical professional soon.")
        sleep.append("Support recovery with a consistent sleep schedule and active stress reduction such as walking, breathing work, or screen limits before bed.")
        focus.append("Treat this result as a prompt for follow-up, not something to ignore.")
    elif probability_percent >= 40:
        diet.append("Small daily improvements in food quality can meaningfully shift moderate-risk patterns over time.")
        sleep.append("Reduce sleep debt and keep bedtime more regular to support blood pressure and recovery.")
        focus.append("You are in a middle zone where habits can still move the trend in a better direction.")
    else:
        wins.append("Your current pattern suggests a lower predicted risk profile than the higher-risk groups.")
        exercise.append("Maintain your current routine and protect the habits that are already helping.")

    if not diet:
        diet.append("Keep meals centered on whole foods, lean protein, vegetables, and fiber to protect heart health.")
    if not exercise:
        exercise.append("Stay active with a mix of walking, mobility work, and regular movement throughout the week.")
    if not sleep:
        sleep.append("Keep sleep regular and aim for a calm wind-down routine to support recovery and heart health.")
    if not checkups:
        checkups.append("Continue periodic screening so you can spot changes early rather than waiting for symptoms.")
    if not focus:
        focus.append("Protect the habits that are keeping your current reading in a healthier range.")

    priority_actions = [
        focus[0],
        checkups[0],
        "Use the app to track trends across visits rather than relying on one-off readings.",
    ]

    return {
        "diet": list(dict.fromkeys(diet)),
        "exercise": list(dict.fromkeys(exercise)),
        "sleep": list(dict.fromkeys(sleep)),
        "checkups": list(dict.fromkeys(checkups)),
        "focus": list(dict.fromkeys(focus)),
        "wins": list(dict.fromkeys(wins)),
        "priority_actions": priority_actions,
        "insight_summary": insights[0] if insights else "Current reading is being interpreted against the saved model and your latest inputs.",
        "urgency": (
            "high" if probability_percent >= 70
            else "moderate" if probability_percent >= 40
            else "lower"
        ),
    }


def save_prediction(patient_id, patient_name, values, prediction):
    created_at = datetime.now().isoformat(timespec="seconds")
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO patient_records (
                patient_id, patient_name, created_at, age, sex, bp, chol, ecg,
                risk_probability, confidence, prediction_label
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                patient_id,
                patient_name,
                created_at,
                values["age"],
                values["sex"],
                values["bp"],
                values["chol"],
                values["ecg"],
                prediction["probability"],
                prediction["confidence"],
                prediction["label"],
            ),
        )


def get_patient_history(patient_id, limit=6, ascending=False):
    if not patient_id:
        return []

    order = "ASC" if ascending else "DESC"
    with get_connection() as connection:
        rows = connection.execute(
            f"""
            SELECT patient_name, patient_id, created_at, age, sex, bp, chol, ecg,
                   risk_probability, confidence, prediction_label
            FROM patient_records
            WHERE patient_id = ?
            ORDER BY datetime(created_at) {order}
            LIMIT ?
            """,
            (patient_id, limit),
        ).fetchall()

    return [
        {
            "patient_name": row["patient_name"],
            "patient_id": row["patient_id"],
            "created_at": row["created_at"],
            "created_label": format_timestamp(row["created_at"]),
            "age": row["age"],
            "sex": row["sex"],
            "bp": row["bp"],
            "chol": row["chol"],
            "ecg": row["ecg"],
            "risk_probability": round(float(row["risk_probability"]), 1),
            "confidence": round(float(row["confidence"]), 1),
            "prediction_label": row["prediction_label"],
        }
        for row in rows
    ]


def get_latest_patient_record(patient_id):
    history = get_patient_history(patient_id, limit=1, ascending=False)
    return history[0] if history else None


def search_patients(query, limit=8):
    cleaned = query.strip()
    if not cleaned:
        return []

    pattern = f"%{cleaned}%"
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT patient_id, patient_name, MAX(datetime(created_at)) AS last_seen, COUNT(*) AS visit_count
            FROM patient_records
            WHERE patient_id LIKE ? OR patient_name LIKE ?
            GROUP BY patient_id, patient_name
            ORDER BY datetime(last_seen) DESC
            LIMIT ?
            """,
            (pattern, pattern, limit),
        ).fetchall()

    return [
        {
            "patient_id": row["patient_id"],
            "patient_name": row["patient_name"],
            "last_seen": format_timestamp(row["last_seen"]),
            "visit_count": row["visit_count"],
        }
        for row in rows
    ]


def get_admin_settings():
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT admin_id, admin_name, password_hash, updated_at
            FROM admin_settings
            WHERE id = 1
            """
        ).fetchone()

    return dict(row) if row else None


def update_admin_settings(admin_id, admin_name, password):
    with get_connection() as connection:
        connection.execute(
            """
            UPDATE admin_settings
            SET admin_id = ?, admin_name = ?, password_hash = ?, updated_at = ?
            WHERE id = 1
            """,
            (
                admin_id.strip().upper(),
                admin_name.strip(),
                generate_password_hash(password),
                datetime.now().isoformat(timespec="seconds"),
            ),
        )


def get_admin_patient_history(patient_id, limit=50):
    return get_patient_history(patient_id, limit=limit, ascending=False)


def export_rows_to_csv(rows, filename):
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "patient_id",
            "patient_name",
            "created_at",
            "age",
            "sex",
            "bp",
            "chol",
            "ecg",
            "risk_probability",
            "confidence",
            "prediction_label",
        ]
    )

    for row in rows:
        writer.writerow(
            [
                row["patient_id"],
                row["patient_name"],
                row["created_at"],
                row["age"],
                row["sex"],
                row["bp"],
                row["chol"],
                row["ecg"],
                row["risk_probability"],
                row["confidence"],
                row["prediction_label"],
            ]
        )

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def get_high_risk_alerts(limit=8):
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT patient_id, patient_name, created_at, risk_probability, prediction_label, bp, chol, ecg
            FROM patient_records
            WHERE risk_probability >= 70
            ORDER BY risk_probability DESC, datetime(created_at) DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return [
        {
            "patient_id": row["patient_id"],
            "patient_name": row["patient_name"],
            "created_label": format_timestamp(row["created_at"]),
            "risk_probability": round(float(row["risk_probability"]), 1),
            "prediction_label": row["prediction_label"],
            "bp": row["bp"],
            "chol": row["chol"],
            "ecg": row["ecg"],
        }
        for row in rows
    ]


def get_admin_analytics():
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT
                COUNT(*) AS visit_count,
                COUNT(DISTINCT patient_id) AS patient_count,
                AVG(risk_probability) AS avg_risk,
                SUM(CASE WHEN risk_probability >= 70 THEN 1 ELSE 0 END) AS high_risk_count,
                SUM(CASE WHEN risk_probability >= 40 AND risk_probability < 70 THEN 1 ELSE 0 END) AS moderate_risk_count
            FROM patient_records
            """
        ).fetchone()
        weekly_row = connection.execute(
            """
            SELECT COUNT(*) AS weekly_visits
            FROM patient_records
            WHERE datetime(created_at) >= datetime('now', '-7 days')
            """
        ).fetchone()

    return {
        "visit_count": row["visit_count"] or 0,
        "patient_count": row["patient_count"] or 0,
        "avg_risk": round(float(row["avg_risk"] or 0), 1),
        "high_risk_count": row["high_risk_count"] or 0,
        "moderate_risk_count": row["moderate_risk_count"] or 0,
        "weekly_visits": weekly_row["weekly_visits"] or 0,
    }


def require_admin_session():
    viewer = current_patient()
    return viewer is not None and viewer.get("role") == "admin"


def get_account(patient_id):
    if not patient_id:
        return None

    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT patient_id, patient_name, password_hash, created_at
            FROM patient_accounts
            WHERE patient_id = ?
            """,
            (patient_id.strip().upper(),),
        ).fetchone()

    return dict(row) if row else None


def create_account(patient_id, patient_name, password):
    patient_id = patient_id.strip().upper()
    patient_name = patient_name.strip()

    existing_account = get_account(patient_id)
    if existing_account is not None:
        return False, "This patient ID already has an account. Please log in instead."

    latest_record = get_latest_patient_record(patient_id)
    if latest_record and latest_record["patient_name"].casefold() != patient_name.casefold():
        return False, f"This patient ID has existing records for {latest_record['patient_name']}."

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO patient_accounts (patient_id, patient_name, password_hash, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                patient_id,
                patient_name,
                generate_password_hash(password),
                datetime.now().isoformat(timespec="seconds"),
            ),
        )

    return True, None


def authenticate_account(patient_id, password):
    account = get_account(patient_id)
    if account is None:
        return None

    if not check_password_hash(account["password_hash"], password):
        return None

    return account


def is_admin_credentials(user_id, password):
    settings = get_admin_settings()
    if settings is None:
        return False
    return (
        user_id.strip().upper() == settings["admin_id"].upper()
        and check_password_hash(settings["password_hash"], password)
    )


def login_account(account):
    session["patient_id"] = account["patient_id"]
    session["patient_name"] = account["patient_name"]
    session["role"] = "patient"


def login_admin():
    settings = get_admin_settings()
    session["patient_id"] = settings["admin_id"].upper()
    session["patient_name"] = settings["admin_name"]
    session["role"] = "admin"


def logout_account():
    session.pop("patient_id", None)
    session.pop("patient_name", None)
    session.pop("role", None)


def current_patient():
    patient_id = session.get("patient_id")
    patient_name = session.get("patient_name")
    if not patient_id or not patient_name:
        return None
    return {"patient_id": patient_id, "patient_name": patient_name, "role": session.get("role", "patient")}


def format_timestamp(value):
    try:
        return datetime.fromisoformat(value).strftime("%d %b %Y, %I:%M %p")
    except ValueError:
        return value


def build_comparison(current_values, prediction, previous_record):
    if previous_record is None:
        return {
            "status": "first_visit",
            "headline": "First tracked visit",
            "summary": "This is your first saved reading, so future visits can now be compared against it.",
            "change_percent": None,
            "improvements": [],
            "declines": [],
            "tips": [],
            "delta": 0,
        }

    previous_risk = float(previous_record["risk_probability"])
    current_risk = float(prediction["probability"])
    delta = round(current_risk - previous_risk, 1)

    if delta <= -0.1:
        status = "improved"
        headline = "Health trend improved"
        summary = f"Predicted risk dropped by {abs(delta):.1f}% compared with your previous saved visit."
    elif delta >= 0.1:
        status = "declined"
        headline = "Health trend declined"
        summary = f"Predicted risk increased by {delta:.1f}% compared with your previous saved visit."
    else:
        status = "stable"
        headline = "Health trend stayed stable"
        summary = "Predicted risk is nearly unchanged compared with your last saved visit."

    improvements = []
    declines = []
    tips = []

    for field, config in TREND_FIELDS.items():
        previous_value = previous_record[field]
        current_value = current_values[field]
        change = current_value - previous_value
        if change == 0:
            continue

        better = change < 0 if config["lower_is_better"] else change > 0
        detail = (
            f"{config['label']} moved from {previous_value} to {current_value} {config['unit']}."
            if field != "ecg"
            else f"{config['label']} moved from level {previous_value} to level {current_value}."
        )

        if better:
            improvements.append(detail)
        else:
            declines.append(detail)
            tips.append(config["tip"])

    if not tips and status == "declined":
        tips.append("Review sleep, stress, exercise consistency, and follow-up measurements to identify what shifted.")

    return {
        "status": status,
        "headline": headline,
        "summary": summary,
        "change_percent": abs(delta),
        "trend_symbol": "↓" if status == "improved" else "↑" if status == "declined" else "→" if status == "stable" else "•",
        "trend_label": "Improved" if status == "improved" else "Declined" if status == "declined" else "Stable" if status == "stable" else "Baseline",
        "improvements": improvements,
        "declines": declines,
        "tips": list(dict.fromkeys(tips)),
        "next_focus": declines[0] if declines else (
            improvements[0] if improvements else "Keep building steady habits and compare again after your next reading."
        ),
        "delta": delta,
    }


def format_training_summary():
    _, _, model_metadata = get_artifacts()
    metrics = model_metadata.get("metrics", {})
    trained_at = model_metadata.get("trained_at")
    trained_label = None
    if trained_at:
        try:
            trained_label = datetime.fromisoformat(trained_at).strftime("%d %b %Y, %I:%M %p")
        except ValueError:
            trained_label = trained_at

    return {
        "model_name": model_metadata.get("model_name", "Unknown model"),
        "data_source": model_metadata.get("data_source", "Unknown"),
        "status": model_metadata.get("status", "legacy"),
        "trained_at": trained_label,
        "accuracy": metrics.get("accuracy"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f1": metrics.get("f1"),
        "roc_auc": metrics.get("roc_auc"),
        "candidate_results": model_metadata.get("candidate_results", []),
        "notes": model_metadata.get("notes", []),
    }


def build_dashboard_context(patient, form_data, errors=None, auth_error=None, auth_success=None):
    errors = errors or []
    patient_id = patient["patient_id"]
    patient_name = patient["patient_name"]
    patient_history = get_patient_history(patient_id, limit=6, ascending=False)
    patient_history_chart = get_patient_history(patient_id, limit=12, ascending=True)
    latest_record = patient_history[0] if patient_history else None

    return {
        "logged_in": True,
        "is_admin": False,
        "patient": patient,
        "feature_ranges": FEATURE_RANGES,
        "patient_profile": {
            "patient_name": patient_name,
            "patient_id": patient_id,
            "visit_count": len(patient_history),
            "last_seen": latest_record["created_label"] if latest_record else "No saved visits yet",
        },
        "form_data": form_data,
        "errors": errors,
        "auth_error": auth_error,
        "auth_success": auth_success,
        "prediction": None,
        "comparison": None,
        "patient_history": patient_history,
        "patient_history_chart": patient_history_chart,
        "training_summary": format_training_summary(),
    }


def build_admin_context(auth_success=None, admin_error=None):
    analytics = get_admin_analytics()
    search_query = request.args.get("admin_search", "").strip()
    selected_patient_id = request.args.get("selected_patient", "").strip().upper()
    search_results = search_patients(search_query, limit=12) if search_query else []
    selected_patient_history = get_admin_patient_history(selected_patient_id, limit=25) if selected_patient_id else []
    selected_patient_summary = search_patients(selected_patient_id, limit=1)[0] if selected_patient_history else None

    with get_connection() as connection:
        recent_rows = connection.execute(
            """
            SELECT patient_id, patient_name, created_at, risk_probability, prediction_label
            FROM patient_records
            ORDER BY datetime(created_at) DESC
            LIMIT 8
            """
        ).fetchall()

    recent_patients = [
        {
            "patient_id": row["patient_id"],
            "patient_name": row["patient_name"],
            "created_label": format_timestamp(row["created_at"]),
            "risk_probability": round(float(row["risk_probability"]), 1),
            "prediction_label": row["prediction_label"],
        }
        for row in recent_rows
    ]

    admin_settings = get_admin_settings()
    alerts = get_high_risk_alerts(limit=10)

    return {
        "logged_in": True,
        "is_admin": True,
        "patient": {"patient_id": admin_settings["admin_id"], "patient_name": admin_settings["admin_name"], "role": "admin"},
        "admin_profile": {
            "visit_count": analytics["visit_count"],
            "patient_count": analytics["patient_count"],
            "admin_id": admin_settings["admin_id"],
            "admin_name": admin_settings["admin_name"],
        },
        "admin_analytics": analytics,
        "admin_search_query": search_query,
        "admin_search_results": search_results,
        "selected_patient_id": selected_patient_id,
        "selected_patient_summary": selected_patient_summary,
        "selected_patient_history": selected_patient_history,
        "recent_patients": recent_patients,
        "risk_alerts": alerts,
        "auth_success": auth_success,
        "admin_error": admin_error,
        "training_summary": format_training_summary(),
    }


init_db()


@app.post("/login")
def login():
    patient_id = request.form.get("patient_id", "").strip().upper()
    password = request.form.get("password", "")

    account = get_account(patient_id)
    if account is None:
        return render_template(
            "index.html",
            logged_in=False,
            auth_tab="login",
            auth_error="No user found. Please sign up first.",
        )

    if not check_password_hash(account["password_hash"], password):
        return render_template(
            "index.html",
            logged_in=False,
            auth_tab="login",
            auth_error="Incorrect password. Please try again.",
        )

    login_account(account)
    return redirect(url_for("index", welcome="1"))


@app.post("/admin-login")
def admin_login():
    admin_id = request.form.get("admin_id", "").strip().upper()
    password = request.form.get("password", "")

    if not is_admin_credentials(admin_id, password):
        return render_template(
            "index.html",
            logged_in=False,
            auth_tab="admin",
            auth_error="Invalid admin ID or password.",
        )

    login_admin()
    return redirect(url_for("index", welcome="1"))


@app.post("/register")
def register():
    patient_id = request.form.get("patient_id", "").strip().upper()
    patient_name = request.form.get("patient_name", "").strip()
    password = request.form.get("password", "")
    confirm_password = request.form.get("confirm_password", "")

    if not patient_id or not patient_name or not password:
        return render_template(
            "index.html",
            logged_in=False,
            auth_tab="register",
            auth_error="Patient ID, patient name, and password are required.",
        )

    if password != confirm_password:
        return render_template(
            "index.html",
            logged_in=False,
            auth_tab="register",
            auth_error="Passwords do not match.",
        )

    success, message = create_account(patient_id, patient_name, password)
    if not success:
        return render_template(
            "index.html",
            logged_in=False,
            auth_tab="register",
            auth_error=message,
        )

    account = authenticate_account(patient_id, password)
    login_account(account)
    return redirect(url_for("index", welcome="1"))


@app.post("/logout")
def logout():
    logout_account()
    return redirect(url_for("index"))


@app.get("/admin/export/all")
def admin_export_all():
    if not require_admin_session():
        return redirect(url_for("index"))

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT patient_id, patient_name, created_at, age, sex, bp, chol, ecg,
                   risk_probability, confidence, prediction_label
            FROM patient_records
            ORDER BY datetime(created_at) DESC
            """
        ).fetchall()

    normalized_rows = [dict(row) for row in rows]
    return export_rows_to_csv(normalized_rows, "heartiq_all_records.csv")


@app.get("/admin/export/patient/<patient_id>")
def admin_export_patient(patient_id):
    if not require_admin_session():
        return redirect(url_for("index"))

    history = get_patient_history(patient_id, limit=500, ascending=True)
    return export_rows_to_csv(history, f"{patient_id}_admin_export.csv")


@app.post("/admin/delete-patient/<patient_id>")
def admin_delete_patient(patient_id):
    if not require_admin_session():
        return redirect(url_for("index"))

    with get_connection() as connection:
        connection.execute("DELETE FROM patient_records WHERE patient_id = ?", (patient_id,))
        connection.execute("DELETE FROM patient_accounts WHERE patient_id = ?", (patient_id,))

    return redirect(url_for("index", admin_message=f"Removed records for {patient_id}."))


@app.post("/admin/clear-records")
def admin_clear_records():
    if not require_admin_session():
        return redirect(url_for("index"))

    with get_connection() as connection:
        connection.execute("DELETE FROM patient_records")
        connection.execute("DELETE FROM patient_accounts")

    return redirect(url_for("index", admin_message="Cleared all patient records and accounts."))


@app.post("/admin/update-credentials")
def admin_update_credentials():
    if not require_admin_session():
        return redirect(url_for("index"))

    admin_id = request.form.get("admin_id", "").strip().upper()
    admin_name = request.form.get("admin_name", "").strip()
    password = request.form.get("password", "")
    confirm_password = request.form.get("confirm_password", "")

    if not admin_id or not admin_name or not password:
        return redirect(url_for("index", admin_error="Admin ID, admin name, and password are required."))

    if password != confirm_password:
        return redirect(url_for("index", admin_error="Admin passwords do not match."))

    update_admin_settings(admin_id, admin_name, password)
    session["patient_id"] = admin_id
    session["patient_name"] = admin_name
    session["role"] = "admin"
    return redirect(url_for("index", admin_message="Admin credentials updated successfully."))


@app.get("/export/history")
def export_history():
    patient = current_patient()
    if patient is None:
        return redirect(url_for("index"))

    history = get_patient_history(patient["patient_id"], limit=500, ascending=True)
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "patient_id",
            "patient_name",
            "created_at",
            "age",
            "sex",
            "bp",
            "chol",
            "ecg",
            "risk_probability",
            "confidence",
            "prediction_label",
        ]
    )

    for row in history:
        writer.writerow(
            [
                row["patient_id"],
                row["patient_name"],
                row["created_at"],
                row["age"],
                row["sex"],
                row["bp"],
                row["chol"],
                row["ecg"],
                row["risk_probability"],
                row["confidence"],
                row["prediction_label"],
            ]
        )

    filename = f"{patient['patient_id']}_heartiq_history.csv"
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.route("/", methods=["GET", "POST"])
def index():
    patient = current_patient()
    if patient is None:
        return render_template(
            "index.html",
            logged_in=False,
            auth_tab="login",
            auth_error=None,
        )

    if patient.get("role") == "admin":
        return render_template(
            "index.html",
            **build_admin_context(
                auth_success=(
                    request.args.get("admin_message")
                    or ("Welcome back, " + patient["patient_name"] if request.args.get("welcome") == "1" else None)
                ),
                admin_error=request.args.get("admin_error"),
            ),
        )

    form_data = get_default_form_data()
    context = build_dashboard_context(
        patient,
        form_data=form_data,
        auth_success="Welcome back, " + patient["patient_name"] if request.args.get("welcome") == "1" else None,
    )

    if request.method == "POST":
        form_data, errors = parse_prediction_form(request.form)
        context = build_dashboard_context(patient, form_data=form_data, errors=errors)

        if not errors:
            previous_record = get_latest_patient_record(patient["patient_id"])
            prediction = build_prediction(form_data)
            comparison = build_comparison(form_data, prediction, previous_record)
            save_prediction(patient["patient_id"], patient["patient_name"], form_data, prediction)
            context = build_dashboard_context(patient, form_data=form_data)
            context["prediction"] = prediction
            context["comparison"] = comparison

    return render_template("index.html", **context)


if __name__ == "__main__":
    app.run(debug=True)
