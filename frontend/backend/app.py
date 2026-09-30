from __future__ import annotations

from datetime import datetime, timedelta, timezone
from functools import wraps
import json
import logging
import os
from pathlib import Path
import random
import re
import secrets
import smtplib
import sqlite3
import threading
import time
from email.message import EmailMessage
from typing import Any, Callable

from flask import Flask, jsonify, redirect, render_template, request, session, url_for
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

try:
    from dotenv import load_dotenv
except ImportError as exc:
    raise RuntimeError(
        "python-dotenv is required to load the .env file. Install it with 'pip install python-dotenv'."
    ) from exc

from werkzeug.security import check_password_hash, generate_password_hash

from model_predict import FloodModelRegistry, FloodPredictor, PredictionError, RainfallFeatureStore
from sms_service import SMSService
from telegram_service import fetch_telegram_updates, send_telegram, verify_telegram_bot_token
from weather_fetch import OpenWeatherClient, WeatherAPIError

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("app")

E164_PHONE_RE = re.compile(r"^\+[1-9]\d{7,14}$")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
COORDINATE_LOCATION_RE = re.compile(r"^-?\d+(?:\.\d+)?,-?\d+(?:\.\d+)?$")

ENV_KEYS = {
    "OPENWEATHER_API_KEY",
    "TWILIO_ACCOUNT_SID",
    "TWILIO_AUTH_TOKEN",
    "TWILIO_PHONE",
    "USER_PHONE",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID",
    "TELEGRAM_POLL_INTERVAL_SECONDS",
    "DEFAULT_STATE",
    "DEFAULT_MODEL_FILE",
    "SMS_COOLDOWN_SECONDS",
    "SMS_TRIGGER_THRESHOLD",
    "HEAT_ALERT_THRESHOLD",
    "COLD_ALERT_THRESHOLD",
    "RAIN_ALERT_THRESHOLD",
    "AQI_ALERT_THRESHOLD",
    "AUTO_ALERTS_ENABLED",
    "AUTO_ALERT_INTERVAL_SECONDS",
    "PORT",
    "LOG_LEVEL",
    "FLASK_SECRET_KEY",
    "ADMIN_EMAIL",
    "SMTP_HOST",
    "SMTP_PORT",
    "SMTP_USER",
    "SMTP_PASSWORD",
    "SMTP_FROM_EMAIL",
    "SMTP_USE_TLS",
    "EMAIL_USER",
    "EMAIL_PASS",
}

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
ENV_CANDIDATES = [
    BASE_DIR / ".env",
    PROJECT_ROOT / ".env",
]

LOADED_ENV_PATH: Path | None = None
for candidate in ENV_CANDIDATES:
    if candidate.exists():
        load_dotenv(dotenv_path=candidate, override=False)
        LOADED_ENV_PATH = candidate
        break

app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me")
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=False,
)

DEFAULT_STATE = os.getenv("DEFAULT_STATE", "kerala").strip().lower()
DEFAULT_MODEL_FILE = os.getenv("DEFAULT_MODEL_FILE", "flood_model.pkl").strip()
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "").strip().lower()

DEFAULT_MODEL_PATH = BASE_DIR / DEFAULT_MODEL_FILE
MODEL_REGISTRY = {
    "kerala": DEFAULT_MODEL_PATH,
}

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "").strip()
TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID", "").strip()
TWILIO_AUTH = os.getenv("TWILIO_AUTH_TOKEN", "").strip()
TWILIO_PHONE = os.getenv("TWILIO_PHONE", "").strip()
USER_PHONE = os.getenv("USER_PHONE", "").strip()

SMS_COOLDOWN_SECONDS_RAW = os.getenv("SMS_COOLDOWN_SECONDS", "1800").strip()
SMS_TRIGGER_THRESHOLD_RAW = os.getenv("SMS_TRIGGER_THRESHOLD", "0.7").strip()
HEAT_ALERT_THRESHOLD_RAW = os.getenv("HEAT_ALERT_THRESHOLD", "30").strip()
COLD_ALERT_THRESHOLD_RAW = os.getenv("COLD_ALERT_THRESHOLD", "0").strip()
RAIN_ALERT_THRESHOLD_RAW = os.getenv("RAIN_ALERT_THRESHOLD", "0.1").strip()
AQI_ALERT_THRESHOLD_RAW = os.getenv("AQI_ALERT_THRESHOLD", "150").strip()
ALERT_ACK_REMINDER_SECONDS_RAW = os.getenv("ALERT_ACK_REMINDER_SECONDS", "1800").strip()
ALERT_ACK_MAX_SMS_ATTEMPTS_RAW = os.getenv("ALERT_ACK_MAX_SMS_ATTEMPTS", "5").strip()
SMTP_HOST = os.getenv("SMTP_HOST", "").strip()
SMTP_PORT_RAW = os.getenv("SMTP_PORT", "587").strip()
SMTP_USER = os.getenv("SMTP_USER", "").strip()
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "").strip()
SMTP_FROM_EMAIL = os.getenv("SMTP_FROM_EMAIL", SMTP_USER).strip()
SMTP_USE_TLS = os.getenv("SMTP_USE_TLS", "true").strip().lower() in {"1", "true", "yes", "on"}
EMAIL_USER = os.getenv("EMAIL_USER", SMTP_USER).strip()
EMAIL_PASS = os.getenv("EMAIL_PASS", SMTP_PASSWORD).strip()
OTP_EXPIRY_MINUTES = 10

DB_PATH = BASE_DIR / "app.db"
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
TRAINED_MODEL_PATH = BASE_DIR / "trained_model.pkl"
USERS: list[dict[str, Any]] = []
LATEST_DATASET_PATH: Path | None = None
SIMULATION_MODE = False
SIMULATION_THRESHOLD = 0.7
SIMULATION_ALERT_COOLDOWN_SECONDS = 10
SIMULATION_MAX_ALERTS = 20
SIMULATION_STATUS: dict[str, Any] = {
    "running": False,
    "dataset": None,
    "total_rows": 0,
    "rows_processed": 0,
    "alerts_triggered": 0,
    "started_at": None,
    "completed_at": None,
    "last_alert_at": None,
    "summary": None,
    "error": None,
}
SIMULATION_THREAD: threading.Thread | None = None
SIMULATION_LOCK = threading.Lock()


def validate_env_file_format(env_path: Path) -> list[str]:
    errors: list[str] = []

    for line_number, raw_line in enumerate(env_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            errors.append(f"{env_path}:{line_number} must use KEY=VALUE format: {raw_line}")
            continue

        key, _value = line.split("=", 1)
        key = key.strip()
        if not key:
            errors.append(f"{env_path}:{line_number} has an empty environment variable name.")
            continue
        if key not in ENV_KEYS:
            logger.warning("Unknown environment variable in %s line %s: %s", env_path, line_number, key)

    return errors


def parse_int_env(name: str, raw_value: str, minimum: int | None = None) -> int:
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer. Received: {raw_value!r}") from exc
    if minimum is not None and value < minimum:
        raise RuntimeError(f"{name} must be >= {minimum}. Received: {value}")
    return value


def parse_float_env(name: str, raw_value: str, minimum: float | None = None, maximum: float | None = None) -> float:
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a float. Received: {raw_value!r}") from exc
    if minimum is not None and value < minimum:
        raise RuntimeError(f"{name} must be >= {minimum}. Received: {value}")
    if maximum is not None and value > maximum:
        raise RuntimeError(f"{name} must be <= {maximum}. Received: {value}")
    return value


AUTO_ALERTS_ENABLED_RAW = os.getenv("AUTO_ALERTS_ENABLED", "true").strip().lower()
AUTO_ALERT_INTERVAL_SECONDS_RAW = os.getenv("AUTO_ALERT_INTERVAL_SECONDS", "300").strip()
TELEGRAM_POLL_INTERVAL_SECONDS_RAW = os.getenv("TELEGRAM_POLL_INTERVAL_SECONDS", "5").strip()
AUTO_ALERTS_ENABLED = AUTO_ALERTS_ENABLED_RAW in {"1", "true", "yes", "on"}
AUTO_ALERT_INTERVAL_SECONDS = parse_int_env("AUTO_ALERT_INTERVAL_SECONDS", AUTO_ALERT_INTERVAL_SECONDS_RAW, minimum=30)
TELEGRAM_POLL_INTERVAL_SECONDS = parse_int_env("TELEGRAM_POLL_INTERVAL_SECONDS", TELEGRAM_POLL_INTERVAL_SECONDS_RAW, minimum=1)
TELEGRAM_ACK_RETENTION_HOURS = 24
AUTO_ALERTS_THREAD: threading.Thread | None = None
TELEGRAM_POLLING_THREAD: threading.Thread | None = None
TELEGRAM_POLLING_STARTED = False
AUTO_ALERTS_LOCK = threading.Lock()
TELEGRAM_POLLING_LOCK = threading.Lock()
AUTO_ALERTS_STATUS: dict[str, Any] = {
    "enabled": AUTO_ALERTS_ENABLED,
    "interval_seconds": AUTO_ALERT_INTERVAL_SECONDS,
    "running": False,
    "last_started_at": None,
    "last_completed_at": None,
    "last_trigger": None,
    "last_result": None,
    "last_error": None,
}
TELEGRAM_POLLING_STATUS: dict[str, Any] = {
    "running": False,
    "last_started_at": None,
    "last_completed_at": None,
    "last_error": None,
    "last_update_id": None,
}


def validate_phone_env(name: str, value: str) -> str:
    if not E164_PHONE_RE.match(value):
        raise RuntimeError(f"{name} must be in E.164 format, for example '+919876543210'. Received: {value!r}")
    return value


def mask_secret(value: str) -> str:
    if not value:
        return "<missing>"
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}{'*' * (len(value) - 8)}{value[-4:]}"


def log_sms_env_summary() -> None:
    logger.info(
        "Twilio env summary env_path=%s sid=%s token=%s from_phone=%s default_to=%s",
        LOADED_ENV_PATH,
        mask_secret(TWILIO_SID),
        mask_secret(TWILIO_AUTH),
        TWILIO_PHONE or "<missing>",
        USER_PHONE or "<unset>",
    )
    print(
        "TWILIO DEBUG:",
        {
            "env_path": str(LOADED_ENV_PATH) if LOADED_ENV_PATH else None,
            "account_sid": mask_secret(TWILIO_SID),
            "auth_token": mask_secret(TWILIO_AUTH),
            "twilio_phone": TWILIO_PHONE or "<missing>",
            "user_phone": USER_PHONE or "<unset>",
        },
    )


def validate_startup_config() -> None:
    errors: list[str] = []

    if LOADED_ENV_PATH is not None:
        logger.info("Loaded environment variables from %s", LOADED_ENV_PATH)
        errors.extend(validate_env_file_format(LOADED_ENV_PATH))
    else:
        logger.info("No .env file found in %s, using process environment", ENV_CANDIDATES)

    model_path = MODEL_REGISTRY.get(DEFAULT_STATE)
    if model_path is None:
        errors.append(f"No model path configured for default state '{DEFAULT_STATE}'.")
    elif not model_path.exists():
        errors.append(f"Model file does not exist: {model_path}")

    if not OPENWEATHER_API_KEY:
        errors.append("OPENWEATHER_API_KEY is missing.")
    if not TWILIO_SID:
        errors.append("TWILIO_ACCOUNT_SID is missing.")
    if not TWILIO_AUTH:
        errors.append("TWILIO_AUTH_TOKEN is missing.")
    if not TWILIO_PHONE:
        errors.append("TWILIO_PHONE is missing.")
    else:
        try:
            validate_phone_env("TWILIO_PHONE", TWILIO_PHONE)
        except RuntimeError as exc:
            errors.append(str(exc))

    try:
        parse_int_env("SMS_COOLDOWN_SECONDS", SMS_COOLDOWN_SECONDS_RAW, minimum=0)
    except RuntimeError as exc:
        errors.append(str(exc))

    try:
        parse_float_env("SMS_TRIGGER_THRESHOLD", SMS_TRIGGER_THRESHOLD_RAW, minimum=0.0, maximum=1.0)
    except RuntimeError as exc:
        errors.append(str(exc))

    for env_name, raw_value in (
        ("HEAT_ALERT_THRESHOLD", HEAT_ALERT_THRESHOLD_RAW),
        ("COLD_ALERT_THRESHOLD", COLD_ALERT_THRESHOLD_RAW),
        ("RAIN_ALERT_THRESHOLD", RAIN_ALERT_THRESHOLD_RAW),
        ("AQI_ALERT_THRESHOLD", AQI_ALERT_THRESHOLD_RAW),
    ):
        try:
            parse_float_env(env_name, raw_value)
        except RuntimeError as exc:
            errors.append(str(exc))

    try:
        parse_int_env("SMTP_PORT", SMTP_PORT_RAW, minimum=1)
    except RuntimeError as exc:
        errors.append(str(exc))

    if errors:
        logger.error("Startup validation failed.")
        for error in errors:
            logger.error("Config error: %s", error)
        raise RuntimeError("Critical startup configuration is missing. Application startup aborted.")

    logger.info("Startup config check passed")
    logger.info("Weather service configured")
    logger.info("SMS service configured")
    logger.info("Model file found at %s", model_path)


validate_startup_config()
log_sms_env_summary()

SMS_COOLDOWN_SECONDS = parse_int_env("SMS_COOLDOWN_SECONDS", SMS_COOLDOWN_SECONDS_RAW, minimum=0)
SMS_TRIGGER_THRESHOLD = parse_float_env(
    "SMS_TRIGGER_THRESHOLD",
    SMS_TRIGGER_THRESHOLD_RAW,
    minimum=0.0,
    maximum=1.0,
)
HEAT_ALERT_THRESHOLD = parse_float_env("HEAT_ALERT_THRESHOLD", HEAT_ALERT_THRESHOLD_RAW)
COLD_ALERT_THRESHOLD = parse_float_env("COLD_ALERT_THRESHOLD", COLD_ALERT_THRESHOLD_RAW)
RAIN_ALERT_THRESHOLD = parse_float_env("RAIN_ALERT_THRESHOLD", RAIN_ALERT_THRESHOLD_RAW)
AQI_ALERT_THRESHOLD = parse_float_env("AQI_ALERT_THRESHOLD", AQI_ALERT_THRESHOLD_RAW)
ALERT_ACK_REMINDER_SECONDS = parse_int_env("ALERT_ACK_REMINDER_SECONDS", ALERT_ACK_REMINDER_SECONDS_RAW, minimum=60)
ALERT_ACK_MAX_SMS_ATTEMPTS = parse_int_env("ALERT_ACK_MAX_SMS_ATTEMPTS", ALERT_ACK_MAX_SMS_ATTEMPTS_RAW, minimum=1)
SMTP_PORT = parse_int_env("SMTP_PORT", SMTP_PORT_RAW, minimum=1)

weather_client = OpenWeatherClient(api_key=OPENWEATHER_API_KEY)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

model_registry = FloodModelRegistry(model_paths=MODEL_REGISTRY, default_state=DEFAULT_STATE)
if TRAINED_MODEL_PATH.exists():
    model_registry.set_model_path(DEFAULT_STATE, TRAINED_MODEL_PATH)
feature_store = RainfallFeatureStore()
flood_predictor = FloodPredictor(model_registry=model_registry, feature_store=feature_store)

sms_service = SMSService(
    account_sid=TWILIO_SID,
    auth_token=TWILIO_AUTH,
    from_phone=TWILIO_PHONE,
    to_phone=USER_PHONE or None,
    cooldown_seconds=SMS_COOLDOWN_SECONDS,
    trigger_threshold=SMS_TRIGGER_THRESHOLD,
)


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_db_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                phone TEXT NOT NULL UNIQUE,
                password TEXT NOT NULL,
                is_admin INTEGER NOT NULL DEFAULT 0,
                is_verified INTEGER NOT NULL DEFAULT 0,
                verification_token TEXT,
                reset_token TEXT,
                reset_token_expires_at TEXT,
                telegram_chat_id TEXT,
                lat REAL NOT NULL,
                lon REAL NOT NULL,
                state TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                phone TEXT NOT NULL,
                location_label TEXT NOT NULL,
                alert_key TEXT NOT NULL UNIQUE,
                priority TEXT NOT NULL,
                alert_types TEXT NOT NULL,
                message TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                sms_attempts INTEGER NOT NULL DEFAULT 0,
                last_sms_sent_at TEXT,
                acknowledged_at TEXT,
                acknowledgement_source TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS telegram_pending_links (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                telegram_username TEXT,
                telegram_chat_id TEXT NOT NULL UNIQUE,
                raw_payload TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS telegram_poller_state (
                state_key TEXT PRIMARY KEY,
                state_value TEXT,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS telegram_acknowledgements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                chat_id TEXT NOT NULL,
                acknowledged_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_name TEXT NOT NULL,
                message TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
            """
        )
        existing_user_columns = {
            str(row["name"])
            for row in conn.execute("PRAGMA table_info(users)").fetchall()
        }
        if "reset_token" not in existing_user_columns:
            conn.execute("ALTER TABLE users ADD COLUMN reset_token TEXT")
        if "reset_token_expires_at" not in existing_user_columns:
            conn.execute("ALTER TABLE users ADD COLUMN reset_token_expires_at TEXT")
        if "telegram_chat_id" not in existing_user_columns:
            conn.execute("ALTER TABLE users ADD COLUMN telegram_chat_id TEXT")
        conn.commit()


def hash_password(password: str) -> str:
    return generate_password_hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    return check_password_hash(password_hash, password)


def user_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": int(row["id"]),
        "name": str(row["name"]),
        "email": str(row["email"]),
        "phone": str(row["phone"]),
        "is_admin": bool(row["is_admin"]),
        "is_verified": bool(row["is_verified"]),
        "telegram_chat_id": str(row["telegram_chat_id"]) if row["telegram_chat_id"] else None,
        "lat": float(row["lat"]),
        "lon": float(row["lon"]),
        "state": str(row["state"]),
    }


def get_user_by_email(email: str) -> sqlite3.Row | None:
    with get_db_connection() as conn:
        return conn.execute("SELECT * FROM users WHERE email = ?", (email.strip().lower(),)).fetchone()


def get_user_by_id(user_id: int) -> sqlite3.Row | None:
    with get_db_connection() as conn:
        return conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()


def get_user_by_reset_token(token: str) -> sqlite3.Row | None:
    with get_db_connection() as conn:
        return conn.execute("SELECT * FROM users WHERE reset_token = ?", (token,)).fetchone()


def normalize_telegram_identity(value: str | None) -> str:
    normalized = str(value or "").strip().lower()
    if normalized.startswith("@"):
        normalized = normalized[1:]
    return re.sub(r"\s+", " ", normalized)


def normalize_phone_digits(value: str | None) -> str:
    return re.sub(r"\D+", "", str(value or ""))


def extract_phone_from_telegram_start(message_text: str | None) -> str | None:
    normalized_text = str(message_text or "").strip()
    if not normalized_text:
        return None
    parts = normalized_text.split(maxsplit=1)
    if len(parts) < 2:
        return None
    return normalize_phone_digits(parts[1])


def find_user_by_phone_for_telegram_link(phone_number: str | None) -> sqlite3.Row | None:
    normalized_phone = normalize_phone_digits(phone_number)
    if len(normalized_phone) < 10:
        return None

    with get_db_connection() as conn:
        rows = conn.execute("SELECT * FROM users").fetchall()

    for row in rows:
        stored_phone = normalize_phone_digits(row["phone"])
        if not stored_phone:
            continue
        if stored_phone == normalized_phone:
            return row
        if len(normalized_phone) == 10 and stored_phone.endswith(normalized_phone):
            return row
    return None


def find_user_for_telegram_link(username: str | None, message_text: str | None = None) -> sqlite3.Row | None:
    candidate_values = [
        normalize_telegram_identity(username),
        normalize_telegram_identity(message_text),
    ]
    candidate_values = [value for value in candidate_values if value]
    if not candidate_values:
        return None

    with get_db_connection() as conn:
        rows = conn.execute("SELECT * FROM users").fetchall()

    for row in rows:
        email = str(row["email"]).strip().lower()
        email_local = email.split("@", 1)[0]
        name = normalize_telegram_identity(row["name"])
        for candidate in candidate_values:
            if candidate in {email, email_local, name}:
                return row
    return None


def store_pending_telegram_link(chat_id: str, username: str | None, payload: dict[str, Any]) -> None:
    now_iso = datetime.now(timezone.utc).isoformat()
    normalized_username = normalize_telegram_identity(username) or None
    raw_payload = json.dumps(payload)
    with get_db_connection() as conn:
        existing = conn.execute(
            "SELECT id FROM telegram_pending_links WHERE telegram_chat_id = ?",
            (chat_id,),
        ).fetchone()
        if existing is None:
            conn.execute(
                """
                INSERT INTO telegram_pending_links (
                    telegram_username, telegram_chat_id, raw_payload, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (normalized_username, chat_id, raw_payload, now_iso, now_iso),
            )
        else:
            conn.execute(
                """
                UPDATE telegram_pending_links
                SET telegram_username = ?, raw_payload = ?, updated_at = ?
                WHERE telegram_chat_id = ?
                """,
                (normalized_username, raw_payload, now_iso, chat_id),
            )
        conn.commit()


def update_user_telegram_chat_id(user_id: int, chat_id: str, username: str | None = None) -> dict[str, Any]:
    normalized_username = normalize_telegram_identity(username) or None
    with get_db_connection() as conn:
        conn.execute(
            "UPDATE users SET telegram_chat_id = ? WHERE id = ?",
            (chat_id, user_id),
        )
        conn.execute(
            "DELETE FROM telegram_pending_links WHERE telegram_chat_id = ? OR telegram_username = ?",
            (chat_id, normalized_username),
        )
        conn.commit()
    row = get_user_by_id(user_id)
    if row is None:
        raise RuntimeError("Unable to refresh linked Telegram user.")
    sync_users_from_db()
    return user_row_to_dict(row)


def link_pending_telegram_for_user(*, user_id: int, email: str, name: str) -> dict[str, Any] | None:
    email_value = normalize_telegram_identity(email)
    email_local = normalize_telegram_identity(email.split("@", 1)[0] if "@" in email else email)
    name_value = normalize_telegram_identity(name)
    with get_db_connection() as conn:
        row = conn.execute(
            """
            SELECT * FROM telegram_pending_links
            WHERE telegram_username IN (?, ?, ?)
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (email_value, email_local, name_value),
        ).fetchone()
    if row is None:
        return None
    return update_user_telegram_chat_id(
        user_id,
        str(row["telegram_chat_id"]),
        str(row["telegram_username"]) if row["telegram_username"] else None,
    )


def get_telegram_poller_state(state_key: str) -> str | None:
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT state_value FROM telegram_poller_state WHERE state_key = ?",
            (state_key,),
        ).fetchone()
    return str(row["state_value"]) if row is not None and row["state_value"] is not None else None


def set_telegram_poller_state(state_key: str, state_value: str) -> None:
    now_iso = datetime.now(timezone.utc).isoformat()
    with get_db_connection() as conn:
        existing = conn.execute(
            "SELECT state_key FROM telegram_poller_state WHERE state_key = ?",
            (state_key,),
        ).fetchone()
        if existing is None:
            conn.execute(
                "INSERT INTO telegram_poller_state (state_key, state_value, updated_at) VALUES (?, ?, ?)",
                (state_key, state_value, now_iso),
            )
        else:
            conn.execute(
                "UPDATE telegram_poller_state SET state_value = ?, updated_at = ? WHERE state_key = ?",
                (state_value, now_iso, state_key),
            )
        conn.commit()


def build_telegram_ack_reply_markup(user_id: int | None) -> dict[str, Any] | None:
    if user_id is None:
        return None
    return {
        "inline_keyboard": [
            [
                {
                    "text": "\u2705 Acknowledge",
                    "callback_data": f"ack_{user_id}",
                }
            ]
        ]
    }


def store_telegram_acknowledgement(*, user_id: int, chat_id: str, acknowledged_at: str | None = None) -> dict[str, Any]:
    ack_time = acknowledged_at or datetime.now(timezone.utc).isoformat()
    with get_db_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO telegram_acknowledgements (user_id, chat_id, acknowledged_at)
            VALUES (?, ?, ?)
            """,
            (user_id, chat_id, ack_time),
        )
        ack_id = int(cursor.lastrowid)
        conn.commit()
        row = conn.execute(
            "SELECT id, user_id, chat_id, acknowledged_at FROM telegram_acknowledgements WHERE id = ?",
            (ack_id,),
        ).fetchone()
    if row is None:
        raise RuntimeError("Unable to save Telegram acknowledgement.")
    return {
        "id": int(row["id"]),
        "user_id": int(row["user_id"]),
        "chat_id": str(row["chat_id"]),
        "acknowledged_at": str(row["acknowledged_at"]),
    }


def acknowledge_latest_alert_for_user_id(user_id: int) -> dict[str, Any] | None:
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT * FROM alerts WHERE user_id = ? AND status = 'pending' ORDER BY created_at DESC LIMIT 1",
            (user_id,),
        ).fetchone()
    if row is None:
        return None
    return acknowledge_alert(alert_id=int(row["id"]), source="telegram_button")


def list_telegram_acknowledgements(limit: int = 100) -> list[dict[str, Any]]:
    query = """
        SELECT
            telegram_acknowledgements.*,
            users.name AS user_name,
            users.phone AS user_phone
        FROM telegram_acknowledgements
        LEFT JOIN users ON users.id = telegram_acknowledgements.user_id
        ORDER BY telegram_acknowledgements.acknowledged_at DESC
        LIMIT ?
    """
    with get_db_connection() as conn:
        rows = conn.execute(query, (limit,)).fetchall()
    items: list[dict[str, Any]] = []
    for row in rows:
        items.append(
            {
                "id": int(row["id"]),
                "user_id": int(row["user_id"]),
                "chat_id": str(row["chat_id"]),
                "acknowledged_at": str(row["acknowledged_at"]),
                "user_name": str(row["user_name"]) if row["user_name"] else "Unknown user",
                "phone": str(row["user_phone"]) if row["user_phone"] else None,
            }
        )
    return items


def cleanup_expired_telegram_acknowledgements() -> int:
    cutoff_iso = (datetime.now(timezone.utc) - timedelta(hours=TELEGRAM_ACK_RETENTION_HOURS)).isoformat()
    with get_db_connection() as conn:
        cursor = conn.execute(
            "DELETE FROM telegram_acknowledgements WHERE acknowledged_at < ?",
            (cutoff_iso,),
        )
        deleted_count = int(cursor.rowcount or 0)
        conn.commit()
    if deleted_count > 0:
        logger.info(
            "Deleted expired Telegram acknowledgements count=%s retention_hours=%s cutoff=%s",
            deleted_count,
            TELEGRAM_ACK_RETENTION_HOURS,
            cutoff_iso,
        )
    return deleted_count


def serialize_message_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": int(row["id"]),
        "user_name": str(row["user_name"]),
        "message": str(row["message"]),
        "timestamp": str(row["timestamp"]),
    }


def save_community_message(user_name: str, message: str) -> dict[str, Any]:
    now_iso = datetime.now(timezone.utc).isoformat()
    with get_db_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO messages (user_name, message, timestamp)
            VALUES (?, ?, ?)
            """,
            (user_name.strip(), message.strip(), now_iso),
        )
        message_id = int(cursor.lastrowid)
        conn.commit()
        row = conn.execute("SELECT * FROM messages WHERE id = ?", (message_id,)).fetchone()
    if row is None:
        raise RuntimeError("Unable to save community message.")
    return serialize_message_row(row)


def list_community_messages(limit: int = 200) -> list[dict[str, Any]]:
    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM messages ORDER BY id ASC LIMIT ?",
            (max(1, min(limit, 500)),),
        ).fetchall()
    return [serialize_message_row(row) for row in rows]


def sync_users_from_db() -> list[dict[str, Any]]:
    global USERS
    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT id, name, email, phone, is_admin, is_verified, telegram_chat_id, lat, lon, state FROM users WHERE is_verified = 1"
        ).fetchall()
    USERS = [user_row_to_dict(row) for row in rows]
    return USERS


def require_text_field(payload: dict[str, Any], field_name: str) -> str:
    value = payload.get(field_name)
    if value is None or str(value).strip() == "":
        raise ValueError(f"{field_name} is required")
    return str(value).strip()


def require_float_field(payload: dict[str, Any], field_name: str) -> float:
    raw_value = payload.get(field_name)
    if raw_value is None or str(raw_value).strip() == "":
        raise ValueError(f"{field_name} is required")
    try:
        return float(str(raw_value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a valid number") from exc


def send_verification_email(email: str, token: str) -> str:
    verify_url = url_for("verify_email", token=token, _external=True)
    message = EmailMessage()
    message["Subject"] = "Verify your account"
    message["From"] = SMTP_FROM_EMAIL or SMTP_USER or "noreply@example.com"
    message["To"] = email
    message.set_content(
        "Click this link to verify your account:\n"
        f"{verify_url}"
    )

    if SMTP_HOST and SMTP_USER and SMTP_PASSWORD and SMTP_FROM_EMAIL:
        try:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as server:
                if SMTP_USE_TLS:
                    server.starttls()
                server.login(SMTP_USER, SMTP_PASSWORD)
                server.send_message(message)
            logger.info("Verification email sent to %s", email)
        except Exception as exc:
            logger.exception("Failed to send verification email")
            raise RuntimeError(f"Unable to send verification email: {exc}") from exc
    else:
        logger.warning("SMTP not fully configured. Falling back to terminal verification link for %s", email)

    print("VERIFICATION LINK:", verify_url)
    return verify_url


def send_password_reset_email(email: str, token: str) -> str:
    reset_url = url_for("reset_password", token=token, _external=True)
    message = EmailMessage()
    message["Subject"] = "Reset your password"
    message["From"] = SMTP_FROM_EMAIL or SMTP_USER or "noreply@example.com"
    message["To"] = email
    message.set_content(
        "Click this link to set a new password:\n"
        f"{reset_url}\n\n"
        "This link expires in 1 hour."
    )

    if SMTP_HOST and SMTP_USER and SMTP_PASSWORD and SMTP_FROM_EMAIL:
        try:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as server:
                if SMTP_USE_TLS:
                    server.starttls()
                server.login(SMTP_USER, SMTP_PASSWORD)
                server.send_message(message)
            logger.info("Password reset email sent to %s", email)
        except Exception as exc:
            logger.exception("Failed to send password reset email")
            raise RuntimeError(f"Unable to send password reset email: {exc}") from exc
    else:
        logger.warning("SMTP not fully configured. Falling back to terminal reset link for %s", email)

    print("PASSWORD RESET LINK:", reset_url)
    return reset_url


def generate_email_otp() -> str:
    return f"{random.randint(100000, 999999):06d}"


def send_signup_otp_email(email: str, otp: str) -> None:
    sender_email = EMAIL_USER or SMTP_FROM_EMAIL
    if not EMAIL_USER or not EMAIL_PASS:
        raise RuntimeError("EMAIL_USER or EMAIL_PASS is not configured.")

    message = EmailMessage()
    message["Subject"] = "Your Signup OTP"
    message["From"] = sender_email
    message["To"] = email
    message.set_content(
        "Your AI Disaster Management signup OTP is:\n"
        f"{otp}\n\n"
        f"This OTP expires in {OTP_EXPIRY_MINUTES} minutes."
    )

    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=20) as server:
            server.starttls()
            server.login(EMAIL_USER, EMAIL_PASS)
            server.send_message(message)
        logger.info("Signup OTP email sent to %s", email)
    except Exception as exc:
        logger.exception("Failed to send signup OTP email")
        raise RuntimeError(f"Unable to send OTP email: {exc}") from exc


def allowed_csv_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() == "csv"


def latest_dataset_path() -> Path | None:
    global LATEST_DATASET_PATH
    if LATEST_DATASET_PATH is not None and LATEST_DATASET_PATH.exists():
        return LATEST_DATASET_PATH
    datasets = sorted(UPLOAD_DIR.glob("*.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
    LATEST_DATASET_PATH = datasets[0] if datasets else None
    return LATEST_DATASET_PATH


def restore_default_model() -> None:
    model_registry.set_model_path(DEFAULT_STATE, DEFAULT_MODEL_PATH)


def load_dataset_frame(dataset_path: Path) -> pd.DataFrame:
    read_attempts = [
        {"sep": None, "engine": "python", "on_bad_lines": "skip"},
        {"sep": ",", "engine": "python", "on_bad_lines": "skip"},
        {"sep": ";", "engine": "python", "on_bad_lines": "skip"},
        {"sep": "\t", "engine": "python", "on_bad_lines": "skip"},
    ]

    last_error: Exception | None = None
    for options in read_attempts:
        try:
            frame = pd.read_csv(dataset_path, **options)
            frame.columns = [str(column).strip().lower() for column in frame.columns]
            return frame
        except Exception as exc:
            last_error = exc

    raise ValueError(f"Unable to read dataset: {last_error}") from last_error


def validate_dataset_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required_columns = ["rainfall", "temperature", "humidity", "flood_risk"]
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Dataset missing required columns: {', '.join(missing_columns)}")

    cleaned = frame.copy()
    if "date" in cleaned.columns:
        cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")

    for column in ["rainfall", "temperature", "humidity", "flood_risk"]:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned = cleaned.dropna(subset=["rainfall", "temperature", "humidity", "flood_risk"])
    if cleaned.empty:
        raise ValueError("Dataset contains no valid rows after cleaning.")

    return cleaned


def validate_simulation_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required_columns = ["rainfall", "temperature", "humidity"]
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Dataset missing required columns for simulation: {', '.join(missing_columns)}")

    cleaned = frame.copy()
    for column in ["rainfall", "temperature", "humidity"]:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    optional_numeric = ["flood_risk", "wind_speed", "earthquake_magnitude"]
    for column in optional_numeric:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    if "date" in cleaned.columns:
        cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")

    cleaned = cleaned.dropna(subset=["rainfall", "temperature", "humidity"])
    if cleaned.empty:
        raise ValueError("Dataset contains no valid simulation rows after cleaning.")
    return cleaned


def summarize_dataset(frame: pd.DataFrame) -> dict[str, Any]:
    summary = {
        "rows": int(len(frame)),
        "columns": frame.columns.tolist(),
        "missing_values": {column: int(value) for column, value in frame.isnull().sum().to_dict().items()},
        "average_rainfall": round(float(frame["rainfall"].mean()), 2),
        "max_temperature": round(float(frame["temperature"].max()), 2),
        "average_humidity": round(float(frame["humidity"].mean()), 2),
        "flood_frequency": round(float((frame["flood_risk"] > 0).mean()), 4),
    }
    logger.info("Dataset summary: %s", summary)
    return summary


def train_model_from_dataset(frame: pd.DataFrame) -> dict[str, Any]:
    features = frame[["rainfall", "temperature", "humidity"]]
    target = frame["flood_risk"].apply(lambda value: 1 if float(value) >= 0.5 else 0)

    stratify_target = target if target.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=stratify_target,
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test) if len(X_test) > 0 else model.predict(X_train)
    truth = y_test if len(X_test) > 0 else y_train
    accuracy = float(accuracy_score(truth, predictions))

    joblib.dump(model, TRAINED_MODEL_PATH)
    model_registry.set_model_path(DEFAULT_STATE, TRAINED_MODEL_PATH)
    logger.info("Trained model saved to %s with accuracy=%.4f", TRAINED_MODEL_PATH, accuracy)

    return {
        "model_path": str(TRAINED_MODEL_PATH),
        "model_type": type(model).__name__,
        "accuracy": round(accuracy, 4),
        "trained_rows": int(len(frame)),
        "feature_names": ["rainfall", "temperature", "humidity"],
    }


def build_historical_insights() -> dict[str, Any]:
    dataset_path = latest_dataset_path()
    if dataset_path is None:
        raise ValueError("No uploaded dataset available.")

    frame = validate_dataset_frame(load_dataset_frame(dataset_path))
    average_rainfall = round(float(frame["rainfall"].mean()), 2)
    max_temperature = round(float(frame["temperature"].max()), 2)
    flood_frequency = round(float((frame["flood_risk"] > 0).mean()), 4)

    trend_summary = "Stable pattern"
    if "date" in frame.columns and frame["date"].notna().any():
        dated = frame.dropna(subset=["date"]).sort_values("date")
        if len(dated) >= 2:
            rainfall_delta = float(dated["rainfall"].iloc[-1] - dated["rainfall"].iloc[0])
            if rainfall_delta > 0:
                trend_summary = "Rainfall trend increasing over time"
            elif rainfall_delta < 0:
                trend_summary = "Rainfall trend decreasing over time"

    return {
        "dataset": str(dataset_path),
        "average_rainfall": average_rainfall,
        "max_temperature": max_temperature,
        "flood_frequency": flood_frequency,
        "trend_summary": trend_summary,
    }


def reset_simulation_status(dataset_path: Path, total_rows: int) -> None:
    global SIMULATION_MODE
    with SIMULATION_LOCK:
        SIMULATION_MODE = True
        SIMULATION_STATUS.update(
            {
                "running": True,
                "dataset": str(dataset_path),
                "total_rows": int(total_rows),
                "rows_processed": 0,
                "alerts_triggered": 0,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "completed_at": None,
                "last_alert_at": None,
                "summary": None,
                "error": None,
            }
        )


def finalize_simulation_status(*, summary: dict[str, Any] | None = None, error: str | None = None) -> None:
    global SIMULATION_MODE
    with SIMULATION_LOCK:
        SIMULATION_MODE = False
        SIMULATION_STATUS["running"] = False
        SIMULATION_STATUS["completed_at"] = datetime.now(timezone.utc).isoformat()
        SIMULATION_STATUS["summary"] = summary
        SIMULATION_STATUS["error"] = error


def should_send_simulation_alert(
    *,
    risk_level: str,
    probability: float,
    last_risk_by_location: dict[str, str],
    location_label: str,
    alerts_sent: int,
) -> bool:
    if alerts_sent >= SIMULATION_MAX_ALERTS:
        return False
    if risk_level != "HIGH" and probability <= SIMULATION_THRESHOLD:
        return False

    now = time.time()
    with SIMULATION_LOCK:
        last_alert_at = SIMULATION_STATUS["last_alert_at"]
        cooldown_ok = last_alert_at is None or (now - float(last_alert_at)) >= SIMULATION_ALERT_COOLDOWN_SECONDS
    if not cooldown_ok:
        return False

    if last_risk_by_location.get(location_label) == risk_level:
        return False
    return True


def build_simulation_alerts(row: pd.Series, flood_probability: float, risk_level: str) -> list[tuple[str, str]]:
    alerts: list[tuple[str, str]] = []
    location_label = str(row.get("location", DEFAULT_STATE)).strip() or DEFAULT_STATE

    if risk_level == "HIGH" or flood_probability > SIMULATION_THRESHOLD:
        alerts.append(("FLOOD", f"Flood risk is {risk_level} at {location_label}."))
    if float(row.get("temperature", 0.0)) >= 32:
        alerts.append(("HEAT", "Heatwave conditions detected. Stay indoors and hydrate."))
    elif float(row.get("temperature", 0.0)) >= 28:
        alerts.append(("WARM", "Warm conditions detected. Stay hydrated."))
    if float(row.get("rainfall", 0.0)) > 0:
        alerts.append(("RAIN", "Heavy rain signal detected."))
    if "wind_speed" in row.index and pd.notna(row.get("wind_speed")) and float(row.get("wind_speed", 0.0)) >= 60:
        alerts.append(("CYCLONE", "Cyclone-like wind conditions detected."))
    if "earthquake_magnitude" in row.index and pd.notna(row.get("earthquake_magnitude")) and float(row.get("earthquake_magnitude", 0.0)) >= 4.5:
        alerts.append(("EARTHQUAKE", "Earthquake activity detected in historical dataset."))
    return alerts


def build_simulation_sms(location_label: str, dataset_date: str, risk_level: str, alerts: list[tuple[str, str]]) -> str:
    lines = [
        "\U000026A0\U0000FE0F ALERT (Simulation)",
        f"Date: {dataset_date}",
        f"Location: {location_label}",
        f"Flood risk: {risk_level}",
    ]
    lines.extend(message for _, message in alerts)
    lines.append("Stay alert!")
    return "\n".join(lines)


def simulate_dataset_alerts(file_path: Path) -> None:
    try:
        frame = validate_simulation_frame(load_dataset_frame(file_path))
        reset_simulation_status(file_path, len(frame))
        sync_users_from_db()

        last_risk_by_location: dict[str, str] = {}
        alerts_sent = 0
        processed_rows = 0

        for index, row in frame.iterrows():
            location_label = str(row.get("location", DEFAULT_STATE)).strip() or DEFAULT_STATE
            state = str(row.get("state", DEFAULT_STATE)).strip().lower() or DEFAULT_STATE
            dataset_date = str(row.get("date", "")) if pd.notna(row.get("date")) else "Unknown"
            location_key = f"simulation:{location_label}:{index}"

            prediction = flood_predictor.predict(
                state=state,
                location_key=location_key,
                rainfall=float(row["rainfall"]),
                temperature=float(row["temperature"]),
                humidity=float(row["humidity"]),
            )
            probability = float(prediction["flood_probability"])
            risk_level = str(prediction["flood_prediction"])
            alerts = build_simulation_alerts(row, probability, risk_level)

            if should_send_simulation_alert(
                risk_level=risk_level,
                probability=probability,
                last_risk_by_location=last_risk_by_location,
                location_label=location_label,
                alerts_sent=alerts_sent,
            ):
                sms_message = build_simulation_sms(location_label, dataset_date, risk_level, alerts)
                telegram_message = build_telegram_alert(
                    location_label,
                    " / ".join(alert_type for alert_type, _ in alerts) or risk_level,
                    sms_message,
                )
                for user in USERS:
                    print("Sending SMS to:", user["phone"])
                    sms_service.send_custom_sms(
                        sms_message,
                        to_phone=str(user["phone"]),
                        priority="HIGH" if risk_level == "HIGH" else "MEDIUM",
                        recipient_key=f"simulation:{user['phone']}:{location_label}",
                    )
                    if not should_send_telegram_for_location(user, state):
                        logger.info(
                            "Skipping user %s due to location mismatch user_location=%s alert_location=%s",
                            user.get("id", user.get("phone")),
                            normalize_alert_location(str(user.get("state", "")).strip(), fallback=DEFAULT_STATE),
                            normalize_alert_location(state, fallback=DEFAULT_STATE),
                        )
                        continue
                    logger.info(
                        "Sending alert to %s for location %s",
                        user.get("id", user.get("phone")),
                        normalize_alert_location(state, fallback=DEFAULT_STATE),
                    )
                    personalized_telegram_message = personalize_telegram_message(telegram_message, str(user.get("name", "User")))
                    telegram_result = send_telegram_to_user(personalized_telegram_message, user)
                    logger.info(
                        "Telegram result for simulation user=%s phone=%s result=%s",
                        user.get("name"),
                        user["phone"],
                        telegram_result,
                    )
                alerts_sent += 1
                last_risk_by_location[location_label] = risk_level
                with SIMULATION_LOCK:
                    SIMULATION_STATUS["alerts_triggered"] = alerts_sent
                    SIMULATION_STATUS["last_alert_at"] = time.time()

            processed_rows += 1
            with SIMULATION_LOCK:
                SIMULATION_STATUS["rows_processed"] = processed_rows

            time.sleep(1)

        finalize_simulation_status(
            summary={
                "mode": "simulation",
                "total_rows_processed": processed_rows,
                "alerts_triggered": alerts_sent,
                "dataset": str(file_path),
            }
        )
    except Exception as exc:
        logger.exception("Dataset simulation failed")
        finalize_simulation_status(error=str(exc))


def current_user() -> dict[str, Any] | None:
    user_id = session.get("user_id")
    if not user_id:
        return None
    row = get_user_by_id(int(user_id))
    if row is None:
        session.clear()
        return None
    return user_row_to_dict(row)


@app.context_processor
def inject_template_globals() -> dict[str, Any]:
    user = current_user()
    return {
        "nav_user": user,
        "nav_is_authenticated": user is not None,
    }


def wants_json_response() -> bool:
    return request.path.startswith(
        (
            "/predict-live",
            "/process-group-alerts",
            "/users",
            "/add-user",
            "/current-user",
            "/upload-dataset",
            "/remove-dataset",
            "/historical-insights",
            "/start-simulation",
            "/simulation-status",
            "/delete-user",
        )
    )


def login_required(view: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(view)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        user = current_user()
        if user is None:
            if wants_json_response():
                return jsonify({"success": False, "error": "Authentication required"}), 401
            return redirect(url_for("login_page"))
        return view(*args, **kwargs)

    return wrapped


def admin_required(view: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(view)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        user = current_user()
        if user is None:
            if wants_json_response():
                return jsonify({"success": False, "error": "Authentication required"}), 401
            return redirect(url_for("login_page"))
        if not user["is_admin"]:
            if wants_json_response():
                return jsonify({"success": False, "error": "Admin access required"}), 403
            return redirect(url_for("dashboard"))
        return view(*args, **kwargs)

    return wrapped


def dependency_status() -> dict[str, object]:
    model_ok = False
    model_info: dict[str, object] | None = None
    model_error: str | None = None

    try:
        model_info = model_registry.get_model_info(DEFAULT_STATE)
        model_ok = True
    except PredictionError as exc:
        model_error = str(exc)

    weather_ok = bool(OPENWEATHER_API_KEY)
    sms_ok = all([TWILIO_SID, TWILIO_AUTH, TWILIO_PHONE])

    overall_ok = model_ok and weather_ok and sms_ok
    return {
        "status": "ok" if overall_ok else "error",
        "default_state": DEFAULT_STATE,
        "user_count": len(sync_users_from_db()),
        "services": {
            "model": {"ok": model_ok, "details": model_info, "error": model_error},
            "weather": {"ok": weather_ok, "configured": weather_ok},
            "sms": {
                "ok": sms_ok,
                "configured": sms_ok,
                "threshold": SMS_TRIGGER_THRESHOLD,
                "cooldown_seconds": SMS_COOLDOWN_SECONDS,
            },
        },
    }


def build_location_label(city: str, lat: float, lon: float) -> str:
    normalized_city = str(city or "").strip()
    if normalized_city:
        return normalized_city
    return f"registered area near {format_coordinate(lat, is_latitude=True)}, {format_coordinate(lon, is_latitude=False)}"


def format_coordinate(value: float, *, is_latitude: bool) -> str:
    direction = "N" if is_latitude else "E"
    if value < 0:
        direction = "S" if is_latitude else "W"
    return f"{abs(value):.3f}{direction}"


def humanize_location_label(location_label: str) -> str:
    normalized = str(location_label or "").strip()
    if not normalized:
        return "your registered area"
    if COORDINATE_LOCATION_RE.match(normalized):
        lat_text, lon_text = normalized.split(",", 1)
        try:
            return f"registered area near {format_coordinate(float(lat_text), is_latitude=True)}, {format_coordinate(float(lon_text), is_latitude=False)}"
        except ValueError:
            return normalized
    return normalized


def build_alerts(
    *,
    location_label: str,
    flood_probability: float,
    rainfall: float,
    temperature: float,
    aqi_value: float,
    flood_threshold: float,
) -> list[tuple[str, str]]:
    alerts: list[tuple[str, str]] = []
    if flood_probability >= flood_threshold:
        alerts.append(("FLOOD", f"Flood risk is elevated at {location_label}. Move to a safer area if needed."))
    if temperature <= COLD_ALERT_THRESHOLD:
        alerts.append(("COLD", "Hey, wear your sweaters and stay warm."))
    if temperature >= HEAT_ALERT_THRESHOLD:
        alerts.append(("HEAT", "Heat alert. Stay indoors and hydrate frequently."))
    if rainfall >= RAIN_ALERT_THRESHOLD:
        alerts.append(("RAIN", "Rain expected. Carry an umbrella."))
    if aqi_value >= AQI_ALERT_THRESHOLD:
        alerts.append(("AQI", "Unhealthy air, wear mask."))
    return alerts


def determine_priority(alerts: list[tuple[str, str]]) -> str | None:
    if not alerts:
        return None
    alert_types = {alert_type for alert_type, _ in alerts}
    if "FLOOD" in alert_types or "AQI" in alert_types:
        return "HIGH"
    if "HEAT" in alert_types or "RAIN" in alert_types or "COLD" in alert_types:
        return "MEDIUM"
    return "LOW"


def build_smart_action_lines(alert_types: list[str], priority: str) -> list[str]:
    actions: list[str] = []
    alert_set = set(alert_types)
    if "FLOOD" in alert_set:
        actions.append("Move valuables up, avoid low-lying roads, and keep your phone charged.")
    if "HEAT" in alert_set:
        actions.append("Limit outdoor activity, drink water, and check on elderly family members.")
    if "RAIN" in alert_set and "FLOOD" not in alert_set:
        actions.append("Keep an umbrella ready and avoid waterlogged streets.")
    if "AQI" in alert_set:
        actions.append("Use a mask outside and keep windows closed if possible.")
    if "COLD" in alert_set:
        actions.append("Wear warm layers and reduce long exposure outdoors.")
    if not actions:
        actions.append("Stay alert and follow local safety guidance.")
    if priority == "HIGH":
        actions.append("Reply OK, SAFE, or RECEIVED once you have seen this alert.")
    else:
        actions.append("Reply OK once you have seen this message.")
    return actions[:3]


def build_alert_sms(user_name: str, location_label: str, alerts: list[tuple[str, str]], priority: str) -> str:
    safe_location_label = humanize_location_label(location_label)
    alert_types = [alert_type for alert_type, _ in alerts]
    headline = "Critical alert" if priority == "HIGH" else "Safety alert"
    lines = [
        f"\U0001F6A8 {headline} for {user_name}",
        f"\U0001F4CD Location: {safe_location_label}",
        f"\U0001F4CA Severity: {priority}",
        "",
    ]
    for index, (_, message) in enumerate(alerts, start=1):
        lines.append(f"{index}. {message}")
    lines.append("")
    lines.append("Recommended action:")
    for action in build_smart_action_lines(alert_types, priority):
        lines.append(f"- {action}")
    return "\n".join(lines)


def build_telegram_alert(location_label: str, alert_type: str, details: str) -> str:
    normalized_location = humanize_location_label(location_label)
    normalized_type = str(alert_type or "GENERAL").strip() or "GENERAL"
    normalized_details = str(details or "").strip() or "Stay alert and monitor updates."
    return "\n".join(
        [
            "\U0001F6A8 ALERT",
            f"Location: {normalized_location}",
            f"Type: {normalized_type}",
            f"Details: {normalized_details}",
        ]
    )


def personalize_telegram_message(message: str, user_name: str | None) -> str:
    normalized_message = str(message or "")
    normalized_name = str(user_name or "").strip() or "User"
    personalized = re.sub(
        r"(Critical alert for )([^\n]+)",
        lambda match: f"{match.group(1)}{normalized_name}",
        normalized_message,
        count=1,
    )
    personalized = re.sub(
        r"(Safety alert for )([^\n]+)",
        lambda match: f"{match.group(1)}{normalized_name}",
        personalized,
        count=1,
    )
    return personalized


def send_telegram_to_user(message: str, user: dict[str, Any] | None, *, fallback: bool = False) -> dict[str, Any]:
    chat_id = str(user.get("telegram_chat_id", "")).strip() if user else ""
    reply_markup = build_telegram_ack_reply_markup(int(user["id"])) if user and user.get("id") is not None else None
    logger.info(
        "Preparing Telegram send user_id=%s email=%s stored_chat_id=%s fallback=%s",
        user.get("id") if user else None,
        user.get("email") if user else None,
        chat_id or None,
        fallback,
    )
    if chat_id:
        return send_telegram(message, chat_id=chat_id, reply_markup=reply_markup)
    if fallback:
        logger.warning(
            "Telegram fallback send is using TELEGRAM_CHAT_ID from environment because user_id=%s has no linked chat_id",
            user.get("id") if user else None,
        )
        return send_telegram(message, reply_markup=reply_markup)
    logger.info("Telegram skipped: no linked chat_id for user=%s", user.get("email") if user else None)
    return {
        "sent": False,
        "reason": "No Telegram chat linked",
        "error": None,
        "chat_id": None,
    }


def handle_telegram_start_message(chat_id: str, username: str | None, message_text: str | None, update: dict[str, Any]) -> dict[str, Any]:
    logger.info("Telegram chat_id received chat_id=%s username=%s message=%s", chat_id, username, message_text)

    phone_number = extract_phone_from_telegram_start(message_text)
    if phone_number is None:
        logger.info("Telegram link skipped: phone number missing chat_id=%s username=%s", chat_id, username)
        prompt_result = send_telegram(
            "Please send /start followed by your registered phone number. Example: /start 9372631037",
            chat_id=chat_id,
        )
        return {
            "linked": False,
            "reason": "Phone number missing",
            "chat_id": chat_id,
            "username": username,
            "telegram_result": prompt_result,
        }

    if len(phone_number) < 10:
        logger.info("Telegram link skipped: invalid phone number chat_id=%s username=%s phone=%s", chat_id, username, phone_number)
        prompt_result = send_telegram(
            "Invalid phone number. Please send /start followed by your registered phone number. Example: /start 9372631037",
            chat_id=chat_id,
        )
        return {
            "linked": False,
            "reason": "Phone number invalid",
            "chat_id": chat_id,
            "username": username,
            "phone_number": phone_number,
            "telegram_result": prompt_result,
        }

    user_row = find_user_by_phone_for_telegram_link(phone_number)
    if user_row is None:
        store_pending_telegram_link(chat_id, username, update)
        logger.info("Telegram link skipped: no user found for phone=%s chat_id=%s username=%s", phone_number, chat_id, username)
        prompt_result = send_telegram(
            "No account was found for that phone number. Please use the phone number registered in the Disaster Management System.",
            chat_id=chat_id,
        )
        return {
            "linked": False,
            "reason": "User not found for phone number",
            "chat_id": chat_id,
            "username": username,
            "phone_number": phone_number,
            "telegram_result": prompt_result,
        }

    linked_user = update_user_telegram_chat_id(int(user_row["id"]), chat_id, username)
    confirmation_result = send_telegram("Telegram connected successfully \u2705", chat_id=chat_id)
    latest_alert = get_latest_alert()
    latest_alert_result: dict[str, Any] | None = None
    if latest_alert:
        latest_alert_result = send_telegram(str(latest_alert["message"]), chat_id=chat_id)
        logger.info(
            "Sent latest alert to newly linked Telegram user_id=%s chat_id=%s alert_id=%s",
            linked_user["id"],
            chat_id,
            latest_alert["id"],
        )
    logger.info(
        "Linked Telegram chat_id=%s to user_id=%s phone=%s email=%s username=%s stored_chat_id=%s",
        chat_id,
        linked_user["id"],
        linked_user["phone"],
        linked_user["email"],
        username,
        linked_user.get("telegram_chat_id"),
    )
    return {
        "linked": True,
        "user_id": linked_user["id"],
        "email": linked_user["email"],
        "phone": linked_user["phone"],
        "chat_id": chat_id,
        "telegram_result": confirmation_result,
        "latest_alert_result": latest_alert_result,
    }


def process_telegram_update(update: dict[str, Any]) -> None:
    callback_query = update.get("callback_query")
    if callback_query:
        callback_data = str(callback_query.get("data", "")).strip()
        callback_message = callback_query.get("message") or {}
        callback_chat = callback_message.get("chat") or {}
        chat_id_raw = callback_chat.get("id")
        chat_id = str(chat_id_raw).strip() if chat_id_raw is not None else ""
        if callback_data.startswith("ack_"):
            user_id_raw = callback_data.removeprefix("ack_").strip()
            try:
                user_id = int(user_id_raw)
            except ValueError:
                logger.warning("Telegram acknowledgement ignored due to invalid callback data=%s", callback_data)
                return
            acknowledged_alert = acknowledge_latest_alert_for_user_id(user_id)
            acknowledgement = store_telegram_acknowledgement(user_id=user_id, chat_id=chat_id)
            logger.info(
                "Acknowledgement received user_id=%s timestamp=%s",
                user_id,
                acknowledgement["acknowledged_at"],
            )
            if chat_id:
                send_telegram("\u2705 Alert acknowledged. Stay safe!", chat_id=chat_id)
            if acknowledged_alert is None:
                logger.info("Telegram acknowledgement stored with no pending alert found for user_id=%s", user_id)
        return

    message = update.get("message") or update.get("edited_message") or {}
    if not message:
        return

    message_text = str(message.get("text", "")).strip()
    if not message_text.startswith("/start"):
        return

    chat = message.get("chat") or {}
    sender = message.get("from") or {}
    chat_id_raw = chat.get("id")
    if chat_id_raw is None:
        return

    chat_id = str(chat_id_raw).strip()
    username = str(sender.get("username", "")).strip() or None
    logger.info("Telegram /start received chat_id=%s username=%s text=%s", chat_id, username, message_text)
    handle_telegram_start_message(chat_id, username, message_text, update)


def telegram_polling_loop() -> None:
    state_key = "last_update_id"
    raw_last_update_id = get_telegram_poller_state(state_key)
    last_update_id = int(raw_last_update_id) if raw_last_update_id and raw_last_update_id.isdigit() else None

    TELEGRAM_POLLING_STATUS["last_started_at"] = datetime.now(timezone.utc).isoformat()
    TELEGRAM_POLLING_STATUS["running"] = True
    TELEGRAM_POLLING_STATUS["last_error"] = None
    TELEGRAM_POLLING_STATUS["last_update_id"] = last_update_id

    logger.info(
        "Starting Telegram polling listener interval=%ss last_update_id=%s",
        TELEGRAM_POLL_INTERVAL_SECONDS,
        last_update_id,
    )

    while True:
        try:
            cleanup_expired_telegram_acknowledgements()
            response = fetch_telegram_updates(
                offset=(last_update_id + 1) if last_update_id is not None else None,
                timeout=max(1, TELEGRAM_POLL_INTERVAL_SECONDS),
            )
            if not response.get("ok"):
                TELEGRAM_POLLING_STATUS["last_error"] = str(response.get("description") or "Telegram polling failed")
                time.sleep(TELEGRAM_POLL_INTERVAL_SECONDS)
                continue

            updates = response.get("result", [])
            for update in updates:
                update_id = update.get("update_id")
                if update_id is None:
                    continue
                process_telegram_update(update)
                last_update_id = int(update_id)
                set_telegram_poller_state(state_key, str(last_update_id))
                TELEGRAM_POLLING_STATUS["last_update_id"] = last_update_id

            TELEGRAM_POLLING_STATUS["last_completed_at"] = datetime.now(timezone.utc).isoformat()
            TELEGRAM_POLLING_STATUS["last_error"] = None
            if not updates:
                time.sleep(TELEGRAM_POLL_INTERVAL_SECONDS)
        except Exception as exc:
            logger.exception("Telegram polling loop failed")
            TELEGRAM_POLLING_STATUS["last_error"] = str(exc)
            time.sleep(TELEGRAM_POLL_INTERVAL_SECONDS)


def start_telegram_polling_listener() -> None:
    global TELEGRAM_POLLING_STARTED, TELEGRAM_POLLING_THREAD
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        logger.info("Telegram polling listener not started because TELEGRAM_BOT_TOKEN is missing")
        return
    with TELEGRAM_POLLING_LOCK:
        if TELEGRAM_POLLING_STARTED:
            logger.info(
                "Telegram polling listener start skipped because it is already running pid=%s thread=%s",
                os.getpid(),
                TELEGRAM_POLLING_THREAD.name if TELEGRAM_POLLING_THREAD is not None else None,
            )
            return
        if TELEGRAM_POLLING_THREAD is not None and TELEGRAM_POLLING_THREAD.is_alive():
            TELEGRAM_POLLING_STARTED = True
            logger.info(
                "Telegram polling listener already active pid=%s thread=%s",
                os.getpid(),
                TELEGRAM_POLLING_THREAD.name,
            )
            return
        TELEGRAM_POLLING_THREAD = threading.Thread(target=telegram_polling_loop, name="telegram-polling", daemon=True)
        TELEGRAM_POLLING_STARTED = True
        logger.info(
            "Starting Telegram polling listener once pid=%s thread=%s",
            os.getpid(),
            TELEGRAM_POLLING_THREAD.name,
        )
        TELEGRAM_POLLING_THREAD.start()


def is_acknowledgement_message(message_body: str) -> bool:
    normalized_body = re.sub(r"[^A-Z0-9\s]", " ", message_body.strip().upper())
    normalized_body = re.sub(r"\s+", " ", normalized_body).strip()
    if not normalized_body:
        return False

    direct_matches = {
        "OK",
        "OKEY",
        "OKAY",
        "K",
        "YES",
        "SEEN",
        "SAFE",
        "RECEIVED",
        "NOTED",
        "ACK",
        "ACKNOWLEDGED",
    }
    if normalized_body in direct_matches:
        return True

    acknowledgement_phrases = (
        "I AM SAFE",
        "IM SAFE",
        "MESSAGE RECEIVED",
        "ALERT RECEIVED",
        "I GOT IT",
        "GOT IT",
        "SEEN IT",
        "I HAVE SEEN THIS",
        "RECEIVED THIS",
        "NOTED THANKS",
    )
    return any(phrase in normalized_body for phrase in acknowledgement_phrases)


def validate_signup_payload(payload: dict[str, Any]) -> dict[str, Any]:
    name = require_text_field(payload, "name")
    email = require_text_field(payload, "email").lower()
    phone = require_text_field(payload, "phone")
    password = require_text_field(payload, "password")
    state = require_text_field(payload, "state").lower()
    lat = require_float_field(payload, "lat")
    lon = require_float_field(payload, "lon")

    if not EMAIL_RE.match(email):
        raise ValueError("email must be a valid email address")
    try:
        validate_phone_env("phone", phone)
    except RuntimeError as exc:
        raise ValueError(str(exc)) from exc
    if len(password) < 8:
        raise ValueError("password must be at least 8 characters")

    return {
        "name": name,
        "email": email,
        "phone": phone,
        "password": password,
        "lat": lat,
        "lon": lon,
        "state": state,
    }


def validate_admin_user_payload(payload: dict[str, Any]) -> dict[str, Any]:
    data = validate_signup_payload(payload)
    data["is_verified"] = bool(payload.get("is_verified", True))
    return data


def validate_new_password(password: str) -> str:
    normalized = str(password or "")
    if len(normalized) < 8:
        raise ValueError("password must be at least 8 characters")
    return normalized


def serialize_user(user: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": int(user["id"]),
        "name": str(user["name"]),
        "email": str(user["email"]),
        "phone": str(user["phone"]),
        "is_admin": bool(user["is_admin"]),
        "is_verified": bool(user["is_verified"]),
        "telegram_chat_id": str(user["telegram_chat_id"]) if user.get("telegram_chat_id") else None,
        "lat": float(user["lat"]),
        "lon": float(user["lon"]),
        "state": str(user["state"]),
    }


def build_alert_key(phone: str, location_label: str, alerts: list[tuple[str, str]]) -> str:
    normalized_phone = phone.strip()
    normalized_location = location_label.strip().lower()
    types_key = ",".join(sorted(alert_type for alert_type, _ in alerts))
    return f"{normalized_phone}|{normalized_location}|{types_key}"


def serialize_alert_row(row: sqlite3.Row) -> dict[str, Any]:
    alert_types_raw = str(row["alert_types"])
    try:
        alert_types = json.loads(alert_types_raw)
    except json.JSONDecodeError:
        alert_types = []
    return {
        "id": int(row["id"]),
        "user_id": int(row["user_id"]) if row["user_id"] is not None else None,
        "phone": str(row["phone"]),
        "location_label": str(row["location_label"]),
        "alert_key": str(row["alert_key"]),
        "priority": str(row["priority"]),
        "alert_types": alert_types,
        "message": str(row["message"]),
        "status": str(row["status"]),
        "sms_attempts": int(row["sms_attempts"]),
        "last_sms_sent_at": str(row["last_sms_sent_at"]) if row["last_sms_sent_at"] else None,
        "acknowledged_at": str(row["acknowledged_at"]) if row["acknowledged_at"] else None,
        "acknowledgement_source": str(row["acknowledgement_source"]) if row["acknowledgement_source"] else None,
        "created_at": str(row["created_at"]),
        "updated_at": str(row["updated_at"]),
    }


def get_latest_alert() -> dict[str, Any] | None:
    with get_db_connection() as conn:
        row = conn.execute("SELECT * FROM alerts ORDER BY created_at DESC LIMIT 1").fetchone()
    return serialize_alert_row(row) if row is not None else None


def upsert_pending_alert(
    *,
    user_id: int | None,
    phone: str,
    location_label: str,
    alerts: list[tuple[str, str]],
    priority: str,
    message: str,
) -> dict[str, Any]:
    now_iso = datetime.now(timezone.utc).isoformat()
    alert_key = build_alert_key(phone, location_label, alerts)
    alert_types = [alert_type for alert_type, _ in alerts]
    with get_db_connection() as conn:
        row = conn.execute("SELECT * FROM alerts WHERE alert_key = ?", (alert_key,)).fetchone()
        if row is None:
            cursor = conn.execute(
                """
                INSERT INTO alerts (
                    user_id, phone, location_label, alert_key, priority, alert_types, message,
                    status, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)
                """,
                (
                    user_id,
                    phone,
                    location_label,
                    alert_key,
                    priority,
                    json.dumps(alert_types),
                    message,
                    now_iso,
                    now_iso,
                ),
            )
            alert_id = int(cursor.lastrowid)
        else:
            alert_id = int(row["id"])
            reset_ack_state = str(row["status"]) == "acknowledged"
            conn.execute(
                """
                UPDATE alerts
                SET user_id = ?, priority = ?, alert_types = ?, message = ?, updated_at = ?,
                    status = 'pending',
                    sms_attempts = CASE WHEN ? THEN 0 ELSE sms_attempts END,
                    last_sms_sent_at = CASE WHEN ? THEN NULL ELSE last_sms_sent_at END,
                    acknowledged_at = CASE WHEN ? THEN NULL ELSE acknowledged_at END,
                    acknowledgement_source = CASE WHEN ? THEN NULL ELSE acknowledgement_source END
                WHERE id = ?
                """,
                (
                    user_id,
                    priority,
                    json.dumps(alert_types),
                    message,
                    now_iso,
                    int(reset_ack_state),
                    int(reset_ack_state),
                    int(reset_ack_state),
                    int(reset_ack_state),
                    alert_id,
                ),
            )
        conn.commit()
        saved_row = conn.execute("SELECT * FROM alerts WHERE id = ?", (alert_id,)).fetchone()
    if saved_row is None:
        raise RuntimeError("Unable to save alert record.")
    return serialize_alert_row(saved_row)


def record_alert_sms_attempt(alert_id: int, sent: bool) -> dict[str, Any]:
    now_iso = datetime.now(timezone.utc).isoformat()
    with get_db_connection() as conn:
        conn.execute(
            """
            UPDATE alerts
            SET sms_attempts = sms_attempts + ?, last_sms_sent_at = ?, updated_at = ?
            WHERE id = ?
            """,
            (1 if sent else 0, now_iso if sent else None, now_iso, alert_id),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM alerts WHERE id = ?", (alert_id,)).fetchone()
    if row is None:
        raise RuntimeError("Unable to refresh alert record.")
    return serialize_alert_row(row)


def list_pending_alerts(*, phone: str | None = None, user_id: int | None = None) -> list[dict[str, Any]]:
    query = "SELECT * FROM alerts WHERE status = 'pending'"
    params: list[Any] = []
    if phone:
        query += " AND phone = ?"
        params.append(phone)
    if user_id is not None:
        query += " AND user_id = ?"
        params.append(user_id)
    query += " ORDER BY created_at DESC"
    with get_db_connection() as conn:
        rows = conn.execute(query, tuple(params)).fetchall()
    return [serialize_alert_row(row) for row in rows]


def normalize_alert_location(value: str | None, *, fallback: str | None = None) -> str:
    normalized = str(value or "").strip().lower()
    if normalized:
        return normalized
    return str(fallback or "").strip().lower()


def should_send_telegram_for_location(user: dict[str, Any] | None, alert_location: str | None) -> bool:
    if not user:
        return False
    user_location = normalize_alert_location(str(user.get("state", "")).strip(), fallback=DEFAULT_STATE)
    target_location = normalize_alert_location(alert_location, fallback=DEFAULT_STATE)
    return bool(user_location) and bool(target_location) and user_location == target_location


def list_alert_overview(limit: int = 100) -> list[dict[str, Any]]:
    query = """
        SELECT
            alerts.*,
            users.name AS user_name,
            users.email AS user_email,
            users.state AS user_state
        FROM alerts
        LEFT JOIN users ON users.id = alerts.user_id
        WHERE alerts.status = 'pending'
        ORDER BY alerts.updated_at DESC, alerts.created_at DESC
        LIMIT ?
    """
    with get_db_connection() as conn:
        rows = conn.execute(query, (limit,)).fetchall()

    overview: list[dict[str, Any]] = []
    for row in rows:
        item = serialize_alert_row(row)
        item["user_name"] = str(row["user_name"]) if row["user_name"] else "Unknown user"
        item["user_email"] = str(row["user_email"]) if row["user_email"] else None
        item["user_state"] = str(row["user_state"]) if row["user_state"] else None
        overview.append(item)
    return overview


def acknowledge_alert(*, alert_id: int, source: str) -> dict[str, Any] | None:
    now_iso = datetime.now(timezone.utc).isoformat()
    with get_db_connection() as conn:
        conn.execute(
            """
            UPDATE alerts
            SET status = 'acknowledged', acknowledged_at = ?, acknowledgement_source = ?, updated_at = ?
            WHERE id = ? AND status = 'pending'
            """,
            (now_iso, source, now_iso, alert_id),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM alerts WHERE id = ?", (alert_id,)).fetchone()
    return serialize_alert_row(row) if row is not None else None


def acknowledge_latest_alert_for_phone(phone: str, message_body: str) -> dict[str, Any] | None:
    if not is_acknowledgement_message(message_body):
        return None
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT * FROM alerts WHERE phone = ? AND status = 'pending' ORDER BY created_at DESC LIMIT 1",
            (phone,),
        ).fetchone()
    if row is None:
        return None
    return acknowledge_alert(alert_id=int(row["id"]), source="sms_reply")


def should_send_alert_reminder(alert_record: dict[str, Any]) -> bool:
    if alert_record["status"] != "pending":
        logger.info(
            "Alert reminder stopped for alert_id=%s phone=%s because it was acknowledged at %s",
            alert_record["id"],
            alert_record["phone"],
            alert_record["acknowledged_at"],
        )
        return False
    if alert_record["sms_attempts"] >= ALERT_ACK_MAX_SMS_ATTEMPTS:
        logger.info(
            "Alert reminder stopped for alert_id=%s phone=%s because max attempts=%s was reached",
            alert_record["id"],
            alert_record["phone"],
            ALERT_ACK_MAX_SMS_ATTEMPTS,
        )
        return False
    last_sent_at = alert_record["last_sms_sent_at"]
    if not last_sent_at:
        logger.info(
            "Alert reminder allowed for alert_id=%s phone=%s because no previous SMS was sent",
            alert_record["id"],
            alert_record["phone"],
        )
        return True
    try:
        last_sent_dt = datetime.fromisoformat(last_sent_at)
    except ValueError:
        logger.warning(
            "Alert reminder allowed for alert_id=%s phone=%s because last_sms_sent_at=%r was invalid",
            alert_record["id"],
            alert_record["phone"],
            last_sent_at,
        )
        return True
    elapsed_seconds = (datetime.now(timezone.utc) - last_sent_dt).total_seconds()
    if elapsed_seconds >= AUTO_ALERT_INTERVAL_SECONDS:
        logger.info(
            "Alert reminder allowed for alert_id=%s phone=%s after %.1fs elapsed (interval=%ss)",
            alert_record["id"],
            alert_record["phone"],
            elapsed_seconds,
            AUTO_ALERT_INTERVAL_SECONDS,
        )
        return True
    logger.info(
        "Alert reminder waiting for interval for alert_id=%s phone=%s elapsed=%.1fs interval=%ss",
        alert_record["id"],
        alert_record["phone"],
        elapsed_seconds,
        AUTO_ALERT_INTERVAL_SECONDS,
    )
    return False


def create_user(
    *,
    name: str,
    email: str,
    phone: str,
    password: str,
    lat: float,
    lon: float,
    state: str,
    is_admin: bool,
    is_verified: bool,
    verification_token: str | None,
) -> dict[str, Any]:
    with get_db_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO users (name, email, phone, password, is_admin, is_verified, verification_token, lat, lon, state, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                name,
                email,
                phone,
                hash_password(password),
                int(is_admin),
                int(is_verified),
                verification_token,
                lat,
                lon,
                state,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
        user_id = int(cursor.lastrowid)
    row = get_user_by_id(user_id)
    if row is None:
        raise RuntimeError("User creation failed.")
    linked_user = link_pending_telegram_for_user(user_id=user_id, email=email, name=name)
    sync_users_from_db()
    return linked_user or user_row_to_dict(row)


def process_user_alert(user: dict[str, Any]) -> dict[str, Any]:
    user_id = int(user["id"]) if user.get("id") is not None else None
    name = str(user.get("name", "User")).strip() or "User"
    phone = str(user.get("phone", "")).strip()
    lat = float(user["lat"])
    lon = float(user["lon"])
    state = str(user.get("state", DEFAULT_STATE)).strip().lower() or DEFAULT_STATE

    logger.info("Processing alert for user=%s lat=%s lon=%s state=%s", name, lat, lon, state)

    weather = weather_client.fetch_live_weather(lat=lat, lon=lon)
    location_label = build_location_label(weather.city, lat, lon)

    location_key = f"{state}:{lat:.4f},{lon:.4f}"
    prediction = flood_predictor.predict(
        state=state,
        location_key=location_key,
        rainfall=weather.rainfall,
        temperature=weather.temperature,
        humidity=weather.humidity,
    )

    flood_probability = prediction["flood_probability"]
    alerts = build_alerts(
        location_label=location_label,
        flood_probability=flood_probability,
        rainfall=weather.rainfall,
        temperature=weather.temperature,
        aqi_value=weather.aqi["value"],
        flood_threshold=SMS_TRIGGER_THRESHOLD,
    )
    priority = determine_priority(alerts)
    alert_record: dict[str, Any] | None = None

    logger.info(
        "Alert evaluation for user=%s phone=%s location=%s flood_probability=%.4f alerts=%s priority=%s",
        name,
        phone,
        location_label,
        flood_probability,
        [alert_type for alert_type, _ in alerts],
        priority,
    )

    if alerts and priority:
        sms_message = build_alert_sms(name, location_label, alerts, priority)
        telegram_message = build_telegram_alert(
            location_label,
            " / ".join(alert_type for alert_type, _ in alerts),
            sms_message,
        )
        alert_record = upsert_pending_alert(
            user_id=user_id,
            phone=phone,
            location_label=location_label,
            alerts=alerts,
            priority=priority,
            message=sms_message,
        )
        if should_send_alert_reminder(alert_record):
            logger.info(
                "Sending reminder SMS for alert_id=%s user=%s phone=%s sms_attempts=%s",
                alert_record["id"],
                name,
                phone,
                alert_record["sms_attempts"],
            )
            print("Sending SMS to:", phone)
            sms_result = sms_service.send_custom_sms(
                sms_message,
                to_phone=phone,
                priority=priority,
                recipient_key=f"alert:{alert_record['id']}",
            )
            if sms_result.get("sent"):
                alert_record = record_alert_sms_attempt(alert_record["id"], sent=True)
            print("SMS RESULT:", {"name": name, "phone": phone, "result": sms_result})
            logger.info("SMS result for user=%s phone=%s result=%s", name, phone, sms_result)
            personalized_telegram_message = personalize_telegram_message(telegram_message, name)
            telegram_result = send_telegram_to_user(personalized_telegram_message, user)
            logger.info("Telegram result for user=%s phone=%s result=%s", name, phone, telegram_result)
        else:
            sms_result = {
                "sent": False,
                "reason": "Waiting for resend interval or acknowledgement",
                "error": None,
                "phone": phone,
                "priority": priority,
            }
            print("SMS RESULT:", {"name": name, "phone": phone, "result": sms_result})
            logger.info("SMS skipped for user=%s phone=%s result=%s", name, phone, sms_result)
    else:
        sms_result = {
            "sent": False,
            "reason": "No alerts generated",
            "error": None,
            "phone": phone,
            "priority": None,
        }
        print(
            "SMS RESULT:",
            {
                "name": name,
                "phone": phone,
                "result": sms_result,
                "flood_probability": round(flood_probability, 4),
                "temperature": temperature if "temperature" in locals() else weather.temperature,
                "rainfall": weather.rainfall,
                "aqi": weather.aqi["value"],
                "thresholds": {
                    "flood": SMS_TRIGGER_THRESHOLD,
                    "heat": HEAT_ALERT_THRESHOLD,
                    "rain": RAIN_ALERT_THRESHOLD,
                    "aqi": AQI_ALERT_THRESHOLD,
                },
            },
        )
        logger.info(
            "SMS skipped for user=%s phone=%s reason=%s flood_probability=%.4f temperature=%.2f rainfall=%.2f aqi=%s thresholds=%s",
            name,
            phone,
            sms_result["reason"],
            flood_probability,
            weather.temperature,
            weather.rainfall,
            weather.aqi["value"],
            {
                "flood": SMS_TRIGGER_THRESHOLD,
                "heat": HEAT_ALERT_THRESHOLD,
                "rain": RAIN_ALERT_THRESHOLD,
                "aqi": AQI_ALERT_THRESHOLD,
            },
        )

    pending_alerts = list_pending_alerts(phone=phone, user_id=user_id)

    return {
        "user_id": user_id,
        "name": name,
        "phone": phone,
        "city": weather.city,
        "location": location_label,
        "latitude": lat,
        "longitude": lon,
        "state": state,
        "temperature": weather.temperature,
        "weather": weather.condition,
        "condition": weather.condition,
        "rainfall": weather.rainfall,
        "humidity": weather.humidity,
        "aqi": weather.aqi,
        "flood_probability": round(flood_probability, 4),
        "flood_prediction": prediction["flood_prediction"],
        "flood_risk": prediction["flood_prediction"],
        "alerts": alerts,
        "priority": priority,
        "hourly": weather.hourly,
        "daily": weather.daily,
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "sms_alert": sms_result,
        "pending_alerts": pending_alerts,
    }


def process_registered_user_alerts() -> dict[str, Any]:
    sync_users_from_db()
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    print("ALL USERS:", USERS)
    logger.info("ALL USERS: %s", USERS)

    for user in USERS:
        try:
            results.append(process_user_alert(user))
        except (WeatherAPIError, PredictionError, ValueError) as exc:
            logger.error("Failed to process user=%s: %s", user.get("name", "unknown"), exc)
            errors.append(
                {
                    "name": str(user.get("name", "unknown")),
                    "phone": str(user.get("phone", "")),
                    "error": str(exc),
                }
            )

    return {
        "processed_users": len(results),
        "failed_users": len(errors),
        "results": results,
        "errors": errors,
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }


def run_registered_user_alerts(trigger: str) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc).isoformat()
    AUTO_ALERTS_STATUS["last_started_at"] = started_at
    AUTO_ALERTS_STATUS["last_trigger"] = trigger
    AUTO_ALERTS_STATUS["running"] = True
    AUTO_ALERTS_STATUS["last_error"] = None

    with AUTO_ALERTS_LOCK:
        try:
            result = process_registered_user_alerts()
            AUTO_ALERTS_STATUS["last_result"] = result
            AUTO_ALERTS_STATUS["last_completed_at"] = datetime.now(timezone.utc).isoformat()
            return result
        except Exception as exc:
            AUTO_ALERTS_STATUS["last_error"] = str(exc)
            AUTO_ALERTS_STATUS["last_completed_at"] = datetime.now(timezone.utc).isoformat()
            logger.exception("Registered alert processing failed for trigger=%s", trigger)
            raise
        finally:
            AUTO_ALERTS_STATUS["running"] = False


def auto_alert_monitor_loop() -> None:
    logger.info(
        "Automatic alert monitor started interval=%ss enabled=%s",
        AUTO_ALERT_INTERVAL_SECONDS,
        AUTO_ALERTS_ENABLED,
    )
    while True:
        if SIMULATION_MODE:
            time.sleep(min(AUTO_ALERT_INTERVAL_SECONDS, 30))
            continue
        try:
            run_registered_user_alerts("automatic")
        except Exception:
            logger.exception("Automatic alert cycle failed")
        time.sleep(AUTO_ALERT_INTERVAL_SECONDS)


def start_auto_alert_monitor() -> None:
    global AUTO_ALERTS_THREAD
    if not AUTO_ALERTS_ENABLED:
        logger.info("Automatic alert monitor disabled by configuration")
        return
    if AUTO_ALERTS_THREAD is not None and AUTO_ALERTS_THREAD.is_alive():
        return
    AUTO_ALERTS_THREAD = threading.Thread(target=auto_alert_monitor_loop, name="auto-alert-monitor", daemon=True)
    AUTO_ALERTS_THREAD.start()


@app.route("/")
def root() -> Any:
    return redirect(url_for("dashboard" if current_user() else "login_page"))


@app.route("/signup", methods=["GET", "POST"])
def signup() -> Any:
    if request.method == "GET":
        return render_template("signup.html", default_state=DEFAULT_STATE)

    try:
        payload = request.get_json(silent=True) or request.form.to_dict()
        data = validate_signup_payload(payload)
        expects_json = request.is_json

        if get_user_by_email(data["email"]) is not None:
            return jsonify({"success": False, "error": "Email already registered."}), 409

        is_first_user = get_user_by_id(1) is None
        is_admin = bool(is_first_user or (ADMIN_EMAIL and data["email"] == ADMIN_EMAIL))
        token = secrets.token_urlsafe(32)
        user = create_user(
            name=data["name"],
            email=data["email"],
            phone=data["phone"],
            password=data["password"],
            lat=data["lat"],
            lon=data["lon"],
            state=data["state"],
            is_admin=is_admin,
            is_verified=False,
            verification_token=token,
        )
        verify_url = send_verification_email(data["email"], token)
        otp = generate_email_otp()
        session["signup_otp"] = otp
        session["signup_otp_email"] = data["email"]
        session["signup_otp_user_id"] = int(user["id"])
        session["signup_otp_expires_at"] = (datetime.now(timezone.utc) + timedelta(minutes=OTP_EXPIRY_MINUTES)).isoformat()
        send_signup_otp_email(data["email"], otp)

        response_payload = {
            "success": True,
            "message": "Signup successful. Check your email for the OTP.",
            "user": serialize_user(user),
            "verify_url": verify_url,
            "redirect_url": url_for("verify_otp"),
        }
        if expects_json:
            return jsonify(response_payload), 201
        return redirect(url_for("verify_otp"))
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except sqlite3.IntegrityError as exc:
        return jsonify({"success": False, "error": str(exc)}), 409
    except Exception as exc:
        logger.exception("Signup failed")
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/login", methods=["GET", "POST"])
def login_page() -> Any:
    if request.method == "GET":
        return render_template("login.html")

    try:
        payload = request.get_json(silent=True) or request.form.to_dict()
        email = require_text_field(payload, "email").lower()
        password = require_text_field(payload, "password")

        row = get_user_by_email(email)
        if row is None or not verify_password(password, str(row["password"])):
            return jsonify({"success": False, "error": "Invalid credentials."}), 401
        otp_verified_user_id = session.get("otp_verified_user_id")
        otp_verified_for_user = otp_verified_user_id is not None and int(otp_verified_user_id) == int(row["id"])
        if not bool(row["is_verified"]) and not otp_verified_for_user:
            return jsonify({"success": False, "error": "Please verify your email first"}), 403

        session["user_id"] = int(row["id"])
        return jsonify({"success": True, "message": "Login successful", "user": user_row_to_dict(row)})
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except Exception as exc:
        logger.exception("Login failed")
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/verify-otp", methods=["GET", "POST"])
def verify_otp() -> Any:
    if request.method == "GET":
        return render_template("verify_otp.html", email=session.get("signup_otp_email", ""), message="")

    payload = request.get_json(silent=True) or request.form.to_dict()
    otp = require_text_field(payload, "otp")
    session_otp = str(session.get("signup_otp", "")).strip()
    signup_email = str(session.get("signup_otp_email", "")).strip()
    signup_user_id = session.get("signup_otp_user_id")
    expires_at_raw = str(session.get("signup_otp_expires_at", "")).strip()

    if not session_otp or not signup_email or signup_user_id is None:
        error_message = "OTP session expired. Please sign up again."
        if request.is_json:
            return jsonify({"success": False, "error": error_message}), 400
        return render_template("verify_otp.html", email="", message=error_message)

    try:
        expires_at = datetime.fromisoformat(expires_at_raw)
    except ValueError:
        expires_at = None
    if expires_at is None or expires_at <= datetime.now(timezone.utc):
        session.pop("signup_otp", None)
        session.pop("signup_otp_email", None)
        session.pop("signup_otp_user_id", None)
        session.pop("signup_otp_expires_at", None)
        error_message = "OTP expired. Please sign up again."
        if request.is_json:
            return jsonify({"success": False, "error": error_message}), 400
        return render_template("verify_otp.html", email=signup_email, message=error_message)

    if otp != session_otp:
        error_message = "Invalid OTP. Please try again."
        if request.is_json:
            return jsonify({"success": False, "error": error_message}), 400
        return render_template("verify_otp.html", email=signup_email, message=error_message)

    with get_db_connection() as conn:
        conn.execute(
            "UPDATE users SET is_verified = 1, verification_token = NULL WHERE id = ?",
            (int(signup_user_id),),
        )
        conn.commit()
    sync_users_from_db()
    session["otp_verified_user_id"] = int(signup_user_id)
    session.pop("signup_otp", None)
    session.pop("signup_otp_email", None)
    session.pop("signup_otp_user_id", None)
    session.pop("signup_otp_expires_at", None)

    if request.is_json:
        return jsonify({"success": True, "message": "OTP verified successfully.", "redirect_url": url_for("login_page")})
    return redirect(url_for("login_page"))


@app.route("/forgot-password", methods=["POST"])
def forgot_password() -> Any:
    try:
        payload = request.get_json(silent=True) or request.form.to_dict()
        email = require_text_field(payload, "email").lower()
        row = get_user_by_email(email)
        response: dict[str, Any] = {
            "success": True,
            "message": "If that email is registered, a password reset link has been sent.",
        }
        if row is None:
            return jsonify(response)

        token = secrets.token_urlsafe(32)
        expires_at = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        with get_db_connection() as conn:
            conn.execute(
                "UPDATE users SET reset_token = ?, reset_token_expires_at = ? WHERE id = ?",
                (token, expires_at, int(row["id"])),
            )
            conn.commit()
        reset_url = send_password_reset_email(email, token)
        response["reset_url"] = reset_url
        return jsonify(response)
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except Exception as exc:
        logger.exception("Forgot password failed")
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/reset-password/<token>", methods=["GET", "POST"])
def reset_password(token: str) -> Any:
    row = get_user_by_reset_token(token)
    if row is None:
        if request.method == "GET":
            return render_template("reset_password.html", token=token, message="Invalid or expired reset link.", is_valid=False)
        return jsonify({"success": False, "error": "Invalid or expired reset link."}), 400

    expires_at_raw = str(row["reset_token_expires_at"] or "").strip()
    try:
        expires_at = datetime.fromisoformat(expires_at_raw)
    except ValueError:
        expires_at = None
    is_expired = expires_at is None or expires_at <= datetime.now(timezone.utc)
    if is_expired:
        with get_db_connection() as conn:
            conn.execute(
                "UPDATE users SET reset_token = NULL, reset_token_expires_at = NULL WHERE id = ?",
                (int(row["id"]),),
            )
            conn.commit()
        if request.method == "GET":
            return render_template("reset_password.html", token=token, message="This reset link has expired. Request a new one.", is_valid=False)
        return jsonify({"success": False, "error": "This reset link has expired. Request a new one."}), 400

    if request.method == "GET":
        return render_template("reset_password.html", token=token, message="", is_valid=True)

    try:
        payload = request.get_json(silent=True) or request.form.to_dict()
        password = validate_new_password(require_text_field(payload, "password"))
        confirm_password = require_text_field(payload, "confirm_password")
        if password != confirm_password:
            return jsonify({"success": False, "error": "Passwords do not match."}), 400

        with get_db_connection() as conn:
            conn.execute(
                "UPDATE users SET password = ?, reset_token = NULL, reset_token_expires_at = NULL WHERE id = ?",
                (hash_password(password), int(row["id"])),
            )
            conn.commit()
        sync_users_from_db()
        return jsonify({"success": True, "message": "Password updated successfully. You can log in now."})
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except Exception as exc:
        logger.exception("Reset password failed")
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/logout", methods=["GET"])
def logout() -> Any:
    session.clear()
    return redirect(url_for("login_page"))


@app.route("/current-user", methods=["GET"])
@login_required
def current_user_route() -> Any:
    user = current_user()
    assert user is not None
    return jsonify({"success": True, "user": user, "pending_alerts": list_pending_alerts(phone=str(user["phone"]), user_id=int(user["id"]))})


@app.route("/send-message", methods=["POST"])
@login_required
def send_message_route() -> Any:
    user = current_user()
    assert user is not None
    payload = request.get_json(silent=True) or request.form.to_dict()
    try:
        message = require_text_field(payload, "message")
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    if len(message) > 1000:
        return jsonify({"success": False, "error": "message must be 1000 characters or fewer"}), 400

    saved_message = save_community_message(str(user["name"]), message)
    return jsonify({"success": True, "message_item": saved_message}), 201


@app.route("/get-messages", methods=["GET"])
@login_required
def get_messages_route() -> Any:
    limit_raw = str(request.args.get("limit", "200")).strip()
    try:
        limit = int(limit_raw)
    except ValueError:
        return jsonify({"success": False, "error": "limit must be numeric"}), 400
    return jsonify({"success": True, "messages": list_community_messages(limit=limit)})


@app.route("/alerts/pending", methods=["GET"])
@login_required
def pending_alerts_route() -> Any:
    user = current_user()
    assert user is not None
    return jsonify({"success": True, "alerts": list_pending_alerts(phone=str(user["phone"]), user_id=int(user["id"]))})


@app.route("/alerts/overview", methods=["GET"])
@admin_required
def alert_overview_route() -> Any:
    limit_raw = str(request.args.get("limit", "100")).strip()
    try:
        limit = int(limit_raw)
    except ValueError:
        return jsonify({"success": False, "error": "limit must be numeric"}), 400
    limit = max(1, min(limit, 500))
    alerts = list_alert_overview(limit=limit)
    pending_count = sum(1 for alert in alerts if alert["status"] == "pending")
    acknowledged_count = sum(1 for alert in alerts if alert["status"] == "acknowledged")
    return jsonify(
        {
            "success": True,
            "alerts": alerts,
            "count": len(alerts),
            "pending_count": pending_count,
            "acknowledged_count": acknowledged_count,
        }
    )


@app.route("/admin/telegram-acknowledgements", methods=["GET"])
@admin_required
def telegram_acknowledgements_route() -> Any:
    cleanup_expired_telegram_acknowledgements()
    limit_raw = str(request.args.get("limit", "100")).strip()
    try:
        limit = int(limit_raw)
    except ValueError:
        return jsonify({"success": False, "error": "limit must be numeric"}), 400
    limit = max(1, min(limit, 500))
    items = list_telegram_acknowledgements(limit=limit)
    return jsonify({"success": True, "acknowledgements": items, "count": len(items)})


@app.route("/alerts/acknowledge", methods=["POST"])
@login_required
def acknowledge_alert_route() -> Any:
    user = current_user()
    assert user is not None
    payload = request.get_json(silent=True) or {}
    alert_id_raw = payload.get("alert_id")
    if alert_id_raw is None or str(alert_id_raw).strip() == "":
        return jsonify({"success": False, "error": "alert_id is required"}), 400
    try:
        alert_id = int(str(alert_id_raw).strip())
    except ValueError:
        return jsonify({"success": False, "error": "alert_id must be numeric"}), 400

    with get_db_connection() as conn:
        row = conn.execute("SELECT * FROM alerts WHERE id = ?", (alert_id,)).fetchone()
    if row is None:
        return jsonify({"success": False, "error": "Alert not found"}), 404
    if str(row["phone"]) != str(user["phone"]):
        return jsonify({"success": False, "error": "You can only acknowledge your own alerts"}), 403

    acknowledged = acknowledge_alert(alert_id=alert_id, source="dashboard")
    return jsonify(
        {
            "success": True,
            "alert": acknowledged,
            "alerts": list_pending_alerts(phone=str(user["phone"]), user_id=int(user["id"])),
        }
    )


@app.route("/sms/test", methods=["POST"])
@login_required
def test_sms_route() -> Any:
    user = current_user()
    assert user is not None

    payload = request.get_json(silent=True) or {}
    phone = str(payload.get("phone", user["phone"])).strip()
    message = str(payload.get("message", "")).strip()
    if not message:
        message = (
            f"AI Disaster Management test SMS for {user['name']}.\n"
            f"Sent at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}."
        )

    try:
        validate_phone_env("phone", phone)
    except RuntimeError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400

    sms_result = sms_service.send_custom_sms(
        message,
        to_phone=phone,
        priority="HIGH",
        recipient_key=f"manual-test:{phone}",
    )
    telegram_message = build_telegram_alert("Manual Test", "TEST", message)
    telegram_result = send_telegram_to_user(telegram_message, user, fallback=True)
    status_code = 200 if sms_result.get("sent") else 502
    return jsonify(
        {
            "success": bool(sms_result.get("sent")),
            "message": message,
            "sms_result": sms_result,
            "telegram_result": telegram_result,
        }
    ), status_code


@app.route("/sms/reply", methods=["POST"])
def sms_reply_webhook() -> Any:
    from_phone = str(request.form.get("From", "")).strip()
    message_body = str(request.form.get("Body", "")).strip()
    if not from_phone:
        return ("", 204)
    acknowledged = acknowledge_latest_alert_for_phone(from_phone, message_body)
    if acknowledged is None:
        return ("", 204)
    logger.info("Alert acknowledged by SMS reply for phone=%s alert_id=%s", from_phone, acknowledged["id"])
    return ("", 204)


@app.route("/telegram-webhook", methods=["POST"])
def telegram_webhook() -> Any:
    update = request.get_json(silent=True) or {}
    message = update.get("message") or update.get("edited_message") or {}
    chat = message.get("chat") or {}
    sender = message.get("from") or {}

    chat_id_raw = chat.get("id")
    if chat_id_raw is None:
        logger.info("Telegram webhook received update without chat_id")
        return jsonify({"success": True, "linked": False, "reason": "No chat_id in update"}), 200

    chat_id = str(chat_id_raw).strip()
    username = str(sender.get("username", "")).strip() or None
    message_text = str(message.get("text", "")).strip() or None

    result = handle_telegram_start_message(chat_id, username, message_text, update)
    return jsonify({"success": True, **result}), 200


@app.route("/admin/telegram-bot-check", methods=["GET"])
def telegram_bot_check_route() -> Any:
    response = verify_telegram_bot_token()
    status_code = 200 if response.get("status_code") == 200 and response.get("ok") is True else 502
    return jsonify(response), status_code


@app.route("/verify/<token>", methods=["GET"])
def verify_email(token: str) -> Any:
    try:
        with get_db_connection() as conn:
            row = conn.execute("SELECT * FROM users WHERE verification_token = ?", (token,)).fetchone()
            if row is None:
                return render_template("login.html", message="Invalid or expired verification link.")
            conn.execute(
                "UPDATE users SET is_verified = 1, verification_token = NULL WHERE id = ?",
                (int(row["id"]),),
            )
            conn.commit()
        sync_users_from_db()
        return render_template("login.html", message="Email verified. You can now log in.")
    except Exception:
        logger.exception("Email verification failed")
        return render_template("login.html", message="Unable to verify email right now.")


@app.route("/dashboard", methods=["GET"])
@login_required
def dashboard() -> Any:
    user = current_user()
    return render_template(
        "index.html",
        current_user_json=json.dumps(user),
        is_admin=user["is_admin"],
        default_state=DEFAULT_STATE,
    )


@app.route("/about", methods=["GET"])
@login_required
def about_page() -> Any:
    return render_template("about.html")


@app.route("/emergency", methods=["GET"])
@login_required
def emergency_page() -> Any:
    emergency_contacts = [
        {"label": "Police", "number": "100", "description": "For law and order emergencies and immediate police assistance."},
        {"label": "Ambulance", "number": "108", "description": "For medical emergencies and ambulance dispatch support."},
        {"label": "Fire", "number": "101", "description": "For fire accidents, rescue needs, and hazardous incidents."},
        {"label": "Disaster Management", "number": "1078", "description": "For regional emergency coordination and disaster-response help."},
    ]
    return render_template("emergency.html", emergency_contacts=emergency_contacts)


@app.route("/support", methods=["GET"])
@login_required
def support_page() -> Any:
    support_details = {
        "email": "support@aidisastermanagement.in",
        "customer_care": "+91 98765 43210",
        "helpline": "+91 11 2345 6789",
        "hours": "24/7 emergency monitoring support with daytime customer care assistance.",
    }
    return render_template("support.html", support=support_details)


@app.route("/contacts", methods=["GET"])
@login_required
def contacts_page() -> Any:
    user = current_user()
    assert user is not None
    saved_contacts = [
        {"label": "Registered Account", "value": str(user["phone"]), "description": "Primary mobile number used for alerts and acknowledgements."},
        {"label": "Account Email", "value": str(user["email"]), "description": "Email used for login, recovery, and system updates."},
    ]
    return render_template("contacts.html", saved_contacts=saved_contacts, user=user)


@app.route("/users", methods=["GET"])
@admin_required
def list_users() -> Any:
    try:
        with get_db_connection() as conn:
            rows = conn.execute(
                "SELECT id, name, email, phone, is_admin, is_verified, telegram_chat_id, lat, lon, state FROM users ORDER BY id ASC"
            ).fetchall()
        users = [user_row_to_dict(row) for row in rows]
        return jsonify({"success": True, "users": [serialize_user(user) for user in users], "count": len(users)})
    except Exception as exc:
        logger.exception("Failed to list users")
        return jsonify({"success": False, "error": str(exc), "users": [], "count": 0}), 500


@app.route("/add-user", methods=["POST"])
@admin_required
def add_user() -> Any:
    try:
        payload = request.get_json(silent=True) or {}
        logger.info("Add user payload received: %s", payload)
        data = validate_admin_user_payload(payload)

        if get_user_by_email(data["email"]) is not None:
            return jsonify({"success": False, "error": "Email already registered."}), 409

        token = None if data["is_verified"] else secrets.token_urlsafe(32)
        user = create_user(
            name=data["name"],
            email=data["email"],
            phone=data["phone"],
            password=data["password"],
            lat=data["lat"],
            lon=data["lon"],
            state=data["state"],
            is_admin=False,
            is_verified=data["is_verified"],
            verification_token=token,
        )
        verify_url = send_verification_email(data["email"], token) if token else None
        print("ALL USERS:", sync_users_from_db())
        return jsonify(
            {
                "success": True,
                "message": "User added successfully",
                "user": serialize_user(user),
                "users": [serialize_user(registered_user) for registered_user in sync_users_from_db()],
                "verify_url": verify_url,
            }
        ), 201
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except sqlite3.IntegrityError as exc:
        return jsonify({"success": False, "error": str(exc)}), 409
    except Exception as exc:
        logger.exception("Unexpected error in /add-user")
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/delete-user", methods=["POST"])
@admin_required
def delete_user() -> Any:
    try:
        payload = request.get_json(silent=True) or {}
        user_id = payload.get("user_id")
        if user_id is None or str(user_id).strip() == "":
            return jsonify({"success": False, "error": "user_id is required"}), 400

        user_id_int = int(str(user_id).strip())
        actor = current_user()
        row = get_user_by_id(user_id_int)
        if row is None:
            return jsonify({"success": False, "error": "User not found"}), 404
        if actor is not None and int(actor["id"]) == user_id_int:
            return jsonify({"success": False, "error": "You cannot delete your own account"}), 400

        with get_db_connection() as conn:
            conn.execute("DELETE FROM users WHERE id = ?", (user_id_int,))
            conn.commit()

        users = [serialize_user(user) for user in sync_users_from_db()]
        return jsonify(
            {
                "success": True,
                "message": "User removed successfully",
                "users": users,
                "count": len(users),
            }
        )
    except ValueError:
        return jsonify({"success": False, "error": "user_id must be numeric"}), 400
    except Exception as exc:
        logger.exception("Delete user failed")
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/upload-dataset", methods=["POST"])
@admin_required
def upload_dataset() -> Any:
    global LATEST_DATASET_PATH

    try:
        if "file" not in request.files:
            return jsonify({"success": False, "error": "CSV file is required."}), 400

        dataset_file = request.files["file"]
        if dataset_file.filename is None or dataset_file.filename.strip() == "":
            return jsonify({"success": False, "error": "CSV file is required."}), 400
        if not allowed_csv_file(dataset_file.filename):
            return jsonify({"success": False, "error": "Only CSV uploads are allowed."}), 400

        safe_name = f"{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{Path(dataset_file.filename).name}"
        dataset_path = UPLOAD_DIR / safe_name
        dataset_file.save(dataset_path)
        LATEST_DATASET_PATH = dataset_path

        loaded_frame = load_dataset_frame(dataset_path)
        simulation_frame = validate_simulation_frame(loaded_frame)
        dataset_summary = summarize_dataset(simulation_frame)
        training_result = None
        if "flood_risk" in simulation_frame.columns:
            training_frame = validate_dataset_frame(simulation_frame)
            training_result = train_model_from_dataset(training_frame)

        with SIMULATION_LOCK:
            SIMULATION_STATUS.update(
                {
                    "dataset": str(dataset_path),
                    "total_rows": int(len(simulation_frame)),
                    "rows_processed": 0,
                    "alerts_triggered": 0,
                    "summary": None,
                    "error": None,
                }
            )
        global SIMULATION_MODE
        SIMULATION_MODE = True

        return jsonify(
            {
                "success": True,
                "message": "Dataset uploaded successfully.",
                "dataset_path": str(dataset_path),
                "dataset_summary": dataset_summary,
                "training": training_result,
                "simulation_mode": SIMULATION_MODE,
            }
        ), 201
    except ValueError as exc:
        logger.error("Dataset validation failed: %s", exc)
        return jsonify({"success": False, "error": str(exc)}), 400
    except Exception as exc:
        logger.exception("Dataset upload failed")
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/remove-dataset", methods=["POST"])
@admin_required
def remove_dataset() -> Any:
    global LATEST_DATASET_PATH, SIMULATION_MODE, SIMULATION_THREAD

    try:
        if SIMULATION_THREAD is not None and SIMULATION_THREAD.is_alive():
            return jsonify({"success": False, "error": "Stop or wait for the running simulation before removing the dataset."}), 409

        dataset_path = latest_dataset_path()
        removed_files: list[str] = []

        if dataset_path is not None and dataset_path.exists():
            dataset_path.unlink()
            removed_files.append(str(dataset_path))

        if TRAINED_MODEL_PATH.exists():
            TRAINED_MODEL_PATH.unlink()
            removed_files.append(str(TRAINED_MODEL_PATH))

        LATEST_DATASET_PATH = None
        SIMULATION_THREAD = None
        SIMULATION_MODE = False
        restore_default_model()
        with SIMULATION_LOCK:
            SIMULATION_STATUS.update(
                {
                    "running": False,
                    "dataset": None,
                    "total_rows": 0,
                    "rows_processed": 0,
                    "alerts_triggered": 0,
                    "started_at": None,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "last_alert_at": None,
                    "summary": {
                        "mode": "live",
                        "total_rows_processed": 0,
                        "alerts_triggered": 0,
                        "dataset": None,
                    },
                    "error": None,
                }
            )

        return jsonify(
            {
                "success": True,
                "message": "Dataset removed. System reverted to normal live mode.",
                "removed_files": removed_files,
                "simulation_mode": SIMULATION_MODE,
                "model_path": str(model_registry.get_model_path(DEFAULT_STATE)),
            }
        )
    except Exception as exc:
        logger.exception("Dataset removal failed")
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/historical-insights", methods=["GET"])
@admin_required
def historical_insights() -> Any:
    try:
        return jsonify({"success": True, **build_historical_insights()})
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 404
    except Exception as exc:
        logger.exception("Historical insights failed")
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/start-simulation", methods=["POST"])
@admin_required
def start_simulation() -> Any:
    global SIMULATION_THREAD

    dataset_path = latest_dataset_path()
    if dataset_path is None:
        return jsonify({"success": False, "error": "No uploaded dataset available."}), 404

    if SIMULATION_THREAD is not None and SIMULATION_THREAD.is_alive():
        return jsonify({"success": False, "error": "Simulation is already running."}), 409

    SIMULATION_THREAD = threading.Thread(target=simulate_dataset_alerts, args=(dataset_path,), daemon=True)
    SIMULATION_THREAD.start()
    return jsonify({"success": True, "message": "Simulation started.", "dataset": str(dataset_path)})


@app.route("/simulation-status", methods=["GET"])
@admin_required
def simulation_status() -> Any:
    with SIMULATION_LOCK:
        return jsonify({"success": True, "simulation_mode": SIMULATION_MODE, **SIMULATION_STATUS})


@app.route("/auto-alert-status", methods=["GET"])
@admin_required
def auto_alert_status() -> Any:
    return jsonify(
        {
            "success": True,
            **AUTO_ALERTS_STATUS,
            "thread_alive": AUTO_ALERTS_THREAD.is_alive() if AUTO_ALERTS_THREAD is not None else False,
            "simulation_mode": SIMULATION_MODE,
        }
    )


@app.route("/predict-live", methods=["POST"])
@login_required
def predict_live() -> Any:
    try:
        logger.info("API request received for /predict-live")
        payload = request.get_json(silent=True) or {}
        user = current_user()
        assert user is not None

        lat = payload.get("lat", user["lat"])
        lon = payload.get("lon", user["lon"])
        state = str(payload.get("state", user["state"])).strip().lower()
        phone = str(payload.get("phone", user["phone"])).strip()
        name = str(payload.get("name", user["name"])).strip() or user["name"]

        live_user = {
            "name": name,
            "phone": phone,
            "lat": float(lat),
            "lon": float(lon),
            "state": state,
        }
        result = process_user_alert(live_user)
        result["mode"] = "simulation" if SIMULATION_MODE else "live"
        if user["is_admin"]:
            result["group_alerts"] = process_registered_user_alerts()
        return jsonify(result)
    except (TypeError, ValueError):
        return jsonify({"success": False, "error": "'lat' and 'lon' must be numeric"}), 400
    except WeatherAPIError as exc:
        logger.error("Weather service failed: %s", exc)
        return jsonify({"success": False, "error": str(exc)}), 502
    except PredictionError as exc:
        logger.error("Model service failed: %s", exc)
        return jsonify({"success": False, "error": str(exc)}), 500
    except Exception as exc:
        logger.exception("Unexpected backend error")
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/process-group-alerts", methods=["POST", "GET"])
@admin_required
def process_group_alerts() -> Any:
    return jsonify({"success": True, **run_registered_user_alerts("manual")})


@app.route("/health", methods=["GET"])
def health() -> Any:
    status = dependency_status()
    code = 200 if status["status"] == "ok" else 503
    return jsonify(status), code


@app.route("/model-info", methods=["GET"])
@login_required
def model_info() -> Any:
    state = request.args.get("state", DEFAULT_STATE)
    try:
        return jsonify(model_registry.get_model_info(state))
    except PredictionError as exc:
        logger.error("Model info lookup failed: %s", exc)
        return jsonify({"success": False, "error": str(exc)}), 500


init_db()
sync_users_from_db()


def should_start_background_jobs() -> bool:
    if __name__ != "__main__":
        return True
    debug_enabled = os.getenv("FLASK_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"} or app.debug
    if not debug_enabled:
        return True
    return os.getenv("WERKZEUG_RUN_MAIN") == "true"


if should_start_background_jobs():
    start_auto_alert_monitor()
    start_telegram_polling_listener()


if __name__ == "__main__":
    flask_debug_enabled = os.getenv("FLASK_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
        debug=flask_debug_enabled,
        use_reloader=False,
    )
