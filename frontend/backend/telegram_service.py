from __future__ import annotations

import logging
import os
import time
from typing import Any
from urllib.parse import urlparse

import requests

logger = logging.getLogger("telegram")
DEFAULT_TIMEOUT_SECONDS = 10
DEFAULT_RETRY_ATTEMPTS = 2


def _env_value(*names: str) -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""


def _env_flag(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _normalize_proxy_url(proxy_url: str) -> str:
    proxy_url = proxy_url.strip()
    if not proxy_url:
        return ""

    parsed = urlparse(proxy_url)
    if parsed.scheme:
        # HTTPS destinations should still tunnel through an HTTP proxy URL.
        if parsed.scheme.lower() == "https":
            return parsed._replace(scheme="http").geturl()
        return proxy_url

    return f"http://{proxy_url}"


def _mask_token(token: str) -> str:
    if len(token) <= 10:
        return "***"
    return f"{token[:6]}...{token[-4:]}"


def _build_telegram_url(token: str, endpoint: str) -> str:
    return f"https://api.telegram.org/bot{token}/{endpoint}"


def _sanitize_for_log(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sanitize_for_log(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_log(item) for item in value]
    if isinstance(value, str) and "bot" in value and "api.telegram.org" in value:
        return value.replace("/bot", "/bot***")
    return value


def _resolve_telegram_proxies() -> dict[str, str]:
    http_proxy = _normalize_proxy_url(_env_value("HTTP_PROXY", "http_proxy"))
    https_proxy = _normalize_proxy_url(_env_value("HTTPS_PROXY", "https_proxy"))

    if not http_proxy and not https_proxy:
        logger.info("Telegram proxy config: no HTTP(S) proxy detected in environment")
        return {}

    proxy = http_proxy or https_proxy
    proxies = {
        "http": proxy,
        "https": https_proxy or proxy,
    }

    # Ensure both schemes use http:// to avoid SSL handshake errors with HTTP proxies.
    proxies["http"] = _normalize_proxy_url(proxies["http"])
    proxies["https"] = _normalize_proxy_url(proxies["https"])
    logger.info(
        "Telegram proxy config detected: http=%s https=%s",
        proxies["http"],
        proxies["https"],
    )
    return proxies


def _telegram_request(
    method: str,
    endpoint: str,
    *,
    params: dict[str, Any] | None = None,
    json_payload: dict[str, Any] | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    max_attempts: int = 1,
) -> dict[str, Any]:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        logger.warning("Telegram request skipped: TELEGRAM_BOT_TOKEN is missing")
        return {
            "ok": False,
            "result": [],
            "description": "Telegram bot token not configured",
            "status_code": None,
            "endpoint": endpoint,
        }

    url = _build_telegram_url(token, endpoint)
    masked_url = _build_telegram_url(_mask_token(token), endpoint)
    verify_ssl = _env_flag("TELEGRAM_SSL_VERIFY", default=True)
    proxies = _resolve_telegram_proxies()
    request_context = {
        "endpoint": endpoint,
        "method": method.upper(),
        "url": masked_url,
        "params": _sanitize_for_log(params or {}),
        "json_payload": _sanitize_for_log(json_payload or {}),
        "timeout": timeout,
        "verify_ssl": verify_ssl,
        "proxies_enabled": bool(proxies),
    }

    if not verify_ssl:
        logger.warning("Telegram SSL verification is disabled for debugging only")

    logger.info("Telegram request prepared: %s", request_context)

    for attempt in range(1, max_attempts + 1):
        if attempt > 1:
            logger.warning(
                "Retrying Telegram request endpoint=%s attempt=%s/%s",
                endpoint,
                attempt,
                max_attempts,
            )

        try:
            with requests.Session() as session:
                # We explicitly manage proxies here so requests does not silently pick
                # up incompatible environment settings.
                session.trust_env = False
                response = session.request(
                    method=method.upper(),
                    url=url,
                    params=params,
                    json=json_payload,
                    timeout=timeout,
                    verify=verify_ssl,
                    proxies=proxies or None,
                )
        except requests.exceptions.SSLError as exc:
            logger.exception("Telegram SSL error while calling %s", masked_url)
            return {
                "ok": False,
                "result": [],
                "description": (
                    f"SSL error while calling Telegram: {exc}. "
                    "If a proxy is required, use http:// for both HTTP_PROXY and HTTPS_PROXY. "
                    "For temporary debugging only, set TELEGRAM_SSL_VERIFY=false."
                ),
                "status_code": None,
                "endpoint": endpoint,
                "request": request_context,
                "exception_type": type(exc).__name__,
            }
        except requests.exceptions.ProxyError as exc:
            logger.exception("Telegram proxy error while calling %s", masked_url)
            return {
                "ok": False,
                "result": [],
                "description": (
                    f"Proxy error while calling Telegram: {exc}. "
                    "Check HTTP_PROXY/HTTPS_PROXY values or remove proxy usage if not required."
                ),
                "status_code": None,
                "endpoint": endpoint,
                "request": request_context,
                "exception_type": type(exc).__name__,
            }
        except requests.exceptions.Timeout as exc:
            logger.exception("Telegram request timed out while calling %s", masked_url)
            if attempt < max_attempts:
                time.sleep(1)
                continue
            return {
                "ok": False,
                "result": [],
                "description": f"Telegram request timed out after {timeout} seconds: {exc}",
                "status_code": None,
                "endpoint": endpoint,
                "request": request_context,
                "exception_type": type(exc).__name__,
            }
        except requests.RequestException as exc:
            logger.exception("Telegram request failed while calling %s", masked_url)
            if attempt < max_attempts:
                time.sleep(1)
                continue
            return {
                "ok": False,
                "result": [],
                "description": f"Telegram request failed: {exc}",
                "status_code": None,
                "endpoint": endpoint,
                "request": request_context,
                "exception_type": type(exc).__name__,
            }

        try:
            data = response.json()
        except ValueError:
            data = {
                "ok": False,
                "result": [],
                "description": response.text or "Telegram returned a non-JSON response",
            }

        response_summary = {
            "endpoint": endpoint,
            "status_code": response.status_code,
            "ok": bool(data.get("ok")),
            "error_code": data.get("error_code"),
            "description": data.get("description"),
            "response_json": _sanitize_for_log(data),
        }
        logger.info("Telegram response: %s", response_summary)

        if response.status_code == 409:
            description = str(
                data.get("description")
                or "Telegram webhook is still active for this bot. Disable the webhook before using polling."
            )
            logger.warning("Telegram polling conflict: %s", description)
            return {
                "ok": False,
                "result": [],
                "description": description,
                "status_code": 409,
                "endpoint": endpoint,
                "request": request_context,
                "response_json": data,
                "error_code": data.get("error_code"),
            }

        if response.status_code >= 500 and attempt < max_attempts:
            time.sleep(1)
            continue

        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            logger.exception("Telegram API returned HTTP %s for %s", response.status_code, masked_url)
            return {
                "ok": False,
                "result": [],
                "description": f"Telegram API HTTP {response.status_code}: {data.get('description') or exc}",
                "status_code": response.status_code,
                "endpoint": endpoint,
                "request": request_context,
                "response_json": data,
                "error_code": data.get("error_code"),
            }

        data.setdefault("status_code", response.status_code)
        data.setdefault("endpoint", endpoint)
        data.setdefault("request", request_context)
        data.setdefault("response_json", data.copy())
        if not data.get("ok"):
            logger.error(
                "Telegram API %s failed status_code=%s error_code=%s description=%s",
                endpoint,
                response.status_code,
                data.get("error_code"),
                data.get("description"),
            )
        return data

    return {
        "ok": False,
        "result": [],
        "description": f"Telegram request failed after {max_attempts} attempts",
        "status_code": None,
        "endpoint": endpoint,
        "request": request_context,
    }


def fetch_telegram_updates(*, offset: int | None = None, timeout: int = 30) -> dict[str, Any]:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        logger.warning("Telegram polling skipped: TELEGRAM_BOT_TOKEN is missing")
        return {
            "ok": False,
            "result": [],
            "description": "Telegram bot token not configured",
        }

    params: dict[str, Any] = {
        "timeout": timeout,
    }
    if offset is not None:
        params["offset"] = offset

    return _telegram_request(
        "GET",
        "getUpdates",
        params=params,
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )


def verify_telegram_bot_token() -> dict[str, Any]:
    response = _telegram_request(
        "GET",
        "getMe",
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    if response.get("status_code") == 200 and response.get("ok") is True:
        result = response.get("result") or {}
        logger.info(
            "Telegram bot token verified successfully bot_id=%s username=%s",
            result.get("id"),
            result.get("username"),
        )
    else:
        logger.warning(
            "Telegram bot token verification failed status_code=%s error_code=%s description=%s",
            response.get("status_code"),
            response.get("error_code"),
            response.get("description"),
        )
    return response


def send_telegram(
    message: str,
    chat_id: str | int | None = None,
    reply_markup: dict[str, Any] | None = None,
) -> dict[str, Any]:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    resolved_chat_id = str(chat_id).strip() if chat_id is not None else os.getenv("TELEGRAM_CHAT_ID", "").strip()

    if not token or not resolved_chat_id:
        logger.warning(
            "Telegram skipped: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is missing token_present=%s chat_id=%s",
            bool(token),
            resolved_chat_id or None,
        )
        return {
            "sent": False,
            "reason": "Telegram credentials not configured",
            "error": None,
            "chat_id": resolved_chat_id or None,
        }

    payload = {
        "chat_id": resolved_chat_id,
        "text": message,
    }
    if reply_markup is not None:
        payload["reply_markup"] = reply_markup

    logger.info("Telegram send attempt chat_id=%s payload=%s", resolved_chat_id, _sanitize_for_log(payload))
    data = _telegram_request(
        "POST",
        "sendMessage",
        json_payload=payload,
        timeout=DEFAULT_TIMEOUT_SECONDS,
        max_attempts=DEFAULT_RETRY_ATTEMPTS,
    )
    status_code = data.get("status_code")
    api_ok = data.get("ok") is True
    logger.info(
        "Telegram send validation chat_id=%s status_code=%s ok=%s full_response=%s",
        resolved_chat_id,
        status_code,
        api_ok,
        _sanitize_for_log(data),
    )

    if status_code != 200 or not api_ok:
        error_description = data.get("description", "Telegram API returned an error")
        error_code = data.get("error_code")
        if status_code == 403:
            logger.error(
                "Telegram send failed: bot blocked or user never started the bot chat_id=%s error_code=%s description=%s",
                resolved_chat_id,
                error_code,
                error_description,
            )
        elif status_code == 400:
            logger.error(
                "Telegram send failed: invalid chat_id or malformed payload chat_id=%s error_code=%s description=%s",
                resolved_chat_id,
                error_code,
                error_description,
            )
        else:
            logger.error(
                "Telegram send failed chat_id=%s status_code=%s error_code=%s description=%s",
                resolved_chat_id,
                status_code,
                error_code,
                error_description,
            )
        return {
            "sent": False,
            "reason": "Telegram API returned an error",
            "error": error_description,
            "error_code": error_code,
            "chat_id": resolved_chat_id,
            "status_code": status_code,
            "response": data,
        }

    logger.info("Telegram message sent successfully chat_id=%s", resolved_chat_id)
    return {
        "sent": True,
        "reason": "Telegram message sent",
        "error": None,
        "chat_id": resolved_chat_id,
        "status_code": status_code,
        "message_id": ((data.get("result") or {}).get("message_id") if isinstance(data.get("result"), dict) else None),
        "response": data,
    }
