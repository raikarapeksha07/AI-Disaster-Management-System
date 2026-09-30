from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any

from twilio.rest import Client

logger = logging.getLogger("sms")
E164_PHONE_RE = re.compile(r"^\+[1-9]\d{7,14}$")


@dataclass
class AlertState:
    last_risk: str = "LOW"
    last_sent_epoch: float = 0.0


class SMSService:
    """Send alerts with recipient-aware cooldown and priority handling."""

    def __init__(
        self,
        account_sid: str,
        auth_token: str,
        from_phone: str,
        to_phone: str | None = None,
        cooldown_seconds: int = 60,
        trigger_threshold: float = 0.1,
    ) -> None:
        self.account_sid = account_sid.strip()
        self.auth_token = auth_token.strip()
        self.from_phone = from_phone.strip()
        self.default_to_phone = to_phone.strip() if to_phone else None
        self.cooldown_seconds = cooldown_seconds
        self.trigger_threshold = trigger_threshold
        self._state_by_recipient: dict[str, AlertState] = {}
        self._client: Client | None = None
        self._client_identity: tuple[str, str] | None = None

        sid, token, active_from_phone = self._resolve_config()
        self._client = self._build_client(sid, token)
        self._client_identity = (sid, token)
        self._debug_log_config("SMS client initialized", sid=sid, token=token, from_phone=active_from_phone)

    def process_flood_alert(
        self,
        city: str,
        flood_probability: float,
        risk_level: str,
        to_phone: str | None = None,
    ) -> dict[str, Any]:
        recipient_phone = to_phone or self.default_to_phone
        recipient_key = self._recipient_key(recipient_phone, city)
        state = self._state_by_recipient.get(recipient_key, AlertState())

        now = time.time()
        cooldown_ok = (now - state.last_sent_epoch) >= self.cooldown_seconds
        threshold_ok = flood_probability >= self.trigger_threshold

        logger.info(
            "SMS evaluation city=%s probability=%.4f risk=%s threshold_ok=%s cooldown_ok=%s to=%s",
            city,
            flood_probability,
            risk_level,
            threshold_ok,
            cooldown_ok,
            recipient_phone,
        )

        sent = False
        reason = "Not triggered"
        error = None

        if not recipient_phone:
            reason = "Recipient phone not configured"
        elif not threshold_ok:
            reason = "Probability below threshold"
            logger.info("SMS skipped: probability below threshold")
        elif not cooldown_ok:
            reason = "Cooldown active"
            logger.info("SMS skipped: cooldown active")
        else:
            send_result = self._send_sms(
                body=self._build_message(city, flood_probability, risk_level),
                recipient_phone=recipient_phone,
                context=f"city={city} risk={risk_level}",
                priority=risk_level,
            )
            sent = bool(send_result["sent"])
            reason = str(send_result["reason"])
            error = send_result["error"]
            if sent:
                state.last_sent_epoch = now
                state.last_risk = risk_level

        self._state_by_recipient[recipient_key] = state

        return {
            "sent": sent,
            "reason": reason,
            "error": error,
            "phone": recipient_phone,
        }

    def send_custom_sms(
        self,
        message: str,
        to_phone: str | None = None,
        *,
        priority: str = "MEDIUM",
        recipient_key: str | None = None,
    ) -> dict[str, Any]:
        recipient_phone = to_phone or self.default_to_phone
        if not recipient_phone:
            return {
                "sent": False,
                "reason": "Recipient phone not configured",
                "error": None,
                "phone": None,
                "priority": priority,
            }

        state_key = recipient_key or self._recipient_key(recipient_phone, priority)
        state = self._state_by_recipient.get(state_key, AlertState())
        now = time.time()
        bypass_cooldown = priority.upper() == "HIGH"
        cooldown_ok = bypass_cooldown or (now - state.last_sent_epoch) >= self.cooldown_seconds

        if not cooldown_ok:
            logger.info("Custom SMS skipped: cooldown active for %s", recipient_phone)
            return {
                "sent": False,
                "reason": "Cooldown active",
                "error": None,
                "phone": recipient_phone,
                "priority": priority,
            }

        send_result = self._send_sms(
            body=message,
            recipient_phone=recipient_phone,
            context="custom",
            priority=priority,
        )
        if send_result["sent"]:
            state.last_sent_epoch = now
            state.last_risk = priority.upper()
            self._state_by_recipient[state_key] = state
            return {
                "sent": True,
                "reason": "Alert sent immediately" if bypass_cooldown else "Alert sent",
                "error": None,
                "phone": recipient_phone,
                "priority": priority,
                "from_phone": send_result["from_phone"],
            }
        return {
            "sent": False,
            "reason": str(send_result["reason"]),
            "error": send_result["error"],
            "phone": recipient_phone,
            "priority": priority,
            "from_phone": send_result["from_phone"],
        }

    @staticmethod
    def _build_message(city: str, flood_probability: float, risk_level: str) -> str:
        probability_percent = round(flood_probability * 100, 2)
        return (
            "EARLY FLOOD WARNING!\n"
            f"City: {city}\n"
            f"Risk Level: {risk_level}\n"
            f"Flood Probability: {probability_percent}%\n"
            "Stay alert and monitor updates."
        )

    @staticmethod
    def _recipient_key(phone: str | None, suffix: str) -> str:
        normalized_phone = phone or "unknown"
        normalized_suffix = suffix.strip().lower() if suffix else "default"
        return f"{normalized_phone}:{normalized_suffix}"

    def _send_sms(
        self,
        *,
        body: str,
        recipient_phone: str,
        context: str,
        priority: str,
    ) -> dict[str, Any]:
        sid, token, from_phone = self._resolve_config()
        sid_display = self._mask_secret(sid)
        token_display = self._mask_secret(token)
        from_display = from_phone or "<missing>"

        self._debug_log_config(
            f"Preparing SMS send ({context})",
            sid=sid,
            token=token,
            from_phone=from_phone,
        )
        logger.info(
            "Sending SMS priority=%s from=%s to=%s context=%s sid=%s token=%s",
            priority,
            from_display,
            recipient_phone,
            context,
            sid_display,
            token_display,
        )
        print(
            "TWILIO DEBUG:",
            {
                "account_sid": sid_display,
                "auth_token": token_display,
                "twilio_phone": from_display,
                "to_phone": recipient_phone,
                "context": context,
                "priority": priority,
            },
        )

        if not sid or not token:
            return {
                "sent": False,
                "reason": "Twilio credentials not configured",
                "error": "TWILIO_ACCOUNT_SID or TWILIO_AUTH_TOKEN is missing.",
                "from_phone": from_phone or None,
            }

        from_error = self._validate_phone("TWILIO_PHONE", from_phone)
        if from_error:
            return {
                "sent": False,
                "reason": "Twilio sender number invalid",
                "error": from_error,
                "from_phone": from_phone or None,
            }

        recipient_error = self._validate_phone("recipient phone", recipient_phone)
        if recipient_error:
            return {
                "sent": False,
                "reason": "Recipient phone invalid",
                "error": recipient_error,
                "from_phone": from_phone,
            }

        client = self._build_client(sid, token)
        try:
            sent_message = client.messages.create(
                body=body,
                from_=from_phone,
                to=recipient_phone,
            )
            logger.info("SMS sent successfully sid=%s to=%s from=%s priority=%s", sent_message.sid, recipient_phone, from_phone, priority)
            return {
                "sent": True,
                "reason": "Alert sent",
                "error": None,
                "from_phone": from_phone,
            }
        except Exception as exc:
            error_text = str(exc)
            logger.error("Twilio SMS send failed from=%s to=%s context=%s error=%s", from_phone, recipient_phone, context, error_text)
            return {
                "sent": False,
                "reason": self._classify_twilio_error(error_text),
                "error": error_text,
                "from_phone": from_phone,
            }

    def _resolve_config(self) -> tuple[str, str, str]:
        sid = os.getenv("TWILIO_ACCOUNT_SID", self.account_sid).strip()
        token = os.getenv("TWILIO_AUTH_TOKEN", self.auth_token).strip()
        from_phone = os.getenv("TWILIO_PHONE", self.from_phone).strip()
        return sid, token, from_phone

    def _build_client(self, sid: str, token: str) -> Client:
        identity = (sid, token)
        if self._client is None or self._client_identity != identity:
            self._client = Client(sid, token)
            self._client_identity = identity
        return self._client

    @staticmethod
    def _validate_phone(label: str, phone: str) -> str | None:
        if not phone:
            return f"{label} is missing."
        if not E164_PHONE_RE.match(phone):
            return f"{label} must be in E.164 format like '+919876543210'. Received: {phone!r}"
        return None

    @staticmethod
    def _classify_twilio_error(error_text: str) -> str:
        lowered = error_text.lower()
        if "mismatch" in lowered and "from" in lowered and "account" in lowered:
            return "Twilio 'From' number does not belong to the configured account"
        if "unverified" in lowered or "verify" in lowered:
            return "Recipient number is not verified for this Twilio account"
        if "trial" in lowered:
            return "Twilio trial restriction blocked the SMS"
        return "Twilio send failed"

    @staticmethod
    def _mask_secret(value: str) -> str:
        if not value:
            return "<missing>"
        if len(value) <= 8:
            return "*" * len(value)
        return f"{value[:4]}{'*' * (len(value) - 8)}{value[-4:]}"

    def _debug_log_config(self, message: str, *, sid: str, token: str, from_phone: str) -> None:
        sid_display = self._mask_secret(sid)
        token_display = self._mask_secret(token)
        from_display = from_phone or "<missing>"
        logger.info("%s sid=%s token=%s from_phone=%s", message, sid_display, token_display, from_display)
        print(
            "TWILIO DEBUG:",
            {
                "message": message,
                "account_sid": sid_display,
                "auth_token": token_display,
                "twilio_phone": from_display,
            },
        )
