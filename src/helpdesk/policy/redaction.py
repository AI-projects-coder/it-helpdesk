"""Redacts PII before logging or tracing."""
import re
from typing import Any

# Order matters: more specific patterns run first so a card number isn't
# partially eaten by the phone pattern.
_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("API_KEY", re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9_-]{16,}\b")),
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("CARD", re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{1,4}\b")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    # Requires separators so ids like CUST_05883 / TCKT_025902 are left alone.
    ("PHONE", re.compile(r"(?<!\w)(?:\+?\d{1,3}[ .-]?)?\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}\b")),
    ("IP", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("SECRET", re.compile(r"(?i)\b(password|passwd|pwd|token|secret)\s*[:=]\s*\S+")),
]

SENSITIVE_KEYS = {"password", "api_key", "token", "secret", "authorization"}


def redact(text: str) -> str:
    for label, pattern in _PATTERNS:
        if label == "SECRET":
            text = pattern.sub(lambda m: f"{m.group(1)}: [REDACTED_SECRET]", text)
        else:
            text = pattern.sub(f"[REDACTED_{label}]", text)
    return text


def redact_obj(obj: Any) -> Any:
    """Recursively redact strings in dicts/lists; mask values of sensitive keys."""
    if isinstance(obj, str):
        return redact(obj)
    if isinstance(obj, dict):
        return {
            k: "[REDACTED_SECRET]" if str(k).lower() in SENSITIVE_KEYS else redact_obj(v)
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return type(obj)(redact_obj(v) for v in obj)
    return obj
