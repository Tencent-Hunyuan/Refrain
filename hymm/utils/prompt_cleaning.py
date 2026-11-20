"""Shared prompt normalization used by training and inference."""

from __future__ import annotations

import html
import re
from typing import Any


# Accept the historical numeric form as well as malformed/empty payloads such
# as <ID:>, <ID:a>, or <ID:1,2>.  These tags are metadata and must never reach
# the text encoder.
_RAW_ID_TAG = r"<\s*ID\s*:\s*[^<>]*?>"
_ESCAPED_ID_TAG = r"&lt;\s*ID\s*:\s*(?:(?!&gt;).)*?&gt;"
_ID_TAG = rf"(?:{_RAW_ID_TAG}|{_ESCAPED_ID_TAG})"
_ID_CONNECTOR = r"(?:[,，、;；/&]+|\b(?:and|or)\b|和|与)"

_WRAPPED_ID_GROUP_RE = re.compile(
    rf"""
    [\(（]\s*
    {_ID_TAG}
    (?:\s*(?:{_ID_CONNECTOR}\s*)+\s*{_ID_TAG})*
    \s*(?:{_ID_CONNECTOR}\s*)*
    [\)）]
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)
_ID_TAG_RE = re.compile(_ID_TAG, flags=re.IGNORECASE)
# Repair common truncated tags found in historical captions, for example
# ``<ID:5.`` (missing ``>`` before sentence punctuation) and ``(<ID:2)``
# (missing ``>`` before the closing parenthesis).
_MALFORMED_ID_TAG_RE = re.compile(
    r"<\s*ID\s*:\s*[A-Za-z0-9_-]*(?:\s*[,，]\s*[A-Za-z0-9_-]+)*\s*>?",
    flags=re.IGNORECASE,
)
_EMPTY_ID_WRAPPER_RE = re.compile(
    rf"[\(（]\s*(?:{_ID_CONNECTOR}\s*)*[\)）]",
    flags=re.IGNORECASE,
)
_FPS_PREFIX_RE = re.compile(r"^FPS:\s*\d+(?:\.\d+)?\s*,\s*", flags=re.IGNORECASE)


def strip_identity_tags(text: Any, *, strip_fps_prefix: bool = False) -> Any:
    """Remove all ``<ID:...>`` metadata from one prompt string.

    Covered forms include:

    - ``(<ID:1>)``
    - ``(<ID:1>,<ID:2>)`` and arbitrarily longer lists
    - Chinese parentheses/commas
    - standalone tags
    - HTML-escaped tags
    - malformed or empty payloads such as ``<ID:>``

    Non-string values are returned unchanged so the helper is safe at shared
    tokenization boundaries.
    """

    if not isinstance(text, str):
        return text
    if not text:
        return text

    cleaned = html.unescape(text)
    if strip_fps_prefix:
        cleaned = _FPS_PREFIX_RE.sub("", cleaned)

    # Remove whole wrappers first.  Removing standalone tokens first would
    # leave punctuation artifacts such as "(,)" for multi-ID groups.
    previous = None
    while previous != cleaned:
        previous = cleaned
        cleaned = _WRAPPED_ID_GROUP_RE.sub("", cleaned)

    cleaned = _ID_TAG_RE.sub("", cleaned)
    cleaned = _MALFORMED_ID_TAG_RE.sub("", cleaned)
    cleaned = _EMPTY_ID_WRAPPER_RE.sub("", cleaned)

    # Normalize punctuation/spacing left around removed metadata.
    cleaned = re.sub(r"([,，])\s*(?:[,，]\s*)+", r"\1 ", cleaned)
    cleaned = re.sub(r",\s*\.", ".", cleaned)
    cleaned = re.sub(r"\.\s*,", ".", cleaned)
    cleaned = re.sub(r"\s+([,.;:!?，。；：！？])", r"\1", cleaned)
    cleaned = re.sub(r"([\(（\[])\s+", r"\1", cleaned)
    cleaned = re.sub(r"\s+([\)）\]])", r"\1", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = cleaned.strip()
    cleaned = re.sub(r"^[,，;；]\s*", "", cleaned)
    return cleaned


def strip_identity_tags_nested(value: Any, *, strip_fps_prefix: bool = False) -> Any:
    """Apply :func:`strip_identity_tags` to strings in a list/tuple."""

    if isinstance(value, str):
        return strip_identity_tags(value, strip_fps_prefix=strip_fps_prefix)
    if isinstance(value, list):
        return [
            strip_identity_tags_nested(item, strip_fps_prefix=strip_fps_prefix)
            for item in value
        ]
    if isinstance(value, tuple):
        return tuple(
            strip_identity_tags_nested(item, strip_fps_prefix=strip_fps_prefix)
            for item in value
        )
    return value
