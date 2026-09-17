"""``x-opencode-session`` — OpenCode relay session-affinity header.

OpenCode (opencode.ai Zen/Go/free relay) pins requests that share an
``x-opencode-session`` value to the same upstream backend, which is what
keeps its prompt cache warm across the turns of one conversation. The value
uses a persisted OpenCode-format identifier for free-tier access. Paid Zen/Go
requests retain Hermes's original routing scope and client identity.
Its routing scope is resolved like the other affinity hints Hermes already sends
(OpenRouter's sticky ``session_id``, xAI's ``x-grok-conv-id``): the
host-declared routing scope first, then the ambient conversation root, then
the physical session id — normalized through ``_cache_scope_from_session_id``
so cron fires of one job share a scope.

Every OpenCode request — main turn on any transport, auxiliary calls
(compression, titles, vision, MoA) — goes through :func:`opencode_session_headers`
so the header cannot drift per code path.
"""

from __future__ import annotations

import hashlib
import re
import secrets
import sqlite3
import threading
import time
from contextlib import closing
from typing import Any, Optional

OPENCODE_SESSION_HEADER = "x-opencode-session"
OPENCODE_USER_AGENT = "opencode/1.18.31"
_SESSION_PATTERN = re.compile(r"ses_[0-9a-f]{12}[0-9A-Za-z]{14}\Z")
_SESSION_LOCK = threading.Lock()
_LAST_TIMESTAMP = 0
_COUNTER = 0


def opencode_session_id(scope: str) -> str:
    """Reuse a persisted official-format ID for this profile's routing scope.

    SQLite's unique scope key and INSERT OR IGNORE make simultaneous processes
    agree on one ID. Store a scope hash rather than the original routing key.
    Native OpenCode IDs already carry their identity and pass through.
    """
    if _SESSION_PATTERN.fullmatch(scope):
        return scope
    from hermes_constants import get_hermes_home, mkdir_under_hermes_home

    home = mkdir_under_hermes_home(get_hermes_home())
    scope_hash = hashlib.sha256(scope.encode("utf-8")).hexdigest()
    with closing(sqlite3.connect(home / "opencode_sessions.db", timeout=10)) as db, db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS sessions "
            "(scope_hash TEXT PRIMARY KEY, session_id TEXT NOT NULL)"
        )
        row = db.execute("SELECT session_id FROM sessions WHERE scope_hash = ?", (scope_hash,)).fetchone()
        if row is None:
            db.execute(
                "INSERT OR IGNORE INTO sessions (scope_hash, session_id) VALUES (?, ?)",
                (scope_hash, _new_opencode_session_id()),
            )
            row = db.execute("SELECT session_id FROM sessions WHERE scope_hash = ?", (scope_hash,)).fetchone()
        return row[0]


def _new_opencode_session_id() -> str:
    """OpenCode's descending milliseconds/counter encoding and base62 suffix."""
    global _LAST_TIMESTAMP, _COUNTER
    with _SESSION_LOCK:
        now = time.time_ns() // 1_000_000
        if now != _LAST_TIMESTAMP:
            _LAST_TIMESTAMP, _COUNTER = now, 0
        _COUNTER += 1
        encoded = (~(now * 0x1000 + _COUNTER)) & ((1 << 48) - 1)
        alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        suffix = "".join(alphabet[b % 62] for b in secrets.token_bytes(14))
        return f"ses_{encoded:012x}{suffix}"


def is_opencode_target(provider: Optional[str], base_url: Optional[str]) -> bool:
    """True when *provider* or *base_url* addresses the OpenCode relay.

    Matches the built-in opencode-zen/go/free providers, custom
    ``opencode-<family>-*`` providers, and any base_url hosted on opencode.ai.
    """
    try:
        from hermes_cli.models import opencode_provider_family

        if opencode_provider_family(provider) is not None:
            return True
    except Exception:
        pass
    try:
        from agent.anthropic_endpoints import _is_opencode_endpoint

        return _is_opencode_endpoint(str(base_url or ""))
    except Exception:
        return False


def opencode_session_headers(
    provider: Optional[str],
    base_url: Optional[str],
    session_id: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> dict[str, str]:
    """Relay affinity for all OpenCode targets; compatibility metadata only for free access."""
    if not is_opencode_target(provider, base_url):
        return {}
    try:
        from agent.portal_tags import get_affinity_scope, get_conversation_context
        from agent.transports.codex import _cache_scope_from_session_id

        key = _cache_scope_from_session_id(
            # Top-level session_id → OpenRouter's sticky routing key. Per their prompt-caching docs it is
            # used directly as the routing key instead of hashing the opening messages, and it activates
            # stickiness on the first successful request rather than only after a cache hit. Resolve it from
            # the declared routing scope first (set only by a host that names its own conversation, #96811),
            # then the ambient conversation contextvar, with the explicit argument as fallback. The gap this
            # closes is the auxiliary call sites — compression, title generation, vision, web_extract,
            # session_search, MoA slots — which funnel through ``agent.auxiliary_client``. That module has
            # no session handle and passes no ``session_id``, so those calls sent NO sticky key at all and
            # each routed independently of the conversation it belonged to (#70820). Mirrors the Nous Portal
            # profile, which resolves the same way (f2f4df064d). The ambient value is the session-lineage
            # ROOT, so it also stays stable for installs that opt out of the default ``compression.in_place:
            # true`` and across delegate-subagent trees.
            get_affinity_scope() or get_conversation_context() or session_id
        )
    except Exception:
        key = str(session_id or "")
    if not is_opencode_free_target(provider, base_url, model, api_key):
        return {OPENCODE_SESSION_HEADER: key} if key else {}
    return {
        "User-Agent": OPENCODE_USER_AGENT,
        "x-opencode-client": "cli",
        OPENCODE_SESSION_HEADER: opencode_session_id(key or ""),
    }


def is_opencode_free_target(provider, base_url, model=None, api_key=None) -> bool:
    """Use the same free-tier classification for session metadata and tool adaptation."""
    if not is_opencode_target(provider, base_url):
        return False
    from hermes_cli.models import (
        OPENCODE_ZEN_FREE_KEYLESS_PLACEHOLDER,
        _opencode_free_known_model_slugs,
        normalize_opencode_model_id,
        opencode_provider_family,
    )

    return (
        opencode_provider_family(provider) == "opencode-free"
        or api_key == OPENCODE_ZEN_FREE_KEYLESS_PLACEHOLDER
        or normalize_opencode_model_id(provider, model).lower() in _opencode_free_known_model_slugs()
    )


def merge_opencode_session_headers(
    kwargs: dict[str, Any],
    provider: Optional[str],
    base_url: Optional[str],
    session_id: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> dict[str, Any]:
    """Merge the affinity header into ``kwargs["extra_headers"]`` (in place).

    Existing per-request headers win, so a caller-pinned value is preserved.
    Non-OpenCode targets are left untouched.
    """
    headers = opencode_session_headers(provider, base_url, session_id, model, api_key)
    if headers:
        existing = kwargs.get("extra_headers")
        merged = dict(existing) if isinstance(existing, dict) else {}
        for key, value in headers.items():
            merged.setdefault(key, value)
        kwargs["extra_headers"] = merged
    return kwargs
