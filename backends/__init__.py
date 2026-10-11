# backends/__init__.py
"""
Backend registry and factory.

Backends are registered by string key in BACKEND_MAP.  The factory
``get_backend`` instantiates and initializes a backend from its key.
"""

from __future__ import annotations

import logging
from typing import Dict, FrozenSet, List, Type

from .base import BaseTTSBackend
from .qwen3 import Qwen3Backend
from .edge import EdgeTTSBackend
from .omnivoice import OmniVoiceBackend
from utils import BackendInitializationError

logger = logging.getLogger(__name__)

# ─── Registry ────────────────────────────────────────────────────────────────
# Typed mapping of string names → backend classes.
# Used by the UI to populate dropdowns and the engine to instantiate workers.
BACKEND_MAP: Dict[str, Type[BaseTTSBackend]] = {
    "qwen3":    Qwen3Backend,
    "edge":     EdgeTTSBackend,
    "omnivoice": OmniVoiceBackend,
}

# Keys that are not owned by any single backend but still affect
# whether cached audio should be considered stale.
_GLOBAL_AUDIO_SETTINGS_KEYS: FrozenSet[str] = frozenset({"active_backend"})


# ─── Public helpers ──────────────────────────────────────────────────────────

def get_backend_info() -> dict[str, dict]:
    """Return a mapping of backend_name → {available, description}."""
    info = {}
    for name, cls in BACKEND_MAP.items():
        info[name] = {
            "available": cls.is_available(),
            "description": getattr(cls, "DESCRIPTION", name),
        }
    return info

def get_backend_descriptions() -> dict[str, str]:
    """Build a description dictionary from all registered backends.

    Returns:
        dict: Mapping of backend_name → description string.
              Always includes a "default" key as fallback.
    """
    descriptions: dict[str, str] = {"default": BaseTTSBackend.DESCRIPTION}
    for name, cls in BACKEND_MAP.items():
        descriptions[name] = getattr(cls, "DESCRIPTION", BaseTTSBackend.DESCRIPTION)
    return descriptions


def get_all_audio_settings_keys() -> list[str]:
    """Collect every config key that can invalidate cached audio.

    Combines global keys and per-backend keys, deduplicating while
    preserving insertion order.
    """
    seen: set[str] = set()
    result: list[str] = []

    # Global keys first
    for key in _GLOBAL_AUDIO_SETTINGS_KEYS:
        if key not in seen:
            seen.add(key)
            result.append(key)

    # Per-backend keys
    for cls in BACKEND_MAP.values():
        for key in getattr(cls, "AUDIO_SETTINGS_KEYS", []):
            if key not in seen:
                seen.add(key)
                result.append(key)

    return result


def get_backend(backend_name: str, config: dict) -> BaseTTSBackend:
    """Factory: return an initialized backend instance.

    Args:
        backend_name: Key in BACKEND_MAP (e.g. ``"qwen3"``).
        config:       Application configuration dictionary.

    Returns:
        A fully initialized BaseTTSBackend ready for ``generate_batch``.

    Raises:
        ValueError:  If the backend name is unknown.
        BackendInitializationError: If the backend fails to initialize.
    """
    backend_class = BACKEND_MAP.get(backend_name)

    if not backend_class:
        available = list(BACKEND_MAP.keys())
        raise ValueError(
            f"Backend '{backend_name}' not found. Available: {available}"
        )

    logger.info(f"[Backend Factory] Instantiating '{backend_name}'...")
    instance = backend_class(config)

    try:
        instance.initialize()
        logger.info(f"[Backend Factory] '{backend_name}' initialized successfully")
        return instance
    except Exception as e:
        # Attempt cleanup to prevent VRAM / resource leaks
        logger.error(f"[Backend Factory] Initialization failed for '{backend_name}': {e}")
        if hasattr(instance, "cleanup"):
            try:
                instance.cleanup()
            except Exception:
                pass  # Suppress cleanup errors during crash handling

        raise BackendInitializationError(
            f"Failed to initialize backend '{backend_name}': {e}"
        ) from e