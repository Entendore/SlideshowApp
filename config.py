# config.py
"""
Application Configuration Manager.

Singleton pattern with thread-safe initialization, auto-save on mutation,
and resolution validation (even dimensions required by FFmpeg yuv420p).
"""

from __future__ import annotations

import copy
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def _compute_app_root() -> str:
    """Compute the application root directory from this file's location."""
    return str(Path(__file__).resolve().parent)


class AppConfig:
    _instance: Optional[AppConfig] = None
    _initialized: bool = False
    APP_ROOT: str = ""  # Set lazily in __init__ to avoid import-time CWD dependency

    # ─── Resolution presets (single source of truth) ─────────────────────
    # Both Settings and New Project dialogs derive from these.
    RESOLUTION_PRESETS: Dict[str, Tuple[int, int]] = {
        "4K UHD (2160p)": (3840, 2160),
        "1440p QHD":      (2560, 1440),
        "1080p FHD":      (1920, 1080),
        "720p HD":        (1280,  720),
        "540p qHD":       ( 960,  540),
        "480p SD":        ( 854,  480),
        "360p":           ( 640,  360),
    }

    PORTRAIT_PRESETS: Dict[str, Tuple[int, int]] = {
        "4K UHD Portrait":    (2160, 3840),
        "1440p QHD Portrait": (1440, 2560),
        "1080p FHD Portrait": (1080, 1920),
        "720p HD Portrait":   ( 720, 1280),
        "540p qHD Portrait":  ( 540,  960),
        "480p SD Portrait":   ( 480,  854),
        "360p Portrait":      ( 360,  640),
    }

    ASPECT_RATIOS: Dict[str, Tuple[int, int]] = {
        "16:9": (16, 9),
        "4:3":  (4,  3),
        "1:1":  (1,  1),
        "21:9": (21, 9),
    }

    # ─── Defaults ────────────────────────────────────────────────────────────

    DEFAULT_CONFIG: Dict[str, Any] = {
        # General & Video
        "width": 1280, "height": 720, "fps": 30,
        "encoder": "auto", "preset": "fast",
        "transition_duration": 0.5, "theme": "Dark",
        # Project Management
        "projects_root": "Projects", "output_dir": "Output",
        # Render
        "render_workers": 3,
        # Hugging Face
        "hf_cache_dir": "",  # defaults to OS-appropriate location if empty
        "hf_datasets_dir": "",
        "hf_use_symlinks": False,
        "instruction_folder_root": "",
        # Active Backend
        "active_backend": "qwen3",
        # Qwen3
        "qwen3_mode": "voice_custom",
        "qwen3_size": "1.7B",
        "qwen3_device_map": "cuda:0",
        "qwen3_dtype": "float16",
        "qwen3_attn_implementation": "flash_attention_2",
        "qwen3_base_model_id": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "qwen3_voicedesign_model_id": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        "qwen3_custom_model_id": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "qwen3_speaker": "Vivian",
        "qwen3_instruct": "",
        "voice_references_root": "voicereferences",
        "qwen3_ref_audio": "", "qwen3_ref_text": "",
        "qwen3_vd_description": "", "qwen3_vd_save_name": "my_designed_voice",
        "qwen3_language": "English",
        # Edge TTS
        "edge_language": "English", "edge_voice": "en-US-JennyNeural",
        "edge_rate": "+0%", "edge_pitch": "+0Hz", "edge_volume": "+0%",
        # OmniVoice
        "omnivoice_mode": "auto",
        "omnivoice_model_id": "k2-fsa/OmniVoice",
        "omnivoice_language": "English",
        "omnivoice_ref_audio": "", "omnivoice_ref_text": "",
        "omnivoice_instruct": "", "omnivoice_use_batch": True,
        # Text Chunking
        "enable_text_chunking": True, "chunk_max_chars": 500,
        "chunk_max_sentences": 5, "chunk_min_chars": 50,
        "inter_slide_gap_ms": 80, "chunk_overlap_chars": 30,
        "chunk_crossfade_ms": 30, "chunk_warn_threshold": 1000,
        "chunk_strategy": "sentence",
        "chunk_max_duration_seconds": 0.0,
        # Silent / Image
        "silent_slide_duration": 5.0, "pdf_import_dpi": 200,
        # Persistent Workers
        # Logging
        "verbose": False,
        "log_console_level": "INFO",       # DEBUG | INFO | WARNING | ERROR
        "log_file_level": "DEBUG",         # DEBUG | INFO | WARNING | ERROR
        "log_file_path": "",               # empty = auto (logs/app.log), "off" = disabled
        "log_file_max_mb": 5,              # rotation size in MB
        "log_file_backups": 3,             # number of rotated backups
        "log_gui_level": "WARNING",        # DEBUG | INFO | WARNING | ERROR
    }

    _init_lock = threading.Lock()

    # ─── Singleton ───────────────────────────────────────────────────────────

    def __new__(cls) -> AppConfig:
        with cls._init_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._initialized = False
            return cls._instance

    def __init__(self) -> None:
        with AppConfig._init_lock:
            if AppConfig._initialized:
                return
            if not AppConfig.APP_ROOT:
                AppConfig.APP_ROOT = _compute_app_root()
            self._lock = threading.RLock()  # Reentrant to allow load() -> save() nesting
            self.config_path = str(Path(__file__).parent / "settings.json")
            self.settings: dict = copy.deepcopy(self.DEFAULT_CONFIG)
            self.load()
            self._sanitize_and_fix_resolution()
            self._validate_numeric_settings()
            AppConfig._initialized = True

    # ─── Resolution helpers ─────────────────────────────────────────────────

    @staticmethod
    def sanitize_resolution(width: int, height: int) -> Tuple[int, int]:
        """Ensure width and height are positive even integers (FFmpeg yuv420p)."""
        width  = max(64, int(width))
        height = max(64, int(height))
        if width  % 2: width  += 1
        if height % 2: height += 1
        return width, height

    def _sanitize_and_fix_resolution(self) -> None:
        w, h = self.sanitize_resolution(
            self.settings.get("width", 1280),
            self.settings.get("height", 720),
        )
        if w != self.settings.get("width") or h != self.settings.get("height"):
            logger.warning(
                f"[Settings] Resolution sanitized: "
                f"{self.settings.get('width')}×{self.settings.get('height')} → {w}×{h}"
            )
            self.settings["width"]  = w
            self.settings["height"] = h
            self.save()

    def set_resolution(self, width: int, height: int) -> None:
        """Set width + height atomically with validation."""
        width, height = self.sanitize_resolution(width, height)
        old_w, old_h = self.settings.get("width"), self.settings.get("height")
        self.settings["width"]  = width
        self.settings["height"] = height
        if old_w != width or old_h != height:
            logger.info(f"[Settings] Resolution: {old_w}×{old_h} → {width}×{height}")
        self.save()

    @classmethod
    def get_height_for_aspect_ratio(cls, width: int, ratio_name: str = "16:9") -> int:
        rw, rh = cls.ASPECT_RATIOS.get(ratio_name, (16, 9))
        _, height = cls.sanitize_resolution(width, int(width * rh / rw))
        return height

    @classmethod
    def get_width_for_aspect_ratio(cls, height: int, ratio_name: str = "16:9") -> int:
        rw, rh = cls.ASPECT_RATIOS.get(ratio_name, (16, 9))
        width, _ = cls.sanitize_resolution(int(height * rw / rh), height)
        return width

    # ─── Core methods ────────────────────────────────────────────────────────

    def load(self) -> None:
        with self._lock:
            path = Path(self.config_path)
            if path.exists():
                try:
                    loaded = json.loads(path.read_text(encoding="utf-8"))
                    merged = copy.deepcopy(self.DEFAULT_CONFIG)
                    merged.update(loaded)
                    self.settings = merged
                    logger.info("[Settings] Configuration loaded.")
                except Exception as exc:
                    logger.error(f"[Settings] Load error: {exc}")
            else:
                # First run — write default config to disk
                self.save()

    def _validate_numeric_settings(self) -> None:
        """Clamp numeric config values to safe ranges after loading."""
        _VALIDATORS = {
            "fps": (1, 120),
            "render_workers": (1, 16),
            "chunk_max_chars": (1, 10000),
            "chunk_max_sentences": (1, 50),
            "chunk_min_chars": (0, 5000),
            "inter_slide_gap_ms": (0, 2000),
            "chunk_overlap_chars": (0, 500),
            "chunk_crossfade_ms": (0, 500),
            "transition_duration": (0.0, 10.0),
            "silent_slide_duration": (0.5, 300.0),
            "pdf_import_dpi": (72, 600),
            "chunk_max_duration_seconds": (0.0, 120.0),
        }
        for key, (lo, hi) in _VALIDATORS.items():
            val = self.settings.get(key)
            if val is not None:
                try:
                    clamped = max(lo, min(hi, float(val)))
                    if isinstance(self.DEFAULT_CONFIG.get(key), int):
                        clamped = int(clamped)
                    if clamped != val:
                        logger.warning(
                            f"[Settings] Clamped '{key}': {val} -> {clamped}"
                        )
                        self.settings[key] = clamped
                except (ValueError, TypeError):
                    pass

    def save(self) -> None:
        try:
            with self._lock:
                path = Path(self.config_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(self.settings, indent=4, ensure_ascii=False),
                    encoding="utf-8",
                )
        except Exception as exc:
            logger.error(f"[Settings] Save error: {exc}")

    def _save_snapshot(self, snapshot: dict) -> None:
        """Write a settings snapshot to disk without acquiring the lock.

        Used by set() and update() after they have already taken a
        snapshot while holding the lock, avoiding TOCTOU races.
        """
        try:
            path = Path(self.config_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(snapshot, indent=4, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.error(f"[Settings] Save error: {exc}")

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self.settings.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a value with validation and auto-save."""
        if key in ("width", "height"):
            try:
                value = int(value)
            except (ValueError, TypeError):
                logger.error(f"[Settings] Invalid {key} value: {value!r}")
                return
            value = max(64, value)
            if value % 2:
                adjusted = value + 1
                logger.warning(
                    f"[Settings] Adjusted '{key}' {value} → {adjusted} (even required)"
                )
                value = adjusted

        with self._lock:
            old_value = self.settings.get(key)
            self.settings[key] = value
            changed = (old_value != value)
            if changed:
                # Snapshot settings while holding the lock to prevent
                # lost updates from concurrent mutations.
                snapshot = copy.deepcopy(self.settings)

        if changed:
            def _fmt(v: Any, max_len: int = 80) -> str:
                s = repr(v)
                return s if len(s) <= max_len else s[:max_len - 3] + "..."
            logger.info(
                f"[Settings] Changed '{key}': {_fmt(old_value)} -> {_fmt(value)}"
            )
            self._save_snapshot(snapshot)

    def update(self, data: dict) -> None:
        """Batch-update with resolution validation and change detection."""
        for dim_key in ("width", "height"):
            if dim_key in data:
                try:
                    data[dim_key] = int(data[dim_key])
                except (ValueError, TypeError):
                    logger.error(
                        f"[Settings] Invalid {dim_key} in batch update: "
                        f"{data[dim_key]!r}"
                    )
                    data.pop(dim_key)

        if "width" in data or "height" in data:
            with self._lock:
                w = data.get("width", self.settings.get("width", 1280))
                h = data.get("height", self.settings.get("height", 720))
            w, h = self.sanitize_resolution(w, h)
            data["width"]  = w
            data["height"] = h
        with self._lock:
            changed = any(self.settings.get(k) != v for k, v in data.items())
            if not changed:
                return
            self.settings.update(data)
            snapshot = copy.deepcopy(self.settings)
        logger.info(
            f"[Settings] Batch updated {len(data)} key(s)"
        )
        self._save_snapshot(snapshot)