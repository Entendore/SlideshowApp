# utils.py
"""
Core utilities, data structures, styling, and file import/export logic.

Handles manifest management, FFmpeg detection, silent audio generation,
theme management, and the heavy lifting of importing PDFs, PPTXs, 
and images into HTML/TXT slide pairs.
"""

from __future__ import annotations

import contextlib
import html
import io
import json
import logging
import logging.handlers
import os
import re
import shutil
import subprocess
import glob
import time
import warnings
import wave
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple, Union

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


# ═══════════════════════════════════════════════════════════════════════════════
#  SHARED FORMATTING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def human_bytes(n: int) -> str:
    """Return a human-friendly byte size string (e.g. '1.3 MB')."""
    if n < 0:
        return f"-{human_bytes(-n)}"
    if n < 1024:
        return f"{n} B"
    elif n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    elif n < 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MB"
    elif n < 1024 * 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024 * 1024):.1f} GB"
    else:
        return f"{n / (1024 * 1024 * 1024 * 1024):.2f} TB"


def fmt_duration(seconds: float) -> str:
    """Format seconds into human-readable hr:min:sec string.

    Examples:
        5.4      -> "5.4s"
        65.0     -> "1m 5s"
        3661.2   -> "1h 1m 1s"
    """
    if seconds < 0:
        return "0s"
    total = int(seconds)
    hrs, rem = divmod(total, 3600)
    mins, secs = divmod(rem, 60)

    if hrs:
        return f"{hrs}h {mins}m {secs}s"
    if mins:
        return f"{mins}m {secs}s"
    return f"{seconds:.1f}s"


class VerboseFilter(logging.Filter):
    """Suppresses noisy DEBUG messages from third-party libraries.

    When verbose logging is OFF this filter drops **all** DEBUG-level messages
    from chatty libraries (PIL, asyncio, urllib3, httpx, httpcore).
    INFO and above always pass through unchanged.

    When verbose is ON the filter is simply not added, so every message
    (including DEBUG) is visible.
    """

    def filter(self, record):
        # INFO and above always pass — only suppress DEBUG noise
        if record.levelno >= logging.INFO:
            return True

        # Suppress all DEBUG messages from noisy third-party loggers
        return False


# ═══════════════════════════════════════════════════════════════════════════════
#  LOGGING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LogConfig:
    """Structured configuration for all three log destinations.

    Each destination has its own level so you can e.g. keep the GUI clean
    (WARNING+) while the file captures everything (DEBUG+).

    Attributes:
        console_level:  Minimum level for terminal output.  Default: INFO.
        file_level:     Minimum level for the log file.  Default: DEBUG.
        file_path:      Path to the log file.  ``None`` disables file logging.
                        Relative paths are resolved against the app root.
        file_max_bytes: Max size per log file before rotation.  Default: 5 MB.
        file_backup_count: Number of rotated backup files to keep.  Default: 3.
        gui_level:      Minimum level for the GUI log widget (via QueueLogHandler).
                        Default: WARNING.  Set to INFO to see render progress
                        lines; set to DEBUG for full worker output.
        verbose:        If True, third-party library DEBUG noise is shown
                        on all handlers.  If False (default), a VerboseFilter
                        suppresses chatty loggers (PIL, asyncio, urllib3, ...).
    """
    console_level: Union[str, int] = logging.INFO
    file_level: Union[str, int] = logging.DEBUG
    file_path: Optional[str] = None
    file_max_bytes: int = 5 * 1024 * 1024   # 5 MB
    file_backup_count: int = 3
    gui_level: Union[str, int] = logging.WARNING
    verbose: bool = False

    # ── Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def from_dict(d: dict) -> "LogConfig":
        """Build a LogConfig from a flat settings dict.

        Recognised keys (all optional):
            log_console_level, log_file_level, log_file_path,
            log_file_max_mb, log_file_backups, log_gui_level, verbose.
        """
        def _level(key: str, default: int) -> int:
            val = d.get(key, default)
            if isinstance(val, str):
                return getattr(logging, val.upper(), default)
            return int(val)

        file_path = d.get("log_file_path")
        file_max_bytes = int(d.get("log_file_max_mb", 5)) * 1024 * 1024

        return LogConfig(
            console_level=_level("log_console_level", logging.INFO),
            file_level=_level("log_file_level", logging.DEBUG),
            file_path=file_path,
            file_max_bytes=file_max_bytes,
            file_backup_count=int(d.get("log_file_backups", 3)),
            gui_level=_level("log_gui_level", logging.WARNING),
            verbose=bool(d.get("verbose", False)),
        )


# Well-known tag used to identify handlers managed by setup_logging()
# so we can remove/replace them safely.
_HANDLER_TAG = "_slideshow_handler"


def _tag_handler(handler: logging.Handler, tag: str = _HANDLER_TAG) -> None:
    """Attach an invisible tag so we can identify our handlers later."""
    setattr(handler, _HANDLER_TAG, tag)


def _is_our_handler(handler: logging.Handler) -> bool:
    return getattr(handler, _HANDLER_TAG, None) == _HANDLER_TAG


def _remove_our_handlers(root: logging.Logger) -> list[logging.Handler]:
    """Remove all handlers placed by a previous setup_logging() call."""
    removed = [h for h in root.handlers if _is_our_handler(h)]
    for h in removed:
        root.removeHandler(h)
    return removed


# ── Named formatters ─────────────────────────────────────────────────

def _console_formatter() -> logging.Formatter:
    """Compact format for terminal — no date (assume terminal shows it)."""
    return logging.Formatter(
        "[%(asctime)s.%(msecs)03d] [%(levelname)-8s] %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def _file_formatter() -> logging.Formatter:
    """Compact format for log file — PID, time, level, logger name."""
    return logging.Formatter(
        "[%(process)d] [%(asctime)s.%(msecs)03d] [%(levelname)-8s] "
        "[%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def setup_logging(
    log_config: LogConfig | dict | None = None,
    **kwargs,
) -> None:
    """Configure the root logger with independent levels per destination.

    Can be called multiple times safely — previous handlers are replaced.

    Args:
        log_config: A ``LogConfig`` instance, a plain dict (passed to
                    ``LogConfig.from_dict``), or ``None`` for defaults.
        **kwargs:   Legacy compat — ``level`` or ``log_file`` are mapped
                    into a default LogConfig.
    """
    root = logging.getLogger()

    # ── Resolve config ────────────────────────────────────────────────
    if isinstance(log_config, dict):
        log_config = LogConfig.from_dict(log_config)

    if log_config is None:
        # Legacy positional/keyword support
        level = kwargs.get("level", logging.INFO)
        log_file = kwargs.get("log_file", None)
        log_config = LogConfig(
            console_level=level,
            file_level=level,
            file_path=log_file,
        )

    # Root logger must accept the most permissive level we need
    root.setLevel(logging.DEBUG)

    # ── Remove previous handlers (allow safe re-configuration) ───────
    _remove_our_handlers(root)

    # ── 1. Console handler ────────────────────────────────────────────
    console = logging.StreamHandler()
    console.setLevel(log_config.console_level)
    console.setFormatter(_console_formatter())
    _tag_handler(console)
    root.addHandler(console)

    # ── 2. File handler (rotating) ────────────────────────────────────
    if log_config.file_path:
        log_dir = os.path.dirname(log_config.file_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            log_config.file_path,
            mode="w",
            maxBytes=log_config.file_max_bytes,
            backupCount=log_config.file_backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(log_config.file_level)
        file_handler.setFormatter(_file_formatter())
        _tag_handler(file_handler)
        root.addHandler(file_handler)

    # ── 3. Verbose filter for third-party noise ──────────────────────
    _NOISY_LOGGERS = [
        "PIL.PngImagePlugin", "PIL.Image", "asyncio",
        "urllib3.connectionpool", "httpx", "httpcore",
        "PIL", "py.warnings",
    ]
    for name in _NOISY_LOGGERS:
        lib_log = logging.getLogger(name)
        lib_log.filters = [
            f for f in lib_log.filters if not isinstance(f, VerboseFilter)
        ]
    if not log_config.verbose:
        vf = VerboseFilter()
        for name in _NOISY_LOGGERS:
            logging.getLogger(name).addFilter(vf)

    # ── 4. Redirect Python warnings into logging ──────────────────────
    _warn_logger = logging.getLogger("py.warnings")

    def _log_warning(message, category, filename, lineno, file=None, line=None):
        _warn_logger.debug(
            "%s: %s (%s:%s)",
            category.__name__, message, filename, lineno,
        )

    warnings.showwarning = _log_warning



_SLIDE_FILE_RE = re.compile(r"slide(\d+)\.html$", re.IGNORECASE)

# ═══════════════════════════════════════════════════════════════════════════════
#  DATA STRUCTURES & MESSAGES
# ═══════════════════════════════════════════════════════════════════════════════

class RenderPhase(str, Enum):
    """Fine-grained render pipeline phases with UI-friendly labels and progress ranges.

    Each phase has a `label` (shown in the UI) and a `progress_range` tuple
    `(start_pct, end_pct)` used by the GUI to produce smooth, continuous
    progress bar motion instead of discrete jumps.

    Sub-phases within a major stage are nested as (stage, detail) pairs.
    """
    # ── Stage 1: Scanning (0% – 5%) ────────────────────────────────────
    SCANNING          = "Scanning"

    # ── Stage 2: Audio (5% – 50%) ─────────────────────────────────────
    AUDIO_INIT        = "Audio Init"           # loading backend / model
    AUDIO_GENERATING  = "Generating Audio"     # TTS in progress
    AUDIO_COMPLETE    = "Audio Complete"       # all clips ready

    # ── Stage 3: Timeline (50% – 55%) ─────────────────────────────────
    BUILDING_TIMELINE = "Building Timeline"

    # ── Stage 4: Video (55% – 92%) ────────────────────────────────────
    VIDEO_STARTING    = "Video Starting"       # launching FFmpeg
    VIDEO_RENDERING   = "Rendering Video"      # frame-by-frame capture
    VIDEO_MUXING      = "Muxing Audio"         # combining audio + video

    # ── Stage 5: Finalizing (92% – 100%) ──────────────────────────────
    FINALIZING        = "Finalizing"
    THUMBNAIL         = "Generating Thumbnail"

    # ── Special states ────────────────────────────────────────────────
    CACHE_SKIP        = "Cache Hit"            # video up-to-date, skipped


# Phase → (ui_label, start_pct, end_pct)
# Provides a smooth 0→100% progress curve across the entire pipeline.
PHASE_PROGRESS_MAP: dict[str, tuple[str, int, int]] = {
    RenderPhase.SCANNING:          ("Scanning project files...",        0,   5),
    RenderPhase.AUDIO_INIT:        ("Initializing audio backend...",    5,  10),
    RenderPhase.AUDIO_GENERATING:  ("Generating audio...",             10,  48),
    RenderPhase.AUDIO_COMPLETE:    ("Audio generation complete",       48,  50),
    RenderPhase.BUILDING_TIMELINE: ("Assembling timeline...",          50,  55),
    RenderPhase.VIDEO_STARTING:    ("Starting video encoder...",       55,  58),
    RenderPhase.VIDEO_RENDERING:   ("Rendering video frames...",       58,  90),
    RenderPhase.VIDEO_MUXING:      ("Muxing audio & video...",         90,  92),
    RenderPhase.FINALIZING:        ("Finalizing output...",            92,  97),
    RenderPhase.THUMBNAIL:         ("Generating thumbnail...",         97,  99),
    RenderPhase.CACHE_SKIP:        ("Cache hit — skipping video",     50,  50),
    # Legacy aliases (backward compat with persisted manifests)
    "Processing Audio":   ("Processing audio...",               5,  50),
    "Generating Audio":   ("Generating audio...",              10,  48),
}


def get_phase_progress(phase: str) -> tuple[str, int, int]:
    """Return (label, start_pct, end_pct) for a phase string.

    Falls back gracefully for legacy phase values.
    """
    if phase in PHASE_PROGRESS_MAP:
        return PHASE_PROGRESS_MAP[phase]
    # Try direct enum lookup for any new phase not yet in the map
    try:
        enum_val = RenderPhase(phase)
        if enum_val in PHASE_PROGRESS_MAP:
            return PHASE_PROGRESS_MAP[enum_val]
    except ValueError:
        pass
    return ("Processing...", 0, 100)

class RenderStatus(str, Enum):
    NEW = "New"
    PENDING = "Pending"
    RENDERING = "Rendering"
    READY = "Ready"
    ERROR = "Error"
    OUTDATED = "Outdated"

STATUS_SORT_PRIORITY = {
    RenderStatus.RENDERING: 0,
    RenderStatus.ERROR: 1,
    RenderStatus.OUTDATED: 2,
    RenderStatus.NEW: 3,
    RenderStatus.PENDING: 4,
    RenderStatus.READY: 5,
}

@dataclass
class PhaseMessage:
    project_name: str
    phase: str
    label: str = ""           # Override UI label (auto-resolved from PHASE_PROGRESS_MAP if empty)
    progress_pct: int = -1    # Override progress (auto-resolved from phase range if -1)
    elapsed: float = 0.0      # Seconds elapsed since render start (0 = unknown)

@dataclass
class SlideProgressMessage:
    project_name: str
    current: int
    total: int

@dataclass
class ChunkProgressMessage:
    project_name: str
    current: int
    total: int
    message: str = ""
    slide_count: int = 0  # actual number of slides (vs total TTS segments)

@dataclass
class ErrorMessage:
    project_name: str
    error_text: str

@dataclass
class FinalizedMessage:
    project_name: str

@dataclass
class LogMessage:
    project_name: str
    level: str
    text: str
    phase: str = ""         # Optional: current render phase for context coloring


# ═══════════════════════════════════════════════════════════════════════════════
#  CUSTOM EXCEPTIONS
# ═══════════════════════════════════════════════════════════════════════════════

class BackendGenerationError(Exception): pass
class FFmpegError(Exception): pass
class VideoRenderError(Exception): pass
class BackendInitializationError(Exception): pass


# ═══════════════════════════════════════════════════════════════════════════════
#  FILE FILTERS
# ═══════════════════════════════════════════════════════════════════════════════

FILE_FILTER_IMPORT = "Importable Files (*.pdf *.pptx *.txt *.csv *.json *.png *.jpg *.jpeg);;All Files (*)"
FILE_FILTER_SCRIPT = "Script Files (*.txt *.json *.pdf);;Text Files (*.txt);;All Files (*)"
FILE_FILTER_AUDIO = "Audio Files (*.wav *.mp3 *.ogg *.flac);;WAV Files (*.wav);;All Files (*)"
FILE_FILTER_IMAGE = "Image Files (*.png *.jpg *.jpeg);;All Files (*)"
FILE_FILTER_VIDEO = "Video Files (*.mp4 *.avi *.mov);;All Files (*)"

# ═══════════════════════════════════════════════════════════════════════════════
#  STYLING & PALETTE
# ═══════════════════════════════════════════════════════════════════════════════

class Palette:
    BG = "#1e1e1e"
    BG_LIGHT = "#252526"
    BG_DARK = "#181818"
    TEXT = "#d4d4d4"
    TEXT_DIM = "#808080"
    ACCENT = "#0078d4"
    ACCENT_HOVER = "#1a8ae8"
    BORDER = "#3c3c3c"
    ERROR = "#f44747"
    SUCCESS = "#6a9955"
    WARNING = "#FFA500"
    
    LOG_DEBUG = "#608b4e"
    LOG_INFO = "#d4d4d4"
    LOG_WARN = "#cca700"
    LOG_ERROR = "#f44747"
    LOG_CRITICAL = "#ff0000"

def _base_style() -> str:
    return f"color: {Palette.TEXT}; background-color: {Palette.BG_LIGHT}; border: 1px solid {Palette.BORDER};"

def style_input(padding="6px", radius="4px") -> str:
    return f"{_base_style()} padding: {padding}; border-radius: {radius};"

def style_input_wide(padding="6px", radius="4px") -> str:
    return style_input(padding, radius) + " min-width: 180px;"

def style_button(padding="6px 12px", radius="4px") -> str:
    return (
        f"color: {Palette.TEXT}; background-color: {Palette.BG_DARK}; "
        f"border: 1px solid {Palette.BORDER}; padding: {padding}; border-radius: {radius};"
    )

def style_button_small() -> str:
    return style_button(padding="3px 8px", radius="3px")

def style_primary_button() -> str:
    return (
        f"color: white; background-color: {Palette.ACCENT}; border: none; "
        f"padding: 6px 12px; border-radius: 4px; font-weight: bold;"
    )

def style_blue_button() -> str:
    return style_primary_button()

def style_green_button() -> str:
    return (
        f"color: white; background-color: {Palette.SUCCESS}; border: none; "
        f"padding: 6px 12px; border-radius: 4px; font-weight: bold;"
    )

def style_stop_button() -> str:
    return (
        f"color: white; background-color: {Palette.ERROR}; border: none; "
        f"padding: 6px 12px; border-radius: 4px; font-weight: bold;"
    )

def style_radio() -> str:
    return f"color: {Palette.TEXT};"

def style_checkbox() -> str:
    return f"color: {Palette.TEXT};"

def style_group_box() -> str:
    return (
        f"QGroupBox {{ color: {Palette.TEXT}; border: none; font-weight: bold; margin-top: 10px; }} "
        f"QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 5px; }}"
    )

def style_group_box_bordered() -> str:
    return (
        f"QGroupBox {{ color: {Palette.TEXT}; border: 1px solid {Palette.BORDER}; border-radius: 4px; "
        f"margin-top: 10px; padding-top: 15px; font-weight: bold; }} "
        f"QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 5px; }}"
    )

def style_summary_label() -> str:
    return f"color: {Palette.TEXT_DIM}; font-style: italic;"

def style_log_text() -> str:
    return (
        f"background-color: {Palette.BG_DARK}; color: {Palette.TEXT}; "
        f"border: 1px solid {Palette.BORDER}; font-family: 'Consolas', 'Courier New', monospace;"
    )

def style_context_menu() -> str:
    return (
        f"QMenu {{ background-color: {Palette.BG_LIGHT}; color: {Palette.TEXT}; border: 1px solid {Palette.BORDER}; }} "
        f"QMenu::item {{ padding: 5px 20px; }} "
        f"QMenu::item:selected {{ background-color: {Palette.ACCENT}; }}"
    )

def list_widget_stylesheet_from_theme() -> str:
    return (
        f"QListWidget {{ background-color: {Palette.BG_DARK}; border: 1px solid {Palette.BORDER}; }} "
        f"QListWidget::item {{ padding: 5px; }}"
    )

def project_list_stylesheet() -> str:
    return list_widget_stylesheet_from_theme() + (
        "QListWidget::item { background: transparent; border: none; }"
        "QListWidget::item:selected { background: transparent; border: none; }"
        "QListWidget::item:hover   { background: transparent; border: none; }"
    )

def style_page_widget() -> str:
    """Styles the base container widget for a tab or page."""
    return (
        f"QWidget {{ background-color: {Palette.BG_DARK}; color: {Palette.TEXT}; border: none; }}"
    )

def style_scroll_area() -> str:
    """Styles QScrollArea to match the dark theme."""
    return (
        f"QScrollArea {{ background-color: {Palette.BG_DARK}; border: none; }}"
        f"QScrollBar:vertical {{ background: {Palette.BG_DARK}; width: 10px; margin: 0px; }}"
        f"QScrollBar::handle:vertical {{ background: {Palette.BORDER}; min-height: 20px; border-radius: 5px; }}"
        f"QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}"
    )

def style_tab_widget() -> str:
    """Styles QTabWidget panes."""
    return (
        f"QTabWidget::pane {{ border: 1px solid {Palette.BORDER}; background: {Palette.BG_DARK}; }}"
    )

def style_label_header() -> str:
    """Styles large header labels."""
    return f"color: {Palette.TEXT}; font-size: 14px; font-weight: bold;"

def style_label_subtitle() -> str:
    """Styles smaller descriptive labels."""
    return f"color: {Palette.TEXT_DIM}; font-size: 11px;"

def style_list_widget() -> str:
    """Styles generic QListWidgets."""
    return (
        f"QListWidget {{ background-color: {Palette.BG_DARK}; border: 1px solid {Palette.BORDER}; "
        f"color: {Palette.TEXT}; outline: none; }}"
        f"QListWidget::item {{ padding: 5px; }}"
        f"QListWidget::item:selected {{ background-color: {Palette.ACCENT}; }}"
    )

def style_progress_bar() -> str:
    """Styles the QProgressBar."""
    return (
        f"QProgressBar {{ border: 1px solid {Palette.BORDER}; border-radius: 4px; "
        f"background-color: {Palette.BG_DARK}; text-align: center; color: {Palette.TEXT}; }}"
        f"QProgressBar::chunk {{ background-color: {Palette.ACCENT}; border-radius: 3px; }}"
    )

def style_tool_button() -> str:
    """Styles flat tool buttons."""
    return (
        f"QToolButton {{ background: transparent; border: none; color: {Palette.TEXT}; "
        f"padding: 4px; }}"
        f"QToolButton:hover {{ background-color: {Palette.BG_LIGHT}; }}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  SETTINGS-SPECIFIC STYLE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def style_combo(padding="6px 10px", radius="4px") -> str:
    """Styles QComboBox for the settings UI with dropdown arrow."""
    return (
        f"QComboBox {{ color: {Palette.TEXT}; background-color: {Palette.BG_LIGHT}; "
        f"border: 1px solid {Palette.BORDER}; border-radius: {radius}; "
        f"padding: {padding}; min-height: 22px; }}"
        f"QComboBox:hover {{ border-color: {Palette.ACCENT}; }}"
        f"QComboBox::drop-down {{ border: none; width: 24px; }}"
        f"QComboBox::down-arrow {{ image: none; border-left: 5px solid transparent; "
        f"border-right: 5px solid transparent; border-top: 6px solid {Palette.TEXT_DIM}; "
        f"margin-right: 8px; }}"
        f"QComboBox QAbstractItemView {{ color: {Palette.TEXT}; "
        f"background-color: {Palette.BG_LIGHT}; border: 1px solid {Palette.BORDER}; "
        f"selection-background-color: {Palette.ACCENT}; selection-color: white; "
        f"outline: none; padding: 4px; }}"
    )

def style_spinbox(padding="4px 8px", radius="4px") -> str:
    """Styles QSpinBox / QDoubleSpinBox with up/down buttons."""
    return (
        f"QSpinBox, QDoubleSpinBox {{ color: {Palette.TEXT}; "
        f"background-color: {Palette.BG_LIGHT}; border: 1px solid {Palette.BORDER}; "
        f"border-radius: {radius}; padding: {padding}; min-height: 22px; }}"
        f"QSpinBox:hover, QDoubleSpinBox:hover {{ border-color: {Palette.ACCENT}; }}"
        f"QSpinBox::up-button, QDoubleSpinBox::up-button {{ "
        f"subcontrol-origin: border; subcontrol-position: top right; "
        f"width: 20px; border-left: 1px solid {Palette.BORDER}; border-bottom: 1px solid {Palette.BORDER}; "
        f"border-top-right-radius: {radius}; }}"
        f"QSpinBox::down-button, QDoubleSpinBox::down-button {{ "
        f"subcontrol-origin: border; subcontrol-position: bottom right; "
        f"width: 20px; border-left: 1px solid {Palette.BORDER}; "
        f"border-bottom-right-radius: {radius}; }}"
        f"QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{ "
        f"width: 8px; height: 8px; border-left: 4px solid transparent; "
        f"border-right: 4px solid transparent; border-bottom: 5px solid {Palette.TEXT_DIM}; }}"
        f"QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{ "
        f"width: 8px; height: 8px; border-left: 4px solid transparent; "
        f"border-right: 4px solid transparent; border-top: 5px solid {Palette.TEXT_DIM}; }}"
    )

def style_checkbox_rich() -> str:
    """Full-featured checkbox style for settings pages with indicator theming."""
    return (
        f"QCheckBox {{ color: {Palette.TEXT}; spacing: 8px; }}"
        f"QCheckBox::indicator {{ width: 16px; height: 16px; "
        f"border: 1px solid {Palette.BORDER}; border-radius: 3px; "
        f"background-color: {Palette.BG_LIGHT}; }}"
        f"QCheckBox::indicator:hover {{ border-color: {Palette.ACCENT}; }}"
        f"QCheckBox::indicator:checked {{ background-color: {Palette.ACCENT}; "
        f"border-color: {Palette.ACCENT}; "
        f"image: none; }}"
        f"QCheckBox::indicator:checked:hover {{ background-color: {Palette.ACCENT_HOVER}; "
        f"border-color: {Palette.ACCENT_HOVER}; }}"
    )

def style_settings_sidebar() -> str:
    """Styles the settings sidebar QListWidget with icons and hover/active states."""
    return (
        f"QListWidget {{ background-color: {Palette.BG_LIGHT}; border: none; "
        f"border-right: 1px solid {Palette.BORDER}; font-size: 13px; outline: none; }}"
        f"QListWidget::item {{ padding: 14px 16px; color: {Palette.TEXT_DIM}; "
        f"border: none; border-left: 3px solid transparent; }}"
        f"QListWidget::item:selected {{ background-color: {Palette.BG_DARK}; "
        f"color: {Palette.TEXT}; border-left: 3px solid {Palette.ACCENT}; }}"
        f"QListWidget::item:hover:!selected {{ background-color: {Palette.BG_DARK}; "
        f"color: {Palette.TEXT}; }}"
    )

def style_settings_search() -> str:
    """Styles the settings search bar."""
    return (
        f"QLineEdit {{ padding: 8px 12px; background-color: {Palette.BG_DARK}; "
        f"color: {Palette.TEXT}; border: none; border-bottom: 1px solid {Palette.BORDER}; "
        f"font-size: 13px; }}"
        f"QLineEdit:focus {{ border-bottom: 2px solid {Palette.ACCENT}; }}"
        f"QLineEdit::placeholder {{ color: {Palette.TEXT_DIM}; }}"
    )

def style_settings_page() -> str:
    """Styles a settings content page (QWidget inside the stacked widget)."""
    return (
        f"QWidget {{ background-color: {Palette.BG_DARK}; color: {Palette.TEXT}; }}"
    )

def style_settings_scroll() -> str:
    """Styles scroll areas inside settings pages."""
    return (
        f"QScrollArea {{ background-color: {Palette.BG_DARK}; border: none; }}"
        f"QScrollBar:vertical {{ background: {Palette.BG_DARK}; width: 8px; margin: 0px; }}"
        f"QScrollBar::handle:vertical {{ background: {Palette.BORDER}; min-height: 30px; "
        f"border-radius: 4px; }}"
        f"QScrollBar::handle:vertical:hover {{ background: {Palette.TEXT_DIM}; }}"
        f"QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}"
        f"QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: none; }}"
    )

def style_form_label() -> str:
    """Styles QFormLayout label (left column) for settings pages."""
    return (
        f"color: {Palette.TEXT_DIM}; font-size: 12px; padding-top: 4px;"
    )

def style_section_header() -> str:
    """Styles a section header label within settings pages."""
    return (
        f"color: {Palette.TEXT}; font-size: 15px; font-weight: bold; "
        f"padding: 8px 0px 4px 0px;"
    )

def style_section_desc() -> str:
    """Styles a section description label below a header."""
    return (
        f"color: {Palette.TEXT_DIM}; font-size: 12px; padding: 0px 0px 12px 0px;"
    )

def style_group_box_settings() -> str:
    """Styles QGroupBox for settings pages — subtle card-like appearance."""
    return (
        f"QGroupBox {{ color: {Palette.TEXT}; background-color: {Palette.BG_LIGHT}; "
        f"border: 1px solid {Palette.BORDER}; border-radius: 6px; "
        f"margin-top: 14px; padding: 16px; padding-top: 24px; font-weight: bold; font-size: 13px; }}"
        f"QGroupBox::title {{ subcontrol-origin: margin; left: 14px; padding: 0 6px; "
        f"color: {Palette.TEXT}; }}"
    )

def style_browse_button() -> str:
    """Styles the small 'Browse...' / '...' buttons in settings."""
    return (
        f"QPushButton {{ color: {Palette.TEXT}; background-color: {Palette.BG_LIGHT}; "
        f"border: 1px solid {Palette.BORDER}; border-radius: 4px; "
        f"padding: 5px 12px; min-width: 60px; }}"
        f"QPushButton:hover {{ border-color: {Palette.ACCENT}; color: {Palette.ACCENT}; }}"
        f"QPushButton:pressed {{ background-color: {Palette.ACCENT}; color: white; }}"
    )

def style_diagnostics_text() -> str:
    """Styles the QTextEdit used in the diagnostics page."""
    return (
        f"QTextEdit {{ background-color: {Palette.BG_LIGHT}; color: {Palette.TEXT}; "
        f"border: 1px solid {Palette.BORDER}; border-radius: 6px; "
        f"padding: 12px; font-size: 13px; }}"
        f"QTextEdit h3 {{ color: {Palette.ACCENT}; }}"
    )

def style_copy_button() -> str:
    """Styles the 'Copy to Clipboard' style button for diagnostics."""
    return (
        f"QPushButton {{ color: white; background-color: {Palette.ACCENT}; "
        f"border: none; border-radius: 4px; padding: 8px 16px; font-weight: bold; }}"
        f"QPushButton:hover {{ background-color: {Palette.ACCENT_HOVER}; }}"
        f"QPushButton:pressed {{ background-color: #005a9e; }}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  THEMES
# ═══════════════════════════════════════════════════════════════════════════════

_CURRENT_THEME_NAME = "Dark"

_THEMES = {
    "Dark": Palette,
}

def get_theme() -> dict:
    """Return the current theme as a dictionary with 'name' and 'palette' keys."""
    return {
        "name": _CURRENT_THEME_NAME,
        "palette": _THEMES.get(_CURRENT_THEME_NAME, Palette),
    }

def set_theme(name: str) -> None:
    """Set the active theme by name."""
    global _CURRENT_THEME_NAME
    if name in _THEMES:
        _CURRENT_THEME_NAME = name
        logger.info(f"[UI] Theme switched to '{name}'")
    else:
        logger.warning(f"[UI] Unknown theme '{name}', keeping '{_CURRENT_THEME_NAME}'")

def list_themes() -> list[str]:
    return list(_THEMES.keys())


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS & FILE UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def natural_sort_key(text: str) -> list:
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', str(text))]

def detect_ffmpeg() -> Optional[str]:
    """Detect FFmpeg executable."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    # Check common paths
    common_paths = [
        r"C:\ffmpeg\bin\ffmpeg.exe",
        "/usr/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
    ]
    for p in common_paths:
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None

def generate_silent_wav(output_path: str, duration: float) -> bool:
    """Generates a silent WAV file.

    FIX: Clamps duration to a minimum of 0.01s to avoid creating
    degenerate 0-byte WAV files that break downstream duration
    calculation (wave.open on a 0-frame WAV raises an exception).
    """
    try:
        # Guard against non-positive duration
        if duration <= 0:
            duration = 0.01
            logger.debug(f"Clamped silent WAV duration to {duration}s (was non-positive)")
        sample_rate = 44100
        num_channels = 1
        num_frames = int(sample_rate * duration)
        
        with wave.open(output_path, 'w') as w:
            w.setnchannels(num_channels)
            w.setsampwidth(2)  # 16-bit
            w.setframerate(sample_rate)
            # Write silent frames (zeros)
            w.writeframes(b'\x00\x00' * num_frames)
        return True
    except Exception as exc:
        logger.error(f"Failed to generate silent WAV ({type(exc).__name__}: {exc})", exc_info=True)
        return False

def is_image_slide(html_path: str) -> bool:
    """Checks if an HTML slide is just an image wrapper."""
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read().lower()
        # Simple heuristic: contains an img tag with object-fit contain/cover
        return '<img' in content and 'object-fit' in content
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════════════
#  HTML / SLIDE GENERATORS
# ═══════════════════════════════════════════════════════════════════════════════

def get_blank_slide_html() -> str:
    return """<!DOCTYPE html>
<html><head><style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  html, body { width: 100%; height: 100%; background: #000; color: #fff; display: flex; align-items: center; justify-content: center; }
</style></head><body></body></html>"""

def get_default_slide_html() -> str:
    return """<!DOCTYPE html>
<html><head><style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  html, body { width: 100%; height: 100%; background: #000; color: #fff; display: flex; align-items: center; justify-content: center; font-family: sans-serif; font-size: 4vw; padding: 5vw; }
</style></head><body><div class="content">Slide Content</div></body></html>"""

def get_image_slide_html(image_rel_path: str) -> str:
    safe_path = (image_rel_path
        .replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;"))
    return f"""<!DOCTYPE html>
<html><head><style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  html, body {{ width: 100%; height: 100%; overflow: hidden; background: #000; }}
  img {{ width: 100%; height: 100%; object-fit: contain; }}
</style></head><body>
  <img src="{safe_path}" />
</body></html>"""

def create_slide_file(project_path: str, slide_num: int, html_content: str, txt_content: str = "") -> None:
    """Creates the HTML and TXT files for a specific slide number."""
    html_path = os.path.join(project_path, f"slide{slide_num}.html")
    txt_path = os.path.join(project_path, f"slide{slide_num}.txt")
    
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(txt_content.strip())

def get_next_slide_number(project_path: str) -> int:
    """Finds the next available slide number in a project directory."""
    if not os.path.isdir(project_path):
        return 1
    existing = [0]
    for f in os.listdir(project_path):
        match = _SLIDE_FILE_RE.match(f)
        if match:
            existing.append(int(match.group(1)))
    return max(existing) + 1


# ═══════════════════════════════════════════════════════════════════════════════
#  PROJECT & MANIFEST MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def get_project_dirs() -> list[str]:
    """Returns a list of valid project directory paths."""
    from config import AppConfig
    config = AppConfig()
    projects_root = config.get("projects_root", "Projects")
    # Always resolve to absolute path — the default "Projects" is relative
    if not os.path.isabs(projects_root):
        projects_root = os.path.join(AppConfig.APP_ROOT, projects_root)

    if not os.path.exists(projects_root):
        return []

    dirs = []
    for entry in os.listdir(projects_root):
        full_path = os.path.join(projects_root, entry)
        if not os.path.isdir(full_path):
            continue
        # Use the module-level _SLIDE_FILE_RE constant
        try:
            has_slides = any(_SLIDE_FILE_RE.match(f) for f in os.listdir(full_path))
        except OSError:
            has_slides = False
        if has_slides:
            dirs.append(full_path)
    return sorted(dirs, key=natural_sort_key)

def initialize_project_files(project_path: str, config: dict = None) -> None:
    """Initializes a new project directory with a first blank slide."""
    os.makedirs(project_path, exist_ok=True)
    create_slide_file(project_path, 1, get_default_slide_html(), "")
    # Write a project manifest so the library can immediately display
    # slide count and source info for this brand-new project.
    project_name = os.path.basename(project_path)
    save_project_manifest(project_path, {
        "version": 2,
        "name": project_name,
        "title": None,
        "source": {
            "type": "blank",
            "file": None,
            "pdf_import_dpi": None,
        },
        "slide_count": 1,
        "status": "new",
        "created": _now_iso(),
    })
    # Also update the global library manifest
    update_library_manifest(project_path)

def get_library_manifest_path() -> str:
    from config import AppConfig
    return os.path.join(AppConfig.APP_ROOT, "library_manifest.json")

def load_project_manifest(project_path: str) -> dict:
    """Load a project manifest.

    Manifest schema:
        version: 2
        name: str
        title: str | null
        created: ISO timestamp
        modified: epoch (last render/import/verify timestamp)
        source: {type, file, pdf_import_dpi, resolution: {width, height, aspect}}
        status: "new" | "incomplete" | "not rendered" | "completed" | "error"
        slide_count: int
        render: {width, height, fps, video_hash, audio_hash}
        output: {video: filename}
        thumbnail: {file, width, height, hash, eligible}
        assets_meta: {filename: {mtime, size}}
    """
    manifest_path = os.path.join(project_path, "manifest.json")
    if not os.path.exists(manifest_path):
        return {}
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        logger.warning(f"Could not load manifest for {os.path.basename(project_path)} ({type(exc).__name__}: {exc})", exc_info=True)
        return {}

    return data


def get_source_resolution(manifest: dict) -> dict:
    """Extract source resolution from the manifest."""
    return manifest.get("source", {}).get("resolution") or {}


def get_source_type(manifest: dict) -> str:
    """Extract source type string from the manifest."""
    src = manifest.get("source", "")
    if isinstance(src, dict):
        return src.get("type", "")
    return str(src)


def get_render_hashes(manifest: dict) -> tuple:
    """Extract (video_hash, audio_hash) from the manifest."""
    r = manifest.get("render", {})
    return r.get("video_hash", ""), r.get("audio_hash", "")


def get_status(manifest: dict) -> str:
    """Extract status from the manifest."""
    return manifest.get("status", "new")


def set_source_resolution(project_path: str, width: int, height: int) -> None:
    """Write source resolution into the project manifest."""
    manifest = load_project_manifest(project_path)
    aspect = _aspect_ratio_label(width, height)
    if "source" not in manifest:
        manifest["source"] = {}
    manifest["source"]["resolution"] = {
        "width": width, "height": height, "aspect": aspect,
    }
    save_project_manifest(project_path, manifest)

def save_project_manifest(project_path: str, data: dict) -> None:
    manifest_path = os.path.join(project_path, "manifest.json")
    try:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except Exception as exc:
        logger.error(f"Failed to save manifest for {os.path.basename(project_path)} ({type(exc).__name__}: {exc})", exc_info=True)

def update_library_manifest(project_path: str) -> None:
    """Updates the global library manifest with current project status.

    The library manifest acts as a cache so refresh_library() can display
    rich info without re-scanning every project directory on disk.
    """
    lib_path = get_library_manifest_path()
    data = {}
    if os.path.exists(lib_path):
        try:
            with open(lib_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            logger.error(f"Failed to read library manifest, starting fresh ({type(exc).__name__}: {exc})", exc_info=True)
            data = {}

    project_name = os.path.basename(project_path)
    manifest = load_project_manifest(project_path)

    # Count slides from disk for accuracy
    try:
        slide_count = sum(1 for f in os.listdir(project_path) if _SLIDE_FILE_RE.match(f))
    except Exception:
        slide_count = manifest.get("slide_count", 0)

    entry = {
        "path": project_path,
        "slide_count": slide_count,
        "status": manifest.get("status", "new"),
        "last_rendered": manifest.get("modified", manifest.get("timestamp", 0)),
        "created": manifest.get("created", ""),
        "title": manifest.get("title") or project_name,
    }
    # Source type
    src = manifest.get("source", {})
    if isinstance(src, dict):
        entry["source"] = src.get("type", "")
    else:
        entry["source"] = str(src)
    # Propagate thumbnail metadata so the library knows about it
    if manifest.get("thumbnail"):
        entry["thumbnail"] = manifest["thumbnail"]
    # Propagate render section (width/height/fps + hashes)
    if manifest.get("render"):
        entry["render"] = manifest["render"]
    # Propagate source resolution (auto-detected or manually verified)
    src_res = get_source_resolution(manifest)
    if src_res:
        entry["source_resolution"] = src_res

    data[project_name] = entry

    try:
        with open(lib_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except Exception as exc:
        logger.error(f"Failed to update library manifest ({type(exc).__name__}: {exc})", exc_info=True)

def build_manifest_from_timeline(
    project_name: str, project_path: str, config: dict,
    timeline: list, output_dir: str, final_video: str,
    render_mode: str, assets_meta: dict,
    existing_manifest: dict | None = None,
) -> dict:
    """Constructs the v2 manifest dictionary for a completed render.

    Preserves import-time fields (``source``, ``created``, ``slide_count``)
    from *existing_manifest* so that a successful render does not wipe metadata
    added during import.
    """
    data = {
        "version": 2,
        "name": project_name,
        "modified": time.time(),
        "status": "completed",
        "slide_count": len(timeline),
        "output": {
            "video": os.path.basename(final_video),
        },
        "render": {
            "width": config.get("width"),
            "height": config.get("height"),
            "fps": config.get("fps"),
        },
        "assets_meta": assets_meta,
    }
    # Preserve import-time fields that would otherwise be lost
    if existing_manifest:
        for key in ("title", "created", "source"):
            if key in existing_manifest and existing_manifest[key]:
                data[key] = existing_manifest[key]
    return data

def get_library_status_data(project_dirs: list, active_procs: dict, pending: list) -> list:
    """Generates UI-friendly status data for projects.

    Status values are capitalized strings matching RenderStatus enum /
    StatusTheme keys: "New", "Outdated", "Pending", "Rendering", "Ready", "Error".
    """
    data = []
    active_paths = {p_info["path"] for p_info in active_procs.values()}
    pending_paths = set(pending)

    for path in project_dirs:
        name = os.path.basename(path)
        manifest = load_project_manifest(path)

        if path in active_paths:
            status = RenderStatus.RENDERING
            detail = "Rendering..."
        elif path in pending_paths:
            status = RenderStatus.PENDING
            detail = "Queued"
        elif manifest.get("status") == "completed":
            status = RenderStatus.READY
            detail = "Completed"
        elif manifest.get("status") == "new":
            status = RenderStatus.NEW
            detail = "New Project"
        else:
            status = RenderStatus.OUTDATED
            detail = "Needs Render"

        # Enrich with manifest data for better library display
        slide_count = manifest.get("slide_count")
        # If slide_count missing, count from disk
        if slide_count is None:
            slide_count = 0
            try:
                slide_count = sum(1 for f in os.listdir(path) if _SLIDE_FILE_RE.match(f))
            except Exception:
                pass

        data.append({
            "name": name,
            "path": path,
            "status": status,
            "detail": detail,
            "slide_count": slide_count,
            "source": get_source_type(manifest),
            "last_rendered": manifest.get("modified", manifest.get("timestamp", 0)),
            "created": manifest.get("created", ""),
            "source_resolution": get_source_resolution(manifest),
        })
    return data

def should_skip_render(project_path: str, config: dict) -> bool:
    """Checks if a project is already up-to-date with current settings.

    Uses video/audio setting hashes to determine if a re-render is needed.
    """
    manifest = load_project_manifest(project_path)
    if manifest.get("status") != "completed":
        return False

    # Lazy import to avoid circular dependency (engines imports utils)
    try:
        from engines import _get_audio_settings_hash, _get_video_settings_hash

        prev_video_hash, prev_audio_hash = get_render_hashes(manifest)

        # Check video settings hash (covers width, height, fps, encoder,
        # preset, transition_duration)
        current_video_hash = _get_video_settings_hash(config)
        if current_video_hash != prev_video_hash:
            return False

        # Check audio settings hash (covers backend, voice, language, etc.)
        current_audio_hash = _get_audio_settings_hash(config)
        if current_audio_hash != prev_audio_hash:
            return False
    except ImportError:
        # If engines can't be imported (e.g. missing torch/numpy),
        # assume render is needed to be safe.
        return False

    return True


# ═══════════════════════════════════════════════════════════════════════════════
#  IMPORT LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

def _create_import_slide_files(project_path: str, slide_num: int, img_rel_path: str, text_content: str = ""):
    """Helper specifically for image imports to generate the wrapper HTML and TXT."""
    html_content = get_image_slide_html(img_rel_path)
    create_slide_file(project_path, slide_num, html_content, text_content)

def import_image_as_slide(image_path: str, project_path: str, config: dict) -> int:
    """Imports a single image as a new slide at the end of the project."""
    images_dir = os.path.join(project_path, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    slide_num = get_next_slide_number(project_path)
    ext = os.path.splitext(image_path)[1]
    img_name = f"slide{slide_num}{ext}"
    img_path = os.path.join(images_dir, img_name)
    
    shutil.copy2(image_path, img_path)
    _create_import_slide_files(project_path, slide_num, f"images/{img_name}", "")
    
    logger.info(f"[Import] Image imported as slide {slide_num}")
    return 1


def _resolve_on_exists(project_path: str, project_name: str, on_exists: str) -> None:
    """Handle an existing project directory according to the on_exists policy.

    Args:
        on_exists: "error"  – raise ValueError (default, backward-compatible)
                   "overwrite" – delete the existing directory
                   "append"    – keep the directory; caller should use append logic
    """
    if not os.path.exists(project_path):
        return

    if on_exists == "overwrite":
        # Safety: verify we're inside projects_root to prevent path traversal
        real = os.path.realpath(project_path)
        real_root = os.path.realpath(os.path.dirname(project_path))
        if not real.startswith(real_root + os.sep) and real != real_root:
            raise ValueError(f"Refusing to delete path outside parent directory: {project_path}")
        shutil.rmtree(project_path)
        logger.info(f"[Import] Overwriting existing project: {project_name}")
    elif on_exists == "append":
        logger.info(f"[Import] Appending to existing project: {project_name}")
    else:  # "error" or any unknown value
        raise ValueError(
            f"Project '{project_name}' already exists. "
            f"Choose 'Overwrite' or 'Append' to proceed."
        )


def import_pdf_as_project(
    pdf_path: str,
    config: dict,
    project_name: str = None,
    on_exists: str = "error",
) -> str:
    """Import a PDF file as a new project, converting each page to a slide."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError(
            "PyMuPDF is required for PDF import. "
            "Install via: pip install PyMuPDF"
        )

    if not os.path.isfile(pdf_path):
        raise ValueError(f"PDF file not found: {pdf_path}")

    # Resolve project name
    if not project_name:
        project_name = os.path.splitext(os.path.basename(pdf_path))[0]

    # Create the project directory
    from config import AppConfig
    projects_root = config.get("projects_root", "Projects")
    if not os.path.isabs(projects_root):
        projects_root = os.path.join(AppConfig.APP_ROOT, projects_root)
    project_path = os.path.join(projects_root, project_name)

    _resolve_on_exists(project_path, project_name, on_exists)

    os.makedirs(project_path, exist_ok=True)

    # Get DPI setting
    dpi = config.get("pdf_import_dpi", 200)
    zoom = dpi / 72.0  # 72 is the default PDF resolution

    # ── Open the PDF and process pages ──────────────────────────────────
    # FIX #1: Suppress MuPDF's noisy C-level stderr warnings about
    # "No common ancestor in structure tree". These are non-fatal
    # structural warnings — the pages are still perfectly readable.
    doc = None
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            doc = fitz.open(pdf_path)

            if doc.is_encrypted:
                raise ValueError(
                    f"PDF '{os.path.basename(pdf_path)}' is encrypted and "
                    f"cannot be imported."
                )

            total_pages = len(doc)
            if total_pages == 0:
                raise ValueError(
                    f"PDF '{os.path.basename(pdf_path)}' has no pages."
                )

            logger.info(
                f"[Import] PDF has {total_pages} page(s), "
                f"rendering at {dpi} DPI"
            )

            images_dir = os.path.join(project_path, "images")
            os.makedirs(images_dir, exist_ok=True)

            for page_num in range(total_pages):
                page = doc[page_num]
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat, alpha=False)

                img_filename = f"slide{page_num + 1}.png"
                img_path = os.path.join(images_dir, img_filename)
                pix.save(img_path)
                pix = None  # Free memory immediately

                # Create the slide HTML (using same template as PDF append)
                _create_import_slide_files(
                    project_path, page_num + 1,
                    f"images/{img_filename}", ""
                )

                logger.debug(
                    f"[Import] Rendered page {page_num + 1}/{total_pages}"
                )

        logger.info(
            f"[Import] Successfully imported {total_pages} page(s) "
            f"from '{os.path.basename(pdf_path)}'"
        )

        # Save a project manifest for consistency with other import paths
        manifest_data = {
            "version": 2,
            "name": project_name,
            "source": {
                "type": "pdf",
                "file": os.path.basename(pdf_path),
                "pdf_import_dpi": dpi,
            },
            "slide_count": total_pages,
            "status": "new",
            "created": _now_iso(),
        }
        save_project_manifest(project_path, manifest_data)

        # Auto-detect source resolution and update config
        auto_set_resolution_from_source(project_path, config)

        # Also update the global library manifest (after resolution is set)
        update_library_manifest(project_path)

    except ImportError:
        raise
    except ValueError:
        raise
    except Exception as exc:
        logger.error(
            f"[Import] Failed to import PDF "
            f"'{os.path.basename(pdf_path)}': {exc}",
            exc_info=True,
        )
        raise ValueError(f"Failed to import PDF: {exc}") from exc
    finally:
        # FIX #2: Guarantee the document is closed, even on error
        if doc is not None:
            try:
                doc.close()
            except Exception:
                pass

    return project_path



def _create_image_slide_html(html_path: str, img_src: str, width: int = 1280):  # noqa: ARG001 – kept for API compatibility
    """Create a minimal HTML file that displays an image full-screen."""
    # FIX: Escape special HTML characters in img_src to prevent broken HTML
    # or attribute injection from paths containing quotes/angle brackets.
    safe_src = img_src.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")
    html_content = f"""<!DOCTYPE html>
<html>
<head>
<style>
  body {{
    margin: 0;
    padding: 0;
    background: #000;
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100vh;
    overflow: hidden;
  }}
  img {{
    max-width: 100%;
    max-height: 100vh;
    object-fit: contain;
  }}
</style>
</head>
<body>
  <img src="{safe_src}" alt="Slide">
</body>
</html>"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)


def detect_source_resolution(project_path: str) -> Optional[Tuple[int, int]]:
    """Detect the natural resolution/aspect ratio of the project's source content.

    Scans the first slide to find an underlying source image and returns its
    pixel dimensions.  This is used to auto-set the render resolution to match
    the source material (e.g. portrait 1080×1920, square 1080×1080).

    Returns (width, height) of the first source image, or None if detection
    fails (e.g. text-only slides with no images).
    """
    try:
        from PIL import Image as _PILImage
    except ImportError:
        return None

    # 1. Check for image slides (direct image wrapper HTML)
    html_files = sorted(
        glob.glob(os.path.join(project_path, "slide*.html")),
        key=natural_sort_key,
    )
    if not html_files:
        return None

    first_html = html_files[0]

    # Try to extract the image path from an image-wrapper slide
    direct_img = _get_direct_image_path_from_html(first_html, project_path)
    if direct_img and os.path.exists(direct_img):
        try:
            with _PILImage.open(direct_img) as img:
                return img.size
        except Exception:
            pass

    # 2. Check images/ directory for the first slide image
    images_dir = os.path.join(project_path, "images")
    if os.path.isdir(images_dir):
        img_files = sorted(
            glob.glob(os.path.join(images_dir, "slide1.*")),
        )
        if img_files:
            try:
                with _PILImage.open(img_files[0]) as img:
                    return img.size
            except Exception:
                pass

    return None


def _get_direct_image_path_from_html(html_path: str, project_path: str) -> Optional[str]:
    """Extract the image path from an image-wrapper HTML slide."""
    if not is_image_slide(html_path):
        return None
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            content = f.read()
        m = re.search(r'src=["\']([^"\']+)["\']', content)
        if m:
            src = html.unescape(m.group(1))
            if not os.path.isabs(src):
                src = os.path.join(project_path, src)
            return src
    except Exception:
        pass
    return None


def auto_set_resolution_from_source(
    project_path: str, config,
) -> Optional[Tuple[int, int]]:
    """Detect source resolution and record it in the project manifest.

    The detected resolution is stored under ``source.resolution`` in the
    manifest.  The *global config is NOT mutated* — each project keeps its
    own resolution preference in its manifest.  The render pipeline reads
    ``render.width/height`` (written after a successful render) or falls back
    to the global config, so there is no need to change the global setting
    per-import.

    Returns the detected (width, height) if found, or None.
    """
    detected = detect_source_resolution(project_path)
    if detected is None:
        return None

    src_w, src_h = detected
    _save_source_resolution(project_path, src_w, src_h)
    logger.info(
        f"[Auto-Detect] Source resolution detected: {src_w}×{src_h} → "
        f"recorded in manifest (global config unchanged)"
    )
    return (src_w, src_h)


def _save_source_resolution(
    project_path: str, width: int, height: int
) -> None:
    """Record the auto-detected source resolution in the project manifest."""
    set_source_resolution(project_path, width, height)


def _aspect_ratio_label(w: int, h: int) -> str:
    """Return a human-readable aspect ratio label like '16:9' or '9:16'."""
    from math import gcd
    g = gcd(w, h)
    rw, rh = w // g, h // g
    # Simplify common ratios
    common = {
        (16, 9): "16:9", (9, 16): "9:16",
        (4, 3): "4:3", (3, 4): "3:4",
        (1, 1): "1:1", (21, 9): "21:9",
    }
    return common.get((rw, rh), f"{rw}:{rh}")


def detect_html_resolution(project_path: str) -> Optional[Tuple[int, int]]:
    """Detect resolution from HTML slide container CSS.

    Looks for explicit pixel dimensions in the first slide's HTML:
      - ``width: <N>px`` and ``height: <N>px`` in inline styles
      - ``<meta name="viewport" content="width=<N>">`` tags
      - ``width`` / ``height`` attributes on the ``<html>`` or ``<body>``
      - Tailwind arbitrary value classes on ``.ppt-slide``:
        ``w-[<N>px]`` and ``min-h-[<N>px]`` / ``h-[<N>px]``

    Returns (width, height) if found, or None.
    """
    html_files = sorted(
        glob.glob(os.path.join(project_path, "slide*.html")),
        key=natural_sort_key,
    )
    if not html_files:
        return None

    try:
        with open(html_files[0], "r", encoding="utf-8") as f:
            content = f.read()
    except Exception:
        return None

    # 1. Check for width/height in inline style attributes
    #    Matches patterns like: style="...width: 1920px;...height: 1080px;..."
    w_match = re.search(r'(?:^|[\s;"])width\s*:\s*(\d+)\s*px', content, re.IGNORECASE)
    h_match = re.search(r'(?:^|[\s;"])height\s*:\s*(\d+)\s*px', content, re.IGNORECASE)
    if w_match and h_match:
        return (int(w_match.group(1)), int(h_match.group(1)))

    # 2. Check for viewport meta tag
    vp = re.search(
        r'<meta\s+name=["\']viewport["\']\s+content=["\']([^"\']+)["\']',
        content, re.IGNORECASE,
    )
    if vp:
        vw = re.search(r'width=(\d+)', vp.group(1))
        vh = re.search(r'height=(\d+)', vp.group(1))
        if vw and vh:
            return (int(vw.group(1)), int(vh.group(1)))

    # 3. Check for width/height attributes on html or body
    tag = re.search(
        r'<(?:html|body)\s+[^>]*width=["\']?(\d+)["\']?[^>]*height=["\']?(\d+)["\']?',
        content, re.IGNORECASE,
    )
    if tag:
        return (int(tag.group(1)), int(tag.group(2)))

    # 4. Check for Tailwind arbitrary value classes on .ppt-slide
    #    Detects patterns like:
    #      .ppt-slide { @apply relative w-[1280px] min-h-[720px] ... }
    #    or:
    #      class="ppt-slide w-[1920px] h-[1080px] ..."
    #    Extracts w-[<N>px] as width, min-h-[<N>px] or h-[<N>px] as height.
    # Arbitrary value regex: w-[<digits>px], min-h-[<digits>px], h-[<digits>px]
    tw_w = re.search(r'w-\[(\d+)px\]', content)
    tw_h = re.search(r'(?:min-h|h)-\[(\d+)px\]', content)
    if tw_w and tw_h:
        return (int(tw_w.group(1)), int(tw_h.group(1)))

    return None


def verify_project_resolution(project_path: str) -> dict:
    """Verify a project's resolution by inspecting source files.

    Checks, in order:
      1. PNG/JPG files in the ``images/`` directory
      2. Direct image referenced by image-wrapper HTML slides
      3. HTML container CSS for explicit pixel dimensions

    Returns a dict with:
      - ``resolution``: (w, h) tuple or None
      - ``method``: "png", "html_css", or None
      - ``details``: human-readable string
    """
    project_name = os.path.basename(project_path)
    result = {"resolution": None, "method": None, "details": ""}

    # Method 1: Inspect images/ directory
    images_dir = os.path.join(project_path, "images")
    if os.path.isdir(images_dir):
        img_files = sorted(glob.glob(os.path.join(images_dir, "slide1.*")))
        if not img_files:
            img_files = sorted(
                glob.glob(os.path.join(images_dir, "*"))
            )
        for img_path in img_files[:5]:  # Check up to 5 images
            try:
                from PIL import Image as _PILImage
                with _PILImage.open(img_path) as img:
                    w, h = img.size
                    result["resolution"] = (w, h)
                    result["method"] = "png"
                    result["details"] = (
                        f"Detected from {os.path.basename(img_path)}: "
                        f"{w}x{h} ({_aspect_ratio_label(w, h)})"
                    )
                    return result
            except Exception:
                continue

    # Method 2: Check direct image from HTML
    html_files = sorted(
        glob.glob(os.path.join(project_path, "slide*.html")),
        key=natural_sort_key,
    )
    if html_files:
        direct_img = _get_direct_image_path_from_html(
            html_files[0], project_path
        )
        if direct_img and os.path.exists(direct_img):
            try:
                from PIL import Image as _PILImage
                with _PILImage.open(direct_img) as img:
                    w, h = img.size
                    result["resolution"] = (w, h)
                    result["method"] = "png"
                    result["details"] = (
                        f"Detected from HTML image reference "
                        f"({os.path.basename(direct_img)}): {w}x{h} "
                        f"({_aspect_ratio_label(w, h)})"
                    )
                    return result
            except Exception:
                pass

        # Method 3: HTML/CSS dimensions (includes Tailwind .ppt-slide classes)
        html_res = detect_html_resolution(project_path)
        if html_res:
            w, h = html_res
            result["resolution"] = (w, h)
            result["method"] = "html_css"
            result["details"] = (
                f"Detected from HTML/CSS: {w}x{h} "
                f"({_aspect_ratio_label(w, h)})"
            )
            return result

    result["details"] = "No resolution could be detected from source files."
    return result

def import_pdf_append(pdf_path: str, project_path: str, config: dict) -> int:
    """Appends PDF pages as image slides to an existing project."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError(
            "PyMuPDF is required to import PDF files. "
            "Install via: pip install PyMuPDF"
        )

    if not os.path.isfile(pdf_path):
        raise ValueError(f"PDF file not found: {pdf_path}")

    images_dir = os.path.join(project_path, "images")
    os.makedirs(images_dir, exist_ok=True)

    dpi = config.get("pdf_import_dpi", 200)
    zoom = dpi / 72.0

    doc = None
    try:
        # Suppress MuPDF C-level stderr warnings (non-fatal structural warnings)
        with contextlib.redirect_stderr(io.StringIO()):
            doc = fitz.open(pdf_path)

        if doc.is_encrypted:
            doc.close()
            doc = None
            raise ValueError(
                f"PDF '{os.path.basename(pdf_path)}' is encrypted and "
                f"cannot be imported."
            )

        total_pages = len(doc)
        if total_pages == 0:
            doc.close()
            doc = None
            raise ValueError(
                f"PDF '{os.path.basename(pdf_path)}' has no pages."
            )

        start_slide = get_next_slide_number(project_path)

        for page_num in range(total_pages):
            slide_num = start_slide + page_num
            page = doc.load_page(page_num)

            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)

            img_name = f"slide{slide_num}.png"
            img_path = os.path.join(images_dir, img_name)
            pix.save(img_path)
            pix = None  # Free memory immediately

            text = page.get_text("text")
            _create_import_slide_files(
                project_path, slide_num, f"images/{img_name}", text
            )

            logger.debug(
                f"[Import] Appended page {page_num + 1}/{total_pages} "
                f"as slide {slide_num}"
            )

        logger.info(
            f"[Import] PDF appended: {total_pages} slide(s) to "
            f"'{os.path.basename(project_path)}'"
        )

        # Update manifest slide_count and propagate to library manifest
        manifest = load_project_manifest(project_path)
        manifest["slide_count"] = sum(
            1 for f in os.listdir(project_path) if _SLIDE_FILE_RE.match(f)
        )
        save_project_manifest(project_path, manifest)
        update_library_manifest(project_path)

        return total_pages

    except ImportError:
        raise
    except ValueError:
        raise
    except Exception as exc:
        logger.error(
            f"[Import] Failed to append PDF "
            f"'{os.path.basename(pdf_path)}': {exc}",
            exc_info=True,
        )
        raise ValueError(f"Failed to append PDF: {exc}") from exc
    finally:
        if doc is not None:
            try:
                doc.close()
            except Exception:
                pass

def _extract_pptx_text(pptx_path: str) -> list[dict]:
    """Uses python-pptx to natively extract text and speaker notes."""
    from pptx import Presentation
    prs = Presentation(pptx_path)
    slides_data = []
    
    for slide in prs.slides:
        slide_text = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                for paragraph in shape.text_frame.paragraphs:
                    text = paragraph.text.strip()
                    if text:
                        slide_text.append(text)
                        
        notes_text = ""
        if slide.has_notes_slide:
            notes_frame = slide.notes_slide.notes_text_frame
            notes_text = notes_frame.text.strip()
            
        slides_data.append({
            "body_text": "\n".join(slide_text),
            "notes_text": notes_text
        })
    return slides_data


def _extract_pptx_images(pptx_path: str, images_dir: str) -> dict[int, list[str]]:
    """Extract embedded images from a PPTX file into the project images/ folder.

    Walks every slide's shape tree looking for picture shapes (``PP_SHAPE_TYPE.PICTURE``
    or shapes with an ``image`` blob).  Each extracted image is saved with a unique
    name that encodes the slide number and a per-slide index so that the
    HTML template can reference it.

    Args:
        pptx_path:  Path to the source .pptx file.
        images_dir: Path to the project's ``images/`` directory (created if missing).

    Returns:
        A dict mapping slide_number (1-based) to a list of relative image paths
        saved for that slide, e.g. ``{1: ["images/embedded_1_0.png"], ...}``.
    """
    from pptx import Presentation
    from pptx.enum.shapes import MSO_SHAPE_TYPE

    os.makedirs(images_dir, exist_ok=True)
    prs = Presentation(pptx_path)

    # Collect extracted image paths per slide
    slide_images: dict[int, list[str]] = {}
    used_names: set[str] = set()

    for slide_idx, slide in enumerate(prs.slides, start=1):
        img_idx = 0
        for shape in slide.shapes:
            image_blob = None
            ext = "png"

            # Direct picture shapes (MSO_SHAPE_TYPE.PICTURE)
            if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                try:
                    image_blob = shape.image.blob
                    ext = _pptx_image_ext(shape.image.content_type)
                except Exception as exc:
                    logger.debug(
                        f"[PPTX Import] Could not read image from shape on "
                        f"slide {slide_idx}: {exc}"
                    )
            elif hasattr(shape, "image"):
                # Fallback: some shapes (e.g. grouped or placeholder) may
                # expose an image attribute even if shape_type != PICTURE.
                try:
                    image_blob = shape.image.blob
                    ext = _pptx_image_ext(shape.image.content_type)
                except Exception:
                    pass

            if image_blob is None or len(image_blob) == 0:
                continue

            # Generate a unique filename: embedded_<slide>_<idx>.<ext>
            img_name = f"embedded_{slide_idx}_{img_idx}.{ext}"
            # Avoid collisions
            while img_name in used_names:
                img_idx += 1
                img_name = f"embedded_{slide_idx}_{img_idx}.{ext}"
            used_names.add(img_name)

            img_path = os.path.join(images_dir, img_name)
            try:
                with open(img_path, "wb") as f:
                    f.write(image_blob)
            except Exception as exc:
                logger.warning(
                    f"[PPTX Import] Failed to write image {img_name} ({type(exc).__name__}: {exc})",
                    exc_info=True,
                )
                continue

            rel_path = f"images/{img_name}"
            slide_images.setdefault(slide_idx, []).append(rel_path)
            img_idx += 1

    if slide_images:
        total = sum(len(v) for v in slide_images.values())
        logger.info(
            f"[PPTX Import] Extracted {total} embedded image(s) "
            f"from {len(slide_images)} slide(s)"
        )
    return slide_images


def _pptx_image_ext(content_type: str) -> str:
    """Map a MIME content-type from python-pptx to a file extension."""
    mapping = {
        "image/png": "png",
        "image/jpeg": "jpg",
        "image/gif": "gif",
        "image/bmp": "bmp",
        "image/tiff": "tiff",
        "image/svg+xml": "svg",
        "image/webp": "webp",
        "image/x-emf": "emf",
        "image/x-wmf": "wmf",
    }
    return mapping.get(content_type, "png")

def import_pptx_as_project(pptx_path: str, config: dict, project_name: str = None, on_exists: str = "error") -> str:
    """Converts PPTX to project, using python-pptx for text and LibreOffice for visuals."""
    try:
        from pptx import Presentation
    except ImportError:
        raise ImportError("python-pptx is required. Run: pip install python-pptx")

    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        raise ImportError(
            "LibreOffice is required to render PPTX visuals. "
            "Please install it and ensure 'soffice' is in your system PATH."
        )
        
    if not project_name:
        project_name = os.path.splitext(os.path.basename(pptx_path))[0]
        
    projects_root = config.get("projects_root", "Projects")
    if not os.path.isabs(projects_root):
        from config import AppConfig
        projects_root = os.path.join(AppConfig.APP_ROOT, projects_root)
    project_path = os.path.join(projects_root, project_name)
    _resolve_on_exists(project_path, project_name, on_exists)
    os.makedirs(project_path, exist_ok=True)
    images_dir = os.path.join(project_path, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # Track 1: Text
    logger.info("[Import] Extracting text from PPTX via python-pptx...")
    pptx_data = _extract_pptx_text(pptx_path)
    
    # Track 1b: Embedded images — extract native-resolution images from
    # the PPTX so they are available in the project images/ folder for
    # re-use or manual editing.
    logger.info("[Import] Extracting embedded images from PPTX...")
    _extract_pptx_images(pptx_path, images_dir)
    
    # Track 2: Visuals
    temp_dir = os.path.join(project_path, "_temp_conversion")
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        cmd = [soffice, "--headless", "--convert-to", "pdf", "--outdir", temp_dir, pptx_path]
        logger.info("[Import] Converting PPTX to PDF via LibreOffice for visuals...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            raise RuntimeError(f"LibreOffice conversion failed: {result.stderr}")
            
        pdf_files = list(Path(temp_dir).glob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError("LibreOffice did not produce a PDF file.")
            
        temp_pdf_path = str(pdf_files[0])
        
        try:
            import fitz
        except ImportError:
            raise ImportError(
                "PyMuPDF is required to render PPTX page images. "
                "Install via: pip install PyMuPDF"
            )
        dpi = config.get("pdf_import_dpi", 200)
        zoom = dpi / 72.0
        
        doc = None
        try:
            doc = fitz.open(temp_pdf_path)
            if len(doc) != len(pptx_data):
                logger.warning(
                    f"[Import] Slide count mismatch! python-pptx found {len(pptx_data)} slides, "
                    f"LibreOffice rendered {len(doc)} pages. Aligning by minimum."
                )
                
            for page_num in range(len(doc)):
                slide_num = page_num + 1
                page = doc.load_page(page_num)
                
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img_name = f"slide{slide_num}.png"
                img_path = os.path.join(images_dir, img_name)
                pix.save(img_path)
                
                text_content = ""
                if page_num < len(pptx_data):
                    data = pptx_data[page_num]
                    if data["notes_text"]:
                        text_content = data["notes_text"]
                    else:
                        text_content = data["body_text"]
                        
                _create_import_slide_files(project_path, slide_num, f"images/{img_name}", text_content)
        finally:
            if doc is not None:
                doc.close()
    except ImportError:
        raise
    except (ValueError, RuntimeError, FileNotFoundError):
        if on_exists != "append" and os.path.isdir(project_path):
            shutil.rmtree(project_path, ignore_errors=True)
        raise
    except Exception as exc:
        logger.error(
            f"[Import] Failed to import PPTX "
            f"'{os.path.basename(pptx_path)}': {exc}",
            exc_info=True,
        )
        if on_exists != "append" and os.path.isdir(project_path):
            shutil.rmtree(project_path, ignore_errors=True)
        raise ValueError(f"Failed to import PPTX: {exc}") from exc
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    # Write project manifest — use actual slide count from disk
    actual_slide_count = sum(1 for f in os.listdir(project_path) if _SLIDE_FILE_RE.match(f))
    manifest_data = {
        "version": 2,
        "name": project_name,
        "source": {
            "type": "pptx",
            "file": os.path.basename(pptx_path),
        },
        "slide_count": actual_slide_count or len(pptx_data),
        "status": "new",
        "created": _now_iso(),
    }
    save_project_manifest(project_path, manifest_data)

    # Auto-detect source resolution and update config
    auto_set_resolution_from_source(project_path, config)

    # Also update the global library manifest (after resolution is set)
    update_library_manifest(project_path)

    logger.info(f"[Import] PPTX imported: {len(pptx_data)} slides -> {project_path}")
    return project_path

def import_pptx_append(pptx_path: str, project_path: str, config: dict) -> int:
    """Appends PPTX slides to an existing project."""
    # Not natively supported with perfect numbering in this simplified helper,
    # so we instantiate a temporary project and copy files over.
    temp_project_name = f"_temp_{os.path.basename(pptx_path)}"
    projects_root = config.get("projects_root", "Projects")
    if not os.path.isabs(projects_root):
        from config import AppConfig
        projects_root = os.path.join(AppConfig.APP_ROOT, projects_root)
    temp_project_path = os.path.join(projects_root, temp_project_name)
    if os.path.isdir(temp_project_path):
        shutil.rmtree(temp_project_path, ignore_errors=True)
    try:
        temp_project_path = import_pptx_as_project(pptx_path, config, project_name=temp_project_name, on_exists="overwrite")
    except Exception:
        if os.path.isdir(temp_project_path):
            shutil.rmtree(temp_project_path, ignore_errors=True)
        raise
    
    start_slide = get_next_slide_number(project_path)
    count = 0
    
    try:
        for f in sorted(os.listdir(temp_project_path), key=natural_sort_key):
            match = _SLIDE_FILE_RE.match(f)
            if match:
                old_num = int(match.group(1))
                new_num = start_slide + count
                
                # Copy HTML
                old_html = os.path.join(temp_project_path, f)
                new_html = os.path.join(project_path, f"slide{new_num}.html")
                # Update image path inside html
                with open(old_html, "r", encoding="utf-8") as fh:
                    content = fh.read()
                content = content.replace(f"images/slide{old_num}.png", f"images/slide{new_num}.png")
                with open(new_html, "w", encoding="utf-8") as fh:
                    fh.write(content)
                
                # Copy TXT
                old_txt = os.path.join(temp_project_path, f"slide{old_num}.txt")
                new_txt = os.path.join(project_path, f"slide{new_num}.txt")
                if os.path.exists(old_txt):
                    shutil.copy2(old_txt, new_txt)
                    
                # Copy Image
                old_img = os.path.join(temp_project_path, "images", f"slide{old_num}.png")
                new_img = os.path.join(project_path, "images", f"slide{new_num}.png")
                if os.path.exists(old_img):
                    shutil.copy2(old_img, new_img)
                    
                count += 1
    finally:
        shutil.rmtree(temp_project_path, ignore_errors=True)
    logger.info(f"[Import] PPTX appended: {count} slides to {project_path}")

    # Update manifest slide_count and propagate to library manifest
    manifest = load_project_manifest(project_path)
    manifest["slide_count"] = sum(
        1 for f in os.listdir(project_path) if _SLIDE_FILE_RE.match(f)
    )
    save_project_manifest(project_path, manifest)
    update_library_manifest(project_path)

    return count

def import_file_as_project(file_path: str, config: dict, project_name: str = None, on_exists: str = "error") -> str:
    """Dispatcher to create a new project from a given file.

    Args:
        on_exists: "error"  – raise ValueError if project exists (default)
                   "overwrite" – delete existing project and re-create
                   "append"    – append slides to the existing project
    """
    ext = os.path.splitext(file_path)[1].lower()

    # For "append" mode, delegate to the append dispatcher directly
    if on_exists == "append":
        # We need a project_path to append to
        if not project_name:
            project_name = os.path.splitext(os.path.basename(file_path))[0]
        from config import AppConfig
        projects_root = config.get("projects_root", "Projects")
        if not os.path.isabs(projects_root):
            projects_root = os.path.join(AppConfig.APP_ROOT, projects_root)
        project_path = os.path.join(projects_root, project_name)
        if not os.path.isdir(project_path):
            # Project doesn't exist yet — fall through to create it
            on_exists = "error"
        else:
            count = import_file_append(file_path, project_path, config)
            update_library_manifest(project_path)
            logger.info(
                f"[Import] Appended {count} slide(s) to '{project_name}'"
            )
            return project_path

    if ext == ".pdf":
        return import_pdf_as_project(file_path, config, project_name, on_exists=on_exists)
    elif ext == ".pptx":
        return import_pptx_as_project(file_path, config, project_name, on_exists=on_exists)
    elif ext == ".json":
        return import_json_speaker_script_as_project(file_path, config, project_name, on_exists=on_exists)
    elif ext in (".txt", ".csv"):
        # Basic text import
        if not project_name:
            project_name = os.path.splitext(os.path.basename(file_path))[0]
        projects_root = config.get("projects_root", "Projects")
        if not os.path.isabs(projects_root):
            projects_root = os.path.join(AppConfig.APP_ROOT, projects_root)
        project_path = os.path.join(projects_root, project_name)
        _resolve_on_exists(project_path, project_name, on_exists)
        os.makedirs(project_path, exist_ok=True)

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        create_slide_file(project_path, 1, get_default_slide_html(), text)
        save_project_manifest(project_path, {
            "version": 2,
            "name": project_name,
            "source": {
                "type": ext.lstrip("."),
                "file": os.path.basename(file_path),
            },
            "slide_count": 1,
            "status": "new",
            "created": _now_iso(),
        })
        update_library_manifest(project_path)
        return project_path
    elif ext in (".png", ".jpg", ".jpeg"):
        if not project_name:
            project_name = os.path.splitext(os.path.basename(file_path))[0]
        projects_root = config.get("projects_root", "Projects")
        if not os.path.isabs(projects_root):
            projects_root = os.path.join(AppConfig.APP_ROOT, projects_root)
        project_path = os.path.join(projects_root, project_name)
        _resolve_on_exists(project_path, project_name, on_exists)
        os.makedirs(project_path, exist_ok=True)
        import_image_as_slide(file_path, project_path, config)
        save_project_manifest(project_path, {
            "version": 2,
            "name": project_name,
            "source": {
                "type": "image",
                "file": os.path.basename(file_path),
            },
            "slide_count": 1,
            "status": "new",
            "created": _now_iso(),
        })
        update_library_manifest(project_path)

        # Auto-detect source resolution and update config
        auto_set_resolution_from_source(project_path, config)

        return project_path
    else:
        raise ValueError(f"Unsupported file type for import: {ext}")


def import_file_append(file_path: str, project_path: str, config: dict) -> int:
    """Dispatcher to append a file to an existing project."""
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == ".pdf":
        count = import_pdf_append(file_path, project_path, config)
    elif ext == ".pptx":
        count = import_pptx_append(file_path, project_path, config)  
    elif ext == ".json":
        count = import_json_speaker_script_append(file_path, project_path, config)
    elif ext in (".txt", ".csv"):
        slide_num = get_next_slide_number(project_path)
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        create_slide_file(project_path, slide_num, get_default_slide_html(), text)
        count = 1
    elif ext in (".png", ".jpg", ".jpeg"):
        count = import_image_as_slide(file_path, project_path, config)
    else:
        raise ValueError(f"Unsupported file type for append: {ext}")

    # Ensure manifest slide_count is always up to date.
    # Sub-appenders (PDF/PPTX/JSON) may have already updated, but re-reading
    # and writing is cheap and guarantees correctness for all paths.
    try:
        manifest = load_project_manifest(project_path)
        manifest["slide_count"] = sum(
            1 for f in os.listdir(project_path) if _SLIDE_FILE_RE.match(f)
        )
        save_project_manifest(project_path, manifest)
        update_library_manifest(project_path)
    except Exception as exc:
        logger.warning(f"[Import] Failed to update manifest after append ({type(exc).__name__}: {exc})", exc_info=True)

    return count

# ═══════════════════════════════════════════════════════════════════════════════
#  JSON SPEAKER SCRIPT IMPORT
# ═══════════════════════════════════════════════════════════════════════════════

def parse_speaker_script_json(file_path: str) -> dict:
    """Parse a JSON speaker script file.
    
    Handles multiple formats:
    
    Format 1 — Object with slides array:
    {
        "presentation": "Rules of Chess",
        "slides": [
            {"slide_number": 1, "title": "(untitled)", "speaker_script": "Welcome..."}
        ]
    }
    
    Format 2 — Flat array of slide objects or strings:
    [
        {"speaker_script": "Welcome..."},
        "Second slide narration text"
    ]
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Support both top-level array and top-level object
    if isinstance(data, list):
        # Flat array: each element is a slide
        slides_raw = data
        title = ""
    elif isinstance(data, dict):
        # 1. Extract presentation title (supports multiple key names)
        title = str(
            data.get("presentation")
            or data.get("title")
            or data.get("name")
            or ""
        ).strip()
        
        # 2. Extract slides array (supports multiple key names)
        # Use explicit None checks — an empty list [] is valid and
        # must not cause a fallthrough to the next key.
        slides_raw = None
        for _key in ("slides", "slide_data", "scenes"):
            if _key in data:
                slides_raw = data[_key]
                break
        if slides_raw is None:
            slides_raw = []
        if not isinstance(slides_raw, list):
            raise ValueError(
                "JSON 'slides' (or 'slide_data'/'scenes') must be an array"
            )
    else:
        raise ValueError(
            f"JSON root must be an object or array, "
            f"got {type(data).__name__}"
        )

    parsed_slides = []
    for i, entry in enumerate(slides_raw):
        # Handle string entries directly — each string is one slide's narration
        if isinstance(entry, str):
            script_text = entry.strip()
            if script_text:
                parsed_slides.append({
                    "slide_number": i + 1,
                    "content": script_text,
                })
            continue

        if not isinstance(entry, dict):
            continue
            
        # Specifically target "speaker_script", fallback to other common keys
        script_text = str(
            entry.get("speaker_script")
            or entry.get("content")
            or entry.get("text")
            or entry.get("narration")
            or entry.get("script")
            or ""
        ).strip()
        
        if script_text:
            parsed_slides.append({
                "slide_number": i + 1,
                "content": script_text,
            })

    if not parsed_slides:
        raise ValueError("No slide text found in the JSON file.")

    return {"title": title, "slides": parsed_slides}



def import_json_speaker_script_as_project(
    file_path: str, config: dict, project_name: str = None, on_exists: str = "error"
) -> str:
    """Import a JSON speaker script as a new project.

    Each entry in the JSON 'slides' array becomes one slide.  Slide
    numbers are assigned sequentially (1, 2, 3, ...) regardless of the
    'slide_number' values in the JSON, so there are never gaps.

    Args:
        on_exists: "error"  – raise ValueError if project exists (default)
                   "overwrite" – delete existing project and re-create
                   "append"    – append slides to the existing project
    """
    if not os.path.isfile(file_path):
        raise ValueError(f"JSON file not found: {file_path}")

    parsed = parse_speaker_script_json(file_path)

    if not project_name:
        project_name = parsed["title"] or os.path.splitext(os.path.basename(file_path))[0]

    # Sanitize name
    project_name = re.sub(r'[\\/*?:"<>|]', "", project_name).strip()
    if not project_name or project_name in (".", "..") or project_name.startswith(".."):
        project_name = "Untitled_Script"

    projects_root = config.get("projects_root", "Projects")
    if not os.path.isabs(projects_root):
        from config import AppConfig
        projects_root = os.path.join(AppConfig.APP_ROOT, projects_root)
    project_path = os.path.join(projects_root, project_name)

    # Handle append mode separately — delegate to the append function
    if on_exists == "append" and os.path.isdir(project_path):
        count = import_json_speaker_script_append(file_path, project_path, config)
        logger.info(
            f"[JSON Import] Appended {count} slide(s) to existing project '{project_name}'"
        )
        return project_path

    _resolve_on_exists(project_path, project_name, on_exists)

    os.makedirs(project_path, exist_ok=True)

    # Use sequential slide numbering to avoid gaps
    for idx, slide_info in enumerate(parsed["slides"], start=1):
        script_text = slide_info["content"]

        html_path = os.path.join(project_path, f"slide{idx}.html")
        txt_path = os.path.join(project_path, f"slide{idx}.txt")

        # 1. Write ONLY the speaker_script text to the .txt file for TTS
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(script_text)

        # 2. Write HTML file (inject the text so the visual slide isn't blank)
        html_content = get_default_slide_html()
        # Escape HTML characters and convert newlines to <br> for the visual slide
        safe_text = script_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
        html_content = html_content.replace("Slide Content", safe_text)

        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

    manifest_data = {
        "version": 2,
        "name": project_name,
        "source": {
            "type": "json_speaker_script",
        },
        "title": parsed["title"],
        "slide_count": len(parsed["slides"]),
        "status": "new",
        "created": _now_iso(),
    }
    save_project_manifest(project_path, manifest_data)
    update_library_manifest(project_path)

    logger.info(
        f"[JSON Import] Created project '{project_name}' with "
        f"{len(parsed['slides'])} slide(s) from speaker script"
    )
    return project_path

def import_json_speaker_script_append(
    file_path: str, project_path: str, config: dict
) -> int:
    """Append slides from a JSON speaker script to an existing project.

    Slides are appended sequentially starting from the next available
    slide number in the project, ignoring the 'slide_number' values
    from the JSON.
    """
    if not os.path.isfile(file_path):
        raise ValueError(f"JSON file not found: {file_path}")

    if not os.path.isdir(project_path):
        raise ValueError(f"Project directory not found: {project_path}")

    parsed = parse_speaker_script_json(file_path)
    next_slide = get_next_slide_number(project_path)

    count = 0
    for slide_info in parsed["slides"]:
        slide_num = next_slide + count
        script_text = slide_info["content"]  # The extracted speaker_script text
        
        html_path = os.path.join(project_path, f"slide{slide_num}.html")
        txt_path = os.path.join(project_path, f"slide{slide_num}.txt")
        
        # 1. Write ONLY the speaker_script text to the .txt file
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(script_text)
            
        # 2. Write HTML file
        html_content = get_default_slide_html()
        safe_text = script_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
        html_content = html_content.replace("Slide Content", safe_text)
        
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        count += 1

    logger.info(
        f"[JSON Import] Appended {count} slide(s) from speaker script "
        f"to '{os.path.basename(project_path)}'"
    )

    # Update manifest slide_count and propagate to library manifest
    manifest = load_project_manifest(project_path)
    manifest["slide_count"] = sum(
        1 for f in os.listdir(project_path) if _SLIDE_FILE_RE.match(f)
    )
    save_project_manifest(project_path, manifest)
    update_library_manifest(project_path)

    return count