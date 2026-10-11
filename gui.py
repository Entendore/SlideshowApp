# gui.py
"""
Main application window — project library, tabs, render scheduler.
This module is ONLY imported by the main GUI process. Worker processes
will never import this module, preventing Qt from initializing in the
child processes and saving significant memory.
"""
from __future__ import annotations
import logging
import multiprocessing
import os
import queue
import shutil
import subprocess
import threading
import time
from datetime import datetime
from typing import Optional
from PySide6.QtCore import Qt, QTimer, QThread, Signal, QObject
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication, QAbstractItemView, QComboBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QInputDialog, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMainWindow, QMessageBox, QMenu, QProgressBar,
    QPushButton, QSplitter, QTabWidget, QTextEdit, QVBoxLayout, QWidget,
)
from config import AppConfig
# ═══════════════════════════════════════════════════════════════════════════════
from dialogs import (
    NewProjectTabWidget, ProjectListItemWidget, ProjectSelectionManager,
    SettingsTabWidget, SlideEditorTabWidget,
)
from engines import render_worker_loop, generate_thumbnail_worker
from utils import (
    detect_ffmpeg,
    get_library_status_data,
    get_project_dirs, get_theme, import_file_append, import_file_as_project,
    import_pdf_append, import_pdf_as_project, import_image_as_slide,
    list_themes, load_project_manifest,
    natural_sort_key, set_theme,
    should_skip_render, update_library_manifest,
    ChunkProgressMessage, ErrorMessage, FinalizedMessage,
    LogMessage, PhaseMessage, RenderPhase, RenderStatus,
    SlideProgressMessage, STATUS_SORT_PRIORITY, Palette, style_input,
    get_phase_progress, fmt_duration, VerboseFilter,
    style_input_wide, style_button, style_button_small,
    style_primary_button, style_blue_button, style_green_button,
    style_stop_button, style_radio, style_checkbox, style_group_box,
    style_group_box_bordered, style_summary_label, style_log_text,
    FILE_FILTER_IMPORT,
)
logger = logging.getLogger(__name__)
# ═══════════════════════════════════════════════════════════════════════════════
#  SAFE WIDGET ACCESS HELPER
# ═══════════════════════════════════════════════════════════════════════════════


def _widget_ok(widget) -> bool:
    """Return True if the Qt widget's C++ object is still alive."""
    try:
        from shiboken6 import isValid
        return isValid(widget)
    except ImportError:
        pass
    try:
        widget.objectName()
        return True
    except RuntimeError:
        return False
# ═══════════════════════════════════════════════════════════════════════════════
#  QUEUE READER THREAD
# ═══════════════════════════════════════════════════════════════════════════════


class QueueReaderThread(QThread):
    """Dedicated thread that blocks on multiprocessing.Queue.get() and
    emits a Qt Signal for each incoming message."""
    message_received = Signal(object)

    def __init__(self, msg_queue: multiprocessing.Queue, parent=None):
        super().__init__(parent)
        self._queue = msg_queue
        self._stop_event = threading.Event()

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                msg = self._queue.get(timeout=0.2)
                if msg is None:
                    break
                self.message_received.emit(msg)
            except queue.Empty:
                continue
            except OSError:
                break
            except Exception:
                if not self._stop_event.is_set():
                    logger.error("[QueueReader] Unexpected error", exc_info=True)
                continue

    def stop(self) -> None:
        self._stop_event.set()
        try:
            self._queue.put(None)
        except Exception:
            pass

    def graceful_stop(self, timeout_ms: int = 3000) -> None:
        try:
            self.message_received.disconnect()
        except RuntimeError:
            pass
        self.stop()
        self.wait(timeout_ms)
# ═══════════════════════════════════════════════════════════════════════════════
#  THREAD-SAFE LOG APPEND SIGNAL
# ═══════════════════════════════════════════════════════════════════════════════


class _LogSignalHelper(QObject):
    append_html = Signal(str)
# ═══════════════════════════════════════════════════════════════════════════════
#  COLORED LOG HANDLER
# ═══════════════════════════════════════════════════════════════════════════════


class ColoredLogHandler(logging.Handler):
    LEVEL_COLORS: dict[int, str] = {
        logging.DEBUG:    Palette.LOG_DEBUG,
        logging.INFO:     Palette.LOG_INFO,
        logging.WARNING:  Palette.LOG_WARN,
        logging.ERROR:    Palette.LOG_ERROR,
        logging.CRITICAL: Palette.LOG_CRITICAL,
    }
    LEVEL_TAGS: dict[int, str] = {
        logging.DEBUG:    "DBG",
        logging.INFO:     "INF",
        logging.WARNING:  "WRN",
        logging.ERROR:    "ERR",
        logging.CRITICAL: "CRT",
    }

    def __init__(self, text_widget: QTextEdit, max_lines: int = 5000):
        super().__init__()
        self._widget = text_widget
        self._max_lines = max_lines
        self._closed = False
        self.setLevel(logging.DEBUG)
        self._signal_helper = _LogSignalHelper()
        self._signal_helper.append_html.connect(self._do_append_html)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._signal_helper.append_html.disconnect()
        except RuntimeError:
            pass
        logging.getLogger().removeHandler(self)
        super().close()

    def _do_append_html(self, html: str) -> None:
        if self._closed:
            return
        try:
            sb = self._widget.verticalScrollBar()
            at_bottom = sb.value() >= sb.maximum() - 30
            self._widget.append(html)
            self._trim_lines()
            if at_bottom:
                sb.setValue(sb.maximum())
        except RuntimeError:
            self._closed = True

    def _format_record_html(self, record: logging.LogRecord) -> str:
        msg = self.format(record)
        color = self.LEVEL_COLORS.get(record.levelno, "#D4D4D4")
        tag = self.LEVEL_TAGS.get(record.levelno, "???")
        timestamp = datetime.now().strftime("%H:%M:%S")
        safe_msg = msg.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return (
            f'<span style="color:#666666">[{timestamp}]</span> '
            f'<span style="color:{color};font-weight:bold">[{tag}]</span> '
            f'<span style="color:{color}">{safe_msg}</span>'
        )

    def _format_worker_log_html(self, level: str, project_name: str, text: str,
                                phase: str = "") -> str:
        level_map = {
            "DEBUG": logging.DEBUG, "INFO": logging.INFO,
            "WARNING": logging.WARNING, "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        levelno = level_map.get(level, logging.INFO)
        if levelno < self.level:
            return ""
        color = self.LEVEL_COLORS.get(levelno, "#D4D4D4")
        tag = self.LEVEL_TAGS.get(levelno, "???")
        timestamp = datetime.now().strftime("%H:%M:%S")
        safe_msg = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        # Phase tag: short colored chip showing pipeline stage
        phase_html = ""
        if phase:
            phase_short = phase.replace("_", " ").title()
            phase_html = (
                f'<span style="color:#b8b8b8;background-color:#333333;'
                f'padding:0px 4px;border-radius:3px;font-size:10px;'
                f'margin-right:4px">[{phase_short}]</span> '
            )
        return (
            f'<span style="color:#666666">[{timestamp}]</span> '
            f'<span style="color:{color};font-weight:bold">[{tag}]</span> '
            f'<span style="color:#569CD6">[{project_name}]</span> '
            f'{phase_html}'
            f'<span style="color:{color}">{safe_msg}</span>'
        )

    def emit(self, record: logging.LogRecord) -> None:
        if self._closed:
            return
        try:
            html = self._format_record_html(record)
            self._signal_helper.append_html.emit(html)
        except Exception:
            self.handleError(record)

    def append_worker_log(self, level: str, project_name: str, text: str,
                          phase: str = "") -> None:
        if self._closed:
            return
        html = self._format_worker_log_html(level, project_name, text, phase)
        if html:
            self._signal_helper.append_html.emit(html)

    def _trim_lines(self) -> None:
        doc = self._widget.document()
        if doc.blockCount() > self._max_lines:
            cursor = self._widget.textCursor()
            cursor.movePosition(cursor.MoveOperation.Start)
            cursor.movePosition(
                cursor.MoveOperation.Down,
                cursor.MoveMode.KeepAnchor,
                doc.blockCount() - self._max_lines,
            )
            cursor.removeSelectedText()
            cursor.deleteChar()
# ═══════════════════════════════════════════════════════════════════════════════
#  STYLESHEETS
# ═══════════════════════════════════════════════════════════════════════════════


def _project_list_stylesheet() -> str:
    from utils import list_widget_stylesheet_from_theme
    base = list_widget_stylesheet_from_theme()
    return base + (
        "QListWidget::item { background: transparent; padding: 0px; margin: 0px; border: none; }"
        "QListWidget::item:selected { background: transparent; border: none; }"
        "QListWidget::item:hover   { background: transparent; border: none; }"
        "QListWidget::item:selected:active { background: transparent; border: none; }"
    )
# ═══════════════════════════════════════════════════════════════════════════════
#  MESSAGE DISPATCH TABLE
# ═══════════════════════════════════════════════════════════════════════════════


def _handle_slide_progress(win: MainWindow, msg: SlideProgressMessage) -> None:
    if win._should_throttle_progress(msg.project_name):
        return
    if msg.total > 0:
        slide_frac = msg.current / msg.total
        # Map slide progress into the VIDEO_RENDERING phase range (58–90%)
        _, vid_start, vid_end = get_phase_progress(str(RenderPhase.VIDEO_RENDERING))
        pct = int(vid_start + slide_frac * (vid_end - vid_start))
        win.lbl_status.setText(f"Rendering {msg.project_name}: Slide {msg.current}/{msg.total}")
        win.progress_bar.setValue(pct)
        progress_str = f"Slide {msg.current}/{msg.total} ({pct}%)"
        win.update_item_status(
            msg.project_name, str(RenderPhase.VIDEO_RENDERING),
            progress_str, progress_pct=pct,
        )


def _handle_phase(win: MainWindow, msg: PhaseMessage) -> None:
    # Use the rich label/progress from the message if provided,
    # otherwise resolve from PHASE_PROGRESS_MAP.
    if msg.label and msg.progress_pct >= 0:
        label = msg.label
        pct = msg.progress_pct
    else:
        label, start, end = get_phase_progress(msg.phase)
        pct = (start + end) // 2
    # Build status text with elapsed time and ETA
    elapsed = getattr(msg, "elapsed", 0.0) or 0.0
    elapsed_str = ""
    eta_str = ""
    if elapsed > 0:
        elapsed_str = f" ({fmt_duration(elapsed)} elapsed)"
        # Estimate remaining time based on progress
        if pct > 2:
            total_est = elapsed / (pct / 100.0)
            remaining = max(0, total_est - elapsed)
            eta_str = f"  ~{fmt_duration(remaining)} left"
    phase_display = msg.phase.replace("_", " ").title()
    status_text = f"{msg.project_name}: [{phase_display}] {label}{eta_str}{elapsed_str}"
    win.lbl_status.setText(status_text)
    win.progress_bar.setValue(pct)
    # Use the sub-phase name as the status key for the project list item,
    # so StatusTheme can color it differently per sub-phase.
    win.update_item_status(
        msg.project_name, msg.phase,
        f"{label} — {pct}%",
        progress_pct=pct,
    )
    # Track render start time for ETA calculation
    if msg.phase == RenderPhase.SCANNING:
        win._render_start_times[msg.project_name] = time.time()


def _handle_error(win: MainWindow, msg: ErrorMessage) -> None:
    project_name = getattr(msg, "project_name", "Unknown")
    error_text = getattr(msg, "error_text", str(msg))
    logger.error(f"[Render Error] [{project_name}]: {error_text}")
    elapsed_str = win._get_elapsed(project_name)
    detail = "Failed \u2014 Check Logs"
    if elapsed_str:
        detail = f"Failed ({elapsed_str}) \u2014 Check Logs"
    win.update_item_status(project_name, RenderStatus.ERROR, detail)


def _handle_chunk_progress(win: MainWindow, msg: ChunkProgressMessage) -> None:
    if win._should_throttle_progress(msg.project_name):
        return
    # Map chunk progress into the AUDIO_GENERATING phase range (10–48%)
    pct = 0
    if msg.total > 0:
        chunk_frac = msg.current / msg.total
        _, aud_start, aud_end = get_phase_progress(str(RenderPhase.AUDIO_GENERATING))
        pct = int(aud_start + chunk_frac * (aud_end - aud_start))
        win.progress_bar.setValue(pct)
    # Build a clear status: "segment X/Y (for N slides)"
    # so the user never confuses segment count with slide count.
    slide_info = ""
    if msg.slide_count > 0 and msg.slide_count != msg.total:
        slide_info = f" (for {msg.slide_count} slide(s))"
    display_msg = f"Audio segment {msg.current}/{msg.total}{slide_info}"
    win.lbl_status.setText(f"{msg.project_name}: {display_msg}")
    win.update_item_status(
        msg.project_name, str(RenderPhase.AUDIO_GENERATING),
        f"Audio: {display_msg}", progress_pct=pct if msg.total > 0 else 0,
    )


def _handle_finalized(win: MainWindow, msg: FinalizedMessage) -> None:
    project_name = getattr(msg, "project_name", "Unknown")
    elapsed_str = win._get_elapsed(project_name)
    detail = "Completed"
    if elapsed_str:
        detail = f"Completed ({elapsed_str})"
    win.update_item_status(project_name, RenderStatus.READY, detail)
    win._clear_render_timer(project_name)
    if elapsed_str:
        logger.info(f"[Scheduler] {project_name} finished in {elapsed_str}")
    win._mark_library_dirty()


def _handle_log_message(win: MainWindow, msg) -> None:
    level = getattr(msg, "level", "INFO")
    project_name = getattr(msg, "project_name", "")
    text = getattr(msg, "text", str(msg))
    phase = getattr(msg, "phase", "")
    if win._log_handler:
        win._log_handler.append_worker_log(level, project_name, text, phase)


_MESSAGE_HANDLERS = {
    SlideProgressMessage.__name__:    _handle_slide_progress,
    PhaseMessage.__name__:            _handle_phase,
    ErrorMessage.__name__:            _handle_error,
    ChunkProgressMessage.__name__:    _handle_chunk_progress,
    FinalizedMessage.__name__:        _handle_finalized,
    LogMessage.__name__:              _handle_log_message,
}
# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ═══════════════════════════════════════════════════════════════════════════════


class _ImportWorker(QThread):
    """Runs file-import (which may invoke LibreOffice) off the GUI thread."""
    finished = Signal(str, object)  # (file_path, result_path_or_exception)

    def __init__(self, file_path, config, project_name=None, on_exists="error", parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.config = config
        self.project_name = project_name
        self.on_exists = on_exists

    def run(self):
        try:
            result = import_file_as_project(
                self.file_path, self.config,
                project_name=self.project_name,
                on_exists=self.on_exists,
            )
            self.finished.emit(self.file_path, result)
        except Exception as exc:
            self.finished.emit(self.file_path, exc)


class _AppendImportWorker(QThread):
    """Runs file-append import off the GUI thread."""
    finished = Signal(str, object)  # (file_path, result_count_or_exception)

    def __init__(self, file_path, project_path, config, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.project_path = project_path
        self.config = config

    def run(self):
        try:
            count = import_file_append(
                self.file_path, self.project_path, self.config
            )
            self.finished.emit(self.file_path, count)
        except Exception as exc:
            self.finished.emit(self.file_path, exc)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = AppConfig()
        self.project_dirs = get_project_dirs()
        self.max_workers = self.config.get("render_workers", 3)
        self.active_processes: dict[int, dict] = {}
        self.pending_projects: list[str] = []
        self._mp_ctx = multiprocessing.get_context("spawn")
        self.render_queue: multiprocessing.Queue = self._mp_ctx.Queue()
        # ── Work queue (shared by all persistent workers) ────────────
        self._worker_queues: dict[int, multiprocessing.Queue] = {}
        self._worker_pool: list[multiprocessing.Process] = []
        # ── Worker health tracking ───────────────────────────────────
        self._pool_cooldown_until: float = 0.0  # timestamp — don't respawn before this
        self._draining: bool = False  # True while graceful drain is active
        # ── General state ────────────────────────────────────────────
        self._refreshing_library = False
        self._render_start_times: dict[str, float] = {}
        self._log_handler: Optional[ColoredLogHandler] = None
        self._verbose_filters: list = []  # VerboseFilter instances for dynamic toggling
        self._library_dirty = False
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.timeout.connect(self._do_dirty_refresh)
        self._last_progress_update: dict[str, float] = {}
        self._progress_throttle_interval = 0.2
        self._shutting_down = False
        self._stopping = False
        # Debounce timer for auto-thumbnail on slide 1 edit
        self._thumb_debounce = QTimer(self)
        self._thumb_debounce.setSingleShot(True)
        self._thumb_debounce.setInterval(2000)
        self._thumb_debounce.timeout.connect(self._on_thumb_debounce_timeout)
        self._pending_thumb_project: Optional[str] = None
        self.init_ui()
        self._setup_gui_logging()
        self._ensure_worker_pool()
        self._cleanup_stale_temp()
        self._pool_timer = QTimer(self)
        self._pool_timer.timeout.connect(self._pool_tick)
        self._pool_timer.start(500)
        self.refresh_library()
        self._queue_reader = QueueReaderThread(self.render_queue, parent=self)
        self._queue_reader.message_received.connect(self._on_queue_message)
        self._queue_reader.start()
    # ══════════════════════════════════════════════════════════════════════
    #  GUI LOGGING SETUP
    # ══════════════════════════════════════════════════════════════════════

    def _setup_gui_logging(self) -> None:
        self._log_handler = ColoredLogHandler(self.log_text)
        self._log_handler.setFormatter(logging.Formatter("%(message)s"))
        root = logging.getLogger()
        root.addHandler(self._log_handler)
        for handler in root.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, ColoredLogHandler):
                handler.setFormatter(logging.Formatter(
                    "[%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s", datefmt="%H:%M:%S",
                ))
        # Apply human-readable debug filter for PIL/asyncio messages
        self._apply_verbose_filters(self.config.get("verbose", False))
        logger.info("[App] GUI logging initialized with color-coded output")

    def _set_log_level(self, level_name: str) -> None:
        level_map = {
            "All": logging.DEBUG, "Debug+": logging.DEBUG,
            "Info+": logging.INFO, "Warning+": logging.WARNING, "Error+": logging.ERROR,
        }
        level = level_map.get(level_name, logging.DEBUG)
        if self._log_handler:
            self._log_handler.setLevel(level)
        logger.info(f"[App] GUI log level set to: {level_name}")
    # ══════════════════════════════════════════════════════════════════════
    #  RENDER TIMING HELPERS
    # ══════════════════════════════════════════════════════════════════════

    def _start_render_timer(self, project_name: str) -> None:
        self._render_start_times[project_name] = time.time()

    def _get_elapsed(self, project_name: str) -> str:
        start = self._render_start_times.get(project_name)
        if start is None:
            return ""
        return fmt_duration(time.time() - start)

    def _format_elapsed(self, project_name: str) -> str:
        start = self._render_start_times.pop(project_name, None)
        if start is None:
            return ""
        return fmt_duration(time.time() - start)

    def _clear_render_timer(self, project_name: str) -> None:
        self._render_start_times.pop(project_name, None)
    # ══════════════════════════════════════════════════════════════════════
    #  BATCHED LIBRARY REFRESH
    # ══════════════════════════════════════════════════════════════════════

    def _mark_library_dirty(self) -> None:
        self._library_dirty = True
        self._refresh_timer.start(150)

    def _do_dirty_refresh(self) -> None:
        if self._library_dirty and not self._shutting_down:
            self._library_dirty = False
            self.refresh_library()
    # ══════════════════════════════════════════════════════════════════════
    #  PROGRESS THROTTLING
    # ══════════════════════════════════════════════════════════════════════

    def _should_throttle_progress(self, project_name: str) -> bool:
        now = time.time()
        last = self._last_progress_update.get(project_name, 0.0)
        if now - last < self._progress_throttle_interval:
            return True
        self._last_progress_update[project_name] = now
        return False
    # ══════════════════════════════════════════════════════════════════════
    #  UI SETUP
    # ══════════════════════════════════════════════════════════════════════

    def init_ui(self):
        self.setWindowTitle("AI Slideshow Renderer (Multi-Process)")
        self.resize(1400, 900)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        # ── Left Panel: Library ──────────────────────────────────────────
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        lbl_lib = QLabel("Project Library")
        lbl_lib.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        left_layout.addWidget(lbl_lib)
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search...")
        self.search_edit.setStyleSheet(style_input_wide())
        self.search_edit.textChanged.connect(self.filter_library)
        left_layout.addWidget(self.search_edit)
        sort_layout = QHBoxLayout()
        sort_layout.setContentsMargins(0, 0, 0, 0)
        self.sort_combo = QComboBox()
        self.sort_combo.addItems(["Name (A->Z)", "Name (Z->A)", "Status", "Last Rendered"])
        self.sort_combo.setStyleSheet(style_input_wide())
        self.sort_combo.currentIndexChanged.connect(self.refresh_library)
        sort_layout.addWidget(QLabel("Sort:"))
        sort_layout.addWidget(self.sort_combo)
        left_layout.addLayout(sort_layout)
        theme_row = QHBoxLayout()
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(list_themes())
        self.theme_combo.setCurrentText(get_theme()["name"])
        self.theme_combo.currentTextChanged.connect(self.on_theme_changed)
        theme_row.addWidget(QLabel("Theme:"))
        theme_row.addWidget(self.theme_combo)
        left_layout.addLayout(theme_row)
        self.project_list = QListWidget()
        self.project_list.setSelectionMode(
            QAbstractItemView.SelectionMode.MultiSelection
        )
        self.project_list.itemDoubleClicked.connect(self.on_project_double_click)
        self.project_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.project_list.customContextMenuRequested.connect(self.show_library_context_menu)
        self.selection_mgr = ProjectSelectionManager(self.project_list)
        self.project_list.itemSelectionChanged.connect(self.selection_mgr.refresh)
        self.project_list.setStyleSheet(_project_list_stylesheet())
        left_layout.addWidget(self.project_list)
        lib_btn_layout = QHBoxLayout()
        btn_new = QPushButton("✨ New Project")
        btn_new.setToolTip("Create a new project — blank, from file, or with a narration script")
        btn_new.clicked.connect(self.create_new_project)
        btn_import = QPushButton("📂 Quick Import")
        btn_import.setToolTip("Pick a file and create a project from it in one step")
        btn_import.clicked.connect(self.import_file_as_new_project)
        btn_edit = QPushButton("Edit Project")
        btn_edit.clicked.connect(self.open_editor)
        btn_delete = QPushButton("Delete Project")
        btn_delete.clicked.connect(self.delete_project)
        lib_btn_layout.addWidget(btn_new)
        lib_btn_layout.addWidget(btn_import)
        lib_btn_layout.addWidget(btn_edit)
        lib_btn_layout.addWidget(btn_delete)
        left_layout.addLayout(lib_btn_layout)
        splitter.addWidget(left_panel)
        # ── Right Panel: Tabs & Logs ─────────────────────────────────────
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        self.tabs = QTabWidget()
        self.tabs.setTabsClosable(True)
        self.tabs.setStyleSheet(
            f"QTabWidget::pane {{ border: none; background: {Palette.BG}; }}"
        )
        self.tabs.tabCloseRequested.connect(self.close_tab)
        self.tabs.setDocumentMode(True)
        self.open_settings_tab(initial=True)
        right_layout.addWidget(self.tabs, stretch=2)
        grp_controls = QGroupBox("Render Controls")
        grp_controls.setStyleSheet(style_group_box_bordered())
        ctrl_layout = QFormLayout()
        self.lbl_workers = QLabel(f"Max Parallel Jobs: {self.max_workers}")
        self.lbl_active_jobs = QLabel("Active: 0")
        self.lbl_queue_count = QLabel("Queue: 0")
        self.lbl_status = QLabel("Idle")
        self.lbl_status.setWordWrap(True)
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p% — %v/%m")
        self.progress_bar.setTextVisible(True)
        btn_render_sel = QPushButton("Render Selected")
        btn_render_sel.setStyleSheet(style_blue_button())
        btn_render_sel.clicked.connect(self.render_selected)
        btn_render_all = QPushButton("Render All (Outdated)")
        btn_render_all.setStyleSheet(style_green_button())
        btn_render_all.clicked.connect(self.render_all_outdated)
        btn_force_sel = QPushButton("Force Render Selected")
        btn_force_sel.setStyleSheet(style_primary_button())
        btn_force_sel.setToolTip(
            "Re-render selected project(s) unconditionally.\n"
            "Bypasses the 'up-to-date' cache check — useful when\n"
            "resolution, encoder, or TTS settings have changed."
        )
        btn_force_sel.clicked.connect(self.render_selected_force)
        self.btn_stop = QPushButton("Stop All")
        self.btn_stop.setStyleSheet(style_stop_button())
        self.btn_stop.clicked.connect(self.stop_all_renders)
        self.btn_stop.setEnabled(False)
        ctrl_layout.addRow(self.lbl_workers)
        ctrl_layout.addRow(self.lbl_active_jobs)
        ctrl_layout.addRow(self.lbl_queue_count)
        ctrl_layout.addRow("Status:", self.lbl_status)
        ctrl_layout.addRow(self.progress_bar)
        btn_row1 = QHBoxLayout()
        btn_row1.addWidget(btn_render_sel)
        btn_row1.addWidget(btn_render_all)
        ctrl_layout.addRow(btn_row1)
        btn_row1b = QHBoxLayout()
        btn_row1b.addWidget(btn_force_sel)
        ctrl_layout.addRow(btn_row1b)
        btn_row2 = QHBoxLayout()
        btn_row2.addWidget(self.btn_stop)
        ctrl_layout.addRow(btn_row2)
        # ── Thumbnail button (independent of render) ────────────────
        btn_thumb = QPushButton("Generate Thumbnail")
        btn_thumb.setStyleSheet(style_button())
        btn_thumb.setToolTip(
            "Generate a PNG thumbnail from the first slide.\n"
            "This runs independently — no video re-render needed.\n"
            "Resolution is auto-detected from the slide's render dimensions."
        )
        btn_thumb.clicked.connect(self.generate_thumbnail_selected)
        ctrl_layout.addRow(btn_thumb)
        grp_controls.setLayout(ctrl_layout)
        right_layout.addWidget(grp_controls)
        grp_logs = QGroupBox("Application Logs")
        grp_logs.setStyleSheet(style_group_box_bordered())
        log_layout = QVBoxLayout()
        log_toolbar = QHBoxLayout()
        log_toolbar.setContentsMargins(0, 0, 0, 0)
        self.log_level_combo = QComboBox()
        self.log_level_combo.addItems(["All", "Info+", "Warning+", "Error+"])
        self.log_level_combo.setStyleSheet(style_input(padding="4px", radius="3px"))
        self.log_level_combo.currentTextChanged.connect(self._set_log_level)
        btn_clear_log = QPushButton("Clear")
        btn_clear_log.setStyleSheet(style_button_small())
        btn_clear_log.clicked.connect(self._clear_log)
        log_toolbar.addWidget(QLabel("Level:"))
        log_toolbar.addWidget(self.log_level_combo)
        log_toolbar.addStretch()
        log_toolbar.addWidget(btn_clear_log)
        log_layout.addLayout(log_toolbar)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet(style_log_text())
        log_layout.addWidget(self.log_text)
        grp_logs.setLayout(log_layout)
        right_layout.addWidget(grp_logs)
        splitter.addWidget(right_panel)
        splitter.setSizes([300, 1100])

    def _clear_log(self) -> None:
        self.log_text.clear()
        logger.info("[App] Log cleared")

    def generate_thumbnail_selected(self) -> None:
        """Launch standalone thumbnail workers for all selected projects."""
        paths = self.get_selected_projects()
        if not paths:
            QMessageBox.information(self, "Info", "No projects selected.")
            return
        worker_config = self.config.settings.copy()
        for project_path in paths:
            project_name = os.path.basename(project_path)

            # ── Landscape-only pre-check ────────────────────────────
            _cw = worker_config.get("width")
            _ch = worker_config.get("height")
            if _cw is not None and _ch is not None:
                if int(_cw) <= int(_ch):
                    logger.info(
                        f"[Scheduler] Skipping thumbnail for "
                        f"'{project_name}' — non-landscape "
                        f"resolution ({_cw}x{_ch})"
                    )
                    self.statusBar().showMessage(
                        f"Thumbnail skipped for '{project_name}' "
                        f"— non-landscape aspect ratio",
                        5000,
                    )
                    continue

            # Skip if already running
            if any(
                p_info["path"] == project_path
                for p_info in self.active_processes.values()
            ):
                logger.warning(
                    f"[Scheduler] Skipping thumbnail for '{project_name}' "
                    f"— already active"
                )
                continue
            proc = self._mp_ctx.Process(
                target=generate_thumbnail_worker,
                args=(project_path, self.render_queue, worker_config, True),
            )
            proc.start()
            self.active_processes[proc.pid] = {
                "proc": proc, "path": project_path,
            }
            logger.info(
                f"[Scheduler] Started thumbnail worker PID {proc.pid} "
                f"for '{project_name}'"
            )
        if len(paths) == 1:
            self.lbl_status.setText(
                f"Generating thumbnail: {os.path.basename(paths[0])}"
            )
        else:
            self.lbl_status.setText(
                f"Generating thumbnails for {len(paths)} project(s)"
            )
        self.update_stats()
    # ══════════════════════════════════════════════════════════════════════
    #  TAB MANAGEMENT
    # ══════════════════════════════════════════════════════════════════════

    def open_settings_tab(self, initial=False):
        for i in range(self.tabs.count()):
            if isinstance(self.tabs.widget(i), SettingsTabWidget):
                self.tabs.setCurrentIndex(i)
                return
        settings_widget = SettingsTabWidget(self.config)
        settings_widget.settings_changed.connect(self._on_settings_changed)
        settings_widget.library_refresh_requested.connect(self.refresh_library)
        idx = self.tabs.addTab(settings_widget, "⚙ Settings")
        self.tabs.setCurrentIndex(idx)

    def _refresh_settings_after_import(self) -> None:
        """After an import that may have auto-detected resolution, refresh
        any open SettingsTabWidget so its combos reflect the new values."""
        from dialogs import SettingsTabWidget
        for i in range(self.tabs.count()):
            widget = self.tabs.widget(i)
            if isinstance(widget, SettingsTabWidget):
                try:
                    widget.refresh_ui_values()
                    widget.refresh_backend_settings_ui()
                except Exception as exc:
                    logger.debug(f"Error refreshing settings tab: {exc}")

    # Loggers whose DEBUG spam should be suppressed when verbose=off
    _NOISY_LOGGERS = [
        "PIL.PngImagePlugin",
        "PIL.Image",
        "asyncio",
        "urllib3.connectionpool",
        "httpx",
        "httpcore",
        "PIL",
        "py.warnings",
    ]

    def _apply_verbose_filters(self, verbose: bool) -> None:
        """Add or remove VerboseFilter on noisy third-party loggers.
        When *verbose* is False (the default) a ``VerboseFilter`` is added
        to every logger in ``_NOISY_LOGGERS`` so their DEBUG-level messages
        are suppressed.  INFO and above always pass through.
        When *verbose* is True the filters are removed so all raw messages
        (including DEBUG) pass through unchanged.
        """
        # Remove any existing filters first
        for vf in self._verbose_filters:
            for name in self._NOISY_LOGGERS:
                logging.getLogger(name).removeFilter(vf)
        self._verbose_filters.clear()
        if not verbose:
            vf = VerboseFilter()
            for name in self._NOISY_LOGGERS:
                logging.getLogger(name).addFilter(vf)
            self._verbose_filters.append(vf)

    def _on_settings_changed(self, key: str, value) -> None:
        """Handle signals from the Settings tab."""
        if key == "__generate_thumbnail__":
            self.generate_thumbnail_selected()
        elif key == "verbose":
            self._apply_verbose_filters(bool(value))

    # ── Auto-thumbnail on first slide edit ───────────────────────

    def _on_slide_content_saved(self, project_path: str, slide_index: int) -> None:
        """When slide 1 is saved, schedule a thumbnail re-render if needed."""
        if slide_index != 0:
            return
        # Quick mtime check — avoid spawning a process if thumbnail is newer
        if not self._first_slide_needs_thumbnail_update(project_path):
            return
        # Debounce: restart the timer so rapid edits don't spawn multiple workers
        self._pending_thumb_project = project_path
        self._thumb_debounce.start()

    def _on_thumb_debounce_timeout(self) -> None:
        """Fire the thumbnail worker after debounce period."""
        project_path = self._pending_thumb_project
        self._pending_thumb_project = None
        if not project_path:
            return
        project_name = os.path.basename(project_path)

        # ── Landscape-only pre-check ────────────────────────────────
        _cw = self.config.settings.get("width")
        _ch = self.config.settings.get("height")
        if _cw is not None and _ch is not None:
            if int(_cw) <= int(_ch):
                logger.info(
                    f"[UI] Skipping auto-thumbnail for '{project_name}' "
                    f"— non-landscape resolution ({_cw}x{_ch})"
                )
                self.statusBar().showMessage(
                    f"Auto-thumbnail skipped for '{project_name}' "
                    f"— non-landscape aspect ratio",
                    5000,
                )
                return

        # Don't spawn if a process for this project is already active
        if any(
            p_info["path"] == project_path
            for p_info in self.active_processes.values()
        ):
            logger.debug(
                f"[UI] Skipping auto-thumbnail for '{project_name}' — already active"
            )
            return
        worker_config = self.config.settings.copy()
        proc = self._mp_ctx.Process(
            target=generate_thumbnail_worker,
            args=(project_path, self.render_queue, worker_config, False),
        )
        proc.start()
        self.active_processes[proc.pid] = {
            "proc": proc, "path": project_path,
        }
        logger.info(
            f"[UI] Auto-thumbnail triggered for '{project_name}' (PID {proc.pid})"
        )

    def _first_slide_needs_thumbnail_update(self, project_path: str) -> bool:
        """Cheap mtime check: is slide1.html newer than the thumbnail?"""
        import glob as _glob
        from utils import natural_sort_key as _nsk
        html_files = sorted(
            _glob.glob(os.path.join(project_path, "slide*.html")),
            key=_nsk,
        )
        if not html_files:
            return False
        first_html = html_files[0]
        # Find thumbnail path using the window's live config
        output_dir = self.config.get("output_dir", "Output")
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(AppConfig.APP_ROOT, output_dir)
        thumb_path = os.path.join(
            output_dir, f"{os.path.basename(project_path)}_thumbnail.png"
        )
        if not os.path.exists(thumb_path):
            return True  # No thumbnail yet — needs one
        return os.path.getmtime(first_html) > os.path.getmtime(thumb_path)

    def open_project_tab(self, path: str):
        project_name = os.path.basename(path)
        for i in range(self.tabs.count()):
            widget = self.tabs.widget(i)
            if isinstance(widget, SlideEditorTabWidget) and getattr(widget, "project_path", None) == path:
                self.tabs.setCurrentIndex(i)
                if hasattr(widget, "refresh_from_disk"):
                    widget.refresh_from_disk()
                return
        editor = SlideEditorTabWidget(path, self.config, self)
        editor.content_saved.connect(self._on_slide_content_saved)
        idx = self.tabs.addTab(editor, f"📝 {project_name}")
        self.tabs.setCurrentIndex(idx)

    def close_tab(self, index: int):
        widget = self.tabs.widget(index)
        if isinstance(widget, SettingsTabWidget):
            widget._flush_pending_settings()
            self.tabs.removeTab(index)
            return
        if isinstance(widget, NewProjectTabWidget):
            if hasattr(widget, 'cleanup'):
                widget.cleanup()
            if hasattr(widget, '_on_created_callback'):
                widget._on_created_callback = None
        if isinstance(widget, SlideEditorTabWidget):
            widget.save_and_cleanup()
        self.tabs.removeTab(index)
    # ══════════════════════════════════════════════════════════════════════
    #  LIBRARY & PROJECT MANAGEMENT
    # ══════════════════════════════════════════════════════════════════════

    def on_theme_changed(self, theme_name: str):
        try:
            set_theme(theme_name)
            logger.info(f"[UI] Switched to theme: {theme_name}")
            self.project_list.setStyleSheet(_project_list_stylesheet())
            self.refresh_library()
        except Exception as exc:
            logger.error(f"[UI] Failed to change theme ({type(exc).__name__}: {exc})", exc_info=True)

    def refresh_library(self):
        if self._refreshing_library or self._shutting_down:
            return
        self._refreshing_library = True
        try:
            self._do_refresh_library()
        finally:
            self._refreshing_library = False

    def _do_refresh_library(self):
        current_selections = self.get_selected_projects()
        self.project_dirs = get_project_dirs()
        # Backfill manifests for any project missing one (e.g. older projects)
        for p in self.project_dirs:
            manifest_path = os.path.join(p, "manifest.json")
            if not os.path.exists(manifest_path):
                update_library_manifest(p)
        self.project_list.blockSignals(True)
        self.project_list.clear()
        project_data = get_library_status_data(
            self.project_dirs, self.active_processes, self.pending_projects
        )
        sort_mode = self.sort_combo.currentText()
        if sort_mode == "Name (A->Z)":
            project_data.sort(key=lambda x: natural_sort_key(x["name"]))
        elif sort_mode == "Name (Z->A)":
            project_data.sort(key=lambda x: natural_sort_key(x["name"]), reverse=True)
        elif sort_mode == "Status":
            project_data.sort(key=lambda x: STATUS_SORT_PRIORITY.get(x["status"], 99))
        elif sort_mode == "Last Rendered":
            project_data.sort(key=lambda x: x.get("last_rendered") or 0.0, reverse=True)
        for pd in project_data:
            list_widget = ProjectListItemWidget(
                pd["name"], pd["path"], self.selection_mgr.state
            )
            list_widget.update_status(
                pd["status"], 0, pd["detail"],
                slide_count=pd.get("slide_count", 0),
                source=pd.get("source", ""),
                source_resolution=pd.get("source_resolution"),
            )
            item = QListWidgetItem()
            item.setSizeHint(list_widget.sizeHint())
            item.setData(Qt.ItemDataRole.UserRole, pd["path"])
            self.project_list.addItem(item)
            self.project_list.setItemWidget(item, list_widget)
        self.project_list.blockSignals(False)
        # Restore previously selected items (multi-select aware)
        for sel_path in current_selections:
            for i in range(self.project_list.count()):
                item = self.project_list.item(i)
                if item.data(Qt.ItemDataRole.UserRole) == sel_path:
                    item.setSelected(True)
                    break
        self.selection_mgr.refresh()
        self.update_stats()

    def get_selected_project_path(self) -> Optional[str]:
        items = self.project_list.selectedItems()
        return items[0].data(Qt.ItemDataRole.UserRole) if items else None

    def get_selected_projects(self) -> list[str]:
        return [
            item.data(Qt.ItemDataRole.UserRole)
            for item in self.project_list.selectedItems()
        ]

    def create_new_project(self, prefill_import_path: str = ""):
        """Open the unified New Project tab.  If *prefill_import_path* is given
        the Import radio is auto-selected and the path pre-filled."""
        for i in range(self.tabs.count()):
            if isinstance(self.tabs.widget(i), NewProjectTabWidget):
                self.tabs.setCurrentIndex(i)
                if prefill_import_path:
                    widget = self.tabs.widget(i)
                    widget.radio_import.setChecked(True)
                    widget.import_path_edit.setText(prefill_import_path)
                    if not widget.name_edit.text().strip():
                        widget.name_edit.setText(
                            os.path.splitext(os.path.basename(prefill_import_path))[0]
                        )
                return
        np_tab = NewProjectTabWidget(
            self.config,
            self.on_new_project_created,
            prefill_import_path=prefill_import_path,
        )
        idx = self.tabs.addTab(np_tab, "+ New Project")
        self.tabs.setCurrentIndex(idx)

    def on_new_project_created(self, project_path: str):
        QTimer.singleShot(0, lambda: self._handle_new_project(project_path))

    def _handle_new_project(self, project_path: str):
        try:
            for i in range(self.tabs.count()):
                if isinstance(self.tabs.widget(i), NewProjectTabWidget):
                    self.tabs.removeTab(i)
                    break
            # Ensure library manifest is up-to-date before refreshing the list
            update_library_manifest(project_path)
            self.refresh_library()
            self.open_project_tab(project_path)
        except Exception as exc:
            logger.error(f"[UI] Error handling new project: {exc}", exc_info=True)
    # ──────────────────────────────────────────────────────────────────────
    # FIX #4: Removed the FIRST duplicate `import_file_as_new_project()`
    # that merely called `self.create_new_project(prefill_import_path=…)`.
    # The second, more complete version below is kept as the single
    # canonical implementation.
    # ──────────────────────────────────────────────────────────────────────
    _IMPORT_FILTER = (
        "Importable Files (*.pdf *.pptx *.txt *.csv *.json);;"
        "PowerPoint Files (*.pptx);;PDF Files (*.pdf);;"
        "Text Files (*.txt);;CSV Files (*.csv);;"
        "JSON Files (*.json);;All Files (*)"
    )

    def import_file_as_new_project(self):
        """Quick-import: pick a file and create a project from it in one step."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Import File as New Project", "", self._IMPORT_FILTER
        )
        if not file_path:
            return
        default_name = os.path.splitext(os.path.basename(file_path))[0]
        project_name, ok = QInputDialog.getText(
            self, "Project Name",
            "Enter a name for the new project (leave blank to use filename):",
            text=default_name,
        )
        if not ok:
            return
        project_name = project_name.strip() or None
        # Check if project already exists and ask the user
        on_exists = "error"
        from dialogs import _ask_overwrite_or_append
        effective_name = project_name or os.path.splitext(os.path.basename(file_path))[0]
        _projects_root = self.config.get("projects_root", "Projects")
        if not os.path.isabs(_projects_root):
            _projects_root = os.path.join(AppConfig.APP_ROOT, _projects_root)
        candidate_path = os.path.join(_projects_root, effective_name)
        if os.path.exists(candidate_path):
            on_exists = _ask_overwrite_or_append(self, effective_name)
            if on_exists is None:
                return  # User cancelled
        self.lbl_status.setText(f"Importing '{os.path.basename(file_path)}'...")

        def _on_import_done(src_path, result):
            if isinstance(result, Exception):
                if isinstance(result, ImportError):
                    logger.error(f"[Import] Missing library: {result}")
                    QMessageBox.critical(self, "Missing Library", str(result))
                elif isinstance(result, ValueError):
                    logger.warning(f"[Import] Invalid file: {result}")
                    QMessageBox.warning(self, "Import Error", str(result))
                else:
                    logger.error(
                        f"[Import] Failed to import '{os.path.basename(src_path)}': {result}",
                        exc_info=True,
                    )
                    QMessageBox.critical(self, "Error", f"Failed to import file:\n{result}")
                self.lbl_status.setText("Import failed")
                return
            logger.info(
                f"[Import] Created project from "
                f"'{os.path.basename(src_path)}' \u2192 "
                f"'{os.path.basename(result)}'"
            )
            update_library_manifest(result)
            self.refresh_library()
            self._refresh_settings_after_import()
            self.open_project_tab(result)
            self.lbl_status.setText("Import complete")

        self._import_worker = _ImportWorker(file_path, self.config.settings,
                               project_name=project_name, on_exists=on_exists,
                               parent=self)
        self._import_worker.finished.connect(_on_import_done)
        self._import_worker.start()

    def filter_library(self):
        query = self.search_edit.text().strip().lower()
        for i in range(self.project_list.count()):
            item = self.project_list.item(i)
            name = os.path.basename(item.data(Qt.ItemDataRole.UserRole))
            item.setHidden(query not in name.lower())

    def show_library_context_menu(self, pos):
        item = self.project_list.itemAt(pos)
        if not item:
            return
        path = item.data(Qt.ItemDataRole.UserRole)
        menu = QMenu(self)
        menu.setStyleSheet(
            "QMenu { background-color: #2d2d2d; color: white; } "
            "QMenu::item { padding: 5px 20px; } "
            "QMenu::item:selected { background-color: #0078d4; }"
        )
        action_open = menu.addAction("Open Folder")
        action_edit = menu.addAction("Edit Project")
        action_render = menu.addAction("Render Now")
        action_output = menu.addAction("Show Output Video")
        menu.addSeparator()
        action_import_append = menu.addAction("Import & Append Slides\u2026")
        menu.addSeparator()
        action_delete = menu.addAction("Delete Project")
        action = menu.exec(self.project_list.mapToGlobal(pos))

        # Gather all selected paths for multi-selection-aware actions
        all_selected = self.get_selected_projects()

        if action == action_open:
            self.open_project_path(path)
        elif action == action_edit:
            targets = all_selected if len(all_selected) > 1 else [path]
            for p in targets:
                self.open_project_tab(p)
        elif action == action_render:
            targets = all_selected if len(all_selected) > 1 else [path]
            for p in targets:
                self.add_to_queue(p)
        elif action == action_output:
            self.show_output_video(path)
        elif action == action_import_append:
            self.import_file_append_to_project(path)
        elif action == action_delete:
            targets = all_selected if len(all_selected) > 1 else [path]
            if len(targets) == 1:
                self.delete_project_for(targets[0])
            else:
                confirm = QMessageBox.question(
                    self, "Delete Projects",
                    f"Delete {len(targets)} selected project(s)?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )
                if confirm == QMessageBox.StandardButton.Yes:
                    for p in targets:
                        self.delete_project_for(p)

    def open_project_path(self, path: str):
        import platform
        system = platform.system()
        if system == "Windows":
            os.startfile(path)
        elif system == "Darwin":
            subprocess.call(["open", path])
        else:
            subprocess.call(["xdg-open", path])

    def open_editor(self):
        paths = self.get_selected_projects()
        if not paths:
            QMessageBox.information(self, "Info", "No project selected.")
            return
        for path in paths:
            self.open_project_tab(path)

    def delete_project(self):
        paths = self.get_selected_projects()
        if not paths:
            self.statusBar().showMessage("No project selected", 3000)
            return
        if len(paths) == 1:
            self.delete_project_for(paths[0])
            return
        confirm = QMessageBox.question(
            self, "Delete Projects",
            f"Delete {len(paths)} selected project(s)?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm == QMessageBox.StandardButton.Yes:
            for path in paths:
                self.delete_project_for(path)

    def show_output_video(self, path: str):
        from config import AppConfig
        manifest = load_project_manifest(path)
        output_info = manifest.get("output", {})
        # v2: output.video; v1: output.video_filename
        video_filename = output_info.get("video") or output_info.get("video_filename")
        # v2: output dir removed (use global config); v1: output.dir
        output_dir = output_info.get("dir") or self.config.get("output_dir", "Output")
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(AppConfig.APP_ROOT, output_dir)
        if not video_filename:
            QMessageBox.information(self, "Info", "No output video found.")
            return
        video_path = os.path.join(output_dir, video_filename)
        if os.path.exists(video_path):
            self.open_project_path(video_path)
        else:
            QMessageBox.information(self, "Info", "Video file does not exist.")

    def delete_project_for(self, path: str):
        if path in self.pending_projects:
            self.pending_projects.remove(path)
            logger.info(
                f"[UI] Removed '{os.path.basename(path)}' from render queue"
            )
            self.refresh_library()
            return
        for p_info in self.active_processes.values():
            if p_info["path"] == path:
                QMessageBox.warning(
                    self, "Busy",
                    "Cannot delete project currently rendering."
                )
                return
        project_name = os.path.basename(path)
        confirm = QMessageBox.question(
            self, "Delete Project", f"Delete {project_name}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm == QMessageBox.StandardButton.Yes:
            try:
                if os.path.exists(path):
                    shutil.rmtree(path)
                    logger.info(f"[UI] Deleted project: {project_name}")
                self.refresh_library()
            except Exception as exc:
                logger.error(
                    f"[UI] Failed to delete project '{project_name}' ({type(exc).__name__}: {exc})",
                    exc_info=True,
                )
                QMessageBox.critical(self, "Error", f"Failed to delete project:\n{exc}")

    def on_project_double_click(self, item):
        # IMPROVE: Double-click opens the project editor tab, not the
        # file explorer.  The context menu already has "Open Folder"
        # for the file explorer use case.
        path = item.data(Qt.ItemDataRole.UserRole)
        self.open_project_tab(path)
    # ══════════════════════════════════════════════════════════════════════
    #  FILE IMPORT HANDLERS
    # ══════════════════════════════════════════════════════════════════════

    def import_file_append_to_project(self, project_path: str):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Import & Append Slides", "", FILE_FILTER_IMPORT
        )
        if not file_path:
            return
        project_name = os.path.basename(project_path)
        self.lbl_status.setText(f"Appending to '{project_name}'...")

        def _on_append_done(src_path, result):
            if isinstance(result, ImportError):
                logger.error(f"[Import] Missing library: {result}")
                QMessageBox.critical(self, "Missing Library", str(result))
                self.lbl_status.setText("Import failed")
            elif isinstance(result, ValueError):
                logger.warning(f"[Import] Invalid input: {result}")
                QMessageBox.warning(self, "Import Error", str(result))
                self.lbl_status.setText("Import failed")
            elif isinstance(result, Exception):
                logger.error(f"[Import] Append failed: {result}", exc_info=True)
                QMessageBox.critical(self, "Error", f"Failed to import file:\n{result}")
                self.lbl_status.setText("Import failed")
            else:
                logger.info(
                    f"[Import] Appended {result} slide(s) to '{project_name}' "
                    f"from '{os.path.basename(src_path)}'"
                )
                QMessageBox.information(
                    self, "Import Complete",
                    f"Appended {result} slide(s) to {project_name}."
                )
                update_library_manifest(project_path)
                self.refresh_library()
                for i in range(self.tabs.count()):
                    widget = self.tabs.widget(i)
                    if (
                        isinstance(widget, SlideEditorTabWidget)
                        and getattr(widget, "project_path", None) == project_path
                        and hasattr(widget, "refresh_from_disk")
                    ):
                        widget.refresh_from_disk()
                        break
                self.lbl_status.setText("Import complete")

        self._append_import_worker = _AppendImportWorker(
            file_path, project_path, self.config.settings, parent=self
        )
        self._append_import_worker.finished.connect(_on_append_done)
        self._append_import_worker.start()
    # ══════════════════════════════════════════════════════════════════════
    #  SCHEDULER & RENDER LOGIC
    # ══════════════════════════════════════════════════════════════════════

    def render_selected(self):
        paths = self.get_selected_projects()
        if not paths:
            QMessageBox.information(self, "Info", "No projects selected.")
            return
        if len(paths) == 1:
            if should_skip_render(paths[0], self.config.settings):
                self.statusBar().showMessage("Project is already up to date or queued", 3000)
                return
            self.add_to_queue(paths[0])
            return
        busy_paths, ready_paths = [], []
        active_path_set = {p_info["path"] for p_info in self.active_processes.values()}
        for path in paths:
            if path in active_path_set or path in self.pending_projects:
                busy_paths.append(os.path.basename(path))
            else:
                ready_paths.append(path)
        if not ready_paths:
            QMessageBox.information(
                self, "All Busy",
                "All selected projects are already rendering or queued."
            )
            return
        msg_parts = [f"Add {len(ready_paths)} project(s) to render queue?"]
        if busy_paths:
            msg_parts.append(f"\n\nSkipped (already active/queued): {len(busy_paths)}")
            msg_parts.extend(f"  • {n}" for n in busy_paths[:5])
            if len(busy_paths) > 5:
                msg_parts.append(f"  ... and {len(busy_paths) - 5} more")
        msg_parts.append("\n\nReady to render:")
        msg_parts.extend(f"  • {os.path.basename(p)}" for p in ready_paths[:10])
        if len(ready_paths) > 10:
            msg_parts.append(f"  ... and {len(ready_paths) - 10} more")
        reply = QMessageBox.question(
            self, "Batch Render", "\n".join(msg_parts),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if reply == QMessageBox.StandardButton.Yes:
            for path in ready_paths:
                self.add_to_queue(path, schedule=False)
            self.refresh_library()
            logger.info(
                f"[Scheduler] Batch render: {len(ready_paths)} project(s) queued"
            )
            self.lbl_status.setText(f"Added {len(ready_paths)} projects to queue")
            if not self._pool_timer.isActive():
                self._pool_timer.start(500)

    def render_selected_force(self):
        """Force-render selected projects, bypassing all cache checks.

        Sets a ``_force_render`` flag in each project's manifest so the
        worker pipeline always re-renders the video even if nothing
        appears to have changed.  This is the fix for resolution mismatch
        scenarios where the global config changed after a previous render.
        """
        paths = self.get_selected_projects()
        if not paths:
            QMessageBox.information(self, "Info", "No projects selected.")
            return
        from utils import load_project_manifest, save_project_manifest
        for path in paths:
            manifest = load_project_manifest(path)
            manifest["_force_render"] = True
            save_project_manifest(path, manifest)
            self.add_to_queue(path, schedule=False)
        self.refresh_library()
        logger.info(
            f"[Scheduler] Force-render: {len(paths)} project(s) queued"
        )
        self.lbl_status.setText(f"Force-render: {len(paths)} project(s) queued")
        if not self._pool_timer.isActive():
            self._pool_timer.start(500)

    def render_all_outdated(self):
        paths = [
            self.project_list.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self.project_list.count())
        ]
        active_path_set = {
            p_info["path"] for p_info in self.active_processes.values()
        }
        count = 0
        for path in paths:
            if should_skip_render(path, self.config.settings):
                continue
            if path in active_path_set or path in self.pending_projects:
                continue
            self.add_to_queue(path, schedule=False)
            count += 1
        if count:
            self.refresh_library()
            logger.info(
                f"[Scheduler] Queued {count} outdated project(s) for re-render"
            )
            if not self._pool_timer.isActive():
                self._pool_timer.start(500)
        else:
            QMessageBox.information(
                self, "Up to Date",
                "All projects are already up to date based on current settings."
            )

    def add_to_queue(self, project_path: str, schedule: bool = True):
        if project_path in self.pending_projects:
            return
        if any(p["path"] == project_path for p in self.active_processes.values()):
            return
        self.pending_projects.append(project_path)
        logger.info(
            f"[Scheduler] Queued '{os.path.basename(project_path)}' for rendering"
        )
        self.update_stats()
        if schedule and not self._pool_timer.isActive():
            self._pool_timer.start(500)

    def schedule_jobs(self):
        if self._shutting_down:
            return
        # Don't dispatch during graceful drain
        if self._draining:
            return
        # Re-read max_workers so live config changes take effect
        self.max_workers = self.config.get("render_workers", 3)
        # Dynamic pool resize: grow or shrink the pool to match max_workers
        self._resize_worker_pool()
        if not self.pending_projects and not self.active_processes:
            self._pool_timer.stop()
            self.lbl_status.setText("All Tasks Completed")
            self.progress_bar.setValue(0)
            self.btn_stop.setEnabled(False)
            logger.info("[Scheduler] All rendering tasks finished")
            self.refresh_library()
            return
        self.btn_stop.setEnabled(True)
        started = False
        # Count only pool-dispatched items (real render jobs, not thumbs)
        in_flight = sum(
            1 for v in self.active_processes.values()
            if v.get("pool_dispatched", False)
        )
        while in_flight < self.max_workers and self.pending_projects:
            next_project = self.pending_projects.pop(0)
            self.start_worker(next_project)
            in_flight += 1
            started = True
        if started:
            self.update_stats()
            self._mark_library_dirty()

    def start_worker(self, project_path: str):
        """Feed a project to the persistent worker pool via the work queue."""
        ffmpeg = detect_ffmpeg()
        if not ffmpeg:
            project_name = os.path.basename(project_path)
            logger.error(
                f"[Scheduler] Cannot start worker for '{project_name}': "
                f"FFmpeg not found."
            )
            self.update_item_status(
                project_name, RenderStatus.ERROR, "FFmpeg not found"
            )
            return
        self._ensure_worker_pool()
        project_name = os.path.basename(project_path)
        # Dispatch to a specific idle worker (queue is empty)
        dispatched = False
        for proc in self._worker_pool:
            if proc.is_alive() and proc.pid in self._worker_queues:
                try:
                    self._worker_queues[proc.pid].put_nowait(project_path)
                    dispatched = True
                    break
                except queue.Full:
                    continue
        if not dispatched:
            # All busy — put in first alive worker's queue
            for proc in self._worker_pool:
                if proc.is_alive() and proc.pid in self._worker_queues:
                    try:
                        self._worker_queues[proc.pid].put(project_path, timeout=2)
                    except (queue.Full, OSError):
                        logger.warning("[GUI] Worker queue full, skipping project dispatch")
                    break
        # Track via a synthetic slot so the UI knows how many are active.
        slot_id = f"pool-{project_path}"
        self.active_processes[slot_id] = {
            "proc": None,
            "path": project_path,
            "pool_dispatched": True,
        }
        self._start_render_timer(project_name)
        active_render_count = sum(
            1 for v in self.active_processes.values()
            if v.get("pool_dispatched", False)
        )
        logger.info(
            f"[Scheduler] Dispatched '{project_name}' to worker pool "
            f"({active_render_count} projects in-flight)"
        )
        self.lbl_status.setText(f"Active Jobs: {active_render_count}")
    # ══════════════════════════════════════════════════════════════════════
    #  TEMP CLEANUP
    # ══════════════════════════════════════════════════════════════════════

    def _cleanup_stale_temp(self):
        temp_base = os.path.join(AppConfig.APP_ROOT, "temp")
        if not os.path.isdir(temp_base):
            return
        try:
            entries = os.listdir(temp_base)
            if not entries:
                os.rmdir(temp_base)
                return
            logger.info(
                f"[Startup] Cleaning up {len(entries)} stale temp directory(ies)"
            )
            for entry in entries:
                entry_path = os.path.join(temp_base, entry)
                try:
                    if os.path.isdir(entry_path):
                        shutil.rmtree(entry_path, ignore_errors=True)
                        logger.debug(f"[Startup] Removed stale temp: {entry}")
                except Exception as exc:
                    logger.warning(
                        f"[Startup] Could not remove stale temp '{entry}' ({type(exc).__name__}: {exc})",
                        exc_info=True,
                    )
            try:
                if not os.listdir(temp_base):
                    os.rmdir(temp_base)
            except OSError:
                pass
        except Exception as exc:
            logger.warning(
                f"[Startup] Error during stale temp cleanup ({type(exc).__name__}: {exc})",
                exc_info=True,
            )
    # ══════════════════════════════════════════════════════════════════════
    #  STOP / CANCEL
    # ══════════════════════════════════════════════════════════════════════

    def stop_all_renders(self):
        logger.info("[Scheduler] Stopping all renders by user request...")
        self._stopping = True
        self._draining = True
        # Stop the pool timer FIRST so _pool_tick cannot respawn workers
        # during the termination loop below.
        self._pool_timer.stop()
        # Set cooldown so the reaper doesn't immediately respawn workers
        self._pool_cooldown_until = time.time() + 3.0
        self.pending_projects.clear()
        # Drain the work queue so workers don't pick up more projects
        self._drain_work_queue()
        # Terminate pool workers that may have already dequeued work
        for proc in self._worker_pool:
            try:
                proc.terminate()
                proc.join(timeout=2)
            except Exception:
                pass
        self._worker_pool.clear()
        for wq in self._worker_queues.values():
            try:
                wq.close()
                wq.join_thread()
            except Exception:
                pass
        self._worker_queues.clear()
        for pid, proc_data in list(self.active_processes.items()):
            proc = proc_data.get("proc")
            pname = os.path.basename(proc_data.get("path", "unknown"))
            if proc is not None:
                try:
                    proc.terminate()
                    proc.join(timeout=1.0)
                    if proc.is_alive():
                        proc.kill()
                        logger.warning(
                            f"[Scheduler] Force-killed process {pid} ({pname})"
                        )
                    else:
                        logger.info(
                            f"[Scheduler] Terminated process {pid} ({pname})"
                        )
                except Exception as exc:
                    logger.error(
                        f"[Scheduler] Error stopping process {pid} ({pname}) ({type(exc).__name__}: {exc})",
                        exc_info=True,
                    )
        self.active_processes.clear()
        self._render_start_times.clear()
        self._last_progress_update.clear()
        self._draining = False
        self._stopping = False
        self.update_stats()
        self.refresh_library()
        self.lbl_status.setText("Stopped by user")
        self.btn_stop.setEnabled(False)
    # ══════════════════════════════════════════════════════════════════════
    #  MESSAGE HANDLING
    # ══════════════════════════════════════════════════════════════════════

    def _on_queue_message(self, msg) -> None:
        if self._shutting_down or getattr(self, '_stopping', False):
            return
        try:
            # Legacy LOG tuple format
            if isinstance(msg, tuple) and len(msg) >= 2 and msg[0] == "LOG":
                if self._log_handler and not self._log_handler._closed:
                    self._log_handler._signal_helper.append_html.emit(msg[1])
                return
            # Dispatch by type name
            handler = _MESSAGE_HANDLERS.get(type(msg).__name__)
            if handler:
                handler(self, msg)
            else:
                logger.debug(
                    f"[Queue] Unhandled message type: {type(msg).__name__}"
                )
            # Clean up pool-dispatched slot when project finishes
            if isinstance(msg, (FinalizedMessage, ErrorMessage)):
                pname = getattr(msg, "project_name", None)
                if pname:
                    slot_to_remove = None
                    for sid, sdata in list(self.active_processes.items()):
                        if sdata.get("pool_dispatched") and os.path.basename(sdata.get("path", "")) == pname:
                            slot_to_remove = sid
                            break
                    if slot_to_remove:
                        del self.active_processes[slot_to_remove]
                        self._clear_render_timer(pname)
                        self._last_progress_update.pop(pname, None)
        except Exception as exc:
            logger.error(f"[Queue] Processing error: {exc}", exc_info=True)
    # ══════════════════════════════════════════════════════════════════════
    #  PROCESS REAPER
    # ══════════════════════════════════════════════════════════════════════

    def _reap_processes(self) -> None:
        if self._shutting_down:
            return
        dead_found = False
        # Reap non-pool processes (thumbnail workers, etc.)
        for pid in list(self.active_processes):
            proc_data = self.active_processes[pid]
            if proc_data.get("pool_dispatched", False):
                continue
            proc = proc_data.get("proc")
            if proc is None:
                continue
            if not proc.is_alive():
                project_name = os.path.basename(proc_data["path"])
                exit_code = proc.exitcode
                if exit_code == 0:
                    status = RenderStatus.READY
                    logger.info(
                        f"[Scheduler] Worker PID {pid} ({project_name}) "
                        f"completed successfully"
                    )
                elif exit_code is not None and exit_code < 0:
                    status = RenderStatus.ERROR
                    logger.error(
                        f"[Scheduler] Worker PID {pid} ({project_name}) "
                        f"killed by signal {-exit_code}"
                    )
                else:
                    status = RenderStatus.ERROR
                    logger.error(
                        f"[Scheduler] Worker PID {pid} ({project_name}) "
                        f"exited with code {exit_code}"
                    )
                if status == RenderStatus.ERROR:
                    elapsed_str = self._format_elapsed(project_name)
                    detail = "Failed \u2014 Check Logs"
                    if elapsed_str:
                        detail = f"Failed ({elapsed_str}) \u2014 Check Logs"
                    self.update_item_status(project_name, status, detail)
                else:
                    self._clear_render_timer(project_name)
                    self.update_item_status(project_name, status)
                self._last_progress_update.pop(project_name, None)
                del self.active_processes[pid]
                dead_found = True
        # Reap dead pool workers and respawn replacements.
        # Workers are permanent — if one dies (crash, OOM) we just
        # replace it.  The worker's own try/except + FinalizedMessage
        # handles error reporting for the project it was working on.
        if time.time() < self._pool_cooldown_until or self._draining:
            return
        alive = []
        for p in self._worker_pool:
            if p.is_alive():
                alive.append(p)
            else:
                logger.warning(
                    f"[WorkerPool] Worker PID {p.pid} died (code={p.exitcode}) "
                    f"— will respawn"
                )
                self._worker_queues.pop(p.pid, None)
        if len(alive) != len(self._worker_pool):
            self._worker_pool = alive
            self._ensure_worker_pool()
        if dead_found:
            self.update_stats()
            self._mark_library_dirty()
    #  STATUS UPDATES & POOL MANAGEMENT
    # ══════════════════════════════════════════════════════════════════════
    # ══════════════════════════════════════════════════════════

    def _pool_tick(self) -> None:
        """Combined pool maintenance tick: reap dead workers then schedule jobs."""
        self._reap_processes()
        self.schedule_jobs()

    def _ensure_worker_pool(self) -> None:
        """Make sure exactly *max_workers* persistent pool processes are alive.

        Respects the cooldown timer so workers aren't immediately respawned
        after a user-initiated stop.
        """
        # Respect cooldown (set by stop_all_renders)
        if time.time() < self._pool_cooldown_until:
            return
        # Never spawn workers while draining (stop_all_renders in progress)
        if self._draining:
            return
        self.max_workers = self.config.get("render_workers", 3)
        # Remove dead workers and clean up their tracking state
        alive = []
        for p in self._worker_pool:
            if p.is_alive():
                alive.append(p)
            else:
                self._worker_queues.pop(p.pid, None)
        self._worker_pool = alive
        while len(self._worker_pool) < self.max_workers:
            ffmpeg = detect_ffmpeg()
            if not ffmpeg:
                logger.warning(
                    "[WorkerPool] FFmpeg not found — skipping worker spawn"
                )
                break
            worker_config = self.config.settings.copy()
            worker_config["ffmpeg_path"] = ffmpeg
            worker_q = self._mp_ctx.Queue()
            proc = self._mp_ctx.Process(
                target=render_worker_loop,
                args=(worker_q, self.render_queue, worker_config),
                daemon=True,
            )
            proc.start()
            self._worker_pool.append(proc)
            self._worker_queues[proc.pid] = worker_q
            logger.info(
                f"[WorkerPool] Spawned persistent worker PID {proc.pid} "
                f"({len(self._worker_pool)}/{self.max_workers})"
            )

    def _resize_worker_pool(self) -> None:
        """Dynamically grow or shrink the pool to match the current max_workers.

        Called from schedule_jobs() so the pool adapts to live config changes
        without needing an app restart.  Excess workers are sent a single
        poison pill to shut them down gracefully; missing workers are spawned.
        """
        target = self.max_workers  # already updated by schedule_jobs()
        alive = [p for p in self._worker_pool if p.is_alive()]
        self._worker_pool = alive

        if len(alive) == target:
            return
        old_count = len(alive)

        if len(alive) > target:
            # Shrink: send poison pills to excess workers
            excess = len(alive) - target
            shrunk = 0
            for proc in alive[:excess + 2]:  # +2 buffer for race
                if shrunk >= excess:
                    break
                if proc.pid in self._worker_queues:
                    try:
                        self._worker_queues[proc.pid].put(None, timeout=1)
                        shrunk += 1
                    except Exception:
                        pass
            logger.info(
                f"[WorkerPool] Resizing {old_count} → {target} "
                f"(sent {shrunk} targeted shutdown pill(s) to reach {target})"
            )
        elif len(alive) < target:
            # Grow: _ensure_worker_pool will spawn the missing ones
            logger.info(
                f"[WorkerPool] Resizing {old_count} → {target} "
                f"(need {target - len(alive)} more worker(s))"
            )
            self._ensure_worker_pool()

    def _drain_work_queue(self) -> None:
        """Remove all pending items from all per-worker queues."""
        for wq in self._worker_queues.values():
            try:
                while True:
                    wq.get_nowait()
            except Exception:
                pass

    def _shutdown_worker_pool(self) -> None:
        """Drain work queue, send poison pills and wait for pool workers to exit."""
        # Drain pending work first so workers don't start new renders during shutdown
        self._drain_work_queue()
        self.pending_projects.clear()
        for proc in self._worker_pool:
            if proc.pid in self._worker_queues:
                try:
                    self._worker_queues[proc.pid].put(None, timeout=2)
                except Exception:
                    pass
        for proc in self._worker_pool:
            try:
                proc.join(timeout=5)
                if proc.is_alive():
                    proc.terminate()
                    logger.warning(f"[WorkerPool] Force-terminated PID {proc.pid}")
            except Exception:
                pass
        self._worker_pool.clear()
        for wq in self._worker_queues.values():
            try:
                wq.close()
                wq.join_thread()
            except Exception:
                pass
        self._worker_queues.clear()


    def update_item_status(
        self,
        project_name: str,
        status: str,
        progress_text: str = "",
        project_path: Optional[str] = None,
        progress_pct: int = 0,
    ):
        """Update the status display for a single project in the library list."""
        if self._shutting_down:
            return
        try:
            for i in range(self.project_list.count()):
                item = self.project_list.item(i)
                path = item.data(Qt.ItemDataRole.UserRole)
                is_match = (
                    (path == project_path)
                    if project_path
                    else (os.path.basename(path) == project_name)
                )
                if is_match:
                    list_widget = self.project_list.itemWidget(item)
                    if list_widget and _widget_ok(list_widget):
                        list_widget.update_status(
                            status, progress_pct, progress_text
                        )
                    break
        except RuntimeError:
            pass

    def closeEvent(self, event):
        """Handle window close — prompt if renders are active, save editor tabs."""
        if self._shutting_down:
            event.accept()
            return
        # Check for active renders
        active_count = sum(
            1 for v in self.active_processes.values()
            if v.get("pool_dispatched", False)
            or (v.get("proc") is not None and v["proc"].is_alive())
        )
        if active_count > 0:
            reply = QMessageBox.question(
                self, "Confirm Exit",
                f"{active_count} render(s) are still active. Exit anyway?\n"
                "Active renders will be terminated.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        # Stop all timers FIRST to prevent post-close callbacks
        self._pool_timer.stop()
        self._refresh_timer.stop()
        self._thumb_debounce.stop()
        # Save and close all editor tabs
        for i in range(self.tabs.count() - 1, -1, -1):
            widget = self.tabs.widget(i)
            if isinstance(widget, SlideEditorTabWidget):
                try:
                    widget.save_and_cleanup()
                except Exception:
                    pass
        # Perform full shutdown cleanup
        self._shutting_down = True
        self._on_about_to_quit()
        event.accept()

    def _on_about_to_quit(self) -> None:
        """Backup cleanup called when the QApplication is about to quit.
        This handles the case where the event loop exits without
        ``closeEvent`` firing (e.g. on Windows when the window is
        destroyed externally).
        """
        if self._shutting_down:
            return
        self._shutting_down = True
        # Stop timers to prevent post-close callbacks
        self._pool_timer.stop()
        self._refresh_timer.stop()
        self._thumb_debounce.stop()
        # Save and close all editor tabs
        for i in range(self.tabs.count() - 1, -1, -1):
            widget = self.tabs.widget(i)
            if isinstance(widget, SlideEditorTabWidget):
                try:
                    widget.save_and_cleanup()
                except Exception:
                    pass
        # Stop the queue reader thread
        self._queue_reader.graceful_stop()
        # Terminate all active worker processes
        for pid, proc_data in list(self.active_processes.items()):
            proc = proc_data.get("proc")
            if proc is not None:
                try:
                    proc.terminate()
                except Exception:
                    pass
        # Close the shared message queue
        try:
            self.render_queue.close()
            self.render_queue.join_thread()
        except Exception:
            pass
        # Send poison pills to persistent workers and wait
        self._shutdown_worker_pool()
        # Close log handler
        if self._log_handler:
            self._log_handler.close()
        logger.info("[App] Shutdown cleanup complete")

    def update_stats(self):
        """Refresh the render statistics labels."""
        # Count pool-dispatched render jobs
        active_count = sum(
            1 for v in self.active_processes.values()
            if v.get("pool_dispatched", False)
        )
        # Also count thumbnail/other real processes
        active_count += sum(
            1 for v in self.active_processes.values()
            if not v.get("pool_dispatched", False) and v.get("proc") is not None
        )
        queue_count = len(self.pending_projects)
        pool_alive = sum(1 for p in self._worker_pool if p.is_alive())
        self.lbl_workers.setText(
            f"Max Parallel Jobs: {self.max_workers} "
            f"(pool: {pool_alive} alive)"
        )
        self.lbl_active_jobs.setText(f"Active: {active_count}")
        self.lbl_queue_count.setText(f"Queue: {queue_count}")
