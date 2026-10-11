# dialogs.py
from __future__ import annotations
import os
import sys
import re
import json 
import shutil
import logging
import html
import platform
import glob
import math
import threading
from datetime import datetime
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QFormLayout, QSpinBox, QDoubleSpinBox, QFrame,
    QApplication, QComboBox, QPushButton, QHBoxLayout, QVBoxLayout,
    QSplitter, QMessageBox, QSizePolicy, QTextEdit, QGroupBox,
    QCheckBox, QLabel, QLineEdit, QFileDialog, QTabWidget,
    QPlainTextEdit, QStackedWidget, QListWidget, QListWidgetItem, QScrollArea,
    QRadioButton
)
from PySide6.QtCore import Qt, QThread, QTimer, QSize, QUrl, Signal, QObject
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtGui import QShortcut, QKeySequence, QFont, QColor, QPalette

# Local imports
from utils import (
    create_slide_file, get_default_slide_html, get_next_slide_number,
    import_file_as_project, import_file_append, import_pdf_append, import_image_as_slide,
    get_blank_slide_html, initialize_project_files, is_image_slide,
    natural_sort_key, Palette, style_input, style_button, style_primary_button,
    style_page_widget, style_group_box, style_radio, style_checkbox,
    style_summary_label,  FILE_FILTER_IMPORT, FILE_FILTER_SCRIPT,
)

from config import AppConfig
try:
    import backends
except ImportError:
    backends = None

logger = logging.getLogger(__name__)

# =================================================================
# ILLEGAL FILENAME CHARACTERS (cross-platform)
# =================================================================

_ILLEGAL_NAME_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _ask_overwrite_or_append(parent: QWidget, project_name: str):
    """Show a dialog asking the user what to do when a project already exists.

    Returns:
        "overwrite" – delete existing project and re-create
        "append"   – add slides to the existing project
        None       – user cancelled
    """
    msg = QMessageBox(parent)
    msg.setIcon(QMessageBox.Icon.Warning)
    msg.setWindowTitle("Project Already Exists")
    msg.setText(f"Project '{project_name}' already exists.")
    msg.setInformativeText("What would you like to do?")
    btn_overwrite = msg.addButton("Overwrite", QMessageBox.ButtonRole.AcceptRole)
    btn_append = msg.addButton("Append", QMessageBox.ButtonRole.AcceptRole)
    btn_cancel = msg.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
    msg.setDefaultButton(btn_append)
    msg.exec()
    clicked = msg.clickedButton()
    if clicked == btn_overwrite:
        return "overwrite"
    elif clicked == btn_append:
        return "append"
    else:
        return None


def _sanitize_project_name(name: str) -> str:
    """Remove characters that are illegal in directory names on
    Windows, macOS, or Linux.  Also strips leading/trailing whitespace
    and dots (trailing dots are invalid on Windows).

    FIX: Returns a non-empty fallback name if the result would be empty
    (e.g. user entered only illegal characters).
    """
    name = name.strip()
    name = _ILLEGAL_NAME_RE.sub('_', name)
    name = name.rstrip('.')
    # Prevent empty name which would break project directory creation
    if not name:
        name = "Untitled_Project"
    return name


# =================================================================
# MODULAR SUB-COMPONENTS
# =================================================================

class StatusStrip(QFrame):
    """
    Left-edge colored strip indicating project status.
    Self-contained: accepts a color and renders itself.
    """
    STRIP_WIDTH = 5
    STRIP_RADIUS_TOP = "3px"
    STRIP_RADIUS_BOT = "3px"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(self.STRIP_WIDTH)
        self._color = "#555555"
        self._apply_style()

    def set_color(self, hex_color: str):
        self._color = hex_color
        self._apply_style()

    def _apply_style(self):
        self.setStyleSheet(
            f"background-color: {self._color}; "
            f"border-top-left-radius: {self.STRIP_RADIUS_TOP}; "
            f"border-bottom-left-radius: {self.STRIP_RADIUS_BOT};"
        )


class StatusBadge(QLabel):
    """
    Compact status indicator badge (e.g. "DONE", "WAIT", "ERR").
    Self-contained: accepts label text, bg, text, and border colors.
    """
    BADGE_W = 52
    BADGE_H = 22
    BADGE_RADIUS = 11
    BADGE_FONT_SIZE = 10
    BADGE_FONT_WEIGHT = "bold"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFixedSize(self.BADGE_W, self.BADGE_H)
        self._bg = "#444444"
        self._fg = "#999999"
        self._border = "#555555"
        self._apply_style()

    def set_appearance(self, text: str, bg: str, fg: str, border: str):
        self.setText(text)
        self._bg = bg
        self._fg = fg
        self._border = border
        self._apply_style()

    def _apply_style(self):
        self.setStyleSheet(f"""
            QLabel {{
                background-color: {self._bg};
                color: {self._fg};
                border-radius: {self.BADGE_RADIUS}px;
                font-size: {self.BADGE_FONT_SIZE}px;
                font-weight: {self.BADGE_FONT_WEIGHT};
                border: 1px solid {self._border};
            }}
        """)


class MetaLabel(QLabel):
    """
    Two-line content block: title (bold, larger) + detail (muted, smaller).
    Self-contained with its own layout.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWordWrap(True)
        self.setStyleSheet("background: transparent;")

    def set_title(self, text: str, color: str = "#e0e0e0",
                  size: int = 14, weight: str = "bold"):
        self.setText(text)
        self.setStyleSheet(
            f"color: {color}; font-size: {size}px; "
            f"font-weight: {weight}; background: transparent;"
        )


class DetailRow(QWidget):
    """
    Row below the title showing status detail and the badge.
    [detail_text ─────── stretch ─────── BADGE]
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.detail = QLabel()
        self.detail.setStyleSheet("background: transparent;")
        self.detail.setSizePolicy(QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Preferred)

        self.badge = StatusBadge()

        layout.addWidget(self.detail, stretch=1)
        layout.addWidget(self.badge)

    def set_detail(self, text: str, color: str = "#888888",
                   size: int = 11, weight: str = "normal"):
        self.detail.setText(text)
        self.detail.setStyleSheet(
            f"color: {color}; font-size: {size}px; "
            f"font-weight: {weight}; background: transparent;"
        )

    def set_badge(self, text: str, bg: str, fg: str, border: str):
        self.badge.set_appearance(text, bg, fg, border)


# =================================================================
# STATUS THEME RESOLVER (Centralized Status → Color Mapping)
# =================================================================

class StatusTheme:
    """
    Single source of truth for status → color mapping.
    All visual components query this instead of hardcoding colors.

    Keys use the **capitalized** form that matches RenderStatus enum values
    (e.g. "Rendering", "Ready", "Outdated", "Error", "Pending") so that
    update_status() lookups work without case conversion.

    IMPROVED: Sub-phase keys (e.g. "Generating Audio", "Rendering Video")
    provide finer visual feedback during the render pipeline.
    """

    # ── Major status states ─────────────────────────────────────────────

    # Strip colors
    STRIP = {
        "default":  "#555555",
        "New":      "#f0ad4e",     # brand-new project, never rendered
        "Outdated": "#f0ad4e",     # needs render (was Outdated/unknown)
        "Pending":  "#f0ad4e",     # queued
        "Rendering":"#0078d4",     # actively rendering (generic fallback)
        "Ready":    "#28a745",     # completed
        "Error":    "#dc3545",     # failed
        # ── Sub-phase strips (while status is "Rendering") ──
        "Scanning":          "#5c8ab5",
        "Audio Init":        "#5c7ab5",
        "Generating Audio":  "#4a90d9",
        "Audio Complete":    "#4a90d9",
        "Building Timeline": "#6a7ad9",
        "Video Starting":    "#7a6ad9",
        "Rendering Video":   "#0078d4",
        "Muxing Audio":      "#0078d4",
        "Finalizing":        "#4a9ad9",
        "Generating Thumbnail":"#4a9ad9",
        "Cache Hit":         "#6a9955",
    }

    # Badge colors: (bg, text, border)
    BADGE = {
        "default":  ("#444444", "#999999", "#555555"),
        "New":      ("#f0ad4e", "#1a1a1a", "#d99a3e"),
        "Outdated": ("#f0ad4e", "#1a1a1a", "#d99a3e"),
        "Pending":  ("#f0ad4e", "#1a1a1a", "#d99a3e"),
        "Rendering":("#0078d4", "#ffffff", "#006abc"),
        "Ready":    ("#28a745", "#ffffff", "#1e7e34"),
        "Error":    ("#dc3545", "#ffffff", "#bd2130"),
        # ── Sub-phase badges ──
        "Scanning":          ("#3a6a90", "#ffffff", "#2a5a80"),
        "Audio Init":        ("#3a6a90", "#ffffff", "#2a5a80"),
        "Generating Audio":  ("#3070b3", "#ffffff", "#205fa3"),
        "Audio Complete":    ("#3070b3", "#ffffff", "#205fa3"),
        "Building Timeline": ("#4a5ab3", "#ffffff", "#3a4aa3"),
        "Video Starting":    ("#5a4ab3", "#ffffff", "#4a3aa3"),
        "Rendering Video":   ("#0078d4", "#ffffff", "#006abc"),
        "Muxing Audio":      ("#0078d4", "#ffffff", "#006abc"),
        "Finalizing":        ("#3070b3", "#ffffff", "#205fa3"),
        "Generating Thumbnail":("#3070b3", "#ffffff", "#205fa3"),
        "Cache Hit":         ("#4a7a3a", "#ffffff", "#3a6a2a"),
    }

    # Badge labels
    LABEL = {
        "default":  "NEW",
        "New":      "NEW",
        "Outdated": "TODO",
        "Pending":  "WAIT",
        "Rendering":"...",
        "Ready":    "DONE",
        "Error":    "ERR",
        # ── Sub-phase badge labels ──
        "Scanning":          "SCAN",
        "Audio Init":        "TTS",
        "Generating Audio":  "TTS",
        "Audio Complete":    "TTS",
        "Building Timeline": "TL",
        "Video Starting":    "VID",
        "Rendering Video":   "VID",
        "Muxing Audio":      "MUX",
        "Finalizing":        "FIN",
        "Generating Thumbnail":"THM",
        "Cache Hit":         "SKIP",
    }

    # Detail text defaults
    DETAIL_COLOR = {
        "default":  "#888888",
        "New":      "#c8a460",
        "Outdated": "#c8a460",
        "Pending":  "#c8a460",
        "Rendering":"#66a3d2",
        "Ready":    "#6abf7b",
        "Error":    "#e87878",
        # ── Sub-phase detail colors ──
        "Scanning":          "#7aafcf",
        "Audio Init":        "#7a9fcf",
        "Generating Audio":  "#6aafdf",
        "Audio Complete":    "#6aafdf",
        "Building Timeline": "#8a9fdf",
        "Video Starting":    "#9a8fdf",
        "Rendering Video":   "#66a3d2",
        "Muxing Audio":      "#66a3d2",
        "Finalizing":        "#6aafdf",
        "Generating Thumbnail":"#6aafdf",
        "Cache Hit":         "#8abf8a",
    }

    @classmethod
    def strip_color(cls, status: str) -> str:
        return cls.STRIP.get(status, cls.STRIP["default"])

    @classmethod
    def badge_colors(cls, status: str) -> tuple:
        return cls.BADGE.get(status, cls.BADGE["default"])

    @classmethod
    def badge_label(cls, status: str) -> str:
        return cls.LABEL.get(status, cls.LABEL["default"])

    @classmethod
    def detail_color(cls, status: str) -> str:
        return cls.DETAIL_COLOR.get(status, cls.DETAIL_COLOR["default"])


# =================================================================
# SELECTION STATE (Decoupled from visual)
# =================================================================

class SelectionState:
    """
    Tracks which items are selected in the list.
    Supports multi-select via a set of paths.
    """
    def __init__(self):
        self._selected_paths: set[str] = set()

    @property
    def selected_path(self) -> str:
        """Return the first (or only) selected path for backward compat."""
        return next(iter(self._selected_paths), "") if self._selected_paths else ""

    @property
    def selected_paths(self) -> set[str]:
        return self._selected_paths

    def select(self, path: str):
        self._selected_paths.add(path)

    def deselect(self, path: str):
        self._selected_paths.discard(path)

    def deselect_all(self):
        self._selected_paths.clear()

    def is_selected(self, path: str) -> bool:
        return path in self._selected_paths


# =================================================================
# PROJECT LIST ITEM WIDGET (Modular Composition)
# =================================================================

class ProjectListItemWidget(QWidget):
    """
    A single project row in the library list.

    Architecture:
        ┌────┬──────────────────────────────┐
        │    │  Title                        │
        │ S  │  Detail text ─────── [BADGE] │
        │    │                               │
        └────┴──────────────────────────────┘
         Strip   Content (MetaLabel + DetailRow)

    Selection is self-managed via set_selected() — the widget
    updates its own background/border without external help.
    """

    ITEM_HEIGHT = 62
    ITEM_RADIUS = 6
    ITEM_MARGIN = "2px 4px 2px 4px"

    # Background states
    BG_NORMAL    = "#2d2d2d"
    BG_HOVER     = "#353535"
    BG_SELECTED  = "#1a4a7a"
    BG_SEL_HOVER = "#1f5a94"

    # Border states
    BORDER_NORMAL   = "1px solid #3a3a3a"
    BORDER_SELECTED = "2px solid #2196F3"

    # Title color states
    TITLE_NORMAL   = "#e0e0e0"
    TITLE_SELECTED = "#ffffff"

    def __init__(self, name: str, path: str,
                 selection_state: SelectionState, parent=None):
        super().__init__(parent)
        self.name = name
        self.path = path
        self._sel_state = selection_state
        self._hovered = False
        self._current_status = "unknown"
        self._progress_pct = 0
        self._detail_text = "Not Rendered"

        self.setObjectName("projectItem")
        self.setFixedHeight(self.ITEM_HEIGHT)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMouseTracking(True)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        self._build_ui()
        self._apply_visual_state()

    # ── Construction ─────────────────────────────────────────────

    def _build_ui(self):
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Strip
        self.strip = StatusStrip()
        outer.addWidget(self.strip)

        # Content
        content = QVBoxLayout()
        content.setContentsMargins(12, 8, 12, 8)
        content.setSpacing(4)

        self.title_label = MetaLabel()
        self.title_label.set_title(self.name)

        self.detail_row = DetailRow()
        self.detail_row.set_detail("Not Rendered")
        self.detail_row.set_badge("TODO", "#3a3a3a", "#888888", "#4a4a4a")

        content.addWidget(self.title_label)
        content.addWidget(self.detail_row)

        outer.addLayout(content, stretch=1)

    # ── Public API ───────────────────────────────────────────────

    def set_selected(self, selected: bool):
        """Set selection state and update visuals immediately."""
        if selected:
            self._sel_state.select(self.path)
        else:
            self._sel_state.deselect(self.path)
        self._apply_visual_state()

    def is_selected(self) -> bool:
        return self._sel_state.is_selected(self.path)

    def update_status(self, status: str, progress_pct: int = 0,
                      detail_text: str = "", slide_count: int = 0,
                      source: str = "", source_resolution: dict = None):
        """Update the project's rendering status and refresh all sub-components."""
        self._current_status = status
        self._progress_pct = progress_pct
        self._detail_text = detail_text or self._default_detail(status)

        # Enrich detail text with slide count, source, and resolution
        extra_parts = []
        if slide_count:
            extra_parts.append(f"{slide_count} slide{'s' if slide_count != 1 else ''}")
        if source:
            source_display = {
                "pdf": "PDF", "pptx": "PPTX", "json_speaker_script": "JSON",
                "json": "JSON", "txt": "TXT", "csv": "CSV", "image": "Image",
                "blank": "Blank",
            }.get(source, source.upper() if source else "")
            if source_display:
                extra_parts.append(f"from {source_display}")
        if source_resolution and isinstance(source_resolution, dict):
            w = source_resolution.get("width")
            h = source_resolution.get("height")
            aspect = source_resolution.get("aspect") or source_resolution.get("aspect_label", "")
            if w and h:
                res_str = f"{w}x{h}"
                if aspect:
                    res_str += f" ({aspect})"
                extra_parts.append(res_str)
        if extra_parts:
            self._detail_text = f"{self._detail_text}  ·  {'  '.join(extra_parts)}"

        # Strip
        self.strip.set_color(StatusTheme.strip_color(status))

        # Badge
        bg, fg, border = StatusTheme.badge_colors(status)
        label = self._badge_label(status, progress_pct)
        self.detail_row.set_badge(label, bg, fg, border)

        # Detail text
        det_color = StatusTheme.detail_color(status)
        self.detail_row.set_detail(self._detail_text, color=det_color)

    def current_status(self) -> str:
        return self._current_status

    # ── Visual State Engine ──────────────────────────────────────

    def _apply_visual_state(self):
        """Compute and apply background, border, and title color
        based on (selected, hovered) combination."""
        sel = self.is_selected()
        hov = self._hovered

        if sel and hov:
            bg = self.BG_SEL_HOVER
        elif sel:
            bg = self.BG_SELECTED
        elif hov:
            bg = self.BG_HOVER
        else:
            bg = self.BG_NORMAL

        border = self.BORDER_SELECTED if sel else self.BORDER_NORMAL
        title_color = self.TITLE_SELECTED if sel else self.TITLE_NORMAL

        self.setStyleSheet(f"""
            #projectItem {{
                background-color: {bg};
                border: {border};
                border-radius: {self.ITEM_RADIUS}px;
                margin: {self.ITEM_MARGIN};
            }}
        """)

        self.title_label.set_title(self.name, color=title_color)

    # ── Mouse Events ─────────────────────────────────────────────

    def enterEvent(self, event):
        self._hovered = True
        self._apply_visual_state()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hovered = False
        self._apply_visual_state()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        """Click self-selects and also notifies the QListWidget.

        - Plain click: clear all, select this item.
        - Ctrl+Click: toggle this item's selection (add/remove).
        - Shift+Click: range-select from anchor to this item.
        - Ctrl+Shift+Click: add range from anchor to this item.

        The event is accepted so it does NOT propagate to the
        QListWidget's viewport, which would re-handle the click and
        fight with our custom selection logic.
        """
        if event.button() == Qt.MouseButton.LeftButton:
            modifiers = event.modifiers()
            ctrl = bool(modifiers & Qt.KeyboardModifier.ControlModifier)
            shift = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
            list_widget = self._find_parent_list()
            if list_widget:
                # Find this widget's index in the list
                my_index = -1
                my_item = None
                for i in range(list_widget.count()):
                    item = list_widget.item(i)
                    w = list_widget.itemWidget(item)
                    if w is self:
                        my_index = i
                        my_item = item
                        break

                if my_index >= 0 and my_item is not None:
                    mgr = self._get_selection_manager(list_widget)

                    # Block signals so itemSelectionChanged → refresh()
                    # does not fire during our manual selection updates
                    list_widget.blockSignals(True)

                    if shift and not ctrl:
                        # ── Shift+Click: range select ──
                        anchor = mgr.anchor_index if mgr else -1
                        if anchor < 0:
                            anchor = my_index
                        lo = min(anchor, my_index)
                        hi = max(anchor, my_index)
                        # Clear all first
                        self._clear_all_selection(list_widget)
                        # Select the range
                        for i in range(lo, hi + 1):
                            item = list_widget.item(i)
                            if item:
                                item.setSelected(True)
                                w = list_widget.itemWidget(item)
                                if isinstance(w, ProjectListItemWidget):
                                    w._sel_state.select(w.path)
                        list_widget.setCurrentItem(my_item)
                        if mgr:
                            mgr.anchor_index = anchor  # keep original anchor
                    elif ctrl and shift:
                        # ── Ctrl+Shift+Click: add range to existing selection ──
                        anchor = mgr.anchor_index if mgr else -1
                        if anchor < 0:
                            anchor = my_index
                        lo = min(anchor, my_index)
                        hi = max(anchor, my_index)
                        for i in range(lo, hi + 1):
                            item = list_widget.item(i)
                            if item:
                                item.setSelected(True)
                                w = list_widget.itemWidget(item)
                                if isinstance(w, ProjectListItemWidget):
                                    w._sel_state.select(w.path)
                        list_widget.setCurrentItem(my_item)
                        if mgr:
                            mgr.anchor_index = anchor
                    elif ctrl:
                        # ── Ctrl+Click: toggle this item ──
                        was_selected = my_item.isSelected()
                        my_item.setSelected(not was_selected)
                        if was_selected:
                            self._sel_state.deselect(self.path)
                        else:
                            self._sel_state.select(self.path)
                        list_widget.setCurrentItem(my_item)
                        if mgr:
                            mgr.anchor_index = my_index
                    else:
                        # ── Plain click: clear all, select this one ──
                        self._clear_all_selection(list_widget)
                        my_item.setSelected(True)
                        list_widget.setCurrentItem(my_item)
                        self._sel_state.select(self.path)
                        if mgr:
                            mgr.anchor_index = my_index

                    list_widget.blockSignals(False)

                    # Update visuals on all widgets (since signals were blocked)
                    self._refresh_all_visuals(list_widget)

        event.accept()  # Prevent propagation to QListWidget viewport

    def _get_selection_manager(self, list_widget):
        """Get the ProjectSelectionManager from the parent MainWindow if available."""
        # Walk up to find the MainWindow or any widget that holds selection_mgr
        p = self.parent()
        while p is not None:
            if hasattr(p, 'selection_mgr'):
                return p.selection_mgr
            p = p.parent()
        return None

    @staticmethod
    def _clear_all_selection(list_widget):
        """Clear selection on all items in the list."""
        for i in range(list_widget.count()):
            item = list_widget.item(i)
            item.setSelected(False)
            w = list_widget.itemWidget(item)
            if isinstance(w, ProjectListItemWidget):
                w._sel_state.deselect(w.path)

    @staticmethod
    def _refresh_all_visuals(list_widget):
        """Refresh visual state on all item widgets."""
        for i in range(list_widget.count()):
            item = list_widget.item(i)
            w = list_widget.itemWidget(item)
            if isinstance(w, ProjectListItemWidget):
                w._apply_visual_state()

    def _find_parent_list(self) -> QListWidget | None:
        """Walk up the widget hierarchy to find the parent QListWidget."""
        p = self.parent()
        while p is not None:
            if isinstance(p, QListWidget):
                return p
            p = p.parent()
        return None

    # ── Helpers ──────────────────────────────────────────────────

    @staticmethod
    def _default_detail(status: str) -> str:
        defaults = {
            "New": "New Project",
            "Outdated": "Needs Render",
            "Pending": "Queued",
            "Rendering": "Processing...",
            "Ready": "Completed",
            "Error": "Failed — Check Logs",
            # ── Sub-phase defaults ──
            "Scanning":          "Scanning project files...",
            "Audio Init":        "Initializing audio backend...",
            "Generating Audio":  "Generating TTS audio...",
            "Audio Complete":    "Audio generation complete",
            "Building Timeline": "Assembling timeline...",
            "Video Starting":    "Starting video encoder...",
            "Rendering Video":   "Rendering video frames...",
            "Muxing Audio":      "Muxing audio & video...",
            "Finalizing":        "Finalizing output...",
            "Generating Thumbnail":"Generating thumbnail...",
            "Cache Hit":         "Video up-to-date, skipped render",
        }
        return defaults.get(status, "Not Rendered")

    @staticmethod
    def _badge_label(status: str, pct: int) -> str:
        # Show percentage for any rendering sub-phase when pct > 0
        if pct > 0 and status in (
            "Rendering", "Scanning", "Audio Init", "Generating Audio",
            "Audio Complete", "Building Timeline", "Video Starting",
            "Rendering Video", "Muxing Audio", "Finalizing",
            "Generating Thumbnail", "Cache Hit",
        ):
            return f"{pct}%"
        return StatusTheme.badge_label(status)


# =================================================================
# SELECTION MANAGER (Decoupled from MainWindow)
# =================================================================

class ProjectSelectionManager:
    """
    Manages selection across all ProjectListItemWidget instances
    in a QListWidget. Replaces the manual loops in MainWindow.

    Usage:
        manager = ProjectSelectionManager(list_widget)
        manager.refresh()          # sync from QListWidget selection
        manager.select_path(path)  # programmatic selection
    """

    def __init__(self, list_widget: QListWidget):
        self._list = list_widget
        self._state = SelectionState()
        self._anchor_index = -1  # for Shift+Click range selection

    @property
    def state(self) -> SelectionState:
        return self._state

    @property
    def anchor_index(self) -> int:
        return self._anchor_index

    @anchor_index.setter
    def anchor_index(self, value: int):
        self._anchor_index = value

    def refresh(self):
        """Sync selection state from the QListWidget's current selection."""
        self._state.deselect_all()
        for item in self._list.selectedItems():
            path = item.data(Qt.ItemDataRole.UserRole)
            if path:
                self._state.select(path)
        self._update_all_widgets()

    def select_path(self, path: str):
        """Programmatically select the item with the given path (single)."""
        self._state.deselect_all()
        self._state.select(path)
        # Clear Qt selection, then select only the target
        for i in range(self._list.count()):
            item = self._list.item(i)
            item.setSelected(False)
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == path:
                item.setSelected(True)
                self._list.setCurrentItem(item)
                break
        self._update_all_widgets()

    def selected_path(self) -> str:
        return self._state.selected_path

    def _update_all_widgets(self):
        """Push current selection state to every item widget."""
        for i in range(self._list.count()):
            item = self._list.item(i)
            widget = self._list.itemWidget(item)
            if isinstance(widget, ProjectListItemWidget):
                widget._apply_visual_state()


# =================================================================
# SETTINGS TAB WIDGET
# =================================================================


class _HardwareDetectThread(QThread):
    """One-shot QThread that probes CUDA devices and emits the result
    via the built-in ``finished`` signal (result stored on ``self``)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.result: tuple = ("error", "thread not run")
        self._stop_event = threading.Event()

    def run(self):
        if self._stop_event.is_set():
            return
        try:
            import torch
            if self._stop_event.is_set():
                return
            torch.cuda.init()
            if torch.cuda.is_available():
                count = torch.cuda.device_count()
                devices = []
                for i in range(count):
                    name = torch.cuda.get_device_name(i)
                    devices.append((f"cuda:{i} ({name})", f"cuda:{i}"))
                self.result = ("found", devices)
            else:
                self.result = ("none", [])
        except ImportError:
            self.result = ("no_torch", [])
        except Exception as exc:
            self.result = ("error", str(exc))


class _ResolutionVerifyThread(QThread):
    """QThread that runs resolution verification off the GUI thread.
    
    Emits ``result_ready(dict)`` with the verification result.
    """
    result_ready = Signal(object)

    def __init__(self, project_path: str, parent=None):
        super().__init__(parent)
        self.project_path = project_path

    def run(self):
        try:
            from utils import verify_project_resolution
            result = verify_project_resolution(self.project_path)
        except Exception as exc:
            result = {
                "resolution": None,
                "method": None,
                "details": f"Error: {exc}",
            }
        self.result_ready.emit(result)


class _BatchResolutionVerifyThread(QThread):
    """QThread that runs resolution verification on ALL projects.

    Emits ``project_done(str, dict)`` for each completed project
    (project_name, result_dict) and ``all_done(int, int)`` when finished
    (total_count, success_count).
    """
    project_done = Signal(str, object)  # project_name, result
    all_done = Signal(int, int)          # total, successes

    def __init__(self, project_dirs: list, parent=None):
        super().__init__(parent)
        self.project_dirs = project_dirs

    def run(self):
        from utils import verify_project_resolution
        total = len(self.project_dirs)
        successes = 0
        for project_path in self.project_dirs:
            project_name = os.path.basename(project_path)
            try:
                result = verify_project_resolution(project_path)
            except Exception as exc:
                result = {
                    "resolution": None,
                    "method": None,
                    "details": f"Error: {exc}",
                }
            if result.get("resolution") is not None:
                successes += 1
            self.project_done.emit(project_name, result)
        self.all_done.emit(total, successes)


class SettingsTabWidget(QWidget):
    """Widget to be embedded in a QTabWidget for settings."""

    log_signal = Signal(str)
    settings_changed = Signal(str, object)  # (key, value)
    library_refresh_requested = Signal()     # emitted after batch verify

    AUTO_SAVE_DELAY_MS = 600


    def __init__(self, config, parent=None):
        super().__init__(parent)

        self.config = config 


        # Debounce timer for settings that change rapidly
        self._settings_timer = QTimer(self)
        self._settings_timer.setSingleShot(True)
        self._settings_timer.setInterval(self.AUTO_SAVE_DELAY_MS)
        self._settings_timer.timeout.connect(self._flush_pending_settings)
        self._pending_settings: dict = {}

        self.backend_descriptions = (
            backends.get_backend_descriptions()
            if backends else {"default": "(backends module not available)"}
        )

        # Main Layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Search Bar
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Search settings...")
        self.search_bar.setStyleSheet(
            "padding: 5px; background: #333; color: white; "
            "border-radius: 0px; border-bottom: 1px solid #444;"
        )
        self.search_bar.textChanged.connect(self.filter_sidebar)
        main_layout.addWidget(self.search_bar)

        # Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # --- LEFT: SIDEBAR ---
        self.sidebar = QListWidget()
        self.sidebar.setFixedWidth(220)
        self.sidebar.setStyleSheet("""
            QListWidget { background-color: #252526; border: none; font-size: 14px; }
            QListWidget::item { padding: 12px; color: #cccccc; border-bottom: 1px solid #333; }
            QListWidget::item:selected { background-color: #007acc; color: white; }
            QListWidget::item:hover { background-color: #2d2d30; }
        """)

        self.sidebar.addItem("General & Video")
        self.sidebar.addItem("Backend Settings")
        self.sidebar.addItem("Hardware & Performance")
        self.sidebar.addItem("Text Chunking")
        self.sidebar.addItem("Paths & Cache")
        self.sidebar.addItem("Diagnostics")

        self.sidebar.setCurrentRow(0)
        self.sidebar.currentRowChanged.connect(self.change_page)
        splitter.addWidget(self.sidebar)

        # --- RIGHT: CONTENT STACK ---
        self.stack = QStackedWidget()
        splitter.addWidget(self.stack)
        splitter.setStretchFactor(1, 4)

        # Build Pages
        self.page_general = self._create_general_page()
        self.page_backend_settings = self._create_backend_settings_page()
        self.page_hardware = self._create_hardware_page()
        self.page_chunking = self._create_chunking_page()
        self.page_paths = self._create_paths_page()
        self.page_diagnostics = self._create_diagnostics_page()

        self.stack.addWidget(self.page_general)
        self.stack.addWidget(self.page_backend_settings)
        self.stack.addWidget(self.page_hardware)
        self.stack.addWidget(self.page_chunking)
        self.stack.addWidget(self.page_paths)
        self.stack.addWidget(self.page_diagnostics)

        # Ctrl+S shortcut for settings
        self._save_shortcut = QShortcut(QKeySequence("Ctrl+S"), self)
        self._save_shortcut.activated.connect(self._flush_pending_settings)

        # Initialize — block signals so programmatic setValue/setText
        # calls don't cascade through _debounce_set during init.
        self.blockSignals(True)
        try:
            self.refresh_ui_values()
            self.refresh_backend_settings_ui()
        finally:
            self.blockSignals(False)

    # ==========================================================
    # DEBOUNCED SETTINGS HELPER
    # ==========================================================
    def _debounce_set(self, key, value):
        """
        Queue a settings change.  Skips work entirely when the new
        value is identical to the current in-memory value, preventing
        spurious writes during refresh_ui_values() or widget init.
        """
        current = self.config.settings.get(key)
        if current == value and key not in self._pending_settings:
            return  # nothing to do

        self._pending_settings[key] = value
        # Apply to in-memory settings immediately so reads are consistent
        self.config.settings[key] = value
        self._settings_timer.start()  # restart the debounce window
        self.settings_changed.emit(key, value)
        
        if key in ('width', 'height'):
            self._push_resolution_to_editors()

    def _flush_pending_settings(self):
        """Write all pending settings to config (and disk).

        NOTE: _debounce_set pre-sets values in-memory for consistency,
        so AppConfig.set() sees no change and skips its own save.
        We must force a disk write here.
        """
        if not self._pending_settings:
            return
        for key, value in self._pending_settings.items():
            self.config.settings[key] = value
        self._pending_settings.clear()
        self.config.save()
        logger.debug("[Settings] Flushed pending settings to disk.")

    # ==========================================================
    # RESOLUTION HELPERS
    # ==========================================================

    def _push_resolution_to_editors(self):
        """Find all open SlideEditorTabWidget instances and update their
        preview panels with the new resolution from settings."""
        try:
            # Walk up to find the MainWindow
            parent = self.parent()
            main_window = None
            while parent:
                if hasattr(parent, 'tabs'):  # MainWindow has self.tabs
                    main_window = parent
                    break
                parent = parent.parent()
            
            if not main_window:
                return
            
            # Get current resolution from config
            width = self.config.get('width', 1280)
            height = self.config.get('height', 720)
            
            # Find all open editor tabs
            for i in range(main_window.tabs.count()):
                widget = main_window.tabs.widget(i)
                if isinstance(widget, SlideEditorTabWidget):
                    # Update the preview panel inside the editor
                    if hasattr(widget, 'preview_panel') and hasattr(widget.preview_panel, 'update_preview_resolution'):
                        widget.preview_panel.update_preview_resolution(width, height)
        except Exception as e:
            logger.error(f"Error pushing resolution to editors ({type(e).__name__}: {e})", exc_info=True)

    def _on_generate_thumbnail_clicked(self):
        """Emit signal so MainWindow can launch the standalone thumbnail worker."""
        self.settings_changed.emit("__generate_thumbnail__", True)

    # _apply_resolution_to_config defined below (line ~1381) with
    # the additional _update_aspect_preview() call.

    def _update_resolution_preview(self):
        """Update the small label showing the actual pixel dimensions."""
        data = self.resolution_combo.currentData()
        orientation = self.orientation_combo.currentData()
        if data:
            base_w, base_h = data
            if orientation == "portrait":
                # Portrait presets store actual portrait (w, h)
                if base_w > base_h:
                    w, h = base_h, base_w
                else:
                    w, h = base_w, base_h
            else:
                if base_h > base_w:
                    w, h = base_h, base_w
                else:
                    w, h = base_w, base_h
            self.resolution_preview.setText(f"Output: {w}x{h} pixels")
        else:
            self.resolution_preview.setText("")

    def _sync_resolution_combos(self):
        """Set resolution and orientation combos from current config width/height."""
        w = self.config.get('width', 1280)
        h = self.config.get('height', 720)

        # Determine orientation
        if h > w:
            orientation = "portrait"
            base_w, base_h = h, w  # swap to landscape reference
        else:
            orientation = "landscape"
            base_w, base_h = w, h

        # Repopulate combo with correct preset list for orientation
        self._populate_resolution_combo(orientation)

        # Find matching preset
        matched = False
        for i in range(self.resolution_combo.count()):
            data = self.resolution_combo.itemData(i)
            if data and data[0] == w and data[1] == h:
                self.resolution_combo.blockSignals(True)
                self.resolution_combo.setCurrentIndex(i)
                self.resolution_combo.blockSignals(False)
                matched = True
                break

        # Fallback: try matching the landscape base to a landscape preset
        if not matched and orientation == "landscape":
            for i in range(self.resolution_combo.count()):
                data = self.resolution_combo.itemData(i)
                if data and data[0] == base_w and data[1] == base_h:
                    self.resolution_combo.blockSignals(True)
                    self.resolution_combo.setCurrentIndex(i)
                    self.resolution_combo.blockSignals(False)
                    matched = True
                    break

        # Set orientation
        orient_idx = self.orientation_combo.findData(orientation)
        if orient_idx >= 0:
            self.orientation_combo.blockSignals(True)
            self.orientation_combo.setCurrentIndex(orient_idx)
            self.orientation_combo.blockSignals(False)

        # Update preview label
        self._update_resolution_preview()

    # ==========================================================
    # NAVIGATION
    # ==========================================================
    def change_page(self, index):
        self.stack.setCurrentIndex(index)

    def filter_sidebar(self, text):
        search = text.lower()
        for i in range(self.sidebar.count()):
            item = self.sidebar.item(i)
            item.setHidden(search not in item.text().lower())

    def refresh_ui_values(self):
        """Reloads UI from config. Blocks signals to prevent cascading saves."""
        if not hasattr(self, 'backend_combo'):
            return
        # Block signals while programmatically updating widgets
        self.resolution_combo.blockSignals(True)
        self.orientation_combo.blockSignals(True)
        self.fps_spin.blockSignals(True)
        self.enc_combo.blockSignals(True)
        self.worker_spin.blockSignals(True)
        self.trans_spin.blockSignals(True)
        self.chk_verbose.blockSignals(True)
        
        try:
            # General — Resolution
            self._sync_resolution_combos()
            self.fps_spin.setValue(self.config.get('fps', 30))
            self.enc_combo.setCurrentText(self.config.get('encoder', 'auto'))
            self.worker_spin.setValue(self.config.get('render_workers', 3))
            self.out_dir_edit.setText(self.config.get('output_dir', 'Output'))
            self.proj_root_edit.setText(self.config.get('projects_root', 'Projects'))
            self.trans_spin.setValue(self.config.get('transition_duration', 0.5))

            # Backend Settings
            current_backend = self.config.get('active_backend', 'qwen3')
            index = self.backend_combo.findData(current_backend)
            if index >= 0:
                self.backend_combo.setCurrentIndex(index)

            # Hardware
            self.qwen3_device_combo.setCurrentText(
                self.config.get('qwen3_device_map', 'cuda:0'))
            self.qwen3_dtype_combo.setCurrentText(
                self.config.get('qwen3_dtype', 'float16'))
            self.chk_flash_attn.setChecked(
                self.config.get('qwen3_attn_implementation') == "flash_attention_2")
            self.qwen3_size_combo.setCurrentText(
                self.config.get('qwen3_size', '1.7B'))

            # Text Chunking
            self.chk_enable_chunking.setChecked(
                self.config.get('enable_text_chunking', True))
            self.spin_chunk_max_chars.setValue(
                self.config.get('chunk_max_chars', 500))
            self.spin_chunk_max_sentences.setValue(
                self.config.get('chunk_max_sentences', 5))
            self.spin_chunk_min_chars.setValue(
                self.config.get('chunk_min_chars', 50))
            self.spin_warn_threshold.setValue(
                self.config.get('chunk_warn_threshold', 1000))

            # Diagnostics
            self.chk_verbose.setChecked(
                self.config.get('verbose', False))
        finally:
            # Always unblock signals
            self.resolution_combo.blockSignals(False)
            self.orientation_combo.blockSignals(False)
            self.fps_spin.blockSignals(False)
            self.enc_combo.blockSignals(False)
            self.worker_spin.blockSignals(False)
            self.trans_spin.blockSignals(False)
            self.chk_verbose.blockSignals(False)
            pass  # thumbnail resolution auto-detected — no combo to unblock

    # ==========================================================
    # PAGE: GENERAL & VIDEO
    # ==========================================================
    def _create_general_page(self):
        page = QWidget()
        layout = QFormLayout(page)

        # Create preview label BEFORE populating combos so that
        # currentIndexChanged signals fired during addItem can safely
        # reference self.resolution_preview.
        self.resolution_preview = QLabel("")
        self.resolution_preview.setStyleSheet(
            "color: #888888; font-size: 11px; padding-left: 2px;")

        res_layout = QHBoxLayout()
        self.resolution_combo = QComboBox()
        self.resolution_combo.blockSignals(True)
        self._populate_resolution_combo("landscape")
        self.resolution_combo.blockSignals(False)

        self.orientation_combo = QComboBox()
        self.orientation_combo.blockSignals(True)
        self.orientation_combo.addItem("Landscape (Horizontal)", "landscape")
        self.orientation_combo.addItem("Portrait (Vertical)", "portrait")
        self.orientation_combo.blockSignals(False)

        self.resolution_combo.currentIndexChanged.connect(
            self._apply_resolution_to_config)
        self.orientation_combo.currentIndexChanged.connect(
            self._on_orientation_changed)

        res_layout.addWidget(self.resolution_combo, stretch=3)
        res_layout.addWidget(self.orientation_combo, stretch=2)
        layout.addRow("Resolution:", res_layout)
        layout.addRow("", self.resolution_preview)

        # ── Visual Aspect Ratio Preview Widget ──
        self.aspect_preview_widget = QWidget()
        self.aspect_preview_widget.setFixedHeight(60)
        self.aspect_preview_widget.setStyleSheet("background: transparent;")
        
        aspect_layout = QHBoxLayout(self.aspect_preview_widget)
        aspect_layout.setContentsMargins(0, 0, 0, 0)
        
        self.aspect_box = QFrame()
        self.aspect_box.setStyleSheet(
            "background-color: #0078d4; border-radius: 4px;"
        )
        self.aspect_label = QLabel("16:9")
        self.aspect_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.aspect_label.setStyleSheet(
            "color: white; font-weight: bold; font-size: 12px; background: transparent;"
        )
        
        aspect_layout.addStretch()
        aspect_layout.addWidget(self.aspect_box, stretch=0)  # Box scales to represent ratio
        aspect_layout.addWidget(self.aspect_label, stretch=0)
        aspect_layout.addStretch()
        
        layout.addRow("Aspect Ratio:", self.aspect_preview_widget)
        
        # Initial update
        self._update_aspect_preview()

        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(10, 60)
        self.fps_spin.valueChanged.connect(
            lambda v: self._debounce_set('fps', v))
        layout.addRow("FPS:", self.fps_spin)

        self.enc_combo = QComboBox()
        self.enc_combo.addItems(["auto", "libx264", "h264_nvenc"])
        self.enc_combo.currentIndexChanged.connect(
            lambda i: self._debounce_set('encoder', self.enc_combo.itemText(i))
        )
        layout.addRow("Encoder:", self.enc_combo)

        self.worker_spin = QSpinBox()
        self.worker_spin.setRange(1, 8)
        self.worker_spin.valueChanged.connect(
            lambda v: self._debounce_set('render_workers', v))
        layout.addRow("Render Workers:", self.worker_spin)

        # Output Directory
        out_layout = QHBoxLayout()
        self.out_dir_edit = QLineEdit()
        self.out_dir_edit.setText(self.config.get('output_dir', 'Output'))
        self.out_dir_edit.editingFinished.connect(
            lambda: self._debounce_set('output_dir', self.out_dir_edit.text())
        )
        btn_browse = QPushButton("Browse...")
        btn_browse.clicked.connect(self.browse_output_dir)
        out_layout.addWidget(self.out_dir_edit)
        out_layout.addWidget(btn_browse)
        layout.addRow("Output Dir:", out_layout)

        # Projects Root — now passes config_key for reliable persistence
        proj_layout = QHBoxLayout()
        self.proj_root_edit = QLineEdit()
        self.proj_root_edit.setText(self.config.get('projects_root', 'Projects'))
        self.proj_root_edit.editingFinished.connect(
            lambda: self._debounce_set('projects_root',
                                       self.proj_root_edit.text())
        )
        btn_browse_proj = QPushButton("Browse...")
        btn_browse_proj.clicked.connect(
            lambda: self.browse_generic(
                self.proj_root_edit, True, config_key='projects_root')
        )
        proj_layout.addWidget(self.proj_root_edit)
        proj_layout.addWidget(btn_browse_proj)
        layout.addRow("Projects Root:", proj_layout)

        self.trans_spin = QDoubleSpinBox()
        self.trans_spin.setRange(0.0, 10.0)
        self.trans_spin.setSingleStep(0.1)
        self.trans_spin.setSuffix(" s")
        self.trans_spin.valueChanged.connect(
            lambda v: self._debounce_set('transition_duration', v))
        layout.addRow("Transition Duration:", self.trans_spin)

        # ── Thumbnail Settings ──────────────────────────────────────────
        thumb_group = QGroupBox("YouTube Thumbnail")
        thumb_group.setStyleSheet(style_group_box())
        thumb_layout = QFormLayout(thumb_group)

        self.btn_generate_thumbnail = QPushButton("Generate Thumbnail for Selected Project")
        self.btn_generate_thumbnail.setStyleSheet(style_primary_button())
        self.btn_generate_thumbnail.setToolTip(
            "Generate a PNG thumbnail from the first slide of the selected "
            "project. This runs independently — no video re-render needed."
        )
        self.btn_generate_thumbnail.clicked.connect(
            self._on_generate_thumbnail_clicked
        )
        thumb_layout.addRow(self.btn_generate_thumbnail)

        thumb_info = QLabel(
            "Resolution is auto-detected from the slide's render dimensions.\n"
            "The full slide is captured without any cropping."
        )
        thumb_info.setStyleSheet("color: #888888; font-size: 11px;")
        thumb_info.setWordWrap(True)
        thumb_layout.addRow(thumb_info)

        layout.addRow(thumb_group)

        return page

    # ==========================================================
    # PAGE: BACKEND SETTINGS
    # ==========================================================
    def _create_backend_settings_page(self):
        page = QWidget()
        main_layout = QVBoxLayout(page)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # 1. Header & Selector
        header_layout = QVBoxLayout()
        title = QLabel("<h2>Backend Configuration</h2>")
        header_layout.addWidget(title)

        selector_layout = QHBoxLayout()
        selector_layout.addWidget(QLabel("<b>Active Engine:</b>"))

        if backends is None:
            lbl = QLabel("Backend module is not available.")
            main_layout.addWidget(lbl)  # Use existing layout, not a new one
            return page
        from backends import get_backend_info
        
        backend_info = get_backend_info()
        current_backend = self.config.get("active_backend", "edge")
        
        # 1. Create the combo box
        self.backend_combo = QComboBox()
        
        # 2. Populate it dynamically
        for backend_name, info in backend_info.items():
            # Make the name look nice in the UI (e.g., "edge" -> "Edge")
            display_name = backend_name.replace("_", " ").title()
            self.backend_combo.addItem(display_name, userData=backend_name)
            
            # Grey out backends whose dependencies aren't installed
            if not info["available"]:
                item = self.backend_combo.model().item(self.backend_combo.count() - 1)
                item.setEnabled(False)
                item.setToolTip("Install required dependencies to enable this backend")
                
        # 3. Fallback logic: if the configured backend isn't available, 
        # select the first available one
        if not backend_info.get(current_backend, {}).get("available", False):
            for backend_name, info in backend_info.items():
                if info["available"]:
                    current_backend = backend_name
                    break

        # 4. Set the active selection in the UI
        idx = self.backend_combo.findData(current_backend)
        if idx >= 0:
            self.backend_combo.setCurrentIndex(idx)

        # 5. Connect signal and add to layout
        self.backend_combo.currentIndexChanged.connect(
            self.on_backend_selection_changed
        )
        selector_layout.addWidget(self.backend_combo, 1)

        # Mode Selector
        self.mode_label = QLabel("Mode:")
        self.mode_combo = QComboBox()
        self.mode_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)
        selector_layout.addWidget(self.mode_label)
        selector_layout.addWidget(self.mode_combo)

        selector_layout.addStretch()
        header_layout.addLayout(selector_layout)

        # Description Area
        self.backend_desc_label = QLabel("Select a backend to view details.")
        self.backend_desc_label.setWordWrap(True)
        header_layout.addWidget(self.backend_desc_label)

        main_layout.addLayout(header_layout)

        # 2. Scroll Area for Content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(scroll_content)
        self.scroll_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Group: Backend Parameters
        self.backend_params_group = QGroupBox("Backend Specific Parameters")
        self.backend_params_layout = QVBoxLayout()
        self.backend_params_group.setLayout(self.backend_params_layout)
        self.scroll_layout.addWidget(self.backend_params_group)

        # Group: Diagnostics & Testing
        diag_group = QGroupBox("Diagnostics & Testing")
        diag_layout = QVBoxLayout()
        diag_layout.addWidget(
            QLabel("Run backend diagnostics from the Diagnostics sidebar page."))
        diag_group.setLayout(diag_layout)
        self.scroll_layout.addWidget(diag_group)

        self.scroll_layout.addStretch()

        scroll.setWidget(scroll_content)
        main_layout.addWidget(scroll)

        return page

    def on_backend_selection_changed(self, index):
        backend_name = self.backend_combo.currentData()
        if backend_name is None:
            return
        self._debounce_set('active_backend', backend_name)
        self.refresh_backend_settings_ui()

    def on_mode_changed(self, index):
        backend_name = self.config.get('active_backend', 'qwen3')
        mode_key = self.mode_combo.currentData()
        self._debounce_set(f"{backend_name}_mode", mode_key)
        self.load_backend_widget()

    def refresh_backend_settings_ui(self):
        backend_name = self.config.get('active_backend', 'qwen3')

        desc = self.backend_descriptions.get(
            backend_name, self.backend_descriptions.get("default"))
        self.backend_desc_label.setText(
            f"<b>{backend_name.upper()}</b>: {desc}")

        backend_class = backends.BACKEND_MAP.get(backend_name) if backends else None
        has_modes = False

        if backend_class and hasattr(backend_class, 'get_ui_options'):
            options = backend_class.get_ui_options()
            modes = options.get("modes", [])

            if modes:
                has_modes = True
                self.mode_combo.blockSignals(True)
                self.mode_combo.clear()
                for key, display_name in modes:
                    self.mode_combo.addItem(display_name, key)

                current_mode = self.config.get(f"{backend_name}_mode")
                idx = self.mode_combo.findData(current_mode)
                if idx >= 0:
                    self.mode_combo.setCurrentIndex(idx)
                self.mode_combo.blockSignals(False)

        self.mode_label.setVisible(has_modes)
        self.mode_combo.setVisible(has_modes)

        self.load_backend_widget()

    def _update_aspect_preview(self):
        """Update the visual aspect ratio box and label."""
        data = self.resolution_combo.currentData()
        orientation = self.orientation_combo.currentData()
        
        if not data:
            return
            
        base_w, base_h = data
        if orientation == "portrait":
            # Portrait presets store actual portrait (w, h)
            if base_w > base_h:
                w, h = base_h, base_w
            else:
                w, h = base_w, base_h
        else:
            if base_h > base_w:
                w, h = base_h, base_w
            else:
                w, h = base_w, base_h
        
        # Calculate greatest common divisor for the ratio label (e.g., 16:9)
        gcd = math.gcd(w, h)
        ratio_w = w // gcd
        ratio_h = h // gcd
        
        self.aspect_label.setText(f"{ratio_w}:{ratio_h}")
        
        # Scale the box to fit the preview area (max 80px wide, 45px tall)
        max_w_px = 80
        max_h_px = 45
        
        scale_w = max_w_px / w if w > 0 else 1
        scale_h = max_h_px / h if h > 0 else 1
        scale = min(scale_w, scale_h)
        
        box_w = int(w * scale)
        box_h = int(h * scale)
        
        self.aspect_box.setFixedSize(box_w, box_h)

    def _populate_resolution_combo(self, orientation: str):
        """Populate the resolution combo with the appropriate preset list."""
        self.resolution_combo.blockSignals(True)
        self.resolution_combo.clear()
        source = (AppConfig.PORTRAIT_PRESETS if orientation == "portrait"
                  else AppConfig.RESOLUTION_PRESETS)
        for name, (w, h) in source.items():
            self.resolution_combo.addItem(name, (w, h))
        self.resolution_combo.blockSignals(False)

    def _on_orientation_changed(self, index):
        """When the user switches orientation, repopulate presets and re-apply."""
        orientation = self.orientation_combo.currentData()
        if not orientation:
            return
        self._populate_resolution_combo(orientation)
        # Now apply the new resolution
        self._apply_resolution_to_config()

    def _apply_resolution_to_config(self):
        """Read current combo values and write width/height to config."""
        data = self.resolution_combo.currentData()
        orientation = self.orientation_combo.currentData()
        if data:
            base_w, base_h = data
            if orientation == "portrait":
                # Portrait presets already store (w, h) as portrait
                # Landscape presets need swapping
                if base_w > base_h:
                    w, h = base_h, base_w
                else:
                    w, h = base_w, base_h
            else:
                # Landscape mode: if data is portrait-shaped, swap
                if base_h > base_w:
                    w, h = base_h, base_w
                else:
                    w, h = base_w, base_h
            self._debounce_set('width', w)
            self._debounce_set('height', h)
        self._update_resolution_preview()
        self._update_aspect_preview()

    def load_backend_widget(self):
        while self.backend_params_layout.count():
            item = self.backend_params_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        backend_name = self.config.get('active_backend', 'qwen3')

        try:
            backend_class = backends.BACKEND_MAP.get(backend_name) if backends else None
            if backend_class:
                if hasattr(backend_class, 'get_settings_widget'):
                    if self.mode_combo.isVisible():
                        mode = self.mode_combo.currentData()
                    else:
                        mode = self.config.get(
                            f"{backend_name}_mode", "default")

                    backend_widget = backend_class.get_settings_widget(
                        mode, self.config, self,
                        save_callback=self._debounce_set
                    )
                    self.backend_params_layout.addWidget(backend_widget)
                else:
                    self.backend_params_layout.addWidget(
                        QLabel(
                            f"{backend_name} has no configurable parameters."))
            else:
                self.backend_params_layout.addWidget(
                    QLabel("Backend implementation not found."))
        except Exception as e:
            self.backend_params_layout.addWidget(
                QLabel(f"<span style='color:red'>Error loading UI: {e}</span>"))
            logger.error(f"Error loading backend settings ({type(e).__name__}: {e})", exc_info=True)

    # ==========================================================
    # PAGE: HARDWARE
    # ==========================================================
    def _create_hardware_page(self):
        page = QWidget()
        layout = QFormLayout(page)

        dev_row = QHBoxLayout()
        self.qwen3_device_combo = QComboBox()
        self.qwen3_device_combo.addItem("cpu", "cpu")
        self.qwen3_device_combo.addItem("cuda:0", "cuda:0")
        self.qwen3_device_combo.addItem("cuda:1", "cuda:1")
        self.qwen3_device_combo.currentIndexChanged.connect(
            lambda i: self._debounce_set(
                'qwen3_device_map',
                self.qwen3_device_combo.itemData(i) or "cpu")
        )

        btn_detect = QPushButton("Auto-Detect GPU")
        btn_detect.clicked.connect(self.auto_detect_hardware)

        dev_row.addWidget(self.qwen3_device_combo)
        dev_row.addWidget(btn_detect)
        layout.addRow("Compute Device:", dev_row)

        self.qwen3_dtype_combo = QComboBox()
        self.qwen3_dtype_combo.addItems(["bfloat16", "float16"])
        self.qwen3_dtype_combo.currentTextChanged.connect(
            lambda t: self._debounce_set('qwen3_dtype', t))
        layout.addRow("Data Type:", self.qwen3_dtype_combo)

        self.chk_flash_attn = QCheckBox("Flash Attention 2")
        self.chk_flash_attn.setToolTip(
            "Faster, lower VRAM. Requires RTX 30/40 series.")
        self.chk_flash_attn.toggled.connect(
            lambda c: self._debounce_set(
                'qwen3_attn_implementation',
                "flash_attention_2" if c else "eager")
        )
        layout.addRow(self.chk_flash_attn)

        self.qwen3_size_combo = QComboBox()
        self.qwen3_size_combo.addItems(["1.7B", "0.6B"])
        self.qwen3_size_combo.setCurrentText(
            self.config.get('qwen3_size', '1.7B'))
        self.qwen3_size_combo.currentTextChanged.connect(
            lambda t: self._debounce_set('qwen3_size', t))
        layout.addRow("Model Size:", self.qwen3_size_combo)

        return page

    def auto_detect_hardware(self):
        """Auto-detect GPU hardware. Runs CUDA init in a one-shot
        QThread to avoid freezing the GUI for 1-3 seconds."""
        # Cancel any previous in-flight detection
        if hasattr(self, '_detect_thread') and self._detect_thread.isRunning():
            self._detect_thread._stop_event.set()
            self._detect_thread.finished.disconnect()
            self._detect_thread.quit()
            self._detect_thread.wait(2000)

        btn = self.sender()
        if btn:
            btn.setEnabled(False)
            btn.setText("Detecting...")

        def _on_done(result):
            # Ignore if this is a stale thread (user clicked again)
            if thread is not getattr(self, '_detect_thread', None):
                return
            try:
                if btn:
                    btn.setEnabled(True)
                    btn.setText("Auto-Detect GPU")
            except RuntimeError:
                return  # Widget already destroyed
            status, data = result
            if status == "found":
                count = len(data)
                self.qwen3_device_combo.blockSignals(True)
                self.qwen3_device_combo.clear()
                self.qwen3_device_combo.addItem("cpu", "cpu")
                for label, value in data:
                    self.qwen3_device_combo.addItem(label, value)
                self.qwen3_device_combo.setCurrentIndex(1)
                self.qwen3_device_combo.blockSignals(False)
                selected = self.qwen3_device_combo.currentData() or "cuda:0"
                self._debounce_set('qwen3_device_map', selected)
                QMessageBox.information(
                    self, "Success", f"Found {count} GPU(s).")
            elif status == "none":
                QMessageBox.warning(
                    self, "No GPU", "No CUDA-compatible GPU found.")
                self.qwen3_device_combo.setCurrentText("cpu")
            elif status == "no_torch":
                QMessageBox.critical(self, "Error", "PyTorch not installed.")
            else:
                QMessageBox.critical(self, "Error", f"Detection failed: {data}")

        thread = _HardwareDetectThread(self)
        self._detect_thread = thread
        thread.finished.connect(lambda: _on_done(thread.result))
        thread.start()

    # ==========================================================
    # PAGE: TEXT CHUNKING (Long Text Handling)
    # ==========================================================
    def _create_chunking_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        info_group = QGroupBox("Long Text Handling")
        info_layout = QVBoxLayout()
        info_label = QLabel(
            "<b>Automatic Text Chunking for TTS</b><br><br>"
            "When enabled, long text inputs are automatically split into "
            "smaller chunks that the TTS model can process reliably. Each "
            "chunk is generated separately and then concatenated to produce "
            "the final audio output.<br><br>"
            "<i>This prevents audio generation failures for long paragraphs "
            "and ensures consistent voice quality across chunked segments.</i>"
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #cccccc; padding: 10px;")
        info_layout.addWidget(info_label)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        settings_group = QGroupBox("Chunking Settings")
        form_layout = QFormLayout()

        self.chk_enable_chunking = QCheckBox("Enable Automatic Text Chunking")
        self.chk_enable_chunking.setToolTip(
            "When disabled, long texts may cause TTS failures")
        self.chk_enable_chunking.setChecked(True)
        self.chk_enable_chunking.toggled.connect(
            lambda c: self._debounce_set('enable_text_chunking', c))
        form_layout.addRow(self.chk_enable_chunking)

        self.spin_chunk_max_chars = QSpinBox()
        self.spin_chunk_max_chars.setRange(100, 2000)
        self.spin_chunk_max_chars.setSingleStep(50)
        self.spin_chunk_max_chars.setSuffix(" chars")
        self.spin_chunk_max_chars.setToolTip(
            "Maximum characters per audio chunk.")
        self.spin_chunk_max_chars.valueChanged.connect(
            lambda v: self._debounce_set('chunk_max_chars', v))
        form_layout.addRow("Max Chars per Chunk:", self.spin_chunk_max_chars)

        self.spin_chunk_max_sentences = QSpinBox()
        self.spin_chunk_max_sentences.setRange(1, 10)
        self.spin_chunk_max_sentences.setToolTip(
            "Maximum sentences per chunk.")
        self.spin_chunk_max_sentences.valueChanged.connect(
            lambda v: self._debounce_set('chunk_max_sentences', v))
        form_layout.addRow("Max Sentences per Chunk:",
                           self.spin_chunk_max_sentences)

        self.spin_chunk_min_chars = QSpinBox()
        self.spin_chunk_min_chars.setRange(10, 200)
        self.spin_chunk_min_chars.setSingleStep(10)
        self.spin_chunk_min_chars.setSuffix(" chars")
        self.spin_chunk_min_chars.setToolTip(
            "Minimum characters to form a valid chunk.")
        self.spin_chunk_min_chars.valueChanged.connect(
            lambda v: self._debounce_set('chunk_min_chars', v))
        form_layout.addRow("Min Chunk Size:", self.spin_chunk_min_chars)

        self.spin_warn_threshold = QSpinBox()
        self.spin_warn_threshold.setRange(500, 5000)
        self.spin_warn_threshold.setSingleStep(100)
        self.spin_warn_threshold.setSuffix(" chars")
        self.spin_warn_threshold.setToolTip(
            "Log a warning when text exceeds this length")
        self.spin_warn_threshold.valueChanged.connect(
            lambda v: self._debounce_set('chunk_warn_threshold', v))
        form_layout.addRow("Warning Threshold:", self.spin_warn_threshold)

        settings_group.setLayout(form_layout)
        layout.addWidget(settings_group)

        impact_group = QGroupBox("Estimated Impact")
        impact_layout = QVBoxLayout()
        impact_label = QLabel(
            "<b>Recommended Settings by Use Case:</b><br><br>"
            "• <b>Short slides (~200 chars):</b> Chunking rarely needed, "
            "defaults work fine<br>"
            "• <b>Medium slides (~500 chars):</b> Default settings "
            "(500 chars/chunk) recommended<br>"
            "• <b>Long narratives (~1000+ chars):</b> Consider 400-500 "
            "chars/chunk for best quality<br>"
            "• <b>Very long text (~2000+ chars):</b> Reduce to 300-400 "
            "chars if quality issues occur<br><br>"
            "<i>Note: Smaller chunks = more API calls but more reliable "
            "output. Larger chunks = fewer calls but may hit model limits.</i>"
        )
        impact_label.setWordWrap(True)
        impact_label.setStyleSheet("color: #aaaaaa; padding: 10px;")
        impact_layout.addWidget(impact_label)
        impact_group.setLayout(impact_layout)
        layout.addWidget(impact_group)

        layout.addStretch()
        return page

    # ==========================================================
    # PAGE: PATHS & CACHE
    # ==========================================================
    def _create_paths_page(self):
        page = QWidget()
        layout = QFormLayout(page)

        def make_validated_row(label, key, is_dir=True):
            row = QHBoxLayout()
            edit = QLineEdit()
            edit.setText(self.config.get(key, ""))

            def check_path():
                text = edit.text()
                self._debounce_set(key, text)

                exists = (os.path.isdir(text) if is_dir
                          else os.path.exists(text))
                p = edit.palette()
                if exists:
                    p.setColor(QPalette.ColorRole.Base, QColor("#1e2f23"))
                    p.setColor(QPalette.ColorRole.Text, QColor("#a5d6a7"))
                else:
                    p.setColor(QPalette.ColorRole.Base, QColor("#3d2220"))
                    p.setColor(QPalette.ColorRole.Text, QColor("#ef9a9a"))
                edit.setPalette(p)

            edit.editingFinished.connect(check_path)
            check_path()

            btn = QPushButton("...")
            btn.setMaximumWidth(30)
            btn.clicked.connect(
                lambda: self.browse_generic(edit, is_dir, config_key=key))

            row.addWidget(QLabel(label))
            row.addWidget(edit)
            row.addWidget(btn)
            return row

        layout.addRow(make_validated_row("HF Model Cache:", "hf_cache_dir"))
        layout.addRow(make_validated_row("HF Datasets:", "hf_datasets_dir"))
        layout.addRow(
            make_validated_row("Voice References:", "voice_references_root"))
        layout.addRow(make_validated_row(
            "Voice Design Cache:", "instruction_folder_root"))

        self.chk_symlinks = QCheckBox("Use Symlinks (Linux/Mac only)")
        self.chk_symlinks.setChecked(
            self.config.get("hf_use_symlinks", False))
        self.chk_symlinks.toggled.connect(
            lambda c: self._debounce_set("hf_use_symlinks", c))
        layout.addRow(self.chk_symlinks)

        return page

    def browse_output_dir(self):
        d = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", self.out_dir_edit.text())
        if d:
            self.out_dir_edit.setText(d)
            # Emit editingFinished so the persisted path is consistent
            # with the browse_generic pattern
            self.out_dir_edit.editingFinished.emit()

    def browse_generic(self, line_edit, is_dir, config_key=None):
        result = None
        if is_dir:
            result = QFileDialog.getExistingDirectory(
                self, "Select Directory", line_edit.text())
        else:
            result, _ = QFileDialog.getOpenFileName(
                self, "Select File", line_edit.text())

        if result:
            line_edit.setText(result)
            # setText() does NOT trigger editingFinished, so we must
            # save explicitly to avoid the "displayed but not persisted" bug.
            if config_key:
                self._debounce_set(config_key, result)
            # Also trigger the editingFinished handlers if any
            line_edit.editingFinished.emit()

    # ==========================================================
    # PAGE: DIAGNOSTICS
    # ==========================================================
    def _create_diagnostics_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        # ── Verbose Logging Toggle ──────────────────────────────────────
        self.chk_verbose = QCheckBox("Verbose Logging")
        self.chk_verbose.setStyleSheet(style_checkbox())
        self.chk_verbose.setToolTip(
            "Show detailed debug messages in the log panel\n"
            "(PIL, asyncio, and engine internals)."
        )
        self.chk_verbose.stateChanged.connect(
            lambda state: self._debounce_set('verbose', bool(state))
        )
        layout.addWidget(self.chk_verbose)

        # ── Resolution Verification ────────────────────────────────────
        res_group = QGroupBox("Resolution Verification")
        res_group.setStyleSheet(style_group_box())
        res_layout = QFormLayout(res_group)

        res_layout.addRow(
            QLabel("Select a project to verify its source resolution "
                   "(inspects PNG files, HTML/CSS, and .ppt-slide classes):")
        )

        self._verify_res_combo = QComboBox()
        self._verify_res_combo.setStyleSheet(style_input())
        self._verify_res_combo.setMinimumWidth(300)
        res_layout.addRow("Project:", self._verify_res_combo)

        self._verify_res_label = QLabel("No verification run yet.")
        self._verify_res_label.setWordWrap(True)
        self._verify_res_label.setStyleSheet(style_summary_label())
        res_layout.addRow("Result:", self._verify_res_label)

        btn_row = QHBoxLayout()
        btn_verify = QPushButton("Verify Resolution")
        btn_verify.setStyleSheet(style_primary_button())
        btn_verify.setToolTip(
            "Inspect the selected project's images/ folder and HTML slides to\n"
            "detect the source resolution. Updates the project manifest\n"
            "with the detected resolution and sets status."
        )
        btn_verify.clicked.connect(self._on_verify_resolution)
        btn_row.addWidget(btn_verify)

        btn_verify_all = QPushButton("Verify All Projects")
        btn_verify_all.setStyleSheet(style_primary_button())
        btn_verify_all.setToolTip(
            "Run resolution verification on ALL projects in the library.\n"
            "Each project's manifest is updated with detected resolution.\n"
            "Results are shown below."
        )
        btn_verify_all.clicked.connect(self._on_verify_all_resolution)
        self._verify_all_btn = btn_verify_all  # store ref for enable/disable
        btn_row.addWidget(btn_verify_all)

        res_layout.addRow(btn_row)

        # Batch results area for "Verify All"
        self._verify_all_text = QTextEdit()
        self._verify_all_text.setReadOnly(True)
        self._verify_all_text.setMaximumHeight(160)
        self._verify_all_text.setStyleSheet(
            "background-color: #1e1e2e; color: #cdd6f4; "
            "font-family: monospace; font-size: 11px; "
            "border: 1px solid #45475a; border-radius: 4px;"
        )
        self._verify_all_text.setPlaceholderText(
            "Batch verification results will appear here...")
        res_layout.addRow("All Projects:", self._verify_all_text)

        layout.addWidget(res_group)

        # ── System Information ──────────────────────────────────────────
        info_text = QTextEdit()
        info_text.setReadOnly(True)

        try:
            import torch
            cuda_ver = torch.version.cuda or "N/A"
            gpu_name = (torch.cuda.get_device_name(0)
                        if torch.cuda.is_available() else "No GPU")
            torch_ver = torch.__version__
        except ImportError:
            cuda_ver = "N/A"
            gpu_name = "N/A"
            torch_ver = "Not Installed"

        report = f"""
    <h3>System Information</h3>
    <b>OS:</b> {platform.system()} {platform.release()}<br>
    <b>Python:</b> {sys.version.split()[0]}<br>
    <b>PyTorch:</b> {torch_ver}<br>
    <b>CUDA:</b> {cuda_ver}<br>
    <b>GPU:</b> {gpu_name}<br>
    <h3>Configuration</h3>
    <b>Active Backend:</b> {self.config.get('active_backend', 'qwen3')}<br>
    <b>Cache Dir:</b> {self.config.get('hf_cache_dir')}
        """
        info_text.setHtml(report)

        btn_copy = QPushButton("Copy Report to Clipboard")
        btn_copy.clicked.connect(
            lambda: QApplication.clipboard().setText(info_text.toPlainText()))

        layout.addWidget(info_text)
        layout.addWidget(btn_copy)

        # Populate project combo when the page is first shown
        QTimer.singleShot(100, self._populate_verify_res_combo)

        return page

    def _populate_verify_res_combo(self):
        """Fill the resolution verification combo with project names."""
        from utils import get_project_dirs
        projects_root = self.config.get("projects_root", "Projects")
        if not os.path.isabs(projects_root):
            projects_root = os.path.join(
                AppConfig.APP_ROOT, projects_root)
        project_dirs = get_project_dirs()
        self._verify_res_combo.clear()
        for pd in project_dirs:
            self._verify_res_combo.addItem(os.path.basename(pd), pd)

    def _on_verify_resolution(self):
        """Run resolution verification on the selected project in a thread."""
        project_path = self._verify_res_combo.currentData()
        if not project_path:
            QMessageBox.warning(self, "No Project", "Select a project first.")
            return

        project_name = self._verify_res_combo.currentText()
        self._verify_res_label.setText(f"Verifying {project_name}...")

        self._verify_thread = _ResolutionVerifyThread(project_path, self)
        self._verify_thread.result_ready.connect(
            lambda result, pp=project_path: self._on_verify_result(
                project_name, result, pp
            )
        )
        self._verify_thread.start()

    def _on_verify_result(self, project_name: str, result: dict,
                          project_path: str = ""):
        """Handle resolution verification result."""
        from utils import (
            save_project_manifest, load_project_manifest,
            update_library_manifest, _aspect_ratio_label,
        )

        res = result.get("resolution")
        method = result.get("method")
        details = result.get("details", "")

        if res is None:
            self._verify_res_label.setText(
                f"<b>{project_name}:</b> {details}"
            )
            QMessageBox.warning(
                self, "Verification Failed",
                f"Could not detect resolution for {project_name}.\n\n{details}",
            )
            return

        w, h = res
        aspect = _aspect_ratio_label(w, h)

        # Save detected resolution to manifest
        manifest = {}
        try:
            from utils import set_source_resolution
            set_source_resolution(project_path, w, h)
            manifest = load_project_manifest(project_path)
            # Update status: "incomplete" if previously completed
            # (resolution changed), "not rendered" otherwise
            if manifest.get("status") == "completed":
                manifest["status"] = "incomplete"
            elif manifest.get("status") not in ("incomplete", "not rendered"):
                manifest["status"] = "not rendered"
            save_project_manifest(project_path, manifest)
            update_library_manifest(project_path)
        except Exception as exc:
            logger.error(f"[Settings] Failed to save resolution ({type(exc).__name__}: {exc})", exc_info=True)

        self._verify_res_label.setText(
            f"<b>{project_name}:</b> {w}x{h} ({aspect}) "
            f"[{method}] — status updated"
        )
        QMessageBox.information(
            self, "Resolution Verified",
            f"<b>{project_name}</b><br><br>"
            f"{details}<br><br>"
            f"Status set to: <b>{manifest.get('status', 'unknown')}</b>",
        )

    # ── Batch Resolution Verification ──────────────────────────────────

    def _on_verify_all_resolution(self):
        """Run resolution verification on ALL projects in a background thread."""
        from utils import get_project_dirs
        project_dirs = get_project_dirs()
        if not project_dirs:
            QMessageBox.information(
                self, "No Projects",
                "No projects found in the library to verify.")
            return

        self._verify_all_text.clear()
        self._verify_all_text.append(
            f"Starting verification of {len(project_dirs)} projects...\n")

        # Disable button to prevent double-clicks
        self._verify_all_btn.setEnabled(False)

        self._batch_verify_thread = _BatchResolutionVerifyThread(
            project_dirs, self)
        self._batch_verify_thread.project_done.connect(
            self._on_batch_project_done)
        self._batch_verify_thread.all_done.connect(
            self._on_batch_verify_done)
        self._batch_verify_thread.start()

    def _on_batch_project_done(self, project_name: str, result: dict):
        """Handle one project result from batch verification."""
        from utils import (
            get_project_dirs, save_project_manifest,
            load_project_manifest, update_library_manifest,
            _aspect_ratio_label,
        )

        res = result.get("resolution")
        method = result.get("method")
        details = result.get("details", "")

        if res is None:
            self._verify_all_text.append(
                f"  [SKIP] {project_name}: {details}")
            return

        w, h = res
        aspect = _aspect_ratio_label(w, h)

        # Save detected resolution to project manifest + library manifest
        try:
            # Find the full project path
            project_path = ""
            for pd in get_project_dirs():
                if os.path.basename(pd) == project_name:
                    project_path = pd
                    break
            if project_path:
                from utils import set_source_resolution
                set_source_resolution(project_path, w, h)
                manifest = load_project_manifest(project_path)
                if manifest.get("status") == "completed":
                    manifest["status"] = "incomplete"
                elif manifest.get("status") not in ("incomplete", "not rendered"):
                    manifest["status"] = "not rendered"
                save_project_manifest(project_path, manifest)
                update_library_manifest(project_path)
        except Exception as exc:
            logger.error(
                f"[Settings] Batch verify save failed for {project_name} ({type(exc).__name__}: {exc})",
                exc_info=True,
            )

        self._verify_all_text.append(
            f"  [OK]   {project_name}: {w}x{h} ({aspect}) [{method}]")

    def _on_batch_verify_done(self, total: int, successes: int):
        """Called when batch verification finishes all projects."""
        failed = total - successes
        self._verify_all_text.append(
            f"\nDone: {successes}/{total} verified, "
            f"{failed} could not be detected.")

        if hasattr(self, '_verify_all_btn'):
            self._verify_all_btn.setEnabled(True)

        # Request main window to refresh the library list so resolution
        # updates are visible immediately
        self.library_refresh_requested.emit()

# ═══════════════════════════════════════════════════════════════════════════════
#  NEW PROJECT TAB WIDGET — Unified project creation with optional import
#  and speaker script support
# ═══════════════════════════════════════════════════════════════════════════════

class NewProjectTabWidget(QWidget):
    """Unified project creation widget — minimalist styling."""

    def __init__(
        self,
        config: AppConfig,
        on_created_callback,
        prefill_import_path: str = "",
        parent=None,
    ):
        super().__init__(parent)
        self.config = config
        self._on_created = on_created_callback
        self._prefill_import = prefill_import_path
        self._cached_script_path = None
        self._cached_script_text = ""
        self.init_ui()

    # ══════════════════════════════════════════════════════════════════════
    #  UI CONSTRUCTION
    # ══════════════════════════════════════════════════════════════════════

    def init_ui(self):
        self.setStyleSheet(style_page_widget())
        root = QVBoxLayout(self)
        root.setContentsMargins(40, 32, 40, 24)
        root.setSpacing(0)

        # ── Header ────────────────────────────────────────────────────────
        title = QLabel("New Project")
        title.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {Palette.TEXT}; margin-bottom: 2px;")
        root.addWidget(title)

        subtitle = QLabel(
            "Create a blank project, import a file, or attach a narration script."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet(f"color: {Palette.TEXT_DIM}; font-size: 12px; margin-bottom: 20px;")
        root.addWidget(subtitle)

        # ── Project Name ──────────────────────────────────────────────────
        grp_name = QGroupBox("Project Name")
        grp_name.setStyleSheet(style_group_box())
        name_lay = QVBoxLayout()
        name_lay.setContentsMargins(0, 6, 0, 4)
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("e.g. My Presentation")
        self.name_edit.setStyleSheet(style_input())
        self.name_edit.textChanged.connect(self._update_summary)
        name_lay.addWidget(self.name_edit)
        grp_name.setLayout(name_lay)
        root.addWidget(grp_name)

        # ── Slide Content Source ──────────────────────────────────────────
        grp_content = QGroupBox("Slide Content")
        grp_content.setStyleSheet(style_group_box())
        content_lay = QVBoxLayout()
        content_lay.setContentsMargins(0, 6, 0, 4)
        content_lay.setSpacing(8)

        self.radio_blank = QRadioButton("Blank project")
        self.radio_blank.setStyleSheet(style_radio())
        self.radio_blank.setChecked(True)

        self.radio_import = QRadioButton("Import from file")
        self.radio_import.setStyleSheet(style_radio())

        import_row = QHBoxLayout()
        import_row.setContentsMargins(20, 0, 0, 0)
        self.import_path_edit = QLineEdit()
        self.import_path_edit.setPlaceholderText("PDF, PPTX, TXT, CSV, or JSON…")
        self.import_path_edit.setStyleSheet(style_input())
        self.import_path_edit.setEnabled(False)
        self.import_path_edit.textChanged.connect(self._update_summary)

        self.import_browse_btn = QPushButton("Browse")
        self.import_browse_btn.setStyleSheet(style_button())
        self.import_browse_btn.setEnabled(False)
        self.import_browse_btn.setFixedWidth(72)
        self.import_browse_btn.clicked.connect(self._browse_import_file)

        import_row.addWidget(self.import_path_edit, stretch=1)
        import_row.addWidget(self.import_browse_btn)

        self.radio_blank.toggled.connect(self._on_content_mode_changed)
        self.radio_import.toggled.connect(self._on_content_mode_changed)

        content_lay.addWidget(self.radio_blank)
        content_lay.addWidget(self.radio_import)
        content_lay.addLayout(import_row)
        grp_content.setLayout(content_lay)
        root.addWidget(grp_content)

        # ── Narration Script ──────────────────────────────────────────────
        grp_script = QGroupBox("Narration Script")
        grp_script.setStyleSheet(style_group_box())
        script_lay = QVBoxLayout()
        script_lay.setContentsMargins(0, 6, 0, 4)
        script_lay.setSpacing(8)

        script_hint = QLabel("Optional — PDF, TXT, CSV, or JSON with blank-line or structured sections.")
        script_hint.setWordWrap(True)
        script_hint.setStyleSheet(f"color: {Palette.TEXT_DIM}; font-size: 11px;")

        script_row = QHBoxLayout()
        self.script_path_edit = QLineEdit()
        self.script_path_edit.setPlaceholderText("Select a .txt script file…")
        self.script_path_edit.setStyleSheet(style_input())
        self.script_path_edit.textChanged.connect(self._update_summary)

        self.script_browse_btn = QPushButton("Browse")
        self.script_browse_btn.setStyleSheet(style_button())
        self.script_browse_btn.setFixedWidth(72)
        self.script_browse_btn.clicked.connect(self._browse_script_file)

        script_row.addWidget(self.script_path_edit, stretch=1)
        script_row.addWidget(self.script_browse_btn)

        self.chk_override_text = QCheckBox("Replace existing slide text with script")
        self.chk_override_text.setChecked(True)
        self.chk_override_text.setStyleSheet(style_checkbox())

        script_lay.addWidget(script_hint)
        script_lay.addLayout(script_row)
        script_lay.addWidget(self.chk_override_text)
        grp_script.setLayout(script_lay)
        root.addWidget(grp_script)

        # ── Resolution ────────────────────────────────────────────────────
        grp_res = QGroupBox("Resolution")
        grp_res.setStyleSheet(style_group_box())
        res_lay = QHBoxLayout()
        res_lay.setContentsMargins(0, 6, 0, 4)

        self.resolution_combo = QComboBox()
        self.resolution_combo.setStyleSheet(style_input())
        self.resolution_combo.addItem("--")  # placeholder before _sync

        self.new_orientation_combo = QComboBox()
        self.new_orientation_combo.setStyleSheet(style_input())
        self.new_orientation_combo.addItem("Landscape", "landscape")
        self.new_orientation_combo.addItem("Portrait", "portrait")

        # Sync combo to current config, then wire signals
        self._sync_new_project_resolution()
        self.new_orientation_combo.currentIndexChanged.connect(
            self._on_new_orientation_changed)
        self.resolution_combo.currentIndexChanged.connect(
            self._update_summary)

        res_lay.addWidget(self.resolution_combo, stretch=3)
        res_lay.addWidget(self.new_orientation_combo, stretch=2)
        grp_res.setLayout(res_lay)
        root.addWidget(grp_res)

        # ── Summary ───────────────────────────────────────────────────────
        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet(style_summary_label())
        self.summary_label.setMinimumHeight(40)
        root.addWidget(self.summary_label)

        root.addSpacing(12)

        # ── Buttons ───────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setStyleSheet(style_button())
        self.btn_cancel.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_cancel.clicked.connect(self._cancel)

        self.btn_create = QPushButton("Create")
        self.btn_create.setStyleSheet(style_primary_button())
        self.btn_create.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_create.clicked.connect(self._create_project)

        btn_row.addWidget(self.btn_cancel)
        btn_row.addSpacing(8)
        btn_row.addWidget(self.btn_create)
        root.addLayout(btn_row)

        root.addStretch()

        # ── Prefill from quick-import ─────────────────────────────────────
        if self._prefill_import:
            self.radio_import.setChecked(True)
            self.import_path_edit.setText(self._prefill_import)
            if not self.name_edit.text().strip():
                self.name_edit.setText(
                    os.path.splitext(os.path.basename(self._prefill_import))[0]
                )

        self._update_summary()

    # ══════════════════════════════════════════════════════════════════════
    #  UI EVENT HANDLERS
    # ══════════════════════════════════════════════════════════════════════

    def _on_content_mode_changed(self):
        if not self.radio_import.isChecked():
            self._update_summary()
            return
        is_import = True
        self.import_path_edit.setEnabled(is_import)
        self.import_browse_btn.setEnabled(is_import)
        self._update_summary()

    def _browse_import_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select File to Import", "", FILE_FILTER_IMPORT
        )
        if path:
            self.import_path_edit.setText(path)
            if not self.name_edit.text().strip():
                self.name_edit.setText(
                    os.path.splitext(os.path.basename(path))[0]
                )

    def _browse_script_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Narration Script", "", FILE_FILTER_SCRIPT
        )
        if path:
            self.script_path_edit.setText(path)
            try:
                content = self._read_file_text(path).strip()
                sections = self._parse_script_sections(content)
                logger.info(
                    f"[NewProject] Script preview: {len(sections)} section(s)"
                )
            except Exception as exc:
                logger.warning(f"[NewProject] Script preview failed ({type(exc).__name__}: {exc})", exc_info=True)

    # ══════════════════════════════════════════════════════════════════════
    #  RESOLUTION HELPERS
    # ══════════════════════════════════════════════════════════════════════

    def _sync_new_project_resolution(self):
        """Populate resolution combo from config and select matching preset."""
        current_w = self.config.get("width", 1280)
        current_h = self.config.get("height", 720)
        orientation = "portrait" if current_h > current_w else "landscape"

        self.new_orientation_combo.blockSignals(True)
        oidx = self.new_orientation_combo.findData(orientation)
        if oidx >= 0:
            self.new_orientation_combo.setCurrentIndex(oidx)
        self.new_orientation_combo.blockSignals(False)

        source = (AppConfig.PORTRAIT_PRESETS if orientation == "portrait"
                  else AppConfig.RESOLUTION_PRESETS)
        self.resolution_combo.blockSignals(True)
        self.resolution_combo.clear()
        matched = False
        for name, (w, h) in source.items():
            self.resolution_combo.addItem(name, (w, h))
            if w == current_w and h == current_h:
                self.resolution_combo.setCurrentIndex(
                    self.resolution_combo.count() - 1)
                matched = True
        if not matched:
            self.resolution_combo.setCurrentIndex(0)
        self.resolution_combo.blockSignals(False)

    def _on_new_orientation_changed(self, index):
        """Repopulate resolution presets when orientation toggles."""
        orientation = self.new_orientation_combo.currentData()
        if not orientation:
            return
        source = (AppConfig.PORTRAIT_PRESETS if orientation == "portrait"
                  else AppConfig.RESOLUTION_PRESETS)
        self.resolution_combo.blockSignals(True)
        self.resolution_combo.clear()
        for name, (w, h) in source.items():
            self.resolution_combo.addItem(name, (w, h))
        self.resolution_combo.blockSignals(False)
        self._update_summary()

    # ══════════════════════════════════════════════════════════════════════
    #  LIVE SUMMARY
    # ══════════════════════════════════════════════════════════════════════

    def _update_summary(self):
        name = self.name_edit.text().strip()
        is_import = self.radio_import.isChecked()
        import_path = self.import_path_edit.text().strip() if is_import else ""
        script_path = self.script_path_edit.text().strip()

        parts = []

        if not name:
            parts.append(f'<span style="color:{Palette.WARNING}">Enter a project name to continue</span>')
        else:
            parts.append(f"<b>{html.escape(name)}</b>")

        if import_path:
            ext = os.path.splitext(import_path)[1].upper().lstrip(".")
            parts.append(f"from {os.path.basename(import_path)} ({ext})")
        else:
            parts.append("1 blank slide")

        if script_path:
            if script_path != self._cached_script_path:
                self._cached_script_path = script_path
                self._cached_script_text = self._read_file_text(script_path) if script_path else ""
            script_text = self._cached_script_text
            try:
                content = script_text.strip()
                sections = self._parse_script_sections(content)
                action = "replace" if self.chk_override_text.isChecked() else "fill"
                parts.append(f"script: {len(sections)} section(s), {action}")
            except Exception:
                parts.append(f"script: {os.path.basename(script_path)} (read error)")

        res_text = self.resolution_combo.currentText().split("(")[0].strip()
        parts.append(res_text)

        self.summary_label.setText("  ·  ".join(parts))
        self.btn_create.setEnabled(bool(name))

    # ══════════════════════════════════════════════════════════════════════
    #  PROJECT CREATION
    # ══════════════════════════════════════════════════════════════════════

    def _create_project(self):
        name = _sanitize_project_name(self.name_edit.text().strip())
        if not name:
            QMessageBox.warning(self, "Missing Name", "Please enter a valid project name.")
            return

        self._set_ui_enabled(False)

        try:
            is_import = self.radio_import.isChecked()
            import_path = self.import_path_edit.text().strip() if is_import else ""
            script_path = self.script_path_edit.text().strip()

            if is_import and import_path and not os.path.isfile(import_path):
                raise ValueError(f"Import file not found: {import_path}")
            if script_path and not os.path.isfile(script_path):
                raise ValueError(f"Script file not found: {script_path}")

            # ── Create base project ───────────────────────────────────────
            on_exists = "error"  # default
            if import_path:
                # Check if project already exists and ask the user
                projects_root = self.config.get("projects_root", "Projects")
                if not os.path.isabs(projects_root):
                    projects_root = os.path.join(AppConfig.APP_ROOT, projects_root)
                candidate_path = os.path.join(projects_root, name)
                if os.path.exists(candidate_path):
                    on_exists = _ask_overwrite_or_append(self, name)
                    if on_exists is None:
                        self._set_ui_enabled(True)
                        return  # User cancelled
                logger.info(
                    f"[NewProject] Importing '{os.path.basename(import_path)}' as '{name}'"
                )
                project_path = import_file_as_project(
                    import_path, self.config.settings, project_name=name,
                    on_exists=on_exists,
                )
            else:
                projects_root = self.config.get("projects_root", "Projects")
                if not os.path.isabs(projects_root):
                    projects_root = os.path.join(AppConfig.APP_ROOT, projects_root)
                project_path = os.path.join(projects_root, name)

                if os.path.exists(project_path):
                    on_exists = _ask_overwrite_or_append(self, name)
                    if on_exists is None:
                        self._set_ui_enabled(True)
                        return  # User cancelled
                    if on_exists == "overwrite":
                        shutil.rmtree(project_path)
                    elif on_exists == "append":
                        # Keep existing project, just apply script later
                        pass

                if on_exists != "append" or not os.path.exists(project_path):
                    os.makedirs(project_path, exist_ok=True)
                    initialize_project_files(project_path)
                    logger.info(f"[NewProject] Created blank project: {name}")
                else:
                    logger.info(f"[NewProject] Appending to existing project: {name}")

            # ── Apply resolution ──────────────────────────────────────────
            res_data = self.resolution_combo.currentData()
            if res_data:
                width, height = res_data
                self.config.set_resolution(width, height)

            # ── Apply speaker script ──────────────────────────────────────
            if script_path:
                updated = self._apply_speaker_script(project_path, script_path)
                logger.info(
                    f"[NewProject] Applied narration script: {updated} slide(s) updated"
                )

            # ── Callback & close ──────────────────────────────────────────
            if self._on_created:
                self._on_created(project_path)
            self._close_tab()

        except ImportError as exc:
            logger.error(f"[NewProject] Missing library: {exc}")
            QMessageBox.critical(self, "Missing Library", str(exc))
            self._set_ui_enabled(True)
        except ValueError as exc:
            logger.warning(f"[NewProject] Invalid input: {exc}")
            QMessageBox.warning(self, "Input Error", str(exc))
            self._set_ui_enabled(True)
        except Exception as exc:
            logger.error(f"[NewProject] Failed: {exc}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to create project:\n{exc}")
            self._set_ui_enabled(True)

    def _set_ui_enabled(self, enabled: bool):
        """Toggle all interactive elements."""
        creating = not enabled
        self.btn_create.setEnabled(enabled)
        self.btn_create.setText("Creating…" if creating else "Create")
        for w in (
            self.name_edit, self.radio_blank, self.radio_import,
            self.import_path_edit, self.import_browse_btn,
            self.script_path_edit, self.script_browse_btn,
            self.chk_override_text, self.resolution_combo,
            self.new_orientation_combo,
            self.btn_cancel,
        ):
            w.setEnabled(enabled)

    # ══════════════════════════════════════════════════════════════════════
    #  SPEAKER SCRIPT ENGINE
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _read_file_text(file_path: str) -> str:
        """Read text content from a script file. Supports PDF, TXT, CSV, and JSON."""
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == ".pdf":
            try:
                import fitz  # PyMuPDF
            except ImportError:
                raise ImportError(
                    "PyMuPDF is required to read PDF scripts. "
                    "Install with: pip install PyMuPDF"
                )
            doc = fitz.open(file_path)
            try:
                text = "\n\n".join(page.get_text() for page in doc)
            finally:
                doc.close()
            return text
            
        elif ext == ".json":
            # Delegate to the proper parser so that structured speaker-script
            # JSON (with a "slides" array) is handled correctly.  The old code
            # joined top-level keys, which turned a 12-slide JSON into 3
            # sections (one per top-level key).
            try:
                from utils import parse_speaker_script_json
            except (ImportError, AttributeError):
                parse_speaker_script_json = None

            if parse_speaker_script_json is not None:
                try:
                    parsed = parse_speaker_script_json(file_path)
                    # Join each slide's content with double-newlines so that
                    # _parse_script_sections() can split them properly.
                    return "\n\n".join(
                        slide["content"] for slide in parsed["slides"]
                    )
                except ValueError:
                    pass  # Not a speaker-script JSON — fall back to generic dump

            # Generic JSON fallback
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return "\n\n".join(
                        item if isinstance(item, str) else json.dumps(item)
                        for item in data
                    )
                elif isinstance(data, dict):
                    return "\n\n".join(
                        f"{k}: {v}" if isinstance(v, str) else f"{k}: {json.dumps(v)}"
                        for k, v in data.items()
                    )
                return str(data)
            
        else:  # .txt, .csv, or anything else -> read as plain text
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()

    @staticmethod
    def _parse_script_sections(content: str) -> list[str]:
        """Split script into sections. Supports blank-line separation and
        labeled format:  Slide 1: text"""
        if not content or not content.strip():
            return []
        labeled = re.findall(
            r"^Slide\s+\d+\s*[:.)]\s*(.+)$",
            content.strip(), re.MULTILINE | re.IGNORECASE,
        )
        if labeled:
            return [s.strip() for s in labeled if s.strip()]
        sections = re.split(r"\n\s*\n", content.strip())
        return [s.strip() for s in sections if s.strip()]

    def _apply_speaker_script(self, project_path: str, script_path: str) -> int:
        """Apply narration script to slide .txt files. Creates extra slides
        if the script has more sections than existing slides."""
        content = self._read_file_text(script_path)

        sections = self._parse_script_sections(content)
        if not sections:
            logger.warning("[NewProject] Speaker script is empty")
            return 0

        override = self.chk_override_text.isChecked()
        txt_files = sorted(
            glob.glob(os.path.join(project_path, "slide*.txt")),
            key=natural_sort_key,
        )
        updated = 0

        # Update existing slides
        for i, txt_file in enumerate(txt_files):
            if i >= len(sections):
                break
            should_write = False
            if override:
                should_write = True
            else:
                try:
                    existing = Path(txt_file).read_text(encoding="utf-8").strip()
                    should_write = not existing
                except Exception:
                    should_write = True
            if should_write:
                with open(txt_file, "w", encoding="utf-8") as f:
                    f.write(sections[i])
                updated += 1

        # Create additional slides for extra sections
        if len(sections) > len(txt_files):
            next_num = get_next_slide_number(project_path)
            for i in range(len(txt_files), len(sections)):
                slide_num = next_num + (i - len(txt_files))
                html_path = os.path.join(project_path, f"slide{slide_num}.html")
                txt_path = os.path.join(project_path, f"slide{slide_num}.txt")
                html_content = get_blank_slide_html()
                with open(html_path, "w", encoding="utf-8") as f:
                    f.write(html_content)
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(sections[i])
                updated += 1

        return updated

    # ══════════════════════════════════════════════════════════════════════
    #  TAB MANAGEMENT
    # ══════════════════════════════════════════════════════════════════════

    def _cancel(self):
        self._close_tab()

    def _close_tab(self):
        parent = self.parent()
        while parent is not None:
            if isinstance(parent, QTabWidget):
                for i in range(parent.count()):
                    if parent.widget(i) is self:
                        parent.removeTab(i)
                        self.deleteLater()
                        return
            parent = parent.parent()
        self.hide()
        self.deleteLater()

# =================================================================
# SLIDE LIST PANEL (Modular Sub-Component)
# =================================================================

class SlideListPanel(QWidget):
    """Left panel: slide navigation, ordering, and management controls."""

    slide_selected = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        header = QLabel("Slides")
        header.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet("color: #ddd; padding: 4px;")
        layout.addWidget(header)

        self.slide_list = QListWidget()
        self.slide_list.setStyleSheet("""
            QListWidget {
                background-color: #1e1e1e;
                color: #cccccc;
                border: 1px solid #333;
                border-radius: 4px;
                font-size: 13px;
                outline: none;
            }
            QListWidget::item {
                padding: 8px 12px;
                border-bottom: 1px solid #2a2a2a;
            }
            QListWidget::item:selected {
                background-color: #007acc;
                color: white;
            }
            QListWidget::item:hover {
                background-color: #2a2d2e;
            }
        """)
        self.slide_list.currentRowChanged.connect(self._on_row_changed)
        layout.addWidget(self.slide_list, stretch=1)

        # Action buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(3)

        self.btn_add = QPushButton("+")
        self.btn_add.setToolTip("Add blank slide after current")
        self.btn_remove = QPushButton("−")
        self.btn_remove.setToolTip("Remove current slide")
        self.btn_duplicate = QPushButton("⧉")
        self.btn_duplicate.setToolTip("Duplicate current slide")
        self.btn_move_up = QPushButton("▲")
        self.btn_move_up.setToolTip("Move slide up")
        self.btn_move_down = QPushButton("▼")
        self.btn_move_down.setToolTip("Move slide down")

        for btn in [self.btn_add, self.btn_remove, self.btn_duplicate,
                     self.btn_move_up, self.btn_move_down]:
            btn.setFixedSize(34, 28)
            btn.setStyleSheet("""
                QPushButton {
                    background: #3c3c3c; color: #ddd; border: 1px solid #555;
                    border-radius: 3px; font-size: 14px; font-weight: bold;
                }
                QPushButton:hover { background: #505050; }
                QPushButton:pressed { background: #007acc; }
            """)
            btn_layout.addWidget(btn)

        layout.addLayout(btn_layout)

    def _on_row_changed(self, row):
        if row >= 0:
            self.slide_selected.emit(row)

    def load_slides(self, slide_files):
        self.slide_list.blockSignals(True)
        self.slide_list.clear()
        for i, path in enumerate(slide_files):
            item = QListWidgetItem(f"  Slide {i + 1}")
            item.setData(Qt.ItemDataRole.UserRole, path)
            item.setSizeHint(QSize(0, 38))
            self.slide_list.addItem(item)
        if slide_files:
            self.slide_list.setCurrentRow(0)
        self.slide_list.blockSignals(False)

    def current_index(self):
        return self.slide_list.currentRow()

    def set_current_index(self, index):
        if 0 <= index < self.slide_list.count():
            self.slide_list.setCurrentRow(index)

    def count(self):
        """Return the total number of slides in the list."""
        return self.slide_list.count()


# =================================================================
# SLIDE PREVIEW PANEL (Modular Sub-Component)
# =================================================================

class SlidePreviewPanel(QWidget):
    """
    Center panel: live HTML preview with zoom/scale controls.

    Uses QWebEngineView to render slides with full CSS support
    (Tailwind, Google Fonts, Material Icons, etc.).
    Scaling is applied via CSS zoom on the page body.
    """

    PRESET_SCALES = ["Fit", "25%", "50%", "75%", "100%", "125%", "150%", "200%"]

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self._scale_percent = 100
        self._fit_mode = False
        self._base_dir = ""
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # --- Zoom Toolbar ---
        self._toolbar = QFrame()
        self._toolbar.setFixedHeight(40)
        self._toolbar.setObjectName("previewToolbar")
        self._toolbar.setStyleSheet("""
            #previewToolbar {
                background: #252526;
                border-bottom: 1px solid #3c3c3c;
            }
            QPushButton {
                background: #3c3c3c; color: #ddd; border: 1px solid #555;
                border-radius: 3px; padding: 3px 10px; font-size: 12px;
                min-height: 24px;
            }
            QPushButton:hover { background: #505050; }
            QPushButton:pressed { background: #007acc; }
            QComboBox {
                background: #3c3c3c; color: #ddd; border: 1px solid #555;
                border-radius: 3px; padding: 3px 8px; font-size: 12px;
                min-height: 24px; min-width: 75px;
            }
            QComboBox::drop-down { border: none; width: 20px; }
            QComboBox QAbstractItemView {
                background: #2d2d2d; color: #ddd;
                selection-background-color: #007acc; outline: none;
            }
            QComboBox QLineEdit {
                background: #3c3c3c; color: #ddd; border: none;
            }
            QLabel { color: #777; font-size: 11px; background: transparent; }
        """)

        tb = QHBoxLayout(self._toolbar)
        tb.setContentsMargins(10, 4, 10, 4)

        self.btn_fit = QPushButton("⊞ Fit to Screen")
        self.btn_fit.setToolTip("Scale slide to fit the preview area")
        self.btn_fit.clicked.connect(self.fit_to_screen)

        self.btn_zoom_out = QPushButton("-")
        self.btn_zoom_out.setFixedSize(28, 28)
        self.btn_zoom_out.setToolTip("Zoom out (-25%)")
        self.btn_zoom_out.clicked.connect(lambda: self._step_scale(-25))

        self.scale_combo = QComboBox()
        self.scale_combo.setEditable(True)
        self.scale_combo.setFixedWidth(85)
        for s in self.PRESET_SCALES:
            self.scale_combo.addItem(s)
        self.scale_combo.setCurrentIndex(4)  # "100%"
        self.scale_combo.currentTextChanged.connect(self._on_scale_text_changed)

        self.btn_zoom_in = QPushButton("+")
        self.btn_zoom_in.setFixedSize(28, 28)
        self.btn_zoom_in.setToolTip("Zoom in (+25%)")
        self.btn_zoom_in.clicked.connect(lambda: self._step_scale(25))

        self.lbl_info = QLabel(f"{self.config.get('width', 1280)} x {self.config.get('height', 720)}")

        tb.addWidget(self.btn_fit)
        tb.addStretch()
        tb.addWidget(self.btn_zoom_out)
        tb.addWidget(self.scale_combo)
        tb.addWidget(self.btn_zoom_in)
        tb.addStretch()
        tb.addWidget(self.lbl_info)

        layout.addWidget(self._toolbar)

                # --- Web View ---
        self.webview = QWebEngineView()
        
        # ── Auto-adjust preview size to match configured render resolution ──
        config_w = self.config.get('width', 1280)
        config_h = self.config.get('height', 720)
        
        # Store base dimensions so load_html() can set the viewport correctly
        self._preview_base_width = config_w
        self._preview_base_height = config_h
        
        # Calculate how large the preview can be on the user's actual screen
        self.webview.setMinimumSize(320, 180)
            
        self._current_zoom = 1.0
        self._current_html = ""
        self._current_base_dir = ""
        self._original_webview_resize = self.webview.resizeEvent
        self.webview.resizeEvent = self._on_webview_resized

        self.webview.loadFinished.connect(self._on_load_finished)
        self.webview.setStyleSheet("background-color: #1a1a1a;")

        layout.addWidget(self.webview, stretch=1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_preview_resolution(self, width, height):
        """Called when the user changes the width/height in Settings.
        Updates the viewport, recalculates the CSS zoom, and re-renders."""
        self._preview_base_width = width
        self._preview_base_height = height
        
        # Re-apply zoom with new dimensions if HTML is loaded
        if hasattr(self, '_current_html') and self._current_html:
            self._apply_zoom_to_preview()

    def load_file(self, html_path: str):
        """Load an HTML slide file from disk into the preview."""
        self._base_dir = os.path.dirname(os.path.abspath(html_path))
        url = QUrl.fromLocalFile(os.path.abspath(html_path))
        self.webview.load(url)

    def load_html(self, html_content: str, base_dir: str):
        """Load an HTML string into the preview with a base URL for relative paths.
        
        Automatically injects a CSS zoom if the project resolution is larger
        than the available preview area, ensuring text remains readable.
        """
        self._base_dir = base_dir
        
        if os.path.isfile(base_dir):
            base_dir = os.path.dirname(base_dir)
        
        abs_dir = os.path.abspath(base_dir).replace(os.sep, '/') + '/'
        base_url = QUrl.fromLocalFile(abs_dir)
        
        # ── Auto-Zoom for High Resolutions ──
        zoom_css = ""
        if hasattr(self, '_preview_base_width') and hasattr(self, 'webview'):
            config_w = self._preview_base_width
            config_h = self._preview_base_height
            
            # Current actual size of the webview widget on screen
            actual_w = self.webview.width()
            
            if actual_w > 0 and config_w > 0:
                # Calculate how much we need to zoom in
                # e.g., if config is 3840 (4K) but widget is 960px wide, zoom = 4.0
                zoom_factor = config_w / actual_w
                
                if zoom_factor > 1.1:
                    # Inject a CSS transform to scale the content up
                    # This makes 4K text readable in a 960px preview window
                    zoom_css = f"""
                    <style id="preview-zoom-style">
                        html {{
                            transform: scale({zoom_factor:.4f});
                            transform-origin: top left;
                            width: {config_w}px !important;
                            height: {config_h}px !important;
                            overflow: hidden !important;
                        }}
                        body {{
                            width: {config_w}px !important;
                            height: {config_h}px !important;
                            overflow: hidden !important;
                        }}
                    </style>
                    """
            
            # Set internal viewport to actual project resolution (not widget size)
            try:
                page = self.webview.page()
                if page:
                    page.setViewportSize(QSize(config_w, config_h))
            except Exception:
                pass
        
        # Inject zoom CSS right before </head>
        if zoom_css and '</head>' in html_content:
            html_content = html_content.replace('</head>', zoom_css + '</head>', 1)
        
        self._current_html = html_content
        self._current_base_dir = base_dir

        self.webview.setHtml(html_content, base_url)

    def refresh_scale(self):
        """Re-apply current scale (call after content change or resize)."""
        if self._fit_mode:
            QTimer.singleShot(30, self.fit_to_screen)
        else:
            self._apply_zoom(self._scale_percent)

    def fit_to_screen(self):
        """Calculate the scale that fits the slide inside the preview area."""
        vw = self.webview.width()
        vh = self.webview.height()
        if vw <= 0 or vh <= 0:
            return
        scale_w = vw / self.config.get('width', 1280 )
        scale_h = vh / self.config.get('height', 720 )
        factor = min(scale_w, scale_h)
        percent = max(10, int(factor * 100))
        self._fit_mode = True
        self._scale_percent = percent
        self._apply_zoom(percent)
        self._update_combo_text("Fit")

    def set_scale(self, percent: int):
        """Set an exact scale percentage and exit fit mode."""
        percent = max(10, min(400, percent))
        self._fit_mode = False
        self._scale_percent = percent
        self._apply_zoom(percent)
        self._update_combo_text(f"{percent}%")

    def current_scale(self) -> int:
        return self._scale_percent

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _apply_zoom(self, percent: int):
        """Apply CSS zoom via transform:scale on the html element (same mechanism
        as _apply_zoom_to_preview) to avoid conflicts between the two zoom paths."""
        scale = percent / 100.0
        base_w = getattr(self, '_preview_base_width', 1280)
        base_h = getattr(self, '_preview_base_height', 720)
        js = f"""
        (function() {{
            document.body.style.zoom = '';
            var el = document.getElementById('preview-zoom-style');
            var css = 'html {{ transform: scale({scale:.4f}); transform-origin: top left; '
                    + 'width: {base_w}px !important; '
                    + 'height: {base_h}px !important; '
                    + 'overflow: hidden !important; }}'
                    + 'body {{ width: {base_w}px !important; '
                    + 'height: {base_h}px !important; '
                    + 'overflow: hidden !important; }}';
            if (!el) {{
                el = document.createElement('style');
                el.id = 'preview-zoom-style';
                document.head.appendChild(el);
            }}
            el.textContent = css;
        }})();
        """
        self.webview.page().runJavaScript(js)

    def _step_scale(self, delta: int):
        new_pct = max(10, min(400, self._scale_percent + delta))
        self.set_scale(new_pct)

    def _on_scale_text_changed(self, text: str):
        text = text.strip()
        if text.lower() == "fit":
            self.fit_to_screen()
            return
        clean = text.replace('%', '').strip()
        try:
            val = int(clean)
            self.set_scale(val)
        except ValueError:
            pass

    def _update_combo_text(self, text: str):
        self.scale_combo.blockSignals(True)
        self.scale_combo.setCurrentText(text)
        self.scale_combo.blockSignals(False)

    def _on_load_finished(self, ok):
        if ok:
            if self._fit_mode:
                QTimer.singleShot(80, self.fit_to_screen)
            else:
                self._apply_zoom(self._scale_percent)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._fit_mode:
            self.fit_to_screen()

    def _on_webview_resized(self, event):
        """Recalculate zoom when the webview widget is resized."""
        if self._original_webview_resize:
            self._original_webview_resize(event)

        # Recalculate and apply zoom
        self._apply_zoom_to_preview()
    
    def _calculate_preview_zoom(self):
        """Calculate how much to scale the content to fit the current widget size."""
        if not hasattr(self, '_preview_base_width') or not hasattr(self, 'webview'):
            return 1.0
        
        widget_w = self.webview.width()
        widget_h = self.webview.height()
        
        if widget_w <= 0 or widget_h <= 0:
            return 1.0
        if self._preview_base_width <= 0 or self._preview_base_height <= 0:
            return 1.0
        
        # Calculate scale to fit the configured resolution into the actual widget area
        # We use 97% to leave a tiny margin so scrollbars don't appear
        scale_w = (widget_w * 0.97) / self._preview_base_width
        scale_h = (widget_h * 0.97) / self._preview_base_height
        
        # Use the smaller scale to ensure the whole slide fits
        zoom = min(scale_w, scale_h)
        
        # Don't zoom beyond 100% (makes low-res slides look pixelated)
        return min(zoom, 1.0)
    
    def _apply_zoom_to_preview(self):
        """Apply the calculated zoom level to the currently loaded HTML."""
        if not hasattr(self, '_current_html') or not self._current_html:
            return
        
        zoom = self._calculate_preview_zoom()
        self._current_zoom = zoom
        
        # Use JavaScript to update the zoom transform in-place,
        # avoiding a full page reload on every resize event.
        js = f"""
        (function() {{
            var el = document.getElementById('preview-zoom-style');
            var css = 'html {{ transform: scale({zoom:.4f}); transform-origin: top left; '
                    + 'width: {self._preview_base_width}px !important; '
                    + 'height: {self._preview_base_height}px !important; '
                    + 'overflow: hidden !important; }}'
                    + 'body {{ width: {self._preview_base_width}px !important; '
                    + 'height: {self._preview_base_height}px !important; '
                    + 'overflow: hidden !important; }}';
            if (!el) {{
                el = document.createElement('style');
                el.id = 'preview-zoom-style';
                document.head.appendChild(el);
            }}
            el.textContent = css;
        }})();
        """
        try:
            page = self.webview.page()
            if page:
                page.runJavaScript(js)
                if hasattr(self, '_preview_base_width') and hasattr(self, '_preview_base_height'):
                    page.setViewportSize(QSize(self._preview_base_width, self._preview_base_height))
        except Exception:
            pass


# =================================================================
# SLIDE EDIT PANEL (Modular Sub-Component)
# =================================================================

class SlideEditPanel(QWidget):
    """Right panel: text narration editor and HTML source editor."""

    content_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._is_loading = False
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #333; border-radius: 4px;
                background: #1e1e1e;
            }
            QTabBar::tab {
                background: #2d2d2d; color: #aaa; padding: 6px 14px;
                border: 1px solid #333; border-bottom: none;
                border-top-left-radius: 4px; border-top-right-radius: 4px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: #1e1e1e; color: #fff;
                border-bottom: 2px solid #007acc;
            }
            QTabBar::tab:hover { background: #383838; }
        """)

        # --- Text Tab ---
        text_widget = QWidget()
        text_layout = QVBoxLayout(text_widget)
        text_layout.setContentsMargins(4, 4, 4, 4)

        text_hint = QLabel("Narration text — sent to TTS for audio generation")
        text_hint.setStyleSheet("color: #888; font-size: 11px; padding: 2px 4px;")
        text_hint.setWordWrap(True)
        text_layout.addWidget(text_hint)

        self.text_editor = QTextEdit()
        self.text_editor.setPlaceholderText(
            "Enter narration text for TTS...\n\n"
            "This text will be converted to speech during rendering."
        )
        self.text_editor.setStyleSheet("""
            QTextEdit {
                background: #1e1e1e; color: #d4d4d4;
                border: 1px solid #333; border-radius: 4px;
                font-size: 14px; padding: 8px;
            }
        """)
        self.text_editor.textChanged.connect(self._on_text_changed)
        text_layout.addWidget(self.text_editor)

        self.tabs.addTab(text_widget, "📝 Text")

        # --- HTML Tab ---
        html_widget = QWidget()
        html_layout = QVBoxLayout(html_widget)
        html_layout.setContentsMargins(4, 4, 4, 4)

        html_hint = QLabel("Slide HTML source — controls visual appearance")
        html_hint.setStyleSheet("color: #888; font-size: 11px; padding: 2px 4px;")
        html_hint.setWordWrap(True)
        html_layout.addWidget(html_hint)

        self.html_editor = QPlainTextEdit()
        self.html_editor.setPlaceholderText("Edit HTML content...")
        self.html_editor.setFont(QFont("Consolas", 10))
        self.html_editor.setStyleSheet("""
            QPlainTextEdit {
                background: #1e1e1e; color: #d4d4d4;
                border: 1px solid #333; border-radius: 4px;
                font-family: Consolas, monospace;
                font-size: 12px; padding: 8px;
            }
        """)
        self.html_editor.textChanged.connect(self._on_html_changed)
        html_layout.addWidget(self.html_editor)

        self.tabs.addTab(html_widget, "💻 HTML")

        layout.addWidget(self.tabs)

    # --- Suppress signals during programmatic content load ---

    def _on_text_changed(self):
        if not self._is_loading:
            self.content_changed.emit()

    def _on_html_changed(self):
        if not self._is_loading:
            self.content_changed.emit()

    def set_loading(self, loading: bool):
        self._is_loading = loading

    # --- Getters / Setters ---

    def get_text(self) -> str:
        return self.text_editor.toPlainText()

    def set_text(self, text: str):
        self._is_loading = True
        self.text_editor.setPlainText(text)
        self._is_loading = False

    def get_html(self) -> str:
        return self.html_editor.toPlainText()

    def set_html(self, html: str):
        self._is_loading = True
        self.html_editor.setPlainText(html)
        self._is_loading = False

    def clear(self):
        self._is_loading = True
        self.text_editor.clear()
        self.html_editor.clear()
        self._is_loading = False


# =================================================================
# SLIDE EDITOR TAB WIDGET (Main Orchestrator)
# =================================================================

class SlideEditorTabWidget(QWidget):
    """
    Modular slide editor with live preview.

    Architecture:
        ┌─────────────┬────────────────────────┬──────────────────┐
        │ SlideList   │  SlidePreviewPanel     │  SlideEditPanel  │
        │ Panel       │  (QWebEngineView +     │  (Text + HTML    │
        │ (left nav)  │   zoom controls)       │   editors)       │
        └─────────────┴────────────────────────┴──────────────────┘

    Features:
        - Debounced auto-save (800 ms after last keystroke)
        - Debounced preview refresh (500 ms after last HTML change)
        - Fit to Screen button with auto-refit on resize
        - Scale percentage control (combo + / - buttons)
        - Full CSS rendering (Tailwind CDN, Google Fonts, Material Icons)
        - Ctrl+S for immediate save
    """

    content_saved = Signal(str, int)  # project_path, slide_index

    AUTO_SAVE_DELAY_MS = 800
    PREVIEW_DELAY_MS = 500

    def __init__(self, project_path, config, parent=None):
        super().__init__(parent)
        self.project_path = project_path
        self.project_name = os.path.basename(project_path)
        self.config = config

        self.vid_width = int(self.config.get('width', 1280))
        self.vid_height = int(self.config.get('height', 720))
        self.slide_files = []
        self.current_index = -1
        self._dirty = False

        # ── Debounce Timers ──────────────────────────────────────
        self._auto_save_timer = QTimer(self)
        self._auto_save_timer.setSingleShot(True)
        self._auto_save_timer.setInterval(self.AUTO_SAVE_DELAY_MS)
        self._auto_save_timer.timeout.connect(self._on_auto_save_timeout)

        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(self.PREVIEW_DELAY_MS)
        self._preview_timer.timeout.connect(self._refresh_preview)

        self._setup_ui()
        self._connect_signals()
        self._load_project()

        # Ctrl+S shortcut
        self._save_shortcut = QShortcut(QKeySequence("Ctrl+S"), self)
        self._save_shortcut.activated.connect(self.save_current_slide)

        # Next/Previous Slide
        self._shortcut_next_slide = QShortcut(QKeySequence("Ctrl+F"), self)
        self._shortcut_next_slide.activated.connect(self._next_slide)

        self._shortcut_prev_slide = QShortcut(QKeySequence("Ctrl+G"), self)
        self._shortcut_prev_slide.activated.connect(self._prev_slide)

        # Add New Slide
        self._shortcut_add_slide = QShortcut(QKeySequence("Ctrl+Shift+N"), self)
        self._shortcut_add_slide.activated.connect(self._add_slide)
        self.refresh_from_disk()

    def refresh_from_disk(self):
        """Re-scan the project directory and reload all slides.
        Called on init and after PDF/image import."""
        if not hasattr(self, 'list_panel'):
            return
        self._load_project()
        if hasattr(self, 'list_panel') and self.list_panel.slide_list.count() > 0:
            self.list_panel.slide_list.setCurrentRow(0)

    # ------------------------------------------------------------------
    # UI Setup
    # ------------------------------------------------------------------

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Header Bar ───────────────────────────────────────────
        header = QFrame()
        header.setFixedHeight(34)
        header.setObjectName("editorHeader")
        header.setStyleSheet("""
            #editorHeader {
                background: #252526;
                border-bottom: 1px solid #3c3c3c;
            }
        """)
        hdr = QHBoxLayout(header)
        hdr.setContentsMargins(10, 0, 10, 0)

        lbl = QLabel(f"📝 {self.project_name}")
        lbl.setStyleSheet(
            "color: #ddd; font-weight: bold; font-size: 13px; "
            "background: transparent;")
        hdr.addWidget(lbl)
        hdr.addStretch()

        self.lbl_dirty = QLabel("✓ Saved")
        self.lbl_dirty.setStyleSheet(
            "color: #6a9955; font-size: 11px; background: transparent;")
        hdr.addWidget(self.lbl_dirty)

        layout.addWidget(header)

        # ── Main Splitter ────────────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setStyleSheet(
            "QSplitter::handle { background: #3c3c3c; width: 2px; }")

        self.list_panel = SlideListPanel()
        self.list_panel.setMinimumWidth(130)
        self.list_panel.setMaximumWidth(260)

        self.preview_panel = SlidePreviewPanel(self.config, self)
        self.preview_panel.setMinimumWidth(350)

        self.edit_panel = SlideEditPanel()
        self.edit_panel.setMinimumWidth(280)

        splitter.addWidget(self.list_panel)
        splitter.addWidget(self.preview_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        splitter.addWidget(self.edit_panel)
        splitter.setStretchFactor(2, 2)
        splitter.setSizes([150, 620, 380])

        layout.addWidget(splitter, stretch=1)

        # --- Import Buttons (add to toolbar area) ---
        self.btn_import_pdf = QPushButton("📄 Import PDF")
        self.btn_import_pdf.setToolTip("Import a PDF — each page becomes a slide")
        self.btn_import_pdf.clicked.connect(self.on_import_pdf)

        self.btn_add_image = QPushButton("🖼 Add Image")
        self.btn_add_image.setToolTip("Add an image as a new slide")
        self.btn_add_image.clicked.connect(self.on_add_image)

        layout.addWidget(self.btn_import_pdf)
        layout.addWidget(self.btn_add_image)

    def _connect_signals(self):
        # Slide list
        self.list_panel.slide_selected.connect(self._on_slide_selected)
        self.list_panel.btn_add.clicked.connect(self._add_slide)
        self.list_panel.btn_remove.clicked.connect(self._remove_slide)
        self.list_panel.btn_duplicate.clicked.connect(self._duplicate_slide)
        self.list_panel.btn_move_up.clicked.connect(self._move_slide_up)
        self.list_panel.btn_move_down.clicked.connect(self._move_slide_down)
        # Editor
        self.edit_panel.content_changed.connect(self._on_content_changed)

    # ------------------------------------------------------------------
    # Project Loading
    # ------------------------------------------------------------------

    def _load_project(self):
        """Scan project directory and populate the slide list."""
        pattern = os.path.join(self.project_path, "slide*.html")
        files = sorted(glob.glob(pattern), key=natural_sort_key)
        self.slide_files = files
        self.list_panel.load_slides(files)
        if files:
            self._on_slide_selected(0)

    # ------------------------------------------------------------------
    # Slide Selection
    # ------------------------------------------------------------------

    def _on_slide_selected(self, index):
        """Load the selected slide into editors and preview."""
        if index < 0 or index >= len(self.slide_files):
            return

        # Save previous if dirty
        if self._dirty and self.current_index >= 0 and self.current_index != index:
            self.save_current_slide()

        self.current_index = index
        html_path = self.slide_files[index]
        txt_path = os.path.splitext(html_path)[0] + ".txt"

        # Read text
        text_content = ""
        if os.path.exists(txt_path):
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    text_content = f.read()
            except Exception as e:
                logger.warning(f"Could not read {os.path.basename(txt_path)} ({type(e).__name__}: {e})", exc_info=True)

        # Read HTML
        html_content = ""
        try:
            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
        except Exception as e:
            logger.warning(f"Could not read {os.path.basename(html_path)} ({type(e).__name__}: {e})", exc_info=True)

        # Load into editors (signals suppressed)
        self.edit_panel.set_loading(True)
        self.edit_panel.set_text(text_content)
        self.edit_panel.set_html(html_content)
        self.edit_panel.set_loading(False)

        # Load into preview from file (ensures CDN resources load correctly)
        self.preview_panel.load_file(html_path)

        self._dirty = False
        self._update_dirty_indicator()

    # ------------------------------------------------------------------
    # Content Change & Auto-Save
    # ------------------------------------------------------------------

    def _on_content_changed(self):
        """Fired when text or HTML editor content changes."""
        self._dirty = True
        self._update_dirty_indicator()
        self._auto_save_timer.start()   # restart debounce
        self._preview_timer.start()     # restart preview debounce

    def _update_dirty_indicator(self):
        if self._dirty:
            self.lbl_dirty.setText("● Unsaved")
            self.lbl_dirty.setStyleSheet(
                "color: #f1c40f; font-size: 11px; background: transparent;")
        else:
            self.lbl_dirty.setText("✓ Saved")
            self.lbl_dirty.setStyleSheet(
                "color: #6a9955; font-size: 11px; background: transparent;")

    def _on_auto_save_timeout(self):
        if self._dirty:
            self.save_current_slide()

    def save_current_slide(self):
        """Save the current slide's text and HTML to disk."""
        if self.current_index < 0 or self.current_index >= len(self.slide_files):
            return

        html_path = self.slide_files[self.current_index]
        txt_path = os.path.splitext(html_path)[0] + ".txt"

        try:
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(self.edit_panel.get_text())
        except Exception as e:
            logger.error(f"Failed to save text ({type(e).__name__}: {e})", exc_info=True)

        try:
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(self.edit_panel.get_html())
        except Exception as e:
            logger.error(f"Failed to save HTML ({type(e).__name__}: {e})", exc_info=True)

        self._dirty = False
        self._update_dirty_indicator()
        self.content_saved.emit(self.project_path, self.current_index)

    # ------------------------------------------------------------------
    # Preview Refresh
    # ------------------------------------------------------------------

    def _refresh_preview(self):
        """Reload preview with the current HTML editor content (live preview)."""
        if self.current_index < 0:
            return
        html = self.edit_panel.get_html()
        if html.strip():
            self.preview_panel.load_html(html, self.project_path)

    # ------------------------------------------------------------------
    # Cleanup (called before tab close)
    # ------------------------------------------------------------------

    def save_and_cleanup(self):
        """Flush pending timers and save before tab removal."""
        self._auto_save_timer.stop()
        self._preview_timer.stop()
        if self._dirty:
            self.save_current_slide()

    # ------------------------------------------------------------------
    # Slide Management
    # ------------------------------------------------------------------

    def _next_slide(self):
        """Navigate to the next slide in the list."""
        idx = self.list_panel.current_index()
        if 0 <= idx < len(self.slide_files) - 1:
            self.list_panel.set_current_index(idx + 1)

    def _prev_slide(self):
        """Navigate to the previous slide in the list."""
        idx = self.list_panel.current_index()
        if idx > 0:
            self.list_panel.set_current_index(idx - 1)

    def _add_slide(self):
        next_num = len(self.slide_files) + 1

        # Find the first unused number to avoid collisions
        existing = set()
        for f in self.slide_files:
            m = re.search(r"slide(\d+)\.html", os.path.basename(f))
            if m:
                existing.add(int(m.group(1)))
        for n in range(1, next_num + 1):
            if n not in existing:
                next_num = n
                break

        html_path = os.path.join(self.project_path, f"slide{next_num}.html")
        txt_path = os.path.join(self.project_path, f"slide{next_num}.txt")

        try:
            create_slide_file(self.project_path, next_num, get_blank_slide_html(), "")
            self._load_project()
            if html_path in self.slide_files:
                self.list_panel.set_current_index(
                    self.slide_files.index(html_path))
        except Exception as e:
            logger.error(f"Failed to add slide ({type(e).__name__}: {e})", exc_info=True)

    def _remove_slide(self):
        if len(self.slide_files) <= 1:
            QMessageBox.information(
                self, "Cannot Remove",
                "Project must have at least one slide.")
            return

        idx = self.list_panel.current_index()
        if idx < 0:
            return

        reply = QMessageBox.question(
            self, "Remove Slide",
            f"Remove Slide {idx + 1}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        html_path = self.slide_files[idx]
        txt_path = os.path.splitext(html_path)[0] + ".txt"
        wav_path = os.path.splitext(html_path)[0] + ".wav"

        try:
            for p in [html_path, txt_path, wav_path]:
                if os.path.exists(p):
                    os.remove(p)
            self._load_project()
        except Exception as e:
            logger.error(f"Failed to remove slide ({type(e).__name__}: {e})", exc_info=True)

    def _duplicate_slide(self):
        idx = self.list_panel.current_index()
        if idx < 0:
            return

        src_html = self.slide_files[idx]
        src_txt = os.path.splitext(src_html)[0] + ".txt"

        # Find next available number
        next_num = len(self.slide_files) + 1
        existing = set()
        for f in self.slide_files:
            m = re.search(r"slide(\d+)\.html", os.path.basename(f))
            if m:
                existing.add(int(m.group(1)))
        for n in range(1, next_num + 1):
            if n not in existing:
                next_num = n
                break

        dst_html = os.path.join(self.project_path, f"slide{next_num}.html")
        dst_txt = os.path.join(self.project_path, f"slide{next_num}.txt")

        try:
            shutil.copy(src_html, dst_html)
            if os.path.exists(src_txt):
                shutil.copy(src_txt, dst_txt)
            else:
                with open(dst_txt, 'w', encoding='utf-8') as f:
                    f.write("")
            self._load_project()
            if dst_html in self.slide_files:
                self.list_panel.set_current_index(
                    self.slide_files.index(dst_html))
        except Exception as e:
            logger.error(f"Failed to duplicate slide ({type(e).__name__}: {e})", exc_info=True)

    def _move_slide_up(self):
        idx = self.list_panel.current_index()
        if idx <= 0:
            return
        self._swap_slide_content(idx, idx - 1)

    def _move_slide_down(self):
        idx = self.list_panel.current_index()
        if idx < 0 or idx >= len(self.slide_files) - 1:
            return
        self._swap_slide_content(idx, idx + 1)

    def _swap_slide_content(self, a, b):
        """Swap the file contents of two slides (preserves filename order).

        Swaps HTML, text (.txt), AND audio (.wav) files so that
        reordering slides keeps narration in sync with visuals.
        """
        ha = self.slide_files[a]
        hb = self.slide_files[b]
        ta = os.path.splitext(ha)[0] + ".txt"
        tb = os.path.splitext(hb)[0] + ".txt"
        wa = os.path.splitext(ha)[0] + ".wav"
        wb = os.path.splitext(hb)[0] + ".wav"

        try:
            # ── Swap HTML ──
            with open(ha, 'r', encoding='utf-8') as f:
                ca = f.read()
            with open(hb, 'r', encoding='utf-8') as f:
                cb = f.read()
            with open(ha, 'w', encoding='utf-8') as f:
                f.write(cb)
            with open(hb, 'w', encoding='utf-8') as f:
                f.write(ca)

            # ── Swap text ──
            tca = ""
            tcb = ""
            if os.path.exists(ta):
                with open(ta, 'r', encoding='utf-8') as f:
                    tca = f.read()
            if os.path.exists(tb):
                with open(tb, 'r', encoding='utf-8') as f:
                    tcb = f.read()
            with open(ta, 'w', encoding='utf-8') as f:
                f.write(tcb)
            with open(tb, 'w', encoding='utf-8') as f:
                f.write(tca)

            # ── Swap audio (.wav) ──
            wca = None
            wcb = None
            if os.path.exists(wa):
                with open(wa, 'rb') as f:
                    wca = f.read()
            if os.path.exists(wb):
                with open(wb, 'rb') as f:
                    wcb = f.read()
            if wca is not None and wcb is not None:
                with open(wa, 'wb') as f:
                    f.write(wcb)
                with open(wb, 'wb') as f:
                    f.write(wca)
            elif wca is not None:
                # Only slide A had audio: move it to B, delete A's
                with open(wb, 'wb') as f:
                    f.write(wca)
                os.remove(wa)
            elif wcb is not None:
                # Only slide B had audio: move it to A, delete B's
                with open(wa, 'wb') as f:
                    f.write(wcb)
                os.remove(wb)

            self._load_project()
            self.list_panel.set_current_index(b)
        except Exception as e:
            logger.error(f"Failed to swap slides ({type(e).__name__}: {e})", exc_info=True)

    # =================================================================
    # PDF & IMAGE IMPORT
    # =================================================================

    def on_import_pdf(self):
        """Import a PDF file, appending each page as a new slide."""
        pdf_path, _ = QFileDialog.getOpenFileName(
            self, "Import PDF", "",
            "PDF Files (*.pdf);;All Files (*)"
        )
        if not pdf_path:
            return
        
        try:
            import fitz  # Early check before modifying project
        except ImportError:
            QMessageBox.critical(
                self, "Missing Dependency",
                "PyMuPDF is required for PDF import.\n\n"
                "Install via: pip install PyMuPDF"
            )
            return
        
        # Confirm with user
        reply = QMessageBox.question(
            self, "Import PDF",
            f"Import {os.path.basename(pdf_path)}?\n\n"
            f"Each page will be appended as a new slide.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        try:
            config = self.config.settings if hasattr(self.config, 'settings') else self.config
            count = import_pdf_append(pdf_path, self.project_path, config)
            QMessageBox.information(
                self, "PDF Imported",
                f"Successfully imported {count} slide(s) from PDF."
            )
            # Refresh the slide list in the editor
            self._refresh_slide_list()
        except Exception as e:
            logger.error(f"PDF import failed ({type(e).__name__}: {e})", exc_info=True)
            QMessageBox.critical(self, "Import Error", f"Failed to import PDF:\n{type(e).__name__}: {e}")

    def on_add_image(self):
        """Add an image file as a new slide."""
        image_path, _ = QFileDialog.getOpenFileName(
            self, "Add Image Slide", "",
            "Image Files (*.png *.jpg *.jpeg *.gif *.bmp *.webp *.tiff);;All Files (*)"
        )
        if not image_path:
            return
        
        try:
            config = self.config.settings if hasattr(self.config, 'settings') else self.config
            html_path = import_image_as_slide(image_path, self.project_path, config=config)
            slide_num = get_next_slide_number(self.project_path) - 1  # The one just created
            QMessageBox.information(
                self, "Image Added",
                f"Added image as slide {slide_num}."
            )
            self._refresh_slide_list()
        except Exception as e:
            logger.error(f"Image import failed ({type(e).__name__}: {e})", exc_info=True)
            QMessageBox.critical(self, "Import Error", f"Failed to add image slide:\n{type(e).__name__}: {e}")

    def _refresh_slide_list(self):
        """Refresh the slide list widget to show newly imported slides."""
        current_idx = self.list_panel.slide_list.currentRow()
        self._load_project()
        if 0 <= current_idx < self.list_panel.slide_list.count():
            self.list_panel.slide_list.setCurrentRow(current_idx)
        else:
            self.list_panel.slide_list.setCurrentRow(0)