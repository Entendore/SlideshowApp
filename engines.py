# engines.py
"""
Render engine — worker process that drives the full pipeline:
scan → audio → timeline → video → mux → manifest.

Runs in a subprocess; communicates with the main GUI via
typed dataclass messages on a multiprocessing.Queue.

IMPROVED:
- Direct image reading via Pillow for PDF/PPTX slides (bypasses Playwright).
- Relays structured log messages to the GUI via QueueLogHandler.
- Per-phase timing and consistent [WORKER {name}] prefixes.
"""

from __future__ import annotations

import asyncio
import glob
import hashlib
import io
import json
import logging
import math
import multiprocessing
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from backends import get_backend, get_all_audio_settings_keys
from text_chunker import TextChunker, ChunkingConfig, preprocess_for_tts
from config import AppConfig

from utils import (
    build_manifest_from_timeline, detect_ffmpeg, generate_silent_wav,
    human_bytes, fmt_duration, VerboseFilter,
    is_image_slide, load_project_manifest, natural_sort_key, save_project_manifest,
    update_library_manifest, BackendGenerationError, ChunkProgressMessage,
    ErrorMessage, FFmpegError, FinalizedMessage, LogMessage, PhaseMessage,
    RenderPhase, RenderStatus, SlideProgressMessage, VideoRenderError,
    get_phase_progress,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SlideTask:
    """Represents one slide's processing state in the render pipeline."""
    index: int
    html_path: str
    txt_path: str
    target_audio_path: str
    text_content: str = ""
    source_audio_path: Optional[str] = None
    needs_tts: bool = False
    is_silent: bool = False
    silent_duration: float = 5.0
    generation_error: Optional[str] = None

    _SLIDE_NUM_RE = re.compile(r"slide(\d+)\.html")

    @property
    def slide_number(self) -> int:
        match = self._SLIDE_NUM_RE.search(Path(self.html_path).name)
        return int(match.group(1)) if match else self.index


@dataclass
class RenderContext:
    """Immutable context carried through the entire pipeline."""
    project_path: str
    temp_dir: str
    config: dict
    ffmpeg_path: str
    render_mode: str


# ═══════════════════════════════════════════════════════════════════════════════
#  QUEUE LOG HANDLER  (relays worker logs to the GUI)
# ═══════════════════════════════════════════════════════════════════════════════

class QueueLogHandler(logging.Handler):
    """Sends structured LogMessage objects through the multiprocessing queue
    so that important worker log entries appear in the GUI log widget.

    IMPROVED: Now tracks the current render phase and includes it in
    LogMessage for phase-aware coloring in the GUI log panel.
    """

    _INFO_WHITELIST: tuple[str, ...] = (
        "Phase", "Partition:", "Smart Cache", "Decision]",
        "Generating", "Playwright", "FFmpeg finished",
        "Manifest saved", "Done.", "Direct image",
        "Slide", "TTS backend reused", "Silent audio",
    )

    def __init__(
        self, msg_queue, project_name: str, min_level: int = logging.WARNING
    ):
        super().__init__()
        self._queue = msg_queue
        self._project_name = project_name
        self.setLevel(logging.DEBUG)
        self._min_level = min_level
        self._current_phase: str = ""   # tracked by engine, set via set_phase()

    def set_phase(self, phase: str) -> None:
        """Update the current phase tag included in subsequent log messages."""
        self._current_phase = phase

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if record.levelno >= self._min_level:
                self._send(record)
                return
            if record.levelno >= logging.INFO:
                msg_text = record.getMessage()
                for pattern in self._INFO_WHITELIST:
                    if pattern in msg_text:
                        self._send(record)
                        return
        except Exception:
            self.handleError(record)

    def _send(self, record: logging.LogRecord) -> None:
        level_name = logging.getLevelName(record.levelno)
        self._queue.put(LogMessage(
            project_name=self._project_name,
            level=level_name,
            text=record.getMessage(),
            phase=self._current_phase,
        ))


def _emit_phase(
    msg_queue, project_name: str, phase: str,
    queue_log_handler: Optional[QueueLogHandler] = None,
    pipeline_start: float = 0.0,
    label: str = "",
    progress_pct: int = -1,
) -> None:
    """Emit a PhaseMessage with auto-resolved label/progress and elapsed time.

    Also updates the QueueLogHandler's tracked phase so subsequent log
    entries carry the phase tag for the GUI.
    """
    if queue_log_handler is not None:
        queue_log_handler.set_phase(str(phase))

    if label == "" or progress_pct == -1:
        resolved_label, start, end = get_phase_progress(str(phase))
        if label == "":
            label = resolved_label
        if progress_pct == -1:
            progress_pct = (start + end) // 2

    elapsed = time.time() - pipeline_start if pipeline_start > 0 else 0.0

    msg_queue.put(PhaseMessage(
        project_name=project_name,
        phase=str(phase),
        label=label,
        progress_pct=progress_pct,
        elapsed=elapsed,
    ))


# ═══════════════════════════════════════════════════════════════════════════════
#  FFMPEG COMMAND BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

class FFmpegCommandBuilder:
    """Builds FFmpeg commands for the video encoding pipeline."""

    _BITRATE_TIERS: list[tuple[int, tuple[str, str, str]]] = [
        (3840, ("16000k", "16000k", "32000k")),
        (2560, ("12000k", "12000k", "24000k")),
        (1920, ("8000k",  "8000k",  "16000k")),
        (1280, ("5000k",  "5000k",  "10000k")),
        (854,  ("3000k",  "3000k",  "6000k")),
        (0,    ("1500k",  "1500k",  "3000k")),
    ]

    @classmethod
    def get_bitrate_settings(cls, width: int, height: int = 0) -> tuple[str, str, str]:
        long_edge = max(width, height)
        for threshold, rates in cls._BITRATE_TIERS:
            if long_edge >= threshold:
                return rates
        return cls._BITRATE_TIERS[-1][1]

    @staticmethod
    def detect_encoder(ffmpeg_path: str) -> str:
        try:
            # Verify an NVIDIA GPU is actually available before selecting nvenc
            subprocess.run(
                ["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                check=True, timeout=10,
            )
            r = subprocess.run(
                [ffmpeg_path, "-encoders"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                timeout=10,
            )
            if "nvenc" in r.stdout or "nvenc" in r.stderr:
                return "h264_nvenc"
        except Exception as exc:
            logger.debug(f"GPU encoder detection failed: {exc}")
        return "libx264"

    @staticmethod
    def build_video_encoder(
        ffmpeg_path: str,
        output_path: str,
        width: int,
        height: int,
        fps: int,
        encoder: str = "auto",
        preset: str = "fast",
    ) -> list[str]:
        from config import AppConfig
        width, height = AppConfig.sanitize_resolution(width, height)

        if encoder == "auto":
            encoder = FFmpegCommandBuilder.detect_encoder(ffmpeg_path)

        gop = max(2, int(fps / 2))
        target_rate, max_rate, buf_size = (
            FFmpegCommandBuilder.get_bitrate_settings(width, height)
        )

        return [
            ffmpeg_path, "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
            "-c:v", encoder, "-preset", preset,
            "-pix_fmt", "yuv420p",
            "-g", str(gop), "-keyint_min", str(gop),
            "-sc_threshold", "0",
            "-b:v", target_rate, "-maxrate", max_rate,
            "-bufsize", buf_size,
            "-movflags", "+faststart",
            output_path,
        ]


# ═══════════════════════════════════════════════════════════════════════════════
#  HASHING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def _get_audio_settings_hash(config: dict) -> str:
    keys_to_hash = get_all_audio_settings_keys()
    data = {k: config.get(k) for k in keys_to_hash}
    for ref_key in ("qwen3_ref_audio", "omnivoice_ref_audio"):
        ref_audio_path = config.get(ref_key, "")
        if ref_audio_path and os.path.exists(ref_audio_path):
            data[f"{ref_key}_mtime"] = os.path.getmtime(ref_audio_path)
    return hashlib.md5(
        json.dumps(data, sort_keys=True).encode()
    ).hexdigest()


def _get_video_settings_hash(config: dict) -> str:
    keys_to_hash = [
        "width", "height", "fps", "encoder",
        "preset", "transition_duration",
    ]
    data = {k: config.get(k) for k in keys_to_hash}
    return hashlib.md5(
        json.dumps(data, sort_keys=True).encode()
    ).hexdigest()


def _compute_thumbnail_content_hash(
    html_path: str, project_path: str
) -> str:
    """Compute an MD5 hash over the first slide's content sources.

    Covers the HTML file content and (if present) the direct source image
    file size + mtime.  Stored in ``manifest["thumbnail"]["hash"]``
    and used to skip thumbnail regeneration when the slide has not changed.
    """
    h = hashlib.md5()
    # Hash HTML content
    try:
        with open(html_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
    except OSError:
        pass
    # Hash direct image metadata (size + mtime) for image slides
    try:
        direct_img = _get_direct_image_path(html_path, project_path)
        if direct_img and os.path.exists(direct_img):
            st = os.stat(direct_img)
            h.update(f"img:{st.st_size}:{st.st_mtime}".encode())
    except Exception:
        pass
    return h.hexdigest()


# ═══════════════════════════════════════════════════════════════════════════════
#  PIPELINE STAGES
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_slide_tasks(
    html_files: List[str], ctx: RenderContext
) -> List[SlideTask]:
    """Phase 1: scan HTML files, determine which need TTS regeneration."""
    project_name = os.path.basename(ctx.project_path)
    manifest = _load_project_manifest(ctx.project_path)

    # Extract audio hash
    _r = manifest.get("render", {})
    prev_audio_hash = _r.get("audio_hash", "")
    current_audio_hash = _get_audio_settings_hash(ctx.config)
    settings_changed = (
        prev_audio_hash != "" and current_audio_hash != prev_audio_hash
    )

    if settings_changed:
        logger.info(
            f"[WORKER {project_name}] Audio settings changed — "
            f"regenerating all audio"
        )
    else:
        logger.info(
            f"[WORKER {project_name}] Audio settings unchanged — "
            f"checking file timestamps"
        )

    enable_chunking = ctx.config.get("enable_text_chunking", True)
    chunk_max_chars = ctx.config.get("chunk_max_chars", 500)
    warn_threshold = ctx.config.get("chunk_warn_threshold", 1000)
    silent_duration = ctx.config.get("silent_slide_duration", 5.0)

    _lang_map = {
        "qwen3": "qwen3_language",
        "edge_tts": "edge_language",
        "omnivoice": "omnivoice_language",
    }
    _active = ctx.config.get("active_backend", "qwen3")
    _lang_key = _lang_map.get(_active, "qwen3_language")

    _chunk_crossfade_scan = int(ctx.config.get("chunk_crossfade_ms", 30))
    chunker = TextChunker(ChunkingConfig(
        max_chars=chunk_max_chars,
        max_sentences=ctx.config.get("chunk_max_sentences", 5),
        min_chunk_chars=ctx.config.get("chunk_min_chars", 50),
        overlap_chars=ctx.config.get("chunk_overlap_chars", 30),
        language=ctx.config.get(_lang_key, "English"),
        chunk_strategy=ctx.config.get("chunk_strategy", "sentence"),
        max_duration_seconds=ctx.config.get("chunk_max_duration_seconds", 0.0),
        crossfade_enabled=_chunk_crossfade_scan > 0,
    )) if enable_chunking else None

    tasks: list[SlideTask] = []
    long_text_slides: list[tuple[int, int]] = []

    for html_file in html_files:
        p_html = Path(html_file)
        p_txt = p_html.with_suffix(".txt")
        slide_num_match = re.search(r"slide(\d+)\.html", p_html.name)
        slide_num = (
            int(slide_num_match.group(1)) if slide_num_match else 0
        )

        target_audio = Path(ctx.temp_dir) / f"a{len(tasks)}.wav"
        text_content = (
            p_txt.read_text(encoding="utf-8").strip()
            if p_txt.exists() else ""
        )
        _is_image = is_image_slide(str(p_html))

        if not text_content:
            if _is_image:
                _is_silent = True
                logger.info(
                    f"[WORKER {project_name}] Slide {slide_num}: Image-only "
                    f"(duration: {fmt_duration(silent_duration)})"
                )
            else:
                text_content = (
                    f"This is Slide {slide_num} with no text."
                )
                _is_silent = False
        else:
            _is_silent = False

        if not _is_silent and text_content:
            text_length = len(text_content)
            if text_length > warn_threshold:
                long_text_slides.append((slide_num, text_length))
                if chunker and chunker.needs_chunking(text_content):
                    chunks = chunker.chunk_text(
                        text_content,
                        ctx.config.get(_lang_key, "English"),
                    )
                    logger.info(
                        f"[WORKER {project_name}] Slide {slide_num}: "
                        f"{text_length} chars → {len(chunks)} TTS segment(s) "
                        f"(chunk_max_chars={chunk_max_chars})"
                    )
                elif not enable_chunking:
                    logger.warning(
                        f"[WORKER {project_name}] Slide {slide_num}: "
                        f"{text_length} chars exceeds threshold but "
                        f"chunking is DISABLED — generation may fail"
                    )

        source_audio = Path(ctx.project_path) / f"slide{slide_num}.wav"
        needs_tts = _determine_needs_tts(
            _is_silent, settings_changed, str(source_audio),
            str(p_txt), slide_num, project_name,
        )

        tasks.append(SlideTask(
            index=len(tasks),
            html_path=str(p_html),
            txt_path=str(p_txt),
            target_audio_path=str(target_audio),
            text_content=text_content,
            source_audio_path=str(source_audio),
            needs_tts=needs_tts,
            is_silent=_is_silent,
            silent_duration=silent_duration,
        ))

    if long_text_slides:
        logger.info(
            f"[WORKER {project_name}] {len(long_text_slides)} slide(s) "
            f"exceed {warn_threshold} chars"
        )
        if not enable_chunking:
            logger.warning(
                f"[WORKER {project_name}] Text chunking is DISABLED — "
                f"long slides may fail"
            )

    logger.info(
        f"[WORKER {project_name}] Phase 1 complete: "
        f"{len(tasks)} slide(s) scanned, "
        f"{sum(1 for t in tasks if t.needs_tts)} need TTS, "
        f"{sum(1 for t in tasks if t.is_silent)} silent/image"
    )
    if long_text_slides and enable_chunking:
        logger.info(
            f"[WORKER {project_name}] Note: {len(long_text_slides)} long "
            f"slide(s) will be split into multiple TTS segments during "
            f"audio generation (chunk_max_chars={chunk_max_chars})"
        )
    return tasks


def _determine_needs_tts(
    is_silent: bool,
    settings_changed: bool,
    source_audio_path: str,
    txt_path: str,
    slide_num: int,
    project_name: str = "",
) -> bool:
    if is_silent:
        return False
    if settings_changed:
        return True
    if not os.path.exists(source_audio_path):
        logger.info(
            f"[WORKER {project_name}] Slide {slide_num}: "
            f"No cached audio — needs TTS"
        )
        return True
    try:
        if os.path.getmtime(txt_path) > os.path.getmtime(source_audio_path):
            logger.info(
                f"[WORKER {project_name}] Slide {slide_num}: "
                f"Text modified — regenerating audio"
            )
            return True
        logger.info(
            f"[WORKER {project_name}] Slide {slide_num}: "
            f"Cached audio is current — reusing"
        )
        return False
    except Exception as exc:
        logger.warning(
            f"[WORKER {project_name}] Slide {slide_num}: "
            f"mtime check failed ({type(exc).__name__}: {exc}) — regenerating",
            exc_info=True,
        )
        return True


# ═══════════════════════════════════════════════════════════════════════════════
#  PERSISTENT TTS BACKEND CACHE
# ═══════════════════════════════════════════════════════════════════════════════
# Module-level state that survives across multiple projects within the
# same worker process.  The model is loaded once and reused until the
# backend name or its config hash changes.

_cached_backend = None

_cached_backend_name: str = ""
_cached_backend_hash: str = ""


def _compute_backend_config_hash(backend_name: str, config: dict) -> str:
    """Return a stable hash of the settings that affect model loading.

    Only keys relevant to the active backend are included so that
    unrelated changes (e.g. video resolution) do NOT trigger a reload.
    """
    try:
        audio_keys = get_all_audio_settings_keys()
    except Exception:
        audio_keys = [k for k in config if any(
            tag in k for tag in (
                "backend", "qwen3", "edge", "omnivoice", "voice",
                "speaker", "language", "dtype", "device", "attn",
                "model_id", "chunk", "hf_cache", "hf_use_sym",
            )
        )]
    hash_dict: Dict[str, Any] = {"active_backend": backend_name}
    for key in sorted(audio_keys):
        if key in config:
            hash_dict[key] = config[key]
    raw = json.dumps(hash_dict, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _get_cached_backend(backend_name: str, config: dict):
    """Return a TTS backend, reusing the cached instance when possible.

    The caller must NOT call ``backend.cleanup()`` — that is handled
    by ``_release_cached_backend()`` when the worker process exits.
    """
    global _cached_backend, _cached_backend_name, _cached_backend_hash

    new_hash = _compute_backend_config_hash(backend_name, config)

    if (
        _cached_backend is not None
        and _cached_backend_name == backend_name
        and _cached_backend_hash == new_hash
    ):
        logger.info(
            f"TTS backend reused ({backend_name}, "
            f"hash={new_hash[:8]}) — no load time"
        )
        return _cached_backend

    # Need to load or reload
    if _cached_backend is not None:
        logger.info(
            f"TTS backend settings changed "
            f"({_cached_backend_name}/{_cached_backend_hash[:8]} -> "
            f"{backend_name}/{new_hash[:8]}) — reloading model"
        )
        try:
            _cached_backend.cleanup()
        except Exception:
            pass

    load_start = time.time()
    logger.info(
        f"Loading TTS backend: {backend_name} "
        f"(config hash: {new_hash[:8]})"
    )
    _cached_backend = get_backend(backend_name, config)
    _cached_backend_name = backend_name
    _cached_backend_hash = new_hash
    logger.info(
        f"TTS backend loaded in {fmt_duration(time.time() - load_start)}"
    )
    return _cached_backend


def _release_cached_backend() -> None:
    """Release the cached TTS backend (called on worker shutdown)."""
    global _cached_backend, _cached_backend_name, _cached_backend_hash
    if _cached_backend is not None:
        logger.info("Releasing cached TTS backend")
        try:
            _cached_backend.cleanup()
        except Exception:
            pass
        _cached_backend = None
        _cached_backend_name = ""
        _cached_backend_hash = ""


def _normalize_audio_to_wav(
    audio_path: str,
    target_path: str,
    sample_rate: int = 24000,
    channels: int = 1,
    ffmpeg_path: str = "ffmpeg",
) -> bool:
    """Normalize an audio file to a consistent WAV format.

    Ensures all audio clips have the same sample rate and channel count
    before concatenation, preventing FFmpeg concat demuxer failures
    and audible artifacts from format mismatches.

    Args:
        audio_path: Source audio file (any format FFmpeg can read).
        target_path: Output WAV file path.
        sample_rate: Target sample rate (default 24000 Hz, common TTS output).
        channels: Target channel count (1=mono, 2=stereo).
        ffmpeg_path: Path to the FFmpeg binary.

    Returns:
        True if normalization succeeded, False otherwise.
    """
    try:
        cmd = [
            ffmpeg_path, "-y",
            "-i", audio_path,
            "-ar", str(sample_rate),
            "-ac", str(channels),
            "-sample_fmt", "s16",
            target_path,
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            logger.warning(
                f"Audio normalization failed for {audio_path}: "
                f"{result.stderr[:200]}"
            )
            return False
        return True
    except Exception as exc:
        logger.warning(
            f"Audio normalization error for {audio_path}: "
            f"{type(exc).__name__}: {exc}"
        )
        return False


def _get_wav_info(wav_path: str) -> Optional[Tuple[int, int, int]]:
    """Read WAV file metadata: (sample_rate, channels, sample_width).

    Returns None if the file cannot be read or is not a valid WAV.
    """
    try:
        import wave
        with wave.open(wav_path, "r") as w:
            return (w.getframerate(), w.getnchannels(), w.getsampwidth())
    except Exception:
        return None


def _get_wav_duration(wav_path: str) -> float:
    """Return the duration of a WAV file in seconds.

    Returns 0.0 if the file cannot be read.
    """
    try:
        import wave
        with wave.open(wav_path, "r") as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  AUDIO POST-PROCESSING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _post_process_clip(
    audio_path: str,
    ffmpeg_path: str,
    project_name: str = "",
    slide_num: int = 0,
    is_chunk_clip: bool = False,
) -> bool:
    """Apply silence trim, fade in/out, and (optionally) loudness normalization.

    Runs a two-pass FFmpeg pipeline:
      Pass 1 (single filter chain):
        1. ``silenceremove`` — strip leading/trailing TTS silence
        2. ``afade t=in`` — fade in to eliminate click at start
        3. ``loudnorm`` — EBU R128 per-clip volume normalization
           (skipped for chunk clips — a single loudnorm is applied after concat)
        4. ``aformat`` — guarantee consistent sample rate / channels
      Pass 2 (non-chunk clips only):
        5. ``afade t=out`` — fade out at the CORRECT end position
           (implemented as a second pass because the clip duration is
           only known after silence removal in pass 1; ``st=0`` would
           incorrectly silence the entire clip)

    For chunk clips (``is_chunk_clip=True``), uses tighter silence
    thresholds (-35 dB) and shorter safety durations to minimise
    residual dead air between chunks.  Both ``loudnorm`` and fade-out
    are omitted: loudnorm is applied once on the concatenated result,
    and crossfade handles the inter-chunk transition.

    If the filter chain fails (e.g. clip is too short for fades),
    the original file is left untouched and the function returns False.

    Returns True on success.
    """
    _fade_dur = 0.04  # 40 ms fade

    if is_chunk_clip:
        # Tighter thresholds: strip more silence to reduce gap artifacts
        # at chunk boundaries.  Shorter safety durations because we
        # know these clips will be concatenated.
        filter_chain = (
            "silenceremove="
            "start_periods=1:start_duration=0.08:start_threshold=-35dB:"
            "stop_periods=1:stop_duration=0.10:stop_threshold=-35dB,"
            f"afade=t=in:st=0:d={_fade_dur},"
            # NOTE: afade=out with st=0 means "start fading out at
            # time 0" which effectively silences the entire clip.
            # FFmpeg requires knowing the duration to set st correctly.
            # We use a two-pass approach: first get duration, then fade.
            # However, since silenceremove already stripped trailing silence,
            # the clip is tight.  We omit the out-fade for chunk clips
            # because crossfade handles the transition.
            "aformat=sample_rates=24000:channel_layouts=mono"
        )
    else:
        filter_chain = (
            "silenceremove="
            "start_periods=1:start_duration=0.15:start_threshold=-40dB:"
            "stop_periods=1:stop_duration=0.2:stop_threshold=-40dB,"
            f"afade=t=in:st=0:d={_fade_dur},"
            # For non-chunk clips, we also omit the out-fade here and
            # instead apply it as a second pass after getting the duration.
            # This avoids the st=0 bug that would silence the whole clip.
            "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=summary,"
            "aformat=sample_rates=24000:channel_layouts=mono"
        )
    tmp_path = audio_path + ".pp.wav"
    try:
        cmd = [
            ffmpeg_path, "-y",
            "-i", audio_path,
            "-af", filter_chain,
            tmp_path,
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
        if result.returncode == 0 and os.path.exists(tmp_path):
            # Apply fade-out as a second pass for non-chunk clips.
            # afade=t=out:st=0 is WRONG (silences entire clip),
            # so we need the actual post-processing duration first.
            if not is_chunk_clip:
                try:
                    dur = _get_wav_duration(tmp_path)
                    if dur > 2 * _fade_dur:
                        fade_out_start = max(0, dur - _fade_dur)
                        fade_tmp = tmp_path + ".fade.wav"
                        fade_cmd = [
                            ffmpeg_path, "-y",
                            "-i", tmp_path,
                            "-af", f"afade=t=out:st={fade_out_start:.3f}:d={_fade_dur}",
                            fade_tmp,
                        ]
                        fr = subprocess.run(
                            fade_cmd,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE,
                            text=True,
                            timeout=30,
                        )
                        if fr.returncode == 0 and os.path.exists(fade_tmp):
                            os.replace(fade_tmp, tmp_path)
                        else:
                            try:
                                os.unlink(fade_tmp)
                            except OSError:
                                pass
                except Exception:
                    pass  # fade-out is non-critical; proceed without it
            os.replace(tmp_path, audio_path)
            return True
        # Filter failed (clip too short, etc.) — leave original
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        return False
    except Exception as exc:
        logger.debug(
            f"[WORKER {project_name}] Post-process slide {slide_num} "
            f"skipped ({type(exc).__name__}: {exc})"
        )
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except OSError:
            pass
        return False


def _loudnorm_clip(
    audio_path: str,
    ffmpeg_path: str,
    project_name: str = "",
    slide_num: int = 0,
) -> bool:
    """Apply EBU R128 loudness normalization to a single audio file.

    Used after concatenating chunk clips so all chunks share one
    loudness target instead of having per-chunk volume jumps.

    Returns True on success.
    """
    filter_chain = (
        "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=summary,"
        "aformat=sample_rates=24000:channel_layouts=mono"
    )
    tmp_path = audio_path + ".ln.wav"
    try:
        cmd = [
            ffmpeg_path, "-y",
            "-i", audio_path,
            "-af", filter_chain,
            tmp_path,
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
        if result.returncode == 0 and os.path.exists(tmp_path):
            os.replace(tmp_path, audio_path)
            return True
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        return False
    except Exception as exc:
        logger.debug(
            f"[WORKER {project_name}] Loudnorm slide {slide_num} "
            f"skipped ({type(exc).__name__}: {exc})"
        )
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except OSError:
            pass
        return False


def _concat_chunk_audio(
    chunk_paths: List[str],
    output_path: str,
    ffmpeg_path: str,
    crossfade_ms: int = 0,
) -> bool:
    """Concatenate multiple chunk audio files into a single file.

    If *crossfade_ms* > 0 and there are 2+ chunks, uses FFmpeg's
    ``acrossfade`` filter for a smooth crossfade between chunks instead
    of hard splicing.  For a single chunk, simply copies the file.

    Returns True on success.
    """
    if not chunk_paths:
        return False
    if len(chunk_paths) == 1:
        try:
            shutil.copy(chunk_paths[0], output_path)
            return True
        except OSError:
            return False

    try:
        if crossfade_ms > 0:
            # ── True crossfade via filter_complex chain ─────────────
            inputs: list[str] = []
            for p in chunk_paths:
                inputs.extend(["-i", p])

            dur = crossfade_ms / 1000.0
            parts: list[str] = []
            prev = "[0]"
            for i in range(1, len(chunk_paths)):
                out = (f"[cf{i}]" if i < len(chunk_paths) - 1
                       else "[aout]")
                parts.append(
                    f"{prev}[{i}]acrossfade=d={dur}:c1=tri:c2=tri{out}"
                )
                prev = out

            cmd = [
                ffmpeg_path, "-y",
                *inputs,
                "-filter_complex", ";".join(parts),
                "-map", "[aout]",
                output_path,
            ]
        else:
            # ── Simple concat demuxer (no crossfade) ───────────────
            listfile = output_path + ".concat.txt"
            with open(listfile, "w", encoding="utf-8") as f:
                for p in chunk_paths:
                    escaped = (os.path.abspath(p).replace(os.sep, "/")
                               .replace("'", "\\'"))
                    f.write(f"file '{escaped}'\n")
            cmd = [
                ffmpeg_path, "-y",
                "-f", "concat", "-safe", "0",
                "-i", listfile,
                "-c", "copy",
                output_path,
            ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            logger.warning(
                f"Chunk concat failed: {result.stderr[:200]}"
            )
            return False

        # Clean up concat listfile (only exists for non-crossfade path)
        if crossfade_ms <= 0:
            listfile = output_path + ".concat.txt"
            if os.path.exists(listfile):
                try:
                    os.unlink(listfile)
                except OSError:
                    pass
        return True
    except Exception as exc:
        logger.warning(f"Chunk concat error: {type(exc).__name__}: {exc}")
        return False


def process_audio_tasks(
    tasks: List[SlideTask], ctx: RenderContext, msg_queue,
    queue_log_handler: Optional[QueueLogHandler] = None,
    pipeline_start: float = 0.0,
) -> None:
    """Phase 2: copy reusable audio, generate silent audio, run TTS backend.

    Uses the persistent backend cache so the TTS model stays loaded in
    memory across multiple projects rendered by the same worker process.

    IMPROVED:
    - Text chunking is wired into the TTS pipeline: long slides are
      split by the engine's chunker, each chunk is generated separately,
      and the resulting audio files are concatenated with crossfade.
    - Per-slide retry (up to 3 attempts) on TTS failure.
    - Post-processing pipeline per clip: silence trim → fade in/out →
      loudness normalization (EBU R128) → format normalization.
    """
    project_name = os.path.basename(ctx.project_path)

    inter_slide_gap_ms = int(ctx.config.get("inter_slide_gap_ms", 0))

    # ── Chunker setup (for engine-side text splitting) ─────────────
    enable_chunking = ctx.config.get("enable_text_chunking", True)
    _lang_map = {
        "qwen3": "qwen3_language",
        "edge_tts": "edge_language",
        "omnivoice": "omnivoice_language",
    }
    _active = ctx.config.get("active_backend", "qwen3")
    _lang_key = _lang_map.get(_active, "qwen3_language")
    _language = ctx.config.get(_lang_key, "English")

    if enable_chunking:
        _chunk_crossfade = int(ctx.config.get("chunk_crossfade_ms", 30))
        chunker = TextChunker(ChunkingConfig(
            max_chars=ctx.config.get("chunk_max_chars", 500),
            max_sentences=ctx.config.get("chunk_max_sentences", 5),
            min_chunk_chars=ctx.config.get("chunk_min_chars", 50),
            overlap_chars=ctx.config.get("chunk_overlap_chars", 30),
            language=_language,
            chunk_strategy=ctx.config.get("chunk_strategy", "sentence"),
            max_duration_seconds=ctx.config.get("chunk_max_duration_seconds", 0.0),
            crossfade_enabled=_chunk_crossfade > 0,
        ))
    else:
        chunker = None

    # ── Partition tasks ────────────────────────────────────────────
    tasks_to_copy = [
        t for t in tasks
        if t.source_audio_path
        and os.path.exists(t.source_audio_path)
        and not t.needs_tts
        and not t.is_silent
    ]
    tasks_to_generate = [t for t in tasks if t.needs_tts]
    tasks_to_silence = [
        t for t in tasks
        if t.is_silent and not os.path.exists(t.target_audio_path)
    ]

    logger.info(
        f"[WORKER {project_name}] Partition: "
        f"{len(tasks_to_copy)} copy, "
        f"{len(tasks_to_generate)} generate, "
        f"{len(tasks_to_silence)} silent"
    )

    # ── Parallel copy & silent generation (I/O-bound) ────────────
    _io_workers = min(4, max(1, (len(tasks_to_copy) + len(tasks_to_silence))))

    def _do_copy(task):
        shutil.copy(task.source_audio_path, task.target_audio_path)

    def _do_silent(task):
        generate_silent_wav(task.target_audio_path, task.silent_duration)

    if tasks_to_copy or tasks_to_silence:
        io_start = time.time()
        with ThreadPoolExecutor(max_workers=_io_workers) as pool:
            future_map = {}
            for task in tasks_to_copy:
                future_map[pool.submit(_do_copy, task)] = task
            for task in tasks_to_silence:
                future_map[pool.submit(_do_silent, task)] = task

            for fut in as_completed(future_map):
                task = future_map[fut]
                try:
                    fut.result()
                    if task.is_silent:
                        logger.info(
                            f"[WORKER {project_name}] Silent audio "
                            f"({fmt_duration(task.silent_duration)}) "
                            f"for slide {task.slide_number}"
                        )
                except Exception as exc:
                    logger.error(
                        f"[WORKER {project_name}] Audio I/O failed for slide "
                        f"{task.slide_number} ({type(exc).__name__}: {exc})",
                        exc_info=True,
                    )
                    task.generation_error = str(exc)
        logger.info(
            f"[WORKER {project_name}] Audio I/O complete in "
            f"{time.time() - io_start:.2f}s "
            f"({len(tasks_to_copy)} copy, {len(tasks_to_silence)} silent, "
            f"{_io_workers} workers)"
        )

    # Check for any audio preparation failures before proceeding
    failed = [t for t in tasks if t.generation_error and not os.path.exists(t.target_audio_path)]
    if failed:
        raise BackendGenerationError(
            f"Audio preparation failed for {len(failed)} slide(s): "
            + "; ".join(f"Slide {t.slide_number}: {t.generation_error}" for t in failed)
        )

    if not tasks_to_generate:
        logger.info(
            f"[WORKER {project_name}] No TTS generation needed — "
            f"all audio ready"
        )
        if inter_slide_gap_ms > 0 or tasks_to_copy:
            _normalize_all_audio(tasks, ctx.ffmpeg_path, project_name)
        return

    # ══════════════════════════════════════════════════════════════
    #  TTS GENERATION  (chunker-wired, with retry)
    # ══════════════════════════════════════════════════════════════
    backend = None
    try:
        backend_name = ctx.config.get("active_backend", "qwen3")
        backend = _get_cached_backend(backend_name, ctx.config)
        logger.info(
            f"[WORKER {project_name}] TTS backend ready: {backend_name}"
        )

        # NOTE: backends may send "Generating slide X/Y" where Y is the
        # total TTS segment count (including chunks), NOT the slide count.
        # We rewrite the message to avoid confusing the user.
        def progress_callback(
            stage: str, current: int, total: int, message: str = ""
        ):
            # Sanitise backend messages that say "slide" when they mean
            # "TTS segment"  (only during the AUDIO_GENERATING phase).
            safe_msg = message
            if message:
                safe_msg = re.sub(
                    r'(?i)\bslide\b', 'segment', message,
                )
            msg_queue.put(ChunkProgressMessage(
                project_name=project_name,
                current=current, total=total, message=safe_msg,
                slide_count=len(tasks_to_generate),
            ))

        if hasattr(backend, 'set_progress_callback'):
            backend.set_progress_callback(progress_callback)

        # ── Build flat batch: preprocess + chunk long texts ─────
        # Every slide's text is preprocessed for TTS (HTML cleanup,
        # abbreviation expansion, URL handling, etc.).  Then long
        # slides are additionally chunked by the engine chunker.
        flat_texts: List[str] = []
        flat_paths: List[str] = []
        # (task_index_in_tasks_to_generate, chunk_index, total_chunks)
        chunk_map: List[Tuple[int, int, int]] = []
        multi_chunk_tasks: set[int] = set()
        _pp_total = 0  # total preprocessing changes across all slides

        for ti, task in enumerate(tasks_to_generate):
            # ── Step 1: Preprocess for TTS ─────────────────────────
            cleaned, pp_stats = preprocess_for_tts(
                task.text_content, _language,
                _overrides=chunker.config.effective_pronunciation_overrides
                if chunker else None,
            )
            if pp_stats.total_changes > 0:
                _pp_total += pp_stats.total_changes
                logger.debug(
                    f"[WORKER {project_name}] Slide "
                    f"{task.slide_number}: preprocessed "
                    f"({pp_stats.total_changes} change(s))"
                )
            # Update task's text with the cleaned version so
            # the pipeline (caching, logging) uses it.
            task.text_content = cleaned

            # ── Step 2: Chunk if needed ──────────────────────────
            if (chunker
                    and chunker.needs_chunking(cleaned)):
                chunks = chunker.chunk_text(cleaned, _language,
                                               preprocess=False)
                for ci, (ct, _) in enumerate(chunks):
                    cp = os.path.join(
                        ctx.temp_dir,
                        f"chunk_{task.slide_number}_{ci}.wav",
                    )
                    flat_texts.append(ct)
                    flat_paths.append(cp)
                    chunk_map.append((ti, ci, len(chunks)))
                multi_chunk_tasks.add(ti)
                logger.info(
                    f"[WORKER {project_name}] Slide {task.slide_number}: "
                    f"{len(cleaned)} chars → {len(chunks)} TTS segment(s)"
                )
            else:
                flat_texts.append(cleaned)
                flat_paths.append(task.target_audio_path)
                chunk_map.append((ti, 0, 1))

        if _pp_total:
            logger.info(
                f"[WORKER {project_name}] TTS preprocessing: "
                f"{_pp_total} total change(s) across "
                f"{len(tasks_to_generate)} slide(s)"
            )

        # ── Chunking summary ────────────────────────────────────
        n_segments = len(flat_texts)
        n_slides = len(tasks_to_generate)
        n_multi = len(multi_chunk_tasks)
        if n_multi:
            # Build a compact per-slide breakdown
            _detail_parts = []
            for j, (ti, ci, ct) in enumerate(chunk_map):
                if ct > 1:
                    slide_num = tasks_to_generate[ti].slide_number
                    _detail_parts.append(f"#{slide_num}→{ct}seg")
            logger.info(
                f"[WORKER {project_name}] Text chunking: "
                f"{n_slides} slide(s) expanded to {n_segments} TTS segment(s)  "
                f"({n_multi} slide(s) chunked: {', '.join(_detail_parts)})"
            )

        total_to_generate = n_slides
        tts_start = time.time()

        _emit_phase(msg_queue, project_name, RenderPhase.AUDIO_GENERATING,
                     queue_log_handler=queue_log_handler,
                     pipeline_start=pipeline_start)

        logger.info(
            f"[WORKER {project_name}] Generating audio: "
            f"{n_segments} TTS segment(s) for {n_slides} slide(s)…"
        )

        # Build a per-slide segment map so the backend can log
        # "segment X/Y (slide N)" with X/Y scoped to each slide.
        # Each entry: (slide_number, segment_index_1based, total_segments_for_slide)
        _seg_slide_map = [
            (tasks_to_generate[ti].slide_number, ci + 1, ct)
            for ti, ci, ct in chunk_map
        ]

        # ── Generate with per-slide retry (max 3 attempts) ──────
        _MAX_RETRIES = 3
        _remaining_indices = list(range(len(flat_texts)))
        _attempt = 0

        while _attempt < _MAX_RETRIES and _remaining_indices:
            _attempt += 1
            _try_texts = [flat_texts[i] for i in _remaining_indices]
            _try_paths = [flat_paths[i] for i in _remaining_indices]

            # Inject mapping for the backend's log messages
            backend.config["_segment_slide_map"] = [
                _seg_slide_map[i] for i in _remaining_indices
            ]
            # The engine already chunked long texts into segments,
            # so disable the backend's own chunking to avoid double-
            # splitting (which inflates segment counts and produces
            # confusing "segment 1/N" per-slide counters).
            _backend_chunking_was = backend.config.get(
                "enable_text_chunking", True
            )
            backend.config["enable_text_chunking"] = False

            _success, _errors = backend.generate_batch(
                _try_texts, _try_paths
            )

            # Restore backend's original chunking setting
            backend.config["enable_text_chunking"] = _backend_chunking_was

            if _success:
                _remaining_indices = []
                break

            # Identify which flat entries failed
            _still_failing = []
            for j, err in enumerate(_errors):
                if err:
                    _still_failing.append(_remaining_indices[j])

            if _attempt < _MAX_RETRIES and _still_failing:
                logger.warning(
                    f"[WORKER {project_name}] TTS attempt "
                    f"{_attempt}/{_MAX_RETRIES}: "
                    f"{len(_still_failing)} segment(s) failed — retrying"
                )
                _remaining_indices = _still_failing
            else:
                # Final attempt or no retries left — collect errors
                # mapped back to tasks
                _error_messages: list[str] = []
                for j, err in enumerate(_errors):
                    if err:
                        fi = _remaining_indices[j] if _remaining_indices else j
                        ti = chunk_map[fi][0]
                        task = tasks_to_generate[ti]
                        task.generation_error = err
                        _error_messages.append(
                            f"Slide {task.slide_number}: {err}"
                        )
                if _error_messages:
                    raise BackendGenerationError(
                        f"Audio generation failed after {_MAX_RETRIES} "
                        f"attempt(s): {'; '.join(_error_messages)}"
                    )

        tts_elapsed = time.time() - tts_start
        tts_time_str = fmt_duration(tts_elapsed)

        logger.info(
            f"[WORKER {project_name}] Audio generation complete: "
            f"{n_segments} TTS segment(s) for {total_to_generate} slide(s) "
            f"in {tts_time_str}"
            + (f" (retry: {_attempt} attempt(s))" if _attempt > 1 else "")
        )

        msg_queue.put(SlideProgressMessage(
            project_name=project_name,
            current=total_to_generate, total=total_to_generate,
        ))

        # Clean up injected mapping
        backend.config.pop("_segment_slide_map", None)

        # ── Post-process individual CHUNK clips (tight silence, no loudnorm) ──
        # Chunk clips get tighter silenceremove (-35 dB) but NO loudnorm.
        # Loudnorm is applied once on the concatenated result to avoid
        # volume jumps between chunks.
        # Also detect clips that become zero-length after aggressive
        # silence removal (e.g., a chunk that was mostly silence).
        if multi_chunk_tasks:
            _chunk_pp_start = time.time()
            _chunk_pp_count = 0
            _dead_chunks: list[tuple[int, int]] = []  # (chunk_map_index, task_index)
            for j, (ti, ci, ct) in enumerate(chunk_map):
                if ct <= 1:
                    continue  # not a multi-chunk slide
                chunk_path = flat_paths[j]
                if not os.path.exists(chunk_path):
                    continue
                if _post_process_clip(
                    chunk_path, ctx.ffmpeg_path,
                    project_name=project_name,
                    slide_num=tasks_to_generate[ti].slide_number,
                    is_chunk_clip=True,
                ):
                    _chunk_pp_count += 1
                # Detect zero-length or near-zero-length clips after trimming
                try:
                    fsize = os.path.getsize(chunk_path)
                    # WAV header is 44 bytes; anything under 200 bytes is
                    # effectively empty audio (< 0.01s at 24kHz mono 16-bit).
                    if fsize < 200:
                        logger.warning(
                            f"[WORKER {project_name}] Chunk clip "
                            f"{os.path.basename(chunk_path)} is near-empty "
                            f"({fsize} bytes) after silence removal — "
                            f"marking for removal"
                        )
                        _dead_chunks.append((j, ti))
                except OSError:
                    pass
            # Remove dead chunk files so they don't break concat
            for j, ti in _dead_chunks:
                dp = flat_paths[j]
                try:
                    os.unlink(dp)
                except OSError:
                    pass
                # Replace with a minimal silent clip so concat still works
                generate_silent_wav(dp, 0.1)
            if _dead_chunks:
                logger.info(
                    f"[WORKER {project_name}] Replaced {_dead_chunks} "
                    f"near-empty chunk clip(s) with 0.1s silence"
                )
            if _chunk_pp_count:
                logger.info(
                    f"[WORKER {project_name}] Pre-concat post-processed "
                    f"{_chunk_pp_count} chunk clip(s) in "
                    f"{time.time() - _chunk_pp_start:.2f}s "
                    f"(tight trim + fade, no loudnorm)"
                )

        # ── Concatenate multi-chunk slides ────────────────────────
        _chunk_crossfade_ms = ctx.config.get("chunk_crossfade_ms", 30)
        for ti in multi_chunk_tasks:
            task = tasks_to_generate[ti]
            cp = [flat_paths[j]
                  for j, (t, _, _) in enumerate(chunk_map) if t == ti]
            logger.info(
                f"[WORKER {project_name}] Merging "
                f"{len(cp)} TTS segment(s) back into slide "
                f"{task.slide_number}"
            )
            if not _concat_chunk_audio(
                cp, task.target_audio_path, ctx.ffmpeg_path,
                crossfade_ms=_chunk_crossfade_ms,
            ):
                # Fallback: first chunk only
                if cp and os.path.exists(cp[0]):
                    shutil.copy(cp[0], task.target_audio_path)

            # Clean up chunk temp files
            for p in cp:
                try:
                    os.unlink(p)
                except OSError:
                    pass

        # ── Single loudnorm pass on concatenated multi-chunk results ──
        # Apply EBU R128 loudness normalization once on each merged slide
        # audio so all chunks share a single loudness target — no volume
        # jumps at chunk boundaries.
        if multi_chunk_tasks:
            _ln_start = time.time()
            _ln_count = 0
            for ti in multi_chunk_tasks:
                task = tasks_to_generate[ti]
                if not os.path.exists(task.target_audio_path):
                    continue
                if _loudnorm_clip(
                    task.target_audio_path, ctx.ffmpeg_path,
                    project_name, task.slide_number,
                ):
                    _ln_count += 1
            if _ln_count:
                logger.info(
                    f"[WORKER {project_name}] Post-concat loudnorm on "
                    f"{_ln_count} merged slide(s) in "
                    f"{time.time() - _ln_start:.2f}s"
                )

        # ── Cache generated audio to project directory ────────────
        for task in tasks_to_generate:
            try:
                if (task.source_audio_path
                        and os.path.exists(task.target_audio_path)):
                    shutil.copy(
                        task.target_audio_path, task.source_audio_path
                    )
            except Exception as exc:
                logger.warning(
                    f"[WORKER {project_name}] Could not cache audio "
                    f"for slide {task.slide_number} "
                    f"({type(exc).__name__}: {exc})",
                    exc_info=True,
                )

    except BackendGenerationError:
        raise
    except Exception as exc:
        logger.error(
            f"[WORKER {project_name}] Critical TTS backend error: {exc}",
            exc_info=True,
        )
        for task in tasks_to_generate:
            task.generation_error = str(exc)
        msg_queue.put(ErrorMessage(
            project_name=project_name,
            error_text=f"TTS backend error: {exc}",
        ))
        raise BackendGenerationError(str(exc)) from exc
    finally:
        if backend and hasattr(backend, "_cleanup_chunk_temp"):
            try:
                backend._cleanup_chunk_temp()
            except Exception as exc:
                logger.warning(
                    f"[WORKER {project_name}] Chunk temp cleanup "
                    f"error ({type(exc).__name__}: {exc})",
                    exc_info=True,
                )

    # ══════════════════════════════════════════════════════════════
    #  POST-PROCESSING PIPELINE (non-chunked clips only)
    # ══════════════════════════════════════════════════════════════
    # Multi-chunk slides were already post-processed above (chunk clips
    # got tight silence trim before concat, then a single loudnorm after
    # concat).  Here we only process slides that were NOT chunked.
    # Per-clip: silence trim → fade in/out → loudnorm
    # Then format normalization for concat compatibility.
    _multi_chunk_set = {tasks_to_generate[ti] for ti in multi_chunk_tasks}
    _pp_start = time.time()
    _pp_count = 0
    for task in tasks:
        if task.is_silent or not os.path.exists(task.target_audio_path):
            continue
        # Skip multi-chunk slides — already post-processed above
        if task in _multi_chunk_set:
            continue
        if _post_process_clip(
            task.target_audio_path, ctx.ffmpeg_path,
            project_name, task.slide_number,
        ):
            _pp_count += 1
    if _pp_count:
        logger.info(
            f"[WORKER {project_name}] Post-processed {_pp_count} "
            f"clip(s) in {time.time() - _pp_start:.2f}s "
            f"(trim + fade + loudnorm)"
        )

    # Format normalization (sample rate / channels) for concat
    _normalize_all_audio(tasks, ctx.ffmpeg_path, project_name)


def _normalize_all_audio(
    tasks: List[SlideTask], ffmpeg_path: str, project_name: str
) -> None:
    """Normalize all task audio files to a consistent WAV format.

    Detects the most common sample rate among existing clips, then
    re-encodes any outliers to match.  This prevents concat demuxer
    failures caused by format mismatches between cached and fresh audio.
    """
    # Step 1: Survey existing audio formats
    sample_rates: Dict[int, int] = {}
    channel_counts: Dict[int, int] = {}

    for task in tasks:
        if not os.path.exists(task.target_audio_path):
            continue
        info = _get_wav_info(task.target_audio_path)
        if info:
            sr, ch, _ = info
            sample_rates[sr] = sample_rates.get(sr, 0) + 1
            channel_counts[ch] = channel_counts.get(ch, 0) + 1

    if not sample_rates:
        return  # No audio to normalize

    # Pick the most common format as the target
    target_sr = max(sample_rates, key=sample_rates.get)
    target_ch = max(channel_counts, key=channel_counts.get) if channel_counts else 1

    # Check if normalization is needed
    all_match = all(
        not os.path.exists(t.target_audio_path)
        or _get_wav_info(t.target_audio_path) == (target_sr, target_ch, 2)
        for t in tasks
    )
    if all_match:
        logger.debug(
            f"[WORKER {project_name}] All audio clips already match "
            f"target format ({target_sr}Hz, {target_ch}ch) — skipping normalization"
        )
        return

    # Step 2: Normalize outliers
    normalized_count = 0
    for task in tasks:
        if not os.path.exists(task.target_audio_path):
            continue
        info = _get_wav_info(task.target_audio_path)
        if info and info[0] == target_sr and info[1] == target_ch and info[2] == 2:
            continue  # Already matches

        # Create a temp file, normalize, then replace
        tmp_path = task.target_audio_path + ".norm.wav"
        if _normalize_audio_to_wav(
            task.target_audio_path, tmp_path,
            sample_rate=target_sr, channels=target_ch,
            ffmpeg_path=ffmpeg_path,
        ):
            try:
                os.replace(tmp_path, task.target_audio_path)
                normalized_count += 1
            except OSError as exc:
                logger.warning(
                    f"[WORKER {project_name}] Could not replace normalized "
                    f"audio for slide {task.slide_number}: {exc}"
                )
                # Clean up temp file
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        else:
            # Clean up temp file if normalization failed
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except OSError:
                pass

    if normalized_count > 0:
        logger.info(
            f"[WORKER {project_name}] Normalized {normalized_count} audio "
            f"clip(s) to {target_sr}Hz/{target_ch}ch"
        )


def build_timeline(tasks: List[SlideTask]) -> List[dict]:
    """Build a timeline entry for each slide with computed duration."""
    timeline: list[dict] = []
    for task in tasks:
        duration = _compute_slide_duration(task)

        direct_image_path = None
        if task.is_silent and is_image_slide(task.html_path):
            try:
                with open(task.html_path, "r", encoding="utf-8") as f:
                    html_content = f.read()
                match = re.search(r'<img[^>]+src=["\']([^"\']+)["\']', html_content, re.IGNORECASE)
                if match:
                    img_rel = match.group(1)
                    direct_image_path = os.path.normpath(
                        os.path.join(
                            os.path.dirname(task.html_path), img_rel
                        )
                    )
            except Exception as exc:
                logger.warning(
                    f"Could not parse direct image path from HTML "
                    f"({type(exc).__name__}: {exc})",
                    exc_info=True,
                )

        timeline.append({
            "html": task.html_path,
            "audio": task.target_audio_path,
            "duration": duration,
            "direct_image": direct_image_path,
        })
    return timeline


def _compute_slide_duration(task: SlideTask) -> float:
    if task.is_silent and not Path(task.target_audio_path).exists():
        return task.silent_duration
    if Path(task.target_audio_path).exists():
        try:
            import wave
            with wave.open(task.target_audio_path, "r") as w:
                return max(0.1, w.getnframes() / float(w.getframerate()))
        except Exception as exc:
            logger.warning(
                f"Could not read WAV duration for {task.target_audio_path} "
                f"({type(exc).__name__}: {exc})",
                exc_info=True,
            )
    if task.is_silent:
        return task.silent_duration
    # CJK-aware fallback: use char count for languages without word separators
    _text = task.text_content
    _word_count = len(re.sub(r'\s', '', _text)) / 3 if re.search(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uAC00-\uD7AF]', _text) else len(_text.split())
    return max(3.0, _word_count / 2.5)


# ═══════════════════════════════════════════════════════════════════════════════
#  WORKER ENTRY POINTS
# ═══════════════════════════════════════════════════════════════════════════════

def _reset_worker_logger(msg_queue, project_name: str, config: dict = None) -> 'QueueLogHandler':
    """(Re-)configure the root logger for a worker process.

    Removes any previous QueueLogHandler, adds a fresh one using
    ``gui_level`` from config (default WARNING).  Also re-applies
    the VerboseFilter based on the current ``verbose`` setting.

    Console and file handlers from the parent process are left intact
    (they survive across forks on Unix, and on Windows each worker
    re-imports utils which calls setup_logging again).
    """
    from utils import LogConfig, _is_our_handler, _console_formatter

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Remove any previous QueueLogHandler (persistent worker renders
    # multiple projects — each gets its own handler with the right name).
    for handler in root.handlers[:]:
        if isinstance(handler, QueueLogHandler):
            root.removeHandler(handler)

    # ── Console handler (add if missing, e.g. fresh worker on Windows) ─
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(logging.Formatter(
            "[%(process)d] [%(asctime)s.%(msecs)03d] [%(levelname)-8s] %(message)s",
            datefmt="%H:%M:%S",
        ))
        root.addHandler(ch)

    # ── Suppress noisy debug messages ──────────────────────────────
    _NOISY_LOGGERS = [
        "PIL.PngImagePlugin", "PIL.Image", "asyncio",
        "urllib3.connectionpool", "httpx", "httpcore",
        "PIL", "py.warnings",
    ]
    verbose = config.get("verbose", False) if config else False

    for name in _NOISY_LOGGERS:
        log = logging.getLogger(name)
        log.filters = [f for f in log.filters if not isinstance(f, VerboseFilter)]

    if not verbose:
        filter_inst = VerboseFilter()
        for name in _NOISY_LOGGERS:
            logging.getLogger(name).addFilter(filter_inst)

    # ── GUI log handler ─────────────────────────────────────────────
    gui_level = logging.WARNING
    if config:
        raw = config.get("log_gui_level", "WARNING")
        if isinstance(raw, str):
            gui_level = getattr(logging, raw.upper(), logging.WARNING)
        else:
            gui_level = int(raw)

    qh = QueueLogHandler(msg_queue, project_name, min_level=gui_level)
    qh.setLevel(logging.DEBUG)
    root.addHandler(qh)
    return qh


def _setup_worker_environment(config: dict) -> None:
    """One-time environment setup (HF cache, symlinks)."""
    cache_dir = config.get(
        "hf_cache_dir", os.path.join(os.getcwd(), "models_cache")
    )
    os.environ["HF_HOME"] = cache_dir
    os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    os.environ["HF_DATASETS_CACHE"] = os.path.join(cache_dir, "datasets")
    if not config.get("hf_use_symlinks", False):
        os.environ["HF_HUB_SYMLINKS"] = "0"
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    # IMPROVE: Consistent with app.py — disable hf_transfer in workers
    # too, since the CLI tool may not be installed in all environments.
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"


def render_worker_loop(
    work_queue: multiprocessing.Queue,
    msg_queue: multiprocessing.Queue,
    config: dict,
) -> None:
    """Persistent render worker — processes multiple projects sequentially.

    Stays alive across renders so the TTS model (cached at module level
    in ``_get_cached_backend``) is only loaded once.  Only exits on a
    poison-pill (``None``) message — workers are **permanent** and are
    respawned by the GUI if they crash (OOM, segfault, etc.).

    Memory management between renders:
    - Runs ``gc.collect()`` + ``torch.cuda.empty_cache()`` (if available)
      after each project to release transient CPU/GPU allocations while
      keeping the TTS model resident in VRAM.
    - Refreshes config from disk before each render so live setting
      changes (backend, voice, encoder, etc.) are picked up without
      restarting the pool.
    """
    _setup_worker_environment(config)
    # No idle timeout — workers are permanent.  Block forever until
    # a project arrives or a poison-pill (None) is sent.
    idle_timeout = None

    pid = os.getpid()
    logger.info(
        f"[Worker {pid}] Persistent worker started (permanent, no idle timeout)"
    )

    # Import signal only on supported platforms
    try:
        import signal
        _current_project: Optional[str] = None

        def _crash_handler(sig, frame):
            """Last-resort handler: report which project we were working on."""
            try:
                pname = os.path.basename(_current_project) if _current_project else "unknown"
                msg_queue.put(ErrorMessage(
                    project_name=pname,
                    error_text=f"Worker PID {pid} killed by signal {sig} while rendering '{pname}'",
                ))
                msg_queue.put(FinalizedMessage(project_name=pname))
            except Exception:
                pass
            # Re-raise to get the default kill behaviour
            signal.signal(sig, signal.SIG_DFL)
            os.kill(os.getpid(), sig)

        signal.signal(signal.SIGTERM, _crash_handler)
        signal.signal(signal.SIGINT, _crash_handler)
    except Exception:
        _current_project = None  # type: ignore[no-redef]

    render_count = 0

    def _release_transient_memory() -> None:
        """Free transient CPU/GPU memory between renders.

        The TTS model stays loaded (cached by _get_cached_backend), but
        intermediate tensors, FFmpeg pipes, Pillow images, etc. are
        released.  On CUDA systems this reclaims fragmented VRAM so the
        next render starts with a clean GPU heap.
        """
        import gc
        collected = gc.collect()
        torch_freed = 0
        try:
            import torch
            if torch.cuda.is_available():
                torch_freed = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
                torch.cuda.empty_cache()
                torch_freed = max(0, torch_freed)
        except Exception:
            pass
        logger.debug(
            f"[Worker {pid}] Memory cleanup: "
            f"gc={collected} objects, cuda_cache_freed≈{torch_freed / 1048576:.0f} MB"
        )

    try:
        while True:
            # ── Wait for next project (or shutdown) ─────────────────
            try:
                work_item = work_queue.get(timeout=idle_timeout)
            except queue.Empty:
                break  # Only reachable if idle_timeout is not None
            except Exception as exc:
                logger.warning(
                    f"[Worker {pid}] Unexpected queue error "
                    f"({type(exc).__name__}: {exc}) — shutting down",
                    exc_info=True,
                )
                break

            if work_item is None:
                logger.info(f"[Worker {pid}] Poison pill — shutting down")
                break

            project_path = work_item
            project_name = os.path.basename(project_path)

            # Track current project for crash handler
            if '_current_project' in dir():
                _current_project = project_path  # type: ignore[name-defined]

            # ── Refresh config from disk (pick up live changes) ─────
            try:
                fresh_cfg = AppConfig()
                merged = config.copy()
                merged.update(fresh_cfg.settings)
                # Preserve worker-only keys that aren't in the saved config
                for k in ("ffmpeg_path",):
                    if k in config and k not in merged:
                        merged[k] = config[k]
                render_config = merged
            except Exception as exc:
                render_config = config
                logger.warning(
                    f"[Worker {pid}] Config refresh failed "
                    f"({type(exc).__name__}: {exc}) — using startup config",
                    exc_info=True,
                )

            qh = _reset_worker_logger(msg_queue, project_name, render_config)
            render_count += 1
            logger.info(
                f"[WORKER {project_name}] Starting render "
                f"(PID {pid}, render #{render_count}, model cached)"
            )

            try:
                _render_single_project(
                    project_path, msg_queue, render_config, queue_log_handler=qh
                )
            except Exception as exc:
                logger.error(
                    f"[WORKER {project_name}] Unhandled error: {exc}",
                    exc_info=True,
                )
                try:
                    msg_queue.put(ErrorMessage(
                        project_name=project_name,
                        error_text=str(exc),
                    ))
                except Exception:
                    pass
                msg_queue.put(FinalizedMessage(project_name=project_name))

            if '_current_project' in dir():
                _current_project = None  # type: ignore[name-defined]

            # ── Release transient memory (keep TTS model loaded) ────
            _release_transient_memory()

            logger.info(
                f"[WORKER {project_name}] Render finished — "
                f"model kept in memory for next project "
                f"(worker total: {render_count} renders)"
            )
    finally:
        _release_cached_backend()
        logger.info(
            f"[Worker {pid}] Exited after {render_count} render(s)"
        )


def render_project_worker(
    project_path: str,
    msg_queue: multiprocessing.Queue,
    config: dict,
) -> str:
    """Single-project render entry point.

    Used by thumbnail generation and other one-off tasks that don't
    benefit from model persistence.
    """
    _setup_worker_environment(config)
    project_name = os.path.basename(project_path)
    qh = _reset_worker_logger(msg_queue, project_name, config)
    return _render_single_project(project_path, msg_queue, config, queue_log_handler=qh)
def _render_single_project(
    project_path: str,
    msg_queue: multiprocessing.Queue,
    config: dict,
    queue_log_handler: Optional[QueueLogHandler] = None
) -> str:
    """Core render logic for a single project.

    Shared by both ``render_worker_loop`` (persistent) and
    ``render_project_worker`` (one-shot).
    """
    project_name = os.path.basename(project_path)

    temp_base_dir = os.path.join(AppConfig.APP_ROOT, "temp")
    project_temp_dir = os.path.join(temp_base_dir, project_name)

    if os.path.exists(project_temp_dir):
        logger.warning(
            f"[WORKER {project_name}] Removing stale temp directory "
            f"from previous run"
        )
        shutil.rmtree(project_temp_dir, ignore_errors=True)

    ctx = RenderContext(
        project_path=project_path,
        temp_dir=project_temp_dir,
        config=config,
        ffmpeg_path=config.get("ffmpeg_path") or detect_ffmpeg(),
        render_mode="full",
    )

    if not ctx.ffmpeg_path:
        msg_queue.put(ErrorMessage(
            project_name=project_name,
            error_text="FFmpeg not found. Install FFmpeg and ensure it is in PATH.",
        ))
        msg_queue.put(FinalizedMessage(project_name=project_name))
        return project_path

    try:
        os.makedirs(ctx.temp_dir, exist_ok=True)
    except Exception as exc:
        logger.error(
            f"[WORKER {project_name}] Cannot create temp directory ({type(exc).__name__}: {exc})",
            exc_info=True,
        )
        msg_queue.put(ErrorMessage(
            project_name=project_name,
            error_text=f"Cannot create temp directory: {exc}",
        ))
        msg_queue.put(FinalizedMessage(project_name=project_name))
        return project_path

    # Work on a local copy to avoid mutating the caller's config dict
    config = dict(config)
    config["_temp_dir"] = ctx.temp_dir
    # Also set it on ctx so backends receive it via ctx.config
    ctx.config["_temp_dir"] = ctx.temp_dir

    # Resolve render resolution — per-project settings take priority
    # over the global config so each project renders at its own resolution.
    #
    # Priority:
    #   1. Manifest render.width/height — explicit per-project override
    #   2. Manifest source.resolution — auto-detected at import/verify time
    #   3. Global config width/height — app-wide default
    #   4. Auto-detect from source files (PNG, HTML/CSS, .ppt-slide)
    #   5. Built-in defaults (1280×720)
    w = None
    h = None

    _proj_manifest = _load_project_manifest(project_path)

    # 1. Per-project render override (set if user manually changed resolution
    #    for this specific project after a previous render)
    _render_section = _proj_manifest.get("render", {})
    if _render_section.get("width") and _render_section.get("height"):
        w = _render_section["width"]
        h = _render_section["height"]

    # 2. Source resolution (auto-detected at import or via Verify Resolution)
    if w is None or h is None:
        _src_res = _proj_manifest.get("source", {}).get("resolution") or {}
        if w is None:
            w = _src_res.get("width")
        if h is None:
            h = _src_res.get("height")

    # 3. Global config fallback
    if w is None:
        w = config.get("width")
    if h is None:
        h = config.get("height")

    # 4. Auto-detect from source files
    if w is None or h is None:
        try:
            from utils import verify_project_resolution, set_source_resolution
            _vresult = verify_project_resolution(project_path)
            _detected = _vresult.get("resolution")
            if _detected:
                _dw, _dh = _detected
                logger.info(
                    f"[WORKER {project_name}] Auto-detected resolution: "
                    f"{_dw}x{_dh} (method: {_vresult.get('method', '?')})"
                )
                try:
                    set_source_resolution(project_path, _dw, _dh)
                except Exception as _save_exc:
                    logger.warning(
                        f"[WORKER {project_name}] Failed to save detected "
                        f"resolution to manifest ({type(_save_exc).__name__}: {_save_exc})",
                        exc_info=True,
                    )
                if w is None:
                    w = _dw
                if h is None:
                    h = _dh
            else:
                logger.info(
                    f"[WORKER {project_name}] Resolution auto-detect found "
                    f"nothing"
                )
        except Exception as _det_exc:
            logger.warning(
                f"[WORKER {project_name}] Resolution auto-detect failed: "
                f"{type(_det_exc).__name__}: {_det_exc}",
                exc_info=True,
            )

    # 5. Hard defaults
    if w is None:
        w = 1280
    if h is None:
        h = 720
    w, h = int(w), int(h)
    w_orig, h_orig = w, h
    w, h = AppConfig.sanitize_resolution(w, h)
    if w != w_orig or h != h_orig:
        logger.warning(
            f"[WORKER {project_name}] Resolution sanitized: "
            f"{w_orig}x{h_orig} -> {w}x{h} "
            f"(even dimensions required by FFmpeg)"
        )
    logger.info(
        f"[WORKER {project_name}] Render resolution: {w}x{h} "
        f"(source: {'manifest render' if _render_section.get('width') else 'source resolution' if _proj_manifest.get('source', {}).get('resolution') else 'global config' if config.get('width') else 'auto-detect' if w != 1280 or h != 720 else 'defaults'})"
    )
    config["width"] = w
    config["height"] = h
    fps = config.get("fps", 30)

    async def run_async():
        pipeline_start = time.time()
        output_dir = os.path.abspath(config.get("output_dir", "Output"))
        os.makedirs(output_dir, exist_ok=True)
        final_video = os.path.join(output_dir, project_name + ".mp4")
        transition_duration = config.get("transition_duration", 0.5)
        transition_frames = int(transition_duration * fps)

        encoder = config.get("encoder", "auto")
        preset = config.get("preset", "fast")

        cmd = FFmpegCommandBuilder.build_video_encoder(
            ffmpeg_path=ctx.ffmpeg_path,
            output_path=os.path.join(ctx.temp_dir, "video_raw.mp4"),
            width=w, height=h, fps=fps,
            encoder=encoder, preset=preset,
        )

        manifest = _load_project_manifest(project_path)
        current_video_hash = _get_video_settings_hash(ctx.config)
        _r = manifest.get("render", {})
        prev_video_hash = _r.get("video_hash", "")

        # ── Phase 1: Scan ────────────────────────────────────────────────
        phase_start = time.time()
        _emit_phase(msg_queue, project_name, RenderPhase.SCANNING,
                     queue_log_handler=queue_log_handler, pipeline_start=pipeline_start)
        logger.info(
            f"[WORKER {project_name}] Phase 1/6: Scanning project files"
        )

        html_files = sorted(
            glob.glob(os.path.join(project_path, "slide*.html")),
            key=natural_sort_key,
        )
        if not html_files:
            msg_queue.put(ErrorMessage(
                project_name=project_name,
                error_text=(
                    "No slide HTML files found in project directory"
                ),
            ))
            return

        tasks = prepare_slide_tasks(html_files, ctx)
        logger.info(
            f"[WORKER {project_name}] Phase 1 complete in "
            f"{time.time() - phase_start:.2f}s — {len(tasks)} slides found"
        )

        # ── Phase 2: Audio ───────────────────────────────────────────────
        phase_start = time.time()
        _emit_phase(msg_queue, project_name, RenderPhase.AUDIO_INIT,
                     queue_log_handler=queue_log_handler, pipeline_start=pipeline_start)
        logger.info(
            f"[WORKER {project_name}] Phase 2/6: Initializing audio backend"
        )
        await asyncio.to_thread(
            process_audio_tasks, tasks, ctx, msg_queue,
            queue_log_handler=queue_log_handler,
            pipeline_start=pipeline_start,
        )
        _emit_phase(msg_queue, project_name, RenderPhase.AUDIO_COMPLETE,
                     queue_log_handler=queue_log_handler, pipeline_start=pipeline_start)
        logger.info(
            f"[WORKER {project_name}] Phase 2 complete in "
            f"{fmt_duration(time.time() - phase_start)}"
        )

        # ── Decision logic ───────────────────────────────────────────────
        needs_video_render = _should_render_video(
            tasks, manifest, current_video_hash, prev_video_hash, ctx
        )

        if not needs_video_render:
            total_elapsed = time.time() - pipeline_start
            _emit_phase(msg_queue, project_name, RenderPhase.CACHE_SKIP,
                         queue_log_handler=queue_log_handler, pipeline_start=pipeline_start)
            logger.info(
                f"[WORKER {project_name}] Video is up to date — "
                f"skipping render (total: {fmt_duration(total_elapsed)})"
            )

            # ── Build timeline (needed for manifest) ─────
            timeline = build_timeline(tasks)

            assets_meta = _collect_assets_meta(timeline, project_path)
            _existing = _load_project_manifest(project_path)
            manifest_data = build_manifest_from_timeline(
                project_name, project_path, config, timeline,
                output_dir, final_video, "full", assets_meta,
                existing_manifest=_existing or None,
            )
            manifest_data["render"]["audio_hash"] = (
                _get_audio_settings_hash(ctx.config)
            )
            manifest_data["render"]["video_hash"] = current_video_hash
            save_project_manifest(project_path, manifest_data)
            update_library_manifest(project_path)

            # ── Thumbnail generation (even on cache-skip) ───────
            try:
                _emit_phase(msg_queue, project_name, RenderPhase.THUMBNAIL,
                             queue_log_handler=queue_log_handler, pipeline_start=pipeline_start)
                await _generate_thumbnail(
                    timeline, ctx, config, output_dir,
                    project_name, msg_queue, force=True,
                )
            except Exception as exc:
                logger.warning(
                    f"[WORKER {project_name}] Thumbnail generation "
                    f"failed (cache-skip path): {type(exc).__name__}: {exc}",
                    exc_info=True,
                )

            return

        # ── Start FFmpeg ─────────────────────────────────────────────────
        ffmpeg_proc, ffmpeg_log_file = _start_ffmpeg(
            cmd, ctx.temp_dir
        )
        if ffmpeg_proc is None:
            msg_queue.put(ErrorMessage(
                project_name=project_name,
                error_text=(
                    "FFmpeg failed to start. Check that FFmpeg is "
                    "installed and the encoder is supported on your "
                    "system."
                ),
            ))
            return

        # ── Phase 3: Timeline ────────────────────────────────────────────
        phase_start = time.time()
        _emit_phase(msg_queue, project_name, RenderPhase.BUILDING_TIMELINE,
                     queue_log_handler=queue_log_handler, pipeline_start=pipeline_start)
        logger.info(
            f"[WORKER {project_name}] Phase 3/6: Building timeline"
        )
        timeline = build_timeline(tasks)
        total_duration = sum(item["duration"] for item in timeline)
        logger.info(
            f"[WORKER {project_name}] Timeline: {len(timeline)} slides, "
            f"total duration {fmt_duration(total_duration)} "
            f"(built in {fmt_duration(time.time() - phase_start)})"
        )

        # ── Phase 4: Video ───────────────────────────────────────────────
        phase_start = time.time()
        _emit_phase(msg_queue, project_name, RenderPhase.VIDEO_STARTING,
                     queue_log_handler=queue_log_handler, pipeline_start=pipeline_start)
        logger.info(
            f"[WORKER {project_name}] Phase 4/6: Starting video encoder "
            f"({w}×{h}, {fps}fps, encoder={encoder})"
        )

        try:
            _emit_phase(msg_queue, project_name, RenderPhase.VIDEO_RENDERING,
                         queue_log_handler=queue_log_handler, pipeline_start=pipeline_start)
            await _render_video_frames(
                timeline, ffmpeg_proc, w, h, fps,
                transition_frames, transition_duration,
                project_name, msg_queue, ctx,
            )
        except Exception as exc:
            logger.error(
                f"[WORKER {project_name}] Video render failed: {exc}",
                exc_info=True,
            )
            _terminate_ffmpeg(ffmpeg_proc)
            # Close log file here since _finalize won't be called
            if ffmpeg_log_file is not None and not ffmpeg_log_file.closed:
                ffmpeg_log_file.close()
            raise
        finally:
            _close_ffmpeg_stdin(ffmpeg_proc)

        video_elapsed = time.time() - phase_start
        logger.info(
            f"[WORKER {project_name}] Phase 4 complete in "
            f"{fmt_duration(video_elapsed)}"
        )

        # ── Phase 5: Muxing ─────────────────────────────────────────────
        _emit_phase(msg_queue, project_name, RenderPhase.VIDEO_MUXING,
                     queue_log_handler=queue_log_handler, pipeline_start=pipeline_start)

        # ── Phase 6: Finalizing ──────────────────────────────────────────
        phase_start = time.time()
        _emit_phase(msg_queue, project_name, RenderPhase.FINALIZING,
                     queue_log_handler=queue_log_handler, pipeline_start=pipeline_start)
        logger.info(
            f"[WORKER {project_name}] Phase 6/6: Finalizing output"
        )

        # FIX #9: Pass ffmpeg_log_file to _finalize so it gets closed
        # even if _finalize raises an exception (e.g., VideoRenderError).
        _finalize(
            ffmpeg_proc, timeline, ctx, config,
            output_dir, final_video, project_name,
            current_video_hash, msg_queue, ffmpeg_log_file,
        )

        finalize_elapsed = time.time() - phase_start
        total_elapsed = time.time() - pipeline_start
        time_str = fmt_duration(total_elapsed)

        # ── Thumbnail generation (always) ──────────────────────────
        try:
            _emit_phase(msg_queue, project_name, RenderPhase.THUMBNAIL,
                         queue_log_handler=queue_log_handler, pipeline_start=pipeline_start)
            await _generate_thumbnail(
                timeline, ctx, config, output_dir,
                project_name, msg_queue, force=True,
            )
        except Exception as exc:
            logger.warning(
                f"[WORKER {project_name}] Thumbnail generation "
                f"failed: {type(exc).__name__}: {exc}",
                exc_info=True,
            )

        logger.info(
            f"[WORKER {project_name}] Render complete: "
            f"{len(timeline)} slides, {fmt_duration(total_duration)} video, "
            f"{w}x{h} — total time: {time_str}"
        )
    try:
        asyncio.run(run_async())
    except BackendGenerationError:
        # Error already reported by process_audio_tasks with a specific message
        pass
    except Exception as exc:
        logger.error(
            f"[WORKER {project_name}] Pipeline crashed: {exc}",
            exc_info=True,
        )
        msg_queue.put(ErrorMessage(
            project_name=project_name,
            error_text=f"Render pipeline error: {exc}",
        ))
    finally:
        _cleanup_temp(ctx.temp_dir, project_name)
        # Guarantee the GUI always receives a FinalizedMessage
        try:
            msg_queue.put(FinalizedMessage(project_name=project_name))
        except Exception:
            pass

    return project_path


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _should_render_video(
    tasks: list[SlideTask],
    manifest: dict,
    current_video_hash: str,
    prev_video_hash: str,
    ctx: RenderContext,
) -> bool:
    project_name = os.path.basename(ctx.project_path)

    # Force-render flag (set by "Force Render Selected" button)
    if manifest.get("_force_render"):
        logger.info(
            f"[WORKER {project_name}] [Decision] Force-render flag set → "
            f"video render required"
        )
        # Clear the flag so subsequent renders use normal logic
        try:
            m = load_project_manifest(ctx.project_path)
            m.pop("_force_render", None)
            save_project_manifest(ctx.project_path, m)
        except Exception:
            pass
        return True

    audio_was_regenerated = any(t.needs_tts for t in tasks)
    video_hash_changed = (current_video_hash != prev_video_hash)

    html_content_changed = False
    assets_meta = manifest.get("assets_meta", {})
    for task in tasks:
        html_name = os.path.basename(task.html_path)
        if html_name in assets_meta:
            prev_mtime = assets_meta[html_name].get("mtime", 0)
            try:
                if os.path.getmtime(task.html_path) > prev_mtime:
                    html_content_changed = True
                    break
            except OSError:
                pass
        else:
            html_content_changed = True
            break

    if audio_was_regenerated:
        logger.info(
            f"[WORKER {project_name}] [Decision] Audio regenerated → "
            f"video render required"
        )
    elif video_hash_changed:
        logger.info(
            f"[WORKER {project_name}] [Decision] Video settings changed → "
            f"video render required"
        )
    elif html_content_changed:
        logger.info(
            f"[WORKER {project_name}] [Decision] HTML content changed → "
            f"video render required"
        )
    else:
        logger.info(
            f"[WORKER {project_name}] [Decision] No changes detected → "
            f"skipping video render"
        )

    return (
        audio_was_regenerated or video_hash_changed or html_content_changed
    )


def _start_ffmpeg(cmd: list[str], temp_dir: str):
    """Launch the FFmpeg encoder subprocess."""
    err_log = None
    try:
        ffmpeg_log_path = os.path.join(temp_dir, "ffmpeg_encode.log")
        err_log = open(ffmpeg_log_path, "w", encoding="utf-8", errors="replace")
        try:
            proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=err_log,
            )
        except Exception:
            err_log.close()
            err_log = None
            raise
        if proc.poll() is not None:
            logger.error(
                f"FFmpeg process exited immediately with code "
                f"{proc.returncode}. See log: {ffmpeg_log_path}"
            )
            err_log.close()
            return None, None
        return proc, err_log
    except FileNotFoundError:
        if err_log is not None and not err_log.closed:
            err_log.close()
        logger.error(
            "FFmpeg binary not found. Ensure FFmpeg is installed and "
            "in PATH."
        )
        return None, None
    except Exception as exc:
        if err_log is not None and not err_log.closed:
            err_log.close()
        logger.error(f"FFmpeg failed to start ({type(exc).__name__}: {exc})", exc_info=True)
        return None, None


def _terminate_ffmpeg(proc) -> None:
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
            proc.wait()


def _close_ffmpeg_stdin(proc) -> None:
    if proc and proc.stdin and not proc.stdin.closed:
        try:
            proc.stdin.close()
        except Exception:
            pass


def _copy_file_buffered(src: str, dst: str, chunk_size: int = 8 * 1024 * 1024) -> None:
    """Copy a file using large buffered reads (default 8 MB chunks).

    Significantly faster than shutil.copy for large video files
    because it avoids the small default buffer size and reduces
    the number of syscalls.
    """
    with open(src, "rb") as f_in, open(dst, "wb") as f_out:
        while True:
            block = f_in.read(chunk_size)
            if not block:
                break
            f_out.write(block)


def _write_frame(proc, frame: np.ndarray) -> None:
    if proc.stdin:
        proc.stdin.write(frame.tobytes())


def _write_frames_safe(proc, frames, label: str) -> None:
    """Write frames to FFmpeg stdin with backpressure-aware flushing.

    Periodically flushes stdin to prevent unbounded buffering when
    the encoder is slower than the frame producer (e.g. high-res
    slides on CPU).  This keeps memory usage stable instead of
    growing linearly with frame count.
    """
    FLUSH_INTERVAL = 60  # flush every 60 frames
    try:
        for count, frame in enumerate(frames, 1):
            _write_frame(proc, frame)
            if count % FLUSH_INTERVAL == 0 and proc.stdin:
                proc.stdin.flush()
    except BrokenPipeError:
        logger.error(
            f"FFmpeg pipe broken during {label} — "
            f"encoder may have crashed"
        )
        _terminate_ffmpeg(proc)
        raise FFmpegError(f"Pipe broken during {label}")


async def _render_video_frames(
    timeline: list[dict],
    ffmpeg_proc,
    w: int, h: int, fps: int,
    transition_frames: int,
    transition_duration: float,
    project_name: str,
    msg_queue,
    ctx: RenderContext,
) -> None:
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        raise RuntimeError(
            "Playwright not installed. Run: "
            "pip install playwright && playwright install chromium"
        )

    total_slides = len(timeline)
    loop = asyncio.get_event_loop()

    # ── Pre-render all slides in parallel ────────────────────────────
    # Direct-image slides are I/O-bound (disk read + PIL resize) and
    # benefit hugely from a thread pool.  HTML slides go through a
    # single Playwright page sequentially (browser context is not
    # thread-safe), but the image slides run concurrently so the
    # browser render for the *next* HTML slide can overlap with
    # image-resize work for *other* slides.
    #
    # Strategy: render everything into a list upfront, using a
    # ThreadPoolExecutor for the sync image loads and the async loop
    # for Playwright HTML renders.  Then feed frames sequentially to
    # FFmpeg without any render latency between slides.

    # Separate indices by type for optimal scheduling
    _image_indices: list[int] = []
    _html_indices: list[int] = []
    for idx in range(total_slides):
        item = timeline[idx]
        if item.get("direct_image") and os.path.exists(item.get("direct_image", "")):
            _image_indices.append(idx)
        else:
            _html_indices.append(idx)

    logger.info(
        f"[WORKER {project_name}] Pre-rendering {total_slides} slides "
        f"({_image_indices and len(_image_indices)} image, "
        f"{_html_indices and len(_html_indices)} HTML) with thread pool"
    )

    slide_images: dict[int, np.ndarray] = {}

    def _load_image_slide(index: int) -> tuple[int, np.ndarray]:
        """Load and resize a direct-image slide (runs in thread pool)."""
        item = timeline[index]
        path = item["direct_image"]
        try:
            img_size = os.path.getsize(path)
            with Image.open(path) as img:
                orig_w, orig_h = img.size
                img = img.resize((w, h), Image.LANCZOS)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                logger.info(
                    f"[WORKER {project_name}] Slide {index + 1}: "
                    f"loaded image ({orig_w}x{orig_h}, "
                    f"{human_bytes(img_size)})"
                )
                return index, np.array(img)
        except Exception as exc:
            logger.error(
                f"[WORKER {project_name}] Direct image load "
                f"failed for slide {index} ({type(exc).__name__}: {exc}). "
                f"Falling back to Playwright.",
                exc_info=True,
            )
            return index, None  # signal to fall back to HTML render

    # Launch image loads in thread pool
    _img_executor = ThreadPoolExecutor(
        max_workers=min(4, max(1, len(_image_indices)))
    )
    img_futures: dict[asyncio.Future, int] = {}

    if _image_indices:
        for idx in _image_indices:
            fut = loop.run_in_executor(_img_executor, _load_image_slide, idx)
            img_futures[fut] = idx

    # Render HTML slides on the async Playwright page
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True, args=["--disable-gpu", "--no-sandbox"]
        )
        page = await browser.new_page()
        await page.set_viewport_size({"width": w, "height": h})

        async def _render_html_slide(index: int) -> np.ndarray:
            item = timeline[index]
            try:
                file_url = (
                    f"file:///"
                    f"{os.path.abspath(item['html']).replace(os.sep, '/')}"
                )
                await page.goto(
                    file_url, wait_until="domcontentloaded"
                )
                try:
                    await page.wait_for_load_state(
                        "networkidle", timeout=2000
                    )
                except Exception:
                    pass
                await asyncio.sleep(0.1)
                img_bytes = await page.screenshot(type="png")
                with Image.open(io.BytesIO(img_bytes)) as img:
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    logger.info(
                        f"[WORKER {project_name}] Slide {index + 1}: "
                        f"rendered HTML ({img.size[0]}x{img.size[1]})"
                    )
                    return np.array(img)
            except Exception as exc:
                logger.error(
                    f"[WORKER {project_name}] Screenshot failed for "
                    f"slide {index} ({type(exc).__name__}: {exc})",
                    exc_info=True,
                )
                return np.zeros((h, w, 3), dtype=np.uint8)

        # Render all HTML slides (sequential — single browser page)
        for idx in _html_indices:
            slide_images[idx] = await _render_html_slide(idx)
            msg_queue.put(SlideProgressMessage(
                project_name=project_name,
                current=idx + 1, total=total_slides,
            ))

        await browser.close()

    # Collect image slide results (may already be done by now)
    for fut, idx in img_futures.items():
        try:
            result_idx, img_array = await fut
            if img_array is not None:
                slide_images[result_idx] = img_array
            else:
                # Fall back: render as HTML (Playwright is closed,
                # use PIL as best-effort or black frame)
                slide_images[result_idx] = np.zeros((h, w, 3), dtype=np.uint8)
                logger.warning(
                    f"[WORKER {project_name}] Slide {result_idx + 1}: "
                    f"using fallback after image load failure"
                )
        except Exception as exc:
            logger.error(
                f"[WORKER {project_name}] Slide {idx + 1} thread failed: {exc}",
                exc_info=True,
            )
            slide_images[idx] = np.zeros((h, w, 3), dtype=np.uint8)

    _img_executor.shutdown(wait=False)
    logger.info(
        f"[WORKER {project_name}] All {total_slides} slides pre-rendered — "
        f"feeding frames to FFmpeg"
    )

    # ── Feed frames to FFmpeg (no render latency) ────────────────────
    img_current = slide_images.get(0, np.zeros((h, w, 3), dtype=np.uint8))

    for i in range(total_slides):
        img_next = (
            slide_images.get(i + 1)
            if i < total_slides - 1 else None
        )

        duration = timeline[i]["duration"]
        is_last = (i == total_slides - 1)

        if is_last:
            static_count = int(math.ceil(duration * fps))
        else:
            static_count = max(
                0,
                int(math.ceil((duration - transition_duration) * fps)),
            )

        if static_count > 0:
            await asyncio.to_thread(
                _write_frames_safe,
                ffmpeg_proc,
                list(img_current for _ in range(static_count)),
                "static frames",
            )

        if (
            not is_last
            and img_next is not None
            and transition_frames > 0
        ):
            # IMPROVE: Use a generator instead of an eager list to avoid
            # allocating transition_frames × width × height × 3 bytes
            # all at once.  At 4K (3840×2160) with 30 transition frames
            # this saves ~700 MB of peak memory per transition.
            def _transition_gen():
                for t in range(1, transition_frames + 1):
                    alpha = t / transition_frames
                    yield (
                        (img_current * (1.0 - alpha) + img_next * alpha)
                    ).astype(np.uint8)

            await asyncio.to_thread(
                _write_frames_safe,
                ffmpeg_proc, _transition_gen(), "transition",
            )

        if img_next is not None:
            img_current = img_next

    logger.info(
        f"[WORKER {project_name}] All frames written to FFmpeg"
    )


def _finalize(
    ffmpeg_proc,
    timeline: list[dict],
    ctx: RenderContext,
    config: dict,
    output_dir: str,
    final_video: str,
    project_name: str,
    current_video_hash: str,
    msg_queue,
    ffmpeg_log_file=None,        # FIX #9: Added parameter
) -> None:
    """Wait for FFmpeg, concat audio, mux into final video, save manifest.

    Performance improvements:
    - Audio concat + AAC encoding merged into a single FFmpeg pass.
    - ``-threads 0`` on both FFmpeg calls to use all CPU cores.
    - Buffered I/O for the fallback video copy (8 MB chunks).
    - Per-step timing so bottlenecks are visible in the log.
    - **PARALLEL**: Audio listfile build + concat starts in a background
      thread while FFmpeg encoder drain is finishing, saving 2-5 s.
    """
    finalize_start = time.time()

    # ── Build audio listfile (fast, I/O-bound) ───────────────────────
    # Extract this into its own function so it can run in a thread
    # alongside the FFmpeg drain wait.
    inter_slide_gap_ms = int(config.get("inter_slide_gap_ms", 0))
    listfile = os.path.join(ctx.temp_dir, "audio_list.txt")
    audio_full = os.path.join(ctx.temp_dir, "audio_full.wav")
    concat_log = os.path.join(ctx.temp_dir, "ffmpeg_concat.log")

    def _build_audio() -> tuple[subprocess.CompletedProcess, float]:
        """Build the crossfade chain / concat listfile, run FFmpeg, return result."""
        local_listfile = os.path.join(ctx.temp_dir, "audio_list.txt")
        local_audio_full = os.path.join(ctx.temp_dir, "audio_full.wav")
        local_concat_log = os.path.join(ctx.temp_dir, "ffmpeg_concat.log")

        n_slides = len(timeline)

        if inter_slide_gap_ms > 0 and n_slides >= 2:
            # ── True crossfade between adjacent slides ────────────
            # Build a filter_complex chain: [0][1]acrossfade...;[cf][2]acrossfade...
            inputs: list[str] = []
            for item in timeline:
                inputs.extend(["-i", item["audio"]])

            dur = inter_slide_gap_ms / 1000.0
            parts: list[str] = []
            prev = "[0]"
            for i in range(1, n_slides):
                out = f"[cf{i}]" if i < n_slides - 1 else "[aout]"
                parts.append(
                    f"{prev}[{i}]acrossfade=d={dur}:c1=tri:c2=tri{out}"
                )
                prev = out

            cmd = [
                ctx.ffmpeg_path, "-y",
                *inputs,
                "-filter_complex", ";".join(parts),
                "-map", "[aout]",
                "-c:a", "pcm_s16le",
                "-ar", "24000", "-ac", "1",
                local_audio_full,
            ]
            audio_start = time.time()
            with open(local_concat_log, "w") as concat_err:
                res = subprocess.run(
                    cmd,
                    stdout=subprocess.DEVNULL, stderr=concat_err,
                    timeout=300,
                )
            audio_elapsed = time.time() - audio_start
            logger.info(
                f"[WORKER {project_name}] Audio crossfade "
                f"({n_slides} slides, {inter_slide_gap_ms}ms) "
                f"in {audio_elapsed:.2f}s"
            )
            return res, audio_elapsed
        else:
            # ── Simple concat (no gap, no crossfade) ─────────────
            with open(local_listfile, "w", encoding="utf-8") as f:
                for item in timeline:
                    abs_path = os.path.abspath(item["audio"]).replace(
                        os.sep, "/"
                    )
                    escaped = (
                        abs_path
                        .replace("\\", "\\\\")
                        .replace("'", "\\'")
                        .replace("\n", "")
                        .replace("\r", "")
                        .replace("#", "\\#")
                        .replace("[", "\\[")
                        .replace("]", "\\]")
                    )
                    f.write(f"file '{escaped}'\n")

            audio_start = time.time()
            with open(local_concat_log, "w") as concat_err:
                res = subprocess.run(
                    [
                        ctx.ffmpeg_path, "-y",
                        "-threads", "0",
                        "-f", "concat", "-safe", "0",
                        "-i", local_listfile,
                        "-c:a", "pcm_s16le",
                        "-ar", "24000", "-ac", "1",
                        local_audio_full,
                    ],
                    stdout=subprocess.DEVNULL, stderr=concat_err,
                    timeout=300,
                )
            audio_elapsed = time.time() - audio_start
            return res, audio_elapsed

    # ── Step 1: FFmpeg drain + Audio concat IN PARALLEL ─────────────
    audio_thread_result: tuple[subprocess.CompletedProcess, float] | None = None
    audio_thread_error: Exception | None = None

    def _audio_thread_target():
        nonlocal audio_thread_result, audio_thread_error
        try:
            audio_thread_result = _build_audio()
        except Exception as exc:
            audio_thread_error = exc

    # Start audio work immediately — it builds the listfile and runs
    # concat while FFmpeg is still draining its encoder buffer.
    audio_start_wall = time.time()
    athread = threading.Thread(target=_audio_thread_target, daemon=True)
    athread.start()

    # Wait for FFmpeg video encoder to finish
    encode_start = time.time()
    encode_elapsed = 0.0
    try:
        if ffmpeg_proc:
            ffmpeg_proc.wait(timeout=600)
    except subprocess.TimeoutExpired:
        # Also wait for audio thread before raising
        athread.join(timeout=5)
        logger.error(
            f"[WORKER {project_name}] FFmpeg drain timed out — terminating"
        )
        _terminate_ffmpeg(ffmpeg_proc)
        raise VideoRenderError(
            "FFmpeg encoding timed out during final drain"
        )
    except OSError as exc:
        logger.error(
            f"[WORKER {project_name}] Error waiting for FFmpeg: {exc}"
        )
    finally:
        # Guarantee the log file is closed even if wait() raises
        if ffmpeg_log_file and not ffmpeg_log_file.closed:
            ffmpeg_log_file.close()

    # Check FFmpeg exit code AFTER the finally block so the log file is closed
    if ffmpeg_proc and ffmpeg_proc.returncode is not None and ffmpeg_proc.returncode != 0:
        athread.join(timeout=5)
        raise VideoRenderError(
            f"FFmpeg encoding failed with exit code {ffmpeg_proc.returncode}"
        )
    else:
        encode_elapsed = time.time() - encode_start
        logger.info(
            f"[WORKER {project_name}] "
            f"FFmpeg encoding finished successfully "
            f"(drain: {fmt_duration(encode_elapsed)})"
        )

    raw_video = os.path.join(ctx.temp_dir, "video_raw.mp4")
    if not os.path.exists(raw_video):
        athread.join(timeout=5)
        raise VideoRenderError(
            f"Raw video file not found after encoding: {raw_video}. "
            f"FFmpeg may have crashed during video capture."
        )

    # ── Step 2: Collect audio thread result ──────────────────────────
    athread.join(timeout=300)
    audio_elapsed = time.time() - audio_start_wall

    if audio_thread_error is not None:
        raise FFmpegError(
            f"Audio concat thread failed: {audio_thread_error}"
        )

    if audio_thread_result is None:
        raise FFmpegError("Audio concat thread did not produce a result")

    res_concat, _audio_concat_elapsed = audio_thread_result

    if res_concat.returncode != 0:
        raise FFmpegError(
            f"Audio concat failed (exit code "
            f"{res_concat.returncode}). Check log: {concat_log}"
        )
    _method = "crossfade" if inter_slide_gap_ms > 0 and len(timeline) >= 2 else "concat"
    logger.info(
        f"[WORKER {project_name}] Audio {_method} "
        f"({fmt_duration(_audio_concat_elapsed)}, "
        f"wall {fmt_duration(audio_elapsed)} incl. overlap with drain)"
    )

    # ── Step 3: Mux video + audio into final output ──────────────────
    # Video stream is copied (no re-encode).  Audio is lossless WAV —
    # encode to AAC in this single pass (no intermediate AAC encode).
    # +faststart is applied so the final file is stream-ready.
    mux_start = time.time()
    mux_log = os.path.join(ctx.temp_dir, "ffmpeg_mux.log")
    cmd_mux = [
        ctx.ffmpeg_path, "-y",
        "-threads", "0",              # use all CPU cores
        "-i", raw_video, "-i", audio_full,
        "-c:v", "copy",
        "-c:a", "aac", "-ar", "48000", "-ac", "2", "-b:a", "384k",
        "-movflags", "+faststart",
        "-shortest",
        final_video,
    ]
    with open(mux_log, "w") as mux_err:
        res_mux = subprocess.run(
            cmd_mux, stdout=subprocess.DEVNULL, stderr=mux_err,
            timeout=300,
        )
    mux_elapsed = time.time() - mux_start

    if res_mux.returncode != 0:
        logger.error(
            f"[WORKER {project_name}] Audio muxing failed "
            f"(exit {res_mux.returncode}). "
            f"Saving video WITHOUT audio as fallback."
        )
        try:
            _copy_file_buffered(raw_video, final_video)
            logger.warning(
                f"[WORKER {project_name}] Saved video without audio "
                f"due to mux error. Check log: {mux_log}"
            )
        except Exception as exc:
            logger.error(
                f"[WORKER {project_name}] Fallback video copy also "
                f"failed ({type(exc).__name__}: {exc})",
                exc_info=True,
            )
    else:
        logger.info(
            f"[WORKER {project_name}] Mux complete "
            f"({fmt_duration(mux_elapsed)}) — "
            f"{final_video} ({human_bytes(os.path.getsize(final_video))})"
        )

    # ── Step 4: Save manifests ───────────────────────────────────────
    manifest_start = time.time()
    assets_meta = _collect_assets_meta(timeline, ctx.project_path)

    _existing = _load_project_manifest(ctx.project_path)
    manifest_data = build_manifest_from_timeline(
        project_name, ctx.project_path, config,
        timeline, output_dir, final_video, "full", assets_meta,
        existing_manifest=_existing or None,
    )
    manifest_data["render"]["audio_hash"] = (
        _get_audio_settings_hash(ctx.config)
    )
    manifest_data["render"]["video_hash"] = current_video_hash
    save_project_manifest(ctx.project_path, manifest_data)
    update_library_manifest(ctx.project_path)

    total_finalize = time.time() - finalize_start
    _overlap_saved = max(0, encode_elapsed - (audio_elapsed - _audio_concat_elapsed))
    logger.info(
        f"[WORKER {project_name}] Finalize complete: "
        f"encode-drain {fmt_duration(encode_elapsed)}, "
        f"audio {fmt_duration(_audio_concat_elapsed)} "
        f"(wall {fmt_duration(audio_elapsed)}, "
        f"~{fmt_duration(_overlap_saved)} overlapped), "
        f"mux {fmt_duration(mux_elapsed)}, "
        f"manifest {fmt_duration(time.time() - manifest_start)} — "
        f"total {fmt_duration(total_finalize)}"
    )


def generate_thumbnail_worker(
    project_path: str,
    msg_queue: multiprocessing.Queue,
    config: dict,
    force: bool = False,
) -> str:
    """Standalone thumbnail worker — runs without the full render pipeline.

    Finds the first slide, renders it via Playwright at the configured
    video resolution, and saves a lossless PNG thumbnail.

    Args:
        project_path: Absolute path to the project directory.
        msg_queue: multiprocessing.Queue for GUI messages.
        config: Application settings dict.
        force: If True, overwrite an existing thumbnail.

    Returns:
        The project_path (same as input).
    """
    project_name = os.path.basename(project_path)

    _reset_worker_logger(msg_queue, project_name, config)
    logger.info(
        f"[THUMB {project_name}] Thumbnail worker started (PID {os.getpid()})"
    )

    try:
        _generate_thumbnail_standalone(
            project_path, config, msg_queue, force
        )
    except Exception as exc:
        logger.error(
            f"[THUMB {project_name}] Thumbnail generation failed: {exc}",
            exc_info=True,
        )
        msg_queue.put(ErrorMessage(
            project_name=project_name,
            error_text=f"Thumbnail generation failed: {exc}",
        ))

    msg_queue.put(FinalizedMessage(project_name=project_name))
    logger.info(f"[THUMB {project_name}] Thumbnail worker finished")
    return project_path


def _detect_slide_resolution(project_path: str, config: dict) -> Tuple[int, int]:
    """Resolve the resolution that was (or will be) used for the video.

    Priority order — the FIRST match wins:
      1. Global config ``width/height`` — matches what the next render will
         produce.  This is the primary source for thumbnail generation.
      2. Manifest ``source.resolution``
         — auto-detected from imported images at import time.
      3. Built-in defaults (1280x720).
    """
    from config import AppConfig

    # 1. Current config — matches what the next render will produce
    w = config.get("width")
    h = config.get("height")
    if w is not None and h is not None:
        return AppConfig.sanitize_resolution(int(w), int(h))

    # 2. Source-detected resolution from import time
    manifest = _load_project_manifest(project_path)
    sr = manifest.get("source", {}).get("resolution") or {}
    w = sr.get("width")
    h = sr.get("height")
    if w is not None and h is not None:
        return AppConfig.sanitize_resolution(int(w), int(h))

    # 3. Hard defaults
    return AppConfig.sanitize_resolution(1280, 720)


def _load_project_manifest(project_path: str) -> dict:
    """Load a project manifest (delegates to utils for auto-migration)."""
    return load_project_manifest(project_path)


def _is_landscape(w: int, h: int) -> bool:
    """Return True if the resolution is widescreen/landscape (width > height)."""
    return w > h


def _generate_thumbnail_standalone(
    project_path: str,
    config: dict,
    msg_queue,
    force: bool = False,
) -> None:
    """Core thumbnail logic for standalone thumbnail generation.

    For imported image slides (PNG/JPG), loads the source file directly
    and resizes to the configured video resolution — no Playwright needed.
    For HTML slides, renders via Playwright at the video resolution.
    Always saves as lossless PNG.
    """
    project_name = os.path.basename(project_path)

    # ── Resolve output directory ────────────────────────────────────
    from config import AppConfig
    output_dir = config.get("output_dir", "Output")
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(AppConfig.APP_ROOT, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # ── Find first slide HTML ───────────────────────────────────────
    html_files = sorted(
        glob.glob(os.path.join(project_path, "slide*.html")),
        key=natural_sort_key,
    )
    if not html_files:
        logger.error(
            f"[THUMB {project_name}] No slide HTML files found"
        )
        return

    first_html = html_files[0]

    # ── Resolve resolution from config (matches video output) ──────
    slide_w, slide_h = _detect_slide_resolution(project_path, config)

    # ── Landscape-only guard ───────────────────────────────────────
    if not _is_landscape(slide_w, slide_h):
        logger.info(
            f"[THUMB {project_name}] Skipped \u2014 non-landscape "
            f"aspect ratio ({slide_w}x{slide_h})"
        )
        msg_queue.put(LogMessage(
            project_name=project_name,
            text=(
                f"Thumbnail skipped \u2014 non-landscape "
                f"aspect ratio ({slide_w}x{slide_h})"
            ),
            level="info",
        ))
        try:
            _man = _load_project_manifest(project_path)
            _man["thumbnail"] = {
                "eligible": False,
                "skipped_reason": "non_landscape_aspect_ratio",
            }
            save_project_manifest(project_path, _man)
        except Exception:
            pass
        return

    # ── Smart Check: skip if thumbnail is up to date ──────────────
    thumb_filename = f"{project_name}_thumbnail.png"
    thumb_path = os.path.join(output_dir, thumb_filename)

    content_hash = _compute_thumbnail_content_hash(first_html, project_path)

    if not force and os.path.exists(thumb_path):
        prev_manifest = _load_project_manifest(project_path)
        prev_hash = prev_manifest.get("thumbnail", {}).get("hash", "")
        if content_hash == prev_hash:
            logger.info(
                f"[THUMB {project_name}] Thumbnail up to date (hash match) — skipping"
            )
            return

    # ── Capture first slide image ───────────────────────────────────
    img: Optional[Image.Image] = None

    # Fast path: image-only slide — load source directly, resize to video res
    direct_image = _get_direct_image_path(first_html, project_path)
    if direct_image and os.path.exists(direct_image):
        try:
            with Image.open(direct_image) as img:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                logger.info(
                    f"[THUMB {project_name}] Loaded first slide from "
                    f"source image ({img.size[0]}x{img.size[1]})"
                )
        except Exception as exc:
            logger.warning(
                f"[THUMB {project_name}] Direct image load failed ({type(exc).__name__}: {exc}). "
                f"Falling back to Playwright.",
                exc_info=True,
            )

    # Fallback: render first slide HTML via Playwright ────────────────
    if img is None:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            logger.error(
                f"[THUMB {project_name}] Cannot generate thumbnail — "
                f"Playwright not installed."
            )
            return

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(
                    headless=True, args=["--disable-gpu", "--no-sandbox"]
                )
                page = browser.new_page()
                page.set_viewport_size(
                    {"width": slide_w, "height": slide_h}
                )
                file_url = (
                    f"file:///"
                    f"{os.path.abspath(first_html).replace(os.sep, '/')}"
                )
                page.goto(file_url, wait_until="domcontentloaded")
                try:
                    page.wait_for_load_state("networkidle", timeout=2000)
                except Exception:
                    pass
                time.sleep(0.1)
                img_bytes = page.screenshot(type="png")
                browser.close()
                with Image.open(io.BytesIO(img_bytes)) as _src:
                    img = (_src.convert("RGB")
                           if _src.mode != "RGB" else _src.copy())
                    logger.info(
                        f"[THUMB {project_name}] Rendered first slide via "
                        f"Playwright ({slide_w}x{slide_h})"
                    )
        except Exception as exc:
            logger.error(
                f"[THUMB {project_name}] Playwright render failed ({type(exc).__name__}: {exc})",
                exc_info=True,
            )
            return

    if img is None:
        logger.error(
            f"[THUMB {project_name}] Could not obtain first slide image"
        )
        return

    # ── Ensure resolution matches video config exactly ─────────────
    if img.size != (slide_w, slide_h):
        img = img.resize((slide_w, slide_h), Image.LANCZOS)
        logger.info(
            f"[THUMB {project_name}] Resized to match video "
            f"resolution: {slide_w}x{slide_h}"
        )

    # ── Save as PNG (lossless, pixel-perfect) ──────────────────────
    img.save(thumb_path, "PNG")
    out_w, out_h = img.size

    thumb_size_kb = os.path.getsize(thumb_path) / 1024
    logger.info(
        f"[THUMB {project_name}] Thumbnail saved: {thumb_path} "
        f"({out_w}x{out_h}, {thumb_size_kb:.0f} KB, PNG)"
    )
    msg_queue.put(LogMessage(
        project_name=project_name,
        text=(
            f"Thumbnail saved: {thumb_filename} "
            f"({out_w}x{out_h}, PNG)"
        ),
        level="info",
    ))

    # ── Update manifest.json with thumbnail metadata ────────────────
    try:
        manifest = _load_project_manifest(project_path)
        manifest["thumbnail"] = {
            "file": thumb_filename,
            "width": slide_w,
            "height": slide_h,
            "hash": content_hash,
            "eligible": True,
        }
        save_project_manifest(project_path, manifest)
        update_library_manifest(project_path)
        logger.debug(
            f"[THUMB {project_name}] manifest.json updated with thumbnail metadata"
        )
    except Exception as exc:
        logger.warning(
            f"[THUMB {project_name}] Failed to update manifest ({type(exc).__name__}: {exc})",
            exc_info=True,
        )


def _get_direct_image_path(html_path: str, project_path: str) -> Optional[str]:
    """Extract the image path from an image-wrapper HTML slide."""
    if not is_image_slide(html_path):
        return None
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            content = f.read()
        m = re.search(r'<img[^>]+src=["\']([^"\']+)["\']', content, re.IGNORECASE)
        if m:
            src = m.group(1)
            if not os.path.isabs(src):
                src = os.path.join(project_path, src)
            return src
    except Exception:
        pass
    return None


async def _generate_thumbnail(
    timeline: list[dict],
    ctx: RenderContext,
    config: dict,
    output_dir: str,
    project_name: str,
    msg_queue,
    force: bool = False,
) -> None:
    """Generate a PNG thumbnail from the first slide.

    For imported image slides, loads the source file directly and
    resizes to the configured video resolution — no Playwright needed.
    For HTML slides, renders via Playwright at the video resolution.
    No cropping, no aspect-ratio filtering.

    This is the pipeline-integrated version called after video render.
    For standalone thumbnail generation, see generate_thumbnail_worker().
    """
    first_item = timeline[0]
    html_path = first_item["html"]

    # ── Resolve resolution to match the video that was just rendered ──
    slide_w, slide_h = _detect_slide_resolution(ctx.project_path, config)

    # ── Landscape-only guard ───────────────────────────────────────
    if not _is_landscape(slide_w, slide_h):
        logger.info(
            f"[WORKER {project_name}] Thumbnail skipped \u2014 "
            f"non-landscape aspect ratio ({slide_w}x{slide_h})"
        )
        msg_queue.put(LogMessage(
            project_name=project_name,
            text=(
                f"Thumbnail skipped \u2014 non-landscape "
                f"aspect ratio ({slide_w}x{slide_h})"
            ),
            level="info",
        ))
        try:
            _man = _load_project_manifest(ctx.project_path)
            _man["thumbnail"] = {
                "eligible": False,
                "skipped_reason": "non_landscape_aspect_ratio",
            }
            save_project_manifest(ctx.project_path, _man)
        except Exception:
            pass
        return

    thumb_filename = f"{project_name}_thumbnail.png"
    thumb_path = os.path.join(output_dir, thumb_filename)

    # ── Smart Check: skip if thumbnail is up to date (hash-based) ──
    content_hash = _compute_thumbnail_content_hash(html_path, ctx.project_path)

    if not force and os.path.exists(thumb_path):
        prev_manifest = _load_project_manifest(ctx.project_path)
        prev_hash = prev_manifest.get("thumbnail", {}).get("hash", "")
        if content_hash == prev_hash:
            logger.info(
                f"[WORKER {project_name}] Thumbnail up to date (hash match) — skipping"
            )
            return

    # ── Capture first slide image ───────────────────────────────────
    img: Optional[Image.Image] = None

    # Fast path: image-only slide — load source directly, resize to video res
    direct_image = first_item.get("direct_image")
    if direct_image and os.path.exists(direct_image):
        try:
            with Image.open(direct_image) as img:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                logger.info(
                    f"[WORKER {project_name}] Thumbnail: loaded first slide "
                    f"from source image ({img.size[0]}x{img.size[1]})"
                )
        except Exception as exc:
            logger.warning(
                f"[WORKER {project_name}] Direct image load failed "
                f"for thumbnail ({type(exc).__name__}: {exc}). Falling back to Playwright.",
                exc_info=True,
            )

    # Fallback: render first slide HTML via Playwright ────────────────
    if img is None:
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            logger.error(
                f"[WORKER {project_name}] Cannot generate thumbnail — "
                f"Playwright not installed."
            )
            return

        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(
                    headless=True, args=["--disable-gpu", "--no-sandbox"]
                )
                page = await browser.new_page()
                await page.set_viewport_size(
                    {"width": slide_w, "height": slide_h}
                )
                file_url = (
                    f"file:///"
                    f"{os.path.abspath(html_path).replace(os.sep, '/')}"
                )
                await page.goto(file_url, wait_until="domcontentloaded")
                try:
                    await page.wait_for_load_state("networkidle", timeout=2000)
                except Exception:
                    pass
                await asyncio.sleep(0.1)
                img_bytes = await page.screenshot(type="png")
                await browser.close()
                with Image.open(io.BytesIO(img_bytes)) as _src:
                    img = (_src.convert("RGB")
                           if _src.mode != "RGB" else _src.copy())
                    logger.info(
                        f"[WORKER {project_name}] Thumbnail: rendered first "
                        f"slide via Playwright ({slide_w}x{slide_h})"
                    )
        except Exception as exc:
            logger.error(
                f"[WORKER {project_name}] Thumbnail Playwright render "
                f"failed ({type(exc).__name__}: {exc})",
                exc_info=True,
            )
            return

    if img is None:
        logger.error(
            f"[WORKER {project_name}] Thumbnail: could not obtain "
            f"first slide image"
        )
        return

    # ── Ensure resolution matches video config exactly ─────────────
    if img.size != (slide_w, slide_h):
        img = img.resize((slide_w, slide_h), Image.LANCZOS)
        logger.info(
            f"[WORKER {project_name}] Thumbnail resized to match "
            f"video resolution: {slide_w}x{slide_h}"
        )

    # ── Save as PNG (lossless, pixel-perfect) ──────────────────────
    img.save(thumb_path, "PNG")
    out_w, out_h = img.size

    thumb_size_kb = os.path.getsize(thumb_path) / 1024
    logger.info(
        f"[WORKER {project_name}] Thumbnail saved: {thumb_path} "
        f"({out_w}x{out_h}, {thumb_size_kb:.0f} KB, PNG)"
    )
    msg_queue.put(LogMessage(
        project_name=project_name,
        text=(
            f"Thumbnail saved: {thumb_filename} "
            f"({out_w}x{out_h}, PNG)"
        ),
        level="info",
    ))

    # ── Update manifest.json with thumbnail metadata ────────────────
    try:
        manifest = _load_project_manifest(ctx.project_path)
        manifest["thumbnail"] = {
            "file": thumb_filename,
            "width": out_w,
            "height": out_h,
            "hash": content_hash,
            "eligible": True,
        }
        save_project_manifest(ctx.project_path, manifest)
        update_library_manifest(ctx.project_path)
        logger.debug(
            f"[WORKER {project_name}] manifest.json updated with thumbnail metadata"
        )
    except Exception as exc:
        logger.warning(
            f"[WORKER {project_name}] Failed to update manifest ({type(exc).__name__}: {exc})",
            exc_info=True,
        )


def _collect_assets_meta(timeline: list[dict], project_path: str) -> dict:
    meta: dict = {}
    for item in timeline:
        html_path = item["html"]
        txt_path = os.path.splitext(html_path)[0] + ".txt"
        for path in (html_path, txt_path):
            if os.path.exists(path):
                meta[os.path.basename(path)] = {
                    "mtime": os.path.getmtime(path),
                    "size": os.path.getsize(path),
                }

    images_dir = os.path.join(project_path, "images")
    if os.path.isdir(images_dir):
        for img_name in os.listdir(images_dir):
            img_path = os.path.join(images_dir, img_name)
            if os.path.isfile(img_path):
                meta[f"images/{img_name}"] = {
                    "mtime": os.path.getmtime(img_path),
                    "size": os.path.getsize(img_path),
                }
    return meta


def _cleanup_temp(temp_dir: str, project_name: str) -> None:
    try:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.debug(
                f"[WORKER {project_name}] Cleaned temp directory"
            )
    except Exception as exc:
        logger.warning(
            f"[WORKER {project_name}] Temp cleanup failed ({type(exc).__name__}: {exc})",
            exc_info=True,
        )

    parent = os.path.dirname(temp_dir)
    if os.path.isdir(parent):
        try:
            if not os.listdir(parent):
                os.rmdir(parent)
                logger.debug(
                    f"[WORKER {project_name}] Removed empty parent "
                    f"temp directory"
                )
        except OSError as exc:
            logger.debug(
                f"[WORKER {project_name}] Parent temp cleanup skipped: "
                f"{exc}"
            )