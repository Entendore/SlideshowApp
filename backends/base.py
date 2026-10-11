# backends/base.py
"""
Abstract base class for all TTS backend plugins.

All backends (Qwen3, Edge, OmniVoice, etc.) must inherit from this class
and implement the abstract methods to be compatible with the render engine.
"""

from __future__ import annotations

import logging
import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from text_chunker import TextChunker, ChunkingConfig

logger = logging.getLogger(__name__)


class BaseTTSBackend(ABC):
    """Contract that every TTS backend must satisfy."""

    DESCRIPTION: str = "Generic backend implementation."
    AUDIO_SETTINGS_KEYS: list[str] = []

    # ─── Construction ────────────────────────────────────────────────────────

    def __init__(self, config: dict) -> None:
        self.config = config
        self._progress_callback: Optional[Callable[[str, int, int, str], None]] = None
        self._chunk_temp_dirs: List[str] = []

    # ─── Context-manager support ─────────────────────────────────────────────

    def __enter__(self) -> BaseTTSBackend:
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.cleanup()
        return False  # do not suppress exceptions

    # ─── Progress reporting ──────────────────────────────────────────────────

    def set_progress_callback(
        self, callback: Optional[Callable[[str, int, int, str], None]]
    ) -> None:
        """Set a callback ``fn(stage, current, total, message)``."""
        self._progress_callback = callback

    def _report_progress(
        self, stage: str, current: int, total: int, message: str = ""
    ) -> None:
        if self._progress_callback:
            self._progress_callback(stage, current, total, message)

    # ─── Temp-directory helpers ──────────────────────────────────────────────

    def _get_local_temp_dir(self, subdir: str = "chunks") -> str:
        """Return a temp directory inside the app's temp tree.

        Uses the project temp dir from config when available, otherwise
        falls back to {cwd}/temp/{subdir}.  Never uses the OS temp.
        """
        base_temp = self.config.get("_temp_dir", "")
        if not base_temp or not os.path.isdir(base_temp):
            base_temp = os.path.join(os.getcwd(), "temp")
        temp_dir = os.path.join(base_temp, subdir)
        os.makedirs(temp_dir, exist_ok=True)

        self._chunk_temp_dirs.append(temp_dir)
        logger.debug(f"[Backend] Created temp dir: {temp_dir}")
        return temp_dir

    def _cleanup_chunk_temp(self) -> None:
        """Remove all chunk temp directories created during generation."""
        base_temp = self.config.get("_temp_dir", "")

        for chunk_dir in self._chunk_temp_dirs:
            try:
                if not os.path.exists(chunk_dir):
                    continue
                shutil.rmtree(chunk_dir, ignore_errors=True)
            except Exception as exc:
                logger.warning(f"[Backend] Failed to clean temp dir {chunk_dir}: {exc}")

        count = len(self._chunk_temp_dirs)
        self._chunk_temp_dirs.clear()
        logger.debug(f"[Backend] Cleaned up {count} chunk temp dir(s)")

    # ─── Input validation ────────────────────────────────────────────────────

    def _validate_batch_inputs(
        self, texts: List[str], output_paths: List[str]
    ) -> None:
        """Validate batch inputs.  Raises ``ValueError`` on problems."""
        if len(texts) != len(output_paths):
            raise ValueError(
                f"texts and output_paths length mismatch: "
                f"{len(texts)} != {len(output_paths)}"
            )

        if not any(t and t.strip() for t in texts):
            raise ValueError("Cannot generate audio: all texts are empty")

        for path in output_paths:
            dir_name = os.path.dirname(path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)

    # ─── Chunking orchestration ──────────────────────────────────────────────

    def _generate_single_with_chunking(
        self,
        text: str,
        output_path: str,
        slide_index: int,
        language: str,
        chunker: Optional[TextChunker],
        generate_single_fn: Callable[[str, str], bool],
        concatenate_fn: Callable[[List[str], str], bool],
        temp_dir: str,
    ) -> Tuple[bool, Optional[str]]:
        """Generate audio for one text item, handling chunking if needed.

        Returns:
            (success, error_message_or_None)
        """
        if not text or not text.strip():
            return False, "Empty text"

        needs_chunking = chunker and chunker.needs_chunking(text)

        if needs_chunking:
            return self._generate_chunked(
                text, output_path, slide_index, language,
                chunker, generate_single_fn, concatenate_fn, temp_dir,
            )
        else:
            return self._generate_unchunked(
                text, output_path, slide_index, generate_single_fn,
            )

    def _generate_chunked(
        self,
        text: str,
        output_path: str,
        slide_index: int,
        language: str,
        chunker: TextChunker,
        generate_single_fn: Callable[[str, str], bool],
        concatenate_fn: Callable[[List[str], str], bool],
        temp_dir: str,
    ) -> Tuple[bool, Optional[str]]:
        """Handle generation with text chunking."""
        logger.info(
            f"[Backend] Segment for slide {slide_index + 1}: Long text "
            f"({len(text)} chars), chunking..."
        )
        chunks = chunker.chunk_text(text, language)
        total_chunks = len(chunks)
        logger.info(
            f"[Backend] Slide {slide_index + 1}: Split into {total_chunks} sub-segment(s)"
        )

        chunk_paths: list[str] = []
        for chunk_idx, (chunk_text, _estimated_duration) in enumerate(chunks):
            chunk_path = os.path.join(
                temp_dir, f"chunk_{slide_index}_{chunk_idx}.wav"
            )
            self._report_progress(
                "chunk", chunk_idx + 1, total_chunks,
                f"Sub-segment {chunk_idx + 1}/{total_chunks}",
            )
            try:
                if generate_single_fn(chunk_text, chunk_path):
                    chunk_paths.append(chunk_path)
                else:
                    return False, f"Failed to generate chunk {chunk_idx + 1}"
            except Exception as exc:
                return False, f"Chunk {chunk_idx + 1}: {exc}"

        if len(chunk_paths) != total_chunks:
            return False, (
                f"Only {len(chunk_paths)}/{total_chunks} chunks generated"
            )

        if not concatenate_fn(chunk_paths, output_path):
            return False, (
                f"Failed to concatenate chunks for slide {slide_index + 1}"
            )

        # Cleanup chunk files on success
        for cp in chunk_paths:
            try:
                os.remove(cp)
            except OSError:
                pass

        return True, None

    def _generate_unchunked(
        self,
        text: str,
        output_path: str,
        slide_index: int,
        generate_single_fn: Callable[[str, str], bool],
    ) -> Tuple[bool, Optional[str]]:
        """Handle generation without chunking."""
        try:
            if generate_single_fn(text, output_path):
                return True, None
            return False, (
                f"Failed to generate audio for slide {slide_index + 1}"
            )
        except Exception as exc:
            return False, str(exc)

    # ─── Abstract interface ──────────────────────────────────────────────────

    @abstractmethod
    def initialize(self) -> None:
        """Load models into memory (VRAM / RAM).

        Called once before generation begins.  Implementations should handle:
        - Downloading models if not present.
        - Loading models onto the correct device.
        - Setting up processors / tokenizers.
        """
        ...

    @abstractmethod
    def generate_batch(
        self, texts: List[str], output_paths: List[str]
    ) -> Tuple[bool, List[str]]:
        """Generate audio for a batch of texts.

        Args:
            texts:        Strings to convert to speech.  Must not all be empty.
            output_paths: Corresponding .wav output paths.

        Returns:
            (success, error_list) — error_list is empty on success.

        Raises:
            ValueError: If all texts are empty or whitespace-only.
        """
        ...

    @abstractmethod
    def cleanup(self) -> None:
        """Unload models and free resources (VRAM, RAM, CUDA caches)."""
        ...