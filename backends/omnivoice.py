# backends/omnivoice.py
"""
OmniVoice TTS Backend Plugin.
Uses the omnivoice CLI tools (omnivoice-infer / omnivoice-infer-batch)
for text-to-speech generation.

Supports 3 modes:
1. Auto Voice - Generates speech in a random/automatic voice
2. Voice Clone - Uses reference audio + transcription
3. Voice Design - Uses voice instruction (e.g., "male, British accent")

Optimized for GTX 1080 (8GB VRAM, Pascal architecture):
- Uses float16 precision
- No Flash Attention 2 (not supported on Pascal)
- Single GPU inference

Installation:
    pip install omnivoice
"""

import os
import gc
import json
import shutil
import logging
import subprocess
import wave
from typing import List, Tuple, Optional, Dict

from text_chunker import TextChunker, ChunkingConfig

from PySide6.QtWidgets import (
    QWidget, QFormLayout, QComboBox, QLineEdit,
    QHBoxLayout, QLabel, QPushButton, QFileDialog, QPlainTextEdit
)
from PySide6.QtCore import QTimer, Qt
import importlib.util
from backends.base import BaseTTSBackend
logger = logging.getLogger(__name__)


class OmniVoiceBackend(BaseTTSBackend):
    """
    Concrete implementation of BaseTTSBackend for OmniVoice.
    
    Uses subprocess calls to omnivoice-infer CLI for generation.
    Batch mode (omnivoice-infer-batch) loads the model once and
    processes all slides — much faster than per-slide CLI calls.
    """

    DESCRIPTION = "OmniVoice: Local GPU TTS with voice cloning, voice design & auto voice. Optimized for GTX 1080 (8GB VRAM, float16, no Flash Attn 2). Install: pip install omnivoice"
    
    AUDIO_SETTINGS_KEYS = [
        "omnivoice_mode", "omnivoice_model_id", "omnivoice_language",
        "omnivoice_ref_audio", "omnivoice_ref_text", "omnivoice_instruct",
        "omnivoice_use_batch",
    ]

    OMNIVOICE_MODES = [
        ("auto", "Auto Voice (Random)"),
        ("voice_clone", "Voice Clone (Reference Audio)"),
        ("voice_design", "Voice Design (Instruction)"),
    ]

    OMNIVOICE_LANGUAGES = {
        "English": "en",
        "Chinese": "zh",
        "Japanese": "ja",
        "Korean": "ko",
        "German": "de",
        "French": "fr",
        "Russian": "ru",
        "Portuguese": "pt",
        "Spanish": "es",
        "Italian": "it",
    }

    OMNIVOICE_LANGUAGE_NAMES = list(OMNIVOICE_LANGUAGES.keys())

    @staticmethod
    def get_ui_options() -> Dict:
        return {
            "modes": OmniVoiceBackend.OMNIVOICE_MODES,
            "languages": OmniVoiceBackend.OMNIVOICE_LANGUAGE_NAMES,
        }
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if PyTorch is installed WITHOUT importing it."""
        return importlib.util.find_spec("torch") is not None

    # =================================================================
    # UI GENERATION FACTORY
    # =================================================================

    @staticmethod
    def get_settings_widget(mode: str, config, parent=None, save_callback=None) -> QWidget:
        widget = QWidget(parent)
        layout = QFormLayout(widget)

        def save(key, value):
            if save_callback:
                save_callback(key, value)
            else:
                config.set(key, value)

        # --- Shared GTX 1080 Notice ---
        gtx_label = QLabel(
            "<i style='color:#7ec8e3;'>☕ GTX 1080 Optimized: float16, no Flash Attention 2, "
            "single-GPU batch inference. Model loads once per render.</i>"
        )
        gtx_label.setWordWrap(True)

        # --- Shared Model ID ---
        model_edit = QLineEdit()
        model_edit.setText(config.get("omnivoice_model_id", "k2-fsa/OmniVoice"))
        model_edit.setPlaceholderText("HuggingFace model ID or local path")
        model_edit.editingFinished.connect(
            lambda: save("omnivoice_model_id", model_edit.text())
        )

        # --- Shared Language ---
        lang_combo = QComboBox()
        lang_combo.addItems(OmniVoiceBackend.OMNIVOICE_LANGUAGE_NAMES)
        lang_combo.setCurrentText(config.get("omnivoice_language", "English"))
        lang_combo.currentTextChanged.connect(lambda t: save("omnivoice_language", t))

        # ---------------------------------------------------------
        # MODE: Auto Voice
        # ---------------------------------------------------------
        if mode == "auto":
            layout.addRow(QLabel("<b>Auto Voice Configuration</b>"))
            layout.addRow(QLabel("<i>Generates speech in an automatic voice.</i>"))
            layout.addRow(QLabel("<i>No reference audio or instructions needed.</i>"))
            layout.addRow(gtx_label)
            layout.addRow("Language:", lang_combo)
            layout.addRow("Model ID:", model_edit)

        # ---------------------------------------------------------
        # MODE: Voice Clone
        # ---------------------------------------------------------
        elif mode == "voice_clone":
            layout.addRow(QLabel("<b>Voice Clone Configuration</b>"))
            layout.addRow(QLabel("<i>Clones a voice from reference audio + transcript.</i>"))
            layout.addRow(QLabel("<i>If ref_text is empty, Whisper auto-transcribes.</i>"))
            layout.addRow(gtx_label)

            ref_row = QHBoxLayout()
            ref_audio_edit = QLineEdit()
            ref_audio_edit.setText(config.get("omnivoice_ref_audio", ""))
            ref_audio_edit.setPlaceholderText("Path to reference .wav file")
            ref_audio_edit.editingFinished.connect(
                lambda: save("omnivoice_ref_audio", ref_audio_edit.text())
            )

            btn_browse = QPushButton("...")
            btn_browse.setMaximumWidth(30)
            btn_browse.clicked.connect(
                lambda: OmniVoiceBackend._browse_audio(
                    ref_audio_edit, ref_text_edit, config
                )
            )

            ref_row.addWidget(ref_audio_edit)
            ref_row.addWidget(btn_browse)
            layout.addRow("Reference Audio:", ref_row)

            ref_text_edit = QLineEdit()
            ref_text_edit.setText(config.get("omnivoice_ref_text", ""))
            ref_text_edit.setPlaceholderText(
                "Transcript of reference audio (optional — auto-transcribed if empty)"
            )
            ref_text_edit.editingFinished.connect(
                lambda: save("omnivoice_ref_text", ref_text_edit.text())
            )
            layout.addRow("Reference Text:", ref_text_edit)

            layout.addRow("Language:", lang_combo)
            layout.addRow("Model ID:", model_edit)

        # ---------------------------------------------------------
        # MODE: Voice Design
        # ---------------------------------------------------------
        elif mode == "voice_design":
            layout.addRow(QLabel("<b>Voice Design Configuration</b>"))
            layout.addRow(QLabel("<i>Generates a voice from a text description.</i>"))
            layout.addRow(gtx_label)

            instruct_edit = QPlainTextEdit()
            instruct_edit.setPlaceholderText(
                "e.g., 'male, British accent' or 'female, soft and warm voice'"
            )
            instruct_edit.setMaximumHeight(80)
            instruct_edit.setPlainText(config.get("omnivoice_instruct", ""))

            _instruct_timer = QTimer(widget)
            _instruct_timer.setSingleShot(True)
            _instruct_timer.setInterval(800)
            _instruct_timer.timeout.connect(
                lambda: save("omnivoice_instruct", instruct_edit.toPlainText())
            )
            instruct_edit.textChanged.connect(_instruct_timer.start)
            layout.addRow("Voice Instruction:", instruct_edit)

            layout.addRow("Language:", lang_combo)
            layout.addRow("Model ID:", model_edit)

        return widget

    @staticmethod
    def _browse_audio(line_edit, text_edit, config):
        fname, _ = QFileDialog.getOpenFileName(
            None,
            "Select Reference Audio",
            config.get("voice_references_root", ""),
            "Audio Files (*.wav *.mp3 *.flac)",
        )
        if fname:
            line_edit.setText(fname)
            config.set("omnivoice_ref_audio", fname)

            if text_edit:
                base_path = os.path.splitext(fname)[0]
                txt_path = base_path + ".txt"

                found_path = None
                if os.path.exists(txt_path):
                    found_path = txt_path
                else:
                    # Case-insensitive fallback
                    dir_path = os.path.dirname(fname) or "."
                    desired_name = os.path.basename(txt_path)
                    try:
                        for f in os.listdir(dir_path):
                            if f.lower() == desired_name.lower():
                                found_path = os.path.join(dir_path, f)
                                break
                    except Exception:
                        pass

                if found_path:
                    try:
                        with open(found_path, "r", encoding="utf-8") as f:
                            content = f.read().strip()
                            text_edit.setText(content)
                            config.set("omnivoice_ref_text", content)
                            logger.info(
                                f"[OmniVoiceBackend] Auto-loaded ref text: {found_path}"
                            )
                    except Exception as e:
                        logger.warning(f"Could not read reference text: {e}")

    # =================================================================
    # INITIALIZATION
    # =================================================================

    def __init__(self, config: dict):
        super().__init__(config)
        self._infer_cmd = self._find_command("omnivoice-infer")
        self._batch_cmd = self._find_command("omnivoice-infer-batch")

        if not self._infer_cmd:
            raise RuntimeError(
                "omnivoice-infer command not found. "
                "Install OmniVoice via: pip install omnivoice"
            )

    @staticmethod
    def _find_command(name: str) -> Optional[str]:
        """Find an executable command in PATH."""
        return shutil.which(name)

    def initialize(self):
        """Validate configuration and log system info."""
        logger.info("[OmniVoiceBackend] Initializing...")

        import torch
        import transformers

        mode = self.config.get("omnivoice_mode", "auto")
        model_id = self.config.get("omnivoice_model_id", "k2-fsa/OmniVoice")

        logger.info(f"[OmniVoiceBackend] Mode: {mode}")
        logger.info(f"[OmniVoiceBackend] Model: {model_id}")
        logger.info(f"[OmniVoiceBackend] CLI: {self._infer_cmd}")

        if self._batch_cmd:
            logger.info(f"[OmniVoiceBackend] Batch CLI: {self._batch_cmd}")
        else:
            logger.warning(
                "[OmniVoiceBackend] omnivoice-infer-batch not found. "
                "Using individual inference only (slower)."
            )

        # Log GPU info for GTX 1080 optimization
        try:
            import torch

            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                vram = torch.cuda.get_device_properties(0).total_mem / (1024**3)
                logger.info(
                    f"[OmniVoiceBackend] GPU: {gpu_name} ({vram:.1f} GB VRAM)"
                )

                major, minor = torch.cuda.get_device_capability(0)
                if major < 8:
                    logger.info(
                        f"[OmniVoiceBackend] Pascal GPU (compute {major}.{minor}). "
                        "Flash Attention 2 unavailable — using default attention."
                    )
                else:
                    logger.info(
                        f"[OmniVoiceBackend] GPU compute capability {major}.{minor}. "
                        "Flash Attention 2 may be available."
                    )
            else:
                logger.warning(
                    "[OmniVoiceBackend] No CUDA GPU found. Will use CPU (very slow)."
                )
        except ImportError:
            logger.warning(
                "[OmniVoiceBackend] PyTorch not available. Cannot check GPU info."
            )

    def cleanup(self):
        """Clean up temporary files and resources."""
        logger.info("[OmniVoiceBackend] Cleanup called")
        self._cleanup_chunk_temp()
        gc.collect()

    # =================================================================
    # GENERATION INTERFACE
    # =================================================================

    def generate_batch(
        self, texts: List[str], output_paths: List[str]
    ) -> Tuple[bool, List[str]]:
        self._validate_batch_inputs(texts, output_paths)

        mode = self.config.get("omnivoice_mode", "auto")
        use_batch = self.config.get("omnivoice_use_batch", True) and self._batch_cmd

        if use_batch:
            try:
                return self._generate_batch_subprocess(texts, output_paths, mode)
            except Exception as e:
                logger.warning(
                    f"[OmniVoiceBackend] Batch inference failed ({e}). "
                    "Falling back to individual inference."
                )
                return self._generate_individual(texts, output_paths, mode)
        else:
            return self._generate_individual(texts, output_paths, mode)

    # =================================================================
    # BATCH SUBPROCESS (omnivoice-infer-batch)
    # =================================================================

    def _generate_batch_subprocess(self, texts, output_paths, mode):
        """Use omnivoice-infer-batch with a JSONL file.

        This loads the model once and processes all items, which is
        MUCH faster than calling omnivoice-infer per slide.
        """
        temp_dir = self._get_local_temp_dir("omnivoice_batch")
        res_dir = os.path.join(temp_dir, "results")
        os.makedirs(res_dir, exist_ok=True)

        jsonl_path = os.path.join(temp_dir, "test_list.jsonl")
        model_id = self.config.get("omnivoice_model_id", "k2-fsa/OmniVoice")
        language = self.config.get("omnivoice_language", "English")
        language_id = self.OMNIVOICE_LANGUAGES.get(language, "en")

        # Build JSONL entries
        id_to_path = {}

        with open(jsonl_path, "w", encoding="utf-8") as f:
            for i, (text, out_path) in enumerate(zip(texts, output_paths)):
                entry_id = f"slide_{i:04d}"
                entry = {
                    "id": entry_id,
                    "text": text.strip(),
                    "language_id": language_id,
                }

                if mode == "voice_clone":
                    ref_audio = self.config.get("omnivoice_ref_audio", "")
                    ref_text = self.config.get("omnivoice_ref_text", "")
                    if ref_audio and os.path.exists(ref_audio):
                        entry["ref_audio"] = os.path.abspath(ref_audio)
                    if ref_text:
                        entry["ref_text"] = ref_text
                elif mode == "voice_design":
                    instruct = self.config.get("omnivoice_instruct", "")
                    if instruct:
                        entry["instruct"] = instruct

                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                id_to_path[entry_id] = out_path

        # Build command
        cmd = [
            self._batch_cmd,
            "--model",
            model_id,
            "--test_list",
            jsonl_path,
            "--res_dir",
            res_dir,
        ]

        logger.info(
            f"[OmniVoiceBackend] Running batch inference for {len(texts)} slides..."
        )

        # Set up environment with cache directories
        env = os.environ.copy()
        cache_dir = self.config.get("hf_cache_dir", "")
        if cache_dir:
            env["HF_HOME"] = cache_dir
            env["HUGGINGFACE_HUB_CACHE"] = cache_dir
            env["TRANSFORMERS_CACHE"] = cache_dir

        # Run the batch command
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=900,  # 15 minute timeout for batch
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("omnivoice-infer-batch timed out after 15 minutes")

        if result.returncode != 0:
            stderr_preview = result.stderr[:500] if result.stderr else "No error output"
            raise RuntimeError(
                f"omnivoice-infer-batch failed (exit {result.returncode}): "
                f"{stderr_preview}"
            )

        logger.info("[OmniVoiceBackend] Batch inference completed. Collecting results...")

        # Collect results — map generated files to output paths
        errors = []
        success_count = 0

        for entry_id, out_path in id_to_path.items():
            generated = None

            # Check for output file with various naming patterns
            candidates = [
                os.path.join(res_dir, f"{entry_id}.wav"),
                os.path.join(res_dir, entry_id, f"{entry_id}.wav"),
                os.path.join(res_dir, f"{entry_id}_0.wav"),
            ]

            for candidate in candidates:
                if os.path.exists(candidate):
                    generated = candidate
                    break

            # If not found, search the results directory
            if not generated and os.path.isdir(res_dir):
                for fname in os.listdir(res_dir):
                    if fname.startswith(entry_id) and fname.endswith(".wav"):
                        generated = os.path.join(res_dir, fname)
                        break
                # Also check subdirectories
                if not generated:
                    for root, dirs, files in os.walk(res_dir):
                        for fname in files:
                            if entry_id in fname and fname.endswith(".wav"):
                                generated = os.path.join(root, fname)
                                break
                        if generated:
                            break

            if generated and os.path.exists(generated):
                try:
                    os.makedirs(
                        os.path.dirname(out_path) if os.path.dirname(out_path) else ".",
                        exist_ok=True,
                    )
                    shutil.copy(generated, out_path)
                    success_count += 1
                    logger.info(
                        f"[OmniVoiceBackend] Collected: {entry_id} -> {out_path}"
                    )
                except Exception as e:
                    errors.append(f"Failed to copy audio for {entry_id}: {e}")
            else:
                errors.append(f"Audio file not generated for {entry_id}")
                logger.warning(f"[OmniVoiceBackend] No output found for {entry_id}")

        logger.info(
            f"[OmniVoiceBackend] Batch results: {success_count}/{len(texts)} successful"
        )

        if success_count == len(texts):
            return True, []
        else:
            return False, errors

    # =================================================================
    # INDIVIDUAL SUBPROCESS (omnivoice-infer)
    # =================================================================

    def _generate_individual(self, texts, output_paths, mode):
        """Generate audio one at a time using omnivoice-infer.

        Fallback when batch inference is not available or fails.
        Each call loads the model fresh, so it's slower than batch mode.
        """
        model_id = self.config.get("omnivoice_model_id", "k2-fsa/OmniVoice")
        language = self.config.get("omnivoice_language", "English")

        enable_chunking = self.config.get("enable_text_chunking", True)
        chunker = self._get_chunker() if enable_chunking else None
        temp_dir = self._get_local_temp_dir("omnivoice_individual")

        errors = []
        success_count = 0

        for i, (text, out_path) in enumerate(zip(texts, output_paths)):
            logger.info(
                f"[OmniVoiceBackend] Generating slide {i + 1}/{len(texts)}..."
            )

            self._report_progress(
                "chunk", i + 1, len(texts), f"Slide {i + 1}/{len(texts)}"
            )

            success, error = self._generate_single_with_chunking(
                text,
                out_path,
                i,
                language,
                chunker,
                self._generate_single_cli,
                self._concatenate_wav_files,
                temp_dir,
            )

            if success:
                success_count += 1
            elif error:
                errors.append(error)

        self._cleanup_chunk_temp()

        if success_count == len(texts):
            return True, []
        else:
            return False, errors

    def _generate_single_cli(self, text: str, output_path: str) -> bool:
        """Generate audio for a single text using omnivoice-infer CLI."""
        if not text or not text.strip():
            return False

        mode = self.config.get("omnivoice_mode", "auto")
        model_id = self.config.get("omnivoice_model_id", "k2-fsa/OmniVoice")

        cmd = [
            self._infer_cmd,
            "--model",
            model_id,
            "--text",
            text.strip(),
            "--output",
            output_path,
        ]

        if mode == "voice_clone":
            ref_audio = self.config.get("omnivoice_ref_audio", "")
            ref_text = self.config.get("omnivoice_ref_text", "")
            if ref_audio and os.path.exists(ref_audio):
                cmd.extend(["--ref_audio", ref_audio])
            if ref_text:
                cmd.extend(["--ref_text", ref_text])
        elif mode == "voice_design":
            instruct = self.config.get("omnivoice_instruct", "")
            if instruct:
                cmd.extend(["--instruct", instruct])

        # Set up environment
        env = os.environ.copy()
        cache_dir = self.config.get("hf_cache_dir", "")
        if cache_dir:
            env["HF_HOME"] = cache_dir
            env["HUGGINGFACE_HUB_CACHE"] = cache_dir
            env["TRANSFORMERS_CACHE"] = cache_dir

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=300,  # 5 minute timeout per slide
            )

            if result.returncode != 0:
                stderr_preview = (
                    result.stderr[:300] if result.stderr else "No error output"
                )
                logger.error(
                    f"[OmniVoiceBackend] omnivoice-infer failed: {stderr_preview}"
                )
                return False

            # Verify output file exists
            if not os.path.exists(output_path):
                logger.warning(
                    f"[OmniVoiceBackend] Output not found at expected path: "
                    f"{output_path}"
                )
                return False

            # Convert to WAV if needed
            if not output_path.endswith(".wav"):
                wav_path = output_path.rsplit(".", 1)[0] + ".wav"
                if self._convert_to_wav(output_path, wav_path):
                    shutil.move(wav_path, output_path)

            return True

        except subprocess.TimeoutExpired:
            logger.error(
                f"[OmniVoiceBackend] Generation timed out for: {text[:50]}..."
            )
            return False
        except Exception as e:
            logger.error(f"[OmniVoiceBackend] Generation failed: {e}")
            return False

    # =================================================================
    # AUDIO UTILITIES
    # =================================================================

    def _convert_to_wav(self, input_path: str, output_path: str) -> bool:
        """Convert audio file to WAV format using FFmpeg."""
        try:
            ffmpeg_path = self.config.get("ffmpeg_path", "ffmpeg")
            result = subprocess.run(
                [
                    ffmpeg_path,
                    "-y",
                    "-i",
                    input_path,
                    "-ar",
                    "24000",
                    "-ac",
                    "1",
                    output_path,
                ],
                capture_output=True,
                timeout=30,
            )
            return result.returncode == 0
        except Exception as e:
            logger.warning(f"[OmniVoiceBackend] Audio conversion failed: {e}")
            return False

    def _concatenate_wav_files(
        self, wav_files: List[str], output_path: str, add_silence_ms: int = 100
    ) -> bool:
        """Concatenate multiple WAV files into a single output file."""
        if not wav_files:
            return False

        if len(wav_files) == 1:
            try:
                shutil.copy(wav_files[0], output_path)
                return True
            except Exception:
                return False

        try:
            audio_data = []
            sample_rate = None
            n_channels = None
            sampwidth = None

            for wav_path in wav_files:
                if not os.path.exists(wav_path):
                    logger.error(f"WAV file not found: {wav_path}")
                    return False

                with wave.open(wav_path, "rb") as wf:
                    if sample_rate is None:
                        sample_rate = wf.getframerate()
                        n_channels = wf.getnchannels()
                        sampwidth = wf.getsampwidth()

                    frames = wf.readframes(wf.getnframes())
                    audio_data.append(frames)

                    # Add silence between chunks
                    if wav_path != wav_files[-1] and add_silence_ms > 0:
                        silence_samples = int(sample_rate * add_silence_ms / 1000)
                        silence_frames = (
                            b"\x00" * (silence_samples * sampwidth * n_channels)
                        )
                        audio_data.append(silence_frames)

            out_dir = os.path.dirname(output_path) if os.path.dirname(output_path) else "."
            os.makedirs(out_dir, exist_ok=True)

            with wave.open(output_path, "wb") as out_wf:
                out_wf.setnchannels(n_channels)
                out_wf.setsampwidth(sampwidth)
                out_wf.setframerate(sample_rate)
                for data in audio_data:
                    out_wf.writeframes(data)

            return True

        except Exception as e:
            logger.error(f"[OmniVoiceBackend] WAV concatenation failed: {e}")
            return False

    def _get_chunker(self) -> TextChunker:
        """Get a configured TextChunker instance from config settings."""
        config = ChunkingConfig(
            max_chars=self.config.get("chunk_max_chars", 500),
            max_sentences=self.config.get("chunk_max_sentences", 5),
            min_chunk_chars=self.config.get("chunk_min_chars", 50),
            language=self.config.get("omnivoice_language", "English"),
        )
        return TextChunker(config)