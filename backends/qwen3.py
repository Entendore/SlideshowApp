# backends/qwen3.py
"""
Qwen3 TTS Backend Plugin.
Strictly implements official API for 3 modes:
1. Voice Clone (Base): generate_voice_clone(ref_audio, ref_text)
2. Voice Design: generate_voice_design(instruct)
3. Voice Custom (CustomVoice): generate_custom_voice(speaker, instruct)

FEATURES:
- Long text handling with automatic chunking
- Audio concatenation for chunked segments
- Progress reporting for multi-chunk generation
"""

import os
import gc
import logging
import numpy as np
import glob
import wave
import tempfile
from datetime import datetime
from typing import List, Tuple, Optional, Dict

# Import text chunker for long text handling
from text_chunker import TextChunker, ChunkingConfig, get_chunker_for_language

# ==============================================================================
# LOGIC IMPORTS
# ==============================================================================
try:
    from qwen_tts import Qwen3TTSModel
    QWEN_AVAILABLE = True
except ImportError:
    Qwen3TTSModel = None
    QWEN_AVAILABLE = False

try:
    import soundfile as sf
except ImportError:
    raise ImportError("Qwen3 Backend requires 'soundfile'. Install via pip.")

# Check for Flash Attention 2 availability
try:
    import flash_attn
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False

# ==============================================================================
# UI IMPORTS
# ==============================================================================
from PySide6.QtWidgets import (
    QWidget, QFormLayout, QComboBox, QLineEdit, 
    QPlainTextEdit, QHBoxLayout, QLabel, QPushButton, QFileDialog, QMessageBox
)

from PySide6.QtCore import QTimer, Qt
import importlib.util
from backends.base import BaseTTSBackend
logger = logging.getLogger(__name__)


class Qwen3Backend(BaseTTSBackend):
    """
    Concrete implementation of BaseTTSBackend for Qwen3.
    """

    DESCRIPTION = "Qwen-Audio: High quality speech synthesis. Supports emotion and style. Requires moderate VRAM."

    AUDIO_SETTINGS_KEYS = [
        "qwen3_mode", "qwen3_size",
        "qwen3_device_map", "qwen3_dtype",
        "qwen3_ref_audio", "qwen3_ref_text",
        "qwen3_vd_description", "qwen3_vd_save_name",
        "qwen3_base_model_id", "qwen3_voicedesign_model_id", "qwen3_custom_model_id",
        "qwen3_speaker", "qwen3_instruct", "qwen3_language",
        "voice_references_root", "instruction_folder_root",
    ]

    QWEN3_LANGUAGES = [
        "Chinese", "English", "Japanese", "Korean", 
        "German", "French", "Russian", "Portuguese", "Spanish", "Italian"
    ]

    # DEFINED MODES
    QWEN3_MODES = [
        ("voice_custom", "Voice Custom (Preset Speakers)"),
        ("voice_clone", "Voice Clone (User Reference)"),
        ("voice_design", "Voice Design (Generate Reference)"),
    ]

    @staticmethod
    def get_ui_options() -> Dict:
        return {
            "modes": Qwen3Backend.QWEN3_MODES,
            "languages": Qwen3Backend.QWEN3_LANGUAGES
        }
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if PyTorch is installed WITHOUT importing it (saves ~1GB RAM)."""
        return importlib.util.find_spec("torch") is not None

    # =================================================================
    # HELPER: DYNAMIC MODEL ID
    # =================================================================
    def _get_model_id(self) -> str:
        mode = self.config.get("qwen3_mode", "voice_custom")
        size = self.config.get("qwen3_size", "1.7B")
        
        # Check for user-overridden model IDs first
        if mode == "voice_design":
            custom = self.config.get("qwen3_voicedesign_model_id", "")
            if custom:
                return custom
            suffix = "VoiceDesign"
        elif mode == "voice_clone":
            custom = self.config.get("qwen3_base_model_id", "")
            if custom:
                return custom
            suffix = "Base"
        elif mode == "voice_custom":
            custom = self.config.get("qwen3_custom_model_id", "")
            if custom:
                return custom
            suffix = "CustomVoice"
        else:
            suffix = "CustomVoice"
            
        return f"Qwen/Qwen3-TTS-12Hz-{size}-{suffix}"

    # =================================================================
    # HELPER: LOCAL MODEL SEARCH
    # =================================================================
    @staticmethod
    def _find_local_model_path(model_id: str, cache_dir: str) -> str:
        if not cache_dir or not os.path.exists(cache_dir):
            return model_id 

        model_author, model_name = model_id.split("/", 1) if "/" in model_id else ("", model_id)
        
        candidates = []
        
        # 1. Standard HF Structure
        hf_folder_name = f"models--{model_author}--{model_name}"
        hf_path = os.path.join(cache_dir, hf_folder_name)
        if os.path.isdir(hf_path):
            snapshots_path = os.path.join(hf_path, "snapshots")
            if os.path.isdir(snapshots_path):
                try:
                    snapshots = os.listdir(snapshots_path)
                    if snapshots:
                        candidates.append(os.path.join(snapshots_path, snapshots[0]))
                except Exception:
                    pass

        # 2. Manual Flat Naming
        candidates.append(os.path.join(cache_dir, model_name))
        candidates.append(os.path.join(cache_dir, model_id.replace("/", "_")))

        # 3. Validate
        for path in candidates:
            if os.path.isdir(path):
                has_config = os.path.exists(os.path.join(path, "config.json"))
                has_weights = glob.glob(os.path.join(path, "*.safetensors")) or \
                              glob.glob(os.path.join(path, "pytorch_model*.bin"))
                
                if has_config and has_weights:
                    logger.info(f"[Qwen3Backend] Found valid local model: {path}")
                    return path

        logger.warning(f"[Qwen3Backend] Model '{model_id}' not found locally. Falling back to download.")
        return model_id

    # =================================================================
    # HELPER: SMART ATTENTION (FIXED PER USER REQUEST)
    # =================================================================
    @staticmethod
    def _get_smart_attn_implementation(device: str, requested_attn: str) -> str:
        """
        Determines the best attention implementation.
        Logic: 
        - If Flash Attention 2 is requested:
            - Use it if library is installed AND GPU is Ampere (RTX 30/40) or newer.
            - Otherwise, fallback to 'eager'.
        - Otherwise: return the requested implementation ('eager', 'sdpa').
        """
        # 1. CPU always eager
        if not device.startswith("cuda"):
            return "eager"
            
        # 2. Handle Flash Attention 2 Request
        if requested_attn == "flash_attention_2":
            # Check Hardware Capability (Requires Compute Capability 8.0+)
            is_hardware_supported = False
            try:
                if torch.cuda.is_available():
                    idx = int(device.split(":")[1]) if ":" in device else 0
                    major, minor = torch.cuda.get_device_capability(idx)
                    if major >= 8:
                        is_hardware_supported = True
                    else:
                        logger.warning(f"[Qwen3Backend] GPU compute capability {major}.{minor} < 8.0. Flash Attention 2 requires Ampere+. Falling back to Eager.")
            except Exception as e:
                logger.warning(f"[Qwen3Backend] Could not detect GPU capability: {e}")

            # Check Library Installation
            if is_hardware_supported:
                if FLASH_ATTN_AVAILABLE:
                    return "flash_attention_2"
                else:
                    logger.warning("[Qwen3Backend] flash_attn library not installed. Falling back to Eager.")
            
            # Fallback if hardware or lib missing
            return "eager"
        
        # 3. Return other requests as-is (e.g., if user explicitly set 'eager' or 'sdpa')
        return requested_attn

    # =================================================================
    # UI GENERATION FACTORY
    # =================================================================
    @staticmethod
    def get_settings_widget(mode: str, config, parent=None, save_callback=None) -> QWidget:
        widget = QWidget(parent)
        layout = QFormLayout(widget)

        def save(key, value):
            if save_callback:
                save_callback(key,value)
            else:
                config.set(key,value)
        
        # ---------------------------------------------------------
        # MODE 1: Voice Custom
        # ---------------------------------------------------------
        if mode == "voice_custom":
            layout.addRow(QLabel("<b>Voice Custom Configuration</b>"))
            layout.addRow(QLabel("<i>Model: Qwen3-TTS-12Hz-1.7B-CustomVoice</i>"))
            
            speaker_edit = QLineEdit()
            speaker_edit.setText(config.get("qwen3_speaker", "Vivian"))
            speaker_edit.setPlaceholderText("e.g., Vivian, Ryan")

            speaker_edit.editingFinished.connect(lambda: save("qwen3_speaker", speaker_edit.text()))
            layout.addRow("Speaker Name:", speaker_edit)
            
            instruct_edit = QLineEdit()
            instruct_edit.setText(config.get("qwen3_instruct", ""))
            instruct_edit.setPlaceholderText("e.g., 'Speak slowly and sadly'")
            instruct_edit.editingFinished.connect(lambda: save("qwen3_instruct", instruct_edit.text()))
            layout.addRow("Style Instruction:", instruct_edit)
            
            lang_combo = QComboBox()
            lang_combo.addItems(Qwen3Backend.QWEN3_LANGUAGES)
            lang_combo.setCurrentText(config.get("qwen3_language", "English"))
            lang_combo.currentTextChanged.connect(lambda t: save("qwen3_language", t))
            layout.addRow("Language:", lang_combo)

        # ---------------------------------------------------------
        # MODE 2: Voice Clone
        # ---------------------------------------------------------
        elif mode == "voice_clone":
            layout.addRow(QLabel("<b>Voice Clone Configuration</b>"))
            layout.addRow(QLabel("<i>Model: Qwen3-TTS-12Hz-1.7B-Base</i>"))
            
            ref_row = QHBoxLayout()
            ref_audio_edit = QLineEdit()
            ref_audio_edit.setText(config.get("qwen3_ref_audio", ""))
            ref_audio_edit.setPlaceholderText("Path to reference .wav file")
            ref_audio_edit.editingFinished.connect(lambda: save("qwen3_ref_audio", ref_audio_edit.text()))
            
            btn_browse = QPushButton("...")
            btn_browse.setMaximumWidth(30)
            btn_browse.clicked.connect(lambda: Qwen3Backend._browse_audio(ref_audio_edit, ref_text_edit, config))
            
            ref_row.addWidget(ref_audio_edit)
            ref_row.addWidget(btn_browse)
            layout.addRow("Reference Audio:", ref_row)
            
            ref_text_edit = QLineEdit()
            ref_text_edit.setText(config.get("qwen3_ref_text", ""))
            ref_text_edit.setPlaceholderText("Transcript of reference audio")
            ref_text_edit.editingFinished.connect(lambda: save("qwen3_ref_text", ref_text_edit.text()))
            layout.addRow("Reference Text:", ref_text_edit)
            
            lang_combo = QComboBox()
            lang_combo.addItems(Qwen3Backend.QWEN3_LANGUAGES)
            lang_combo.setCurrentText(config.get("qwen3_language", "English"))
            lang_combo.currentTextChanged.connect(lambda t: save("qwen3_language", t))
            layout.addRow("Language:", lang_combo)

        # ---------------------------------------------------------
        # MODE 3: Voice Design
        # ---------------------------------------------------------
        elif mode == "voice_design":
            layout.addRow(QLabel("<b>Voice Design Configuration</b>"))
            layout.addRow(QLabel("<i>Model: Qwen3-TTS-12Hz-1.7B-VoiceDesign</i>"))
            
            desc_edit = QPlainTextEdit()
            desc_edit.setPlaceholderText("Describe the voice (e.g., 'A deep male voice, speaking slowly')")
            desc_edit.setMaximumHeight(100)
            desc_edit.setPlainText(config.get("qwen3_vd_description", ""))
            # Debounce: only save after user stops typing for 800ms
            _vd_desc_timer = QTimer(widget)
            _vd_desc_timer.setSingleShot(True)
            _vd_desc_timer.setInterval(800)
            _vd_desc_timer.timeout.connect(lambda: save("qwen3_vd_description", desc_edit.toPlainText()))
            desc_edit.textChanged.connect(_vd_desc_timer.start)
            layout.addRow("Voice Description:", desc_edit)
            
            save_name_edit = QLineEdit()
            save_name_edit.setText(config.get("qwen3_vd_save_name", "my_designed_voice"))
            save_name_edit.editingFinished.connect(lambda: save("qwen3_vd_save_name", save_name_edit.text()))
            layout.addRow("Save Reference As:", save_name_edit)

            lang_combo = QComboBox()
            lang_combo.addItems(Qwen3Backend.QWEN3_LANGUAGES)
            lang_combo.setCurrentText(config.get("qwen3_language", "English"))
            lang_combo.currentTextChanged.connect(lambda t: save("qwen3_language", t))
            layout.addRow("Language:", lang_combo)

        return widget

    @staticmethod
    def _browse_audio(line_edit, text_edit, config):
        fname, _ = QFileDialog.getOpenFileName(
            None, 
            "Select Reference Audio", 
            "", 
            "Audio Files (*.wav *.mp3 *.flac)"
        )
        if fname:
            line_edit.setText(fname)

            config.set("qwen3_ref_audio", fname)
            
            if text_edit:
                # 1. Construct the expected path (e.g. /path/to/whipservoice.txt)
                base_path = os.path.splitext(fname)[0]
                txt_path = base_path + ".txt"
                
                # 2. Check for the file
                found_path = None
                
                if os.path.exists(txt_path):
                    found_path = txt_path
                else:
                    # 3. Fallback: Case-Insensitive Search
                    dir_path = os.path.dirname(fname)
                    if not dir_path:
                        dir_path = "."
                    
                    desired_name = os.path.basename(txt_path)
                    try:
                        files_in_dir = os.listdir(dir_path)
                        for f in files_in_dir:
                            if f.lower() == desired_name.lower():
                                found_path = os.path.join(dir_path, f)
                                logger.info(f"[Qwen3Backend] Found text file via case-insensitive search: {found_path}")
                                break
                    except Exception as e:
                        logger.warning(f"Could not list directory for fallback search: {e}")

                # 4. Load or Show Error
                if found_path:
                    try:
                        with open(found_path, 'r', encoding='utf-8') as f:
                            content = f.read().strip()
                            text_edit.setText(content)
                            config.set("qwen3_ref_text", content)
                            logger.info(f"[Qwen3Backend] Auto-loaded reference text from: {found_path}")
                    except Exception as e:
                        logger.error(f"[Qwen3Backend] Failed to read text file {found_path}: {e}")
                        QMessageBox.warning(None, "Read Error", f"Found a text file at {found_path} but could not read it.\nError: {e}")
                else:
                    logger.warning(f"[Qwen3Backend] Reference text file not found at: {txt_path}")
                    QMessageBox.warning(
                        None, 
                        "Transcript Not Found", 
                        f"Could not find a matching transcript file.\n\n"
                        f"I am looking for this file:\n{txt_path}\n\n"
                        f"Please ensure the text file has the exact same name as the audio file."
                    )

    # =================================================================
    # INITIALIZATION
    # =================================================================

    def __init__(self, config: dict):
        super().__init__(config)
        self.model = None
        self.device = config.get("qwen3_device_map", "cuda:0")
        
        if not QWEN_AVAILABLE:
            raise RuntimeError("Qwen3 library (qwen-tts) is not installed.")

    def initialize(self):
        import torch
        import transformers
    
        if self.model is not None:
            self.cleanup()
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        mode = self.config.get("qwen3_mode")
        logger.info(f"[Qwen3Backend] Initializing for mode: {mode}")

        model_id = self._get_model_id()
        cache_dir = self.config.get("hf_cache_dir")
        model_path = self._find_local_model_path(model_id, cache_dir)

        requested_attn = self.config.get("qwen3_attn_implementation", "flash_attention_2")
        actual_attn = self._get_smart_attn_implementation(self.device, requested_attn)
        
        dtype_str = self.config.get("qwen3_dtype", "float16")
        dtype = getattr(torch, dtype_str, torch.float16) 

        try:
            logger.info(f"[Qwen3Backend] Loading model from: {model_path}")
            logger.info(f"[Qwen3Backend] Device: {self.device}, Dtype: {dtype}, Attention: {actual_attn}")
            
            is_local = os.path.isdir(model_path)
            
            self.model = Qwen3TTSModel.from_pretrained(
                model_path,
                local_files_only=is_local,
                device_map=self.device,
                torch_dtype=dtype,
                attn_implementation=actual_attn
            )
            logger.info("[Qwen3Backend] Model loaded successfully.")
            
        except Exception as e:
            logger.error(f"[Qwen3Backend] Failed to load model: {e}")
            self.model = None
            raise

    def cleanup(self):
        if self.model is not None:
            logger.info("[Qwen3Backend] Cleaning up model resources...")
            del self.model
            self.model = None
        self._cleanup_chunk_temp()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # =================================================================
    # GENERATION INTERFACE
    # =================================================================

    def generate_batch(self, texts: List[str], output_paths: List[str]) -> Tuple[bool, List[str]]:
        # Validate that we have at least some non-empty text
        self._validate_batch_inputs(texts, output_paths)
        valid_texts = [t for t in texts if t and t.strip()]
        if not valid_texts:
            raise ValueError("Cannot generate audio: all texts are empty or whitespace only")
        
        if self.model is None:
            try:
                self.initialize()
            except Exception as e:
                return False, [f"Initialization failed: {e}"]

        mode = self.config.get("qwen3_mode")
        try:
            if mode == "voice_custom":
                return self._gen_voice_custom(texts, output_paths)
            elif mode == "voice_clone":
                return self._gen_voice_clone(texts, output_paths)
            elif mode == "voice_design":
                return self._gen_voice_design(texts, output_paths)
            else:
                return False, [f"Unknown mode: {mode}"]
        except Exception as e:
            logger.exception("[Qwen3Backend] Generation failed")
            return False, [str(e) for _ in texts]

    # =================================================================
    # MODE IMPLEMENTATIONS
    # =================================================================

    def _save_wav(self, wav, sr, path):
        """Helper to save a single wav file, handling tensor conversion."""
        try:
            # Convert Torch Tensor to Numpy if needed
            if hasattr(wav, 'cpu'):
                wav = wav.cpu().numpy()
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            sf.write(path, wav, sr)
            return True
        except Exception as e:
            logger.error(f"Failed to save audio file {path}: {e}")
            return False

    # =================================================================
    # LONG TEXT HANDLING: AUDIO CONCATENATION
    # =================================================================

    def _concatenate_wav_files(self, wav_files: List[str], output_path: str, 
                                add_silence_ms: int = 100) -> bool:
        """
        Concatenate multiple WAV files into a single output file.
        
        Args:
            wav_files: List of paths to WAV files to concatenate
            output_path: Path for the output WAV file
            add_silence_ms: Milliseconds of silence to add between chunks
            
        Returns:
            True if successful, False otherwise
        """
        if not wav_files:
            logger.error("No WAV files to concatenate")
            return False
        
        if len(wav_files) == 1:
            # Single file, just copy it
            try:
                import shutil
                shutil.copy(wav_files[0], output_path)
                return True
            except Exception as e:
                logger.error(f"Failed to copy single WAV file: {e}")
                return False
        
        logger.info(f"[Qwen3Backend] Concatenating {len(wav_files)} audio chunks...")
        
        try:
            # Read all WAV files and collect audio data
            audio_data = []
            sample_rate = None
            
            for wav_path in wav_files:
                if not os.path.exists(wav_path):
                    logger.error(f"WAV file not found: {wav_path}")
                    return False
                
                with wave.open(wav_path, 'rb') as wf:
                    if sample_rate is None:
                        sample_rate = wf.getframerate()
                        n_channels = wf.getnchannels()
                        sampwidth = wf.getsampwidth()
                    elif wf.getframerate() != sample_rate:
                        logger.warning(f"Sample rate mismatch in {wav_path}, converting...")
                    
                    frames = wf.readframes(wf.getnframes())
                    audio_data.append(frames)
                    
                    # Add silence between chunks (except after the last one)
                    if wav_path != wav_files[-1] and add_silence_ms > 0:
                        silence_samples = int(sample_rate * add_silence_ms / 1000)
                        silence_frames = b'\x00' * (silence_samples * sampwidth * n_channels)
                        audio_data.append(silence_frames)
            
            # Write concatenated output
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with wave.open(output_path, 'wb') as out_wf:
                out_wf.setnchannels(n_channels)
                out_wf.setsampwidth(sampwidth)
                out_wf.setframerate(sample_rate)
                
                for data in audio_data:
                    out_wf.writeframes(data)
            
            logger.info(f"[Qwen3Backend] Successfully concatenated to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to concatenate WAV files: {e}")
            return False

    def _get_chunker(self) -> TextChunker:
        """Get a configured TextChunker instance from config settings."""
        config = ChunkingConfig(
            max_chars=self.config.get("chunk_max_chars", 500),
            max_sentences=self.config.get("chunk_max_sentences", 5),
            min_chunk_chars=self.config.get("chunk_min_chars", 50),
            language=self.config.get("qwen3_language", "English")
        )
        return TextChunker(config)

    def _generate_chunks_with_progress(self, chunks: List[Tuple[str, float]], 
                                        output_dir: str, 
                                        generate_func, 
                                        generate_kwargs: dict,
                                        slide_index: int = 0) -> Tuple[bool, List[str]]:
        """
        Generate audio for multiple text chunks and return paths to generated files.
        """
        chunk_paths = []
        total_chunks = len(chunks)
        
        for chunk_idx, (chunk_text, estimated_duration) in enumerate(chunks):
            chunk_path = os.path.join(output_dir, f"chunk_{slide_index}_{chunk_idx}.wav")
            
            logger.info(f"[Qwen3Backend] Generating chunk {chunk_idx + 1}/{total_chunks} "
                       f"({len(chunk_text)} chars, ~{estimated_duration:.1f}s)...")
            
            self._report_progress(
                "chunk", 
                chunk_idx + 1, 
                total_chunks, 
                f"Chunk {chunk_idx + 1}/{total_chunks}"
            )
            
            try:
                result = generate_func(text=[chunk_text], **generate_kwargs)
                
                wavs, sr = result
                
                if isinstance(wavs, list):
                    wav = wavs[0]
                else:
                    wav = wavs
                
                if self._save_wav(wav, sr, chunk_path):
                    chunk_paths.append(chunk_path)
                else:
                    return False, [f"Failed to save chunk {chunk_idx + 1}"]
                    
            except Exception as e:
                logger.error(f"Failed to generate chunk {chunk_idx + 1}: {e}")
                return False, [f"Chunk {chunk_idx + 1} generation failed: {e}"]
        
        return True, chunk_paths

    def _gen_voice_custom(self, texts: List[str], paths: List[str]) -> Tuple[bool, List[str]]:
        speaker = self.config.get("qwen3_speaker", "Vivian")
        instruct = self.config.get("qwen3_instruct", "")
        language = self.config.get("qwen3_language", "English")
        enable_chunking = self.config.get("enable_text_chunking", True)

        logger.info(f"[Qwen3Backend] Running Voice Custom (Speaker: {speaker}) for {len(texts)} items.")
        
        errors = []
        success_count = 0
        
        chunker = self._get_chunker() if enable_chunking else None
        temp_dir = self._get_local_temp_dir("chunks_custom")

        for i, text in enumerate(texts):
            logger.info(f"[Qwen3Backend] Generating slide {i+1}/{len(texts)}...")
            
            needs_chunking = chunker and chunker.needs_chunking(text)
            
            if needs_chunking:
                logger.info(f"[Qwen3Backend] Slide {i+1}: Long text detected ({len(text)} chars), chunking...")
                chunks = chunker.chunk_text(text, language)
                logger.info(f"[Qwen3Backend] Slide {i+1}: Split into {len(chunks)} chunks")
                
                gen_kwargs = {
                    "language": [language],
                    "speaker": [speaker],
                    "instruct": [instruct]
                }
                
                success, result = self._generate_chunks_with_progress(
                    chunks, temp_dir, 
                    self.model.generate_custom_voice,
                    gen_kwargs,
                    slide_index=i
                )
                
                if success:
                    if self._concatenate_wav_files(result, paths[i]):
                        success_count += 1
                        for chunk_path in result:
                            try:
                                os.remove(chunk_path)
                            except:
                                pass
                    else:
                        errors.append(f"Failed to concatenate chunks for slide {i+1}")
                else:
                    errors.extend(result)
                    
            else:
                try:
                    result = self.model.generate_custom_voice(
                        text=[text], 
                        language=[language],
                        speaker=[speaker],
                        instruct=[instruct]
                    )
                    
                    wavs, sr = result
                    
                    if isinstance(wavs, list):
                        wav = wavs[0]
                    else:
                        wav = wavs

                    if self._save_wav(wav, sr, paths[i]):
                        success_count += 1
                    else:
                        errors.append(f"Failed to save file for slide {i+1}")
                        
                except Exception as e:
                    logger.error(f"Voice Custom failed for slide {i+1}: {e}")
                    errors.append(str(e))
        
        self._cleanup_chunk_temp()

        if success_count == len(texts):
            return True, []
        else:
            return False, errors

    def _gen_voice_clone(self, texts: List[str], paths: List[str]) -> Tuple[bool, List[str]]:
        ref_audio_path = self.config.get("qwen3_ref_audio", "")
        ref_text = self.config.get("qwen3_ref_text", "")
        language = self.config.get("qwen3_language", "English")
        enable_chunking = self.config.get("enable_text_chunking", True)

        if not ref_audio_path or not os.path.exists(ref_audio_path):
            return False, ["Reference audio file not found."]

        if not ref_text:
            return False, ["Reference text is required for Voice Clone."]

        logger.info(f"[Qwen3Backend] Running Voice Clone for {len(texts)} items.")

        prompt_items = None
        if hasattr(self.model, 'create_voice_clone_prompt'):
            try:
                logger.info("[Qwen3Backend] Creating optimized voice clone prompt...")
                prompt_items = self.model.create_voice_clone_prompt(
                    ref_audio=ref_audio_path,
                    ref_text=ref_text,
                    x_vector_only_mode=False
                )
            except Exception as e:
                logger.warning(f"Could not create voice clone prompt: {e}")

        errors = []
        success_count = 0
        
        chunker = self._get_chunker() if enable_chunking else None
        temp_dir = self._get_local_temp_dir("chunks_clone")

        for i, text in enumerate(texts):
            logger.info(f"[Qwen3Backend] Generating slide {i+1}/{len(texts)}...")
            
            needs_chunking = chunker and chunker.needs_chunking(text)
            
            if needs_chunking:
                logger.info(f"[Qwen3Backend] Slide {i+1}: Long text detected ({len(text)} chars), chunking...")
                chunks = chunker.chunk_text(text, language)
                logger.info(f"[Qwen3Backend] Slide {i+1}: Split into {len(chunks)} chunks")
                
                gen_kwargs = {
                    "language": [language]
                }
                
                if prompt_items:
                    gen_kwargs["voice_clone_prompt"] = prompt_items
                else:
                    gen_kwargs["ref_audio"] = ref_audio_path
                    gen_kwargs["ref_text"] = ref_text
                
                success, result = self._generate_chunks_with_progress(
                    chunks, temp_dir,
                    self.model.generate_voice_clone,
                    gen_kwargs,
                    slide_index=i
                )
                
                if success:
                    if self._concatenate_wav_files(result, paths[i]):
                        success_count += 1
                        for chunk_path in result:
                            try:
                                os.remove(chunk_path)
                            except:
                                pass
                    else:
                        errors.append(f"Failed to concatenate chunks for slide {i+1}")
                else:
                    errors.extend(result)
                    
            else:
                try:
                    args = {
                        "text": [text],
                        "language": [language]
                    }
                    
                    if prompt_items:
                        args["voice_clone_prompt"] = prompt_items
                    else:
                        args["ref_audio"] = ref_audio_path
                        args["ref_text"] = ref_text
                    
                    wavs, sr = self.model.generate_voice_clone(**args)
                    
                    if isinstance(wavs, list):
                        wav = wavs[0]
                    else:
                        wav = wavs

                    if self._save_wav(wav, sr, paths[i]):
                        success_count += 1
                    else:
                        errors.append(f"Failed to save file for slide {i+1}")

                except Exception as e:
                    logger.error(f"Voice Clone failed for slide {i+1}: {e}")
                    errors.append(str(e))
        
        self._cleanup_chunk_temp()

        if success_count == len(texts):
            return True, []
        else:
            return False, errors

    def _gen_voice_design(self, texts: List[str], paths: List[str]) -> Tuple[bool, List[str]]:
        description = self.config.get("qwen3_vd_description", "")
        save_name = self.config.get("qwen3_vd_save_name", "designed_voice")
        language = self.config.get("qwen3_language", "English")
        enable_chunking = self.config.get("enable_text_chunking", True)
        
        refs_root = self.config.get("voice_references_root", "voicereferences")
        if not os.path.exists(refs_root):
            try:
                os.makedirs(refs_root)
            except OSError:
                pass

        logger.info(f"[Qwen3Backend] Running Voice Design for {len(texts)} items.")

        errors = []
        success_count = 0
        
        chunker = self._get_chunker() if enable_chunking else None
        temp_dir = self._get_local_temp_dir("chunks_design")

        for i, text in enumerate(texts):
            logger.info(f"[Qwen3Backend] Generating slide {i+1}/{len(texts)}...")
            
            needs_chunking = chunker and chunker.needs_chunking(text)
            
            if needs_chunking:
                logger.info(f"[Qwen3Backend] Slide {i+1}: Long text detected ({len(text)} chars), chunking...")
                chunks = chunker.chunk_text(text, language)
                logger.info(f"[Qwen3Backend] Slide {i+1}: Split into {len(chunks)} chunks")
                
                gen_kwargs = {
                    "language": [language],
                    "instruct": [description]
                }
                
                success, result = self._generate_chunks_with_progress(
                    chunks, temp_dir,
                    self.model.generate_voice_design,
                    gen_kwargs,
                    slide_index=i
                )
                
                if success:
                    if self._concatenate_wav_files(result, paths[i]):
                        success_count += 1
                        
                        if i == 0:
                            try:
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                ref_filename = f"{save_name}_{timestamp}.wav"
                                ref_save_path = os.path.join(refs_root, ref_filename)
                                import shutil
                                shutil.copy(paths[i], ref_save_path)
                                logger.info(f"[Qwen3Backend] Saved Voice Design Reference to: {ref_save_path}")
                            except Exception as e:
                                logger.warning(f"Failed to save voice design reference: {e}")
                        
                        for chunk_path in result:
                            try:
                                os.remove(chunk_path)
                            except:
                                pass
                    else:
                        errors.append(f"Failed to concatenate chunks for slide {i+1}")
                else:
                    errors.extend(result)
                    
            else:
                try:
                    wavs, sr = self.model.generate_voice_design(
                        text=[text],
                        language=[language],
                        instruct=[description]
                    )

                    if isinstance(wavs, list):
                        wav = wavs[0]
                    else:
                        wav = wavs
                    
                    if self._save_wav(wav, sr, paths[i]):
                        success_count += 1
                        
                        if i == 0:
                            try:
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                ref_filename = f"{save_name}_{timestamp}.wav"
                                ref_save_path = os.path.join(refs_root, ref_filename)
                                if hasattr(wav, 'cpu'): wav = wav.cpu().numpy()
                                sf.write(ref_save_path, wav, sr)
                                logger.info(f"[Qwen3Backend] Saved Voice Design Reference to: {ref_save_path}")
                            except Exception as e:
                                logger.warning(f"Failed to save voice design reference: {e}")
                    else:
                        errors.append(f"Failed to save file for slide {i+1}")

                except Exception as e:
                    logger.error(f"Voice Design failed for slide {i+1}: {e}")
                    errors.append(str(e))
        
        self._cleanup_chunk_temp()

        if success_count == len(texts):
            return True, []
        else:
            return False, errors