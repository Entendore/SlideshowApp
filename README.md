```markdown
# AI Slideshow Renderer

A desktop application for generating narrated video slideshows using AI-powered Text-to-Speech. Built with PySide6, it renders HTML slides into video with synchronized audio, supporting multiple TTS backends, smart caching, and multi-process parallel rendering.

## Features

- **Multi-Backend TTS** — Plugin architecture supporting Qwen3 (local GPU), Edge TTS (free online), and OmniVoice (local GPU with cloning)
- **Multi-Process Rendering** — Configurable parallel workers (1–8) for batch project rendering
- **Smart Caching** — Audio and video are only regenerated when source text, settings, or HTML content actually change (hash-based detection)
- **HTML Slide Engine** — Slides are HTML files rendered headlessly via Playwright + Chromium, then encoded with FFmpeg
- **Long Text Chunking** — Automatically splits long slide text into TTS-safe segments with sentence boundary preservation, then concatenates audio
- **Slide Transitions** — Configurable crossfade transitions between slides
- **Project Library** — Searchable, sortable project list with status badges and render progress
- **Settings UI** — Categorized settings with debounced auto-save, resolution presets (360p–4K), GPU auto-detection
- **Theming** — Switchable dark themes

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  app.py          MainWindow (PySide6 GUI)               │
│  ├── Project Library (left panel)                       │
│  ├── Tab Editor (settings + per-project slide editors)  │
│  ├── Render Controls (queue, progress, workers)         │
│  └── Application Logs                                   │
├─────────────────────────────────────────────────────────┤
│  engines.py      Multi-process render worker             │
│  ├── Phase 1: Scan slides & smart-cache decisions       │
│  ├── Phase 2: TTS generation via backend plugin         │
│  ├── Phase 3: Build timeline from audio durations       │
│  └── Phase 4: Playwright capture → FFmpeg encode        │
├─────────────────────────────────────────────────────────┤
│  backends/       TTS Backend Plugins                     │
│  ├── base.py     Abstract base class                     │
│  ├── qwen3.py    Qwen3-TTS (Voice Custom / Clone / Design) │
│  ├── edge.py     Microsoft Edge TTS (free, online)       │
│  └── omnivoice.py OmniVoice (local GPU, cloning)         │
├─────────────────────────────────────────────────────────┤
│  text_chunker.py Sentence-aware text splitter for TTS    │
│  config.py       Singleton config manager (settings.json)│
│  dialogs.py      Custom Qt widgets & settings tabs       │
│  utils.py        Helpers, logging, file utilities        │
└─────────────────────────────────────────────────────────┘
```

## Prerequisites

- **Python 3.10+**
- **FFmpeg** — must be on `PATH` or specified in settings
- **CUDA GPU** — required for Qwen3 / OmniVoice backends (Edge TTS is CPU/online only)

## Installation

```bash
# Core dependencies
pip install torch torchaudio
pip install PySide6
pip install Pillow numpy soundfile

# Playwright (headless browser for slide rendering)
pip install playwright
playwright install chromium

# TTS Backends (install what you need)

# Option A: Qwen3 TTS (local GPU)
pip install qwen-tts
pip install flash-attn   # Optional: RTX 30/40 series only

# Option B: Edge TTS (free, online)
pip install edge-tts

# Option C: OmniVoice (local GPU)
pip install omnivoice
```

## Usage

```bash
python app.py
```

### Workflow

1. **Create a Project** — Click "New Project" to create a project folder under `Projects/`
2. **Edit Slides** — Double-click a project to open the slide editor. Each slide is an HTML file (`slide1.html`) with an optional text file (`slide1.txt`) for narration
3. **Configure TTS** — Open ⚙ Settings → Backend Settings. Select your TTS engine, voice, and language
4. **Render** — Select project(s) and click "Render Selected", or use "Render All (Outdated)" to batch-render projects with stale output
5. **Output** — Final videos are saved to the `Output/` directory as `.mp4`

### Project Structure

```
Projects/
└── MyProject/
    ├── slide1.html        # Slide content (HTML)
    ├── slide1.txt         # Narration text for slide 1
    ├── slide1.wav         # Cached audio (auto-generated)
    ├── slide2.html
    ├── slide2.txt
    ├── slide2.wav
    └── manifest.json      # Render metadata & cache hashes
```

## TTS Backends

| Backend | Type | GPU Required | Voice Clone | Voice Design | Internet |
|---------|------|-------------|-------------|--------------|----------|
| **Qwen3** | Local | Yes (CUDA) | ✅ (Base model) | ✅ (VoiceDesign model) | No |
| **Edge TTS** | Online | No | ❌ | ❌ | Yes |
| **OmniVoice** | Local | Yes (CUDA) | ✅ | ✅ | No |


## Smart Cache System

The renderer avoids redundant work by tracking content hashes:

- **Audio Settings Hash** — If TTS backend, voice, language, or reference audio hasn't changed, existing `.wav` files are reused
- **Video Settings Hash** — If resolution, FPS, encoder, or transition settings haven't changed, video re-encoding is skipped
- **File Modification Time** — If `slide1.txt` is newer than `slide1.wav`, audio is regenerated. If HTML is newer than the last render, video is regenerated

This means changing only slide text only regenerates that slide's audio — the entire video is only re-encoded when necessary.

## Configuration

Settings are stored in `settings.json` (auto-created on first run). Key settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `width` / `height` | 1280×720 | Output resolution (presets: 360p–4K, portrait/landscape) |
| `fps` | 30 | Frames per second |
| `encoder` | auto | `libx264`, `h264_nvenc`, or `auto` |
| `render_workers` | 3 | Max parallel render processes |
| `transition_duration` | 0.5s | Crossfade duration between slides |
| `active_backend` | qwen3 | TTS backend to use |
| `hf_cache_dir` | `models_cache/` | HuggingFace model cache path |
| `enable_text_chunking` | true | Split long text for TTS |
| `chunk_max_chars` | 500 | Max characters per TTS chunk |

## Text Chunking

Long narration text is automatically split into sentence-boundary-respecting chunks before TTS generation. This prevents failures on long paragraphs and maintains natural speech patterns.

Supported languages for sentence splitting: English, Chinese, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian.

## License

See repository for license information.
```