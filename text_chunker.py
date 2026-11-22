# text_chunker.py
"""
Text Chunking Utility for TTS Systems.

Handles long text input by splitting into manageable segments while
preserving sentence boundaries and natural speech patterns.

Features:
  - Pre-compiled regex patterns (per-language, cached at module level)
  - Language constants as a Literal type for IDE autocompletion
  - Separated chunking strategies (sentence-based, hard-split)
  - Abbreviation-aware sentence splitting for Latin-script languages
  - URL and SSML-aware splitting (pre-scanned once, not per-boundary)
  - Overlap support at chunk boundaries for smoother TTS continuity
  - Paragraph-aware pre-splitting before sentence grouping
  - Punctuation-pause-aware duration estimation
  - CJK sentence patterns with additional delimiters
  - Mixed-language (CJK + Latin) speech duration estimation
  - Smart min-chunk merging that respects overlap budget
  - TTS preprocessing: HTML/markdown cleanup, abbreviation/symbol
    expansion, emoji removal, and text normalization for speech
  - Number-to-words conversion (integers, decimals, currency, ordinals,
    years, fractions) for natural speech output
  - Expanded abbreviation dictionary (tech, Latin, business) with a
    case-sensitive tier ("US"/"IT"/"ID" never rewritten as words)
  - Enhanced symbol normalization (comparison operators, currency
    symbols, hash, at-sign, tilde, standalone plus/minus)
  - Exponent notation: caret (2^10 → "two to the tenth"), Unicode
    superscripts (10⁶ → "ten to the sixth", xⁿ → "x to the n") and
    subscripts (CO₂ → "CO two")
  - Math glyphs: √ π Δ Ω ∫ ℃ ℉ read as spoken words
  - Programming operators: ==/===/!==/&&/|| spoken correctly
  - Decades and plural years: 1990s → "nineteen nineties",
    the 90s → "the nineties", 1900s → "nineteen hundreds"
  - Tech-term pronunciation table (LaTeX, Nginx, MySQL, IEEE, GIF,
    K8s, i18n, a11y, Ctrl ...) and numeronym expansion
  - Brand pronunciations: lowercase nginx, PyTorch, PySide6, Qt,
    GraphQL, FastAPI, WebGL, OpenGL, TensorRT, VoIP, I2C, SPI, PCIe
  - Common English respellings engines garble: queue → "cue",
    epitome → "eh pit oh mee", hyperbole → "hy per bo lee"
  - IPv4 addresses read digit-by-digit (192.168.1.1 → "one nine two
    dot one six eight dot one dot one", with CIDR /8 → "slash 8")
  - Phone numbers read digit-by-digit (555-123-4567 → "five five
    five, one two three, four five six seven")
  - ISO dates: 2024-01-15 → "January fifteenth, twenty twenty-four"
  - Bare unit durations: 45s → "45 seconds", 24h → "24 hours",
    2w → "2 weeks" (digit-anchored, "10s of thousands" kept intact)
  - Dotted port numbers: "port 8080" → "port eight zero eight zero"
  - Quarters: Q1–Q4 → "quarter one" … "quarter four"; e2e →
    "end to end"; CI/CD → "C I C D"; a.m./p.m. → "A M" / "P M"
  - Rupee currency: ₹1,299 → "one thousand two hundred ninety-nine
    rupees"; paragraph sign ¶ read as "paragraph"; 3*4 and 4×
    read as "3 times 4" / "4 times"; "deg C" → "degrees Celsius"
  - Identifier readability: snake_case and camelCase split into words
    ("my_var_name" → "my var name", "getUserData" → "get User Data")
  - Compound rates: km/h, MB/s, m/s² ("meters per second squared")
    spoken correctly, with or without a leading digit
  - Multiple punctuation normalization (!!! → !, ??? → ?)
  - Improved slash handling (word/word → word or word)
  - Parenthetical pause insertion and hyphen normalization
  - Time expression normalization (5:30 PM → five thirty PM)
  - Math pronunciation pass: tight operators ("3+4" → "3 plus 4",
    "x=5" → "x equals 5"), negative numbers ("-5" → "minus 5"),
    scientific notation (1.5e-3 → "1.5 times 10 to the minus third"),
    Python "**" exponents, primes (f'(x) → "f prime of x"),
    derivatives (dy/dx, d²y/dx², ∂f/∂x), math function calls
    (sin(x) → "sine of x", sqrt(25) → "square root of 25"), log
    bases (log₂ → "log base two"), inverse trigonometrics
    (sin⁻¹(x) → "inverse sine of x", sin^-1(x) likewise), factorial
    (n! → "n factorial"), assignment (":=" → "is defined as"),
    modulo ("5 mod 3" → "5 modulo 3"), the Greek alphabet
    (α → alpha), logic glyphs (¬ ∧ ∨ ⇔), set-theory glyphs
    (∑ ∏ ∂ ∈ ∀ ∃ ∪ ∩ ⊂ ∅ ∝ ≡ ∴ ∵), and variable division
    ("x/2" → "x over 2")
  - Mathematician-name respellings engines garble: Euler → "oiler",
    Euclid → "yoo klid", Fourier → "foor ee ay", Poisson → "pwah sohn",
    eigen- → "eye gen", Laplace, Descartes, Cauchy, Galois, Leibniz,
    Schrödinger, Dirichlet, Chebyshev, Lebesgue, L'Hôpital, de Broglie,
    Poincaré, Bézier, Noether, Lagrange, Fermat, Dijkstra, Erdős ...
  - Math shorthand: iff → "if and only if", w.r.t./wrt → "with respect
    to", WLOG → "without loss of generality", lhs/rhs → "left/right
    hand side", gcd/lcm → "G C D"/"L C M", PDE → "P D E"
  - History pronunciation pass: era markers ("44 BC" → "forty-four
    B C", "AD 1066" → "A D ten sixty-six", "2500 BCE", "CE 220"),
    year spans ("1914-1918" / "between 1914 and 1918" / "from 1914
    to 1918" read year-style on both sides), "c. 1860" → "circa …",
    round years ("in 1900" → "nineteen hundred"), "oh" year tails
    ("1806" → "eighteen oh six"), extra royal/emperor names
    (Ivan IV, Wilhelm II, Catherine II, Constantine I …), and ~55
    respellings for history names/places engines mangle (Versailles,
    Charlemagne, Byzantine, Qing, Tokugawa, Bolshevik, Ypres, ...)
"""

from __future__ import annotations

import functools
import html as _html_mod
import logging
import re
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple

__all__ = [
    "TextChunker", "ChunkingConfig", "ChunkStrategy",
    "chunk_text_for_tts", "get_chunker_for_language",
    "preprocess_for_tts",
]

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
#  LANGUAGE CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

SUPPORTED_LANGUAGE = Literal[
    "English", "Chinese", "Japanese", "Korean",
    "German", "French", "Russian", "Portuguese", "Spanish", "Italian",
]

CJK_LANGUAGES: FrozenSet[str] = frozenset({"Chinese", "Japanese", "Korean"})

SPEAKING_RATES: Dict[str, int] = {
    "English":    150, "Chinese":    200, "Japanese":   200,
    "Korean":     180, "German":     140, "French":     150,
    "Russian":    140, "Portuguese": 150, "Spanish":    160,
    "Italian":    160,
}

DEFAULT_SPEAKING_RATE = 150
_CJK_SPEAKING_RATE = 200  # chars/min for CJK (used in mixed-language mode)

# ═══════════════════════════════════════════════════════════════════════════════
#  PRE-COMPILED REGEX PATTERNS (module-level — compiled once)
# ═══════════════════════════════════════════════════════════════════════════════

_SENTENCE_PATTERNS: Dict[str, re.Pattern] = {
    "English":    re.compile(r'([.!?]+)'),
    "Chinese":    re.compile(r'([。！？…]+)'),
    "Japanese":   re.compile(r'([。！？…]+)'),
    "Korean":     re.compile(r'([.!?]+)'),  # Korean uses ASCII punctuation
    "German":     re.compile(r'([.!?]+)'),
    "French":     re.compile(r'([.!?]+)'),
    "Russian":    re.compile(r'([.!?]+)'),
    "Portuguese": re.compile(r'([.!?]+)'),
    "Spanish":    re.compile(r'([.!?]+)'),
    "Italian":    re.compile(r'([.!?]+)'),
}
_DEFAULT_SENTENCE_PATTERN = re.compile(r'([.!?。！？…]+)')
_PAUSE_PATTERN = re.compile(r'([,;:、，；：—–\-])')

# ─── Paragraph / newline split pattern ────────────────────────────────────────
_PARAGRAPH_PATTERN = re.compile(r'\n\s*\n')

# ─── Multi-dot abbreviations (moved from inside method — compiled once) ──────
_MULTI_DOT_ABBR = re.compile(r'\b(e\.g|i\.e|a\.m|p\.m)\.', re.IGNORECASE)

# ─── URL / domain pattern for pre-scanning (non-split zones) ──────────────────
_KNOWN_TLDS = (
    "com|org|net|edu|gov|io|co|uk|de|fr|jp|cn|ru|es|it|"
    "nl|se|no|ch|au|ca|br|in|mx|tv|me|info|biz|xyz|dev|app|ai"
)
_URL_PATTERN = re.compile(
    r'https?://\S+'                       # http:// or https:// URLs
    r'|www\.\S+'                          # bare www. domains
    r'|\b(?:[A-Za-z0-9-]+\.)+(?:' + _KNOWN_TLDS + r')\b'  # x.tld and x.y.tld
)

# ─── SSML tag pattern for pre-scanning (non-split zones) ──────────────────────
_SSML_TAG_PATTERN = re.compile(r'<[^>]+>')

# ─── Known multi-letter abbreviations (NOT sentence boundaries) ───────────────
# Moved from inline set in _is_sentence_boundary to module-level frozenset.
_KNOWN_ABBREVIATIONS: FrozenSet[str] = frozenset({
    'mr', 'mrs', 'ms', 'dr', 'prof', 'sr', 'jr', 'st',
    'gen', 'rep', 'sen', 'rev', 'hon', 'capt', 'lt', 'col',
    'sgt', 'adv', 'esq', 'pres', 'gov', 'mayor', 'ave', 'blvd',
    'dept', 'inc', 'ltd', 'corp', 'co', 'vs', 'etc',
    'approx', 'avg', 'max', 'min', 'vol', 'no', 'fig', 'ch',
    'pp', 'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug',
    'sep', 'sept', 'oct', 'nov', 'dec',
})

# ─── Constants ────────────────────────────────────────────────────────────────
# Merge tolerance for min-chunk merging (allow 20% over max_chars)
_MERGE_TOLERANCE = 1.2

# CJK Unicode ranges for character classification
_CJK_RANGES: Tuple[Tuple[int, int], ...] = (
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs
    (0x3400, 0x4DBF),   # CJK Unified Ideographs Extension A
    (0x3040, 0x309F),   # Hiragana
    (0x30A0, 0x30FF),   # Katakana
    # NOTE: Hangul (0xAC00-0xD7AF) intentionally NOT counted as CJK here —
    # Korean is space-delimited and word-based, so it is measured at the
    # Korean word rate (180 wpm), not the CJK character rate.
    (0xF900, 0xFAFF),   # CJK Compatibility Ideographs
)

# ─── Punctuation pause durations for duration estimation (seconds) ────────────
_PUNCTUATION_PAUSE: Dict[str, float] = {
    '.':  0.40, '。': 0.40, '!':  0.35, '！': 0.35,
    '?':  0.40, '？': 0.40, '…':  0.50,
    ',':  0.15, '，': 0.15, ';':  0.25, '；': 0.25,
    ':':  0.20, '：': 0.20, '、': 0.10,
    '—':  0.20, '–':  0.15,
    '\n': 0.30,
}

# Characters to strip from sentence edges (quotes, brackets, parentheses)
_SENTENCE_EDGE_CHARS = '"\'""''「」『』()（）[]【】'


# ═══════════════════════════════════════════════════════════════════════════════
#  CHUNK STRATEGY TYPE
# ═══════════════════════════════════════════════════════════════════════════════

ChunkStrategy = Literal[
    "sentence",   # group by sentence count (original default)
    "duration",   # group by estimated speaking duration
    "paragraph",  # prefer paragraph boundaries even if chunks are uneven
    "semantic",   # use conjunction/clause boundaries
    "hybrid",     # combine duration + semantic boundaries
]

_DEFAULT_CHUNK_STRATEGY: ChunkStrategy = "sentence"

# ═══════════════════════════════════════════════════════════════════════════════
#  PROSODY-AWARE CLAUSE BOUNDARIES
# ═══════════════════════════════════════════════════════════════════════════════

# Words that signal a natural clause/sentence start — preferred split points
_CLAUSE_START_WORDS: FrozenSet[str] = frozenset({
    # Transitional / conjunctive adverbs
    'however', 'therefore', 'moreover', 'furthermore', 'nevertheless',
    'consequently', 'meanwhile', 'otherwise', 'instead', 'likewise',
    'similarly', 'accordingly', 'subsequently', 'additionally',
    'alternatively', 'incidentally', 'namely', 'notably',
    # Temporal
    'then', 'next', 'later', 'afterwards', 'finally', 'initially',
    'eventually', 'previously',
    # Causal / contrastive
    'because', 'although', 'while', 'whereas', 'since', 'unless',
    'despite', 'though', 'yet', 'still',
})

# Pre-compiled pattern: detect clause-boundary words at sentence/paragraph start
_RE_CLAUSE_START = re.compile(
    r'^(?:' + '|'.join(
        re.escape(w) for w in sorted(
            _CLAUSE_START_WORDS,
            key=len, reverse=True,
        ) if len(w) > 2
    ) + r')[\s,;:]',
    re.IGNORECASE,
)

# ═══════════════════════════════════════════════════════════════════════════════
#  DIALOGUE / QUOTATION PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════

# Detect quoted speech at sentence start: "Hello," ...
_RE_DIALOGUE_OPEN = re.compile(r'^\s*[“"]')

# Detect dialogue attribution after a quote: ...said. "..."
_RE_DIALOGUE_ATTRIBUTION = re.compile(
    '["\u201c\u201d\u2019\u2018\']'
    r'\s*[,.]*\s*(?:\w+\s+)?'
    r'(?:said|replied|asked|exclaimed|whispered|'
    r'muttered|shouted|cried|called|added|noted|continued|'
    r'explained|remarked|observed|commented|stated|declared|'
    r'announced|responded|answered|interjected)\b[\s,.]',
    re.IGNORECASE,
)




# ═══════════════════════════════════════════════════════════════════════════════
#  LIST / ENUMERATION PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════

# Ordinal enumeration at sentence start: "First,...", "Second,..."
_RE_ENUMERATION_START = re.compile(
    r'^(?:first|second|third|fourth|fifth|sixth|seventh|eighth|'
    r'ninth|tenth|next|last|finally)[\s,;:]',
    re.IGNORECASE,
)

# ═══════════════════════════════════════════════════════════════════════════════
#  CONTEXT-AWARE NUMBER SKIP PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════

# Version strings: v1.2, Version 3.5, iOS 17, Python 3.12.
# Apple codename releases ("macOS Sonoma 14.2.1") are covered by a
# dedicated alternative — without it only "14.2" was consumed and the
# trailing ".1" was left dangling.
_RE_VERSION_STRING = re.compile(
    r'\b(?:v|version|ver)\.?\s*\d[\d.]*(?:[a-zA-Z]\d?)?\b'
    r'|\b(?:iOS|iPadOS|macOS|watchOS|tvOS)\s+'
    r'(?:Sonoma|Sequoia|Ventura|Monterey|Big\s+Sur|Catalina|Mojave|'
    r'High\s+Sierra|Sierra|El\s+Capitan)?\s*\d[\d.]*(?:[a-zA-Z]\d?)?\b'
    r'|\b(?:Android|Windows|Linux|'
    r'Ubuntu|Debian|Fedora|CentOS|Python|Node|Java|Go|Rust|Ruby|PHP|'
    r'Swift|Kotlin|Dart|Flutter|React|Angular|Vue|Next|Nuxt|Django|'
    r'Flask|Spring|Chrome|Firefox|Safari|Edge|Opera)'
    r'\s+\d[\d.]*(?:[a-zA-Z]\d?)?\b',
    re.IGNORECASE,
)

# Model/technical identifiers: GPT-4, GPT-4o, Llama-3, BERT-base
_RE_MODEL_IDENTIFIER = re.compile(
    r'\b(?:GPT|BERT|LLaMA|Llama|Claude|Gemini|Mistral|Phi|Qwen|'
    r'Copilot|Sora|DALL\s*E|Stable\s+Diffusion|Midjourney|'
    r'Whisper|TTS|SSML|HTML|CSS|XML|JSON|YAML|SQL|NoSQL|'
    r'RFC|IEEE|ISO|ASCII|UTF|Unicode)'
    r'-?\d[\d.]*(?:[a-zA-Z]\d?)?(?:-[a-zA-Z]+)?\b',
    re.IGNORECASE,
)

# Technical specs: 1080p, 4K, 720p, H.264, 802.11, USB-C, etc.
# Suffix letters {1,4}: single-letter wifi variants (802.11a/b/g/n/s) were
# previously unprotected — "802.11a" lost its "802" to the integer rule
# ("eight hundred two.11a") and "802.11s" was read "… one one seconds".
_RE_TECH_SPEC = re.compile(
    r'\b\d+[pPkKMGT](?:Hz|bps)?\b'
    r'|\bH\.?\d{3}(?:\.\d+)?\b'
    r'|\b\d{2,3}\.\d{2}(?:[a-zA-Z]{1,4})?\b'   # 802.11 / 802.11a / 802.11ac
    r'|\b\d{3,4}(?:\.\d+)?(?:-[a-zA-Z]{2,4}){1,2}\b'
    r'|\b(?:USB|HDMI|PCIe|DDR|LPDDR|NVMe|SATA|M\.?2|Wi[\s-]?Fi|'
    r'Bluetooth|5G|4G|3G|LTE|OLED|LCD|LED|AMOLED)\s*\d*\b',
    re.IGNORECASE,
)


def _build_number_skip_positions(t: str) -> set:
    """Character positions of digits/dots inside version/model/tech regions.

    Module-level so that *every* rule which must not touch technical tokens
    (the number-conversion section AND the history pass's era markers and
    year spans) rebuilds and consults the same protection set on the
    CURRENT text.  Positions computed before any length-changing
    substitution are stale — rebuilding immediately before each use is the
    structural fix for the original "Windows 11" → "Windows eleven" bug.
    """
    positions: set = set()
    for skip_re in (_RE_VERSION_STRING, _RE_MODEL_IDENTIFIER, _RE_TECH_SPEC):
        for skip_m in skip_re.finditer(t):
            for ci in range(skip_m.start(), skip_m.end()):
                if t[ci].isdigit() or t[ci] == '.':
                    positions.add(ci)
    return positions

# ═══════════════════════════════════════════════════════════════════════════════
#  PRONUNCIATION OVERRIDES
# ═══════════════════════════════════════════════════════════════════════════════

# User-configurable pronunciation overrides applied AFTER standard expansions.
_DEFAULT_PRONUNCIATION_OVERRIDES: Dict[re.Pattern, str] = {}
for _pat, _spk in [
    (r'\bJSON\b', 'jason'),
    (r'\bGPIO\b', 'gee pee eye oh'),
    (r'\bSQL\b', 'S Q L'),
    (r'\bGUI\b', 'gooey'),
    (r'\bJWT\b', 'jot'),
    (r'\bYAML\b', 'yamell'),
    (r'\bSaaS\b', 'sass'),
    (r'\bPaaS\b', 'pass'),
    (r'\bIaaS\b', 'eye as'),
    (r'\bOAuth\b', 'oh auth'),
    (r'\bUNESCO\b', 'you ness co'),
    (r'\bNDA\b', 'en dee ay'),
    (r'\bMoU\b', 'em oh you'),
    (r'\bRFP\b', 'ar eff pee'),
    (r'\bSLA\b', 'ess el ay'),
    (r'\bKPI\b', 'kay pee eye'),
    (r'\bROI\b', 'ar oh eye'),
    (r'\bMVP\b', 'em vee pee'),
    (r'\bYoY\b', 'year over year'),
    (r'\bMoM\b', 'month over month'),
    (r'\bQoQ\b', 'quarter over quarter'),
    (r'\bAPI\b', 'A P I'),
    (r'\bUI\b', 'U I'),
    (r'\bUX\b', 'U X'),
    (r'\bGPU\b', 'G P U'),
    (r'\bCPU\b', 'C P U'),
    (r'\bSSD\b', 'S S D'),
    (r'\bIDE\b', 'I D E'),
    (r'\bSDK\b', 'S D K'),
    (r'\bCLI\b', 'C L I'),
    (r'\bNLP\b', 'en el pee'),
    (r'\bLLM\b', 'el el em'),
    (r'\bGPT\b', 'G P T'),
    (r'\bBERT\b', 'bert'),
    (r'\bRAM\b', 'ram'),
    (r'\bROM\b', 'rom'),
    (r'\bVM\b', 'virtual machine'),
    (r'\bDNS\b', 'D N S'),
    (r'\bSSH\b', 'S S H'),
    (r'\bFTP\b', 'F T P'),
    (r'\bIoT\b', 'I O T'),
    (r'\bHTTP\b', 'H T T P'),
    (r'\bHTTPS\b', 'H T T P S'),
    (r'\bURL\b', 'U R L'),
    (r'\bHTML\b', 'H T M L'),
    (r'\bCSS\b', 'C S S'),
    (r'\bJS\b', 'javascript'),
    (r'\bNoSQL\b', 'no S Q L'),
    (r'\bPDF\b', 'P D F'),
    (r'\bCSV\b', 'C S V'),
    (r'\bWHO\b', 'World Health Organization'),
    (r'\bWTO\b', 'World Trade Organization'),
    (r'\bGDP\b', 'gross domestic product'),
    (r'\bGNP\b', 'gross national product'),
    (r'\bNASA\b', 'NASA'),
    (r'\bNATO\b', 'NATO'),
    (r'\bRADAR\b', 'radar'),
    (r'\bLASER\b', 'laser'),
    (r'\bWiFi\b', 'wi fi'),
    (r'\bPIN\b', 'P I N'),
    (r'\bSIM\b', 'S I M'),
    (r'\bGPS\b', 'G P S'),
    (r'\bTBH\b', 'to be honest'),
    (r'\bFWIW\b', 'for what its worth'),
    (r'\bAFAIK\b', 'as far as I know'),
    (r'\bIMO\b', 'in my opinion'),
    (r'\bIMHO\b', 'in my humble opinion'),
    (r'\bUSA\b', 'U S A'),
    (r'\bUK\b', 'U K'),
    (r'\bEU\b', 'E U'),
    (r'\bUN\b', 'U N'),
    # Title acronyms
    (r'\bCEO\b', 'Chief Executive Officer'),
    (r'\bCTO\b', 'Chief Technology Officer'),
    (r'\bCFO\b', 'Chief Financial Officer'),
    (r'\bCOO\b', 'Chief Operating Officer'),
    (r'\bCIO\b', 'Chief Information Officer'),
    (r'\bCMO\b', 'Chief Marketing Officer'),
    (r'\bCPO\b', 'Chief Product Officer'),
    (r'\bVP\b', 'Vice President'),
    (r'\bSVP\b', 'Senior Vice President'),
    (r'\bEVP\b', 'Executive Vice President'),
    # AI/ML terms
    (r'\bRL\b', 'reinforcement learning'),
    (r'\bML\b', 'machine learning'),
    (r'\bDL\b', 'deep learning'),
]:
    _DEFAULT_PRONUNCIATION_OVERRIDES[re.compile(_pat, re.IGNORECASE)] = _spk

# ═══════════════════════════════════════════════════════════════════════════════
#  PARAGRAPH CONTINUITY SIGNALS
# ═══════════════════════════════════════════════════════════════════════════════

# Words signaling dependence on prior context (pronouns, references)
_RE_CONTEXT_DEPENDENT_START = re.compile(
    r'^(?:this|these|those|that|it|its|such|they|them|their|'
    r'he|she|him|her|his|we|us|our|which|who|whom|whose)'
    r'(?:\s+(?:is|are|was|were|means|shows|proves|suggests|'
    r'implies|indicates|demonstrates|reveals|represents))?\b',
    re.IGNORECASE,
)

# ═══════════════════════════════════════════════════════════════════════════════
#  CROSSFADE PAUSE DURATION BY BOUNDARY TYPE
# ═══════════════════════════════════════════════════════════════════════════════

_BOUNDARY_CROSSFADE: Dict[str, float] = {
    '.':  0.30, '。': 0.30, '!':  0.25, '！': 0.25,
    '?':  0.30, '？': 0.30, '…': 0.35,
    ',':  0.12, '，': 0.12, ';':  0.20, '；': 0.20,
    ':':  0.15, '：': 0.15,
}
_DEFAULT_CROSSFADE = 0.20  # fallback for unknown boundaries

# ═══════════════════════════════════════════════════════════════════════════════
#  CJK NUMBER PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════

# Arabic digits followed by CJK measure/counter words
_RE_CJK_NUMBER_EXPR = re.compile(
    r'(\d+)\s*(?:年|月|日|时|分|秒|号|'
    r'点|度|元|块|毛|角|分|'
    r'厘米|毫米|米|公里|千米|'
    r'克|千克|公斤|吨|兆|亿|万|'
    r'千|百|个|只|条|本|页|章|'
    r'节|篇|段|行|列|人|次|台|'
    r'部|件|套|组)'
)

# CJK digit map for optional conversion
_CJK_DIGIT_MAP = {
    '零': '0', '一': '1', '二': '2', '三': '3',
    '四': '4', '五': '5', '六': '6', '七': '7',
    '八': '8', '九': '9', '两': '2',
}


# ═══════════════════════════════════════════════════════════════════════════════
#  PREPROCESSING PATTERNS (for TTS text cleaning)
# ═══════════════════════════════════════════════════════════════════════════════

# HTML comments
_RE_HTML_COMMENT = re.compile(r'<!--.*?-->', re.DOTALL)

# HTML tags — must start with a letter (or slash+letter) so that prose
# comparisons like "if a < 10 and b > 5" are NOT eaten as pseudo-tags.
_RE_HTML_TAG = re.compile(r'</?[a-zA-Z][^<>]*>')

# Markdown: images ![alt](url) — remove entirely
_RE_MD_IMAGE = re.compile(r'!\[([^\]]*)\]\([^)]*\)')

# Markdown: links [text](url) → keep text
_RE_MD_LINK = re.compile(r'\[([^\]]+)\]\([^)]*\)')

# Markdown: bold **text** or __text__
_RE_MD_BOLD = re.compile(r'\*\*(.+?)\*\*|__(.+?)__')

# Markdown: italic *text* or _text_
# The underscore form requires whitespace (or string start) before the
# opening underscore and whitespace/punctuation after the closing one, so
# snake_case identifiers like "my_var_name" are left untouched.
_RE_MD_ITALIC = re.compile(
    r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)'
    r'|(?:(?<=\s)|^)_([^_\s]+)_(?=[\s.,;:!?)]|$)'
)

# Markdown: inline code `text`
_RE_MD_CODE = re.compile(r'`([^`]+)`')

# Markdown: headings (## Heading → remove ## prefix)
_RE_MD_HEADING = re.compile(r'^(#{1,6})\s+(.*)$', re.MULTILINE)

# Bullet / numbered list markers
# NOTE: only horizontal whitespace before the marker — a class of [\s]*
# would also swallow the blank line that separates paragraphs.
_RE_LIST_MARKER = re.compile(r'^[ \t]*(?:[-*+•]|\d+[.)])\s+', re.MULTILINE)

# Unicode normalization: smart quotes (split by shape)
# Double curly quotes  U+201C/U+201D → ASCII "
_RE_SMART_DOUBLE_QUOTES = re.compile(r'[\u201c\u201d]')
# Single curly quotes/apostrophes U+2018/2019/201A/201B → ASCII '
_RE_SMART_SINGLE_QUOTES = re.compile(r'[\u2018\u2019\u201a\u201b]')

# Unicode normalization: dashes
_RE_DASHES = re.compile(r'[—–]')
_RE_DOUBLE_DASH = re.compile(r'--')

# Non-breaking spaces and zero-width characters
_RE_NBSP = re.compile(r'\u00a0')
_RE_ZERO_WIDTH = re.compile(r'[\u200b\u200c\u200d\ufeff\u2028\u2029]')

# Emojis (common Unicode ranges and ZWJ sequences)
_RE_EMOJI = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002702-\U000027B0"
    "\U000024C2-\U000024FF"  # enclosed alphanumerics
    "\U0001F170-\U0001F1FF"  # enclosed alphanumeric supplement (was 1F251 — too broad, captured CJK ideographs like ㊙)
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U0001FA00-\U0001FA6F"  # chess symbols
    "\U0001FA70-\U0001FAFF"  # symbols extended
    "\U00002600-\U000026FF"  # misc symbols
    "\U00002700-\U000027BF"  # dingbats
    "\U0000FE00-\U0000FE0F"  # variation selectors
    "]+",
    flags=re.UNICODE,
)

# URLs → "link to [domain]" — match the WHOLE URL (path included) so no
# path fragments are left behind and shredded into prose.
_RE_URL = re.compile(r'https?://\S+|www\.\S+')
# Trailing punctuation that does NOT belong to a URL (sentence periods etc.)
# \x27 = ASCII apostrophe; \u201d/\u2019 handled by the regex engine.
_URL_TRAILING_PUNCT = re.compile(r'[.,;:!?)\]}>"\u201d\u2019\x27]+$')

# Emails (Latin only) — user@host.com → "user at host dot com"
_RE_EMAIL = re.compile(r'\b([A-Za-z0-9._%+-]+)@([A-Za-z0-9.-]+)\.([A-Za-z]{2,})\b')

# Abbreviations to expand for speech (Latin only)
_ABBREVIATION_EXPANSIONS: Dict[re.Pattern, str] = {}
for _abbr, _exp in [
    # ── Common Latin abbreviations ──────────────────────────────────────────
    (r'\be\.g\.', 'for example'),
    (r'\bi\.e\.', 'that is'),
    # TRAILING \b GUARD: without it, \betc\.? matched the "etc" PREFIX of
    # "etched" → "et ceterahed", \bcf\.? matched "CFO" → "compare withO".
    # The \b before the optional dot only matches real standalone tokens.
    (r'\betc\b\.?', 'et cetera'),
    (r'\bvs\b\.?', 'versus'),
    (r'\bcf\b\.?', 'compare with'),
    # Requires the dot: bare "ca" (rare word) must not become "circa"
    (r'\bca\.(?=\s|$)', 'circa'),
    (r'\bq\.?\s*v\.?', 'quod vide'),
    (r'\bn\.b\.?', 'note well'),
    (r'\bet\s+al\b\.?', 'and others'),
    (r'\bibid\b\.?', 'ibidem'),
    (r'\bop\.?\s*cit\.?', 'opere citato'),
    (r'\bloc\.?\s*cit\.?', 'loco citato'),
    # NOTE: identity expansions (sic→sic, status quo→status quo, ad hoc,
    # de facto, per se, vice versa, a priori, a posteriori, ipso facto)
    # were removed — they inflated stats.abbreviations_expanded without
    # changing the spoken text.
    # ── General abbreviations ──────────────────────────────────────────────
    (r'\bapprox\.?(?=\s|$)', 'approximately'),
    (r'\bavg\b\.?', 'average'),
    (r'\bmin\.?(?=\s|$)', 'minimum'),
    (r'\bmax\.?(?=\s|$)', 'maximum'),
    (r'\binfo\b\.?', 'information'),
    (r'\bstats\b\.?', 'statistics'),
    (r'\btech\.?(?=\s|$)', 'technology'),
    (r'\brep\.?(?=\s|$)', 'repetitions'),
    (r'\bsec\.?(?=\s|$)', 'seconds'),
    (r'\bhr\.?(?=\s|$)', 'hour'),
    (r'\bhrs\b\.?', 'hours'),
    (r'\bfig\.?(?=\s|$)', 'figure'),
    (r'\bvol\.?(?=\s|$)', 'volume'),
    # Requires the dot AND digits after it: "No. 5" → "number 5", but a
    # sentence-final "no." ("...or no. Next") must stay the word "no".
    # The old (?=\s|$) lookahead turned "no. Next" into "number Next".
    (r'\bno\.(?=\s*\d)', 'number'),
    (r'\bch\.?(?=\s|$)', 'chapter'),
    (r'\bapp\.?(?=\s|$)', 'application'),
    (r'\bproj\.?(?=\s|$)', 'project'),
    (r'\btemp\.?(?=\s|$)', 'temperature'),
    (r'\beq\.?(?=\s|$)', 'equation'),
    (r'\bfn\.?(?=\s|$)', 'function'),
    (r'\bref\.?(?=\s|$)', 'reference'),
    (r'\bdef\.?(?=\s|$)', 'definition'),
    # ── Business / Professional ────────────────────────────────────────────
    (r'\bCEO\b', 'Chief Executive Officer'),
    (r'\bCTO\b', 'Chief Technology Officer'),
    (r'\bCFO\b', 'Chief Financial Officer'),
    (r'\bCOO\b', 'Chief Operating Officer'),
    (r'\bCIO\b', 'Chief Information Officer'),
    (r'\bCMO\b', 'Chief Marketing Officer'),
    (r'\bCPO\b', 'Chief Product Officer'),
    (r'\bVP\b', 'Vice President'),
    (r'\bSVP\b', 'Senior Vice President'),
    (r'\bEVP\b', 'Executive Vice President'),
    (r'\bROI\b', 'return on investment'),
    (r'\bKPI\b', 'key performance indicator'),
    (r'\bYoY\b', 'year over year'),
    (r'\bMoM\b', 'month over month'),
    (r'\bQoQ\b', 'quarter over quarter'),
    (r'\bMVP\b', 'minimum viable product'),
    (r'\bSLA\b', 'service level agreement'),
    (r'\bNDA\b', 'non-disclosure agreement'),
    (r'\bMOU\b', 'memorandum of understanding'),
    (r'\bRFP\b', 'request for proposal'),
    (r'\bFAQ\b', 'frequently asked questions'),
    # ── Technology / Computing ────────────────────────────────────────────
    (r'\bAI\b', 'artificial intelligence'),
    (r'\bAPI\b', 'A P I'),
    (r'\bUI\b', 'U I'),
    (r'\bUX\b', 'U X'),
    (r'\bGPU\b', 'G P U'),
    (r'\bCPU\b', 'C P U'),
    (r'\bRAM\b', 'ram'),
    (r'\bROM\b', 'rom'),
    (r'\bSSD\b', 'S S D'),
    (r'\bHTTP\b', 'H T T P'),
    (r'\bHTTPS\b', 'H T T P S'),
    (r'\bURL\b', 'U R L'),
    (r'\bHTML\b', 'H T M L'),
    (r'\bCSS\b', 'C S S'),
    (r'\bJS\b', 'javascript'),
    (r'\bJSON\b', 'jason'),
    (r'\bSQL\b', 'S Q L'),
    (r'\bNoSQL\b', 'no S Q L'),
    (r'\bDNS\b', 'D N S'),
    (r'\bSSH\b', 'S S H'),
    (r'\bFTP\b', 'F T P'),
    (r'\bIoT\b', 'I O T'),
    (r'\bSaaS\b', 'sass'),
    (r'\bPaaS\b', 'pass'),
    (r'\bIaaS\b', 'iaas'),
    (r'\bVM\b', 'virtual machine'),
    (r'\bIDE\b', 'I D E'),
    (r'\bSDK\b', 'S D K'),
    (r'\bCLI\b', 'C L I'),
    (r'\bGUI\b', 'G U I'),
    (r'\bNLP\b', 'natural language processing'),
    (r'\bLLM\b', 'large language model'),
    (r'\bGPT\b', 'G P T'),
    (r'\bBERT\b', 'bert'),
    (r'\bRL\b', 'reinforcement learning'),
    (r'\bML\b', 'machine learning'),
    (r'\bDL\b', 'deep learning'),
    # ── Common acronyms ────────────────────────────────────────────────────
    (r'\bWiFi\b', 'wi fi'),
    (r'\bPDF\b', 'P D F'),
    (r'\bCSV\b', 'C S V'),
    (r'\bYAML\b', 'yamell'),
    (r'\bJWT\b', 'J W T'),
    (r'\bOAuth\b', 'O auth'),
    (r'\bGDP\b', 'gross domestic product'),
    (r'\bGNP\b', 'gross national product'),
    (r'\bUSA\b', 'U S A'),
    (r'\bUK\b', 'U K'),
    (r'\bEU\b', 'E U'),
    (r'\bUN\b', 'U N'),
    (r'\bWHO\b', 'World Health Organization'),
    (r'\bUNESCO\b', 'you-ness-co'),
    (r'\bWTO\b', 'World Trade Organization'),
    (r'\bPIN\b', 'P I N'),
    (r'\bSIM\b', 'S I M'),
    (r'\bGPS\b', 'G P S'),
    (r'\bRADAR\b', 'radar'),
    (r'\bLASER\b', 'laser'),
    (r'\bTBH\b', 'to be honest'),
    (r'\bFWIW\b', 'for what it is worth'),
    (r'\bAFAIK\b', 'as far as I know'),
    (r'\bIMO\b', 'in my opinion'),
    (r'\bIMHO\b', 'in my humble opinion'),
    # ── Academic / Scientific / Titles ─────────────────────────────────────
    # \b GUARDS REQUIRED: \bDr\.? turned "dream"→"doctoream", "dress"→
    # "doctoress", "drive"→"doctorive"; \bProf\.? turned "profits"→
    # "professorits".  The trailing \b makes the pattern match only the
    # standalone token ("Dr." / "Dr"), never a word prefix.
    (r'\bPh\.?D\b\.?', 'doctor of philosophy'),
    # St. disambiguation: "St. Louis" / "St. Patrick" (capitalized word
    # follows) → saint; "Main St." / "742 Oak St," (end of clause or
    # lowercase follows) → street.  Saint rule runs first.
    (r'\bSt\.?(?=\s+[A-Z])', 'saint'),
    (r'\bSt\.?(?=\s*$|[,.;:!?)])', 'street'),
    (r'\bSt\.?(?=\s+[a-z])', 'street'),
    (r'\bDr\b\.?', 'doctor'),
    (r'\bProf\b\.?', 'professor'),
    (r'\bSr\b\.?', 'senior'),
    (r'\bJr\b\.?', 'junior'),
    # Requires the dot: "200 ms" (milliseconds) must not become "miss"
    # "Ms." is pronounced "Miz" — "miss" was wrong.
    (r'\bMs\.', 'miz'),
    (r'\bMrs\b\.?', 'misses'),
    (r'\bMr\b\.?', 'mister'),
    (r'\bRev\.?(?=\s|$)', 'reverend'),
    (r'\bHon\.?(?=\s|$)', 'honorable'),
    (r'\bCapt\.?(?=\s|$)', 'captain'),
    (r'\bLt\.?(?=\s|$)', 'lieutenant'),
    (r'\bCol\.?(?=\s|$)', 'colonel'),
    # Requires the dot: bare "Gen" is "Gen Z"/"Gen X" — "general Z" was
    # nonsense ("Gen. Sherman" still reads "general Sherman").
    (r'\bGen\.(?=\s|$)', 'general'),
    (r'\bSgt\.?(?=\s|$)', 'sergeant'),
    # ── Misc common abbreviations ─────────────────────────────────────────
    (r'\bmisc\b\.?', 'miscellaneous'),
    (r'\badmin\b\.?', 'administration'),
    (r'\bauth\.?(?=\s|$)', 'authentication'),
    (r'\bconfig\.?(?=\s|$)', 'configuration'),
    (r'\bdir\.?(?=\s|$)', 'directory'),
    (r'\bdoc\.?(?=\s|$)', 'document'),
    (r'\bdocs\b\.?', 'documents'),
    (r'\blib\.?(?=\s|$)', 'library'),
    (r'\bmgr\.?(?=\s|$)', 'manager'),
    (r'\bmgmt\b\.?', 'management'),
    (r'\bdept\.?(?=\s|$)', 'department'),
    (r'\bdiv\.?(?=\s|$)', 'division'),
    (r'\basst\.?(?=\s|$)', 'assistant'),
    (r'\bassoc\.?(?=\s|$)', 'associate'),
    (r'\bacct\.?(?=\s|$)', 'account'),
    (r'\brecon\.?(?=\s|$)', 'reconnaissance'),
    (r'\bpwd\.?(?=\s|$)', 'password'),
    (r'\bmsg\.?(?=\s|$)', 'message'),
    (r'\bmsgs\b\.?', 'messages'),
    (r'\bbtn\.?(?=\s|$)', 'button'),
    (r'\bchk\.?(?=\s|$)', 'check'),
    (r'\bcalc\.?(?=\s|$)', 'calculator'),
    (r'\bdiag\.?(?=\s|$)', 'diagram'),
    (r'\bnum\.?(?=\s|$)', 'number'),
    (r'\bpct\.?(?=\s|$)', 'percent'),
    (r'\bamt\.?(?=\s|$)', 'amount'),
    (r'\best\.(?=\s|$)', 'estimate'),
    (r'\breq\.?(?=\s|$)', 'requirement'),
    (r'\breqd\b\.?', 'required'),
    (r'\bops\.?(?=\s|$)', 'operations'),
    (r'\bdev\.?(?=\s|$)', 'development'),
    (r'\bdevs\b\.?', 'developers'),
    (r'\bprod\.?(?=\s|$)', 'production'),
    (r'\brec\.?(?=\s|$)', 'recommendation'),
    (r'\binc\.?(?=\s|$)', 'incorporated'),
    (r'\bltd\.?(?=\s|$)', 'limited'),
    (r'\bcorp\.?(?=\s|$)', 'corporation'),
    # (?<!\.) protects bare domains like "example.co"
    (r'(?<!\.)\bco\.(?=\s|$)', 'company'),
    (r'\buniv\.?(?=\s|$)', 'university'),
    (r'\blab\.?(?=\s|$)', 'laboratory'),
    (r'\bspec\.?(?=\s|$)', 'specification'),
    (r'\brev\.?(?=\s|$)', 'revision'),
    (r'\bver\.?(?=\s|$)', 'version'),
    (r'\barch\.?(?=\s|$)', 'architecture'),
    (r'\bpkg\.?(?=\s|$)', 'package'),
    (r'\bmod\.?(?=\s|$)', 'module'),
    (r'\blog\.?(?=\s|$)', 'log'),
    (r'\berr\.?(?=\s|$)', 'error'),
    (r'\bval\.?(?=\s|$)', 'value'),
    (r'\bparam\.?(?=\s|$)', 'parameter'),
    (r'\bretval\.?(?=\s|$)', 'return value'),
    (r'\bretvals\b\.?', 'return values'),
    # ── More spoken-friendly acronyms (letter-by-letter) ────────────────
    (r'\bWWI\b', 'World War One'),
    (r'\bWWII\b', 'World War Two'),
    (r'\bXML\b', 'X M L'),
    (r'\bTCP\b', 'T C P'),
    (r'\bUDP\b', 'U D P'),
    (r'\bVPN\b', 'V P N'),
    (r'\bLAN\b', 'L A N'),
    (r'\bWAN\b', 'W A N'),
    (r'\bVLAN\b', 'V L A N'),
    (r'\bSMTP\b', 'S M T P'),
    (r'\bIMAP\b', 'I M A P'),
    (r'\bXSS\b', 'X S S'),
    (r'\bCSRF\b', 'C S R F'),
    (r'\bCORS\b', 'C O R S'),
    (r'\bPHP\b', 'P H P'),
    (r'\bnpm\b', 'N P M'),
    (r'\bAWS\b', 'A W S'),
    (r'\bGCP\b', 'G C P'),
    (r'\bDMZ\b', 'D M Z'),
    (r'\bFBI\b', 'F B I'),
    (r'\bCIA\b', 'C I A'),
    (r'\bNSA\b', 'N S A'),
    (r'\bDNA\b', 'D N A'),
    (r'\bRNA\b', 'R N A'),
    (r'\bHIV\b', 'H I V'),
    (r'\bCPR\b', 'C P R'),
    (r'\bICU\b', 'I C U'),
    (r'\bADHD\b', 'A D H D'),
    (r'\bPTSD\b', 'P T S D'),
    (r'\bOEM\b', 'O E M'),
    (r'\bERP\b', 'E R P'),
    (r'\bCRM\b', 'C R M'),
    (r'\bCMS\b', 'C M S'),
    (r'\bSEO\b', 'S E O'),
    (r'\bOKR\b', 'O K R'),
    (r'\bETF\b', 'E T F'),
    (r'\bIPO\b', 'I P O'),
    (r'\bATM\b', 'A T M'),
    (r'\bDIY\b', 'D I Y'),
    (r'\bASAP\b', 'as soon as possible'),
    (r'\bRSVP\b', 'R S V P'),
    (r'\bAKA\b', 'also known as'),
    (r'\bB2B\b', 'B to B'),
    (r'\bB2C\b', 'B to C'),
    (r'\bP2P\b', 'peer to peer'),
    (r'\bH2O\b', 'H two O'),
    (r'\bCO2\b', 'C O two'),
    (r'\bU\.S\.A\.', 'U S A'),
    (r'\bU\.S\.(?=\s|$|[,.;:!?)])', 'U S'),
    # ── Pronunciation-quality additions (improvement pass 2) ────────────
    (r'\bok\b', 'okay'),
    (r'\bviz\b\.?', 'namely'),
    (r'\bWW1\b', 'World War One'),
    (r'\bWW2\b', 'World War Two'),
    (r'\bTBD\b', 'to be determined'),
    (r'\bTBA\b', 'to be announced'),
    (r'\bN/A\b', 'not applicable'),
    (r'\bI/O\b', 'input or output'),
    (r'\bA/B(?=\s+test(?:s|ing)?\b)', 'A B'),
    # Numeronyms: letter-digit-letter forms engines read as "k-eight-s"
    (r'\bK8s\b', 'kubernetes'),
    (r'\bi18n\b', 'internationalization'),
    (r'\bl10n\b', 'localization'),
    (r'\ba11y\b', 'accessibility'),
    # Dot-less "e.g"/"i.e" variants (the dotted forms matched above)
    (r'\be\.g(?=\s|,|;|$)', 'for example'),
    (r'\bi\.e(?=\s|,|;|$)', 'that is'),
    # ── Pronunciation-quality additions (improvement pass 3) ──────────
    # Dotted a.m./p.m. — engines otherwise read "a dot m".  The dot is
    # REQUIRED so the words "am"/"pm" never match.
    (r'\ba\.m\.', 'A M'),
    (r'\bp\.m\.', 'P M'),
    # "deg" is not an English word — engines say "dehg"
    (r'\bdeg\.?\s*C\b', 'degrees Celsius'),
    (r'\bdeg\.?\s*F\b', 'degrees Fahrenheit'),
    (r'\bdeg\.?(?=$|[\s,.;:!?)])', 'degrees'),
    # "queue" read "kwe-we" by many engines; "cue" is unambiguous
    (r'\bqueues\b', 'cues'),
    (r'\bqueue\b', 'cue'),
    (r'\bqueued\b', 'cued'),
    # Frequently mispronounced English words
    (r'\bepitome\b', 'eh pit oh mee'),
    (r'\bhyperbole\b', 'hy per bo lee'),
    # Numeronym: "e2e" read "ee-two-ee" — spelled out is natural
    (r'\be2e\b', 'end to end'),
    # Fiscal quarters: "Q1 results" → "quarter 1 results" (engines read
    # "Q1" as "kwon" or letter-by-letter awkwardly).  RAW string required:
    # a plain 'quarter \1' would store an octal control character.
    (r'\bQ([1-4])\b', r'quarter \1'),
    # CI/CD is pronounced "see-eye see-dee", never "CI or CD"
    (r'\bCI/CD\b', 'C I C D'),
    # Two-factor authentication shorthand
    (r'\b2FA\b', 'two F A'),
    # ── Pronunciation-quality additions (improvement pass 4) ──────────
    # Notoriously misread word: engines say "seeg" or "seg-yoo"
    (r'\bsegues\b', 'seg ways'),
    (r'\bsegue\b', 'seg way'),
    # ── Pronunciation-quality additions (improvement pass 5) ──────────
    # Math shorthand that engines garble or read literally:
    (r'\biff\b', 'if and only if'),
    (r'\bw\.r\.t\.?(?=\s|$)', 'with respect to'),
    (r'\bwrt\b', 'with respect to'),
    (r'\bwlog\b', 'without loss of generality'),
    (r'\blhs\b', 'left hand side'),
    (r'\brhs\b', 'right hand side'),
    (r'\bgcd\b', 'G C D'),
    (r'\blcm\b', 'L C M'),
    (r'\bresp\.(?=\s|$)', 'respectively'),
    (r'\bthm\.?(?=\s|$)', 'theorem'),
    (r'\bprop\.(?=\s|$)', 'proposition'),
    # ── Mathematician names engines mangle (math lecture decks) ───────
    # Respellings follow the established syllable style ("eh pit oh mee").
    # Case-insensitive: proper nouns like "euler"/"eigen" are not English
    # words, so no lowercase collisions exist.
    (r'\beuler\b', 'oiler'),
    (r'\beulerian\b', 'oiler ee un'),
    (r'\beuclid\b', 'yoo klid'),
    (r'\bfourier\b', 'foor ee ay'),
    (r'\bpoisson\b', 'pwah sohn'),
    (r'\bpoincar[ée]\b', 'pwan ka ray'),
    (r'\bb[ée]zier\b', 'bez ee ay'),
    (r'\bcauchy\b', 'koh shee'),
    (r'\bgalois\b', 'gal wah'),
    (r'\blaplace\b', 'luh ploss'),
    (r'\bdescartes\b', 'day cart'),
    (r'\bdirichlet\b', 'deer ikh lay'),
    (r'\bleibniz\b', 'lype nits'),
    (r'\bchebyshev\b', 'cheh bee shef'),
    (r'\bschr[öo]dinger\b', 'shroh ding er'),
    (r'\bschroedinger\b', 'shroh ding er'),
    (r'\bde\s+broglie\b', 'duh broh glee'),
    (r'\bbroglie\b', 'broh glee'),
    (r"\bl'?h[oô]pital\b", 'loh pee tal'),
    (r'\blebesgue\b', 'luh beg'),
    (r'\bbanach\b', 'bah nahk'),
    (r'\bnoether\b', 'nur ter'),
    (r'\blagrange\b', 'luh grahnj'),
    (r'\bfermat\b', 'fur mah'),
    (r'\berd[őo]s\b', 'air dish'),
    (r'\bmoivre\b', 'mwah vruh'),
    (r'\bdelaunay\b', 'day lon ay'),
    (r'\bweierstrass\b', 'vy er strahss'),
    (r'\blegendre\b', 'luh zhan druh'),
    (r'\btukey\b', 'too kee'),
    (r'\beratosthenes\b', 'air uh tos thuh neez'),
    (r'\bkleene\b', 'klay nee'),
    (r'\bdijkstra\b', 'dik struh'),
    # "eigen-" prefix: covers eigenvalue/eigenvector and snake_case splits
    # ("eigen value") in one pattern; "eigen" is not an English word.
    # The compound form needs a lookahead (no word boundary inside
    # "eigenvalue"), and keeps the value/vector suffix in the text.
    (r'\beigen\b', 'eye gen'),
    (r'\beigen(?=values?|vectors?)', 'eye gen '),
    # ── History names & places engines mangle (history decks) ───────
    # Same syllable-respelling style as the mathematician list; every
    # token is a proper noun, so the case-insensitive tier cannot
    # collide with an ordinary English word.
    # ── Ancient world ──
    (r'\bnebuchadnezzar\b', 'neb uh kud nezz er'),
    (r'\bhammurabi\b', 'hah moo rah bee'),
    (r'\bxerxes\b', 'zerk seez'),
    (r'\bthucydides\b', 'thoo sid ih deez'),
    (r'\bherodotus\b', 'huh rod uh tus'),
    (r'\bptolemaic\b', 'tol uh may ik'),
    (r'\bptolemy\b', 'tol uh mee'),
    (r'\bboudicca\b', 'boo dik ah'),
    (r'\bmesopotamia\b', 'mes uh puh tay mee uh'),
    (r'\baegean\b', 'ih jee un'),
    (r'\bmycenae\b', 'my see nee'),
    (r'\bantioch\b', 'an tee ahk'),
    (r'\bvercingetorix\b', 'vur san guh tor iks'),
    # ── Medieval & early modern Europe ──
    (r'\bcharlemagne\b', 'shar luh mayn'),
    (r'\bbyzantine\b', 'biz un teen'),
    (r'\bplantagenet\b', 'plan tag uh net'),
    (r'\bcarolingian\b', 'kair uh lin jee un'),
    (r'\bmerovingian\b', 'mair uh vin jee un'),
    (r'\bvalois\b', 'val wah'),
    (r'\bagincourt\b', 'azh in koor'),
    (r'\bbayeux\b', 'bye yuh'),
    (r'\bversailles\b', 'ver sigh'),
    (r'\bausterlitz\b', 'aws ter lits'),
    (r'\bmontesquieu\b', 'mont uh skyoo'),
    (r'\brousseau\b', 'roo soh'),
    (r'\btocqueville\b', 'tohk vil'),
    # ── Americas / Africa / Middle East ──
    (r"\bl'?ouverture\b", 'loo ver toor'),
    (r'\btoussaint\b', 'too sant'),
    (r'\btenochtitlan\b', 'te noch tee tlahn'),
    (r'\bumayyad\b', 'oo my yad'),
    (r'\babbasid\b', 'uh bah sid'),
    (r'\bseljuk\b', 'sel jook'),
    (r'\bsaladin\b', 'sal ah deen'),
    # ── Asia ──
    (r'\bmaurya\b', 'mah oo ryah'),
    (r'\bashoka\b', 'uh shoh kah'),
    (r'\bmughal(s?)\b', r'moo gul\1'),
    (r'\bgenghis\b', 'geng gis'),
    (r'\bkublai\b', 'koo blie'),
    (r'\btamerlane\b', 'tam er lane'),
    (r'\bqing\b', 'ching'),
    (r'\bqin\b', 'chin'),
    (r'\bzhou\b', 'joh'),
    (r'\bsui\b', 'sway'),
    (r'\bxia\b', 'shyah'),
    (r'\bzheng\b', 'jung'),
    (r'\bxinhai\b', 'shin high'),
    (r'\bcixi\b', 'tsih shee'),
    (r'\btokugawa\b', 'toh koo gah wah'),
    (r'\bmeiji\b', 'may ee jee'),
    (r'\bheian\b', 'hay ahn'),
    (r'\bkamakura\b', 'kah mah koo rah'),
    (r'\bjoseon\b', 'choh sun'),
    # ── 19th/20th century ──
    (r'\bbolshevik(s?)\b', r'bol shuh vick\1'),
    (r'\bkhrushchev\b', 'kroosh chof'),
    (r'\bweimar\b', 'vy mahr'),
    (r'\breichstag\b', 'rikes tahk'),
    (r'\bluftwaffe\b', 'looft vah fuh'),
    (r'\bblitzkrieg\b', 'blits kreeg'),
    (r'\blusitania\b', 'loo sih tay nee ah'),
    (r'\bypres\b', 'ee pruh'),
    (r'\bapartheid\b', 'uh pahr tate'),
    (r'\btiananmen\b', 'tyen an men'),
    (r'\bperestroika\b', 'per uh stroy kah'),
    # ── More commonly garbled English words (pass 5) ──────────────────
    (r'\bawry\b', 'uh rye'),
    (r'\brendezvous\b', 'ron day voo'),
    (r'\bquays\b', 'keez'),
    (r'\bquay\b', 'kee'),
    (r'\bindictment\b', 'in dite ment'),
    (r'\bindict\b', 'in dite'),
    (r'\bvoil[àa]\b', 'vwah lah'),
]:
    _ABBREVIATION_EXPANSIONS[re.compile(_abbr, re.IGNORECASE)] = _exp

# Case-sensitive expansion tier — compiled WITHOUT re.IGNORECASE because
# the lowercase form is an ordinary word that must never be rewritten:
# "us" (pronoun) vs "US", "it" (pronoun) vs "IT", "id" vs "ID",
# "dc" vs "DC", Greek "eta" vs "ETA", "Ps." (Psalms) vs "PS" (postscript).
_ABBREVIATION_EXPANSIONS_CS: Dict[re.Pattern, str] = {}
for _abbr, _exp in [
    (r'\bUS\b', 'U S'),
    (r'\bIT\b', 'I T'),
    (r'\bID\b', 'I D'),
    (r'\bDC\b', 'D C'),
    (r'\bGIF\b', 'jiff'),
    (r'\bGIFs\b', 'jiffs'),
    (r'\bIEEE\b', 'I triple E'),
    (r'\bISO\b', 'I S O'),
    (r'\bRFC\b', 'R F C'),
    (r'\bUTF\b', 'U T F'),
    (r'\bASCII\b', 'ask ee'),
    (r'\bTTS\b', 'T T S'),
    (r'\bSSML\b', 'S S M L'),
    (r'\bACM\b', 'A C M'),
    (r'\bUSD\b', 'U S D'),
    (r'\bLLVM\b', 'L L V M'),
    (r'\bGCC\b', 'G C C'),
    (r'\bMBA\b', 'M B A'),
    (r'\bGPA\b', 'G P A'),
    (r'\bETA\b', 'E T A'),
    # Tech products whose letter sequences engines mangle
    (r'\bLaTeX\b', 'lah tek'),
    (r'\bTeX\b', 'tek'),
    (r'\bNginx\b', 'engine x'),
    (r'\bMySQL\b', 'my sequel'),
    (r'\bPostgreSQL\b', 'postgres Q L'),
    (r'\bSQLite\b', 'sequel lite'),
    (r'\bMongoDB\b', 'mongo D B'),
    (r'\bNumPy\b', 'num pie'),
    (r'\bSciPy\b', 'sigh pie'),
    (r'\bPyPI\b', 'pie pie eye'),
    (r'\bVue\b', 'view'),
    (r'\bGNU\b', 'guh new'),
    # Keyboard keys shown in shortcut slides
    (r'\bCtrl\b', 'control'),
    (r'\bCmd\b', 'command'),
    (r'\bEsc\b', 'escape'),
    # "P.S." → postscript; case-sensitive so "Ps. 23" (Psalms) survives.
    # The lookahead blocks "PS5" and similar product codes.
    (r'\bP\.?S\.?(?=\s|:|,|$)', 'postscript'),
    # ── Brand pronunciations (improvement pass 3) ───────────────────
    # The dotted-uppercase "Nginx" form was already covered; lowercase
    # "nginx" is the far more common spelling in technical decks.
    (r'\bnginx\b', 'engine x'),
    (r'\bNGINX\b', 'engine x'),
    (r'\bPyTorch\b', 'pie torch'),
    (r'\bPySide6\b', 'pie side six'),
    (r'\bPySide\b', 'pie side'),
    (r'\bQt\b', 'Q T'),
    (r'\bGraphQL\b', 'graph Q L'),
    (r'\bFastAPI\b', 'fast A P I'),
    (r'\bWebGL\b', 'web G L'),
    (r'\bOpenGL\b', 'open G L'),
    (r'\bTensorRT\b', 'tensor R T'),
    (r'\bVoIP\b', 'V O I P'),
    # Embedded-bus names: "I2C" read "eye two see" (never "i-2-c")
    (r'\bI2C\b', 'I two C'),
    (r'\bI2S\b', 'I two S'),
    (r'\bSPI\b', 'S P I'),
    (r'\bPCIe\b', 'P C I express'),
    # Math course shorthand: "PDE" read letter-wise (engines say "pdee");
    # "ODE" already reads as the English word "ode" and stays untouched.
    (r'\bPDE\b', 'P D E'),
]:
    _ABBREVIATION_EXPANSIONS_CS[re.compile(_abbr)] = _exp

# Symbols to normalize (Latin only)
_RE_AMPERSAND = re.compile(r'\s*&\s*')
_RE_SLASH_DIGITS = re.compile(r'(\d+)\s*/\s*(\d+)')
_RE_PLUS_EQUALS = re.compile(r'\+=')
_RE_EQUALS = re.compile(r'(?<=\s)=|=(?=\s)')
_RE_PERCENT = re.compile(r'%')
_RE_DEGREE = re.compile(r'°')
# Asterisk included: "3*4" is multiplication (markdown/code backticks are
# stripped before this rule runs, so no conflict).
_RE_MULTIPLY_DIGITS = re.compile(r'(\d+)\s*[x×·*]\s*(\d+)')

# Trailing times-sign without a second operand: "4× zoom" → "4 times zoom".
# "A×B" (digit on the right) is consumed by the multiply rule above first.
_RE_TRAILING_TIMES_SIGN = re.compile(r'\b(\d{1,3})\s*×(?!\s*\d)')

# ─── Multiple punctuation normalization ──────────────────────────────────────
_RE_MULTI_EXCLAIM = re.compile(r'!{2,}')
_RE_MULTI_QUESTION = re.compile(r'\?{2,}')
_RE_MULTI_PERIOD = re.compile(r'\.{4,}')  # more than ellipsis → ellipsis
_RE_MULTI_COMMA = re.compile(r',{2,}')
_RE_MULTI_SEMICOLON = re.compile(r';{2,}')
_RE_MULTI_COLON = re.compile(r':{2,}')

# ─── Comparison / logic operators (must come BEFORE number patterns) ─────────
_RE_NOT_EQUAL = re.compile(r'!=|!==')
_RE_GTE = re.compile(r'>=')
_RE_LTE = re.compile(r'<=')
_RE_GT = re.compile(r'(?<!=)>')
_RE_LT = re.compile(r'(?<![<]=)<')

# ─── Currency symbols (with optional leading digits) ─────────────────────────
# (?:\.\d+)? — decimals are optional-whole; a bare trailing dot (sentence
# period after money, e.g. "It costs $5.") must NOT be swallowed.
# Optional attached magnitude suffix: "$5K" → "five thousand dollars",
# "$1.5B" → "one point five billion dollars" (previously left a dangling
# "K"/"B" glued to the spoken currency: "five dollarsK").
_RE_CURRENCY_DOLLAR = re.compile(r'\$(\d[\d,]*(?:\.\d+)?)([KMB])?\b')
_RE_CURRENCY_EURO = re.compile(r'€(\d[\d,]*(?:\.\d+)?)([KMB])?\b')
_RE_CURRENCY_POUND = re.compile(r'£(\d[\d,]*(?:\.\d+)?)([KMB])?\b')
_RE_CURRENCY_YEN = re.compile(r'¥([\d,]+)([KMB])?\b')
# Indian rupee: "₹1,299" — the glyph has no native engine reading.
_RE_CURRENCY_RUPEE = re.compile(r'₹(\d[\d,]*(?:\.\d+)?)([KMB])?\b')

# ─── Number patterns (ordered: most specific first) ─────────────────────────
# Ordinal suffixes: 1st, 2nd, 3rd, 4th, 11th, 12th, 13th, 21st, 1234th, etc.
_RE_ORDINAL = re.compile(
    r'\b(0*[1-9]\d{0,3})(st|nd|rd|th)\b', re.IGNORECASE
)
# Time: 5:30, 5:30 PM, 5:30pm, 12:00 AM, etc.
_RE_TIME = re.compile(
    r'\b(\d{1,2}):(\d{2})(?:\s*([AaPp]\.?[Mm]\.?))?\b'
)
# Elapsed h:m:s timestamps: "12:34:56" — the clock-time rule would eat
# "12:34" and leave a dangling ":56" ("twelve thirty-four:fifty-six").
# The colon guards keep MAC-like colon chains ("00:11:22:33") untouched.
_RE_HMS = re.compile(
    r'(?<!\d:)\b(\d{1,2}):([0-5]\d):([0-5]\d)\b(?!:\d)')
# Ratio / aspect ratio: "16:9" → "16 to 9", "3:1" → "3 to 1".  Clock
# times (\d{1,2}:\d{2}) are consumed first by _RE_TIME; whatever digit-
# colon-digit pairs remain are ratios, scores, or aspect ratios — all
# naturally spoken "X to Y".  The colon guards keep h:m:s fragments and
# MAC-like colon chains out of the match.
_RE_RATIO = re.compile(
    r'(?<!:)\b(\d{1,3}(?:\.\d+)?)\s*:\s*(\d{1,2})\b(?!:)')
# Decimal number: 3.14, 0.99, .5 (leading-dot decimals supported)
_RE_DECIMAL = re.compile(r'(?<![A-Za-z\d])(\d+\.\d+|\.\d+)(?![A-Za-z\d])')
# Fraction: 1/2, 3/4, 2/3, etc. (whole or mixed like 1 1/2)
_RE_FRACTION = re.compile(r'\b(\d+)\s*/\s*(\d+)\b')
# Integer with commas: 1,000  /  1,000,000  / standalone integers
_RE_INTEGER = re.compile(r'\b(\d{1,3}(?:,\d{3})+|\d+)\b')

# ─── Slash between words ─────────────────────────────────────────────────────
_RE_WORD_SLASH = re.compile(r'\b([A-Za-z][A-Za-z0-9]*)\s*/\s*([A-Za-z][A-Za-z0-9]*)\b')

# ─── Parenthetical pauses ───────────────────────────────────────────────────
# Replace (content) with , content,  — but preserve content
_RE_PARENTHETICAL = re.compile(r'\(\s*([^)]+?)\s*\)')

# ─── Hyphenated compound words (non-dash range, non minus sign) ──────────────
# Two-part: "well-known"; also matches the outer parts of multi-hyphen
# compounds like "state-of-the-art" (captures "state" and "art").
_RE_HYPHENATED_WORD = re.compile(r'\b([A-Za-z]+)-([A-Za-z]+)\b')
# Multi-hyphen compound: "state-of-the-art", "mother-in-law"
_RE_MULTI_HYPHEN = re.compile(r'\b([A-Za-z]+)(?:-[A-Za-z]+)+\b')

# ─── Hyphenated words to preserve (module-level to avoid re-creation per match) ─
_KEEP_HYPHEN: FrozenSet[str] = frozenset({
    'e-mail', 're-enter', 'co-operate', 'co-ordinate',
    'pre-empt', 're-examine', 're-iterate', 'de-ice',
    'x-ray', 't-shirt', 're-write', 'co-author',
})

# ─── Number words (module-level to avoid re-creation per match) ─
_NUMBER_WORDS: FrozenSet[str] = frozenset({
    'zero', 'one', 'two', 'three', 'four', 'five', 'six',
    'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve',
    'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen',
    'eighteen', 'nineteen', 'twenty', 'thirty', 'forty',
    'fifty', 'sixty', 'seventy', 'eighty', 'ninety',
    'hundred', 'thousand', 'million', 'billion', 'trillion',
    'first', 'second', 'third', 'fourth', 'fifth', 'sixth',
    'seventh', 'eighth', 'ninth', 'tenth', 'eleventh',
    'twelfth', 'thirteenth', 'fourteenth', 'fifteenth',
    'sixteenth', 'seventeenth', 'eighteenth', 'nineteenth',
    'twentieth', 'thirtieth', 'fortieth', 'fiftieth',
    'sixtieth', 'seventieth', 'eightieth', 'ninetieth',
    'hundredth', 'thousandth', 'millionth', 'billionth',
})

# ─── Standalone symbols ─────────────────────────────────────────────────────
_RE_HASH_NUMBER = re.compile(r'#(\d+)')
_RE_STANDALONE_PLUS = re.compile(r'(?<![+\w=])\+(?![=\w])')
_RE_STANDALONE_MINUS = re.compile(r'(?<![-\w=])-(?![=\w\d-])')
_RE_TILDE = re.compile(r'~')
_RE_AT_SIGN = re.compile(r'(?<![A-Za-z0-9.])@(?![A-Za-z0-9.])')

# ─── Tight math operators (no surrounding spaces) ──────────────────────────
# "3+4" and "a+b": both the spaced-equals family and the standalone rules
# require breathing room around "+", so a glued plus dangled.  C++/Ctrl+S/
# "+=/++" are consumed by earlier rules or excluded by the guards.
_RE_TIGHT_PLUS = re.compile(r'(?<=[A-Za-z0-9%])\+(?=[A-Za-z0-9%])')
# Negative numbers: "-5", "x = -5", "x=-5", "(-5)".  Only a minus that is
# anchored by whitespace/string-start, an equals sign or an opening bracket
# AND directly followed by a digit is spoken "minus" — ranges ("2020-2024"),
# phone numbers ("555-123-4567"), ISO dates and hyphenated tokens
# ("GPT-4", "A-5", "mid-90s") all have a word/digit before the dash and are
# left for their own rules.
_RE_NEGATIVE_NUMBER = re.compile(r'(?:(?<!\S)|(?<==)|(?<=[(\[]))-(?=\d)')

# ─── Variable division: "x/2", "2/x" — the fraction rule needs two digits
# and the word/word slash rule two words, so a variable over a digit
# dangled unread.  Token-start guards keep "PS/2" (S preceded by P),
# "HTML/5" and "80/kg" (the "unit" must be a lone letter) untouched.
_RE_VAR_OVER_NUM = re.compile(r'(?<![A-Za-z0-9])([a-zA-Z])\s*/\s*(\d+)\b')
_RE_NUM_OVER_VAR = re.compile(
    r'(?<![\w.])(\d+)\s*/\s*([a-zA-Z])(?![A-Za-z0-9])')


# ═══════════════════════════════════════════════════════════════════════════════
#  PRONUNCIATION: UNITS, MEASURES, AND SPECIAL SYMBOLS
#  (digit-anchored expansions so TTS engines say "five kilograms" instead of
#  "five kay gee"; run BEFORE the abbreviation table so "5 min" wins over
#  min→minimum, and BEFORE number conversion so digits remain to be spelled)
# ═══════════════════════════════════════════════════════════════════════════════

# Tech fixes that must run BEFORE abbreviation overrides: ".js" was eaten by
# the JS→javascript rule ("Node.js" → "Node.javascript"); "C++"/"C#" are
# otherwise silent symbols.
_RE_DOT_JS = re.compile(r'\.js\b', re.IGNORECASE)
_RE_DOT_NET = re.compile(r'\.NET\b')
_RE_C_PLUS_PLUS = re.compile(r'\bC\+\+', re.IGNORECASE)
_RE_C_SHARP = re.compile(r'\b([CF])#', re.IGNORECASE)

# Keyboard shortcuts: "Ctrl+S" / "Cmd+Alt+Del" — engines otherwise read
# the glued "+" as silence or a stutter ("controlS").  The modifier word
# anchors the match, so prose plus signs ("C++", "a + b") are untouched.
_RE_KEY_SHORTCUT = re.compile(
    r'\b((?:Ctrl|Cmd|Alt|Shift|Win|Meta|Fn|Option)'
    r'(?:\s*\+\s*[A-Za-z][A-Za-z0-9]*)+)\b',
    re.IGNORECASE,
)
# "+/-" before a number: the standalone-plus rule shredded it into
# "plus /-3".  Spoken as the ± glyph already is: "plus or minus".
_RE_PLUS_SLASH_MINUS = re.compile(r'\+/-')

# ─── Degree units: must run BEFORE the generic ° → "degrees" rule, else
# "20°C" became "twenty degreesC" with a glued C ─────────────────────────────
_RE_DEGREE_UNIT = re.compile(
    r'(-?)(\d[\d,]*(?:\.\d+)?)\s*°\s*([CFKcfk])\b'
)
_DEGREE_UNIT_WORDS = {
    'C': 'Celsius', 'F': 'Fahrenheit', 'K': 'Kelvin',
    'c': 'Celsius', 'f': 'Fahrenheit', 'k': 'Kelvin',
}

# ─── Unicode fractions → spoken words (½ ⅓ ¾ … are otherwise unread) ────────
_UNICODE_FRACTIONS = {
    '½': 'one half', '⅓': 'one third', '⅔': 'two thirds',
    '¼': 'one quarter', '¾': 'three quarters',
    '⅕': 'one fifth', '⅖': 'two fifths', '⅗': 'three fifths',
    '⅘': 'four fifths', '⅙': 'one sixth', '⅚': 'five sixths',
    '⅐': 'one seventh', '⅑': 'one ninth', '⅒': 'one tenth',
    '⅛': 'one eighth', '⅜': 'three eighths', '⅝': 'five eighths',
    '⅞': 'seven eighths',
}
_RE_UNICODE_FRACTION = re.compile('[' + ''.join(_UNICODE_FRACTIONS.keys()) + ']')

# ─── Superscripts: 5 m² → "5 square meters", E=mc² → "E equals m c squared" ─
_RE_UNIT_SQUARED = re.compile(
    r'\b((?:km|cm|mm|nm|mi|ft|yd|m))\s*²\b', re.IGNORECASE)
_RE_UNIT_CUBED = re.compile(
    r'\b((?:km|cm|mm|nm|mi|ft|yd|m))\s*³\b', re.IGNORECASE)
_SQUARED_UNIT_WORDS = {
    'km': 'square kilometers', 'cm': 'square centimeters',
    'mm': 'square millimeters', 'nm': 'square nanometers',
    'mi': 'square miles', 'ft': 'square feet', 'yd': 'square yards',
    'm': 'square meters',
}
_CUBED_UNIT_WORDS = {
    'km': 'cubic kilometers', 'cm': 'cubic centimeters',
    'mm': 'cubic millimeters', 'nm': 'cubic nanometers',
    'mi': 'cubic miles', 'ft': 'cubic feet', 'yd': 'cubic yards',
    'm': 'cubic meters',
}
_RE_ALPHA_SQUARED = re.compile(r'([A-Za-z])\s*²\b')
_RE_ALPHA_CUBED = re.compile(r'([A-Za-z])\s*³\b')
_RE_DIGIT_SQUARED = re.compile(r'(?<![A-Za-z²])(\d)\s*²\b')
_RE_DIGIT_CUBED = re.compile(r'(?<![A-Za-z³])(\d)\s*³\b')

# ─── Measurement units ───────────────────────────────────────────────────────
# (pattern-fragment, spoken form, case_sensitive).  Digit-anchored: only
# expands when a number directly precedes the unit, so "min" as a plain word
# still becomes "minimum" via the abbreviation table but "5 min" becomes
# "five minutes".  Compound forms must be listed first.
_UNIT_COMPOUNDS = [
    (r'km\s*/\s*h', 'kilometers per hour', False),
    (r'km\s*/\s*s', 'kilometers per second', False),
    (r'm\s*/\s*s', 'meters per second', False),
    # Data throughput: "80 MB/s" — without these compounds the unit rule
    # consumed "MB" and the word/word slash rule later read "/s" as "or s".
    # Case matters: B = bytes, b = bits.
    (r'KB\s*/\s*s', 'kilobytes per second', True),
    (r'MB\s*/\s*s', 'megabytes per second', True),
    (r'GB\s*/\s*s', 'gigabytes per second', True),
    (r'TB\s*/\s*s', 'terabytes per second', True),
    (r'Kb\s*/\s*s', 'kilobits per second', True),
    (r'Mb\s*/\s*s', 'megabits per second', True),
    (r'Gb\s*/\s*s', 'gigabits per second', True),
    (r'Tb\s*/\s*s', 'terabits per second', True),
]
_UNITS_CASE_INSENSITIVE = [
    # time
    (r'mins?', 'minutes'), (r'sec(?!ond)s?', 'seconds'), (r'hrs?', 'hours'),
    (r'yrs?', 'years'), (r'days', 'days'), (r'wks?', 'weeks'),
    (r'ms', 'milliseconds'), (r'ns', 'nanoseconds'),
    # distance
    (r'km', 'kilometers'), (r'cm', 'centimeters'), (r'mm', 'millimeters'),
    (r'nm', 'nanometers'), (r'mi', 'miles'), (r'ft', 'feet'),
    (r'yd', 'yards'),
    # weight / volume
    (r'kg', 'kilograms'), (r'mg', 'milligrams'), (r'lbs?', 'pounds'),
    (r'oz', 'ounces'), (r'gal', 'gallons'), (r'ml', 'milliliters'),
    (r'l', 'liters'),
    # speed / power / misc
    (r'mph', 'miles per hour'), (r'kwh', 'kilowatt hours'),
    (r'kw', 'kilowatts'), (r'hp', 'horsepower'), (r'db', 'decibels'),
    (r'ppm', 'parts per million'), (r'ppb', 'parts per billion'),
    # media / design metrics (digit-anchored, so "60 fps" is expanded
    # but "fps" inside prose is left to the engine)
    (r'fps', 'frames per second'), (r'rpm', 'revolutions per minute'),
    (r'dpi', 'dots per inch'), (r'px', 'pixels'),
    # frequency (case variants like GHZ are common in slides)
    (r'thz', 'terahertz'), (r'ghz', 'gigahertz'), (r'mhz', 'megahertz'),
    (r'khz', 'kilohertz'), (r'hz', 'hertz'),
]
_UNITS_CASE_SENSITIVE = [
    # data size — B/b distinction matters (bytes vs bits)
    (r'KB', 'kilobytes'), (r'MB', 'megabytes'), (r'GB', 'gigabytes'),
    (r'TB', 'terabytes'), (r'PB', 'petabytes'),
    (r'Kb', 'kilobits'), (r'Mb', 'megabits'), (r'Gb', 'gigabits'),
    (r'Kbps', 'kilobits per second'), (r'Mbps', 'megabits per second'),
    (r'Gbps', 'gigabits per second'),
    # binary prefixes (IEC): "5 GiB" → "five gibibytes"
    (r'KiB', 'kibibytes'), (r'MiB', 'mebibytes'), (r'GiB', 'gibibytes'),
    (r'TiB', 'tebibytes'),
    # battery capacity
    (r'mAh', 'milliamp hours'),
    # power — mW (milliwatts) vs MW (megawatts)
    (r'MW', 'megawatts'), (r'GW', 'gigawatts'), (r'mW', 'milliwatts'),
]
_RE_UNIT_TMPL = r'\b(\d+(?:,\d{{3}})*(?:\.\d+)?)\s*(?:{})\b'
_UNIT_EXPANSIONS = []
for _frag, _word, _cs in (
        _UNIT_COMPOUNDS
        + [(f, w, True) for f, w in _UNITS_CASE_SENSITIVE]
        + [(f, w, False) for f, w in _UNITS_CASE_INSENSITIVE]):
    _UNIT_EXPANSIONS.append((
        re.compile(_RE_UNIT_TMPL.format(_frag),
                   0 if _cs else re.IGNORECASE),
        _word,
    ))

# Bare lowercase "m" after a digit = meters (uppercase M is handled by the
# magnitude rule below as "million", and Kelvin "300 K" is left alone).
_RE_BARE_METERS = re.compile(r'\b(\d+(?:,\d{3})*(?:\.\d+)?)\s*(m)\b')


def _singular_unit(word: str) -> str:
    """Plural unit word → singular for exact "1" ("1 kilometers" sounds off)."""
    if word in ('feet',):
        return 'foot'
    tokens = word.split(' ')
    first = tokens[0]
    if first.endswith('s') and not first.endswith('ss') and first != 'per':
        tokens[0] = first[:-1]
    return ' '.join(tokens)


def _singular_measure(word: str) -> str:
    """Multi-word measure → singular: 'square meters' → 'square meter',
    'cubic feet' → 'cubic foot' (the leading word carries no plural)."""
    if word.endswith('feet'):
        return word[:-4] + 'foot'
    if word.endswith('s') and not word.endswith('ss'):
        return word[:-1]
    return word

# ─── Micro units ─────────────────────────────────────────────────────────────────
# [µμ] — U+00B5 micro sign and Greek U+03BC mu both occur in real text
_RE_MICRO_SECONDS = re.compile(r'\b(\d[\d,]*(?:\.\d+)?)\s*[µμ]s\b')
_RE_MICRO_METERS = re.compile(r'\b(\d[\d,]*(?:\.\d+)?)\s*[µμ]m\b')
_RE_MICRO_GRAMS = re.compile(r'\b(\d[\d,]*(?:\.\d+)?)\s*[µμ]g\b')
_RE_MICRO_BARE = re.compile(r'µ')

# ─── Acceleration / kinematic compounds with superscripts ────────────────────
# The generic unit template ends in \b which cannot match after a non-word
# superscript, so m/s² and m²/s need dedicated patterns.
# BUG FIX: m/s² is acceleration — "meters per second SQUARED".  The old
# compound table said "square meters per second", which is m²/s.
# NOTE: these MUST run BEFORE the ²/³ superscript rules — ² is alphanumeric
# in Python's \w (str.isalnum() is True), so "m²/s" would be split into
# "square meters /s" by the unit² rule and "s²" in "m/s²" by the alpha² rule.
_RE_M_PER_S2 = re.compile(
    r'\b(\d[\d,]*(?:\.\d+)?)\s*m\s*/\s*s²(?![A-Za-z0-9])', re.IGNORECASE)
_RE_M2_PER_S = re.compile(
    r'\b(\d[\d,]*(?:\.\d+)?)\s*m²\s*/\s*s\b', re.IGNORECASE)
_RE_M3_PER_S = re.compile(
    r'\b(\d[\d,]*(?:\.\d+)?)\s*m³\s*/\s*s\b', re.IGNORECASE)
_RE_BARE_M_PER_S2 = re.compile(
    r'\bm\s*/\s*s²(?![A-Za-z0-9])', re.IGNORECASE)
_RE_BARE_M2_PER_S = re.compile(r'\bm²\s*/\s*s\b', re.IGNORECASE)
_RE_BARE_M3_PER_S = re.compile(r'\bm³\s*/\s*s\b', re.IGNORECASE)

# ─── Plain-digit powers (no superscript available): "25 m3", "12 m2" ───────
_RE_PLAIN_VOLUME = re.compile(
    r'\b(\d[\d,]*(?:\.\d+)?)\s*(km3|cm3|mm3|m3)\b', re.IGNORECASE)
_RE_PLAIN_AREA = re.compile(
    r'\b(\d[\d,]*(?:\.\d+)?)\s*(km2|cm2|mm2|m2)\b', re.IGNORECASE)
_PLAIN_VOLUME_WORDS = {
    'km3': 'cubic kilometers', 'cm3': 'cubic centimeters',
    'mm3': 'cubic millimeters', 'm3': 'cubic meters',
}
_PLAIN_AREA_WORDS = {
    'km2': 'square kilometers', 'cm2': 'square centimeters',
    'mm2': 'square millimeters', 'm2': 'square meters',
}

# ─── "sq ft" / "cu ft" style abbreviations (no leading digit needed) ────────
_RE_SQ_UNIT = re.compile(
    r'\bsq\.?\s*(ft|yd|mi|km|mm|cm|m|in)\b', re.IGNORECASE)
_SQ_UNIT_WORDS = {
    'ft': 'square feet', 'yd': 'square yards', 'mi': 'square miles',
    'km': 'square kilometers', 'mm': 'square millimeters',
    'cm': 'square centimeters', 'm': 'square meters', 'in': 'square inches',
}
_RE_CU_UNIT = re.compile(
    r'\bcu\.?\s*(ft|yd|mm|cm|m|in)\b', re.IGNORECASE)
_CU_UNIT_WORDS = {
    'ft': 'cubic feet', 'yd': 'cubic yards', 'mm': 'cubic millimeters',
    'cm': 'cubic centimeters', 'm': 'cubic meters', 'in': 'cubic inches',
}

# ─── Bare compound rates (no leading digit): "speed in km/h" ────────────────
# The digit-anchored unit rules skip these and the word/word slash rule
# later would read "km/h" as "km or h" and "80 MB/s" as "megabytes or s".
_RE_BARE_RATE_COMPOUNDS = [
    (re.compile(r'\bkm\s*/\s*h\b', re.IGNORECASE), 'kilometers per hour'),
    (re.compile(r'\bkm\s*/\s*s\b', re.IGNORECASE), 'kilometers per second'),
    (re.compile(r'\bm\s*/\s*s\b', re.IGNORECASE), 'meters per second'),
    (re.compile(r'\bKB\s*/\s*s\b'), 'kilobytes per second'),
    (re.compile(r'\bMB\s*/\s*s\b'), 'megabytes per second'),
    (re.compile(r'\bGB\s*/\s*s\b'), 'gigabytes per second'),
    (re.compile(r'\bTB\s*/\s*s\b'), 'terabytes per second'),
    (re.compile(r'\bKb\s*/\s*s\b'), 'kilobits per second'),
    (re.compile(r'\bMb\s*/\s*s\b'), 'megabits per second'),
    (re.compile(r'\bGb\s*/\s*s\b'), 'gigabits per second'),
    (re.compile(r'\bTb\s*/\s*s\b'), 'terabits per second'),
]

# ─── Caret exponent notation: 10^6, 2^10, x^2, e^x, 10^-6 ───────────────────
# Previously "2^10" was CORRUPTED: the integer rule read the exponent
# ("2^ten") and "10^6 kg" let the unit rule anchor on the exposed "6 kg".
_RE_CARET_NUM = re.compile(
    r'(?<![A-Za-z0-9^])(\d{1,3}(?:,\d{3})+|\d+)\s*\^\s*(-?)(\d+)')
_RE_CARET_NUM_LETTER = re.compile(
    r'(?<![A-Za-z0-9^])(\d{1,3}(?:,\d{3})+|\d+)\s*\^\s*(-?)([A-Za-z])(?![A-Za-z])')
_RE_CARET_ALPHA_NUM = re.compile(r'\b([A-Za-z][A-Za-z0-9]*)\s*\^\s*(-?)(\d+)\b')
_RE_CARET_ALPHA_LETTER = re.compile(
    r'\b([A-Za-z][A-Za-z0-9]*)\s*\^\s*(-?)([A-Za-z])(?![A-Za-z])')
_RE_LEFTOVER_CARET = re.compile(r'\s*\^\s*')

# ─── Unicode superscripts beyond ²/³: 10⁶, 2⁴⁵, 10⁻⁶, xⁿ ───────────────
_SUP_DIGIT_CHARS = '\u00b9\u00b2\u00b3\u2070\u2074\u2075\u2076\u2077\u2078\u2079'
_RE_SUP_RUN_MINUS = re.compile('\u207b[' + _SUP_DIGIT_CHARS + ']+')
_RE_SUP_RUN_MULTI = re.compile('[' + _SUP_DIGIT_CHARS + ']{2,}')
_RE_SUP_RUN_SINGLE = re.compile('[\u00b9\u2070\u2074\u2075\u2076\u2077\u2078\u2079]')
_RE_SUP_N = re.compile('\u207f')
_SUP_DIGIT_VALUES = {
    '\u00b9': '1', '\u00b2': '2', '\u00b3': '3', '\u2070': '0',
    '\u2074': '4', '\u2075': '5', '\u2076': '6', '\u2077': '7',
    '\u2078': '8', '\u2079': '9',
}
# Unicode subscripts: "CO₂" → "CO two" (subscript digits are unread glyphs)
_RE_SUBSCRIPT = re.compile('[\u2080-\u2089]+')

# ─── Math glyphs with no native TTS reading ────────────────────────────────
_RE_MEGOHM = re.compile(r'(\d[\d,]*(?:\.\d+)?)\s*M\s*[\u03a9\u2126]')
_RE_MILLIOHM = re.compile(r'(\d[\d,]*(?:\.\d+)?)\s*m\s*[\u03a9\u2126]')
_RE_KILOHM = re.compile(
    r'(\d[\d,]*(?:\.\d+)?)\s*k\s*[\u03a9\u2126]', re.IGNORECASE)
_RE_OHM = re.compile(r'(\d[\d,]*(?:\.\d+)?)\s*[\u03a9\u2126]')
_RE_BARE_OHM = re.compile('[\u03a9\u2126]')
_RE_CELSIUS_GLYPH = re.compile(r'(-?)(\d[\d,]*(?:\.\d+)?)\s*℃')
_RE_FAHRENHEIT_GLYPH = re.compile(r'(-?)(\d[\d,]*(?:\.\d+)?)\s*℉')
_RE_BARE_CELSIUS = re.compile('℃')
_RE_BARE_FAHRENHEIT = re.compile('℉')
# √ π Δ ∆ ∫ № — common in math/science slides, unread glyphs otherwise
_RE_SQRT = re.compile('√')
_RE_PI = re.compile('π')
_RE_DELTA = re.compile('[\u0394\u2206]')   # Greek capital delta / increment
_RE_INTEGRAL = re.compile('∫')
_RE_NUMERO_SIGN = re.compile('№')

# ─── Programming operators: "1 !== 2" was read "is not equal to equals"
# (!= fired inside !==), "a && b" became "a and and b" (& fired twice). ────
_RE_STRICT_NOT_EQUAL = re.compile(r'!==')
_RE_STRICT_EQUAL = re.compile(r'===')
_RE_DOUBLE_EQUAL = re.compile(r'(?<![=!<>])==(?![=>])')
_RE_AND_AND = re.compile(r'&&')
_RE_OR_OR = re.compile(r'\|\|')

# ═════════════════════════════════════════════════════════════════════════
#  MATH PRONUNCIATION PASS
#  (runs before the superscript/subscript rules — d²y/dx² and log₂ would be
#  shredded — before the caret rules, and before the parenthetical rule,
#  which would speak sin(x) as "sine, x,")
# ═════════════════════════════════════════════════════════════════════════

# Python-style exponent: an unmatched "**" (markdown pairs were already
# stripped) between math operands is an exponent — "2**10" → caret form.
_RE_ASCII_POW = re.compile(r'\*\*')

# Scientific notation: "1e9" / "2.5e-3" — engines read "1e9" as "one e
# nine".  Spoken "1 times 10^9" and the caret rule finishes the job.
# The exponent is capped at 4 digits so absurd inputs keep the old path.
_RE_SCI_NOTATION = re.compile(r'\b(\d+(?:\.\d+)?)[eE]([+-]?\d{1,4})\b')

# Parenthesized exponent on a letter base: "e^(i*pi)" → "e to the i*pi"
# (the leftover-caret rule would otherwise strand the parens for the
# parenthetical rule, which reads them as an aside).  Digit bases keep the
# caret rules, which already speak "ten to the tenth" correctly.
_RE_CARET_PAREN_EXP = re.compile(
    r'\b([A-Za-z][A-Za-z0-9]*)\s*\^\s*\(([^()]{1,15})\)')

# A minus directly before a math function ("-sin(x)", "= -log(x)") is a
# negated value, spoken "minus".  Anchored like the negative-number rule;
# the \b guards keep "-long list" bullets out.
_RE_NEGATIVE_MATH_FUNC = re.compile(
    r'(?:(?<!\S)|(?<==)|(?<=[(\[]))-(?=sin\b|cos\b|tan\b|cot\b|log\b|'
    r'ln\b|sqrt\b|exp\b|lim\b)')

# Derivatives.  d²y/dx² MUST run before the superscript rules (which would
# read "d squared y"), and dy/dx before the word/word slash rule ("dy or
# dx").  \b guards keep "2D/3D" and "id/id" out (no word boundary inside).
_RE_SECOND_DERIV = re.compile(r'\bd²([A-Za-z]+)\s*/\s*d([A-Za-z]+)²\b')
_RE_SECOND_PARTIAL_DERIV = re.compile(
    r'∂²([A-Za-z]+)\s*/\s*∂([A-Za-z]+)²\b')
_RE_PARTIAL_DERIV = re.compile(r'∂([A-Za-z])\s*/\s*∂([A-Za-z])\b')
_RE_DERIV = re.compile(r'\bd([a-zA-Z])\s*/\s*d([a-zA-Z])\b')
_RE_D_OVER_DVAR = re.compile(r'\bd\s*/\s*d([a-zA-Z])\b')

# Prime notation: f'(x), x' = 2 — the apostrophe is spoken "prime".
# Only when followed by an operator or opening paren: a quote-final
# apostrophe ("the 'best'.") and possessives ("it's", "o'clock") never match.
_RE_PRIME = re.compile(
    r"(?<![A-Za-z0-9])([A-Za-z])'(?=\s*[=+\-*/^(])")
# The primed-call form "f'(x)" — after the prime rule the text is
# "f prime(x)"; this rule speaks it "f prime of x".
_RE_PRIME_CALL = re.compile(r'\b([A-Za-z]) prime\s*\(([^()]*)\)')

# Inverse trigonometric notation: sin⁻¹(x) → "inverse sine of x".
# Without this, the superscript rule reads "sin to the minus first x" and
# the caret rule reads "sin^-1" the same wrong way.  Both the Unicode
# superscript form (U+207B U+00B9) and the ASCII caret form are matched;
# the optional trailing "(...)" argument is consumed so the letter-func /
# parenthetical rules never shred it.  Runs BEFORE the superscript rules.
_INVERSE_FUNC_WORDS = {
    'sin': 'inverse sine', 'cos': 'inverse cosine', 'tan': 'inverse tangent',
}
_RE_INVERSE_FUNC_SUB = re.compile(
    r'\b(sin|cos|tan)\s*\u207b\u00b9\s*(?:\(([^()]*)\))?')
_RE_INVERSE_FUNC_CARET = re.compile(
    r'\b(sin|cos|tan)\s*\^\s*\(?\s*-\s*1\s*\)?\s*(?:\(([^()]*)\))?')

# Factorial: "n!" → "n factorial" (engines speak a bare "!" as sentence
# emphasis: "n is 120").  Only single lowercase math variables; the
# lookahead keeps genuine exclamations safe — "Pick n! Then…" (sentence
# end, capital follow) and "a != b" (no space before "=") never match.
_RE_FACTORIAL = re.compile(
    r"\b([a-z])!(?=\s*[,\;:\)\]]|\s+[=(a-z(])")

# Assignment / definition operator: "f(x) := x²" → "is defined as" (the
# leftover-equals rule would speak it "x : equals x squared").  Also the
# Python walrus operator.
_RE_ASSIGN_OP = re.compile(r':=')

# Remainder operator: "5 mod 3" → "5 modulo 3".  MUST run before the
# abbreviation table, whose mod→"module" rule would say "5 module 3".
# Digit-anchored on both sides so tech slides ("auth mod") are untouched.
_RE_MODULO = re.compile(r'(?<=\d\s)mod(?=\s+\d)', re.IGNORECASE)
# Generic single-letter function: f(x), P(A) → "f of x", "P of A" — the
# parenthetical rule would otherwise read the argument as an aside
# ("eff, ex,").  Runs AFTER the named-function rules; a letter glued into
# a longer token ("Ctrl(...)") never matches.
_RE_LETTER_FUNC_CALL = re.compile(r'\b([A-Za-z])\(([^()]{1,40})\)')

# Division spelled with a closing paren or a digit before the slash and a
# letter/paren after: "P(A) / P(B)", "5 / (x+1)" → "divided by" (the
# word/word slash rule would say "or").  The closing paren / spacing is
# kept via capture groups — only the slash is replaced.
_RE_PAREN_SLASH_DIV = re.compile(
    r'(\))\s*/\s*(?=[A-Za-z(])|(?<=\d)(\s*)/\s*(?=\()')

# Tight multiplication with a non-digit left operand: "a*b", "i*pi",
# ") * P(A)" → "times" (digit*digit products belong to the multiply rule,
# which knows the "1920 by 1080" dimension reading).
_RE_TIGHT_MULTIPLY = re.compile(
    r'(?<=[A-Za-z)])\s*\*\s*(?=[A-Za-z0-9(])|(?<=\d)\s*\*\s*(?=[A-Za-z(])')

# Conditional probability pipe: P(A|B) → "A given B".  Only pipes INSIDE
# parentheses convert — shell pipes ("ls | grep"), boolean "p|q" and
# markdown tables keep their bar (a repl checks the enclosing parens).
_RE_PIPE = re.compile(r'\|')

# Math function calls: sin(x) → "sine of x".  The parenthetical-pause rule
# would otherwise speak the argument as an aside ("sine, x,").  Alternation
# is sorted longest-first so "sinh" wins over "sin" and "arcsin" over both.
_MATH_FUNC_SPOKEN = {
    'sin': 'sine of', 'sinh': 'sinch of', 'arcsin': 'arc sine of',
    'cos': 'cosine of', 'cosh': 'cosh of', 'arccos': 'arc cosine of',
    'tan': 'tangent of', 'tanh': 'tanch of', 'arctan': 'arc tangent of',
    'cot': 'cotangent of', 'sqrt': 'square root of', 'log': 'log of',
    'ln': 'natural log of', 'exp': 'exponential of', 'lim': 'limit of',
    'max': 'maximum of', 'min': 'minimum of',
    # Improvement pass 5: the remaining trig/inverse functions and the
    # algebra calls engines read as letter soup ("det(A)" → "dett").
    'sec': 'secant of', 'csc': 'cosecant of', 'det': 'determinant of',
    'gcd': 'G C D of', 'lcm': 'L C M of',
}
_RE_MATH_FUNC_CALL = re.compile(
    r'\b(' + '|'.join(
        sorted(_MATH_FUNC_SPOKEN, key=len, reverse=True)) +
    r')\s*\(([^()]*)\)', re.IGNORECASE)

# Bare function names engines garble ("ln" read as "lin", "sin" as /sɪn/).
# "log"/"max"/"min" are ordinary words or already fine and stay untouched.
_MATH_FUNC_BARE = {
    'sin': 'sine', 'sinh': 'sinch', 'arcsin': 'arc sine',
    'cos': 'cosine', 'arccos': 'arc cosine',
    'tan': 'tangent', 'tanh': 'tanch', 'arctan': 'arc tangent',
    'cot': 'cotangent', 'sqrt': 'square root of',
    'ln': 'natural log', 'exp': 'exponential', 'lim': 'limit',
    # "csc" is never an English word, so the bare form is safe.  Bare
    # "sec" is deliberately ABSENT: the abbreviation table reads
    # "30 sec" as seconds and the rate table reads "ops/sec" — a bare
    # secant expansion would corrupt both.  Call form only (see above).
    'csc': 'cosecant',
}
_RE_MATH_FUNC_BARE = re.compile(
    r'\b(' + '|'.join(
        sorted(_MATH_FUNC_BARE, key=len, reverse=True)) +
    r')\b', re.IGNORECASE)

# Log bases: "log_2 64" / "log₂ 64" → "log base 2 ..." (engines read
# "log underscore two" verbatim or skip the subscript glyph entirely).
# Call forms first so "log_2(8)" becomes "log base 2 of 8" instead of
# leaving "(8)" for the parenthetical rule.
_RE_LOG_BASE_SUB = re.compile(r'\blog([₀-₉]+)\b')
_RE_LOG_BASE_ASCII = re.compile(r'\blog_([0-9]+)\b')
_RE_LOG_BASE_SUB_PAREN = re.compile(r'\blog([₀-₉]+)\s*\(([^()]*)\)')
_RE_LOG_BASE_ASCII_PAREN = re.compile(r'\blog_([0-9]+)\s*\(([^()]*)\)')

# Greek-mu units with a leading digit: "5 μm" → "5 micrometers".  Runs
# BEFORE the Greek glyph table maps a bare μ to "mu".  (U+00B5 µ forms are
# handled by the micro-unit rules; these cover the Greek U+03BC spelling.)
_RE_GREEK_MICRO = re.compile(r'\b(\d[\d,]*(?:\.\d+)?)\s*μ([smg])\b')
_RE_GREEK_MICRO_BARE = re.compile(r'\bμ([smg])\b')
_GREEK_MICRO_WORDS = {
    's': 'microseconds', 'm': 'micrometers', 'g': 'micrograms',
}

# Greek letters and math glyphs beyond π/Δ/Ω (which the explicit-symbol
# loop already speaks).  Look-alike capitals (Α Β Ε Ζ Η Ι Κ Μ Ν Ο Ρ Τ Υ Χ)
# are deliberately NOT mapped — they are keyboard-layout typos away from
# Latin text; Greek omicron 'ο' is excluded for the same reason.  Every
# mapped lowercase name ends in a vowel, so "α²" still resolves through
# the alpha-squared rule to "alpha squared".
_MATH_GLYPH_WORDS = {
    'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'δ': 'delta',
    'ε': 'epsilon', 'ϵ': 'epsilon', 'ζ': 'zeta', 'η': 'eta',
    'θ': 'theta', 'ϴ': 'theta', 'ι': 'iota', 'κ': 'kappa',
    'λ': 'lambda', 'μ': 'mu', 'ν': 'nu', 'ξ': 'xi',
    'ρ': 'rho', 'σ': 'sigma', 'ς': 'sigma', 'τ': 'tau',
    'υ': 'upsilon', 'φ': 'phi', 'ϕ': 'phi', 'χ': 'chi',
    'ψ': 'psi', 'ω': 'omega', 'π': 'pi',
    'Δ': 'delta', '∆': 'delta', 'Γ': 'gamma', 'Θ': 'theta',
    'Λ': 'lambda', 'Ξ': 'xi', 'Σ': 'sigma', 'Π': 'product of',
    'Φ': 'phi', 'Ψ': 'psi',
    '∑': 'sum of', '∏': 'product of', '∂': 'partial',
    '∛': 'cube root of', '∈': 'in', '∉': 'not in',
    '∀': 'for all', '∃': 'there exists', '∪': 'union',
    '∩': 'intersection', '⊂': 'is a subset of',
    '⊃': 'is a superset of', '⊆': 'is a subset of',
    '⊇': 'is a superset of', '∅': 'the empty set',
    '∝': 'is proportional to', '≡': 'is equivalent to',
    '∴': 'therefore', '∵': 'because',
    '⊥': 'is perpendicular to', '∠': 'angle',
    # Improvement pass 5 — logic glyphs (∧/∨/¬ were silently skipped by
    # engines), biconditional arrows, congruence/similarity, higher roots
    # and the contour integral.
    '∧': 'and', '∨': 'or', '¬': 'not',
    '⇔': 'if and only if', '↔': 'if and only if', '↦': 'maps to',
    '≅': 'is congruent to', '≃': 'is approximately equal to',
    '∜': 'fourth root of', '∮': 'contour integral of',
    '∼': 'is similar to',
}
_RE_MATH_GLYPHS = re.compile(
    '|'.join(re.escape(k) for k in
             sorted(_MATH_GLYPH_WORDS, key=len, reverse=True)))

# Equals signs the spaced rule missed ("x=5", "a=b=c"): after the strict/
# double/plus-equals and comparison rules have run, every remaining "=" is
# a lone assignment or comparison.
_RE_EQUALS_LEFTOVER = re.compile(r'=')

# ─── Trailing currency: "50$" → "fifty dollars" (the "$50" prefix form
# was already spoken; the trailing form left a bare dollar sign). ────────────
_RE_TRAILING_DOLLAR = re.compile(r'\b(\d[\d,]*(?:\.\d+)?)\s*\$')

# ─── Identifier readability: snake_case and camelCase ──────────────────────
# TTS engines garble "my_var_name" and "getUserData"; splitting them into
# words is far more intelligible.  Known mixed-case brand names whose
# reading would be damaged by a split are preserved.
_RE_SNAKE_UNDERSCORE = re.compile(r'(?<=\w)_(?=\w)')
_RE_LEFTOVER_UNDERSCORE = re.compile(r'_+')
_RE_CAMEL_WORD = re.compile(r'\b[a-z]+(?:[A-Z]+[a-z]*)+\b')
_RE_CAMEL_SPLIT = re.compile(r'(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])')
_CAMEL_KEEP: FrozenSet[str] = frozenset({
    # Apple-style "i" prefixes and other brands with a lower→upper seam
    'iphone', 'ipad', 'ipod', 'imac', 'ios', 'itunes', 'icloud', 'imessage',
    # Apple OS names MUST survive camel splitting: the version-skip builder
    # runs after this step and matches "macOS 14.2" / "watchOS 10" by
    # brand — a split ("mac OS 14.2") loses the protection and the version
    # digits get cardinally read ("fourteen point two.1").
    'macos', 'watchos', 'tvos', 'ipados', 'visionos',
    'ebay', 'jquery', 'opencv', 'github', 'gitlab', 'whatsapp', 'powerpoint',
    # lowercase-first unit symbols ("5 kW" was already expanded; the bare
    # forms must not be split: pH, dB, mL, kW, GHz, mmHg, kPa, mAh ...)
    'ph', 'db', 'ml', 'kw', 'kwh', 'mv', 'ma', 'kpa', 'mmhg',
    'khz', 'mhz', 'ghz', 'thz', 'mah',
})

# ─── Magnitude suffixes: 10K → "ten thousand", 5M → "five million" ──────────
# Attached form only ("10K"), never spaced ("300 K" may be Kelvin).
# 2K/4K/8K (display resolution), 24K (gold carat) and 401K (retirement plan)
# are excluded — TTS reads those naturally as "… kay".
_RE_MAGNITUDE_KMB = re.compile(r'\b(\d[\d,]*(?:\.\d+)?)([KkMB])\b')
_MAGNITUDE_WORDS = {'K': 'thousand', 'M': 'million', 'B': 'billion'}
_MAGNITUDE_K_EXCLUDE = {2, 4, 8, 24, 401}

# ─── Rate shorthand: "ops/sec", "msg/min" → "per second"/"per minute" ────────
# The word/word slash rule would read them "ops or sec", and the
# abbreviation table's (?=\s|$) lookahead then mangles the exposed
# "sec"/"min" ("ops or seconds", "msg or minimum").  A word followed by a
# time-unit slash is spoken "per".  Must run AFTER the km/h and MB/s
# compound tables (they own those forms) and BEFORE the abbreviation
# table.  Bare "m" is deliberately absent (meters, not minutes).
_RATE_UNIT_WORDS = {
    's': 'second', 'sec': 'second', 'secs': 'seconds',
    'min': 'minute', 'mins': 'minutes',
    'h': 'hour', 'hr': 'hour', 'hrs': 'hours',
    'day': 'day', 'days': 'day',
    'wk': 'week', 'wks': 'weeks',
    'mo': 'month', 'mos': 'months',
    'yr': 'year', 'yrs': 'years', 'y': 'year',
}
_RE_RATE_SLASH = re.compile(
    r'\b([A-Za-z][A-Za-z0-9]*)\s*/\s*('
    + '|'.join(_RATE_UNIT_WORDS) + r')\b',
    re.IGNORECASE,
)

# ─── Informal shorthand ──────────────────────────────────────────────────────
_RE_W_O = re.compile(r'\bw/o\b', re.IGNORECASE)                # w/o → without
_RE_W_SLASH = re.compile(r'\bw/\s*(?=[A-Za-z])', re.IGNORECASE)  # w/ → with

# ─── Symbols without native TTS readings ────────────────────────────────────
_RE_PLUS_MINUS = re.compile(r'±')
_RE_APPROX_EQUAL = re.compile(r'≈')
_RE_NOT_EQUAL_SIGN = re.compile(r'≠')
_RE_INFINITY = re.compile(r'∞')
_RE_DIVIDE_SIGN = re.compile(r'÷')
_RE_ARROW_RIGHT = re.compile(r'→|⇒')
_RE_ASCII_ARROW = re.compile(r'->|=>')   # "A -> B" → "A to B"
_RE_PER_MILLE = re.compile(r'‰')
_RE_SECTION_SIGN = re.compile(r'§')
_RE_PARAGRAPH_SIGN = re.compile('¶')
# Unicode comparison signs — the ASCII forms (>=, <=) are handled in the
# operator pass, but the glyphs themselves have no engine reading.
_RE_GTE_SIGN = re.compile('≥')
_RE_LTE_SIGN = re.compile('≤')
# Silent typographic symbols — engines either skip or stutter on these
# (guillemets « » and angle quotes ‹ › included)
_RE_SILENT_SYMBOLS = re.compile(r'[©®™†‡•◦‣·«»‹›\u00a9\u00ae\u2122]')
_RE_MINUS_SIGN = re.compile(r'−')   # U+2212 math minus → ASCII hyphen

# ─── Month abbreviations: "Jan 15" → "January 15" ───────────────────────────
_RE_MONTH_ABBR = re.compile(
    r'\b(Sept|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?(?=\s*\d)',
    re.IGNORECASE,
)
_MONTH_ABBR_WORDS = {
    'jan': 'January', 'feb': 'February', 'mar': 'March', 'apr': 'April',
    'jun': 'June', 'jul': 'July', 'aug': 'August', 'sep': 'September',
    'sept': 'September', 'oct': 'October', 'nov': 'November',
    'dec': 'December',
}

# ─── Numeric ranges: "10-20" → "10 to 20", "2020—2024" → "2020 to 2024" ─────
# (?<![\d.]) blocks version strings (1.2-3); (?![\d.]) blocks decimals.
_RE_NUMERIC_RANGE = re.compile(
    r'(?<![\d.,])(\d[\d,]*(?:\.\d+)?)\s*[-—]\s*(\d[\d,]*(?:\.\d+)?)(?![\d.])'
)

# ─── Dotted quads (IPv4 addresses, CIDR blocks, host:port) ────────────
# "192.168.1.1" must be read digit-by-digit with "dot" separators — the
# decimal rule otherwise garbles it ("one hundred ninety-two point one six
# eight.one ...").  Optional suffixes: CIDR "10.0.0.0/8" → "... slash 8"
# and "192.168.1.1:8080" → "... port eight zero eight zero".
# Runs BEFORE the decimal rule; dotted VERSION strings ("v2.5.1.2") are
# excluded by the prefix guard below.
_RE_DOTTED_QUAD = re.compile(
    r'\b(\d{1,3}(?:\.\d{1,3}){3})'
    r'(?:(?::(\d{2,5}))|(?:\s*/\s*(\d{1,2})))?')
_RE_VERSION_PREFIX_TAIL = re.compile(
    r'\b(?:v|ver|version|rev|rel|release|build)\.?\s*$', re.IGNORECASE)

# ─── ISO dates: "2024-01-15" → "January fifteenth, twenty twenty-four" ───
# Must run BEFORE the numeric-range rule, which would read "2024-01" as
# "2024 to 01".  _date_to_words validates the parts and rejects garbage.
_RE_ISO_DATE = re.compile(r'\b(20\d{2})-(\d{1,2})-(\d{1,2})\b')

# ─── Phone numbers: "555-123-4567" → digit-by-digit with pauses ──────────
# Without this, the range rule reads "555-123" as "five hundred fifty-five
# to one hundred twenty-three" and the integer rule garbles the rest.
_RE_PHONE_NUMBER = re.compile(r'\b(\d{3})-(\d{3})-(\d{4})\b')

# ─── Dotted port numbers: "port 8080" → "port eight zero eight zero" ─────
# "Port" anchors the reading; "import 1234" / "passport 4567" don't match
# (no word boundary before "port").
_RE_PORT_NUMBER = re.compile(r'\bport\s+(\d{2,5})\b', re.IGNORECASE)

# ─── Bare unit durations: "45s" → "45 seconds", "24h" → "24 hours" ───────
# Digit-anchored lowercase-only: plain words ("hours") and "3D"/"2D"
# (uppercase D deliberately absent — dimensions) are untouched.
# "m" is intentionally excluded — "5m" is meters (bare-meters rule).
_RE_BARE_DURATION = re.compile(r'\b(\d+(?:,\d{3})*(?:\.\d+)?)\s*([shw])\b(?!\s*of\b)')

# ─── Always-idiomatic number phrases ─────────────────────────────────────────
_RE_24_7 = re.compile(r'\b24\s*/\s*7\b')
# "50-50" / "50/50" → "fifty fifty" — the range rule says "fifty to
# fifty" and the fraction rule "fifty divided by fifty"; the idiom is
# always spoken as a bare pair.
_RE_50_50 = re.compile(r'\b50\s*[-/]\s*50\b')

# ─── Roman numerals in anchored contexts ─────────────────────────────────────
_ROMAN_VALUES = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500,
                 'M': 1000}
_ROMAN_VALIDATE = re.compile(
    r'M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$')
_RE_ROMAN_SECTION = re.compile(
    r'\b(chapter|section|part|book|volume|article|act|scene|step|phase|'
    r'title|world\s+war)\s+([IVXLC]+)\b',
    re.IGNORECASE,
)
_RE_ROMAN_NAME = re.compile(
    r'\b(king|queen|pope|emperor|tsar|czar|henry|charles|louis|elizabeth|'
    r'victoria|pius|gregory|benedict|william|richard|george|edward|james|'
    r'alexander|ferdinand|philip|napoleon|john|paul|ivan|peter|catherine|'
    r'nicholas|wilhelm|frederick|constantine|justinian|otto|gustav|'
    r'alfonso|basil|joseph)\s+([IVXLC]+)\b',
    re.IGNORECASE,
)
_RE_ROMAN_CENTURY = re.compile(
    r'\b([IVXLC]+)\s+(centur(?:y|ies))\b', re.IGNORECASE)


def _roman_to_int(s: str) -> int:
    """Roman numeral → int, or -1 if *s* is not a valid Roman numeral."""
    s = s.upper()
    if not s or len(s) > 12 or not _ROMAN_VALIDATE.match(s):
        return -1
    total, prev = 0, 0
    for ch in reversed(s):
        val = _ROMAN_VALUES[ch]
        if val < prev:
            total -= val
        else:
            total += val
            prev = val
    return total


# ═══════════════════════════════════════════════════════════════════════════════
#  NUMBER-TO-WORDS ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

_ONES = [
    '', 'one', 'two', 'three', 'four', 'five', 'six', 'seven',
    'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen',
    'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen',
]
# Digit words for digit-by-digit readings (IPs, ports, phones): unlike
# _ONES, index 0 is 'zero' — _ONES[0] is '' for cardinal composition.
_DIGIT_WORDS = ('zero',) + tuple(_ONES[1:])
_TENS = [
    '', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy',
    'eighty', 'ninety',
]
_THOUSANDS = [
    ('', ''), ('thousand', 'thousand'), ('million', 'million'),
    ('billion', 'billion'), ('trillion', 'trillion'),
    ('quadrillion', 'quadrillion'), ('quintillion', 'quintillion'),
    ('sextillion', 'sextillion'),
]
_MAX_INT_WORDS = 1000 ** 8 - 1  # largest number spellable with _THOUSANDS
_ORDINAL_MAP = {
    'one': 'first', 'two': 'second', 'three': 'third', 'four': 'fourth',
    'five': 'fifth', 'six': 'sixth', 'seven': 'seventh', 'eight': 'eighth',
    'nine': 'ninth', 'ten': 'tenth', 'eleven': 'eleventh',
    'twelve': 'twelfth', 'thirteen': 'thirteenth', 'fourteen': 'fourteenth',
    'fifteen': 'fifteenth', 'sixteen': 'sixteenth', 'seventeen': 'seventeenth',
    'eighteen': 'eighteenth', 'nineteen': 'nineteenth',
    'twenty': 'twentieth', 'thirty': 'thirtieth', 'forty': 'fortieth',
    'fifty': 'fiftieth', 'sixty': 'sixtieth', 'seventy': 'seventieth',
    'eighty': 'eightieth', 'ninety': 'ninetieth',
    'hundred': 'hundredth', 'thousand': 'thousandth',
    'million': 'millionth', 'billion': 'billionth',
}
_FRACTION_WORDS = {
    '2': 'half', '3': 'third', '4': 'quarter', '5': 'fifth',
    '6': 'sixth', '7': 'seventh', '8': 'eighth', '9': 'ninth',
    '10': 'tenth', '11': 'eleventh', '12': 'twelfth',
}

# Month names for date expressions like 12/25/2024 → "December twenty-fifth"
_MONTH_NAMES = (
    'January', 'February', 'March', 'April', 'May', 'June', 'July',
    'August', 'September', 'October', 'November', 'December',
)

# Calendar dates: M/D/YYYY (three parts).  Two-part tokens like 1/2 stay
# fractions — treating every a/b as a date would corrupt ordinary fractions.
_RE_DATE_M_D_Y = re.compile(r'\b(\d{1,2})\s*/\s*(\d{1,2})\s*/\s*(\d{4})\b')

# Year-context words: a 4-digit number right after one of these is read
# year-style ("twenty twenty-four"); otherwise as a cardinal number.
# A comma directly before the number is also year context — history decks
# date events as "Treaty of Versailles, 1919" (cardinal "one thousand nine
# hundred nineteen" was how engines read it).  Month names too: "May 1945"
# is always a month + year (the month-day rule only consumes 1-2 digit
# days, so bare 4-digit months-years were read as cardinals).
_RE_YEAR_CONTEXT = re.compile(
    r'\b(?:in|since|by|until|till|around|circa|year|copyright|©|during|'
    r'between|throughout)\s+$|,\s+$|'
    r'\b(?:january|february|march|april|may|june|july|august|september|'
    r'october|november|december)\s+$',
    re.IGNORECASE,
)

# ─── Historical era markers: "44 BC", "AD 1066", "2500 BCE", "CE 220" ────────
# Engines read the letters "AD" as the word "ad" and often skip "BC"
# entirely, so both markers are spelled out letter-wise.  Both rules are
# digit-anchored (a bare "BC" — British Columbia — or "AD" in prose stays
# untouched) and case-sensitive (lowercase "44 bc" / "ad 1066" is left to
# the engine).  The trailing dot of "B.C." / "A.D." after the year is
# never consumed, so a sentence period survives ("…in 44 B.C. Julius…");
# a dot *before* a year ("A.D. 1066") is internal and is consumed.
_RE_ERA_BEFORE_YEAR = re.compile(r'\b(A\.?D|C\.?E)\.?\s+(\d{1,4}(?:,\d{3})*)\b')
_RE_ERA_AFTER_YEAR = re.compile(
    r'(?<![\d,.])\b(\d{1,4}(?:,\d{3})*)\s*'
    r'(B\.?C(?:\.?E)?|A\.?D|C\.?E)\b(?![A-Za-z])')
_ERA_SPOKEN = {'BC': 'B C', 'BCE': 'B C E', 'AD': 'A D', 'CE': 'C E'}

# ─── Historical year spans: "1914-1918", "between 1914 and 1918" ─────────────
# A 4-digit pair is nearly always a span of years; both sides are read
# year-style in one pass ("nineteen fourteen to nineteen eighteen").
# The generic range rule leaves bare digits, and the second year then
# lost its year reading ("…to one thousand nine hundred eighteen").
# The leading "from"/"between" word is re-emitted in the replacement —
# only the years are rewritten.  Em dashes were normalized in step 6, so
# both "-" and "—" match.  The lookbehind also blocks "-", so chained
# tokens (European phones "020-7946-0958") can never anchor;
# version/tech skip regions are re-checked on the current text before
# every use.
_RE_YEAR_RANGE = re.compile(
    r'(?<![\d.,-])(\d{4})\s*[-—]\s*(\d{4})\s?'
    r'(B\.?C(?:\.?E)?|A\.?D|C\.?E)?\b(?![A-Za-z])')
_RE_YEAR_BETWEEN = re.compile(
    r'\b(between)\s+(\d{4})\s+and\s+(\d{4})\b', re.IGNORECASE)
_RE_YEAR_FROM_TO = re.compile(
    r'\b(from)\s+(\d{4})\s+to\s+(\d{4})\b', re.IGNORECASE)

# "c. 1860" (circa, history notation).  Applied inside the history pass
# BEFORE the era rules — otherwise "c. 1860 BCE" converts the year first
# and "c." no longer sees digits ("circa" is a year-context word, so
# "c. 1860" reads "circa eighteen sixty").  Requires 3-4 digits after
# the dot so list tails ("a, b, c. 5 more") never fire.
_RE_CIRCA_DOT = re.compile(r'\bc\.(?=\s*\d{3,4}\b)', re.IGNORECASE)

# Decades: "the 1990s" → "the nineteen nineties", "the 90s" → "the nineties".
# The integer pattern cannot match "1990s" (no word boundary before the
# trailing "s"), so engines used to read it digit-by-digit + "ess".
_RE_DECADE_4 = re.compile(r'\b(\d{4})s\b')
_RE_DECADE_2 = re.compile(r'\bthe\s+(\d{2})s\b', re.IGNORECASE)
# Apostrophe decades: "the '90s" / "mid-'80s" — without this the bare
# duration rule later read "90s" as "ninety seconds" ("the ninety
# seconds were wild").  The lookbehind (not \b) so "mid-'90s" matches.
_RE_APOS_DECADE = re.compile(r"(?<![A-Za-z0-9])'(\d{2})s\b")
# "mid-90s" (no apostrophe) is always the decade — nobody waits "mid 90
# seconds".  Fixed-width lookbehind keeps bare "90s" (= 90 seconds) with
# the duration rule.
_RE_MID_DECADE = re.compile(r'(?<=mid-)(\d{2})s\b', re.IGNORECASE)
# Possessive decade/year: "1990's style" → "nineteen nineties style"
# (the integer rule read "one thousand nine hundred ninety's").
_RE_YEAR_POSSESSIVE = re.compile(r"\b(\d{4})'s\b")

# "9/11" must be spoken as a date, not the fraction "nine elevenths".
_RE_9_11 = re.compile(r'\b9\s*/\s*11\b')

# Full month + day: "January 15" → "January fifteenth".  The positive
# lookahead requires a word edge; the negative lookahead skips days that
# already carry an ordinal suffix ("January 15th" is handled later).
_RE_MONTH_DAY = re.compile(
    r'\b(January|February|March|April|May|June|July|August|September|'
    r'October|November|December)\s+(\d{1,2})(?=[\s,.;:!?]|$)(?!\s*'
    r'(?:st|nd|rd|th)\b)',
    re.IGNORECASE,
)


def _date_to_words(month: str, day: str, year: str) -> str:
    """Convert M/D/Y date parts to spoken words, or '' if not a valid date."""
    try:
        m_int, d_int, y_int = int(month), int(day), int(year)
    except ValueError:
        return ''
    if not (1 <= m_int <= 12 and 1 <= d_int <= 31):
        return ''
    out = f'{_MONTH_NAMES[m_int - 1]} {_ordinal_to_words(str(d_int), "th")}'
    if 1000 <= y_int <= 9999:
        out += f', {_number_to_words(y_int)}'  # year-style reading
    else:
        out += f', {_int_to_words(y_int)}'
    return out


def _is_year_context(full_text: str, match_start: int) -> bool:
    """True if the ~24 chars before *match_start* end with a year-context word."""
    window = full_text[max(0, match_start - 24):match_start]
    return bool(_RE_YEAR_CONTEXT.search(window))


def _int_to_words(n: int) -> str:
    """Convert a non-negative integer to English words.

    Handles numbers from 0 up to 999 sextillion (10^24 - 1).
    Beyond that, falls back to a comma-grouped digit string, which TTS
    engines read acceptably (previous behavior silently returned 'zero').
    Examples: 0 → 'zero', 7 → 'seven', 42 → 'forty-two',
    1234 → 'one thousand two hundred thirty-four'.
    """
    if n == 0:
        return 'zero'
    if n < 0:
        return 'negative ' + _int_to_words(-n)
    if n > _MAX_INT_WORDS:
        return f"{n:,}"

    parts: List[str] = []
    for i in range(len(_THOUSANDS) - 1, -1, -1):
        group_val, group_name = n // (1000 ** i), _THOUSANDS[i][0]
        if group_val >= 1000:
            group_val = group_val % 1000
        if group_val > 0:
            group_words = _three_digit_group(group_val)
            if group_name:
                group_words += ' ' + group_name
            parts.append(group_words)
        n %= 1000 ** i
        if n == 0:
            break

    return ' '.join(parts) if parts else 'zero'


def _three_digit_group(n: int) -> str:
    """Convert a number 1-999 to English words."""
    result = ''
    if n >= 100:
        result += _ONES[n // 100] + ' hundred'
        n %= 100
        if n > 0:
            result += ' '
    if n >= 20:
        result += _TENS[n // 10]
        n %= 10
        if n > 0:
            result += '-' + _ONES[n]
    elif n > 0:
        result += _ONES[n]
    return result


def _number_to_words(n: int) -> str:
    """Convert integer to words, with year-aware handling for 4-digit numbers.

    Round years of the last millennium read "nineteen hundred" (1100-1999
    ending in 00), and single-digit tails read with "oh" ("eighteen oh
    six" for 1806) — the way years are actually spoken in history and
    date contexts.
    """
    if 1000 <= n <= 9999:
        hi, lo = n // 100, n % 100
        if lo == 0:
            if 1000 < n <= 1999:
                return f'{_int_to_words(hi)} hundred'  # 1900 → "nineteen hundred"
            return _int_to_words(n)  # 1000 → "one thousand", 2000 → "two thousand"
        hi_words = _int_to_words(hi)
        if lo < 10:
            lo_words = 'oh ' + _ONES[lo]  # 1806 → "eighteen oh six"
        else:
            lo_words = _int_to_words(lo)
        return hi_words + ' ' + lo_words
    return _int_to_words(n)


def _decade_to_words(y: int) -> str:
    """Plural year → decade words, or '' when no natural reading exists.

    1990 → 'nineteen nineties', 1900 → 'nineteen hundreds',
    2000 → 'two thousands', 2020 → 'twenty twenties'.
    """
    if not (1000 <= y <= 2099):
        return ''
    hi, lo = y // 100, y % 100
    if lo == 0:
        if y % 1000 == 0:
            return f'{_int_to_words(y // 1000)} thousands'
        return f'{_int_to_words(hi)} hundreds'
    if lo >= 10 and lo % 10 == 0:
        tens = lo // 10
        decade_word = 'tens' if tens == 1 else _TENS[tens].replace('y', 'ies')
        return f'{_int_to_words(hi)} {decade_word}'
    return ''


def _decade_word_from_pair(pair: str) -> str:
    """Two-digit decade ('90') → plural decade word ('nineties'), or ''.

    Only round tens are decades ('80' → 'eighties'); anything else ('84',
    '05') has no natural reading and the caller leaves the token alone.
    """
    dec = int(pair)
    if dec < 10 or dec % 10:
        return ''
    return 'tens' if dec == 10 else _TENS[dec // 10].replace('y', 'ies')


def _caret_exp_words(exp: str) -> str:
    """Exponent → ordinal words ('10' → 'tenth', '0' → 'zeroth')."""
    n = int(exp)
    if n == 0:
        return 'zeroth'
    return _ordinal_to_words(exp, 'th')


def _decimal_to_words(s: str) -> str:
    """Convert a decimal string like '3.14' or '.5' to words.

    '3.14' → 'three point one four'
    '.5' → 'point five'
    '99.9' → 'ninety-nine point nine'
    """
    if '.' in s:
        integer_part, decimal_part = s.split('.', 1)
    else:
        integer_part, decimal_part = s, ''

    words = ''
    if integer_part:
        n = int(integer_part.replace(',', ''))
        words = _int_to_words(n)
    else:
        words = 'zero'

    if decimal_part:
        words += ' point'
        for digit in decimal_part:
            if digit.isdigit():
                # _DIGIT_WORDS, NOT _ONES: _ONES[0] is '' (cardinal
                # composition), so "3.05" was spoken "three point five" —
                # factually wrong (3.05 ≠ 3.5) and "4.0" lost its "zero".
                words += ' ' + _DIGIT_WORDS[int(digit)]
            else:
                words += ' ' + digit
    return words


def _fraction_to_words(numerator: str, denominator: str) -> str:
    """Convert a fraction like '1/2' or '3/4' to words.

    '1/2' → 'one half', '3/4' → 'three quarters',
    '2/3' → 'two thirds', '5/8' → 'five eighths'.
    """
    num = int(numerator)
    den = int(denominator)

    if den == 2:
        if num == 1:
            return 'one half'
        return f'{_int_to_words(num)} halves'

    # Special cases for common denominators
    den_word = _FRACTION_WORDS.get(denominator)
    if den_word:
        num_word = _ONES[num] if num <= 19 else _int_to_words(num)
        # Pluralize: 'quarter' → 'quarters', 'third' → 'thirds'
        if num > 1:
            if den_word.endswith('x'):
                plural = den_word + 'es'
            elif den_word.endswith('f'):
                plural = den_word[:-1] + 'ves'
            else:
                plural = den_word + 's'
        else:
            plural = den_word
        return f'{num_word} {plural}'

    return f'{_int_to_words(num)} divided by {_int_to_words(den)}'


def _ordinal_to_words(num_str: str, suffix: str) -> str:
    """Convert ordinal like '1st', '2nd', '3rd', '4th' to words.

    '1st' → 'first', '2nd' → 'second', '3rd' → 'third',
    '21st' → 'twenty-first', '4th' → 'fourth'.
    """
    n = int(num_str)
    base_words = _int_to_words(n)
    # Look up the last word in the base and convert to ordinal
    last_word = base_words.rsplit(' ', 1)[-1]
    # Handle hyphenated: 'twenty-one' → check 'one'
    if '-' in last_word:
        parts = last_word.split('-')
        last_part = parts[-1]
        if last_part in _ORDINAL_MAP:
            parts[-1] = _ORDINAL_MAP[last_part]
            ordinal_word = '-'.join(parts)
            return base_words.rsplit(' ', 1)[0] + ' ' + ordinal_word if ' ' in base_words else ordinal_word
        return base_words
    else:
        last_part = last_word

    if last_part in _ORDINAL_MAP:
        ordinal_word = _ORDINAL_MAP[last_part]
        if ' ' in base_words:
            return base_words.rsplit(' ', 1)[0] + ' ' + ordinal_word
        return ordinal_word
    return base_words


def _time_to_words(hours: str, minutes: str, ampm: str = '') -> str:
    """Convert time like '5:30 PM' to words.

    '5:30 PM' → 'five thirty PM'
    '5:05' → 'five oh five'
    '12:00 AM' → 'twelve AM'
    """
    h = int(hours)
    m = int(minutes)

    hour_word = _int_to_words(h) if h > 0 else 'twelve'

    if m == 0:
        time_str = hour_word
    elif m < 10:
        time_str = f'{hour_word} oh {_ONES[m]}'
    else:
        time_str = f'{hour_word} {_int_to_words(m)}'

    if ampm:
        ampm_clean = ampm.upper().replace('.', '')
        time_str += ' ' + ampm_clean

    return time_str


def _currency_to_words(amount_str: str, symbol: str, magnitude: str = '') -> str:
    """Convert currency like '$50', '€1,000.50' or '$1.5B' to words.

    '$50' → 'fifty dollars'
    '$1.50' → 'one dollar fifty cents'
    '€1,000' → 'one thousand euros'
    '$5K' → 'five thousand dollars'
    '$1.5B' → 'one point five billion dollars'
    """
    # Clean the amount string
    amount_clean = amount_str.replace(',', '')
    currency_name = {'$': 'dollar', '€': 'euro', '£': 'pound', '¥': 'yen',
                     '₹': 'rupee'}.get(symbol, 'dollar')

    magnitude = (magnitude or '').upper()
    mag_word = {'K': 'thousand', 'M': 'million', 'B': 'billion'}.get(magnitude, '')

    if mag_word and '.' in amount_clean:
        # "$1.5B" → "one point five billion dollars" (trim trailing zeros:
        # "1.50" → "1.5" so it is not spoken "one point five zero")
        trimmed = amount_clean.rstrip('0').rstrip('.')
        return (
            f'{_decimal_to_words(trimmed)} {mag_word} '
            f'{currency_name if symbol == "¥" else currency_name + "s"}'
        )
    if mag_word:
        # "$5K" → 5 × 1,000 → "five thousand dollars"
        int_val = (int(amount_clean) if amount_clean else 0) * {
            'K': 1000, 'M': 1_000_000, 'B': 1_000_000_000,
        }[magnitude]
        int_word = _int_to_words(int_val)
        if symbol == '¥':
            return f'{int_word} yen'
        if int_val == 1:
            return f'{int_word} {currency_name}'
        return f'{int_word} {currency_name}s'

    if '.' in amount_clean:
        integer_part, decimal_part = amount_clean.split('.', 1)
        int_val = int(integer_part) if integer_part else 0
        cents = int(decimal_part[:2].ljust(2, '0')) if decimal_part else 0

        int_word = _int_to_words(int_val)
        # Japanese yen has no plural form
        if symbol == '¥':
            result = f'{int_word} yen'
        elif int_val == 1:
            result = f'{int_word} {currency_name}'
        else:
            result = f'{int_word} {currency_name}s'

        if cents > 0:
            if symbol == '¥':
                result += f' {_int_to_words(cents)} sen'
            else:
                if cents == 1:
                    result += f' {_int_to_words(cents)} cent'
                else:
                    result += f' {_int_to_words(cents)} cents'
        return result
    else:
        int_val = int(amount_clean) if amount_clean else 0
        int_word = _int_to_words(int_val)
        if symbol == '¥':
            return f'{int_word} yen'
        if int_val == 1:
            return f'{int_word} {currency_name}'
        else:
            return f'{int_word} {currency_name}s'


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

@functools.lru_cache(maxsize=4096)
def _is_cjk_char(ch: str) -> bool:
    """Return True if *ch* falls within common CJK Unicode ranges.

    Cached because this is called per-character during duration estimation;
    repeated characters (very common in CJK text) hit the cache instead
    of scanning all six Unicode ranges each time.
    """
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ChunkingConfig:
    """Configuration for text chunking behavior.

    Attributes:
        max_chars: Maximum characters per chunk.
        max_sentences: Maximum sentences per chunk (sentence strategy).
        min_chunk_chars: Minimum characters for a valid chunk.
        overlap_chars: Number of overlapping characters at chunk boundaries.
        respect_sentence_bounds: Whether to avoid splitting mid-sentence.
        language: Target TTS language.
        max_duration_seconds: Maximum estimated speaking duration per chunk.
            When > 0, chunks are also split if their estimated duration
            exceeds this value (used by "duration" and "hybrid" strategies).
        chunk_strategy: Chunking strategy to use. One of:
            "sentence" (default), "duration", "paragraph", "semantic", "hybrid".
        pronunciation_overrides: Optional dict of ``{regex_pattern: spoken_form}``
            applied after standard abbreviation expansion. Allows customizing
            how specific terms are spoken. If None, uses built-in defaults.
        crossfade_enabled: When False, overlap_chars is forced to 0 to prevent
            double-speak artifacts that occur when overlapping text is played
            sequentially without an audio crossfade.
    """
    max_chars: int = 500
    max_sentences: int = 5
    min_chunk_chars: int = 50
    overlap_chars: int = 0
    respect_sentence_bounds: bool = True
    language: str = "English"
    max_duration_seconds: float = 0.0
    chunk_strategy: str = "sentence"
    pronunciation_overrides: Optional[Dict[re.Pattern, str]] = None
    crossfade_enabled: bool = True

    def __post_init__(self) -> None:
        """Validate config values to prevent undefined behavior."""
        if self.max_chars < 1:
            raise ValueError(f"max_chars must be >= 1, got {self.max_chars}")
        if self.max_sentences < 1:
            raise ValueError(f"max_sentences must be >= 1, got {self.max_sentences}")
        if self.min_chunk_chars < 0:
            raise ValueError(f"min_chunk_chars must be >= 0, got {self.min_chunk_chars}")
        if self.min_chunk_chars >= self.max_chars:
            # Auto-clamp instead of raising — a direct ChunkingConfig(...)
            # with a small max_chars used to crash the whole render
            # pipeline for a setting the caller often doesn't care about.
            clamped = max(0, self.max_chars - 1)
            logger.warning(
                f"[Chunker] min_chunk_chars ({self.min_chunk_chars}) >= "
                f"max_chars ({self.max_chars}) — clamped to {clamped}"
            )
            object.__setattr__(self, 'min_chunk_chars', clamped)
        # Validate strategy
        _valid_strategies = {"sentence", "duration", "paragraph", "semantic", "hybrid"}
        if self.chunk_strategy not in _valid_strategies:
            logger.warning(
                f"[Chunker] Unknown chunk_strategy '{self.chunk_strategy}'; "
                f"falling back to 'sentence'. Valid: {_valid_strategies}"
            )
            object.__setattr__(self, 'chunk_strategy', 'sentence')
        # Disable overlap when crossfade is off — overlapping text with
        # a hard concat splice causes audible double-speak at boundaries.
        if not self.crossfade_enabled and self.overlap_chars > 0:
            logger.info(
                f"[Chunker] crossfade_enabled=False — disabling overlap_chars "
                f"({self.overlap_chars}) to prevent double-speak on concat splice"
            )
            object.__setattr__(self, 'overlap_chars', 0)
        # Clamp overlap to a reasonable fraction of max_chars
        max_overlap = self.max_chars // 2
        if self.overlap_chars > max_overlap:
            logger.warning(
                f"[Chunker] overlap_chars ({self.overlap_chars}) clamped to "
                f"{max_overlap} (half of max_chars={self.max_chars})"
            )
            object.__setattr__(self, 'overlap_chars', max_overlap)
        # Normalize pronunciation_overrides: compile raw pattern strings so
        # callers can pass either {"\\bJSON\\b": "jason"} or compiled patterns.
        # Invalid regexes from user settings.json are DROPPED with a warning
        # instead of crashing the render worker (old behavior: re.error).
        if self.pronunciation_overrides is not None:
            normalized: Dict[re.Pattern, str] = {}
            for k, v in self.pronunciation_overrides.items():
                if isinstance(k, str):
                    try:
                        k = re.compile(k, re.IGNORECASE)
                    except re.error as exc:
                        logger.warning(
                            "[Chunker] Dropping invalid pronunciation "
                            f"override {k!r}: {exc}"
                        )
                        continue
                normalized[k] = v
            object.__setattr__(self, 'pronunciation_overrides', normalized)
        # Clamp duration to reasonable range
        if self.max_duration_seconds < 0:
            logger.warning(
                "[Chunker] max_duration_seconds must be >= 0; "
                "setting to 0 (disabled)"
            )
            object.__setattr__(self, 'max_duration_seconds', 0.0)

    @property
    def effective_pronunciation_overrides(self) -> Dict[re.Pattern, str]:
        """Merged pronunciation overrides: built-in defaults + user additions.

        User entries WIN on conflicting patterns.  Previously a user override
        dictionary completely replaced every built-in expansion — adding one
        custom word silently disabled all default pronunciations.
        """
        if self.pronunciation_overrides:
            merged = dict(_DEFAULT_PRONUNCIATION_OVERRIDES)
            merged.update(self.pronunciation_overrides)
            return merged
        return _DEFAULT_PRONUNCIATION_OVERRIDES

    def __repr__(self) -> str:
        return (f"ChunkingConfig(max_chars={self.max_chars}, "
                f"lang={self.language}, strategy={self.chunk_strategy}, "
                f"overlap={self.overlap_chars}, "
                f"max_dur={self.max_duration_seconds}s)")


# ═══════════════════════════════════════════════════════════════════════════════
#  CHUNKER
# ═══════════════════════════════════════════════════════════════════════════════

class TextChunker:
    """Splits long text into TTS-friendly chunks while preserving sentence boundaries.

    Stateless after ``__init__`` — safe to call from multiple threads.

    Features:
    - Abbreviation-aware splitting (won't break on "Dr.", "U.S.", "e.g.")
    - URL-aware splitting (pre-scanned once, O(1) per boundary check)
    - SSML-aware splitting (won't split inside ``<speak>``, ``<break>``, etc.)
    - Paragraph-aware pre-splitting for better chunk boundaries
    - Configurable overlap at chunk boundaries for smoother TTS joins
    - Punctuation-pause-aware duration estimation
    - Mixed-language (CJK + Latin) speech duration estimation
    - CJK patterns include ellipsis and paragraph breaks
    """

    def __init__(self, config: Optional[ChunkingConfig] = None) -> None:
        self.config = config or ChunkingConfig()
        self._sentence_pattern = _SENTENCE_PATTERNS.get(
            self.config.language, _DEFAULT_SENTENCE_PATTERN
        )
        if self.config.language not in _SENTENCE_PATTERNS:
            logger.warning(
                f"[Chunker] Language '{self.config.language}' not in sentence "
                f"patterns; falling back to default. "
                f"Supported: {list(_SENTENCE_PATTERNS.keys())}"
            )

    # ─── Public API ──────────────────────────────────────────────────────────

    def needs_chunking(self, text: str) -> bool:
        """Check if text exceeds the character, sentence, or duration limit."""
        if not text:
            return False
        if len(text) > self.config.max_chars:
            return True
        # Check duration limit before splitting sentences (cheaper)
        if self.config.max_duration_seconds > 0:
            dur = self.estimate_speech_duration(text, self.config.language, is_chunk=False)
            if dur > self.config.max_duration_seconds:
                return True
        sentences = self.split_into_sentences(text, self.config.language)
        return len(sentences) > self.config.max_sentences

    def chunk_text(
        self, text: str, language: Optional[str] = None,
        preprocess: bool = True,
    ) -> List[Tuple[str, float]]:
        """Split text into chunks suitable for TTS processing.

        Returns:
            A list of ``(chunk_text, estimated_duration)`` tuples.
        """
        if language is None:
            language = self.config.language

        if not text or not text.strip():
            return []

        # ── Preprocess text for TTS ──────────────────────────────────────
        if preprocess:
            cleaned_text, pp_stats = preprocess_for_tts(
                text, language,
                _overrides=self.config.effective_pronunciation_overrides,
            )
            if pp_stats.total_changes > 0:
                logger.debug(
                    f"[Chunker] Preprocessing stats: {pp_stats}"
                )
            text = cleaned_text

        text = self._normalize_whitespace(text)

        # Split sentences once, then decide whether to chunk
        sentences = self.split_into_sentences(text, language)

        # Paragraph strategy must honor paragraph markers even for short
        # text (the early return below used to bypass it entirely).
        _has_para_marker = (
            self.config.chunk_strategy == "paragraph"
            and len(sentences) > 1
            and any(s.startswith('\n') for s in sentences[1:])
        )

        if len(text) <= self.config.max_chars and not _has_para_marker:
            if not sentences or len(sentences) <= self.config.max_sentences:
                dur = self.estimate_speech_duration(text, language)
                if (self.config.max_duration_seconds > 0
                        and dur > self.config.max_duration_seconds):
                    pass  # still needs chunking by duration
                else:
                    return [(text, dur)]

        # Text exceeds limits — proceed with chunking
        if not sentences:
            logger.debug(
                f"[Chunker] No sentence boundaries found for {len(text)}-char "
                f"text — falling back to hard split"
            )
            return self._hard_split(text, language)

        # ── Dispatch to the configured strategy ────────────────────────
        strategy = self.config.chunk_strategy
        if strategy == "duration":
            chunks = self._chunk_by_duration(sentences, language)
        elif strategy == "paragraph":
            chunks = self._chunk_by_paragraph(sentences, language)
        elif strategy == "semantic":
            chunks = self._chunk_semantic(sentences, language)
        elif strategy == "hybrid":
            chunks = self._chunk_hybrid(sentences, language)
        else:
            # "sentence" — original default
            chunks = self._chunk_by_sentences(sentences, language)

        # Apply overlap if configured
        if self.config.overlap_chars > 0 and len(chunks) > 1:
            chunks = self._apply_overlap(chunks, language)

        logger.debug(
            f"[Chunker] Split {len(text)}-char text into {len(chunks)} "
            f"chunk(s) (strategy={strategy}, max_chars={self.config.max_chars})"
            + "".join(
                f"\n  [{i+1}] {len(c[0])} chars, ~{c[1]:.1f}s"
                for i, c in enumerate(chunks)
            )
        )
        return chunks

    def split_into_sentences(
        self, text: str, language: Optional[str] = None
    ) -> List[str]:
        """Split text into sentences using language-appropriate delimiters.

        For Latin-script languages, uses abbreviation-aware splitting that
        avoids breaking on "Dr.", "U.S.", "e.g.", etc.  For CJK languages,
        splits on sentence-ending punctuation and paragraph/newline boundaries.
        """
        if language is None:
            language = self.config.language
        if not text or not text.strip():
            return []

        if language in CJK_LANGUAGES:
            return self._split_cjk_sentences(text, language)

        if language in _ABBREVIATION_AWARE_LANGUAGES:
            return self._split_abbreviation_aware(text)

        # Fallback: simple pattern split
        pattern = _SENTENCE_PATTERNS.get(language, self._sentence_pattern)
        return self._simple_pattern_split(text, pattern)

    def estimate_speech_duration(
        self, text: str, language: Optional[str] = None,
        is_chunk: bool = False,
    ) -> float:
        """Estimate audio duration in seconds.

        Mixed-language support: CJK characters and non-CJK words are counted
        separately and estimated with their respective speaking rates, producing
        accurate durations for text like ``"The公司采用new strategy"``.

        Punctuation pauses are added on top of the base speaking duration.

        Args:
            text: The text to estimate duration for.
            language: Target TTS language.
            is_chunk: If True, this is one chunk of a larger text.  The
                fixed TTS engine start/stop overhead (0.3s) is omitted since
                intermediate chunks don't incur engine warm-up latency.
        """
        if language is None:
            language = self.config.language
        non_cjk_rate = SPEAKING_RATES.get(language, DEFAULT_SPEAKING_RATE)

        # ── Classify non-space characters as CJK or non-CJK ──────────────
        cjk_count = 0
        non_cjk_count = 0
        for ch in text:
            if ch.isspace():
                continue
            if _is_cjk_char(ch):
                cjk_count += 1
            else:
                non_cjk_count += 1

        # ── Compute base duration ────────────────────────────────────────
        if cjk_count == 0:
            # Pure non-CJK: word-based
            word_count = len(text.split())
            base_duration = (word_count / non_cjk_rate) * 60
        elif non_cjk_count == 0:
            # Pure CJK: character-based
            base_duration = (cjk_count / _CJK_SPEAKING_RATE) * 60
        else:
            # Mixed: estimate each part with its appropriate rate
            cjk_duration = (cjk_count / _CJK_SPEAKING_RATE) * 60
            # Approximate non-CJK words from character count (~5 chars/word)
            non_cjk_words = non_cjk_count / 5.0
            non_cjk_duration = (non_cjk_words / non_cjk_rate) * 60
            base_duration = cjk_duration + non_cjk_duration

        # Add punctuation pauses (single-pass O(n))
        punct_pause = self._estimate_punctuation_pause(text)

        # Fixed overhead for TTS engine start/stop (only for standalone clips)
        engine_overhead = 0.0 if is_chunk else 0.3
        return max(0.5, base_duration + punct_pause + engine_overhead)

    # ─── Private Helpers ─────────────────────────────────────────────────────

    def _chunk_duration(self, text: str, language: str) -> float:
        """Estimate speech duration for a chunk (no engine overhead).

        Wraps :meth:`estimate_speech_duration` with ``is_chunk=True`` so
        that the fixed 0.3s engine start/stop overhead is excluded.  This
        prevents over-chunking: the overhead is only real for standalone
        clips, not for intermediate segments that will be concatenated.
        """
        return self.estimate_speech_duration(text, language, is_chunk=True)

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """Collapse runs of whitespace (except newlines) into single spaces."""
        text = re.sub(r'[^\S\n]+', ' ', text)
        return text.strip()

    @staticmethod
    def _estimate_punctuation_pause(text: str) -> float:
        """Sum up pause durations for all punctuation marks.

        Single-pass O(n) — iterates the text once with a dict lookup
        per character, instead of calling ``str.count()`` per punctuation
        type (which was O(n × m) with *m* punctuation types).

        Adjacent punctuation like ``!?`` or ``...`` each get their own
        pause, but consecutive identical punctuation (e.g. ``!!`` which
        should have been normalised) is deduplicated to avoid double-counting.
        """
        total = 0.0
        prev_pause_char = ''
        for c in text:
            pause = _PUNCTUATION_PAUSE.get(c, 0.0)
            if pause > 0.0:
                # Deduplicate: if the previous char was the same punctuation,
                # skip it (normalised multi-punct like "!!" → "!" should only
                # count once, but mixed like "!?" each get their pause).
                if c == prev_pause_char:
                    continue
                total += pause
                prev_pause_char = c
            else:
                prev_pause_char = ''
        return total

    # ─── Protected-region pre-scanning ────────────────────────────────────────

    @staticmethod
    def _pre_scan_protected_regions(text: str) -> Set[int]:
        """Pre-scan text for URL and SSML tag positions that must not be split.

        Returns a set of character positions that are inside URLs or SSML
        tags.  This is called **once** per chunk_text invocation and passed
        to :meth:`_is_sentence_boundary`, making each boundary check O(1)
        instead of O(n) per boundary.
        """
        protected: Set[int] = set()

        # Mark URL regions — trim trailing punctuation so a sentence period
        # after a URL ("See https://example.com.") still splits sentences.
        for m in _URL_PATTERN.finditer(text):
            trimmed_len = len(_URL_TRAILING_PUNCT.sub('', m.group(0)))
            protected.update(range(m.start(), m.start() + trimmed_len))

        # Mark SSML tag regions
        for m in _SSML_TAG_PATTERN.finditer(text):
            protected.update(range(m.start(), m.end()))

        return protected

    # ─── Sentence boundary detection ─────────────────────────────────────────

    @staticmethod
    def _is_sentence_boundary(
        text: str,
        punct_start: int,
        punct_end: int,
        protected_positions: Optional[Set[int]] = None,
    ) -> bool:
        """Check if punctuation at ``[punct_start:punct_end]`` is a real
        sentence boundary.

        Returns *False* if the position is inside a protected region
        (URL / SSML tag), the preceding character is a digit, the preceding
        word is a known abbreviation, or the punctuation is part of an
        initialism.

        Args:
            text: The full text being split.
            punct_start: Start index of the punctuation mark(s).
            punct_end: End index (exclusive) of the punctuation mark(s).
            protected_positions: Set of char positions inside URLs / SSML
                tags (output of :meth:`_pre_scan_protected_regions`).
        """
        # ── Protected region check (URL, SSML tag) ───────────────────────
        if protected_positions:
            for p in range(punct_start, punct_end):
                if p in protected_positions:
                    return False

        # Bounded backward scan — O(1) per boundary.  The old
        # ``text[:punct_start].rstrip()`` copied the whole prefix, making
        # sentence splitting O(n²) on large documents.
        lookback_start = max(0, punct_start - 48)
        preceding = text[lookback_start:punct_start].rstrip()
        if not preceding:
            return True

        # ── Decimal numbers: "3.14" — digit before the period AND a digit
        # after it.  A digit followed by space/capital ("In 2024. The next
        # year...") is a real sentence boundary.
        if preceding[-1].isdigit():
            if punct_end < len(text) and text[punct_end].isdigit():
                return False
            return True

        # ── Extract the last word before the punctuation ────────────────
        last_word_match = re.search(r'([A-Za-z]+)$', preceding)
        if not last_word_match:
            return True  # No alphabetic word → real boundary

        last_word = last_word_match.group(1)
        punct_char = text[punct_start] if punct_start < len(text) else ''

        # ── Multi-letter abbreviation: Dr., Mr., Corp., etc. ────────────
        if last_word.lower() in _KNOWN_ABBREVIATIONS:
            return False

        # ── e.g. / i.e. fallback (if pre-scan missed it) ────────────────
        if len(last_word) == 1 and last_word.islower():
            pre_word = preceding[:last_word_match.start()].rstrip()
            if pre_word and pre_word[-1] == '.':
                pre_dot_match = re.search(r'([A-Za-z])\s*$', pre_word[:-1])
                if pre_dot_match and len(pre_dot_match.group(1)) == 1:
                    return False  # Looks like "e.g." or "i.e."

        # ── Single-letter initial: "J. K. Rowling", "U.S.A." ────────────
        if len(last_word) == 1 and last_word.isupper() and punct_char == '.':
            after = text[punct_end:].lstrip()
            if not after:
                return False  # End of text after initial
            # Part of an initialism: "U.S.A."
            if len(after) >= 3 and after[0].isupper() and after[1] == '.':
                return False
            # "J. Smith went home." — uppercase start means real boundary
            if after[0].isupper():
                return True
            return False

        # ── Domain pattern: "example.com", "site.org" ────────────────────
        if re.search(r'\.[A-Za-z]{2,5}$', preceding):
            last_dot_idx = preceding.rfind('.')
            if last_dot_idx >= 0 and ' ' not in preceding[last_dot_idx:]:
                nxt = text[punct_end:punct_end + 1]
                if nxt and not nxt.isspace():
                    return False
                # Period after a domain + whitespace/EOL → real boundary

        return True

    # ─── Prosody-aware boundary detection ────────────────────────────────────

    def _is_clause_boundary(self, sentence: str) -> bool:
        """Check if *sentence* starts with a clause-boundary word.

        These words (e.g. "However,", "Therefore,") signal natural
        breathing points where TTS should start a new chunk.
        """
        return bool(_RE_CLAUSE_START.match(sentence))

    def _is_dialogue_start(self, sentence: str) -> bool:
        """Check if *sentence* starts with quoted dialogue."""
        return bool(_RE_DIALOGUE_OPEN.match(sentence))

    def _is_dialogue_end_with_attribution(self, sentence: str) -> bool:
        """Check if *sentence* ends with dialogue attribution.

        E.g. ``\"I don't know,\" she said.`` — the attribution should
        stay attached to the following quote if possible.
        """
        return bool(_RE_DIALOGUE_ATTRIBUTION.search(sentence))

    def _is_enumeration_start(self, sentence: str) -> bool:
        """Check if *sentence* starts with an ordinal enumeration word."""
        return bool(_RE_ENUMERATION_START.match(sentence))

    def _is_context_dependent_start(self, sentence: str) -> bool:
        """Check if *sentence* starts with a context-dependent pronoun/reference.

        Chunks starting with "This...", "These...", "It..." may lose meaning
        without the preceding context.  The overlap mechanism uses this
        to boost the overlap length.
        """
        return bool(_RE_CONTEXT_DEPENDENT_START.match(sentence))

    def _last_boundary_char(self, text: str) -> str:
        """Return the last punctuation character in *text*, or empty string."""
        for ch in reversed(text.rstrip()):
            if ch in _BOUNDARY_CROSSFADE:
                return ch
        return ''

    # ─── Strategy: chunk by paragraph boundaries ───────────────────────────

    def _chunk_by_paragraph(
        self, sentences: List[str], language: str
    ) -> List[Tuple[str, float]]:
        r"""Group sentences into chunks respecting paragraph breaks.

        Tries to keep each paragraph intact.  If a paragraph exceeds
        *max_chars*, it falls back to sentence-based sub-chunking.

        Paragraph breaks are carried by sentence splitters as a leading
        ``\n`` on the first sentence of each new paragraph — the old
        implementation re-split with ``_PARAGRAPH_PATTERN`` (``\n\s*\n``)
        which could never match that marker, silently merging paragraphs.
        """
        paragraphs: List[str] = []
        current: List[str] = []
        for sentence in sentences:
            stripped = sentence.strip()
            if not stripped:
                continue
            if sentence.startswith('\n') and current:
                paragraphs.append(' '.join(current))
                current = [stripped]
            else:
                current.append(stripped)
        if current:
            paragraphs.append(' '.join(current))

        chunks: List[Tuple[str, float]] = []
        for para in paragraphs:
            if len(para) <= self.config.max_chars:
                dur = self.estimate_speech_duration(para, language)
                if self.config.max_duration_seconds > 0 and dur > self.config.max_duration_seconds:
                    # Paragraph too long by duration — sub-chunk by sentence
                    sub_sentences = self.split_into_sentences(para, language)
                    sub_chunks = self._chunk_by_sentences(sub_sentences, language)
                    chunks.extend(sub_chunks)
                else:
                    chunks.append((para, dur))
            else:
                # Paragraph too long by chars — sub-chunk by sentence
                sub_sentences = self.split_into_sentences(para, language)
                sub_chunks = self._chunk_by_sentences(sub_sentences, language)
                chunks.extend(sub_chunks)

        return chunks

    # ─── Strategy: chunk by estimated speaking duration ────────────────────

    def _chunk_by_duration(
        self, sentences: List[str], language: str
    ) -> List[Tuple[str, float]]:
        """Group sentences into chunks respecting *max_duration_seconds*.

        Falls back to *max_chars* if *max_duration_seconds* is not set.
        """
        max_dur = self.config.max_duration_seconds
        if max_dur <= 0:
            # No duration limit — fall back to sentence-based
            return self._chunk_by_sentences(sentences, language)

        chunks: List[Tuple[str, float]] = []
        current_chunk = ""
        current_duration = 0.0

        for sentence in sentences:
            clean_sentence = sentence.lstrip('\n')
            sent_duration = self._chunk_duration(clean_sentence, language)

            # Incremental duration: add sentence duration to running total.
            # This avoids re-estimating the full combined text each iteration.
            potential_duration = current_duration + sent_duration
            potential_text = (current_chunk + " " + clean_sentence).strip() if current_chunk else clean_sentence

            # Check both duration and hard char limit
            over_duration = potential_duration > max_dur
            over_chars = len(potential_text) > self.config.max_chars

            if over_duration or over_chars:
                if current_chunk:
                    chunks.append((current_chunk.strip(), current_duration))
                if len(clean_sentence) > self.config.max_chars:
                    chunks.extend(self._clause_split(clean_sentence, language, self.config.max_chars))
                    current_chunk = ""
                    current_duration = 0.0
                else:
                    current_chunk = clean_sentence
                    current_duration = sent_duration
            else:
                current_chunk = potential_text
                current_duration = potential_duration

        if current_chunk:
            if len(current_chunk) < self.config.min_chunk_chars and chunks:
                prev_text, _ = chunks[-1]
                combined = (prev_text + " " + current_chunk).strip()
                combined_dur = self._chunk_duration(combined, language)
                fits_chars = len(combined) <= self.config.max_chars * _MERGE_TOLERANCE
                fits_dur = (
                    self.config.max_duration_seconds <= 0
                    or combined_dur <= self.config.max_duration_seconds * 1.1
                )
                if fits_chars and fits_dur:
                    chunks[-1] = (combined, combined_dur)
                else:
                    chunks.append((current_chunk.strip(), current_duration))
            elif len(current_chunk) < self.config.min_chunk_chars and not chunks:
                logger.debug(
                    f"[Chunker] Duration strategy: only chunk is "
                    f"{len(current_chunk)} chars (below min_chunk_chars="
                    f"{self.config.min_chunk_chars}) — no adjacent chunk to merge into"
                )
                chunks.append((current_chunk.strip(), current_duration))
            else:
                chunks.append((current_chunk.strip(), current_duration))

        # Merge any remaining undersized chunks (same treatment as the
        # other strategies — previously skipped for this strategy only).
        self._redistribute_undersized(chunks, language)
        return chunks

    # ─── Strategy: semantic chunking (clause + dialogue + list aware) ──────

    def _chunk_semantic(
        self, sentences: List[str], language: str
    ) -> List[Tuple[str, float]]:
        """Group sentences with awareness of semantic boundaries.

        Prefers splitting at:
        - Clause-boundary words ("However", "Therefore", ...)
        - Dialogue boundaries (before quotes, after attribution)
        - Enumeration starts ("First,", "Second,", ...)
        """
        chunks: List[Tuple[str, float]] = []
        current_chunk = ""
        current_sentences = 0
        effective_max = self.config.max_chars

        for sentence in sentences:
            clean_sentence = sentence.lstrip('\n')
            potential_chunk = (
                (current_chunk + " " + clean_sentence).strip()
                if current_chunk
                else clean_sentence
            )
            over_char_limit = len(potential_chunk) > effective_max
            over_sentence_limit = current_sentences >= self.config.max_sentences

            # Semantic split signal: prefer splitting BEFORE this sentence
            # if it starts with a clause boundary, dialogue, enumeration,
            # or a new paragraph.
            wants_split = (
                self._is_clause_boundary(clean_sentence)
                or (self._is_dialogue_start(clean_sentence) and current_sentences > 0)
                or (self._is_enumeration_start(clean_sentence) and current_sentences > 0)
                or (self._has_paragraph_break(sentence) and current_sentences > 0)
            )

            if (over_char_limit or over_sentence_limit or
                    (wants_split and current_sentences > 0)):
                if current_chunk:
                    chunks.append((
                        current_chunk.strip(),
                        self._chunk_duration(current_chunk, language),
                    ))
                if len(clean_sentence) > effective_max:
                    chunks.extend(self._clause_split(clean_sentence, language, self.config.max_chars))
                    current_chunk = ""
                    current_sentences = 0
                else:
                    current_chunk = clean_sentence
                    current_sentences = 1
            else:
                current_chunk = potential_chunk
                current_sentences += 1

        # Handle the last chunk (same logic as _chunk_by_sentences)
        if current_chunk:
            if len(current_chunk) < self.config.min_chunk_chars and chunks:
                prev_text, _ = chunks[-1]
                combined = (prev_text + " " + current_chunk).strip()
                if len(combined) <= self.config.max_chars * _MERGE_TOLERANCE:
                    chunks[-1] = (combined, self._chunk_duration(combined, language))
                else:
                    chunks.append((current_chunk.strip(), self._chunk_duration(current_chunk, language)))
            else:
                chunks.append((current_chunk.strip(), self._chunk_duration(current_chunk, language)))

        # Balanced-chunk redistribution
        self._redistribute_undersized(chunks, language)

        return chunks

    # ─── Strategy: hybrid (duration + semantic boundaries) ─────────────────

    def _chunk_hybrid(
        self, sentences: List[str], language: str
    ) -> List[Tuple[str, float]]:
        """Combine duration limits with semantic boundary awareness.

        Uses semantic boundary detection for split placement but also
        enforces *max_duration_seconds* and *max_chars*.
        """
        max_dur = self.config.max_duration_seconds
        chunks: List[Tuple[str, float]] = []
        current_chunk = ""
        current_sentences = 0

        for sentence in sentences:
            clean_sentence = sentence.lstrip('\n')
            potential_chunk = (
                (current_chunk + " " + clean_sentence).strip()
                if current_chunk
                else clean_sentence
            )
            potential_duration = self._chunk_duration(potential_chunk, language)

            over_char_limit = len(potential_chunk) > self.config.max_chars
            over_sentence_limit = current_sentences >= self.config.max_sentences
            over_duration = (max_dur > 0 and potential_duration > max_dur)

            # Semantic split signals (includes paragraph boundaries)
            wants_split = (
                self._is_clause_boundary(clean_sentence)
                or (self._is_dialogue_start(clean_sentence) and current_sentences > 0)
                or (self._is_enumeration_start(clean_sentence) and current_sentences > 0)
                or (self._has_paragraph_break(sentence) and current_sentences > 0)
            )

            # Allow semantic splits when approaching (but not necessarily
            # exceeding) limits — previously used identity check `is False`
            # which was both unpythonic and prevented valid splits.
            should_split = (
                over_char_limit
                or over_sentence_limit
                or over_duration
                or (wants_split and current_sentences > 0
                    and not over_duration and not over_char_limit)
            )

            if should_split:
                if current_chunk:
                    chunks.append((
                        current_chunk.strip(),
                        self._chunk_duration(current_chunk, language),
                    ))
                if len(clean_sentence) > self.config.max_chars:
                    chunks.extend(self._clause_split(clean_sentence, language, self.config.max_chars))
                    current_chunk = ""
                    current_sentences = 0
                else:
                    current_chunk = clean_sentence
                    current_sentences = 1
            else:
                current_chunk = potential_chunk
                current_sentences += 1

        # Handle the last chunk
        if current_chunk:
            if len(current_chunk) < self.config.min_chunk_chars and chunks:
                prev_text, _ = chunks[-1]
                combined = (prev_text + " " + current_chunk).strip()
                if len(combined) <= self.config.max_chars * _MERGE_TOLERANCE:
                    chunks[-1] = (combined, self._chunk_duration(combined, language))
                else:
                    chunks.append((current_chunk.strip(), self._chunk_duration(current_chunk, language)))
            else:
                chunks.append((current_chunk.strip(), self._chunk_duration(current_chunk, language)))

        # Balanced-chunk redistribution
        self._redistribute_undersized(chunks, language)

        return chunks

    # ─── Sentence splitting strategies ────────────────────────────────────────

    def _split_abbreviation_aware(self, text: str) -> List[str]:
        """Split text into sentences, respecting abbreviations, URLs, and SSML.

        Strategy:
        1. Pre-scan the full text **once** to mark protected regions
           (URLs, SSML tags, multi-dot abbreviations).
        2. Use the sentence pattern to find candidate split points.
        3. Filter out false positives using the protected-region set and
           abbreviation heuristics.
        4. Strip leading/trailing quotes and parentheses from each sentence.
        """
        pattern = _SENTENCE_PATTERNS.get(
            self.config.language, self._sentence_pattern
        )

        # ── Step 1: Pre-scan all protected positions in one pass ────────
        protected_positions = self._pre_scan_protected_regions(text)

        # Mark multi-dot abbreviation periods as non-boundaries
        for m in _MULTI_DOT_ABBR.finditer(text):
            for ch_pos in range(m.start(), m.end()):
                if text[ch_pos] == '.':
                    protected_positions.add(ch_pos)

        # ── Step 2–3: Find boundaries and extract sentences ─────────────
        sentences: List[str] = []
        last_end = 0

        for match in pattern.finditer(text):
            punct_start = match.start()
            punct_end = match.end()

            if not self._is_sentence_boundary(
                text, punct_start, punct_end, protected_positions
            ):
                continue

            raw = text[last_end:punct_end]
            # Detect paragraph break (double newline) between sentences
            has_para = '\n\n' in raw or (last_end > 0 and text[last_end - 1] == '\n')
            sentence = raw.strip()
            if sentence:
                # Step 4: Strip surrounding quotes/brackets/parens
                sentence = sentence.strip(_SENTENCE_EDGE_CHARS)
                if sentence:
                    # Preserve paragraph-break signal as a leading \n
                    if has_para and sentences:
                        sentence = '\n' + sentence
                    sentences.append(sentence)
            last_end = punct_end

        # Append remaining text after last punctuation
        remaining = text[last_end:].strip()
        if remaining:
            remaining = remaining.strip(_SENTENCE_EDGE_CHARS)
            if remaining:
                sentences.append(remaining)

        # If no sentences found (text with no real punctuation), return whole
        if not sentences and text.strip():
            sentences = [text.strip()]

        return sentences

    def _split_cjk_sentences(self, text: str, language: str) -> List[str]:
        """Split CJK text into sentences.

        Handles:
        - Sentence-ending punctuation (。！？…)
        - Paragraph/newline boundaries as additional split points
        - Consecutive punctuation marks grouped together
        """
        pattern = _SENTENCE_PATTERNS.get(language, _SENTENCE_PATTERNS["Chinese"])

        # First, split by paragraph breaks to get segments
        paragraphs = _PARAGRAPH_PATTERN.split(text)
        sentences: List[str] = []

        for pi, para in enumerate(paragraphs):
            para = para.strip()
            if not para:
                continue

            para_sentences = self._simple_pattern_split(para, pattern)
            if para_sentences:
                # Mark the first sentence of each subsequent paragraph
                # with a leading \n so chunkers can detect paragraph breaks.
                if pi > 0 and sentences:
                    para_sentences[0] = '\n' + para_sentences[0]
                sentences.extend(para_sentences)
            else:
                # No sentence punctuation found — keep the paragraph as-is
                if pi > 0 and sentences:
                    sentences.append('\n' + para)
                else:
                    sentences.append(para)

        return sentences

    def _simple_pattern_split(
        self, text: str, pattern: re.Pattern
    ) -> List[str]:
        """Basic regex split on punctuation pattern."""
        parts = pattern.split(text)
        sentences: List[str] = []

        for i in range(0, len(parts) - 1, 2):
            sentence = parts[i].strip() + parts[i + 1]
            if sentence.strip():
                sentences.append(sentence.strip())

        if len(parts) % 2 == 1 and parts[-1].strip():
            sentences.append(parts[-1].strip())

        return sentences if sentences else (
            [text.strip()] if text.strip() else []
        )

    # ─── Chunking strategies ─────────────────────────────────────────────────

    @staticmethod
    def _has_paragraph_break(sentence: str) -> bool:
        """Check if *sentence* starts with a paragraph break (leading newlines).

        After ``_normalize_whitespace``, paragraph breaks survive as leading
        ``\\n`` characters on the sentence that opens a new paragraph.
        """
        return sentence.startswith('\n')

    def _chunk_by_sentences(
        self, sentences: List[str], language: str
    ) -> List[Tuple[str, float]]:
        """Group sentences into chunks respecting *max_chars* and *max_sentences*.

        Paragraph-aware: prefers splitting before a sentence that starts a
        new paragraph (detected via leading ``\\n``), even if the char/sentence
        limits haven't been reached yet — provided the current chunk already
        has at least one sentence.
        """
        chunks: List[Tuple[str, float]] = []
        current_chunk = ""
        current_sentences = 0
        effective_max = self.config.max_chars

        for sentence in sentences:
            # Strip the paragraph-break marker (\n) for length checks
            # and chunk building — it's only a signal for split decisions.
            clean_sentence = sentence.lstrip('\n')
            potential_chunk = (
                (current_chunk + " " + clean_sentence).strip()
                if current_chunk
                else clean_sentence
            )

            over_char_limit = len(potential_chunk) > effective_max
            over_sentence_limit = current_sentences >= self.config.max_sentences

            # Dialogue attribution anti-split: if the current chunk ends
            # with dialogue attribution (e.g., "...," she said.), prefer
            # keeping the next sentence (likely the quote continuation)
            # in the same chunk, unless we're already over limits.
            _current_ends_with_attribution = (
                current_sentences > 0
                and self._is_dialogue_end_with_attribution(current_chunk)
            )
            _next_is_quote = self._is_dialogue_start(clean_sentence)

            # Paragraph-boundary split: if this sentence opens a new
            # paragraph and we already have content, prefer splitting here
            # to keep paragraphs together within chunks.
            wants_para_split = (
                current_sentences > 0
                and self._has_paragraph_break(sentence)
                and not over_char_limit
                and not _current_ends_with_attribution
            )

            # Dialogue continuity: suppress paragraph/soft splits when
            # the current chunk ends with attribution and next is a quote.
            if (_current_ends_with_attribution and _next_is_quote
                    and not over_char_limit and not over_sentence_limit):
                wants_para_split = False

            if over_char_limit or over_sentence_limit or wants_para_split:
                if current_chunk:
                    chunks.append((
                        current_chunk.strip(),
                        self._chunk_duration(current_chunk, language),
                    ))
                if len(clean_sentence) > effective_max:
                    logger.debug(
                        f"[Chunker] Single sentence ({len(clean_sentence)} chars) "
                        f"exceeds max_chars ({effective_max}) — hard split"
                    )
                    chunks.extend(self._clause_split(clean_sentence, language, self.config.max_chars))
                    current_chunk = ""
                    current_sentences = 0
                else:
                    current_chunk = clean_sentence
                    current_sentences = 1
            else:
                current_chunk = potential_chunk
                current_sentences += 1

        # Handle the last chunk
        if current_chunk:
            if len(current_chunk) < self.config.min_chunk_chars and chunks:
                prev_text, _ = chunks[-1]
                combined = (prev_text + " " + current_chunk).strip()
                if len(combined) <= self.config.max_chars * _MERGE_TOLERANCE:
                    chunks[-1] = (
                        combined,
                        self._chunk_duration(combined, language),
                    )
                else:
                    chunks.append((
                        current_chunk.strip(),
                        self._chunk_duration(current_chunk, language),
                    ))
            else:
                chunks.append((
                    current_chunk.strip(),
                    self._chunk_duration(current_chunk, language),
                ))

        # ── Balanced-chunk redistribution ────────────────────────────────
        # Scan ALL chunks (not just the last) for undersized ones and
        # merge into adjacent chunks when possible.  This handles cases
        # where a very long sentence forces an early split, leaving the
        # next chunk undersized.
        self._redistribute_undersized(chunks, language)

        return chunks

    def _redistribute_undersized(
        self, chunks: List[Tuple[str, float]], language: str
    ) -> None:
        """Merge undersized chunks into neighbors to reduce chunk count.

        Modifies *chunks* in place.  Scans all chunks and merges any
        that are less than 40% of the average length into the adjacent
        chunk (prefer next, fallback to previous) if the combined text
        fits within the merge tolerance.
        """
        if len(chunks) < 2:
            return
        max_merged = self.config.max_chars * _MERGE_TOLERANCE
        # Cache total char count to avoid O(n) recalc per iteration
        # (original code called sum(len(c[0]) for c in chunks) inside
        # the loop, making the overall complexity O(n²)).
        _total_chars = sum(len(c[0]) for c in chunks)
        i = 0
        while i < len(chunks):
            text_i = chunks[i][0]
            avg_len = _total_chars / len(chunks)
            if i == len(chunks) - 1:
                # Last chunk: try merging into previous
                if i > 0:
                    if len(text_i) < avg_len * 0.4:
                        prev_text = chunks[i - 1][0]
                        combined = (prev_text + " " + text_i).strip()
                        if len(combined) <= max_merged:
                            _total_chars = _total_chars - len(chunks[i - 1][0]) - len(text_i) + len(combined)
                            chunks[i - 1] = (
                                combined,
                                self._chunk_duration(combined, language),
                            )
                            chunks.pop(i)
                            continue  # re-check same index
                break
            # Not the last chunk: try merging into the NEXT chunk
            if len(text_i) < avg_len * 0.4:
                next_text = chunks[i + 1][0]
                combined = (text_i + " " + next_text).strip()
                if len(combined) <= max_merged:
                    logger.debug(
                        f"[Chunker] Merging undersized chunk {i} "
                        f"({len(text_i)} chars) into chunk {i + 1}"
                    )
                    _total_chars = _total_chars - len(text_i) - len(next_text) + len(combined)
                    chunks[i + 1] = (
                        combined,
                        self._chunk_duration(combined, language),
                    )
                    chunks.pop(i)
                    continue  # re-check same index (next shifted up)
            i += 1

    def _apply_overlap(
        self, chunks: List[Tuple[str, float]], language: str
    ) -> List[Tuple[str, float]]:
        """Add overlapping text at chunk boundaries for smoother TTS joins.

        Each chunk (except the first) is prepended with the last *N*
        characters from the previous chunk.  The overlap is cut at the
        **last** space within the overlap region to preserve maximum
        context while avoiding mid-word splits.  Falls back to the last
        comma or semicolon if no space is found.

        Improvements over the basic version:
        - **Context-dependent boost**: If the next chunk starts with a
          pronoun or reference word ("This...", "It..."), the overlap
          length is increased to carry more context.
        - **Boundary-aware crossfade**: The extra duration added to
          the chunk depends on the last punctuation of the previous
          chunk (period → longer pause, comma → shorter pause).
        """
        base_overlap = self.config.overlap_chars
        if base_overlap <= 0 or len(chunks) <= 1:
            return chunks

        result: List[Tuple[str, float]] = []
        for i, (text, duration) in enumerate(chunks):
            if i == 0:
                result.append((text, duration))
                continue

            prev_text = chunks[i - 1][0]

            # ── Context-dependent overlap boost ───────────────────────
            # If this chunk starts with a context-dependent pronoun,
            # increase overlap to carry more context from the previous chunk.
            effective_overlap = base_overlap
            if self._is_context_dependent_start(text):
                # Boost by 50% (capped at max_chars // 3)
                boosted = int(base_overlap * 1.5)
                max_overlap = self.config.max_chars // 3
                effective_overlap = min(boosted, max_overlap)
                logger.debug(
                    f"[Chunker] Chunk {i}: context-dependent start detected, "
                    f"overlap boosted {base_overlap} → {effective_overlap} chars"
                )

            overlap_text = prev_text[-effective_overlap:]

            # Cut at the FIRST space so the overlap starts at a word
            # boundary while KEEPING its head context (~overlap chars).
            # The old rfind(' ') kept only the final 1-2 words, making the
            # configured overlap length effectively useless.
            clean_break = overlap_text.find(' ')
            if clean_break > 0:
                overlap_text = overlap_text[clean_break + 1:]
            else:
                # No space found — try first comma or semicolon
                clean_break = min(
                    (p for p in (overlap_text.find(','), overlap_text.find(';')) if p >= 0),
                    default=-1,
                )
                if clean_break > 0:
                    overlap_text = overlap_text[clean_break + 1:].lstrip()

            if not overlap_text:
                result.append((text, duration))
                continue

            # ── Guard: don't let overlap push chunk over max_chars ──
            merged = overlap_text + " " + text
            if len(merged) > self.config.max_chars * 1.15:
                # Trim overlap to fit; find a shorter break point
                shorter_overlap = prev_text[-(base_overlap // 2):]
                clean_break = shorter_overlap.find(' ')
                if clean_break > 0:
                    shorter_overlap = shorter_overlap[clean_break + 1:]
                if shorter_overlap:
                    merged = shorter_overlap + " " + text
                else:
                    merged = text  # abandon overlap entirely

            # ── Boundary-aware crossfade duration ─────────────────────
            # The crossfade pause depends on the last punctuation of the
            # *previous* chunk's original text (before overlap trimming).
            boundary_char = self._last_boundary_char(prev_text)
            crossfade_extra = _BOUNDARY_CROSSFADE.get(
                boundary_char, _DEFAULT_CROSSFADE
            )
            # Add the overlap text's estimated speaking duration so the
            # total accounts for the extra words that will be spoken.
            overlap_spoken_dur = self._chunk_duration(overlap_text, language)
            adjusted_duration = duration + crossfade_extra + overlap_spoken_dur

            logger.debug(
                f"[Chunker] Chunk {i}: added {len(overlap_text)}-char overlap "
                f"(boundary='{boundary_char}', crossfade={crossfade_extra:.2f}s)"
            )
            result.append((merged, adjusted_duration))

        return result

    def _clause_split(
        self, text: str, language: str, max_len: int
    ) -> List[Tuple[str, float]]:
        """Try splitting a long sentence at clause boundaries first.

        Looks for the last comma, semicolon, colon, or dash within
        *max_len* characters.  Falls back to :meth:`_hard_split` if
        no clause boundary is found or if text is already short enough.
        """
        if len(text) <= max_len:
            return [(text, self._chunk_duration(text, language))]

        search = text[:max_len]
        # Find the last clause boundary
        best_pos = -1
        for ch in (';', ',', ':', '—', '–'):
            pos = search.rfind(ch)
            if pos > best_pos:
                best_pos = pos

        if best_pos > self.config.min_chunk_chars:
            part1 = text[:best_pos].strip()
            part2 = text[best_pos + 1:].strip()
            if part2:
                result = [(part1, self._chunk_duration(part1, language))]
                # Recurse for the remainder (it might still be too long)
                result.extend(self._clause_split(part2, language, max_len))
                return result

        # No suitable clause boundary — fall back to hard split
        return self._hard_split(text, language)

    def _hard_split(
        self, text: str, language: Optional[str] = None
    ) -> List[Tuple[str, float]]:
        """Fallback splitter for text with no sentence boundaries.

        Break priority: paragraph/newline → pause markers → spaces → forced.
        """
        if language is None:
            language = self.config.language
        chunks: List[Tuple[str, float]] = []
        remaining = text

        # Safety guard: ensure max_chars > 0 to prevent infinite loop
        effective_max = max(1, self.config.max_chars)

        while len(remaining) > effective_max:
            search_region = remaining[:effective_max]

            # Priority 1: Break at paragraph/newline boundaries
            newline_pos = search_region.rfind('\n')
            if newline_pos > self.config.min_chunk_chars:
                break_point = newline_pos
                chunk = remaining[:break_point].strip()
            else:
                # Priority 2: Break at pause markers (commas, semicolons, etc.)
                # Only pauses that END after min_chunk_chars — otherwise tiny
                # leading fragments ("a," at char 2) become their own chunk.
                matches = [
                    m for m in _PAUSE_PATTERN.finditer(search_region)
                    if m.end() > self.config.min_chunk_chars
                ]
                if matches:
                    break_point = matches[-1].end()
                    chunk = remaining[:break_point].strip()
                else:
                    # Priority 3: Break at last space
                    last_space = search_region.rfind(' ')
                    if last_space > self.config.min_chunk_chars:
                        break_point = last_space
                        chunk = remaining[:break_point].strip()
                    else:
                        # Priority 4: Forced mid-word split (last resort)
                        break_point = effective_max
                        chunk = remaining[:break_point].strip()
                        logger.warning(
                            f"[Chunker] Forced hard split in middle of "
                            f"word/phrase at char {break_point} (no sentence "
                            f"boundary, comma, or space found)"
                        )

            if not chunk:
                # Prevent infinite loop if break_point is 0
                break_point = max(break_point, 1)
                chunk = remaining[:break_point].strip()

            chunks.append((
                chunk, self._chunk_duration(chunk, language)
            ))
            remaining = remaining[break_point:].strip()

        if remaining:
            chunks.append((
                remaining, self._chunk_duration(remaining, language)
            ))

        return chunks


# ═══════════════════════════════════════════════════════════════════════════════
#  LANGUAGE SETS
# ═══════════════════════════════════════════════════════════════════════════════

# Languages that benefit from abbreviation-aware sentence splitting
_ABBREVIATION_AWARE_LANGUAGES: FrozenSet[str] = frozenset({
    "English", "German", "French", "Russian", "Portuguese", "Spanish", "Italian",
})


# ═══════════════════════════════════════════════════════════════════════════════
#  TTS PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PreprocessStats:
    """Statistics from :func:`preprocess_for_tts`."""
    html_entities_decoded: int = 0
    html_tags_stripped: int = 0
    html_comments_stripped: int = 0
    markdown_cleaned: int = 0
    list_markers_removed: int = 0
    unicode_normalized: int = 0
    emojis_removed: int = 0
    urls_replaced: int = 0
    emails_replaced: int = 0
    abbreviations_expanded: int = 0
    symbols_normalized: int = 0
    numbers_converted: int = 0
    punctuation_normalized: int = 0
    slashes_normalized: int = 0
    parentheticals_normalized: int = 0
    hyphens_normalized: int = 0

    @property
    def total_changes(self) -> int:
        return (
            self.html_entities_decoded
            + self.html_tags_stripped
            + self.html_comments_stripped
            + self.markdown_cleaned
            + self.list_markers_removed
            + self.unicode_normalized
            + self.emojis_removed
            + self.urls_replaced
            + self.emails_replaced
            + self.abbreviations_expanded
            + self.symbols_normalized
            + self.numbers_converted
            + self.punctuation_normalized
            + self.slashes_normalized
            + self.parentheticals_normalized
            + self.hyphens_normalized
        )

    def __repr__(self) -> str:
        parts = []
        if self.html_entities_decoded:
            parts.append(f"html_entities={self.html_entities_decoded}")
        if self.html_tags_stripped:
            parts.append(f"html_tags={self.html_tags_stripped}")
        if self.html_comments_stripped:
            parts.append(f"html_comments={self.html_comments_stripped}")
        if self.markdown_cleaned:
            parts.append(f"markdown={self.markdown_cleaned}")
        if self.list_markers_removed:
            parts.append(f"list_markers={self.list_markers_removed}")
        if self.unicode_normalized:
            parts.append(f"unicode={self.unicode_normalized}")
        if self.emojis_removed:
            parts.append(f"emojis={self.emojis_removed}")
        if self.urls_replaced:
            parts.append(f"urls={self.urls_replaced}")
        if self.emails_replaced:
            parts.append(f"emails={self.emails_replaced}")
        if self.abbreviations_expanded:
            parts.append(f"abbreviations={self.abbreviations_expanded}")
        if self.symbols_normalized:
            parts.append(f"symbols={self.symbols_normalized}")
        if self.numbers_converted:
            parts.append(f"numbers={self.numbers_converted}")
        if self.punctuation_normalized:
            parts.append(f"punctuation={self.punctuation_normalized}")
        if self.slashes_normalized:
            parts.append(f"slashes={self.slashes_normalized}")
        if self.parentheticals_normalized:
            parts.append(f"parentheticals={self.parentheticals_normalized}")
        if self.hyphens_normalized:
            parts.append(f"hyphens={self.hyphens_normalized}")
        inner = ", ".join(parts) if parts else "none"
        return f"PreprocessStats({inner})"


def preprocess_for_tts(
    text: str, language: str = "English",
    _overrides: Optional[Dict[re.Pattern, str]] = None,
) -> Tuple[str, PreprocessStats]:
    """Clean and normalize text for TTS synthesis.

    Performs a pipeline of transformations to make text more suitable for
    speech synthesis:
    - Decodes HTML entities (``&amp;`` → ``&``, etc.)
    - Strips HTML tags and comments
    - Cleans markdown formatting (bold, italic, code, links, headings)
    - Removes bullet/numbered list markers
    - Normalizes unicode (smart quotes, dashes, non-breaking spaces, zero-width chars)
    - Removes emojis
    - Replaces URLs with ``"link to [domain]"``
    - Replaces emails with ``"user at host dot com"`` (Latin only)
    - Expands common abbreviations for speech (Latin only)
    - Normalizes symbols: ``&``, ``/``, ``@``, ``=``, ``%``, ``°``, ``×``
      (Latin only)
    - Converts numbers to words: integers, decimals, currency, ordinals,
      years, fractions, time expressions (Latin only)
    - Normalizes multiple punctuation (``!!!`` → ``!``)
    - Converts word/word slashes to ``word or word`` (Latin only)
    - Converts parentheticals to comma-delimited asides (Latin only)
    - Normalizes hyphenated compounds to spaces (Latin only)
    - Collapses whitespace

    Pronunciation-focused expansions (Latin only):
    - Tech tokens: ``C++`` → "C plus plus", ``C#`` → "C sharp",
      ``Node.js`` → "node jay ess", ``.NET`` → "dot net"
    - Measurements digit-anchored: ``5 kg`` → "five kilograms",
      ``10 MB`` → "ten megabytes", ``2.4 GHz`` → "two point four gigahertz",
      ``5 min`` → "five minutes", ``200 ms`` → "two hundred milliseconds"
    - Compound rates with/without digits: ``km/h`` → "kilometers per hour",
      ``80 MB/s`` → "eighty megabytes per second",
      ``5 m/s²`` → "five meters per second squared"
    - Degrees: ``20°C`` → "twenty degrees Celsius", ``36.6℃`` likewise
    - Magnitude suffixes: ``10K`` → "ten thousand", ``$1.5B`` → "one point
      five billion dollars"
    - Roman numerals in context: ``Chapter IV``, ``World War II``,
      ``Henry VIII``, ``the XIX century``
    - Unicode fractions/superscripts/subscripts: ``½`` → "one half",
      ``m²`` → "square meters", ``E=mc²`` → "E=mc squared",
      ``10⁶`` → "ten to the sixth", ``xⁿ`` → "x to the n",
      ``CO₂`` → "CO two"
    - Caret exponents: ``2^10`` → "two to the tenth", ``x^2`` → "x squared"
    - Math glyphs: ``√2`` → "square root of two", ``π`` → "pi",
      ``ΔT`` → "delta T", ``50 Ω`` → "fifty ohms", ``∫`` → "integral of"
    - Operators: ``a == b`` → "is equal to", ``a !== b`` → "is not equal
      to", ``a && b`` → "a and b", ``a || b`` → "a or b"
    - Idioms: ``24/7`` → "twenty-four seven", ``10-20`` → "ten to twenty",
      ``Jan 15`` → "January 15", ``w/`` → "with", ``w/o`` → "without"
    - Decades: ``1990s`` → "nineteen nineties", ``the 90s`` → "the
      nineties", ``1900s`` → "nineteen hundreds"
    - Dates: ``January 15`` → "January fifteenth", ``9/11`` → "September
      eleventh", ``1920×1080`` → "nineteen twenty by ten eighty"
    - Case-sensitive tier: ``US`` → "U S" (lowercase "us" untouched),
      ``LaTeX`` → "lah tek", ``Nginx`` → "engine x", ``GIF`` → "jiff",
      ``IEEE`` → "I triple E", ``Ctrl`` → "control"
    - Numeronyms: ``K8s`` → "kubernetes", ``i18n`` → "internationalization",
      ``a11y`` → "accessibility", ``e2e`` → "end to end"
    - IPv4 addresses digit-by-digit: ``192.168.1.1`` → "one nine two dot
      one six eight dot one dot one", ``10.0.0.0/8`` → "... slash 8"
      (dotted version strings like ``v2.5.1.2`` are excluded)
    - Phone numbers digit-by-digit: ``555-123-4567`` → "five five five,
      one two three, four five six seven"
    - ISO dates: ``2024-01-15`` → "January fifteenth, twenty twenty-four"
    - Bare unit durations: ``45s`` → "45 seconds", ``24h`` → "24 hours",
      ``2w`` → "2 weeks" (``10s of thousands`` idiom and ``3D`` untouched)
    - Dotted ports: ``port 8080`` → "port eight zero eight zero"
    - Fiscal quarters: ``Q1``–``Q4`` → "quarter 1"–"quarter 4"
    - Brand words: lowercase ``nginx`` → "engine x", ``PyTorch`` → "pie
      torch", ``PySide6`` → "pie side six", ``Qt`` → "Q T",
      ``GraphQL`` → "graph Q L", ``VoIP`` → "V O I P", ``I2C`` → "I two C",
      ``PCIe`` → "P C I express"
    - ``CI/CD`` → "C I C D"; ``and/or`` → "and or" (no duplicated "or")
    - Common respellings: ``queue`` → "cue", ``epitome`` → "eh pit oh
      mee", ``hyperbole`` → "hy per bo lee"; ``a.m.``/``p.m.`` → "A M"/"P M";
      ``deg C`` → "degrees Celsius"; ``2FA`` → "two F A"
    - Rupee: ``₹1,299`` → "one thousand two hundred ninety-nine rupees";
      paragraph sign ``¶`` read as "paragraph"; ``3*4``/``4×`` → "3 times
      4"/"4 times"
    - Decimal zeros: ``3.05`` → "three point zero five" (was "three point
      five"), ``4.0`` → "four point zero" (was "four point")
    - Identifiers: ``my_var_name`` → "my var name",
      ``getUserData`` → "get User Data" (brand names preserved)

    Pronunciation pass 4:
    - Ratios: ``16:9`` → "16 to 9", ``3:1`` → "3 to 1"; timestamps:
      ``12:34:56`` → "twelve hours, thirty-four minutes, fifty-six
      seconds"
    - Keyboard shortcuts: ``Ctrl+S`` → "control plus S",
      ``Ctrl+Alt+Del`` → "control plus Alt plus Del"
    - Rate shorthand: ``100 ops/sec`` → "one hundred ops per second",
      ``5 msg/min`` → "5 msg per minute"
    - Negative temperatures: ``-40°C`` → "minus 40 degrees Celsius"
    - Idioms: ``50-50`` → "fifty fifty", ``the '90s`` → "the nineties",
      ``mid-90s`` → "mid-nineties", ``1990's style`` → "nineteen
      nineties style"
    - Glyph comparisons: ``≥``/``≤`` spoken as "is greater/less than or
      equal to"; ``+/-`` → "plus or minus"; ``y/o`` → "years old"
    - Decimal multipliers: ``1.5x`` → "1.5 times"; respellings:
      ``segue`` → "seg way"; guillemets removed

    Math pronunciation pass:
    - Tight operators: ``3+4`` → "3 plus 4", ``x=5`` → "x equals 5",
      ``a^2+b^2=c^2`` → "a squared plus b squared equals c squared"
    - Negative numbers: ``-5`` → "minus 5" (ranges ``2020-2024``, phones
      and hyphenated tokens like ``GPT-4`` keep their dash)
    - Scientific notation: ``1e9`` → "one times ten to the ninth",
      ``2.5e-3`` → "two point five times ten to the minus third"
    - Python exponents: ``2**10`` → "two to the tenth"
    - Primes: ``f'(x)`` → "f prime of x", ``x' = 2`` → "x prime equals
      2" (quotes ``the 'best'.`` and possessives ``it's`` untouched)
    - Derivatives: ``dy/dx`` → "d y by d x", ``d²y/dx²`` → "second
      derivative of y with respect to x", ``∂f/∂x`` → "partial f
      partial x"
    - Function calls: ``sin(x)`` → "sine of x", ``sqrt(25)`` → "square
      root of 25", ``log_2(8)`` → "log base 2 of 8", ``log₂`` → "log
      base two"; bare ``ln`` → "natural log", ``sin θ`` → "sine θ";
      ``sec(x)`` → "secant of x", ``det(A)`` → "determinant of A";
      ``gcd(a, b)`` → "G C D of a, b"
    - Inverse trig: ``sin⁻¹(x)`` / ``sin^-1(x)`` → "inverse sine of x"
      (previously "sin to the minus first x")
    - Factorial: ``n! = 120`` → "n factorial = 120" (sentence-final
      exclamations and ``!=`` are never touched)
    - Assignment: ``f(x) := x²`` → "f of x is defined as x squared"
    - Modulo: ``5 mod 3`` → "5 modulo 3" (was "5 module 3")
    - Greek alphabet: ``α`` → "alpha", ``λ`` → "lambda", ``μ`` → "mu",
      ``Σ`` → "sigma"; digit-anchored ``5 μm`` → "5 micrometers"
    - Logic glyphs: ``¬`` → "not", ``∧`` → "and", ``∨`` → "or",
      ``⇔`` → "if and only if", ``↦`` → "maps to", ``≅`` → "is
      congruent to", ``∜`` → "fourth root of", ``∮`` → "contour
      integral of"
    - Set/logic glyphs: ``∑`` → "sum of", ``∏`` → "product of",
      ``∂`` → "partial", ``∈`` → "in", ``∀`` → "for all",
      ``∃`` → "there exists", ``∪`` → "union", ``∩`` → "intersection",
      ``⊂`` → "is a subset of", ``∅`` → "the empty set",
      ``∝`` → "is proportional to", ``≡`` → "is equivalent to",
      ``∴`` → "therefore"
    - Variable division: ``x/2`` → "x over 2", ``2/x`` → "2 over x"
      (``PS/2``, ``HTML/5``, ``80/kg`` protected)
    - Math shorthand: ``iff`` → "if and only if", ``w.r.t.``/``wrt`` →
      "with respect to", ``WLOG`` → "without loss of generality",
      ``lhs``/``rhs`` → "left/right hand side", ``gcd``/``lcm`` →
      "G C D"/"L C M", ``thm.`` → "theorem", ``resp.`` → "respectively",
      ``PDE`` → "P D E"
    - Mathematician names: ``Euler`` → "oiler", ``Euclid`` → "yoo klid",
      ``Fourier`` → "foor ee ay", ``Poisson`` → "pwah sohn",
      ``eigenvalue`` → "eye gen value", ``Laplace`` → "luh ploss",
      ``Descartes`` → "day cart", ``Cauchy`` → "koh shee",
      ``Galois`` → "gal wah", ``Leibniz`` → "lype nits",
      ``Schrödinger`` → "shroh ding er", ``de Broglie`` → "duh broh glee",
      ``L'Hôpital`` → "loh pee tal", ``Lebesgue`` → "luh beg",
      ``Banach`` → "bah nahk", ``Noether`` → "nur ter",
      ``Lagrange`` → "luh grahnj", ``Fermat`` → "fur mah",
      ``Erdős`` → "air dish", ``Dijkstra`` → "dik struh", ...
    - Word respellings: ``awry`` → "uh rye", ``rendezvous`` → "ron day
      voo", ``quay`` → "kee", ``indict`` → "in dite", ``voilà`` →
      "vwah lah"

    For CJK languages, only HTML/whitespace/emoji/punctuation cleanup is
    performed (abbreviation, symbol, and number expansion are skipped).

    Args:
        text: The raw text to preprocess.
        language: The target TTS language.

    Returns:
        A tuple of ``(cleaned_text, PreprocessStats)``.
    """
    stats = PreprocessStats()
    is_cjk = language in CJK_LANGUAGES

    # ── 1. Decode HTML entities ──────────────────────────────────────────
    if '&' in text:
        try:
            decoded = _html_mod.unescape(text)
            if decoded != text:
                stats.html_entities_decoded += 1
                text = decoded
        except Exception:
            pass

    # ── 2. Strip HTML comments ───────────────────────────────────────────
    if '<!--' in text:
        new_text, n = _RE_HTML_COMMENT.subn('', text)
        if n:
            stats.html_comments_stripped += n
            text = new_text

    # ── 3. Strip HTML tags ───────────────────────────────────────────────
    if '<' in text:
        new_text, n = _RE_HTML_TAG.subn('', text)
        if n:
            stats.html_tags_stripped += n
            text = new_text

    # ── 4. Clean markdown ───────────────────────────────────────────────
    md_changes = 0
    if '![' in text:
        new_text, n = _RE_MD_IMAGE.subn(r'\1', text)
        if n:
            md_changes += n
            text = new_text
    if '](' in text:
        new_text, n = _RE_MD_LINK.subn(r'\1', text)
        if n:
            md_changes += n
            text = new_text
    if '**' in text or '__' in text:
        new_text, n = _RE_MD_BOLD.subn(r'\1\2', text)
        if n:
            md_changes += n
            text = new_text
    if '*' in text or '_' in text:
        new_text, n = _RE_MD_ITALIC.subn(r'\1\2', text)
        if n:
            md_changes += n
            text = new_text
    if '`' in text:
        new_text, n = _RE_MD_CODE.subn(r'\1', text)
        if n:
            md_changes += n
            text = new_text
    if '#' in text:
        # Add a period after heading text to create a sentence boundary.
        # Without this, "## Results" merges into the previous paragraph.
        new_text, n = _RE_MD_HEADING.subn(r'\2.', text)
        if n:
            md_changes += n
            text = new_text
    if md_changes:
        stats.markdown_cleaned += md_changes

    # ── 5. Remove list markers ──────────────────────────────────────────
    new_text, n = _RE_LIST_MARKER.subn('', text)
    if n:
        stats.list_markers_removed += n
        text = new_text

    # ── 6. Normalize unicode ────────────────────────────────────────────
    unicode_changes = 0
    new_text, n = _RE_SMART_DOUBLE_QUOTES.subn('"', text)
    if n:
        unicode_changes += n
        text = new_text
    new_text, n = _RE_SMART_SINGLE_QUOTES.subn("'", text)
    if n:
        unicode_changes += n
        text = new_text
    new_text, n = _RE_DASHES.subn('—', text)
    if n:
        unicode_changes += n
        text = new_text
    new_text, n = _RE_DOUBLE_DASH.subn('—', text)
    if n:
        unicode_changes += n
        text = new_text
    new_text, n = _RE_NBSP.subn(' ', text)
    if n:
        unicode_changes += n
        text = new_text
    new_text, n = _RE_ZERO_WIDTH.subn('', text)
    if n:
        unicode_changes += n
        text = new_text
    if unicode_changes:
        stats.unicode_normalized += unicode_changes

    # ── 7. Remove emojis ────────────────────────────────────────────────
    new_text, n = _RE_EMOJI.subn('', text)
    if n:
        stats.emojis_removed += n
        text = new_text

    # ── 8. Normalize multiple punctuation (all languages) ───────────────
    punct_changes = 0
    new_text, n = _RE_MULTI_EXCLAIM.subn('!', text)
    if n:
        punct_changes += n
        text = new_text
    new_text, n = _RE_MULTI_QUESTION.subn('?', text)
    if n:
        punct_changes += n
        text = new_text
    new_text, n = _RE_MULTI_PERIOD.subn('...', text)
    if n:
        punct_changes += n
        text = new_text
    new_text, n = _RE_MULTI_COMMA.subn(',', text)
    if n:
        punct_changes += n
        text = new_text
    new_text, n = _RE_MULTI_SEMICOLON.subn(';', text)
    if n:
        punct_changes += n
        text = new_text
    new_text, n = _RE_MULTI_COLON.subn(':', text)
    if n:
        punct_changes += n
        text = new_text
    if punct_changes:
        stats.punctuation_normalized += punct_changes

    # ── CJK-specific: normalize digit+CJK-counter expressions ────────
    # Numbers followed by CJK counter/measure words are common in CJK text.
    # Most CJK TTS engines handle these well natively, but we ensure
    # proper spacing for mixed-language text (e.g., "2024年" reads naturally).
    # We do NOT convert digits to CJK words here — CJK TTS engines prefer digits.
    if is_cjk:
        # Ensure proper spacing around CJK number expressions for
        # mixed-language text where digits precede CJK counters.
        # This is a no-op for pure CJK text but helps with "第3章" etc.
        pass  # CJK TTS engines handle natively; placeholder for future expansion

    # ── Latin-only transformations ─────────────────────────────────────
    if not is_cjk:
        # ── 9. Replace URLs ────────────────────────────────────────────
        def _url_repl(m):
            raw = m.group(0)
            stripped = _URL_TRAILING_PUNCT.sub('', raw)
            trailing = raw[len(stripped):]
            domain_m = re.match(r'https?://(?:www\.)?([^/\s]+)', stripped)
            name = domain_m.group(1) if domain_m else stripped
            return f'link to {name}{trailing}'
        new_text, n = _RE_URL.subn(_url_repl, text)
        if n:
            stats.urls_replaced += n
            text = new_text

        # ── 10. Replace emails ──────────────────────────────────────────
        new_text, n = _RE_EMAIL.subn(r'\1 at \2 dot \3', text)
        if n:
            stats.emails_replaced += n
            text = new_text

        # ── 10.5 Pronunciation: units, measures, special symbols ───────
        # Runs BEFORE the abbreviation table (so "5 min" → "five minutes"
        # wins over min→"minimum") and BEFORE number conversion (digits
        # must still be present for the digit-anchored unit patterns).
        pron_changes = 0

        # ASCII arrows first: "A -> B" / "A => B" were previously shredded
        # into "A minus is greater than B" / "A equals is greater than B".
        new_text, n = _RE_ASCII_ARROW.subn(' to ', text)
        if n:
            pron_changes += n
            text = new_text
        # Math minus sign U+2212 → ASCII hyphen so the minus rules apply.
        new_text, n = _RE_MINUS_SIGN.subn('-', text)
        if n:
            stats.unicode_normalized += n
            text = new_text

        # Programming operators BEFORE the generic comparison rules in step
        # 12: "1 !== 2" was read "is not equal to equals" (!= fired inside
        # !==), "a == b" became "a equals equals b", "a && b" became
        # "a and and b" (& fired per character).
        for op_re, spoken in (
                (_RE_STRICT_NOT_EQUAL, ' is not equal to '),
                (_RE_STRICT_EQUAL, ' is equal to '),
                (_RE_DOUBLE_EQUAL, ' is equal to '),
                (_RE_AND_AND, ' and '),
                (_RE_OR_OR, ' or ')):
            new_text, n = op_re.subn(spoken, text)
            if n:
                pron_changes += n
                text = new_text

        # ── Math pronunciation pass ────────────────────────────────
        # Runs before the superscript/subscript rules (which would shred
        # "d²y/dx²" into "d squared y" and "log₂" into "log two"), before
        # the caret rules (which consume "**" and scientific notation) and
        # before the parenthetical rule (which would speak sin(x) as
        # "sine, x,").
        math_changes = 0

        # Definition / walrus operator FIRST: "f(x) := x^2" → "is defined
        # as" (the leftover-equals rule would say "x : equals x squared").
        new_text, n = _RE_ASSIGN_OP.subn(' is defined as ', text)
        if n:
            math_changes += n
            text = new_text

        # Python exponent: unmatched markdown "**" → caret notation.
        new_text, n = _RE_ASCII_POW.subn('^', text)
        if n:
            math_changes += n
            text = new_text

        # Scientific notation: "1e9" → "1 times 10^9"; the caret rule below
        # speaks the power ("ten to the ninth") and the integer rule keeps
        # the small mantissa digit as a digit.
        new_text, n = _RE_SCI_NOTATION.subn(r'\1 times 10^\2', text)
        if n:
            math_changes += n
            text = new_text

        # Parenthesized exponent: "e^(i*pi)" → "e to the i*pi".
        new_text, n = _RE_CARET_PAREN_EXP.subn(r'\1 to the \2', text)
        if n:
            math_changes += n
            text = new_text

        # Negated math functions: "-sin(x)" → "minus sin(x)" (the function
        # rules below then say "minus sine of x").
        new_text, n = _RE_NEGATIVE_MATH_FUNC.subn(' minus ', text)
        if n:
            math_changes += n
            text = new_text

        # Derivatives, most specific first.
        new_text, n = _RE_SECOND_DERIV.subn(
            r'second derivative of \1 with respect to \2', text)
        if n:
            math_changes += n
            text = new_text
        new_text, n = _RE_SECOND_PARTIAL_DERIV.subn(
            r'second partial derivative of \1 with respect to \2', text)
        if n:
            math_changes += n
            text = new_text
        new_text, n = _RE_PARTIAL_DERIV.subn(r'partial \1 partial \2', text)
        if n:
            math_changes += n
            text = new_text
        new_text, n = _RE_DERIV.subn(r'd \1 by d \2', text)
        if n:
            math_changes += n
            text = new_text
        new_text, n = _RE_D_OVER_DVAR.subn(r'd by d \1', text)
        if n:
            math_changes += n
            text = new_text

        # Prime notation first so "f'(x)" chains into "f prime of x".
        new_text, n = _RE_PRIME.subn(r'\1 prime', text)
        if n:
            math_changes += n
            text = new_text
        new_text, n = _RE_PRIME_CALL.subn(r'\1 prime of \2', text)
        if n:
            math_changes += n
            text = new_text

        # Factorial: "n! = 120" → "n factorial = 120".  Lookahead-guarded
        # so sentence-final exclamations and "!=" stay untouched.
        new_text, n = _RE_FACTORIAL.subn(r'\1 factorial', text)
        if n:
            math_changes += n
            text = new_text

        # Variable division: "x/2" → "x over 2", "2/x" → "2 over x".
        # Runs in this early pass, BEFORE the abbreviation table — its
        # HTML→"H T M L" expansion would otherwise give "L/5" a fresh
        # token start and read "HTML/5" as "H T M L over 5".  The token
        # guards keep "PS/2", "80/kg" and "a(b)"-style tokens intact.
        new_text, n = _RE_VAR_OVER_NUM.subn(r'\1 over \2', text)
        if n:
            math_changes += n
            text = new_text
        new_text, n = _RE_NUM_OVER_VAR.subn(r'\1 over \2', text)
        if n:
            math_changes += n
            text = new_text

        # Log bases: "log_2(x)" / "log₂(x)" → "log base 2 of x"; bare
        # "log_2 64" → "log base 2 64".
        new_text, n = _RE_LOG_BASE_SUB_PAREN.subn(
            lambda m: 'log base ' + ' '.join(
                _ONES[ord(d) - 0x2080] for d in m.group(1))
            + f' of {m.group(2).strip()}', text)
        if n:
            math_changes += n
            text = new_text
        new_text, n = _RE_LOG_BASE_ASCII_PAREN.subn(
            r'log base \1 of \2', text)
        if n:
            math_changes += n
            text = new_text
        new_text, n = _RE_LOG_BASE_SUB.subn(
            lambda m: 'log base ' + ' '.join(
                _ONES[ord(d) - 0x2080] for d in m.group(1)), text)
        if n:
            math_changes += n
            text = new_text
        new_text, n = _RE_LOG_BASE_ASCII.subn(r'log base \1', text)
        if n:
            math_changes += n
            text = new_text

        # Inverse trigonometrics BEFORE the function-call / bare-function
        # rules (whose \b(sin)\b would respell "sin" and break the match)
        # and before the superscript/caret rules, which would read
        # "sin⁻¹(x)" as "sin to the minus first x".  Unicode form first,
        # then the ASCII caret form.
        def _inverse_func_repl(m):
            name = _INVERSE_FUNC_WORDS[m.group(1).lower()]
            if m.group(2) is None:
                return f'{name} '
            return f'{name} of {m.group(2).strip()} '
        new_text, n = _RE_INVERSE_FUNC_SUB.subn(_inverse_func_repl, text)
        if n:
            math_changes += n
            text = new_text
        new_text, n = _RE_INVERSE_FUNC_CARET.subn(_inverse_func_repl, text)
        if n:
            math_changes += n
            text = new_text

        # Function calls: "sin(x)" → "sine of x" — repeated so nested calls
        # ("sin(cos(x))") resolve innermost-first.
        def _math_func_call_repl(m):
            spoken = _MATH_FUNC_SPOKEN[m.group(1).lower()]
            return f'{spoken} {m.group(2).strip()}'
        for _ in range(3):
            new_text, n = _RE_MATH_FUNC_CALL.subn(_math_func_call_repl, text)
            if not n:
                break
            math_changes += n
            text = new_text

        # Function-argument division: "P(A) / P(B)" → "P(A) divided by
        # P(B)" — must run BEFORE the letter-function rule below erases
        # the parens that anchor the match.
        new_text, n = _RE_PAREN_SLASH_DIV.subn(
            lambda m: (m.group(1) or '') + ' divided by ', text)
        if n:
            math_changes += n
            text = new_text

        # Conditional-probability pipe INSIDE parens: "P(A|B)" → "P(A given
        # B)" — also before the letter-function rule, while the parens are
        # still present.  Shell pipes, boolean bars and |x| stay untouched.
        def _pipe_given_repl(m):
            open_pos = text.rfind('(', 0, m.start())
            if open_pos == -1 or ')' in text[open_pos:m.start()]:
                return '|'
            close_pos = text.find(')', m.end())
            if close_pos == -1 or '(' in text[m.end():close_pos]:
                return '|'
            return ' given '
        new_text, n = _RE_PIPE.subn(_pipe_given_repl, text)
        if n:
            math_changes += n
            text = new_text

        # Generic single-letter function: "f(x)" → "f of x" (after the
        # named-function rules so sin(x) & co. are already consumed).
        new_text, n = _RE_LETTER_FUNC_CALL.subn(r'\1 of \2', text)
        if n:
            math_changes += n
            text = new_text

        # Bare function names engines garble: "ln x" → "natural log x".
        def _math_func_bare_repl(m):
            return _MATH_FUNC_BARE[m.group(0).lower()]
        new_text, n = _RE_MATH_FUNC_BARE.subn(_math_func_bare_repl, text)
        if n:
            math_changes += n
            text = new_text

        # Greek-mu units BEFORE the glyph table maps a bare μ to "mu".
        def _greek_micro_repl(m):
            return f'{m.group(1)} {_GREEK_MICRO_WORDS[m.group(2)]}'
        new_text, n = _RE_GREEK_MICRO.subn(_greek_micro_repl, text)
        if n:
            math_changes += n
            text = new_text
        new_text, n = _RE_GREEK_MICRO_BARE.subn(
            lambda m: f' {_GREEK_MICRO_WORDS[m.group(1)]} ', text)
        if n:
            math_changes += n
            text = new_text

        # Greek letters and math glyphs: α → "alpha", ∑ → "sum of",
        # ∅ → "the empty set" — engines skip or stutter on the glyphs.
        def _math_glyph_repl(m):
            return f' {_MATH_GLYPH_WORDS[m.group(0)]} '
        new_text, n = _RE_MATH_GLYPHS.subn(_math_glyph_repl, text)
        if n:
            math_changes += n
            text = new_text

        # Remainder operator: "5 mod 3" → "5 modulo 3" — before the
        # abbreviation table, whose mod→"module" rule would say
        # "5 module 3".  Digit-anchored, so prose "mod" is untouched.
        new_text, n = _RE_MODULO.subn(' modulo ', text)
        if n:
            math_changes += n
            text = new_text

        if math_changes:
            stats.symbols_normalized += math_changes

        # ── History pronunciation pass ──────────────────────────────
        # Era markers (BC/BCE/AD/CE) and year spans.  Runs BEFORE the
        # number section: each year is consumed together with its era or
        # span, so the integer rule can never read "1066" as "one
        # thousand sixty-six" or "44 BC" as "forty-four Bik".
        hist_era_changes = 0
        hist_span_changes = 0

        # "c. 1860" → "circa 1860" FIRST — the era/span rules below turn
        # years into words, and the circa lookahead needs digits.
        new_text, n = _RE_CIRCA_DOT.subn('circa', text)
        if n:
            stats.abbreviations_expanded += n
            text = new_text

        # Year spans FIRST (with optional era suffix): "1914-1918" →
        # "nineteen fourteen to nineteen eighteen", "between 1914 and
        # 1918", "from 1914 to 1918", "3000-2500 BC".  Both years are
        # read year-style in a single substitution — the generic range
        # rule leaves digits, and the second year then lost its year
        # reading.  Spans must precede the standalone era rules, which
        # would otherwise eat the second year and leave the first
        # dangling before a dash.
        _hist_skip = _build_number_skip_positions(text)

        def _year_span_repl(joiner, word_group=1):
            def _repl(m):
                nonlocal hist_span_changes
                if any(p in _hist_skip for p in range(m.start(), m.end())):
                    return m.group(0)
                try:
                    a = _number_to_words(
                        int(m.group(word_group).replace(',', '')))
                    b = _number_to_words(
                        int(m.group(word_group + 1).replace(',', '')))
                except ValueError:
                    return m.group(0)
                hist_span_changes += 1
                # "from"/"between" (group 1 of their patterns) are
                # re-emitted so only the years are rewritten.
                lead = f'{m.group(1)} ' if word_group == 2 else ''
                out = f'{lead}{a} {joiner} {b}'
                # Optional era suffix ("3000-2500 BC").
                if m.lastindex and m.lastindex > word_group + 1:
                    era = m.group(m.lastindex)
                    out += f' {_ERA_SPOKEN[era.replace(".", "").upper()]}'
                return out
            return _repl

        new_text, n = _RE_YEAR_RANGE.subn(_year_span_repl('to'), text)
        if n:
            hist_span_changes += n
            text = new_text
        new_text, n = _RE_YEAR_BETWEEN.subn(
            _year_span_repl('and', word_group=2), text)
        if n:
            hist_span_changes += n
            text = new_text
        new_text, n = _RE_YEAR_FROM_TO.subn(
            _year_span_repl('to', word_group=2), text)
        if n:
            hist_span_changes += n
            text = new_text

        # Era before year: "AD 1066" → "A D ten sixty-six", "CE 220" →
        # "C E two hundred twenty".  Case-sensitive — lowercase "ad" in
        # prose is not an era marker.
        def _era_before_repl(m):
            nonlocal hist_era_changes
            era_key = m.group(1).replace('.', '').upper()
            try:
                words = _number_to_words(int(m.group(2).replace(',', '')))
            except ValueError:
                return m.group(0)
            hist_era_changes += 1
            return f'{_ERA_SPOKEN[era_key]} {words}'

        new_text, n = _RE_ERA_BEFORE_YEAR.subn(_era_before_repl, text)
        if n:
            hist_era_changes += n
            text = new_text

        # Year + era: "44 BC" → "forty-four B C", "2500 BCE" → "two
        # thousand five hundred B C E", "1066 AD" → "ten sixty-six A D".
        # Digit-anchored (bare "BC"/"AD" stay); skip regions
        # ("802.11AD"-style tokens) are re-checked on the current text.
        _hist_skip = _build_number_skip_positions(text)

        def _era_after_repl(m):
            nonlocal hist_era_changes
            if any(p in _hist_skip for p in range(m.start(1), m.end(2))):
                return m.group(0)
            era_key = m.group(2).replace('.', '').upper()
            if era_key not in _ERA_SPOKEN:
                return m.group(0)
            try:
                words = _number_to_words(int(m.group(1).replace(',', '')))
            except ValueError:
                return m.group(0)
            hist_era_changes += 1
            return f'{words} {_ERA_SPOKEN[era_key]}'

        new_text, n = _RE_ERA_AFTER_YEAR.subn(_era_after_repl, text)
        if n:
            hist_era_changes += n
            text = new_text

        if hist_era_changes:
            stats.symbols_normalized += hist_era_changes
        if hist_span_changes:
            stats.numbers_converted += hist_span_changes

        # Composed temperature glyphs (℃/℉) — they contain no "°", so the
        # degree-unit and generic degree rules below never see them.
        # Negative readings ("-40℃") speak "minus" — a bare "-" glued to
        # the number was read "dash" or skipped by engines.
        def _temp_glyph_repl(m, unit_word):
            minus = 'minus ' if m.group(1) else ''
            return f'{minus}{m.group(2)} degrees {unit_word}'
        new_text, n = _RE_CELSIUS_GLYPH.subn(
            lambda m: _temp_glyph_repl(m, 'Celsius'), text)
        if n:
            pron_changes += n
            text = new_text
        new_text, n = _RE_FAHRENHEIT_GLYPH.subn(
            lambda m: _temp_glyph_repl(m, 'Fahrenheit'), text)
        if n:
            pron_changes += n
            text = new_text

        # Tech tokens that later stages would corrupt (run FIRST):
        # ".js" was eaten by the JS→javascript override ("Node.js" →
        # "Node.javascript"); "C++"/"C#" were silent symbols.
        new_text, n = _RE_DOT_JS.subn(' jay ess', text)
        if n:
            pron_changes += n
            text = new_text
        new_text, n = _RE_DOT_NET.subn(' dot net', text)
        if n:
            pron_changes += n
            text = new_text
        new_text, n = _RE_C_PLUS_PLUS.subn('C plus plus', text)
        if n:
            pron_changes += n
            text = new_text
        new_text, n = _RE_C_SHARP.subn(r'\1 sharp', text)
        if n:
            pron_changes += n
            text = new_text

        # Keyboard shortcuts: "Ctrl+S" → "Ctrl plus S" (the abbreviation
        # tier later says "control plus S").  Runs BEFORE the abbreviation
        # table; the modifier word anchors the match.
        new_text, n = _RE_KEY_SHORTCUT.subn(
            lambda m: m.group(0).replace('+', ' plus '), text)
        if n:
            pron_changes += n
            text = new_text

        # Dotted quads (IPv4 / CIDR) BEFORE decimals: "192.168.1.1" read
        # digit-by-digit with "dot" separators; "10.0.0.0/8" gains "slash 8".
        # Dotted version strings ("v2.5.1.2", "version 3.5.1.2") are excluded
        # via the version-prefix guard.
        def _dotted_quad_repl(m):
            prefix = text[max(0, m.start() - 16):m.start()]
            if _RE_VERSION_PREFIX_TAIL.search(prefix):
                return m.group(0)
            octets = [
                ' '.join(_DIGIT_WORDS[int(d)] for d in octet)
                for octet in m.group(1).split('.')
            ]
            spoken = ' dot '.join(octets)
            if m.group(2):   # host:port → "... port eight zero eight zero"
                spoken += ' port ' + ' '.join(
                    _DIGIT_WORDS[int(d)] for d in m.group(2))
            elif m.group(3):  # CIDR → "... slash 8"
                spoken += f' slash {m.group(3)}'
            return spoken
        new_text, n = _RE_DOTTED_QUAD.subn(_dotted_quad_repl, text)
        if n:
            stats.numbers_converted += n
            text = new_text

        # Unicode fractions: "½" → "one half" (unread glyph otherwise).
        def _unicode_fraction_repl(m):
            nonlocal pron_changes
            pron_changes += 1
            return ' ' + _UNICODE_FRACTIONS[m.group(0)] + ' '
        new_text, n = _RE_UNICODE_FRACTION.subn(_unicode_fraction_repl, text)
        if n:
            stats.numbers_converted += n
            text = new_text

        # General Unicode superscripts BEFORE the ²/³ rules below: multi-char
        # runs, minus-prefixed runs and single digits other than ²/³.
        # "10⁶" → "10 to the sixth", "2⁴⁵" → "2 to the forty-fifth",
        # "10⁻⁶" → "10 to the minus sixth", "xⁿ" → "x to the n".
        # Single ²/³ are left for the squared/cubed rules beneath.
        sup_seq_changes = 0

        def _sup_ordinal(seq: str) -> str:
            num = ''.join(_SUP_DIGIT_VALUES[c] for c in seq)
            try:
                val = int(num)
            except ValueError:
                return num
            return 'zeroth' if val == 0 else _ordinal_to_words(num, 'th')

        def _sup_run_repl(m):
            nonlocal sup_seq_changes
            sup_seq_changes += 1
            return f' to the {_sup_ordinal(m.group(0))} '

        def _sup_minus_repl(m):
            nonlocal sup_seq_changes
            sup_seq_changes += 1
            return f' to the minus {_sup_ordinal(m.group(0)[1:])} '

        def _sup_n_repl(m):
            nonlocal sup_seq_changes
            sup_seq_changes += 1
            return ' to the n '

        new_text, n = _RE_SUP_RUN_MINUS.subn(_sup_minus_repl, text)
        if n:
            sup_seq_changes += n
            text = new_text
        new_text, n = _RE_SUP_RUN_MULTI.subn(_sup_run_repl, text)
        if n:
            sup_seq_changes += n
            text = new_text
        new_text, n = _RE_SUP_RUN_SINGLE.subn(_sup_run_repl, text)
        if n:
            sup_seq_changes += n
            text = new_text
        new_text, n = _RE_SUP_N.subn(_sup_n_repl, text)
        if n:
            sup_seq_changes += n
            text = new_text

        # Unicode subscripts: "CO₂" → "CO two", "C₁₂H₂₂O₁₁" →
        # "C one two H two two O one one" (digits are unread glyphs).
        def _sub_repl(m):
            nonlocal sup_seq_changes
            sup_seq_changes += 1
            # Subscript digits start at U+2080; ord() arithmetic converts
            # them to ASCII values ('₂' is not accepted by int()).
            return ' ' + ' '.join(
                _ONES[ord(d) - 0x2080] for d in m.group(0)) + ' '
        new_text, n = _RE_SUBSCRIPT.subn(_sub_repl, text)
        if n:
            sup_seq_changes += n
            text = new_text
        if sup_seq_changes:
            stats.numbers_converted += sup_seq_changes

        # Superscripted kinematic compounds BEFORE the ²/³ superscript rules
        # and the generic unit loop — ² is alphanumeric in Python's \w, so
        # "m²/s" would be shredded into "square meters or s" by the unit²
        # rule and "s²" in "m/s²" by the alpha² rule.  Also a BUG FIX:
        # m/s² is acceleration — "meters per second SQUARED" (the old
        # compound table said "square meters per second", which is m²/s);
        # "12 m²/s" → "12 square meters per second".
        kin_changes = 0
        for kin_re, spoken in (
                (_RE_M_PER_S2, r'\1 meters per second squared'),
                (_RE_M2_PER_S, r'\1 square meters per second'),
                (_RE_M3_PER_S, r'\1 cubic meters per second'),
                (_RE_BARE_M_PER_S2, ' meters per second squared '),
                (_RE_BARE_M2_PER_S, ' square meters per second '),
                (_RE_BARE_M3_PER_S, ' cubic meters per second ')):
            new_text, n = kin_re.subn(spoken, text)
            if n:
                kin_changes += n
                text = new_text
        if kin_changes:
            stats.abbreviations_expanded += kin_changes

        # Superscripts BEFORE units: "5 m²" must become "5 square meters",
        # not "5 meters ²"; "E=mc²" → "E=mc squared"; "9²" → "9 squared".
        sup_changes = 0
        def _sup_repl(mapping):
            def _repl(m):
                nonlocal sup_changes
                sup_changes += 1
                word = mapping[m.group(1).lower()]
                # Singular for exact "1": "1 m²" → "1 square meter".
                # Peek behind the unit for the preceding number.
                prefix = text[max(0, m.start() - 12):m.start()]
                num_m = re.search(r'(\d[\d,]*(?:\.\d+)?)\s*$', prefix)
                if num_m:
                    try:
                        if int(num_m.group(1).replace(',', '')
                               .split('.')[0]) == 1:
                            word = _singular_measure(word)
                    except (ValueError, IndexError):
                        pass
                return ' ' + word + ' '
            return _repl
        new_text, n = _RE_UNIT_SQUARED.subn(_sup_repl(_SQUARED_UNIT_WORDS), text)
        if n:
            sup_changes += n
            text = new_text
        new_text, n = _RE_UNIT_CUBED.subn(_sup_repl(_CUBED_UNIT_WORDS), text)
        if n:
            sup_changes += n
            text = new_text
        new_text, n = _RE_ALPHA_SQUARED.subn(r'\1 squared', text)
        if n:
            sup_changes += n
            text = new_text
        new_text, n = _RE_ALPHA_CUBED.subn(r'\1 cubed', text)
        if n:
            sup_changes += n
            text = new_text
        new_text, n = _RE_DIGIT_SQUARED.subn(r'\1 squared', text)
        if n:
            sup_changes += n
            text = new_text
        new_text, n = _RE_DIGIT_CUBED.subn(r'\1 cubed', text)
        if n:
            sup_changes += n
            text = new_text
        if sup_changes:
            stats.numbers_converted += sup_changes

        # Degree units BEFORE the generic ° rule: "20°C" → "20 degrees
        # Celsius" (previously "twenty degreesC" with a glued C).
        # Negative readings ("-40°C") speak "minus", not a bare dash.
        def _degree_unit_repl(m):
            nonlocal pron_changes
            pron_changes += 1
            minus = 'minus ' if m.group(1) else ''
            return (f'{minus}{m.group(2)} degrees '
                    f'{_DEGREE_UNIT_WORDS[m.group(3)]}')
        new_text, n = _RE_DEGREE_UNIT.subn(_degree_unit_repl, text)
        if n:
            stats.symbols_normalized += n
            text = new_text

        # Measurement units: "5 kg" → "5 kilograms", "10 MB" → "10
        # megabytes", "2.4 GHz" → "2.4 gigahertz", "5 min" → "5 minutes".
        # Exact "1" gets a singular unit ("1 kilometer").
        unit_changes = 0
        def _unit_repl(m, word):
            nonlocal unit_changes
            unit_changes += 1
            try:
                is_one = int(m.group(1).replace(',', '').split('.')[0]) == 1
            except (ValueError, IndexError):
                is_one = False
            spoken = _singular_unit(word) if is_one else word
            return f'{m.group(1)} {spoken}'
        for unit_re, unit_word in _UNIT_EXPANSIONS:
            new_text, n = unit_re.subn(
                lambda m, w=unit_word: _unit_repl(m, w), text)
            if n:
                text = new_text
        # "10Mbps" (no space) is matched by the compound/unit rules above;
        # bare lowercase m after a digit is meters.
        new_text, n = _RE_BARE_METERS.subn(
            lambda m: _unit_repl(m, 'meters'), text)
        if n:
            text = new_text
        # Plain-digit powers: "25 m3" → "25 cubic meters", "12 m2" →
        # "12 square meters" (slides rarely carry real superscripts).
        def _plain_power_repl(m, table):
            nonlocal unit_changes
            unit_changes += 1
            num = m.group(1)
            unit = m.group(2).lower()
            word = table[unit]
            try:
                is_one = int(num.replace(',', '').split('.')[0]) == 1
            except (ValueError, IndexError):
                is_one = False
            if is_one:
                word = _singular_measure(word)
            return f'{num} {word}'
        for power_re, power_table in (
                (_RE_PLAIN_VOLUME, _PLAIN_VOLUME_WORDS),
                (_RE_PLAIN_AREA, _PLAIN_AREA_WORDS)):
            new_text, n = power_re.subn(
                lambda m, t=power_table: _plain_power_repl(m, t), text)
            if n:
                text = new_text
        # Real-estate style: "45 sq ft" → "45 square feet", "4 cu ft" →
        # "4 cubic feet" (no leading digit required).
        for unit_re, unit_table in (
                (_RE_SQ_UNIT, _SQ_UNIT_WORDS),
                (_RE_CU_UNIT, _CU_UNIT_WORDS)):
            def _sqcu_repl(m, t=unit_table):
                nonlocal unit_changes
                unit_changes += 1
                return ' ' + t[m.group(1).lower()]
            new_text, n = unit_re.subn(_sqcu_repl, text)
            if n:
                text = new_text
        if unit_changes:
            stats.abbreviations_expanded += unit_changes

        # Micro units: "5 µs" → "5 microseconds"; stray "µ" → " micro ".
        micro_changes = 0
        for micro_re, spoken in (
                (_RE_MICRO_SECONDS, r'\1 microseconds'),
                (_RE_MICRO_METERS, r'\1 micrometers'),
                (_RE_MICRO_GRAMS, r'\1 micrograms')):
            new_text, n = micro_re.subn(spoken, text)
            if n:
                micro_changes += n
                text = new_text
        new_text, n = _RE_MICRO_BARE.subn(' micro ', text)
        if n:
            micro_changes += n
            text = new_text
        if micro_changes:
            stats.symbols_normalized += micro_changes

        # Caret exponent notation: "2^10" → "two to the tenth",
        # "10^-6" → "ten to the minus sixth", "x^2" → "x squared",
        # "e^x" → "e to the x".  Runs AFTER units so "10^6 kg" first
        # becomes "10^6 kilograms" (the unit pattern anchors on the exposed
        # "6 kg") and the caret then converts the whole exponent cleanly —
        # previously the integer rule CORRUPTED the exponent ("2^ten").
        caret_changes = 0

        def _caret_num_repl(m):
            nonlocal caret_changes
            caret_changes += 1
            base = _int_to_words(int(m.group(1).replace(',', '')))
            minus, exp = m.group(2), m.group(3)
            if not minus and exp == '2':
                spoken = 'squared'
            elif not minus and exp == '3':
                spoken = 'cubed'
            else:
                spoken = ('to the ' + ('minus ' if minus else '')
                          + _caret_exp_words(exp))
            return f'{base} {spoken}'

        def _caret_num_letter_repl(m):
            nonlocal caret_changes
            caret_changes += 1
            base = _int_to_words(int(m.group(1).replace(',', '')))
            return (f'{base} to the '
                    f'{"minus " if m.group(2) else ""}{m.group(3)}')

        def _caret_alpha_num_repl(m):
            nonlocal caret_changes
            caret_changes += 1
            minus, exp = m.group(2), m.group(3)
            if not minus and exp == '2':
                return f'{m.group(1)} squared'
            if not minus and exp == '3':
                return f'{m.group(1)} cubed'
            return (f'{m.group(1)} to the '
                    f'{"minus " if minus else ""}{_caret_exp_words(exp)}')

        def _caret_alpha_letter_repl(m):
            nonlocal caret_changes
            caret_changes += 1
            return (f'{m.group(1)} to the '
                    f'{"minus " if m.group(2) else ""}{m.group(3)}')

        new_text, n = _RE_CARET_NUM.subn(_caret_num_repl, text)
        if n:
            text = new_text
        new_text, n = _RE_CARET_NUM_LETTER.subn(_caret_num_letter_repl, text)
        if n:
            text = new_text
        new_text, n = _RE_CARET_ALPHA_NUM.subn(_caret_alpha_num_repl, text)
        if n:
            text = new_text
        new_text, n = _RE_CARET_ALPHA_LETTER.subn(
            _caret_alpha_letter_repl, text)
        if n:
            text = new_text
        # Any leftover caret ("2^3^4" chains) is a silent separator.
        new_text, n = _RE_LEFTOVER_CARET.subn(' ', text)
        if n:
            caret_changes += n
            text = new_text
        if caret_changes:
            stats.numbers_converted += caret_changes

        # Bare compound rates WITHOUT a leading digit ("measured in km/h"):
        # the digit-anchored unit rules above skip them and the word/word
        # slash rule later would read "km/h" as "km or h" and leave
        # "80 MB/s" as "megabytes or s".
        bare_compound_changes = 0
        for cmp_re, spoken in _RE_BARE_RATE_COMPOUNDS:
            new_text, n = cmp_re.subn(spoken, text)
            if n:
                bare_compound_changes += n
                text = new_text
        if bare_compound_changes:
            stats.abbreviations_expanded += bare_compound_changes

        # Rate shorthand: "100 ops/sec" → "100 ops per second", "5
        # msg/min" → "5 msg per minute".  MUST run before the abbreviation
        # table: its (?=\s|$) lookahead mangles the exposed "sec"/"min"
        # after the slash ("ops or seconds", "msg or minimum"), and before
        # the word/word slash rule which would say "ops or sec".
        def _rate_repl(m):
            return (f'{m.group(1)} per '
                    f'{_RATE_UNIT_WORDS[m.group(2).lower()]}')
        new_text, n = _RE_RATE_SLASH.subn(_rate_repl, text)
        if n:
            stats.abbreviations_expanded += n
            text = new_text

        # Informal shorthand: "w/o attachments" → "without attachments",
        # "w/ team" → "with team" (previously "w or the team").  NOTE: w/o
        # must run FIRST — the w/ rule would otherwise eat its slash
        # ("w/o" → "with o").
        shorthand_changes = 0
        new_text, n = _RE_W_O.subn('without', text)
        if n:
            shorthand_changes += n
            text = new_text
        new_text, n = _RE_W_SLASH.subn('with ', text)
        if n:
            shorthand_changes += n
            text = new_text
        if shorthand_changes:
            stats.abbreviations_expanded += shorthand_changes

        # Ohm family, digit-anchored: "50 Ω" → "50 ohms", "10 kΩ" →
        # "10 kilohms", "5 MΩ" → "five megaohms", "2 mΩ" → milliohms.
        # U+03A9 (Greek capital omega) and U+2126 (ohm sign) both occur.
        for ohm_re, spoken in (
                (_RE_MEGOHM, r'\1 megaohms'),
                (_RE_MILLIOHM, r'\1 milliohms'),
                (_RE_KILOHM, r'\1 kilohms'),
                (_RE_OHM, r'\1 ohms')):
            new_text, n = ohm_re.subn(spoken, text)
            if n:
                pron_changes += n
                text = new_text

        # Symbols with no native TTS reading, spoken explicitly.
        explicit_changes = 0
        for sym_re, spoken in (
                (_RE_PLUS_MINUS, ' plus or minus '),
                (_RE_APPROX_EQUAL, ' approximately '),
                (_RE_NOT_EQUAL_SIGN, ' is not equal to '),
                (_RE_INFINITY, ' infinity '),
                (_RE_DIVIDE_SIGN, ' divided by '),
                (_RE_ARROW_RIGHT, ' to '),
                (_RE_PER_MILLE, ' per mille '),
                (_RE_SECTION_SIGN, ' section '),
                (_RE_PARAGRAPH_SIGN, ' paragraph '),
                (_RE_GTE_SIGN, ' is greater than or equal to '),
                (_RE_LTE_SIGN, ' is less than or equal to '),
                (_RE_SQRT, ' square root of '),
                (_RE_PI, ' pi '),
                (_RE_DELTA, ' delta '),
                (_RE_INTEGRAL, ' integral of '),
                (_RE_NUMERO_SIGN, ' number '),
                (_RE_BARE_OHM, ' ohms '),
                (_RE_BARE_CELSIUS, ' degrees Celsius '),
                (_RE_BARE_FAHRENHEIT, ' degrees Fahrenheit ')):
            new_text, n = sym_re.subn(spoken, text)
            if n:
                explicit_changes += n
                text = new_text
        # Silent typographic marks (©, ®, ™, †, ‡, •) — removed.
        new_text, n = _RE_SILENT_SYMBOLS.subn(' ', text)
        if n:
            explicit_changes += n
            text = new_text
        # Trailing currency: "50$" → "fifty dollars" (the "$50" prefix form
        # was already spoken; the trailing form left a bare dollar sign).
        new_text, n = _RE_TRAILING_DOLLAR.subn(
            lambda m: _currency_to_words(m.group(1), '$'), text)
        if n:
            explicit_changes += n
            text = new_text
        if explicit_changes:
            stats.symbols_normalized += explicit_changes

        # ── Identifier readability: snake_case and camelCase ────────────
        # TTS engines garble "my_var_name" and "getUserData"; splitting
        # them into words is far more intelligible.  Known mixed-case
        # brand names are preserved via _CAMEL_KEEP.
        ident_changes = 0
        if '_' in text:
            new_text, n = _RE_SNAKE_UNDERSCORE.subn(' ', text)
            if n:
                ident_changes += n
                text = new_text
            # Leading/trailing underscores ("_private") have no word char
            # on one side — collapse the remainder.
            new_text, n = _RE_LEFTOVER_UNDERSCORE.subn(' ', text)
            if n:
                ident_changes += n
                text = new_text

        def _camel_repl(m):
            word = m.group(0)
            if word.lower() in _CAMEL_KEEP:
                return word
            return _RE_CAMEL_SPLIT.sub(' ', word)

        new_text, n = _RE_CAMEL_WORD.subn(_camel_repl, text)
        if n:
            ident_changes += n
            text = new_text
        if ident_changes:
            # Same word-separation family as the hyphen rules below.
            stats.hyphens_normalized += ident_changes

        if pron_changes:
            stats.symbols_normalized += pron_changes

        # ── 11. Expand abbreviations ───────────────────────────────────
        abbr_changes = 0
        for pattern, expansion in _ABBREVIATION_EXPANSIONS.items():
            new_text, n = pattern.subn(expansion, text)
            if n:
                abbr_changes += n
                text = new_text
        # Case-sensitive tier: "US"/"IT"/"ID" are only expanded in
        # uppercase — the lowercase words ("us", "it", "id") are ordinary
        # English and must survive untouched.
        for pattern, expansion in _ABBREVIATION_EXPANSIONS_CS.items():
            new_text, n = pattern.subn(expansion, text)
            if n:
                abbr_changes += n
                text = new_text
        if abbr_changes:
            stats.abbreviations_expanded += abbr_changes

        # ── 11b. Apply pronunciation overrides ─────────────────────────
        # These run AFTER standard abbreviation expansion so users can
        # override any expansion.  The overrides come from the caller's
        # ChunkingConfig (passed via the new _overrides param).
        if _overrides:
            ov_changes = 0
            for ov_key, ov_spoken in _overrides.items():
                # Accept raw pattern strings (documented usage) as well as
                # pre-compiled patterns — the old code crashed with
                # AttributeError: 'str' object has no attribute 'subn'.
                ov_pattern = (
                    re.compile(ov_key, re.IGNORECASE)
                    if isinstance(ov_key, str) else ov_key
                )
                new_text, n = ov_pattern.subn(ov_spoken, text)
                if n:
                    ov_changes += n
                    text = new_text
            if ov_changes:
                logger.debug(f"[Preprocess] Pronunciation overrides applied: {ov_changes}")
                stats.abbreviations_expanded += ov_changes

        # ── 12. Normalize symbols (comparison operators, currency, etc.) ──
        sym_changes = 0
        # Comparison / logic operators (before number conversion!)
        new_text, n = _RE_NOT_EQUAL.subn(' is not equal to ', text)
        if n:
            sym_changes += n
            text = new_text
        new_text, n = _RE_GTE.subn(' is greater than or equal to ', text)
        if n:
            sym_changes += n
            text = new_text
        new_text, n = _RE_LTE.subn(' is less than or equal to ', text)
        if n:
            sym_changes += n
            text = new_text
        new_text, n = _RE_GT.subn(' is greater than ', text)
        if n:
            sym_changes += n
            text = new_text
        new_text, n = _RE_LT.subn(' is less than ', text)
        if n:
            sym_changes += n
            text = new_text
        # Currency symbols (before number conversion!)
        for sym_re, sym_ch in [
            (_RE_CURRENCY_DOLLAR, '$'),
            (_RE_CURRENCY_EURO, '€'),
            (_RE_CURRENCY_POUND, '£'),
            (_RE_CURRENCY_YEN, '¥'),
            (_RE_CURRENCY_RUPEE, '₹'),
        ]:
            new_text, n = sym_re.subn(
                lambda m, s=sym_ch: _currency_to_words(
                    m.group(1), s, m.group(2) or ''),
                text,
            )
            if n:
                sym_changes += n
                text = new_text
        # Magnitude suffixes AFTER currency (so "$5K" is consumed whole):
        # "10K users" → "10 thousand users", "5M" → "five million".
        # Attached form only; 2K/4K/8K (resolution), 24K (gold) and 401K
        # (retirement) are left for TTS to read naturally as "… kay".
        def _magnitude_repl(m):
            mag = m.group(2)
            if mag in ('K', 'k'):
                try:
                    if int(m.group(1).replace(',', '').split('.')[0]) \
                            in _MAGNITUDE_K_EXCLUDE:
                        return m.group(0)
                except (ValueError, IndexError):
                    return m.group(0)
            return f'{m.group(1)} {_MAGNITUDE_WORDS[mag.upper()]}'
        new_text, n = _RE_MAGNITUDE_KMB.subn(_magnitude_repl, text)
        if n:
            sym_changes += n
            text = new_text
        # Hash number: #123 → "number 123"
        new_text, n = _RE_HASH_NUMBER.subn(r'number \1', text)
        if n:
            sym_changes += n
            text = new_text
        # Dotted port numbers: "port 8080" → "port eight zero eight zero"
        # ("eight thousand eighty" is how the integer rule would read it —
        # port numbers are conventionally digit-by-digit).
        def _port_repl(m):
            return 'port ' + ' '.join(
                _DIGIT_WORDS[int(d)] for d in m.group(1))
        new_text, n = _RE_PORT_NUMBER.subn(_port_repl, text)
        if n:
            sym_changes += n
            text = new_text
        # "+/-" BEFORE the standalone rules — otherwise the plus fired
        # alone and "/-3" was left dangling ("plus /-3 percent").
        new_text, n = _RE_PLUS_SLASH_MINUS.subn(' plus or minus ', text)
        if n:
            sym_changes += n
            text = new_text
        # Tight addition: "3+4" / "a+b" → "3 plus 4" / "a plus b".
        new_text, n = _RE_TIGHT_PLUS.subn(' plus ', text)
        if n:
            sym_changes += n
            text = new_text
        # Tight multiplication with a letter/paren operand: "a*b", "i*pi",
        # ") * P(A)" → "times" (digit*digit keeps the multiply rule below,
        # which owns the "1920 by 1080" dimension reading).
        new_text, n = _RE_TIGHT_MULTIPLY.subn(' times ', text)
        if n:
            sym_changes += n
            text = new_text
        # Negative numbers: "-5" / "x=-5" → "minus 5" (anchored minus only;
        # ranges, phones and hyphenated tokens keep their dash).
        new_text, n = _RE_NEGATIVE_NUMBER.subn(' minus ', text)
        if n:
            sym_changes += n
            text = new_text
        # Standalone plus
        new_text, n = _RE_STANDALONE_PLUS.subn(' plus ', text)
        if n:
            sym_changes += n
            text = new_text
        # Standalone minus
        new_text, n = _RE_STANDALONE_MINUS.subn(' minus ', text)
        if n:
            sym_changes += n
            text = new_text
        # Tilde → approximately
        new_text, n = _RE_TILDE.subn(' approximately ', text)
        if n:
            sym_changes += n
            text = new_text
        # Standalone @
        new_text, n = _RE_AT_SIGN.subn(' at ', text)
        if n:
            sym_changes += n
            text = new_text
        # Legacy symbol patterns
        new_text, n = _RE_AMPERSAND.subn(' and ', text)
        if n:
            sym_changes += n
            text = new_text
        new_text, n = _RE_PLUS_EQUALS.subn(' plus equals ', text)
        if n:
            sym_changes += n
            text = new_text
        new_text, n = _RE_EQUALS.subn(' equals ', text)
        if n:
            sym_changes += n
            text = new_text
        # Any equals the spaced rule missed ("x=5", "a=b=c") — after the
        # strict/double/plus-equals and comparison rules, every remaining
        # "=" is a lone assignment or comparison.
        new_text, n = _RE_EQUALS_LEFTOVER.subn(' equals ', text)
        if n:
            sym_changes += n
            text = new_text
        new_text, n = _RE_DEGREE.subn(' degrees', text)
        if n:
            sym_changes += n
            text = new_text
        # Dimension products: "1920×1080" is spoken "nineteen twenty by ten
        # eighty" — the old code said "one thousand nine hundred twenty
        # times one thousand eighty".  Small operands keep the arithmetic
        # reading ("3 times 3"); both sides ≥ 3 digits read as dimensions.
        def _multiply_repl(m):
            a = m.group(1).replace(',', '')
            b = m.group(2).replace(',', '')
            if len(a) >= 3 and len(b) >= 3:
                try:
                    return (f'{_number_to_words(int(a))} by '
                            f'{_number_to_words(int(b))}')
                except ValueError:
                    pass
            return f'{m.group(1)} times {m.group(2)}'
        new_text, n = _RE_MULTIPLY_DIGITS.subn(_multiply_repl, text)
        if n:
            sym_changes += n
            text = new_text
        # Trailing times-sign: "4× zoom" → "4 times zoom" (a bare × has no
        # engine reading; digit-on-the-right products were consumed above).
        new_text, n = _RE_TRAILING_TIMES_SIGN.subn(r'\1 times', text)
        if n:
            sym_changes += n
            text = new_text
        # Note: % and digit/digit / are handled after number conversion
        if sym_changes:
            stats.symbols_normalized += sym_changes

        # ── 13. Convert numbers to words ─────────────────────────────────
        # Order matters: dates → ordinals → time → ratios → fractions →
        # decimals → integers.  Skip positions for version/model/tech
        # regions are REBUILT immediately before EVERY rule that consults
        # them — positions computed on earlier text are stale after any
        # length-changing substitution (the old bug that turned
        # "Windows 11" into "Windows eleven"; a decimal or multiplier
        # before the version string shifted the offsets after the single
        # mid-section rebuild).
        num_changes = 0

        # 13-prefix. Skip-position builder for version/model/tech regions.
        # (_build_number_skip_positions is module-level so the history
        # pass above shares the same protection set; positions are
        # rebuilt on the current text before EVERY rule that consults
        # them — see the function docstring.)
        _number_skip_positions = _build_number_skip_positions(text)

        # 13a-1. Decades and plural years: "the 1990s" → "the nineteen
        # nineties", "the 90s" → "the nineties", "1900s" → "nineteen
        # hundreds".  The integer pattern cannot match "1990s" (no word
        # boundary before the trailing "s"), so engines read it
        # digit-by-digit plus "ess".  Runs FIRST, while skip positions
        # are still fresh.
        def _decade4_repl(m):
            nonlocal num_changes
            if any(p in _number_skip_positions for p in range(m.start(), m.end())):
                return m.group(0)
            words = _decade_to_words(int(m.group(1)))
            if not words:
                return m.group(0)
            num_changes += 1
            return words
        text = _RE_DECADE_4.sub(_decade4_repl, text)

        def _decade2_repl(m):
            nonlocal num_changes
            dec = int(m.group(1))
            if dec < 10 or dec % 10:
                return m.group(0)
            num_changes += 1
            word = 'tens' if dec == 10 else _TENS[dec // 10].replace('y', 'ies')
            return f'the {word}'
        text = _RE_DECADE_2.sub(_decade2_repl, text)

        # 13a-1b. Apostrophe decades: "the '90s" → "the nineties",
        # "mid-'80s" → "mid-eighties"; and "mid-90s" → "mid-nineties".
        # Without these, the bare-duration rule later read "90s" as
        # "ninety seconds" ("the ninety seconds were wild").
        def _apos_decade_repl(m):
            nonlocal num_changes
            word = _decade_word_from_pair(m.group(1))
            if not word:
                return m.group(0)
            num_changes += 1
            return word

        def _mid_decade_repl(m):
            nonlocal num_changes
            word = _decade_word_from_pair(m.group(1))
            if not word:
                return m.group(0)
            num_changes += 1
            # The (?<=mid-) lookbehind does NOT consume the prefix — it
            # stays in the text, so only the decade word is inserted
            # ("mid-90s" → "mid-nineties", never "mid-mid-nineties").
            return word
        text = _RE_APOS_DECADE.sub(_apos_decade_repl, text)
        text = _RE_MID_DECADE.sub(_mid_decade_repl, text)

        # 13a-1c. Possessive years: "1990's style" → "nineteen nineties
        # style", "2023's revenue" → "twenty twenty-three's revenue"
        # (the integer rule read "one thousand nine hundred ninety's").
        def _year_poss_repl(m):
            nonlocal num_changes
            y = int(m.group(1))
            words = _decade_to_words(y)
            if not words and 1000 <= y <= 2099:
                words = _number_to_words(y) + "'s"
            if not words:
                return m.group(0)
            num_changes += 1
            return words
        text = _RE_YEAR_POSSESSIVE.sub(_year_poss_repl, text)

        # 13a0. Roman numerals in anchored contexts: "Chapter IV" →
        # "Chapter four", "World War II" → "World War two", "Henry VIII" →
        # "Henry the eighth", "the XIX century" → "the nineteenth century".
        # Unanchored roman-looking tokens (MIX, XL clothing sizes, DIY…) are
        # left alone; single L/C/D/M are excluded (size XL, "Plan C").
        def _roman_valid(g: str) -> bool:
            if len(g) == 1 and g.upper() in 'LCDM':
                return False
            return _roman_to_int(g) > 0

        def _roman_section_repl(m):
            nonlocal num_changes
            if not _roman_valid(m.group(2)):
                return m.group(0)
            num_changes += 1
            return f'{m.group(1)} {_int_to_words(_roman_to_int(m.group(2)))}'
        text = _RE_ROMAN_SECTION.sub(_roman_section_repl, text)

        def _roman_name_repl(m):
            nonlocal num_changes
            if not _roman_valid(m.group(2)):
                return m.group(0)
            num_changes += 1
            return (f'{m.group(1)} the '
                    f'{_ordinal_to_words(str(_roman_to_int(m.group(2))), "th")}')
        text = _RE_ROMAN_NAME.sub(_roman_name_repl, text)

        def _roman_century_repl(m):
            nonlocal num_changes
            if _roman_to_int(m.group(1)) <= 0:
                return m.group(0)
            num_changes += 1
            return (f'{_ordinal_to_words(str(_roman_to_int(m.group(1))), "th")} '
                    f'{m.group(2)}')
        text = _RE_ROMAN_CENTURY.sub(_roman_century_repl, text)

        # 13a1. Month abbreviations: "Jan 15" → "January 15" (May is already
        # a full word and is excluded).
        def _month_repl(m):
            nonlocal num_changes
            num_changes += 1
            return _MONTH_ABBR_WORDS[m.group(1).lower()]
        text = _RE_MONTH_ABBR.sub(_month_repl, text)

        # 13a1b. Full month + bare day: "January 15" → "January fifteenth",
        # "March 3" → "March third" (a bare day was read as a cardinal —
        # "March three").  Days already carrying an ordinal suffix are
        # skipped by the pattern's lookahead and handled by the ordinal rule.
        def _month_day_repl(m):
            nonlocal num_changes
            day = m.group(2)
            if int(day) > 31:
                return m.group(0)
            num_changes += 1
            return f'{m.group(1)} {_ordinal_to_words(day, "th")}'
        text = _RE_MONTH_DAY.sub(_month_day_repl, text)

        # 13a1c. ISO dates: "2024-01-15" → "January fifteenth, twenty
        # twenty-four".  BEFORE the range rule, which would otherwise read
        # "2024-01" as "2024 to 01".  _date_to_words validates the parts.
        def _iso_date_repl(m):
            nonlocal num_changes
            words = _date_to_words(m.group(2), m.group(3), m.group(1))
            if words:
                num_changes += 1
                return words
            return m.group(0)
        text = _RE_ISO_DATE.sub(_iso_date_repl, text)

        # 13a1d. Phone numbers: "555-123-4567" → "five five five, one two
        # three, four five six seven" (digit-by-digit with pause commas —
        # the range rule would say "five hundred fifty-five to one hundred
        # twenty-three").
        def _phone_repl(m):
            nonlocal num_changes
            num_changes += 1
            return ', '.join(
                ' '.join(_DIGIT_WORDS[int(d)] for d in grp)
                for grp in m.groups())
        text = _RE_PHONE_NUMBER.sub(_phone_repl, text)

        # 13a2-pre. "50-50" / "50/50" → "fifty fifty" — the range rule
        # would say "fifty to fifty" and the fraction rule "fifty divided
        # by fifty"; the idiom is always spoken as a bare pair.
        def _fifty_fifty_repl(m):
            nonlocal num_changes
            num_changes += 1
            return 'fifty fifty'
        text = _RE_50_50.sub(_fifty_fifty_repl, text)

        # 13a2. Numeric ranges: "10-20" → "10 to 20", "2020—2024" →
        # "2020 to 2024" (previously read as a hyphenated slur).  Matches
        # overlapping version/tech skip regions are preserved.
        _number_skip_positions = _build_number_skip_positions(text)
        def _range_repl(m):
            nonlocal num_changes
            if any(p in _number_skip_positions for p in range(m.start(), m.end())):
                return m.group(0)
            num_changes += 1
            return f'{m.group(1)} to {m.group(2)}'
        text = _RE_NUMERIC_RANGE.sub(_range_repl, text)

        # 13a3. "24/7" → "twenty-four seven" BEFORE the fraction rule would
        # call it "twenty-four sevenths".
        def _always_repl(m):
            nonlocal num_changes
            num_changes += 1
            return 'twenty-four seven'
        text = _RE_24_7.sub(_always_repl, text)

        # 13a3b. "9/11" → "September eleventh" — the fraction rule would
        # otherwise say "nine elevenths".
        text = _RE_9_11.sub('September eleventh', text)

        # 13a. Calendar dates: 12/25/2024 → "December twenty-fifth, 2024"
        # (before fractions — the fraction rule would read 12/25 as a
        # fraction and the slash rule would garble the rest).
        def _date_repl(m):
            nonlocal num_changes
            words = _date_to_words(m.group(1), m.group(2), m.group(3))
            if words:
                num_changes += 1
                return words
            return m.group(0)
        text = _RE_DATE_M_D_Y.sub(_date_repl, text)

        # 13b. Ordinal suffixes: 1st, 2nd, 3rd, 21st, etc.
        # (Skip ordinals inside version/tech/model regions)
        _number_skip_positions = _build_number_skip_positions(text)
        def _ordinal_repl(m):
            nonlocal num_changes
            match_start = m.start()
            if any(p in _number_skip_positions for p in range(match_start, m.end())):
                return m.group(0)
            num_changes += 1
            return _ordinal_to_words(m.group(1), m.group(2))
        text = _RE_ORDINAL.sub(_ordinal_repl, text)

        # 13b-pre. Elapsed h:m:s timestamps: "12:34:56" → "twelve hours,
        # thirty-four minutes, fifty-six seconds".  MUST run before the
        # clock-time rule, which would otherwise consume "12:34" and leave
        # a dangling ":56" ("twelve thirty-four:fifty-six").  Colon guards
        # keep MAC-like chains ("00:11:22:33") untouched.
        _number_skip_positions = _build_number_skip_positions(text)
        def _hms_repl(m):
            nonlocal num_changes
            if any(p in _number_skip_positions for p in range(m.start(), m.end())):
                return m.group(0)
            h, mi, se = int(m.group(1)), int(m.group(2)), int(m.group(3))
            parts = []
            if h:
                parts.append(
                    f'{_int_to_words(h)} hour{"" if h == 1 else "s"}')
            if mi:
                parts.append(
                    f'{_int_to_words(mi)} minute{"" if mi == 1 else "s"}')
            if se or not parts:
                parts.append(
                    f'{_int_to_words(se)} second{"" if se == 1 else "s"}')
            num_changes += 1
            return ', '.join(parts)
        text = _RE_HMS.sub(_hms_repl, text)

        # 13b. Time expressions: 5:30 PM, 12:00 AM, etc.
        def _time_repl(m):
            nonlocal num_changes
            num_changes += 1
            return _time_to_words(m.group(1), m.group(2), m.group(3) or '')
        text = _RE_TIME.sub(_time_repl, text)

        # 13b2. Ratios and aspect ratios: "16:9" → "16 to 9", "3:1" →
        # "3 to 1", "2.35:1" → "2.35 to 1".  The clock-time rule above
        # already consumed \d{1,2}:\d{2} times ("5:30" → "five thirty");
        # whatever digit:digit pairs remain are ratios, scores, or aspect
        # ratios — all naturally spoken "X to Y" (the bare colon was read
        # "sixteen colon nine" or skipped entirely).
        _number_skip_positions = _build_number_skip_positions(text)
        def _ratio_repl(m):
            nonlocal num_changes
            if any(p in _number_skip_positions for p in range(m.start(), m.end())):
                return m.group(0)
            num_changes += 1
            return f'{m.group(1)} to {m.group(2)}'
        text = _RE_RATIO.sub(_ratio_repl, text)

        # 13c. Fractions: 1/2, 3/4, etc. (before decimals, which match \d+.\d+)
        def _fraction_repl(m):
            nonlocal num_changes
            num_changes += 1
            return _fraction_to_words(m.group(1), m.group(2))
        text = _RE_FRACTION.sub(_fraction_repl, text)

        # Refresh skip positions — ordinals/time/ratios above changed text
        # length, so every stored position is now stale.
        _number_skip_positions = _build_number_skip_positions(text)

        # 13d0. Bare unit durations: "45s" → "45 seconds", "24h" → "24
        # hours", "2w" → "2 weeks".  Digit-anchored lowercase-only; "m" is
        # deliberately absent ("5m" = meters via the bare-meters rule), and
        # 4-digit values are skipped ("1990s"-style tokens belong to the
        # decade rule, which already ran; a skipped region like "802.11h"
        # stays untouched via the skip positions).
        _DURATION_UNITS = {
            's': ('seconds', 'second'),
            'h': ('hours', 'hour'),
            'w': ('weeks', 'week'),
        }

        def _duration_repl(m):
            nonlocal num_changes
            if any(p in _number_skip_positions for p in range(m.start(), m.end())):
                return m.group(0)
            val_clean = m.group(1).replace(',', '')
            unit = m.group(2)
            if unit == 's' and len(val_clean.split('.')[0]) == 4:
                return m.group(0)
            try:
                is_one = float(val_clean) == 1.0
            except ValueError:
                is_one = False
            plural, singular = _DURATION_UNITS[unit]
            num_changes += 1
            return f'{m.group(1)} {singular if is_one else plural}'
        text = _RE_BARE_DURATION.sub(_duration_repl, text)

        # 13d-pre. Multiplier shorthand: "3x faster" → "3 times faster",
        # "1.5x faster" → "1.5 times faster".  Digit×digit products were
        # already handled; this covers the bare "Nx" form.  MUST run
        # before the decimal rule (which would otherwise leave "zero
        # point five x" with an unspoken "x").  Version strings ("v1.2x",
        # "802.11x") are protected by skip positions and the (?<![\d.])
        # guard.
        _number_skip_positions = _build_number_skip_positions(text)
        def _multiplier_repl(m):
            nonlocal num_changes
            if any(p in _number_skip_positions for p in range(m.start(), m.end())):
                return m.group(0)
            num_changes += 1
            return f'{m.group(1)} times'
        text = re.sub(
            r'(?<![\d.])\b(\d{1,3}(?:\.\d+)?)\s*[xX]\b(?!\d)',
            _multiplier_repl, text)

        # 13d. Decimal numbers: 3.14, 0.99, etc.
        # (Skip decimals inside version/tech/model regions)
        _number_skip_positions = _build_number_skip_positions(text)
        def _decimal_repl(m):
            nonlocal num_changes
            match_start, match_end = m.start(1), m.end(1)
            if any(p in _number_skip_positions for p in range(match_start, match_end)):
                return m.group(0)
            num_changes += 1
            return _decimal_to_words(m.group(1))
        text = _RE_DECIMAL.sub(_decimal_repl, text)

        # 13e. Percent: "85 percent" → "eighty-five percent"
        # Now that numbers before % are already words, just clean up
        new_text, n = _RE_PERCENT.subn(' percent', text)
        if n:
            text = new_text

        # 13f. Digit/digit slash (remaining after fractions): e.g., in
        # technical contexts like "16/9 aspect ratio"
        new_text, n = _RE_SLASH_DIGITS.subn(r'\1 divided by \2', text)
        if n:
            num_changes += n
            text = new_text

        # 13i. Integers (with commas or standalone)
        _number_skip_positions = _build_number_skip_positions(text)
        def _int_repl(m):
            nonlocal num_changes
            # Skip if any digit in the match falls inside a version/tech region
            match_start, match_end = m.start(1), m.end(1)
            if any(p in _number_skip_positions for p in range(match_start, match_end)):
                return m.group(0)
            num_str = m.group(1).replace(',', '')
            try:
                n_val = int(num_str)
                # Skip single digits and very small numbers — TTS handles
                # 0-9 fine, and conversion of "2" to "two" can sound odd
                # in many contexts.  Convert 10+ and any comma-formatted.
                if n_val < 10 and ',' not in m.group(1):
                    return m.group(0)
                num_changes += 1
                # Read 4-digit numbers as years ONLY in year context
                # ("in 2024"); "The library has 1984 books" must stay a
                # cardinal number.
                if 1000 <= n_val <= 9999 and not _is_year_context(text, m.start(1)):
                    return _int_to_words(n_val)
                return _number_to_words(n_val)
            except (ValueError, OverflowError):
                return m.group(0)
        text = _RE_INTEGER.sub(_int_repl, text)

        if num_changes:
            stats.numbers_converted += num_changes

        # ── 14. Slash between words ────────────────────────────────
        # "client/server" → "client or server", "input/output" → "input or output".
        # "and/or" → "and or": group2 is literally "or", so the generic
        # "group1 or group2" join would duplicate it ("and or or").
        def _word_slash_repl(m):
            if m.group(2).lower() == 'or':
                return m.group(1) + ' or'
            if m.group(1).lower() == 'y' and m.group(2).lower() == 'o':
                return 'years old'
            return m.group(1) + ' or ' + m.group(2)
        new_text, n = _RE_WORD_SLASH.subn(_word_slash_repl, text)
        if n:
            stats.slashes_normalized += n
            text = new_text

        # ── 15. Parenthetical pauses ───────────────────────────────────
        # "(see Figure 1)" → ", see Figure 1,"
        # Skip empty parens "()" and numeric-only "(123)"
        def _counting_paren_repl(m):
            content = m.group(1).strip()
            if not content or content.isdigit():
                return m.group(0)
            # Math operands keep their parens: "5 divided by (x+1)" as
            # "5 divided by, x plus 1," reads like an aside; engines skip
            # the parens and say "five divided by x plus one".
            prefix = text[max(0, m.start() - 12):m.start()]
            if re.search(
                    r'(?:divided by|to the| over| of| given| between)\s*$',
                    prefix):
                return m.group(0)
            stats.parentheticals_normalized += 1
            return ', ' + content + ','

        if '(' in text:
            text = _RE_PARENTHETICAL.sub(_counting_paren_repl, text)

        # ── 16. Hyphenated compound words ──────────────────────────────
        # "well-known" → "well known", "state-of-the-art" → "state of the art"
        # Skip hyphenated numbers (twenty-one is fine), skip ranges (2020-2024)
        # Note: _NUMBER_WORDS and _KEEP_HYPHEN are now module-level frozensets
        # to avoid re-creation on every regex match.
        def _hyphen_repl(m):
            part1, part2 = m.group(1), m.group(2)
            # Don't split if either part is a single letter (e.g., "e-mail")
            if len(part1) <= 1 or len(part2) <= 1:
                return m.group(0)
            # Don't split number compounds: "twenty-one", "ninety-nine",
            # "twenty-first", "one-hundred", etc.
            if part1.lower() in _NUMBER_WORDS or part2.lower() in _NUMBER_WORDS:
                return m.group(0)
            # Don't split common hyphenated words that are better kept
            full = part1 + '-' + part2
            if full.lower() in _KEEP_HYPHEN:
                return m.group(0)
            stats.hyphens_normalized += 1
            return part1 + ' ' + part2

        if '-' in text:
            text = _RE_HYPHENATED_WORD.sub(_hyphen_repl, text)
            # Also handle multi-hyphen compounds: "state-of-the-art"
            # Replace ALL hyphens between alpha parts with spaces
            def _multi_hyphen_repl(m):
                full = m.group(0)
                # Check if any part is a number word
                parts = full.split('-')
                if any(p.lower() in _NUMBER_WORDS for p in parts):
                    return full
                if full.lower() in _KEEP_HYPHEN:
                    return full
                stats.hyphens_normalized += 1
                return ' '.join(parts)
            text = _RE_MULTI_HYPHEN.sub(_multi_hyphen_repl, text)

    # ── Final: Collapse whitespace ────────────────────────────────────────
    text = re.sub(r'[^\S\n]+', ' ', text)
    # (the old second `` +`` substitution was unreachable redundancy —
    # ``[^\S\n]+`` already collapses space runs)
    # Clean up double commas from parenthetical conversion: ", , " → ","
    text = re.sub(r',\s*,', ',', text)
    # Clean up ", ." → "."
    text = re.sub(r',\s*\.', '.', text)
    # Glyph expansions insert spaced words ("x to the n .") — pull a stray
    # space back before the closing punctuation ("n ." → "n.")
    text = re.sub(r' +([.,;:!?])', r'\1', text)
    text = text.strip()

    # ── Log debug stats if changes were made ────────────────────────────
    if stats.total_changes > 0:
        logger.debug(f"[Preprocess] {stats}")

    return text, stats


# ═══════════════════════════════════════════════════════════════════════════════
#  MODULE-LEVEL CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_chunker_for_language(
    language: str, max_chars: int = 500, overlap_chars: int = 0,
    chunk_strategy: str = "sentence", max_duration_seconds: float = 0.0,
) -> TextChunker:
    """Factory: create a ``TextChunker`` configured for *language*.

    Args:
        language: Target TTS language.
        max_chars: Maximum characters per chunk.
        overlap_chars: Overlap at chunk boundaries.
        chunk_strategy: One of "sentence", "duration", "paragraph",
            "semantic", "hybrid".
        max_duration_seconds: Max estimated speaking duration per chunk
            (0 = disabled, used by "duration" and "hybrid" strategies).
    """
    config = ChunkingConfig(
        max_chars=max_chars,
        max_sentences=5 if language not in CJK_LANGUAGES else 3,
        # Clamp min_chunk_chars so small max_chars values don't raise
        # (ChunkingConfig requires min < max).
        min_chunk_chars=min(50, max(0, max_chars - 1)),
        overlap_chars=overlap_chars,
        respect_sentence_bounds=True,
        language=language,
        chunk_strategy=chunk_strategy,
        max_duration_seconds=max_duration_seconds,
    )
    return TextChunker(config)


def chunk_text_for_tts(
    text: str, language: str = "English", max_chars: int = 500,
    chunk_strategy: str = "sentence", max_duration_seconds: float = 0.0,
) -> List[str]:
    """Split text into chunks, returning only the text strings (no durations).

    Args:
        text: The text to chunk.
        language: Target TTS language.
        max_chars: Maximum characters per chunk.
        chunk_strategy: One of "sentence", "duration", "paragraph",
            "semantic", "hybrid".
        max_duration_seconds: Max estimated speaking duration per chunk.
    """
    chunker = get_chunker_for_language(
        language, max_chars,
        chunk_strategy=chunk_strategy,
        max_duration_seconds=max_duration_seconds,
    )
    return [chunk for chunk, _ in chunker.chunk_text(text, language, preprocess=True)]