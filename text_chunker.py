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
  - Expanded abbreviation dictionary (tech, Latin, business)
  - Enhanced symbol normalization (comparison operators, currency
    symbols, hash, at-sign, tilde, standalone plus/minus)
  - Multiple punctuation normalization (!!! → !, ??? → ?)
  - Improved slash handling (word/word → word or word)
  - Parenthetical pause insertion and hyphen normalization
  - Time expression normalization (5:30 PM → five thirty PM)
"""

from __future__ import annotations

import functools
import html as _html_mod
import logging
import re
from dataclasses import dataclass, field
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
    r'|\b\S+\.\S+\.(?:' + _KNOWN_TLDS + r')\b'  # domain-like: x.y.tld
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
    (0xAC00, 0xD7AF),   # Hangul Syllables
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

# Version strings: v1.2, Version 3.5, iOS 17, Python 3.12
_RE_VERSION_STRING = re.compile(
    r'\b(?:v|version|ver)\.?\s*\d[\d.]*(?:[a-zA-Z]\d?)?\b'
    r'|\b(?:iOS|iPadOS|macOS|watchOS|tvOS|Android|Windows|Linux|'
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
_RE_TECH_SPEC = re.compile(
    r'\b\d+[pPkKMGT](?:Hz|bps)?\b'
    r'|\bH\.?\d{3}(?:\.\d+)?\b'
    r'|\b\d{3,4}(?:\.\d+)?(?:-[a-zA-Z]{2,4}){1,2}\b'
    r'|\b(?:USB|HDMI|PCIe|DDR|LPDDR|NVMe|SATA|M\.?2|Wi[\s-]?Fi|'
    r'Bluetooth|5G|4G|3G|LTE|OLED|LCD|LED|AMOLED)\s*\d*\b',
    re.IGNORECASE,
)

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

# HTML tags
_RE_HTML_TAG = re.compile(r'<[^>]+>')

# Markdown: images ![alt](url) — remove entirely
_RE_MD_IMAGE = re.compile(r'!\[([^\]]*)\]\([^)]*\)')

# Markdown: links [text](url) → keep text
_RE_MD_LINK = re.compile(r'\[([^\]]+)\]\([^)]*\)')

# Markdown: bold **text** or __text__
_RE_MD_BOLD = re.compile(r'\*\*(.+?)\*\*|__(.+?)__')

# Markdown: italic *text* or _text_
_RE_MD_ITALIC = re.compile(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)|(?<!_)_(?!_)(.+?)(?<!_)_(?!_)')

# Markdown: inline code `text`
_RE_MD_CODE = re.compile(r'`([^`]+)`')

# Markdown: headings (## Heading → remove ## prefix)
_RE_MD_HEADING = re.compile(r'^(#{1,6})\s+(.*)$', re.MULTILINE)

# Bullet / numbered list markers
_RE_LIST_MARKER = re.compile(r'^[\s]*(?:[-*+•]|\d+[.)])\s+', re.MULTILINE)

# Unicode normalization: smart quotes
_RE_SMART_QUOTES = re.compile(r'[""\u2018\u2019\u201a\u201b]')

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

# URLs → "link to [domain]"
_RE_URL = re.compile(r'https?://(?:www\.)?([^/\s]+)')

# Emails (Latin only) — user@host.com → "user at host dot com"
_RE_EMAIL = re.compile(r'\b([A-Za-z0-9._%+-]+)@([A-Za-z0-9.-]+)\.([A-Za-z]{2,})\b')

# Abbreviations to expand for speech (Latin only)
_ABBREVIATION_EXPANSIONS: Dict[re.Pattern, str] = {}
for _abbr, _exp in [
    # ── Common Latin abbreviations ──────────────────────────────────────────
    (r'\be\.g\.', 'for example'),
    (r'\bi\.e\.', 'that is'),
    (r'\betc\.?', 'et cetera'),
    (r'\bvs\.?', 'versus'),
    (r'\bcf\.?', 'compare with'),
    (r'\bca\.?(?=\s|$)', 'circa'),
    (r'\bq\.?\s*v\.?', 'quod vide'),
    (r'\bn\.b\.?', 'note well'),
    (r'\bad\s+hoc\b', 'ad hoc'),
    (r'\bde\s+facto\b', 'de facto'),
    (r'\bper\s+se\b', 'per se'),
    (r'\bvice\s+versa\b', 'vice versa'),
    (r'\ba\s*priori\b', 'a priori'),
    (r'\ba\s*posteriori\b', 'a posteriori'),
    (r'\bet\s+al\.?', 'and others'),
    (r'\bibid\.?', 'ibidem'),
    (r'\bop\.?\s*cit\.?', 'opere citato'),
    (r'\bloc\.?\s*cit\.?', 'loco citato'),
    (r'\bsic\b', 'sic'),
    (r'\bstatus\s+quo\b', 'status quo'),
    (r'\bipso\s+facto\b', 'ipso facto'),
    # ── General abbreviations ──────────────────────────────────────────────
    (r'\bapprox\.?(?=\s|$)', 'approximately'),
    (r'\bavg\.?', 'average'),
    (r'\bmin\.?(?=\s|$)', 'minimum'),
    (r'\bmax\.?(?=\s|$)', 'maximum'),
    (r'\binfo\.?', 'information'),
    (r'\bstats\.?', 'statistics'),
    (r'\btech\.?(?=\s|$)', 'technology'),
    (r'\brep\.?(?=\s|$)', 'repetitions'),
    (r'\bsec\.?(?=\s|$)', 'seconds'),
    (r'\bhr\.?(?=\s|$)', 'hour'),
    (r'\bhrs\.?', 'hours'),
    (r'\bfig\.?(?=\s|$)', 'figure'),
    (r'\bvol\.?(?=\s|$)', 'volume'),
    (r'\bno\.?(?=\s|$)', 'number'),
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
    (r'\bNASA\b', 'NASA'),
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
    (r'\bNATO\b', 'NATO'),
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
    (r'\bPh\.?D\.?', 'doctor of philosophy'),
    (r'\bDr\.?', 'doctor'),
    (r'\bProf\.?', 'professor'),
    (r'\bSr\.?', 'senior'),
    (r'\bJr\.?', 'junior'),
    (r'\bMs\.?', 'miss'),
    (r'\bMrs\.?', 'misses'),
    (r'\bMr\.?', 'mister'),
    (r'\bRev\.?(?=\s|$)', 'reverend'),
    (r'\bHon\.?(?=\s|$)', 'honorable'),
    (r'\bCapt\.?(?=\s|$)', 'captain'),
    (r'\bLt\.?(?=\s|$)', 'lieutenant'),
    (r'\bCol\.?(?=\s|$)', 'colonel'),
    (r'\bGen\.?(?=\s|$)', 'general'),
    (r'\bSgt\.?(?=\s|$)', 'sergeant'),
    (r'\bSt\.?(?=\s|$)', 'saint'),
    # ── Misc common abbreviations ─────────────────────────────────────────
    (r'\bmisc\.?', 'miscellaneous'),
    (r'\badmin\.?', 'administration'),
    (r'\bauth\.?(?=\s|$)', 'authentication'),
    (r'\bconfig\.?(?=\s|$)', 'configuration'),
    (r'\bdir\.?(?=\s|$)', 'directory'),
    (r'\bdoc\.?(?=\s|$)', 'document'),
    (r'\bdocs\.?', 'documents'),
    (r'\blib\.?(?=\s|$)', 'library'),
    (r'\bmgr\.?(?=\s|$)', 'manager'),
    (r'\bmgmt\.?', 'management'),
    (r'\bdept\.?(?=\s|$)', 'department'),
    (r'\bdiv\.?(?=\s|$)', 'division'),
    (r'\basst\.?(?=\s|$)', 'assistant'),
    (r'\bassoc\.?(?=\s|$)', 'associate'),
    (r'\bacct\.?(?=\s|$)', 'account'),
    (r'\brecon\.?(?=\s|$)', 'reconnaissance'),
    (r'\bpwd\.?(?=\s|$)', 'password'),
    (r'\bmsg\.?(?=\s|$)', 'message'),
    (r'\bmsgs\.?', 'messages'),
    (r'\bbtn\.?(?=\s|$)', 'button'),
    (r'\bchk\.?(?=\s|$)', 'check'),
    (r'\bcalc\.?(?=\s|$)', 'calculator'),
    (r'\bdiag\.?(?=\s|$)', 'diagram'),
    (r'\bnum\.?(?=\s|$)', 'number'),
    (r'\bpct\.?(?=\s|$)', 'percent'),
    (r'\bamt\.?(?=\s|$)', 'amount'),
    (r'\best\.?(?=\s|$)', 'estimate'),
    (r'\breq\.?(?=\s|$)', 'requirement'),
    (r'\breqd\.?', 'required'),
    (r'\bops\.?(?=\s|$)', 'operations'),
    (r'\bdev\.?(?=\s|$)', 'development'),
    (r'\bdevs\.?', 'developers'),
    (r'\bprod\.?(?=\s|$)', 'production'),
    (r'\brec\.?(?=\s|$)', 'recommendation'),
    (r'\binc\.?(?=\s|$)', 'incorporated'),
    (r'\bltd\.?(?=\s|$)', 'limited'),
    (r'\bcorp\.?(?=\s|$)', 'corporation'),
    (r'\bco\.?(?=\s|$)', 'company'),
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
    (r'\bretvals\.?', 'return values'),
]:
    _ABBREVIATION_EXPANSIONS[re.compile(_abbr, re.IGNORECASE)] = _exp

# Symbols to normalize (Latin only)
_RE_AMPERSAND = re.compile(r'\s*&\s*')
_RE_SLASH_DIGITS = re.compile(r'(\d+)\s*/\s*(\d+)')
_RE_PLUS_EQUALS = re.compile(r'\+=')
_RE_EQUALS = re.compile(r'(?<=\s)=|=(?=\s)')
_RE_PERCENT = re.compile(r'%')
_RE_DEGREE = re.compile(r'°')
_RE_MULTIPLY_DIGITS = re.compile(r'(\d+)\s*[x×]\s*(\d+)')

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
_RE_CURRENCY_DOLLAR = re.compile(r'\$(\d[\d,]*\.?\d*)')
_RE_CURRENCY_EURO = re.compile(r'€(\d[\d,]*\.?\d*)')
_RE_CURRENCY_POUND = re.compile(r'£(\d[\d,]*\.?\d*)')
_RE_CURRENCY_YEN = re.compile(r'¥([\d,]+)')

# ─── Number patterns (ordered: most specific first) ─────────────────────────
# Ordinal suffixes: 1st, 2nd, 3rd, 4th, 11th, 12th, 13th, 21st, 22nd, etc.
_RE_ORDINAL = re.compile(
    r'\b(0*[1-9]\d{0,2})(st|nd|rd|th)\b', re.IGNORECASE
)
# Time: 5:30, 5:30 PM, 5:30pm, 12:00 AM, etc.
_RE_TIME = re.compile(
    r'\b(\d{1,2}):(\d{2})(?:\s*([AaPp]\.?[Mm]\.?))?\b'
)
# Decimal number: 3.14, 0.99, .5 (not preceded by digit — that's already matched)
_RE_DECIMAL = re.compile(r'(?<![A-Za-z])(\d+\.\d+)(?![A-Za-z])')
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


# ═══════════════════════════════════════════════════════════════════════════════
#  NUMBER-TO-WORDS ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

_ONES = [
    '', 'one', 'two', 'three', 'four', 'five', 'six', 'seven',
    'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen',
    'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen',
]
_TENS = [
    '', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy',
    'eighty', 'ninety',
]
_THOUSANDS = [
    ('', ''), ('thousand', 'thousand'), ('million', 'million'),
    ('billion', 'billion'), ('trillion', 'trillion'),
]
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


def _int_to_words(n: int) -> str:
    """Convert a non-negative integer to English words.

    Handles numbers from 0 up to 999,999,999,999,999.
    Examples: 0 → 'zero', 7 → 'seven', 42 → 'forty-two',
    1234 → 'one thousand two hundred thirty-four'.
    """
    if n == 0:
        return 'zero'
    if n < 0:
        return 'negative ' + _int_to_words(-n)

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
    """Convert integer to words, with year-aware handling for 4-digit numbers."""
    if 1000 <= n <= 9999:
        # Year-style: 2024 → "twenty twenty-four"
        hi, lo = n // 100, n % 100
        if lo == 0:
            return _int_to_words(n)  # 2000 → "two thousand"
        hi_words = _int_to_words(hi)
        lo_words = _int_to_words(lo)
        return hi_words + ' ' + lo_words
    return _int_to_words(n)


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
                words += ' ' + _ONES[int(digit)]
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


def _currency_to_words(amount_str: str, symbol: str) -> str:
    """Convert currency like '$50' or '€1,000.50' to words.

    '$50' → 'fifty dollars'
    '$1.50' → 'one dollar fifty cents'
    '€1,000' → 'one thousand euros'
    """
    # Clean the amount string
    amount_clean = amount_str.replace(',', '')
    currency_name = {'$': 'dollar', '€': 'euro', '£': 'pound', '¥': 'yen'}.get(symbol, 'dollar')

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
            raise ValueError(
                f"min_chunk_chars ({self.min_chunk_chars}) must be < "
                f"max_chars ({self.max_chars})"
            )
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
        # Clamp duration to reasonable range
        if self.max_duration_seconds < 0:
            logger.warning(
                f"[Chunker] max_duration_seconds must be >= 0; setting to 0 (disabled)"
            )
            object.__setattr__(self, 'max_duration_seconds', 0.0)

    @property
    def effective_pronunciation_overrides(self) -> Dict[re.Pattern, str]:
        """Return pronunciation overrides — user-provided or built-in defaults."""
        if self.pronunciation_overrides is not None:
            return self.pronunciation_overrides
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

        if len(text) <= self.config.max_chars:
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
            base_duration = (cjk_count / 1.5 / _CJK_SPEAKING_RATE) * 60
        else:
            # Mixed: estimate each part with its appropriate rate
            cjk_duration = (cjk_count / 1.5 / _CJK_SPEAKING_RATE) * 60
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
        text = re.sub(r' +', ' ', text)
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

        # Mark URL regions
        for m in _URL_PATTERN.finditer(text):
            protected.update(range(m.start(), m.end()))

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

        preceding = text[:punct_start].rstrip()
        if not preceding:
            return True

        # ── Decimal numbers: "3.14" — digit precedes the period ─────────
        if preceding[-1].isdigit():
            return False

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
                return False

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
        """Group sentences into chunks respecting paragraph breaks.

        Tries to keep each paragraph intact.  If a paragraph exceeds
        *max_chars*, it falls back to sentence-based sub-chunking.
        """
        # Re-join sentences with spaces, split by double-newline (paragraph)
        full_text = ' '.join(s for s in sentences if s.strip())
        paragraphs = [p.strip() for p in _PARAGRAPH_PATTERN.split(full_text) if p.strip()]

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
                if len(combined) <= self.config.max_chars * _MERGE_TOLERANCE:
                    chunks[-1] = (combined, self._chunk_duration(combined, language))
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

            # Find the LAST space to preserve maximum overlap while
            # still breaking at a clean word boundary.
            clean_break = overlap_text.rfind(' ')
            if clean_break > 0:
                overlap_text = overlap_text[clean_break + 1:]
            else:
                # No space found — try last comma or semicolon
                clean_break = max(
                    overlap_text.rfind(','),
                    overlap_text.rfind(';'),
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
                clean_break = shorter_overlap.rfind(' ')
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
                matches = list(_PAUSE_PATTERN.finditer(search_region))
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
    new_text, n = _RE_SMART_QUOTES.subn('"', text)
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
        new_text, n = _RE_URL.subn(r'link to \1', text)
        if n:
            stats.urls_replaced += n
            text = new_text

        # ── 10. Replace emails ──────────────────────────────────────────
        new_text, n = _RE_EMAIL.subn(r'\1 at \2 dot \3', text)
        if n:
            stats.emails_replaced += n
            text = new_text

        # ── 11. Expand abbreviations ───────────────────────────────────
        abbr_changes = 0
        for pattern, expansion in _ABBREVIATION_EXPANSIONS.items():
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
            for ov_pattern, ov_spoken in _overrides.items():
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
        ]:
            new_text, n = sym_re.subn(
                lambda m, s=sym_ch: _currency_to_words(m.group(1), s),
                text,
            )
            if n:
                sym_changes += n
                text = new_text
        # Hash number: #123 → "number 123"
        new_text, n = _RE_HASH_NUMBER.subn(r'number \1', text)
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
        new_text, n = _RE_DEGREE.subn(' degrees', text)
        if n:
            sym_changes += n
            text = new_text
        new_text, n = _RE_MULTIPLY_DIGITS.subn(r'\1 times \2', text)
        if n:
            sym_changes += n
            text = new_text
        # Note: % and digit/digit / are handled after number conversion
        if sym_changes:
            stats.symbols_normalized += sym_changes

        # ── 13. Convert numbers to words ─────────────────────────────────
        # Order matters: ordinals → time → fractions → decimals → integers
        num_changes = 0

        # 13-prefix. Build skip positions for version/model/tech regions
        _number_skip_positions: set = set()
        for skip_re in (_RE_VERSION_STRING, _RE_MODEL_IDENTIFIER, _RE_TECH_SPEC):
            for skip_m in skip_re.finditer(text):
                for ci in range(skip_m.start(), skip_m.end()):
                    if text[ci].isdigit() or text[ci] == '.':
                        _number_skip_positions.add(ci)

        # 13a. Ordinal suffixes: 1st, 2nd, 3rd, 21st, etc.
        # (Skip ordinals inside version/tech/model regions)
        def _ordinal_repl(m):
            nonlocal num_changes
            match_start = m.start()
            if any(p in _number_skip_positions for p in range(match_start, m.end())):
                return m.group(0)
            num_changes += 1
            return _ordinal_to_words(m.group(1), m.group(2))
        text = _RE_ORDINAL.sub(_ordinal_repl, text)

        # 13b. Time expressions: 5:30 PM, 12:00 AM, etc.
        def _time_repl(m):
            nonlocal num_changes
            num_changes += 1
            return _time_to_words(m.group(1), m.group(2), m.group(3) or '')
        text = _RE_TIME.sub(_time_repl, text)

        # 13c. Fractions: 1/2, 3/4, etc. (before decimals, which match \d+.\d+)
        def _fraction_repl(m):
            nonlocal num_changes
            num_changes += 1
            return _fraction_to_words(m.group(1), m.group(2))
        text = _RE_FRACTION.sub(_fraction_repl, text)

        # 13d. Decimal numbers: 3.14, 0.99, etc.
        # (Skip decimals inside version/tech/model regions)
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

        # 13h. Integers (with commas or standalone)
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
                return _number_to_words(n_val)
            except (ValueError, OverflowError):
                return m.group(0)
        text = _RE_INTEGER.sub(_int_repl, text)

        if num_changes:
            stats.numbers_converted += num_changes

        # ── 14. Slash between words ────────────────────────────────────
        # "client/server" → "client or server", "input/output" → "input or output"
        # Skip "and/or" → "and or" (not "and or or")
        new_text, n = _RE_WORD_SLASH.subn(
            lambda m: (
                m.group(1) + ' or ' + m.group(2)
                if m.group(1).lower() != 'and'
                else m.group(0)
            ),
            text,
        )
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
    text = re.sub(r' +', ' ', text)
    # Clean up double commas from parenthetical conversion: ", , " → ","
    text = re.sub(r',\s*,', ',', text)
    # Clean up ", ." → "."
    text = re.sub(r',\s*\.', '.', text)
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
        min_chunk_chars=50,
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