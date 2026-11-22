# AI Slideshow Renderer — Improvement Change Summary

Drop-in replacement files for all 7 source files, verified with a 648-test pytest suite (all passing). Original files are untouched in `upload/`; the fixed versions in this folder have the same names — copy them over your existing files.

| File | Original lines | Fixed lines | Changed lines |
|---|---:|---:|---:|
| app.py | 138 | 146 | 7 |
| config.py | 338 | 414 | 80 |
| dialogs.py | 3,999 | 4,129 | 274 |
| engines.py | 3,531 | 3,723 | 400 |
| gui.py | 2,023 | 2,171 | 392 |
| text_chunker.py | 2,890 | 5,754 | 2,911 |
| utils.py | 2,505 | 2,680 | 373 |

Compatibility: same module names, same public API, same message/queue protocol. No new third-party dependencies.

---

## 1. Critical system-level bugs

### 1.1 Progress reporting was completely broken
`str(RenderPhase.X)` returns `"RenderPhase.X"`, which never matched the phase-name lookup maps on the GUI side, so phase labels/progress stages silently fell through to defaults. All senders and lookups now use `RenderPhase.X.value`.

### 1.2 Cleanup on quit never ran
`_on_about_to_quit()` checked the `_shutting_down` flag that `closeEvent` had just set and returned immediately — so worker pools, temp dirs and model handles were never released on exit. Replaced with a dedicated `_cleanup_done` idempotency flag; both paths now share one cleanup routine.

### 1.3 Editor data loss: dirty buffers written to the wrong slide
Deleting or moving a slide in the editor flushed *pending* text-buffer edits after the row indices had changed, writing slide A's edits into slide B's file. Added `_flush_before_structure_change()`, called before every delete/move/reorder so buffers are flushed while indices are still valid.

### 1.4 Per-project resolution override leaked
The per-project resolution was written to a local copy of the config while the cache-hash was computed from `ctx.config`, so a project rendered once with an override stayed on the old resolution forever (cache always matched). The override is now applied before hashing, consistently.

## 2. Rendering pipeline bugs

### 2.1 JPEG thumbnails always failed
`with Image.open(...)` closed the file before pixel access, so JPEG slides crashed thumbnail generation and fell to the (slow) Playwright path every time. New helper `_load_slide_image_rgb()` loads and converts in one go; also fixed two spots that skipped the Playwright fallback on the wrong condition.

### 2.2 Multi-block audio chain failed at 24 kHz
Silence for dead blocks was generated at 44.1 kHz while the crossfade chain ran at 24 kHz — the whole audio concatenation aborted for any project with a dead block. Silence is now generated at the chain's sample rate with an explicit warning log.

### 2.3 Drift across slides
Slide gaps used `acrossfade`, which overlaps audio with video-boundary misalignment and drifts cumulatively. Replaced with explicit silent-segment concatenation, keeping every slide's audio exactly aligned to its video.

### 2.4 Mux failure produced a permanently "successful" silent video
When audio muxing failed, the code saved a video without audio *and* wrote full success hashes — the next run cache-skipped it and the user was stuck with silent output forever. Now emits `ErrorMessage` to the GUI and records `error` status in the manifest; mux timeout falls back to no-audio without aborting the whole render.

### 2.5 Backend config leaked between renders
The persistent worker mutated `backend.config` in place and never restored it, so settings from one render bled into the next. Wrapped in `try/finally` restoration.

### 2.6 Scheduler could not actually parallelize
Unbounded task queues meant every task landed on worker #1; a hard-killed worker (SIGKILL/OOM) left pool slots occupied forever. Now: `maxsize=1` per-worker queue so `put_nowait` genuinely picks idle workers, failed tasks are re-queued instead of dropped, a pool-slot watchdog finalizes orphaned slots, and retiring workers are removed from the dispatch ring.

### 2.7 Crossfade banding / precision
Fade math on uint8 with NumPy 2 (NEP 50) silently overflowed/banded. Crossfades now run in float32 before converting back.

### 2.8 nvenc detection
Encoder availability is now probed with a real 1-frame encode instead of trusting the binary's presence.

## 3. TTS text corruption (11 fixes, each with a regression test)

Text preprocessing mangled spoken content in these ways — all fixed in `text_chunker.py` / `utils.py`:

1. `$5.` — the sentence boundary regex swallowed the period after abbreviations/amounts.
2. `no` was rewritten to `number` (over-eager "No." expansion).
3. `200 ms` became `200 miss` ("ms" treated as a name).
4. Curly quotes/apostrophes were inconsistently transformed; normalization is now uniform.
5. `a < 10` was treated as an HTML tag and deleted entirely; angle-bracket stripping now only removes real tags.
6. `snake_case_words` was eaten by the italic regex (underscores treated as Markdown emphasis).
7. "Windows 11" was spoken as "eleven" — skip-list positions went stale after earlier substitutions; positions are recomputed.
8. `802.11` and similar version numbers were read as section numbers.
9. Dates were misparsed ("2024-01-15" spoken oddly).
10. URL paths left fragments like "slash docs slash page" in speech.
11. Years in parentheses were misread; year-context handling is now explicit.

### 3.1 Catastrophic prefix-eating bug (found during the pronunciation pass)
Several abbreviation patterns lacked a trailing word boundary, so `\bDr\.?` matched the `Dr` **prefix** of any word: "dream" → "doctoream", "dress" → "doctoress", "drive" → "doctorive", "profits" → "professorits", "etched" → "et ceterahed", "CFO" → "compare withO", "administrator" → "administrationistrator", "src" → "seniorc". All unguarded patterns (`Dr`, `Prof`, `etc`, `vs`, `cf`, `info`, `Mr(s)`, `admin`, …) now carry a `\b…\b` guard. This is arguably the single worst text-corruption bug in the codebase — these are extremely common words.

## 3b. Pronunciation improvements (pass 1)

The TTS preprocessing layer now speaks words correctly instead of reading letters. All rules are digit-anchored or context-guarded, and CJK text is untouched.

**Measurements** (read as words, not letter soup):
- `5 kg` → "five kilograms", `10 km`, `30 MB` → "megabytes", `2.4 GHz` → "gigahertz", `10 Mbps` → "megabits per second", `200 ms` → "two hundred milliseconds", `5 min` → "five minutes" (previously "five **minimum**"), `80 km/h`, `9 m/s`, `5 µs` → "microseconds"
- Byte vs bit distinction preserved (`MB` bytes, `Mb` bits); exact `1` gets singular form (`1 lb` → "one pound")

**Degrees**: `20°C` → "twenty degrees Celsius", `70 °F` → Fahrenheit, `300 °K` → Kelvin (previously "twenty degrees**C**"). Bare `90°` still → "degrees".

**Currency & magnitude**: `$5K` → "five thousand dollars" (previously "five dollars**K**"), `$1.5B` → "one point five billion dollars" (previously "one dollar fifty cents**B**"), `€3M`; bare `10K users` → "ten thousand users", `5M` → "five million". Display resolutions `2K/4K/8K`, gold `24K` and retirement `401K` are left for the engine to read naturally.

**Roman numerals in context**: `Chapter IV` → "Chapter four", `World War II`/`WWII` → "World War Two", `King Henry VIII` → "Henry the eighth", `the XIX century` → "nineteenth century", `Title IX` → "Title nine". Unanchored tokens (`MIX`, clothing `XL`) are untouched.

**Tech tokens**: `C++` → "C plus plus" and `C#`/`F#` → "… sharp" (previously silent symbols), `Node.js` → "node jay ess" (previously "Node.**javascript**"), `.NET` → "dot net", plus ~40 new letter-spoken acronyms (XML, TCP, VPN, FBI, DNA, ADHD, AWS, B2B → "B to B", H2O, CO2, …).

**Idioms & symbols**: `24/7` → "twenty-four seven" (previously "twenty-four **sevenths**"), `10-20` → "ten to twenty", `Jan 15` → "January 15", `w/` → "with", `w/o` → "without" (previously "w or the team"), `3x faster` → "three times faster", `A -> B` and `A → B` → "A to B" (previously "A minus is greater than B"), `±` → "plus or minus", `≈`, `≠`, `÷` spoken; `½` → "one half", `m²` → "square meters", `E=mc²` → "m c squared"; `© ® ™ † ‡ •` are now removed instead of stuttering.

**Title corrections**: `Ms.` is spoken "Miz" (was "miss"); `St.` is "Saint Louis" but "Main **street**" via a capitalized-following-word heuristic.

**User-configurable pronunciation dictionary** (new `pronunciation_overrides` setting in `settings.json`):
```json
"pronunciation_overrides": {"\\bGLM\\b": "G L M", "Zhipu": "Jee-poo"}
```
Entries are case-insensitive regexes applied after the built-in expansions (user entries win). Previously a user override dict **replaced** all built-ins; it now merges. Invalid regexes are dropped with a warning at load time instead of crashing the render worker; non-dict/non-string values are sanitized in `config.py`.

## 3c. Pronunciation improvements (pass 2 — word reading quality)

A second pass focused on how individual words and notations are *spoken*. All new rules are inside the Latin-only branch (CJK output is byte-identical) and every item has regression tests.

**Corruption bugs fixed on the way** (spoken output was wrong, not just ugly):
- `2^10` was read as "two to the **ten**" — the integer rule consumed the exponent. Caret exponents are now converted before number conversion: `2^10` → "two to the tenth", `x^2` → "x squared", `e^x` → "e to the x", `10^-6` → "ten to the minus sixth".
- `10^6 kg` let the unit rule anchor on the exposed `6 kg` — caret conversion runs after unit anchoring, so it becomes "ten to the sixth kilograms".
- `1 !== 2` was read "is not equal to **equals**" (`!=` fired inside `!==`); `a == b` was "a equals equals b"; `a && b` was "a and and b". `===`/`!==`/`==`/`&&`/`||` are now spoken correctly.
- `80 MB/s` became "eighty megabytes **or s**" (the unit rule ate `MB`, the slash rule read `/s`); bare `km/h` was "km or h". Data-rate compounds (`KB/s`…`Tb/s`) exist in digit-anchored **and** bare forms.
- `m/s²` was spoken "square meters per second" — that is m²/s, a different quantity. m/s² → "meters per second squared"; m²/s and m³/s also handled. (Subtle: `²` counts as a word character in Python regexes, so these patterns must run *before* the generic ²/³ rules.)
- `no.` at the end of a rhetorical answer ("…or no. Next") became "number Next" — the expansion now requires digits after the dot (`No. 5` still → "number 5").

**Exponent & chemical notation** (previously unread glyphs): `10⁶` → "ten to the sixth", `2⁴⁵` → "to the forty-fifth", `10⁻⁶` → "to the minus sixth", `xⁿ` → "x to the n"; subscripts `CO₂` → "CO two", `C₁₂H₂₂O₁₁` → "C one two H two two O one one".

**Math glyphs**: `√2` → "square root of 2", `π` → "pi", `ΔT` → "delta T", `50 Ω` → "fifty ohms" (also kΩ/MΩ/mΩ), `∫` → "integral of", `№` → "number", `℃`/`℉` → "degrees Celsius/Fahrenheit".

**Decades & dates**: `1990s` → "nineteen nineties", `1920s` → "nineteen twenties", `1900s` → "nineteen hundreds", `2000s` → "two thousands", `the 90s` → "the nineties" (the old integer pattern could not even match across the trailing `s`); `January 15` → "January fifteenth", `March 3` → "March third" (ordinal days; `January 15th` still handled once); `9/11` → "September eleventh" (was "nine elevenths"); `between 2020` now reads year-style.

**Dimension products**: `1920×1080` → "nineteen twenty **by** ten eighty" (was "one thousand nine hundred twenty times one thousand eighty"); small operands keep the arithmetic reading (`3×3` → "3 times 3").

**Case-sensitive acronym tier** (new) — the lowercase form is an ordinary word and is never touched: `US` → "U S" ("helps **us**" stays), `IT` → "I T" ("do **it**" stays), `ID` → "I D", `DC`, `GIF` → "jiff"/`GIFs` → "jiffs", `IEEE` → "I triple E", `ASCII` → "ask ee", `ISO`/`RFC`/`UTF`/`TTS`/`SSML`/`ACM`/`USD`/`LLVM`/`GCC` letter-spoken, `MBA`/`GPA`/`ETA` letter-spoken, `Ctrl`/`Cmd`/`Esc` → "control/command/escape", `P.S.` → "postscript" (while `Ps. 23` the Psalms stays intact).

**Tech product names** (engines mangle these): `LaTeX` → "lah tek", `TeX` → "tek", `Nginx` → "engine x", `MySQL` → "my sequel", `PostgreSQL` → "postgres Q L", `SQLite` → "sequel lite", `MongoDB` → "mongo D B", `NumPy` → "num pie", `SciPy` → "sigh pie", `PyPI` → "pie pie eye", `Vue` → "view", `GNU` → "guh new".

**Numeronyms & shorthand**: `K8s` → "kubernetes", `i18n` → "internationalization", `l10n` → "localization", `a11y` → "accessibility"; `N/A` → "not applicable", `I/O` → "input or output", `A/B testing` → "A B testing", `TBD`/`TBA`, `WW1`/`WW2`, `viz.` → "namely", dot-less `e.g`/`i.e`, `ok` → "okay".

**More units** (digit-anchored): `60 fps` → "frames per second", `3500 rpm`, `300 dpi`, `1920px` → "pixels", `10 ns` → "nanoseconds", `5000 mAh` → "milliamp hours", `5 GiB` → "gibibytes" (IEC binary prefixes), `3 ppb`; plain-digit powers `25 m3` → "cubic meters", `12 m2` → "square meters"; real-estate style `45 sq ft` → "square feet", `30 sq m`, `4 cu ft`; trailing currency `50$` → "fifty dollars"; exact `1` now also singularizes superscript measures (`1 km²` → "one square kilometer").

**Identifier readability**: `my_var_name` → "my var name" (underscores become spaces; the earlier fix only stopped the italic rule from *deleting* them), `getUserData` → "get User Data", `parseHTMLString` → "parse HTML String" (then the HTML rule spells it). A keep-list preserves mixed-case brands (`iPhone`, `iPad`, `eBay`, `jQuery`, `GitHub`, …) and unit symbols (`pH`, `dB`, `mL`, `kW`, `GHz`, `mmHg`, …).

**Cosmetics**: stray spaces before closing punctuation produced by glyph expansion ("x to the n .") are pulled back ("x to the n."); removed the last two pyflakes findings (unused `field` import, placeholder-less f-string).

User `pronunciation_overrides` still win for any term that survives the built-in tables (the built-in expansions run first; a term already rewritten — e.g. `MySQL` → "my sequel" — can no longer be re-matched, so overrides are best used for custom words).

## 3d. Pronunciation improvements (pass 3 — digit-by-digit readings, brands, respellings)

A third pass targeting what engines demonstrably garble. All rules are Latin-only, digit-anchored or context-guarded, and each has regression tests. Three of the fixes below are *corruption* bugs, not just style.

**Corruption bugs fixed on the way**:
- **Decimals dropped zeros**: `3.05` was spoken "three point five" — factually wrong (3.05 ≠ 3.5) — and `4.0` became "four point". The decimal-digit table now maps `0` → "zero" (the cardinal word table has an empty slot for 0 by design).
- **`and/or` duplication**: the word/word slash rule produced "and or **or**" (group 2 is literally the word "or"). Now "and or"; `client/server` still → "client or server".
- **802.11a/b/g/n/s variants were unprotected**: the tech-spec pattern only allowed 2-4 letter suffixes, so `802.11a` lost its `802` to the integer rule ("eight hundred two.11a") and `802.11s` was read "… one one **seconds**". Suffix quantifier widened to 1-4 letters.
- **Apple codename versions**: `macOS Sonoma 14.2.1` only protected `14.2` and dangled the trailing `.1`; the camelCase splitter also broke `macOS`/`watchOS`/`tvOS` *before* the version-skip builder ran, silently losing protection. Both fixed (codename alternative in the version pattern; Apple OS names added to the camel keep-list).

**IPv4 addresses digit-by-digit** (was "one hundred ninety-two point one six eight.one …"): `192.168.1.1` → "one nine two dot one six eight dot one dot one"; CIDR `10.0.0.0/8` → "… slash 8"; `192.168.1.1:8080` → "… port eight zero eight zero". Dotted version strings (`v2.5.1.2`, `version 3.5.1.2`, `release 1.2.3.4`) are excluded via a version-prefix guard, so they keep engine-native reading.

**Phone numbers digit-by-digit** (the range rule was reading `555-123` as "five hundred fifty-five to one hundred twenty-three"): `555-123-4567` → "five five five, one two three, four five six seven".

**ISO dates**: `2024-01-15` → "January fifteenth, twenty twenty-four" (previously a garbled mix of cardinal + dashes; the range rule also misread `2024-01` as "2024 to 01"). Invalid dates (`2024-13-40`) are left untouched rather than spoken as a wrong date.

**Bare unit durations** (engines read `45s` as "forty-five **ess**"): `45s` → "45 seconds", `24h` → "24 hours", `2w` → "2 weeks", `1h` → "1 hour", `1.5h` → plural. Guards: lowercase-only (`3D`/`2D` dimensions untouched), the `10s of thousands` idiom is excluded, 4-digit values (`1990s`) stay with the decade rule, and version/tech-skip regions (`802.11s`) are honored. `m` is deliberately absent — `5m` remains "5 meters" via the existing bare-meters rule.

**Ports digit-by-digit**: `port 8080` → "port eight zero eight zero" (was "eight thousand eighty"); `port 443` → "port four four three". `import`/`passport` don't match (no word boundary before "port").

**Fiscal quarters**: `Q1`–`Q4` → "quarter 1"–"quarter 4" (case-insensitive; `Q&A` unaffected). **Numeronym**: `e2e` → "end to end".

**Brand words** (case-sensitive tier): lowercase `nginx`/`NGINX` → "engine x" (only the dotted `Nginx` form was covered), `PyTorch` → "pie torch", `PySide6` → "pie side six" / `PySide` → "pie side", `Qt` → "Q T", `GraphQL` → "graph Q L", `FastAPI` → "fast A P I", `WebGL`/`OpenGL` → "web G L"/"open G L", `TensorRT` → "tensor R T", `VoIP` → "V O I P", `I2C`/`I2S` → "I two C"/"I two S", `SPI` → "S P I", `PCIe` → "P C I express".

**Common respellings** (engines say "kwe-we" etc.): `queue` → "cue" (`queues`/`queued` included), `epitome` → "eh pit oh mee", `hyperbole` → "hy per bo lee". **Notation**: `a.m.`/`p.m.` → "A M"/"P M" (dot required — plain "am"/"pm" untouched), `45 deg` → "degrees" with `deg C`/`deg F` → "degrees Celsius/Fahrenheit", `2FA` → "two F A", `CI/CD` → "C I C D" (never "CI or CD"), `¶` → "paragraph", `3*4` → "3 times 4" (asterisk multiplication; markdown is stripped before this runs), trailing `4×` → "4 times" (dimension pairs like `1920×1080` keep the "by" reading).

**Currency**: `₹1,299` → "one thousand two hundred ninety-nine rupees", `₹5K` → "five thousand rupees" (rupee glyph had no engine reading).

User `pronunciation_overrides` continue to win over everything above.

## 3e. Pronunciation improvements (pass 4 — ratios, timestamps, shortcuts, rates)

A fourth pass, triggered by a full-pipeline audit. It fixes one residual *corruption* bug class, then teaches the pipeline a set of notations engines still read badly. All rules are Latin-only and every item has regression tests (`TestPronunciationPass4`, 44 tests).

**Corruption bug fixed on the way — skip positions went stale again**:
- The "Windows 11" fix in section 3 rebuilt the version/tech skip positions only once, mid-way through the number section. Any length-changing substitution *after* that rebuild (a decimal `3.5` → "three point five", a multiplier `5x` → "5 times", a range `10-20` → "10 to 20", a bare duration `45s` → "45 seconds") shifted the offsets and the protection silently missed: "Radius 3.5 meters. Works on Windows 11 too" was spoken "… Works on Windows **eleven** too". Skip positions are now rebuilt immediately before **every** rule that consults them (ranges, ordinals, timestamps, ratios, durations, decimals, multipliers, integers) — the whole bug class is structurally gone, not re-patched.

**Ratios & aspect ratios** (a bare colon was read "sixteen colon nine" or skipped): `16:9` → "16 to 9", `3:1` → "3 to 1", `100:50` → "100 to 50", cinematic `2.35:1` → "2.35 to 1". Clock times keep priority (`5:30` → "five thirty", `12:45 PM` → "twelve forty-five PM"); colon guards leave MAC-like chains (`00:11:22:33:44:55`) and the tail of `12:34:56` untouched.

**Elapsed h:m:s timestamps** (the time rule used to eat `12:34` and dangle a `:56` — "twelve thirty-four:fifty-six"): `12:34:56` → "twelve hours, thirty-four minutes, fifty-six seconds"; `01:07:09` → "one hour, seven minutes, nine seconds"; zeros are skipped (`00:00:30` → "thirty seconds").

**Keyboard shortcuts** (the glued `+` was read as silence or a stutter — "control+Alt"): `Ctrl+S` → "control plus S", `Ctrl+Alt+Del` → "control plus Alt plus Del", `Cmd+Q` → "command plus Q". The modifier word (`Ctrl|Cmd|Alt|Shift|Win|Meta|Fn|Option`, case-insensitive) anchors the match, so prose plus signs and `C++` are untouched.

**Rate shorthand** (was doubly mangled — the slash rule said "ops or sec" and the abbreviation table's `(?=\s|$)` lookahead then read the exposed "sec"/"min": "ops or seconds", "msg or minimum"): `100 ops/sec` → "one hundred operations per second", `5 msg/min` → "5 message per minute", `50 pages/hr` → "pages per hour", `2 builds/day` → "builds per day". Runs before the abbreviation table; the `km/h` and `KB/s` compound tables keep owning those forms; bare `m` is deliberately absent (meters, not minutes).

**Negative temperatures** (the minus was left glued and read "dash"): `-40°C` → "minus forty degrees Celsius", `−10℃` → "minus ten degrees Celsius", `-4℉` → "minus 4 degrees Fahrenheit". Positive readings unchanged.

**Idioms & decades**: `50-50` / `50/50` → "fifty fifty" (was "fifty to fifty" / "fifty divided by fifty"); `the '90s` → "the nineties" — previously "the **ninety seconds**", the bare-duration rule ate the `90s`; `mid-90s` → "mid nineties" and `mid-'80s` → "mid eighties"; `1990's style` → "nineteen nineties style" (was "one thousand nine hundred ninety's style") and `2023's revenue` → "twenty twenty-three's revenue". A bare `90s` without apostrophe/`mid-` prefix stays a duration ("Wait 90s" → "ninety seconds").

**Glyph comparisons & symbols** (no engine reading at all before): `≥` → "is greater than or equal to", `≤` → "is less than or equal to"; `+/-` → "plus or minus" (previously the standalone-plus rule shredded it into "plus /-3 percent"); `y/o` → "years old" (was "y or o"); guillemets `« »` and angle quotes `‹ ›` are now stripped like other silent typographic marks.

**Decimal multipliers** (`1.5x` kept a bare unspoken `x`): `1.5x faster` → "one point five times faster". The multiplier rule now runs before the decimal rule (plain `3x` → "3 times" unchanged; `v1.2x` / `802.11x` version strings still protected by skip positions and the `(?!\d)` guard).

**Respelling**: `segue` → "seg way", `segues` → "seg ways" (engines say "seeg").

User `pronunciation_overrides` still win over everything above.

## 3f. Pronunciation improvements (pass 5 — math)

A dedicated math pass rewrites equations, operators and math notation into
spoken form. It runs *before* the superscript/subscript rules (which would
shred `d²y/dx²` into "d squared y" and `log₂` into "log two"), before the
caret rules, and before the parenthetical rule (which would speak `sin(x)`
as "sine, x,").

**Tight operators** (no spaces): `3+4` → "3 plus 4", `a+b` → "a plus b",
`x=5` → "x equals 5", `a^2+b^2=c^2` → "a squared plus b squared equals c
squared", `E=mc^2` → "E equals mc squared". Every `=` the spaced rule
misses becomes "equals" (strict/double/plus-equals are consumed earlier),
and `a*b` / `i*pi` / `P(A) * P(B)` become "times" (digit×digit keeps the
existing "1920 by 1080" dimension reading).

**Negative numbers**: `-5`, `x = -5`, `x=-5`, `(-5)` → "minus 5" — only
whitespace/equals/paren-anchored dashes fire, so ranges (`2020-2024`),
phones, ISO dates and hyphenated tokens (`GPT-4`, `A-5`, `mid-90s`) keep
their dash. Negated functions too: `-sin(x)` → "minus sine of x".

**Scientific notation & Python exponents**: `1e9` → "one times ten to the
ninth", `2.5e-3` → "… minus third", `2**10` → "two to the tenth",
`2 ** 3` → "two cubed", `e^(i*pi)` → "e to the i times pi".

**Primes**: `f'(x)` → "f prime of x", `x' = 2` → "x prime equals 2".
Quotes (`the 'best'.`) and possessives (`it's`, `o'clock`) never match.

**Derivatives**: `dy/dx` → "d y by d x", `d²y/dx²` → "second derivative of
y with respect to x", `d/dt` → "d by d t", `∂f/∂x` → "partial f partial
x", `∂²u/∂x²` → "second partial derivative of u with respect to x".

**Function calls** (nested calls resolve innermost-first): `sin(x)` →
"sine of x", `sqrt(25)` → "square root of twenty-five", `ln(x)` → "natural
log of x", `log(x)` → "log of x", `exp(x)` → "exponential of x",
`max(a, b)` → "maximum of a, b", plus `sinh/arcsin/arccos/arctan/cosh/
tanh/cot/lim/min`. Generic `f(x)` / `P(A)` → "f of x" / "P of A". Bare
garble-prone names are respelled: `ln` → "natural log", `sin` → "sine",
`cos` → "cosine", `tan` → "tangent", `sqrt` → "square root of",
`exp` → "exponential", `lim` → "limit", `arcsin` → "arc sine" (`log`,
`max`, `min` are real words and stay). Log bases: `log_2(8)` / `log₂(8)`
→ "log base 2 of 8" / "log base two of 8".

**Greek alphabet & math glyphs**: α–ω (plus ϵ ϕ ς ϴ variants) and the
distinctive capitals Γ Θ Λ Ξ Σ Π Φ Ψ are spoken by name ("alpha",
"lambda", "sigma"…); look-alike capitals (Α Β Ε …, Greek-layout typos of
Latin letters) and omicron are deliberately not mapped. `α²` still reads
"alpha squared". Digit-anchored Greek-mu units: `5 μm` → "5 micrometers",
`3 μs`, `2 μg`; the U+00B5 micro rules now accept U+03BC too. New glyphs:
`∑` → "sum of", `∏` → "product of", `∂` → "partial", `∛` → "cube root
of", `∈` → "in", `∉` → "not in", `∀` → "for all", `∃` → "there exists",
`∪` → "union", `∩` → "intersection", `⊂/⊆` → "is a subset of",
`⊃/⊇` → "is a superset of", `∅` → "the empty set", `∝` → "is proportional
to", `≡` → "is equivalent to", `∴` → "therefore", `∵` → "because",
`⊥` → "is perpendicular to", `∠` → "angle".

**Probability & division**: `P(A|B)` → "P of A given B" — only pipes
*inside* parentheses convert, so shell pipes (`ls | grep`), boolean `p|q`
and markdown tables keep their bar, and `|x|` bars are untouched.
`P(A) / P(B)` → "P of A divided by P of B" and `5 / (x+1)` → "5 divided by
(x plus 1)" (the slash rule used to say "or"); `x/2` → "x over 2",
`2/x` → "2 over x" with token guards protecting `PS/2`, `HTML/5` and
`80/kg`. Parens that follow "divided by / of / to the / given" keep their
parens instead of becoming a comma aside.

All of the above is Latin-branch only — CJK text is untouched (tested).

## 3g. Pronunciation improvements (pass 6 — words & math names)

A further pass on what engines demonstrably mispronounce: *mathematician
names*, *math shorthand*, *inverse-trig notation*, and a few everyday
words. All rules are context-guarded, Latin-only, and each has regression
tests (70 new).

**Latent unit-anchor corruption found on the way**: the digit-anchored unit
patterns used `\d[\d,]*` for the leading number, whose character class
includes commas — so the list sequence "…equals 6, L C M…" matched "6," +
" L" and was spoken "6 **liters** C M". The digit group in the unit,
bare-meters and bare-duration templates is now a strict number
(`\d+(?:,\d{3})*`): "1,299 liters" and "2,500 ms" still expand, but a
comma-list before a unit letter can never anchor again.

**Mathematician-name respellings** (engines read these phonetically wrong):
`Euler` → "oiler" (was "YOU-ler"), `Euclid` → "yoo klid", `Fourier` → "foor
ee ay" (was "four-ee-er"), `Poisson` → "pwah sohn" (was "POY-son"),
`Poincaré`/`Poincare` → "pwan ka ray", `Bézier`/`Bezier` → "bez ee ay",
`Cauchy` → "koh shee" (was "COW-chee"), `Galois` → "gal wah" (was
"GAY-loys"), `Laplace` → "luh ploss", `Descartes` → "day cart",
`Dirichlet` → "deer ikh lay", `Leibniz` → "lype nits", `Chebyshev` → "cheh
bee shef", `Schrödinger`/`Schroedinger` → "shroh ding er", `de Broglie` →
"duh broh glee", `L'Hôpital`/`L'Hopital` → "loh pee tal", `Lebesgue` → "luh
beg" (was "luh-BES-goo"), `Banach` → "bah nahk", `Noether` → "nur ter",
`Lagrange` → "luh grahnj", `Fermat` → "fur mah", `Erdős`/`Erdos` → "air
dish", `de Moivre` → "mwah vruh", `Delaunay` → "day lon ay", `Weierstrass`
→ "vy er strahss", `Legendre` → "luh zhan druh", `Tukey` → "too kee",
`Eratosthenes` → "air uh tos thuh neez", `Kleene` → "klay nee",
`Dijkstra` → "dik struh" (was "dij-kstra"), `Eulerian` → "oiler ee un".
Words like "Euclidean"/"Cartesian" are deliberately untouched (\b guards).

**Eigen- family**: `eigenvalue`/`eigenvalues`/`eigenvector`/`eigenvectors`
and snake/camel-split `eigen value` → "eye gen …" (was "AY-gen" or
"EE-gen").

**Math shorthand words**: `iff` → "if and only if", `w.r.t.`/`wrt` → "with
respect to", `WLOG` → "without loss of generality", `lhs`/`rhs` → "left
hand side"/"right hand side", `gcd`/`lcm` → "G C D"/"L C M", `thm.` →
"theorem", `resp.` → "respectively", `prop.` → "proposition", `PDE` → "P D
E" (`ODE` already reads as the word "ode" and stays).

**Inverse trigonometrics** (was "sin to the minus first x"): `sin⁻¹(x)` →
"inverse sine of x", `cos⁻¹(θ)` → "inverse cosine of …", `tan⁻¹` →
"inverse tangent"; the ASCII forms `sin^-1(x)` / `sin^-1` are handled the
same way. Both must run *before* the bare-function respeller (which would
rewrite `sin` → "sine" and break the match) and before the
superscript/caret rules. Plain caret powers (`10^-6` → "ten to the minus
sixth") and hyperbolics (`sinh`, `cosh`) are unaffected.

**Factorial**: `n! = 120` → "n factorial = 120" (was spoken as sentence
emphasis: "n … is one hundred twenty"). Lookahead-guarded so sentence-final
exclamations (`Pick n! Then…`) and `a != b` never match.

**Assignment operator**: `f(x) := x²` → "f of x is defined as x squared"
(the leftover-equals rule would have said "x : equals x squared"); also
the Python walrus operator.

**Modulo**: `5 mod 3` → "5 modulo 3" — digit-anchored on both sides; the
existing mod→"module" abbreviation would otherwise say "5 **module** 3",
while prose/tech "mod" is untouched.

**Logic & extra glyphs** (previously silent or stuttered): `¬` → "not",
`∧` → "and", `∨` → "or", `⇔`/`↔` → "if and only if", `↦` → "maps to",
`≅` → "is congruent to", `≃` → "is approximately equal to", `∜` → "fourth
root of", `∮` → "contour integral of", `∼` → "is similar to".

**More function calls**: `sec(x)` → "secant of x", `csc(x)` → "cosecant of
x" (bare `csc` too — it is never an English word; bare `sec` is
deliberately *not* mapped so "30 sec" and "ops/sec" keep their readings),
`det(A)` → "determinant of A", `gcd(a, b)` → "G C D of a, b", `lcm` → "L C
M of …".

**Everyday words engines garble**: `awry` → "uh rye" (was "AW-ree"),
`rendezvous` → "ron day voo", `quay`/`quays` → "kee"/"keez", `indict` →
"in dite" (was "in-DIKT"), `voilà`/`voila` → "vwah lah".

All pass-6 rules are Latin-branch only — CJK output is byte-identical
(tested). The skip-position rebuild for version/model/tech regions (the
"Windows 11" root cause) was re-verified against math operators: a decimal,
multiplier, currency, caret or fraction before `Windows 11` / `v2.5.1.2`
no longer shifts the protected digits.

## 3h. Pronunciation improvements (pass 7 — history)

A dedicated history pass rewrites era markers, year spans and circa
notation into spoken form, plus ~55 respellings for names and places
engines mangle. It runs *before* the number-conversion section, so each
year is consumed together with its era or span — the integer rule can no
longer read "1066" as "one thousand sixty-six". All rules are
digit-anchored and Latin-only; each has regression tests (90 new).

**Era markers** (previously engines said "in ad 1066" or skipped "BC"
entirely): `44 BC` → "forty-four B C", `2500 BCE` → "two thousand five
hundred B C E", `1066 AD` → "ten sixty-six A D", `CE 220` → "C E two
hundred twenty". Dotted forms (`44 B.C.`, `A.D. 1066`) work too, and the
trailing dot of `B.C.` is never consumed, so a sentence period after the
era survives. Guards: a bare `BC` (British Columbia) or `AD` in prose is
untouched, lowercase `44 bc` / `ad 1066` is left to the engine, and tech
tokens like `802.11AD` are protected via the version/tech skip regions
(the skip-position builder moved to module level so the history pass and
the number section share the same protection set, rebuilt on current text
before every use).

**Year spans** (previously the second year lost its year reading: "1914
to 1918" was spoken "…to one thousand nine hundred eighteen"): `1914-1918`
→ "nineteen fourteen to nineteen eighteen", `1939—1945` likewise, `between
1914 and 1918`, `from 1914 to 1918` (the "from"/"between" word is kept),
`1914-1918 AD` and `3000-2500 BC` convert both years plus the era in one
pass. The lookbehind also blocks "-", so chained tokens (European phones
`020-7946-0958`) and version regions (`Windows 2019-2023`) can never
anchor a span.

**Circa**: `c. 1860` → "circa 1860" (was "see dot…"); the lookahead
requires 3-4 digits so list tails ("a, b, c. 5 more") never fire; runs
inside the history pass before the era rules, otherwise `c. 1860 BCE`
converted the year first and `c.` no longer saw digits. "circa" is a
year-context word, so the year reads year-style.

**Year-style readings** (`_number_to_words`): round years of the last
millennium read "nineteen hundred" — `in 1900` was "one thousand nine
hundred"; single-digit tails read with "oh" — `1806` → "eighteen oh six"
(was "eighteen six"), `2005` → "twenty oh five"; `2000` → "two thousand"
and cardinal contexts (`The library has 1900 books.` → "one thousand nine
hundred") are unchanged. A comma directly before a year is now year
context — history decks date events as "Treaty of Versailles, 1919"
(was "one thousand nine hundred nineteen"). Month names too: `May 1945`
reads "May nineteen forty-five" (the month-day rule only consumes 1-2
digit days, so bare month-year pairs were read as cardinals).

**Ruler numerals extended**: `Ivan IV`, `Wilhelm II`, `Nicholas II`,
`Catherine II`, `Peter I`, `Frederick II`, `Constantine I`, `Justinian I`,
`Otto I`, `Gustav`, `Alfonso`, `Basil`, `Joseph` now read "Ivan the
fourth" style (previously only a smaller ruler list existed).

**"Gen" title bug**: `\bGen\.?(?=\s|$)` expanded *bare* "Gen", so "Gen Z
buyers" was spoken "general Z buyers". The dot is now required
(`Gen. Sherman` still reads "general Sherman"); "Gen Z"/"Gen X" pass
through.

**History respellings** (same syllable style as the mathematician list;
all proper nouns, so the case-insensitive tier cannot collide with English
words): `Versailles` → "ver sigh", `Charlemagne` → "shar luh mayn",
`Byzantine` → "biz un teen", `Nebuchadnezzar` → "neb uh kud nezz er",
`Xerxes` → "zerk seez", `Thucydides` → "thoo sid ih deez",
`Herodotus` → "huh rod uh tus", `Ptolemy`/`Ptolemaic` → "tol uh mee"/"tol
uh may ik", `Boudicca` → "boo dik ah", `Hammurabi` → "hah moo rah bee",
`Mesopotamia` → "mes uh puh tay mee uh", `Aegean` → "ih jee un",
`Mycenae` → "my see nee", `Antioch` → "an tee ahk", `Vercingetorix` →
"vur san guh tor iks", `Plantagenet` → "plan tag uh net", `Carolingian` →
"kair uh lin jee un", `Merovingian` → "mair uh vin jee un", `Valois` →
"val wah", `Agincourt` → "azh in koor", `Bayeux` → "bye yuh",
`Austerlitz` → "aws ter lits", `Montesquieu` → "mont uh skyoo",
`Rousseau` → "roo soh", `Tocqueville` → "tohk vil", `Toussaint` → "too
sant", `L'Ouverture` → "loo ver toor", `Tenochtitlan` → "te noch tee
tlahn", `Umayyad` → "oo my yad", `Abbasid` → "uh bah sid", `Seljuk` →
"sel jook", `Saladin` → "sal ah deen", `Maurya` → "mah oo ryah",
`Ashoka` → "uh shoh kah", `Mughal(s)` → "moo gul(s)", `Genghis` → "geng
gis", `Kublai` → "koo blie", `Tamerlane` → "tam er lane", `Qing` →
"ching", `Qin` → "chin", `Zhou` → "joh", `Sui` → "sway", `Xia` → "shyah",
`Zheng` → "jung", `Xinhai` → "shin high", `Cixi` → "tsih shee",
`Tokugawa` → "toh koo gah wah",
`Meiji` → "may ee jee", `Heian` → "hay ahn", `Kamakura` → "kah mah koo
rah", `Joseon` → "choh sun", `Bolshevik(s)` → "bol shuh vick(s)",
`Khrushchev` → "kroosh chof", `Weimar` → "vy mahr", `Reichstag` → "rikes
tahk", `Luftwaffe` → "looft vah fuh", `Blitzkrieg` → "blits kreeg",
`Lusitania` → "loo sih tay nee ah", `Ypres` → "ee pruh", `Apartheid` →
"uh pahr tate", `Tiananmen` → "tyen an men", `Perestroika` → "per uh
stroy kah". Plurals keep their "s"; ordinary words ("historic", "modern",
"scenic") are untouched (tested), and CJK output is byte-identical
(tested). Known trade-off: the rare Latin phrase "sui generis" now reads
"sway generis" — accepted because "Sui dynasty" is far more common in
slide decks.

## 4. Data integrity & security

- **Atomic JSON writes everywhere** (`atomic_write_json`, `_atomic_write_text`): write to temp file, `fsync`, `os.replace`. No more truncated `settings.json` / manifests on crash or full disk.
- **Corrupt settings.json recovery**: the file is backed up as `settings.json.corrupt` and defaults are restored instead of silently keeping half-loaded state.
- **Library manifest concurrency**: file locking prevents two instances from clobbering each other's manifest; ghost entries (projects whose folder is gone) are pruned on load.
- **Error status survives restart**: projects marked `error` are no longer reset to pending by a re-scan, so failed renders are visible after relaunch.
- **`should_skip_render` checks asset mtimes** against the manifest (`assets_meta`), so editing a slide invalidates the cache even if the manifest hash is unchanged.
- **Path traversal fixed**: project names are sanitized (`_sanitize_project_name`) at every entry point — a name like `../evil` could previously reach `shutil.rmtree`. Illegal characters, traversal segments and empty names are handled with a safe fallback.
- **Thumbnail staleness**: cheap mtime comparison (`slide1.html` vs thumbnail) avoids spawning a process per check.
- **Config path respects `APP_ROOT`**: `settings.json` was previously hardcoded next to the source module, so any launch that redirected `APP_ROOT` (tests, embedded runs) silently wrote/overwrote settings files in the source folder. The path is now derived from `APP_ROOT` — identical location in normal desktop use, sandboxed otherwise.

## 5. Code quality & UX

- `resolve_projects_root()` / `resolve_output_dir()` replace 11 copy-pasted path-resolution blocks.
- Settings-dialog search now matches actual setting labels instead of raw keys.
- Editor shortcuts (Ctrl+Shift+N etc.) are scoped to the editor tab — they no longer add slides while a dialog has focus.
- A failed render leaves the Finalized state as `Error` instead of pretending completion.
- Progress bar state survives a refresh instead of resetting.
- PDF import extracts text and rolls back cleanly on failure instead of leaving a half-created project.
- O(n²) lookups replaced with bounded lookback.
- `app.py` enables `faulthandler` and installs graceful Ctrl+C handling so workers are not orphaned.

## 6. Test suite (`tests/`, 648 tests)

| File | Tests | Covers |
|---|---:|---|
| test_text_chunker.py | 584 | All TTS-corruption regressions (including the prefix-eating `Dr`/`Prof`/`etc`/`CFO` bug), all pronunciation rules from all seven passes (units, degrees, currency/magnitude, Roman numerals, tech tokens, idioms, exponent/superscript/subscript notation, math glyphs, operators, decades, dimension products, case-sensitive tier, numeronyms, identifier splitting, override merging, digit-by-digit IPs/phones/ports, ISO dates, bare durations, quarters, brands, respellings, decimal zeros; ratios/timestamps, keyboard shortcuts, rate shorthand, negative degrees, ≥/≤ and +/- glyphs, segue, decimal multipliers, skip-position refresh regressions; math pass: tight operators, negative numbers, scientific notation, Python `**`, primes, derivatives, function calls, log bases, Greek letters, set glyphs, probability pipes, variable division; pass 6: inverse trig, factorial, `:=`, modulo, logic glyphs, sec/csc/det/gcd/lcm calls, math shorthand, mathematician-name respellings, garbled-word respellings; pass 7 history: era markers BC/BCE/AD/CE, year spans, circa, round-year and "oh" year readings, month-year dates, ruler numerals, history respellings, Gen-title fix) |
| test_utils.py | 39 | Path sanitization/traversal, atomic writes, manifest locking, `should_skip_render` mtime logic, `resolve_projects_root` |
| test_config.py | 25 | Load/save round-trips, corrupt-file recovery, atomic write, defaults merge, `APP_ROOT`-based config path sandboxing, `pronunciation_overrides` validation |

Run with: `python -m pytest tests/ -q` (Python 3.12; no PySide6/torch required — GUI modules are import-mocked in `conftest.py`). The suite is fully hermetic: repeated runs leave no files next to the sources.

## 7. Deliberately not changed (flagged, too risky without a live GUI)

- Frame-data streaming architecture (replacing batch video frames) — high-risk rewrite of the worker protocol.
- Splitting `dialogs.py` (4,129 lines) into modules — would break drop-in replacement.
- Threading message-path identity — touched everywhere, untestable headless.

These are recommendations for a future refactor with the app running.
