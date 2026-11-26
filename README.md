# Text-Based Reel Editor

*🇦🇿 Azərbaycan dilində: [README.az.md](README.az.md)*

Edit vertical (9:16) Instagram Reels **by editing their transcript**. Upload a
clip, strike out the words you don't want, curate AI-suggested B-roll cutaways,
and export a finished Reel with burned-in kinetic subtitles.

```
upload ─► ffmpeg 16 kHz mono ─► Silero VAD ─► Groq whisper-large-v3 (word timestamps)
       ─► LLM terminology fix ─► LLM B-roll moments ─► Pexels 9:16 search
       ─► [ you edit the text ] ─► timeline recalculation ─► ffmpeg render ─► MP4
```

* **Backend** — FastAPI (Python 3.12), fully typed, modular.
* **Frontend** — React 18 + TypeScript + Vite; interactive word tokens, live
  cut preview, inline B-roll placeholders.
* **No mocks.** Every stage calls the real API or the real binary.

---

## 1. Requirements

| Component | Version | Notes |
|---|---|---|
| Python | **3.10 – 3.13** | 3.12 recommended. PyTorch (Silero VAD) has no 3.14 wheels yet. |
| FFmpeg + ffprobe | 6.0+ | `brew install ffmpeg` / `apt install ffmpeg`. **libass strongly recommended** (see §6). |
| Node.js | 18+ | Only to build/serve the UI. |
| Groq API key | — | https://console.groq.com/keys |
| Pexels API key | — | https://www.pexels.com/api/ (optional; disables B-roll if absent) |

## 2. Setup

```bash
git clone <this repo> && cd "AI Video editor"

# ---- backend -------------------------------------------------------------
python3.12 -m venv .venv
source .venv/bin/activate

# Recommended: includes PyTorch + Silero VAD
pip install -r requirements-vad.txt
# ...or core only (falls back to the built-in energy VAD):
# pip install -r requirements.txt

# ---- keys ----------------------------------------------------------------
cp .env.example .env
$EDITOR .env          # set GROQ_API_KEY and PEXELS_API_KEY

# ---- frontend ------------------------------------------------------------
cd frontend && npm install && npm run build && cd ..
```

### Run

```bash
# single process: API + built UI on http://127.0.0.1:8000
./run.sh
```

Or run the two halves separately during development:

```bash
uvicorn app.main:app --reload --port 8000     # terminal 1
cd frontend && npm run dev                    # terminal 2 -> http://localhost:5173
```

The Vite dev server proxies `/api` to `127.0.0.1:8000`, so the browser stays on
one origin (override with `VITE_API_TARGET`).

Health checks: [`/api/health`](http://127.0.0.1:8000/api/health),
[`/api/capabilities`](http://127.0.0.1:8000/api/capabilities),
OpenAPI docs at [`/docs`](http://127.0.0.1:8000/docs).

## 3. Using it

1. **Upload** a vertical MP4/MOV that has an audio track. Ingest runs in the
   background; the UI shows every stage.
2. **Edit the text.** Click a word to strike it out, `Shift`+click for a range,
   the `✕` on a hovered sentence to drop the whole line, `⌘/Ctrl+Z` to undo.
   `Alt`+click scrubs the preview to that word.
3. **Tighten pauses** (on by default) also removes the dead air Silero VAD
   found *inside* surviving passages — silences ≥ `SILENCE_TRIM_MS` only, so
   natural rhythm survives.
4. **Curate B-roll.** Each suggestion offers `Accept` / `Swap` (type a new
   Pexels query) / `Dismiss`. Nothing is baked in without an explicit accept.
5. **Preview** — the player skips removed regions, so you hear the real cut
   before rendering.
6. **Render** and download the MP4.

## 4. Architecture

```
app/
  config.py            all tunables, .env-driven, validated once
  errors.py            typed exceptions -> HTTP status + stable error codes
  schemas.py           Pydantic models (also the on-disk format)
  ffmpeg_utils.py      probe, progress-reporting runner, capability probe, font discovery
  audio_processing.py  16 kHz extraction, Silero VAD, compacted ASR chunks + time mapping
  transcription.py     Groq whisper-large-v3, sentence grouping, terminology normalisation
  llm_broll.py         LLM moment detection, Pexels 9:16 search, clip download
  timeline_manager.py  text edit -> cut list -> re-timed words & B-roll
  subtitles.py         kinetic captions: .ass karaoke + Pillow raster engine
  renderer.py          single-pass ffmpeg filtergraph (cut, join, overlay, burn-in)
  store.py             atomic per-project JSON persistence, per-project locks
  jobs.py              thread-pool job runner with persisted progress
  pipeline.py          stage orchestration for ingest & render
  main.py              FastAPI routes
  ui.py                serves the built SPA
frontend/src/          React UI (components/, api.ts, types.ts)
tests/                 network-free tests for the pure logic
data/projects/<id>/    source.mp4, work/, output/, jobs/, project.json
```

### Three time bases

Every timestamp field names the timeline it belongs to:

| base | meaning |
|---|---|
| `source_*` | the original uploaded video |
| `compact_*` | the silence-stripped audio sent to Whisper (internal to transcription) |
| `output_*` | the final rendered Reel |

`CompactTimeline.to_source()` is the piecewise-linear bridge from the second to
the first; `timeline_manager` builds the bridge from the first to the third.

### Design decisions worth knowing

**Speech-only ASR audio.** The VAD segments are concatenated into FLAC chunks
before upload. Whisper never hears breaths or dead air (fewer hallucinated
filler tokens, smaller uploads), and word timestamps are mapped back to the
source timeline afterwards. A 4 ms fade at each seam prevents click artefacts.

**Timestamps are never rewritten by an LLM.** The terminology pass is a strict
token-for-token substitution: it may only replace an existing token's surface
form. Insertions, deletions, merges, splits and retiming are rejected *in code*
(`_acceptable_replacement`), not merely discouraged in the prompt. The original
ASR spelling is kept in `Word.raw_text` and shown in the UI with a dotted
underline.

**Cuts happen only where words were removed.** Natural pauses inside a
surviving run of words are preserved, so delivery keeps its rhythm. Separately,
`trim_silences` removes VAD-detected dead air ≥ `SILENCE_TRIM_MS`. Removals that
would save less than `MIN_REMOVABLE_GAP_MS` are merged away rather than becoming
a visible jump cut, and no clip is shorter than `MIN_CLIP_MS`.

**Audio joins preserve the timeline exactly.** A hard cut clicks. The default
`crossfade` mode extends each clip's *audio* trim `d` ms past its video cut
point and joins with `acrossfade=d`; because `len(A+d) + len(B) − d == len(A) +
len(B)`, total length is unchanged, so the subtitle timings stay valid. The
dissolve material is the real audio just after the cut — a classic audio
dissolve under a hard picture cut. When a junction has no spare material (a cut
that ends at EOF), the renderer automatically falls back to per-clip micro
fades + hard concat (`AUDIO_JOIN_MODE=fade`). Verified: both modes reproduce the
computed duration to the millisecond.

**One ffmpeg pass.** Cut → concat → normalise to 1080×1920 (scale-to-cover +
centre crop) → B-roll overlays → subtitle burn-in all happen in a single
`filter_complex`, so the video is encoded exactly once. The generated graph is
dumped to `data/projects/<id>/work/filter_complex.txt` for debugging.

**B-roll keeps the dialogue.** Overlays replace the picture only; the stock
clip's own audio is never mapped. Overlapping overlays are resolved
deterministically (earlier wins, later is pushed back or dropped).

## 5. Subtitles: two engines

Both share one layout model (line grouping, per-word highlight states).

* **`ass`** — an Advanced SubStation Alpha script burned in with FFmpeg's `ass`
  filter. Real font shaping, an animated scale-up on the active word, sub-pixel
  outlines. Requires **libass**.
* **`pillow`** — rasterises one transparent PNG per highlight state and
  composites them through the concat demuxer with a single `overlay`. Automatic
  fallback when the FFmpeg build lacks libass/libfreetype. Each word occupies a
  slot sized for its *highlighted* rendition, so the line neither jitters nor
  collides as the highlight advances.

`SUBTITLE_ENGINE=auto` (default) picks `ass` when available, otherwise
`pillow`. `/api/capabilities` reports what was chosen and why.

Fonts are auto-discovered from the usual macOS/Linux/Windows font directories
(Montserrat ExtraBold → Anton → Inter/Roboto/Noto/DejaVu Bold → Arial Bold →
anything bold). Override with `SUBTITLE_FONT_FILE=/path/to/Font.ttf`.

**Glyph coverage is verified, not assumed.** `Impact` and `Arial Black` ship
without a glyph for `ə` (U+0259), so picking them by weight alone would render
every Azerbaijani schwa as an empty box. `font_covers()` renders each of
`ə ğ ı ş ç ö ü` and compares the bitmap against the `.notdef` glyph, so those
faces are skipped unless nothing else is installed — and
`/api/capabilities.font_covers_language` reports the verdict to the UI. The
check uses Pillow only (no extra dependency) and was cross-checked against
fontTools' cmap tables.

## 6. Which model for Azerbaijani?

### Why the original model broke

`llama-3.3-70b-versatile` was **deprecated by Groq on 2026-06-17 and shut down
on 2026-08-16**, together with `llama-3.1-8b-instant`, `qwen/qwen3-32b` and
`meta-llama/llama-4-scout-17b-16e-instruct`. Groq's own migration advice for it
is `openai/gpt-oss-120b` or `qwen/qwen3.6-27b`. The deprecation hits free and
developer tiers; committed-spend enterprise accounts are unaffected. That is why
the app now walks a fallback chain instead of trusting a single hardcoded id.

### Measured, not assumed

Both remaining candidates claim broad multilingual coverage (Qwen3: 119
languages incl. the Turkic family; gpt-oss: 81+ languages) but neither publishes
Azerbaijani numbers, and community reports flag weak gpt-oss translation quality
in some low-resource languages. So the choice was benchmarked with **this app's
own prompt**: 35 Azerbaijani tokens — 13 phonetically mangled English terms that
must be repaired, 22 correct Azerbaijani words (incl. loanwords like `alqoritm`,
`marketinq`, `kontent`) that must be left alone — 3 runs each.

| Model | repaired /13 | corrupted /22 | avg latency |
|---|---|---|---|
| **`openai/gpt-oss-120b`** | **13.0** | **0** | 1.1 s |
| `openai/gpt-oss-20b` | 13.0 | 0 | 0.7 s |
| `groq/compound-mini` | 13.0 | 0 | 3.6 s |
| `qwen/qwen3.6-27b` | 12.0 | 0 | 0.6 s |

All four also handled the two-token `prompt engineering` case without illegally
merging tokens. On the *B-roll* task `gpt-oss-120b` produced the most usable
English search queries (`"video editing timeline with b-roll"`), while
`gpt-oss-20b` was vaguest (`"storytelling graphic"`) and `compound-mini` once
returned a semantically wrong repair (`vörkflou → framework`).

**Verdict:** `openai/gpt-oss-120b` primary, then `openai/gpt-oss-20b` (equally
accurate and faster, weaker at creative queries), then `qwen/qwen3.6-27b`
(fastest, consistently misses one term), then `groq/compound-mini` (correct but
3× slower). That is exactly `GROQ_LLM_MODEL` + `GROQ_LLM_FALLBACKS`.

### Two real bugs the benchmark exposed

1. **The glossary was corrupting correct Azerbaijani.** It listed `algoritm`, so
   *every* model dutifully "fixed" the correct Azerbaijani `alqoritm` into it.
   `DEFAULT_GLOSSARY` (English terms) is now separate from `AZ_PRESERVE`
   (loanwords whose Azerbaijani spelling is already correct), and prompt rule 7
   forbids anglicising them. Corruption went from 1/22 to **0/22 on every model**.
2. **Reasoning tokens were eating the whole budget.** Qwen3 emits
   `<think>…</think>` inside `content`; with `max_tokens=2048` it never reached
   the JSON and returned nothing usable — which looked like "Qwen is bad at
   Azerbaijani" but was purely our bug. Fixed by `strip_reasoning()` plus
   `reasoning_kwargs()` (`reasoning_effort=none` for Qwen, `low` for gpt-oss).
   Note that simply raising `max_tokens` is *not* the fix: 8192 returns HTTP 413
   on this account tier.

### ASR: keep `whisper-large-v3`, not turbo

For Azerbaijani the model size genuinely matters — published FLEURS-az results:

| Model | WER | CER |
|---|---|---|
| **Whisper large-v3** | **21.7 %** | **5.9 %** |
| MMS-1B (azj-latin) | 23.8 % | 5.3 % |
| Whisper medium | 34.3 % | 9.0 % |
| Whisper small | 51.7 % | 14.4 % |

`whisper-large-v3-turbo` prunes the decoder from 32 layers to 4 and trades
~1–2 % WER for ~4× speed, degrading most on low-resource languages — the wrong
trade here, so `GROQ_ASR_MODEL` stays `whisper-large-v3`.

Two Azerbaijani-specific consequences already handled in the code:

* **Turkish misrouting.** With auto-detection Whisper frequently routes
  Azerbaijani audio to Turkish and produces systematic orthographic errors. The
  app always sends `language="az"` explicitly (`ASR_LANGUAGE`), which is the
  documented mitigation.
* **Known error profile** on Azerbaijani: named entities 24.1 %, numerals
  22.9 %, dropped `ə ğ ı ö ş ü ç` 18.9 %. The terminology pass only helps the
  first bucket, so proper nouns and figures still deserve a human read before
  you render.

## 7. Troubleshooting

**`No such filter: 'ass'` / captions look different than expected.**
Your FFmpeg has no libass. Check `/api/capabilities`. Either reinstall FFmpeg
with libass (`brew reinstall ffmpeg`) or keep the automatic `pillow` engine —
both produce burned-in kinetic captions.

**`The model 'x' does not exist or you do not have access to it`.**
Groq retires model IDs. `GROQ_LLM_MODEL` is tried first, then every id in
`GROQ_LLM_FALLBACKS`, so the app keeps working; set `GROQ_LLM_MODEL` to
whatever `GET https://api.groq.com/openai/v1/models` lists for your account.
Transcription (`whisper-large-v3`) and B-roll/terminology are independent —
if the LLM is unavailable, B-roll moment selection degrades to a deterministic
information-density heuristic and the transcript is still delivered. See §6 for the measured ranking on Azerbaijani.

**Rate limits.** Groq and Pexels 429s are retried with exponential backoff and
honour `Retry-After`. If they persist the API returns `429` with
`error.code = upstream_rate_limited` and `error.retry_after`.

**"Silero VAD could not be loaded".** PyTorch isn't installed for your Python
version. Install `requirements-vad.txt` on Python 3.10–3.13. The built-in
energy VAD (adaptive noise floor + hysteresis) is used meanwhile and the
project surfaces a warning.

**"The uploaded file contains no video/audio stream".** Exactly what it says —
`ffprobe` found only one of the two. Re-export with both tracks.

**macOS: `pip` fails with `Symbol not found: _XML_SetAllocTrackerActivationThreshold`.**
Homebrew's Python links `pyexpat` against the system `libexpat`, which on
macOS 26 lacks that symbol; this breaks `venv`/`pip` (not this app). Fix:

```bash
brew install expat
export DYLD_LIBRARY_PATH=/opt/homebrew/opt/expat/lib   # needed for pip/venv only
```

The application itself does not import `pyexpat`, so you do **not** need the
variable set when running the server.

## 8. Configuration

Every value in `.env.example` maps 1:1 to a field on `Settings` in
`app/config.py`. The ones you are most likely to touch:

| Variable | Default | Effect |
|---|---|---|
| `ASR_LANGUAGE` | `az` | Whisper language hint. |
| `ASR_PROMPT` | Azerbaijani + tech/marketing term list | Biases ASR toward your vocabulary. |
| `VAD_PAD_MS` | `65` | Padding around every speech boundary. |
| `SILENCE_TRIM_MS` | `350` | Minimum silence worth cutting. |
| `CUT_PAD_MS` | `40` | Head/tail kept around a run of kept words. |
| `AUDIO_JOIN_MODE` | `crossfade` | `crossfade` or `fade`. |
| `SUBTITLE_HIGHLIGHT_COLOR` | `#FFE94D` | Active-word colour (also per-render in the UI). |
| `SUBTITLE_POSITION` | `lower_third` | `lower_third` or `center`. |
| `VIDEO_CRF` / `VIDEO_PRESET` | `20` / `veryfast` | Quality vs speed. |
| `LOG_LEVEL` | `INFO` | Root logger level; `DEBUG` also dumps the ffmpeg filtergraph. |

## 9. API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | Liveness. |
| `GET` | `/api/capabilities` | FFmpeg build, chosen subtitle engine, key/config status. |
| `POST` | `/api/projects` | Multipart upload; starts ingest. → `Project` |
| `GET` | `/api/projects` | Recent projects. |
| `GET` | `/api/projects/{id}` | Full project incl. transcript + B-roll. |
| `DELETE` | `/api/projects/{id}` | Remove project and all artefacts. |
| `GET` | `/api/projects/{id}/job` | Latest ingest job (progress polling). |
| `GET` | `/api/projects/{id}/timeline` | Timeline for the stored edit. |
| `PUT` | `/api/projects/{id}/edit` | Save struck-out words → `{project, timeline}`. |
| `POST` | `/api/projects/{id}/broll/{sid}/search` | Re-search Pexels with a new query. |
| `POST` | `/api/projects/{id}/broll/{sid}/decision` | `accepted` / `dismissed` / `pending`. |
| `POST` | `/api/projects/{id}/render` | Start a render → `Job`. |
| `GET` | `/api/jobs/{job_id}` | Job status/progress/error. |
| `GET` | `/api/projects/{id}/source` | Source stream (HTTP Range supported). |
| `GET` | `/api/projects/{id}/output` | Rendered stream (Range supported). |
| `GET` | `/api/projects/{id}/download` | Rendered file as an attachment. |

Errors are always `{"error": {"code", "message", "detail?", ...}}` with a stable
`code` (`missing_audio_stream`, `no_speech_detected`, `upstream_rate_limited`,
`invalid_edit`, `ffmpeg_capability_missing`, …) so the UI can react without
string-matching.

## 10. Tests

```bash
pip install pytest
pytest tests -q
```

`tests/` is network-free: timeline arithmetic, silence trimming, subtitle
generation/escaping, ASS validity, and ffprobe error paths (it synthesises its
own fixtures with FFmpeg). The API/render integration checks require live Groq
and Pexels keys and are documented in `tests/README.md`.

## 11. Licence / attribution

Stock footage comes from Pexels under the
[Pexels licence](https://www.pexels.com/license/); `BrollClip` carries `author`,
`author_url` and `page_url` so you can credit creators.
