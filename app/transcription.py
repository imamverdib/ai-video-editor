"""Groq Whisper transcription + post-ASR terminology normalisation.

The transcription step uploads the silence-stripped FLAC chunks produced by
:mod:`app.audio_processing`, asks for word-level timestamps, and maps every
word back onto the original video timeline through the
:class:`~app.audio_processing.CompactTimeline`.

The normalisation step then repairs phonetic misspellings of Azerbaijani and
English tech/marketing terms. It is deliberately implemented as a *token-for-
token substitution*: the LLM may only replace the surface form of an existing
token, never insert, delete, reorder or retime one. That invariant is enforced
in code, not merely requested in the prompt.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import httpx

from .agy_provider import agy_chat, session_scope
from .audio_processing import AsrChunk, CompactTimeline
from .config import settings
from .errors import ConfigurationError, RateLimitedError, UpstreamAPIError
from .runtime import check_cancelled, interruptible_sleep
from .schemas import Sentence, Transcript, Word

log = logging.getLogger(__name__)

ProgressFn = Callable[[float, str], None]

_SENTENCE_END = re.compile(r"[.!?…]+[\"'»)\]]?$")
_TOKEN_SPLIT = re.compile(r"\s+")

#: Words longer than this many characters are never produced by a single
#: Whisper token in Azerbaijani; a "fix" that long is a hallucination.
_MAX_TOKEN_CHARS = 48

#: Several Groq-hosted models emit chain-of-thought inside `content`
#: (Qwen3 uses <think> tags). It must be removed before JSON extraction.
_THINK_BLOCK = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN = re.compile(r"<think(?:ing)?>.*", re.DOTALL | re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    """Drop <think> blocks, including an unterminated trailing one."""
    cleaned = _THINK_BLOCK.sub("", text)
    if "</think" not in cleaned.lower():
        cleaned = _THINK_OPEN.sub("", cleaned)
    return cleaned.strip()


def reasoning_kwargs(model: str) -> dict[str, Any]:
    """Per-family knob that keeps reasoning from eating the token budget.

    Qwen3 on Groq accepts only ``none``/``default``; gpt-oss accepts the
    usual low/medium/high. Left empty for anything else so an unknown model
    never 400s on an unsupported parameter.
    """
    name = model.lower()
    if "qwen" in name:
        return {"reasoning_effort": "none"}
    if "gpt-oss" in name:
        return {"reasoning_effort": "low"}
    return {}


# --------------------------------------------------------------------- clients
def _groq_client():
    if not settings.groq_api_key:
        raise ConfigurationError(
            "GROQ_API_KEY is not set.",
            detail="Add GROQ_API_KEY=... to your .env file and restart the server.",
        )
    try:
        from groq import Groq
    except ImportError as exc:  # pragma: no cover
        raise ConfigurationError(
            "The 'groq' package is not installed.",
            detail="Run: pip install -r requirements.txt",
        ) from exc
    return Groq(api_key=settings.groq_api_key, max_retries=0, timeout=60.0)


def _anthropic_client():
    if not settings.anthropic_api_key:
        raise ConfigurationError(
            "ANTHROPIC_API_KEY is not set but NORMALIZER_PROVIDER=anthropic.",
            detail="Set ANTHROPIC_API_KEY or switch NORMALIZER_PROVIDER to groq/off.",
        )
    try:
        from anthropic import Anthropic
    except ImportError as exc:
        raise ConfigurationError(
            "The 'anthropic' package is not installed.",
            detail="Run: pip install anthropic  (or set NORMALIZER_PROVIDER=groq)",
        ) from exc
    return Anthropic(api_key=settings.anthropic_api_key, max_retries=0)


# ----------------------------------------------------------------- retry logic
#: Groq's ``x-ratelimit-reset-*`` headers use a compound duration ("2m59.56s").
_DURATION = re.compile(
    r"^(?:(?P<h>\d+(?:\.\d+)?)h)?(?:(?P<m>\d+(?:\.\d+)?)m)?"
    r"(?:(?P<s>\d+(?:\.\d+)?)(?:s|ms)?)?$",
    re.IGNORECASE,
)

#: No server hint is ever slept for longer than this. A per-day quota reset can
#: be hours away; sleeping it verbatim wedges the whole job pool.
_MAX_RETRY_SLEEP = 60.0


def parse_retry_hint(raw: object) -> float | None:
    """Parse a Retry-After / rate-limit-reset value into seconds.

    Accepts bare seconds ("3600"), the ``2.5s`` form, Groq's compound duration
    ("1m30s", "7m59.56s") and an RFC 9110 HTTP-date. Returns ``None`` only when
    nothing matches, so a usable hint is never silently discarded.
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        return max(0.0, float(text))
    except ValueError:
        pass

    match = _DURATION.match(text)
    if match and any(match.group(name) for name in ("h", "m", "s")):
        hours = float(match.group("h") or 0.0)
        minutes = float(match.group("m") or 0.0)
        seconds = float(match.group("s") or 0.0)
        if text.lower().endswith("ms"):
            hours = minutes = 0.0
            seconds = float(match.group("s") or 0.0) / 1000.0
        return max(0.0, hours * 3600.0 + minutes * 60.0 + seconds)

    try:
        from email.utils import parsedate_to_datetime

        target = parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None
    if target is None:  # pragma: no cover - parsedate_to_datetime raises instead
        return None
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc) if target.tzinfo else datetime.now()  # noqa: UP017
    return max(0.0, (target - now).total_seconds())


def _retry_after(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    for key in ("retry-after", "x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"):
        hint = parse_retry_hint(headers.get(key))
        if hint is not None:
            return hint
    return None


def _exception_status(exc: Exception) -> int | None:
    return (
        getattr(exc, "status_code", None)
        or getattr(exc, "status", None)
        or getattr(getattr(exc, "response", None), "status_code", None)
    )


def is_transient(exc: Exception) -> bool:
    """Whether ``exc`` is worth retrying.

    Classification is driven by the HTTP status, never by the bare
    ``APIStatusError`` class name: the SDK raises that base class for every
    status it has no subclass for, 413 Payload Too Large included. Retrying a
    413 just re-uploads an oversized file five times and then reports a generic
    upstream failure.
    """
    status = _exception_status(exc)
    if isinstance(status, int):
        return status in (408, 429) or status >= 500
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError)):
        return True
    return type(exc).__name__ in {
        "RateLimitError",
        "APIConnectionError",
        "APITimeoutError",
        "APIConnectionTimeoutError",
        "InternalServerError",
    }


def with_retries(
    call: Callable[[], Any],
    *,
    provider: str,
    what: str,
    max_attempts: int,
    budget_s: float | None = None,
) -> Any:
    """Run ``call`` with exponential backoff on rate limits / transient errors.

    ``budget_s`` bounds the *total* time spent waiting between attempts, so a
    server that asks for a two-hour wait cannot pin a job worker: the call fails
    fast with the hint attached instead.
    """
    last_exc: Exception | None = None
    waited = 0.0
    started = time.monotonic()
    for attempt in range(1, max_attempts + 1):
        check_cancelled()
        try:
            return call()
        except Exception as exc:  # noqa: BLE001 - re-raised below after classification
            last_exc = exc
            name = type(exc).__name__
            if not is_transient(exc) or attempt == max_attempts:
                break
            hinted = _retry_after(exc)
            if hinted:
                delay = min(_MAX_RETRY_SLEEP, hinted)
            else:
                delay = min(_MAX_RETRY_SLEEP, 1.5 * (2 ** (attempt - 1)))
            delay += random.uniform(0, 0.75)  # noqa: S311 - jitter only
            if budget_s is not None and max(waited, time.monotonic() - started) + delay > budget_s:
                log.warning(
                    "%s %s: %.0fs retry budget exhausted after attempt %d; giving up.",
                    provider,
                    what,
                    budget_s,
                    attempt,
                )
                break
            waited += delay
            log.warning(
                "%s %s failed (%s, attempt %d/%d); retrying in %.1fs",
                provider,
                what,
                name,
                attempt,
                max_attempts,
                delay,
            )
            interruptible_sleep(delay)

    assert last_exc is not None
    status = _exception_status(last_exc)
    detail = str(last_exc)[:800]
    if type(last_exc).__name__ == "RateLimitError" or status == 429:
        raise RateLimitedError(
            f"{provider} rate limit reached while {what}.",
            provider=provider,
            retry_after=_retry_after(last_exc),
            detail=detail,
        ) from last_exc
    raise UpstreamAPIError(
        f"{provider} request failed while {what}.",
        provider=provider,
        status=status if isinstance(status, int) else None,
        detail=detail,
    ) from last_exc


def groq_chat(
    system: str,
    user: str,
    *,
    what: str,
    max_tokens: int = 2048,
    temperature: float = 0.0,
) -> str:
    """Chat-completion against Groq, walking the configured model chain.

    ``response_format`` is deliberately NOT used: several Groq-hosted reasoning
    models fail JSON-mode validation, and both call sites parse JSON out of free
    text robustly anyway. A 404 (retired / not-entitled model id) advances to the
    next model in the chain instead of failing the request.
    """
    client = _groq_client()
    chain = settings.groq_llm_chain
    last_error: Exception | None = None

    for model in chain:

        def call(model: str = model) -> Any:
            return client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                **reasoning_kwargs(model),
            )

        try:
            response = with_retries(
                call,
                provider="Groq",
                what=what,
                max_attempts=settings.llm_max_retries,
                budget_s=settings.llm_retry_budget_s,
            )
        except UpstreamAPIError as exc:
            unavailable = exc.status in (400, 404) and (
                "model_not_found" in (exc.detail or "")
                or "does not exist" in (exc.detail or "")
                or "decommissioned" in (exc.detail or "")
            )
            if (
                unavailable or exc.status in (408, 429) or (exc.status or 0) >= 500
            ) and model != chain[-1]:
                log.warning("Groq model '%s' unavailable; trying the next fallback.", model)
                last_error = exc
                continue
            raise
        if model != chain[0]:
            log.info("Groq request served by fallback model '%s'.", model)
        choice = response.choices[0]
        if getattr(choice, "finish_reason", None) == "length":
            log.warning(
                "Groq model '%s' hit the %d-token limit while %s; the reply may be truncated.",
                model,
                max_tokens,
                what,
            )
        content = strip_reasoning(choice.message.content or "")
        if not content or getattr(choice, "finish_reason", None) == "length":
            last_error = UpstreamAPIError("Incomplete model response", provider="Groq")
            if model != chain[-1]:
                continue
            raise last_error
        return content

    assert last_error is not None
    raise last_error


# ------------------------------------------------------------------ ASR parsing
def _as_dict(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return item
    for method in ("model_dump", "dict", "to_dict"):
        fn = getattr(item, method, None)
        if callable(fn):
            try:
                return fn()
            except TypeError:
                continue
    return {
        key: getattr(item, key) for key in ("word", "text", "start", "end") if hasattr(item, key)
    }


def _extract_words(response: Any) -> list[dict[str, Any]]:
    payload = _as_dict(response)
    words = payload.get("words")
    if not words:
        # Fall back to segment-level timings, spread evenly across the tokens.
        derived: list[dict[str, Any]] = []
        for segment in payload.get("segments") or []:
            seg = _as_dict(segment)
            text = str(seg.get("text") or "").strip()
            start, end = float(seg.get("start") or 0.0), float(seg.get("end") or 0.0)
            tokens = [t for t in _TOKEN_SPLIT.split(text) if t]
            if not tokens or end <= start:
                continue
            step = (end - start) / len(tokens)
            for offset, token in enumerate(tokens):
                derived.append(
                    {
                        "word": token,
                        "start": start + offset * step,
                        "end": start + (offset + 1) * step,
                    }
                )
        return derived
    return [_as_dict(word) for word in words]


def _transcribe_chunk_raw(chunk: AsrChunk) -> tuple[list[dict[str, Any]], str | None]:
    """Transcribe one chunk, returning its words and the language Whisper heard."""
    client = _groq_client()

    def call() -> Any:
        with chunk.path.open("rb") as handle:
            return client.audio.transcriptions.create(
                file=(chunk.path.name, handle.read()),
                model=settings.groq_asr_model,
                response_format="verbose_json",
                timestamp_granularities=["word", "segment"],
                language=settings.asr_language,
                prompt=settings.asr_prompt,
                temperature=settings.asr_temperature,
            )

    response = with_retries(
        call,
        provider="Groq",
        what=f"transcribing chunk {chunk.index + 1}",
        max_attempts=settings.asr_max_retries,
        budget_s=settings.asr_retry_budget_s,
    )
    detected = _as_dict(response).get("language")
    return _extract_words(response), (str(detected) if detected else None)


def transcribe_chunk(chunk: AsrChunk) -> list[dict[str, Any]]:
    """Transcribe one compacted FLAC chunk with word-level timestamps."""
    return _transcribe_chunk_raw(chunk)[0]


def transcribe_chunks(
    chunks: Sequence[AsrChunk],
    timeline: CompactTimeline,
    *,
    on_progress: ProgressFn | None = None,
    languages: list[str] | None = None,
) -> list[Word]:
    """Transcribe every chunk and return words on the **source** timeline."""
    words: list[Word] = []
    for position, chunk in enumerate(chunks):
        if on_progress:
            on_progress(position / len(chunks), f"Transcribing chunk {position + 1}/{len(chunks)}")
        raw_words, detected = _transcribe_chunk_raw(chunk)
        if detected and languages is not None:
            languages.append(detected)
        for raw in raw_words:
            text = str(raw.get("word") or raw.get("text") or "").strip()
            if not text:
                continue
            try:
                local_start = float(raw.get("start", 0.0))
                local_end = float(raw.get("end", local_start))
            except (TypeError, ValueError):
                continue

            compact_start = chunk.compact_start + max(0.0, local_start)
            compact_end = chunk.compact_start + max(local_start, local_end)
            compact_start = min(compact_start, chunk.compact_end)
            compact_end = min(max(compact_end, compact_start), chunk.compact_end)

            source_start = timeline.to_source(compact_start)
            source_end = timeline.to_source(compact_end)
            # The compacted audio joins two speech segments with a 4 ms fade, so
            # Whisper regularly emits a word straddling a join. Mapping start and
            # end independently across that discontinuity stretches the word over
            # the whole removed silence (a 0.2s word became 29.2s in testing),
            # which freezes its caption and drags the dead air back into the cut.
            # A word can never be longer in source time than in compact time.
            start_segment = timeline.segment_index(compact_start) if timeline.segments else -1
            if start_segment >= 0 and timeline.segment_index(compact_end) != start_segment:
                segment = timeline.segments[start_segment]
                source_end = min(
                    segment.end,
                    source_start + (compact_end - compact_start),
                )
            if source_end <= source_start:
                # Zero-length after mapping (word sat exactly on a seam):
                # give it a minimum 40 ms so subtitles stay visible.
                source_end = source_start + 0.04
            words.append(
                Word(
                    id=len(words),
                    text=text,
                    start=round(source_start, 4),
                    end=round(source_end, 4),
                    sentence_id=0,
                )
            )
    if on_progress:
        on_progress(1.0, "Transcription complete")

    words.sort(key=lambda w: (w.start, w.end))
    # Whisper's word boundaries are approximate and neighbouring words can
    # overlap by tens of milliseconds. Trim the earlier word's end so the
    # sequence is strictly non-overlapping: starts are the more reliable edge,
    # and the timeline/subtitle code assumes monotonicity.
    for previous, current in zip(words, words[1:], strict=False):
        if previous.end > current.start:
            previous.end = round(max(previous.start + 0.02, current.start), 4)
    for index, word in enumerate(words):
        word.id = index
    return words


# ------------------------------------------------------------ sentence grouping
def group_sentences(
    words: Sequence[Word],
    *,
    max_words: int = 18,
    gap_threshold: float = 0.7,
) -> list[Sentence]:
    """Group words into readable sentences using punctuation and pauses."""
    sentences: list[Sentence] = []
    current: list[Word] = []

    def flush() -> None:
        if not current:
            return
        sentence_id = len(sentences)
        for word in current:
            word.sentence_id = sentence_id
        sentences.append(
            Sentence(
                id=sentence_id,
                word_ids=[w.id for w in current],
                text=" ".join(w.text for w in current),
                start=current[0].start,
                end=current[-1].end,
            )
        )
        current.clear()

    for index, word in enumerate(words):
        current.append(word)
        next_word = words[index + 1] if index + 1 < len(words) else None
        ends_sentence = bool(_SENTENCE_END.search(word.text))
        long_pause = next_word is not None and (next_word.start - word.end) >= gap_threshold
        too_long = len(current) >= max_words
        if next_word is None or ends_sentence or (long_pause and len(current) >= 3) or too_long:
            flush()
    flush()
    return sentences


# ------------------------------------------------- terminology normalisation
_NORMALIZER_SYSTEM = (
    "You are a transcription proofreader for Azerbaijani speech that mixes in "
    "English technology and marketing vocabulary.\n"
    "You receive a numbered list of single word tokens from an ASR system.\n"
    "Your ONLY job: fix phonetic misspellings and wrong casing of real terms.\n\n"
    "Hard rules:\n"
    "1. Return ONLY tokens you are changing.\n"
    "2. A replacement must be exactly ONE token - no spaces, no splitting, no merging.\n"
    "3. Never translate, rephrase, summarise, censor or reorder anything.\n"
    "4. Never touch ordinary Azerbaijani words that are already spelled correctly.\n"
    "5. Keep any trailing punctuation that the original token had.\n"
    "6. If nothing needs fixing, return an empty list.\n"
    "7. NEVER anglicise a loanword that is already spelled correctly in "
    "Azerbaijani (e.g. leave 'alqoritm', 'marketinq' and 'kontent' exactly as "
    "they are - do NOT turn them into 'algorithm', 'marketing' or 'content').\n\n"
    'Respond with JSON only: {"fixes": [{"i": <index>, "t": "<corrected token>"}]}'
)


def _normalizer_user_prompt(
    tokens: list[tuple[int, str]],
    glossary: Iterable[str],
    preserve: Iterable[str] = (),
) -> str:
    listing = "\n".join(f"{index}: {text}" for index, text in tokens)
    context = " ".join(text for _, text in tokens)
    parts = [
        "English terms that keep their English spelling:\n" + ", ".join(glossary),
    ]
    preserve = list(preserve)
    if preserve:
        parts.append(
            "Azerbaijani loanwords that are ALREADY CORRECT - never change these:\n"
            + ", ".join(preserve)
        )
    parts.append("Sentence context (read-only, for disambiguation):\n" + context)
    parts.append("Tokens to proofread:\n" + listing)
    return "\n\n".join(parts)


def _brace_objects(text: str) -> Iterable[str]:
    """Yield every top-level ``{...}`` slice, ignoring braces inside strings.

    A greedy ``\\{.*\\}`` match spans from the first brace to the *last* one, so a
    model that emits a preamble object or repeats itself produced an unparseable
    slice and the whole batch was silently discarded.
    """
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                yield text[start : index + 1]
                start = -1
            elif depth < 0:
                depth = 0


def _parse_fixes(raw: str) -> list[dict[str, Any]]:
    text = strip_reasoning(raw)
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()

    candidates: list[str] = []
    if text.startswith("{"):
        candidates.append(text)
    candidates.extend(_brace_objects(text))
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        fixes = payload.get("fixes")
        if isinstance(fixes, list):
            return fixes
    if text:
        log.warning('Normaliser reply contained no parseable {"fixes": [...]} object.')
    return []


def gemini_chat(
    system: str,
    user: str,
    *,
    what: str,
    max_tokens: int = 2048,
    temperature: float = 0.0,
) -> str:
    """Call Google Gemini API (e.g. Gemini 2.5 / 3.7 Flash) with fallback chain."""
    key = (
        settings.gemini_api_key
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
    )
    if not key:
        if settings.groq_api_key:
            log.info("GEMINI_API_KEY not set; using Groq %s for %s", settings.groq_llm_model, what)
            return groq_chat(
                system, user, what=what, max_tokens=max_tokens, temperature=temperature
            )
        raise ConfigurationError("GEMINI_API_KEY is not set. Set GEMINI_API_KEY in .env.")

    models = [settings.gemini_model] + [
        m.strip() for m in settings.gemini_fallbacks.split(",") if m.strip()
    ]
    last_error: Exception | None = None

    for model in models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        payload = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "responseMimeType": "application/json",
            },
        }

        try:
            with httpx.Client(timeout=45.0) as client:
                res = client.post(url, json=payload, headers={"x-goog-api-key": key})
                if res.status_code == 429:
                    retry_after = res.headers.get("retry-after")
                    raise RateLimitedError(
                        "Gemini rate limited",
                        provider="Gemini",
                        retry_after=parse_retry_hint(retry_after) if retry_after else None,
                    )
                if res.status_code != 200:
                    raise UpstreamAPIError(
                        f"Gemini API returned HTTP {res.status_code}",
                        provider="Gemini",
                        detail="Provider request failed.",
                        status=res.status_code,
                    )
                data = res.json()
                candidates = data.get("candidates") or []
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    text = strip_reasoning(
                        "".join(p.get("text", "") for p in parts if not p.get("thought"))
                    )
                    if text and candidates[0].get("finishReason") in (None, "STOP"):
                        return text
                raise UpstreamAPIError("Gemini boş və ya yarımçıq cavab verdi.", provider="Gemini")
        except Exception as exc:
            check_cancelled()
            log.warning("Gemini model failed for %s (%s)", what, type(exc).__name__)
            last_error = exc
            continue

    if settings.groq_api_key:
        log.warning("All Gemini models failed; falling back to Groq for %s", what)
        return groq_chat(system, user, what=what, max_tokens=max_tokens, temperature=temperature)
    assert last_error is not None
    raise last_error


def _call_normalizer(prompt: str) -> str:
    provider = settings.normalizer_provider
    if provider == "agy":
        return agy_chat(_NORMALIZER_SYSTEM, prompt, key="fixes")
    if provider == "gemini":
        return gemini_chat(
            _NORMALIZER_SYSTEM,
            prompt,
            what="normalising terminology with Gemini",
            max_tokens=2048,
            temperature=0.0,
        )
    if provider == "anthropic":
        client = _anthropic_client()

        def call() -> Any:
            return client.messages.create(
                model=settings.anthropic_model,
                max_tokens=2048,
                system=_NORMALIZER_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )

        response = with_retries(
            call,
            provider="Anthropic",
            what="normalising terminology",
            max_attempts=settings.llm_max_retries,
            budget_s=settings.llm_retry_budget_s,
        )
        return "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )

    return groq_chat(
        _NORMALIZER_SYSTEM,
        prompt,
        what="normalising terminology",
        max_tokens=2048,
        temperature=0.0,
    )


_PROOFREADER_SYSTEM = (
    "Sən Azərbaycan dili və nitq analizi üzrə ekspert redaktorusan.\n"
    "Whisper AI nitqi mətnə çevirərkən (ASR transkripsiyası zamanı) bəzi sözləri yanlış "
    "tələffüzə, sürətli danışığa və ya səs oxşarlığına görə TAMAMİLƏ YANLIŞ, "
    "kontekstə uyğun gəlməyən başqa bir söz kimi yaza bilir "
    "(Məsələn: 'təsirsiz olur' əvəzinə 'həssəsöz / həssasöz olur', 'çirkli paltarlar' "
    "əvəzinə 'çiçdi / çeşdi paltarlar', 'paltaryuyan' əvəzinə 'paltaryan', "
    "'baxış bucağı' əvəzinə 'bağış bucağı', 'məhsuldarlıq' əvəzinə 'məsul darlıq').\n\n"
    "SƏNİN VƏZİFƏN:\n"
    "1. Əvvəlcə sənə verilən BÜTÜN MƏTNİ (TAM VİDEO MƏTNİ) oxuyub videonun mövzusunu, "
    "məğzini və danışanın nə demək istədiyini dərindən başa düşmək.\n"
    "2. Sonra nömrələnmiş sözlərə (tokenlərə) baxaraq kontekstə mənaca qəti uyğun gəlməyən, "
    "lakin tələffüzcə oxşar olan bütün ASR çaşqınlıqlarını, səhv sözləri və orfoqrafik "
    "qüsurları tapmaq.\n"
    "3. Hər bir səhv söz üçün kontekstə və cümlənin mənasına ən uyğun düzgün söz "
    "TÖVSİYƏSİNİ (suggestion) vermək.\n\n"
    "QAYDALAR:\n"
    "- YALNIZ mənaca və ya orfoqrafik cəhətdən həqiqətən səhv olan sözləri seç.\n"
    "- Kontekstə tam uyğun gələn düzgün sözlərə toxunma.\n"
    "- Orijinal durğu işarələrini (., ?, ! və s.) qoru.\n"
    "- Hər bir düzəliş üçün QISA səbəb yaz (maksimum 4-6 söz, məs: 'Məna: həssəsöz ➔ təsirsiz').\n"
    "- Cavabı YALNIZ JSON formatında ver:\n"
    '{"suggestions": [{"i": <token_id>, "t": "<düzgün söz>", "r": "<qısa izahat>"}]}\n'
    'Əgər düzəliş yoxdursa, {"suggestions": []} qaytar.'
)


_AZ_COMMON_FIXES: dict[str, tuple[str, str]] = {
    "həssəsöz": ("təsirsiz", "Məna: həssəsöz ➔ təsirsiz"),
    "həssasöz": ("təsirsiz", "Məna: həssasöz ➔ təsirsiz"),
    "çiçdi": ("çirkli", "Məna: çiçdi ➔ çirkli"),
    "çeşdi": ("çirkli", "Məna: çeşdi ➔ çirkli"),
    "paltaryan": ("paltaryuyan", "Orfoqrafiya: paltaryan ➔ paltaryuyan"),
    "qarab": ("xarab", "Tələffüz: qarab ➔ xarab"),
    "zəhv": ("zəng", "ASR qüsuru: zəhv ➔ zəng"),
    "zəncimi": ("zəngimi", "Tələffüz: zəncimi ➔ zəngimi"),
    "aşmaz": ("açmaz", "Tələffüz: aşmaz ➔ açmaz"),
    "nuhayib": ("nə isə", "ASR qüsuru: nuhayib ➔ nə isə"),
    "dərhaa": ("dərhal", "Orfoqrafiya: dərhaa ➔ dərhal"),
    "düzətmək": ("düzəltmək", "Orfoqrafiya: düzətmək ➔ düzəltmək"),
    "nəsiyyət": ("nəsihət", "Orfoqrafiya: nəsiyyət ➔ nəsihət"),
    "şuur": ("şüur", "Orfoqrafiya: şuur ➔ şüur"),
    "şuurla": ("şüurla", "Orfoqrafiya: şuurla ➔ şüurla"),
    "şuurumuzla": ("şüurumuzla", "Orfoqrafiya: şuurumuzla ➔ şüurumuzla"),
    "inammıram": ("inanmıram", "Orfoqrafiya: inammıram ➔ inanmıram"),
    "inammırıq": ("inanmırıq", "Orfoqrafiya: inammırıq ➔ inanmırıq"),
    "inammırsan": ("inanmırsan", "Orfoqrafiya: inammırsan ➔ inanmırsan"),
    "inammır": ("inanmır", "Orfoqrafiya: inammır ➔ inanmır"),
    "inammırlar": ("inanmırlar", "Orfoqrafiya: inammırlar ➔ inanmırlar"),
    "məndəndə": ("məndən də", "Ayrı yazılış: -də ədatı"),
    "səndəndə": ("səndən də", "Ayrı yazılış: -də ədatı"),
    "ondanda": ("ondan da", "Ayrı yazılış: -da ədatı"),
    "bizdəndə": ("bizdən də", "Ayrı yazılış: -də ədatı"),
    "sizdəndə": ("sizdən də", "Ayrı yazılış: -də ədatı"),
    "onlardanda": ("onlardan da", "Ayrı yazılış: -da ədatı"),
    "hərşey": ("hər şey", "Ayrı yazılış: hər şey"),
    "hərkəs": ("hər kəs", "Ayrı yazılış: hər kəs"),
    "hərvaxt": ("hər vaxt", "Ayrı yazılış: hər vaxt"),
    "heçvaxt": ("heç vaxt", "Ayrı yazılış: heç vaxt"),
    "heçkəs": ("heç kəs", "Ayrı yazılış: heç kəs"),
    "heçnə": ("heç nə", "Ayrı yazılış: heç nə"),
    "heçbir": ("heç bir", "Ayrı yazılış: heç bir"),
    "birşey": ("bir şey", "Ayrı yazılış: bir şey"),
    "biraz": ("bir az", "Ayrı yazılış: bir az"),
    "gəlirəmmi": ("gəlirəm mi", "Ayrı yazılış: -mi sual ədatı"),
    "gedirəmmi": ("gedirəm mi", "Ayrı yazılış: -mi sual ədatı"),
    "olarmı": ("olar mı", "Ayrı yazılış: -mı sual ədatı"),
    "süniintellect": ("süni intellekt", "Termin: süni intellekt"),
    "süniintellekt": ("süni intellekt", "Termin: süni intellekt"),
}


def _builtin_check_tokens(tokens: list[tuple[int, str]]) -> list[dict[str, Any]]:
    results = []
    punc_re = re.compile(r"([.,!?;:…]+)$")
    for idx, text in tokens:
        clean = text.strip()
        match = punc_re.search(clean)
        trailing = match.group(1) if match else ""
        bare = clean[: len(clean) - len(trailing)].lower() if trailing else clean.lower()
        if bare in _AZ_COMMON_FIXES:
            replacement_bare, reason = _AZ_COMMON_FIXES[bare]
            if clean and clean[0].isupper():
                replacement = replacement_bare[0].upper() + replacement_bare[1:] + trailing
            else:
                replacement = replacement_bare + trailing
            results.append({"i": idx, "t": replacement, "r": reason})
    return results


def _proofreader_user_prompt(
    full_text: str,
    tokens: list[tuple[int, str]],
    glossary: Sequence[str] | None = None,
) -> str:
    parts = [
        "TAM VİDEO MƏTNİ (Bütöv kontekst və mövzu):",
        f'"""\n{full_text.strip()}\n"""',
        "",
        "NÖMRƏLƏNMİŞ SÖZLƏR (Aşağıdakı tokenləri kontekstə və tələffüzə görə yoxla):",
    ]
    for idx, text in tokens:
        parts.append(f"[{idx}, {json.dumps(text, ensure_ascii=False)}]")
    if glossary:
        parts.append("")
        parts.append("Düzgün terminlər:")
        parts.append(", ".join(glossary[:40]))
    return "\n".join(parts)


def _call_proofreader(prompt: str) -> str:
    provider = settings.normalizer_provider
    if provider == "agy":
        return agy_chat(_PROOFREADER_SYSTEM, prompt, key="suggestions")
    if provider == "gemini":
        return gemini_chat(
            _PROOFREADER_SYSTEM,
            prompt,
            what="generating spelling suggestions with Gemini",
            max_tokens=3000,
            temperature=0.0,
        )
    if provider == "anthropic":
        client = _anthropic_client()

        def call() -> Any:
            return client.messages.create(
                model=settings.anthropic_model,
                max_tokens=3000,
                system=_PROOFREADER_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )

        response = with_retries(
            call,
            provider="Anthropic",
            what="generating spelling suggestions",
            max_attempts=settings.llm_max_retries,
            budget_s=settings.llm_retry_budget_s,
        )
        return "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )

    return groq_chat(
        _PROOFREADER_SYSTEM,
        prompt,
        what="generating spelling suggestions",
        max_tokens=3000,
        temperature=0.0,
    )


def _parse_suggestions(raw: str) -> list[dict[str, Any]]:
    text = strip_reasoning(raw)
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()

    candidates: list[str] = []
    if text.startswith("{"):
        candidates.append(text)
    candidates.extend(_brace_objects(text))
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        items = payload.get("suggestions") or payload.get("fixes")
        if isinstance(items, list):
            return items
    return []


@session_scope()
def generate_spelling_suggestions(
    words: list[Word],
    *,
    warnings: list[str] | None = None,
) -> tuple[list[Word], int]:
    """Inspect words in full transcript context and attach AI spelling/grammar suggestions.

    Populates ``word.suggestion`` and ``word.suggestion_reason`` on words that
    likely have spelling or acoustic transcription mistakes. Returns the word list and count.
    """
    if not words or settings.normalizer_provider == "off":
        return words, 0

    for word in words:
        word.suggestion = None
        word.suggestion_reason = None
    full_text = " ".join(word.text for word in words)
    batch_size = max(100, settings.normalizer_batch_size)
    count = 0

    for offset in range(0, len(words), batch_size):
        batch = words[offset : offset + batch_size]
        tokens = [(word.id, word.text) for word in batch]
        by_id = {word.id: word for word in batch}

        found_suggestions: dict[int, tuple[str, str]] = {}
        for item in _builtin_check_tokens(tokens):
            found_suggestions[item["i"]] = (item["t"], item["r"])

        try:
            raw = _call_proofreader(
                _proofreader_user_prompt(full_text, tokens, settings.glossary())
            )
            for item in _parse_suggestions(raw):
                try:
                    index = int(item.get("i"))
                except (TypeError, ValueError):
                    continue
                rep = str(item.get("t") or "").strip()
                reason = str(item.get("r") or "").strip()
                if rep:
                    found_suggestions[index] = (rep, reason)
        except Exception as exc:
            check_cancelled()
            if warnings is not None:
                warnings.append(f"Orfoqrafik AI yoxlaması tamamlanmadı (batch {offset}).")
            log.warning("AI proofreader skipped for batch at %d: %s", offset, type(exc).__name__)

        for index, (replacement, reason) in found_suggestions.items():
            word = by_id.get(index)
            if word is None:
                continue
            if not _acceptable_replacement(
                word.text, replacement, allow_spaces=True, allow_semantic=True
            ):
                continue
            word.suggestion = unicodedata.normalize("NFC", replacement)
            word.suggestion_reason = reason or f"Düzəliş: {word.text} ➔ {replacement}"
            count += 1

    log.info("Generated %d spelling/grammar suggestions", count)
    return words, count


#: Unicode categories that must never reach a burned-in caption: control and
#: format characters (U+200B ZERO WIDTH SPACE, U+202E RIGHT-TO-LEFT OVERRIDE,
#: U+FEFF ...) plus every kind of separator. Python's ``\s`` matches none of the
#: format characters, so the whitespace test alone let them through: the Pillow
#: rasteriser then draws a .notdef box, or silently reverses the caption line.
_FORBIDDEN_CATEGORIES = frozenset({"Cc", "Cf", "Co", "Cs", "Cn", "Zs", "Zl", "Zp"})


def _acceptable_replacement(
    original: str,
    replacement: str,
    *,
    allow_spaces: bool = False,
    allow_semantic: bool = False,
) -> bool:
    replacement = unicodedata.normalize("NFC", replacement.strip())
    if not replacement or replacement.lower() == original.lower():
        return False
    if len(replacement) > _MAX_TOKEN_CHARS:
        return False
    if not allow_spaces and _TOKEN_SPLIT.search(replacement):
        return False
    if allow_spaces and len(replacement.split()) > 3:
        return False
    for char in replacement:
        cat = unicodedata.category(char)
        if cat in _FORBIDDEN_CATEGORIES and not (allow_spaces and cat == "Zs" and char == " "):
            return False
    if allow_semantic:
        return len(replacement) <= max(12, int(len(original) * 3.5)) + 12
    if original.lower().strip(".,!?;:…") in settings.az_preserve():
        return False
    if replacement.lower().strip(".,!?;:…") not in {x.lower() for x in settings.glossary()}:
        return False
    # Guard against the LLM swapping in an unrelated word: a phonetic fix keeps
    # roughly the same length.
    return len(replacement) <= max(4, int(len(original) * 2.8)) + 6


@session_scope()
def normalize_terminology(
    words: list[Word],
    *,
    warnings: list[str] | None = None,
) -> tuple[list[Word], int]:
    """Fix phonetic spelling errors in place, never touching ``start``/``end``.

    Returns the same list plus the number of tokens actually rewritten. Any
    upstream failure is swallowed (logged) - normalisation is a nice-to-have and
    must never cost the user their transcript.
    """
    if settings.normalizer_provider == "off" or not words:
        return words, 0

    glossary = settings.glossary()
    batch_size = max(20, settings.normalizer_batch_size)
    changed = 0

    for offset in range(0, len(words), batch_size):
        batch = words[offset : offset + batch_size]
        tokens = [(word.id, word.text) for word in batch]
        by_id = {word.id: word for word in batch}
        try:
            raw = _call_normalizer(
                _normalizer_user_prompt(tokens, glossary, settings.az_preserve())
            )
        except (UpstreamAPIError, ConfigurationError) as exc:
            if warnings is not None:
                warnings.append(f"Termin AI yoxlaması tamamlanmadı (batch {offset}).")
            log.warning("Terminology skipped at %d: %s", offset, type(exc).__name__)
            continue

        applied: set[int] = set()
        for fix in _parse_fixes(raw):
            try:
                index = int(fix.get("i"))
            except (TypeError, ValueError):
                continue
            word = by_id.get(index)
            if word is None:
                continue
            if index in applied:
                # A repeated index would overwrite raw_text with the model's own
                # earlier replacement, destroying the genuine ASR token, and
                # would count the same word twice.
                log.warning("Normaliser returned index %d twice; ignoring the repeat.", index)
                continue
            replacement = str(fix.get("t") or "")
            if not _acceptable_replacement(word.text, replacement):
                continue
            bare = word.text.lower().strip(".,!?;:…")
            if bare in settings.az_preserve():
                continue
            trusted = {
                "vörkflou": "workflow",
                "freyimvörk": "framework",
                "storitelinq": "storytelling",
                "huk": "hook",
            }
            if trusted.get(bare) != replacement.lower().strip(".,!?;:…"):
                word.suggestion = replacement.strip()
                word.suggestion_reason = "Termin düzəlişi — təsdiqinizi gözləyir"
                applied.add(index)
                continue
            if word.raw_text is None:
                word.raw_text = word.text
            word.text = unicodedata.normalize("NFC", replacement.strip())
            applied.add(index)
            changed += 1

    log.info("Terminology normalisation rewrote %d/%d tokens", changed, len(words))
    return words, changed


# --------------------------------------------------------------- orchestration
@session_scope()
def build_transcript(
    chunks: Sequence[AsrChunk],
    timeline: CompactTimeline,
    *,
    normalize: bool = True,
    on_progress: ProgressFn | None = None,
) -> Transcript:
    detected: list[str] = []
    words = transcribe_chunks(chunks, timeline, on_progress=on_progress, languages=detected)
    if not words:
        from .errors import NoSpeechDetectedError

        raise NoSpeechDetectedError(
            "Whisper returned no words for this audio.",
            detail="The clip may contain only music or unintelligible noise.",
        )

    warnings: list[str] = []
    normalized_count = 0
    suggestions_count = 0
    if normalize:
        if on_progress:
            on_progress(0.85, "Generating spelling suggestions")
        words, suggestions_count = generate_spelling_suggestions(words, warnings=warnings)
        if on_progress:
            on_progress(0.95, "Normalising terminology")
        words, normalized_count = normalize_terminology(words, warnings=warnings)
        suggestions_count = sum(bool(w.suggestion) for w in words)

    sentences = group_sentences(words)
    # Record what Whisper actually heard rather than what we asked it to decode:
    # a mixed-language Reel was previously always stamped with ASR_LANGUAGE.
    language = detected[0] if detected else settings.asr_language
    if detected and any(item != settings.asr_language for item in detected):
        log.warning(
            "Whisper detected language %s but ASR_LANGUAGE=%s was requested; the "
            "transcript may have been transliterated.",
            ", ".join(sorted(set(detected))),
            settings.asr_language,
        )
    return Transcript(
        language=language,
        text=" ".join(word.text for word in words),
        words=words,
        sentences=sentences,
        normalization_applied=normalized_count > 0,
        normalized_word_count=normalized_count,
        asr_chunks=len(chunks),
        suggestions_count=suggestions_count,
        ai_warnings=warnings,
        ai_provider=settings.normalizer_provider,
        ai_model=settings.agy_model if settings.normalizer_provider == "agy" else None,
    )
