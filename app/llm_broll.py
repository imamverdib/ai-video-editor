"""Contextual B-roll: LLM moment detection + Pexels vertical clip search.

Flow
----
1. An LLM reads the numbered sentences of the transcript and returns 2-3
   high-impact moments that benefit from a visual cutaway, each with an English
   stock-footage search query (English queries hit far more Pexels inventory
   than Azerbaijani ones).
2. Each query is executed against the Pexels video API restricted to portrait
   orientation, and the best 9:16 rendition of every result is kept.
3. The user then accepts / swaps / dismisses each suggestion - nothing is baked
   into the render without an explicit accept.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from pathlib import Path
from typing import Any

import httpx

from .agy_provider import agy_chat
from .config import settings
from .errors import ConfigurationError, RateLimitedError, UpstreamAPIError
from .runtime import check_cancelled
from .schemas import BrollClip, BrollStatus, BrollSuggestion, Transcript
from .transcription import (
    _anthropic_client,
    gemini_chat,
    groq_chat,
    strip_reasoning,
    with_retries,
)

log = logging.getLogger(__name__)

PEXELS_SEARCH_URL = "https://api.pexels.com/videos/search"
_TARGET_ASPECT = 9 / 16

_SYSTEM_PROMPT = (
    "You are a senior short-form video editor who cuts vertical Instagram Reels.\n"
    "You are given the numbered sentences of a Reel's spoken script (Azerbaijani, "
    "mixed with English tech/marketing terms).\n"
    "Your job: pick the best 2-3 moments to cut away to visual B-roll footage that "
    "illustrates what is being said.\n\n"
    "Rules:\n"
    "1. Pick ONLY moments that have a strong, concrete visual equivalent.\n"
    "2. Every search query MUST be concrete, visual, keyword-rich and in "
    "ENGLISH (2-4 words, describing what the camera sees - not the abstract idea).\n\n"
    "Respond with JSON only:\n"
    '{"moments": [{"sentence_id": <int>, "query": "<english search query>", '
    '"keywords": ["<tag>", "<tag>"], "reason": "<why, max 12 words>", '
    '"duration": <seconds 1.5-5>}]}'
)


# ------------------------------------------------------------------ LLM moments
def _call_llm(prompt: str) -> str:
    provider = settings.normalizer_provider
    if provider == "agy":
        return agy_chat(_SYSTEM_PROMPT, prompt, key="moments")
    if provider == "off":
        return '{"moments": []}'
    if provider == "gemini":
        return gemini_chat(
            _SYSTEM_PROMPT,
            prompt,
            what="selecting B-roll moments with Gemini",
            max_tokens=1500,
            temperature=0.2,
        )
    if provider == "anthropic":
        client = _anthropic_client()

        def call() -> Any:
            return client.messages.create(
                model=settings.anthropic_model,
                max_tokens=1500,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )

        response = with_retries(
            call,
            provider="Anthropic",
            what="selecting B-roll moments",
            max_attempts=settings.llm_max_retries,
        )
        return "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )

    return groq_chat(
        _SYSTEM_PROMPT,
        prompt,
        what="selecting B-roll moments",
        max_tokens=1500,
        temperature=0.2,
    )


def _parse_moments(raw: str) -> list[dict[str, Any]]:
    text = strip_reasoning(raw)
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return []
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    moments = payload.get("moments")
    return [m for m in moments if isinstance(m, dict)] if isinstance(moments, list) else []


def _heuristic_moments(transcript: Transcript) -> list[dict[str, Any]]:
    """Deterministic fallback when the LLM is unreachable.

    Scores sentences by information density (length + glossary hits + digits)
    and spreads the picks across the timeline so cutaways never bunch up.
    """
    glossary = {term.lower() for term in settings.glossary()}
    scored: list[tuple[float, Any]] = []
    for sentence in transcript.sentences:
        tokens = [t.lower().strip(".,!?…") for t in sentence.text.split()]
        if len(tokens) < 4:
            continue
        hits = sum(1 for token in tokens if token in glossary)
        digits = sum(1 for token in tokens if any(ch.isdigit() for ch in token))
        score = len(tokens) * 0.4 + hits * 4.0 + digits * 3.0
        scored.append((score, sentence))

    scored.sort(key=lambda item: item[0], reverse=True)
    wanted = settings.broll_max_suggestions
    picked: list[Any] = []
    min_gap = 3.0
    for _, sentence in scored:
        if all(abs(sentence.start - other.start) >= min_gap for other in picked):
            picked.append(sentence)
        if len(picked) >= wanted:
            break
    picked.sort(key=lambda s: s.start)

    moments: list[dict[str, Any]] = []
    for sentence in picked:
        keywords = [
            token.strip(".,!?…")
            for token in sentence.text.split()
            if token.lower().strip(".,!?…") in glossary
        ][:3]
        query = " ".join(keywords) if keywords else "person working laptop office"
        moments.append(
            {
                "sentence_id": sentence.id,
                "query": query,
                "keywords": keywords,
                "reason": "High information density (auto-selected)",
                "duration": settings.broll_default_duration,
            }
        )
    return moments


# ---------------------------------------------------------------- Pexels search
def _score_video_file(file: dict[str, Any]) -> tuple[int, float]:
    width = int(file.get("width") or 0)
    height = int(file.get("height") or 0)
    if width <= 0 or height <= 0:
        return (-1, 0.0)
    vertical = height > width
    aspect_error = abs((width / height) - _TARGET_ASPECT)
    # Prefer: vertical, >= 1080 tall, aspect close to 9:16, then bigger.
    tier = 0
    if vertical:
        tier += 4
    if height >= 1280:
        tier += 2
    elif height >= 720:
        tier += 1
    if aspect_error < 0.05:
        tier += 2
    return (tier, -aspect_error + min(height, 2160) / 10000.0)


def _best_rendition(video: dict[str, Any]) -> dict[str, Any] | None:
    files = [
        file
        for file in (video.get("video_files") or [])
        if file.get("link") and (file.get("file_type") or "video/mp4") == "video/mp4"
    ]
    if not files:
        return None
    return max(files, key=_score_video_file)


def _clip_id(provider_id: str, link: str) -> str:
    digest = hashlib.sha1(f"{provider_id}:{link}".encode()).hexdigest()[:16]  # noqa: S324
    return f"px_{digest}"


def search_pexels(query: str, *, per_page: int | None = None) -> list[BrollClip]:
    """Search Pexels for portrait video clips matching ``query``."""
    if not settings.pexels_api_key:
        raise ConfigurationError(
            "PEXELS_API_KEY is not set.",
            detail="Add PEXELS_API_KEY=... to your .env to enable B-roll search.",
        )
    per_page = per_page or settings.broll_search_results
    headers = {"Authorization": settings.pexels_api_key}

    def fetch(params: dict[str, Any]) -> dict[str, Any]:
        def call() -> dict[str, Any]:
            with httpx.Client(timeout=30.0) as client:
                response = client.get(PEXELS_SEARCH_URL, params=params, headers=headers)
            if response.status_code == 429:
                raise RateLimitedError(
                    "Pexels rate limit reached.",
                    provider="Pexels",
                    retry_after=_header_float(response, "retry-after"),
                    detail="Pexels allows 200 requests/hour on the free tier.",
                )
            if response.status_code == 401:
                raise UpstreamAPIError(
                    "Pexels rejected the API key.",
                    provider="Pexels",
                    status=401,
                    detail="Check PEXELS_API_KEY in your .env.",
                )
            response.raise_for_status()
            return response.json()

        return with_retries(
            call,
            provider="Pexels",
            what=f"searching '{query}'",
            max_attempts=3,
        )

    payload = fetch(
        {
            "query": query,
            "orientation": "portrait",
            "per_page": per_page,
            "size": "medium",
        }
    )
    videos = payload.get("videos") or []
    if not videos:
        # Portrait inventory can be thin; retry unconstrained and crop later.
        payload = fetch({"query": query, "per_page": per_page})
        videos = payload.get("videos") or []

    clips: list[BrollClip] = []
    for video in videos:
        rendition = _best_rendition(video)
        if not rendition:
            continue
        user = video.get("user") or {}
        clips.append(
            BrollClip(
                id=_clip_id(str(video.get("id")), str(rendition["link"])),
                provider="pexels",
                provider_id=str(video.get("id")),
                width=int(rendition.get("width") or 0),
                height=int(rendition.get("height") or 0),
                duration=float(video.get("duration") or 0.0),
                download_url=str(rendition["link"]),
                preview_image=video.get("image"),
                author=user.get("name"),
                author_url=user.get("url"),
                page_url=video.get("url"),
                query=query,
            )
        )
    clips.sort(
        key=lambda clip: (
            not clip.is_vertical,
            abs(clip.width / max(clip.height, 1) - _TARGET_ASPECT),
        )
    )
    return clips


def _header_float(response: httpx.Response, key: str) -> float | None:
    raw = response.headers.get(key)
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


# ------------------------------------------------------------------ suggestions
def build_suggestions(transcript: Transcript) -> tuple[list[BrollSuggestion], list[str]]:
    """Detect B-roll moments and pre-fill each with Pexels candidates."""
    warnings: list[str] = []
    if not transcript.sentences:
        return [], warnings

    listing = "\n".join(f"{sentence.id}: {sentence.text}" for sentence in transcript.sentences)
    prompt = (
        f"Reel duration: {transcript.sentences[-1].end:.1f}s. "
        f"Pick between {settings.broll_min_suggestions} and "
        f"{settings.broll_max_suggestions} moments.\n\nSentences:\n{listing}"
    )

    try:
        moments = _parse_moments(_call_llm(prompt))
    except (UpstreamAPIError, ConfigurationError) as exc:
        log.warning("B-roll LLM selection failed: %s", exc)
        warnings.append(f"B-roll moment detection fell back to heuristics ({exc}).")
        moments = []
    if not moments:
        if not warnings:
            warnings.append("The LLM returned no usable B-roll moments; used heuristics.")
        moments = _heuristic_moments(transcript)

    by_sentence = {sentence.id: sentence for sentence in transcript.sentences}
    suggestions: list[BrollSuggestion] = []
    seen_sentences: set[int] = set()

    for moment in moments[: settings.broll_max_suggestions]:
        try:
            sentence_id = int(moment.get("sentence_id"))
        except (TypeError, ValueError):
            continue
        sentence = by_sentence.get(sentence_id)
        if sentence is None or sentence_id in seen_sentences:
            continue
        seen_sentences.add(sentence_id)

        query = str(moment.get("query") or "").strip()[:120]
        if not query:
            continue
        keywords = [str(k)[:40] for k in (moment.get("keywords") or []) if str(k).strip()][:5]

        try:
            duration = float(moment.get("duration") or settings.broll_default_duration)
        except (TypeError, ValueError):
            duration = settings.broll_default_duration
        duration = max(1.5, min(settings.broll_max_duration, duration))
        span = max(0.0, sentence.end - sentence.start)
        duration = min(duration, span) if span >= 1.5 else duration

        suggestion = BrollSuggestion(
            id=f"br_{uuid.uuid4().hex[:10]}",
            sentence_id=sentence_id,
            anchor_word_id=sentence.word_ids[0] if sentence.word_ids else 0,
            source_start=round(sentence.start, 3),
            source_end=round(sentence.start + duration, 3),
            reason=str(moment.get("reason") or "")[:160],
            query=query,
            keywords=keywords,
        )
        try:
            suggestion.candidates = search_pexels(query)
            if not suggestion.candidates:
                suggestion.search_error = f"No Pexels results for '{query}'."
            else:
                suggestion.selected_clip_id = suggestion.candidates[0].id
        except (UpstreamAPIError, ConfigurationError) as exc:
            suggestion.search_error = str(exc)
            warnings.append(f"Pexels search for '{query}' failed: {exc}")
        suggestions.append(suggestion)

    suggestions.sort(key=lambda item: item.source_start)
    return suggestions, warnings


def refresh_candidates(suggestion: BrollSuggestion, query: str) -> BrollSuggestion:
    """Re-run the stock search for a suggestion with a user-supplied query."""
    clips = search_pexels(query)
    suggestion.query = query
    suggestion.candidates = clips
    suggestion.search_error = None if clips else f"No Pexels results for '{query}'."
    suggestion.selected_clip_id = clips[0].id if clips else None
    suggestion.status = BrollStatus.pending
    return suggestion


# -------------------------------------------------------------------- download
def download_clip(clip: BrollClip, cache_dir: Path) -> Path:
    return with_retries(
        lambda: _download_clip_once(clip, cache_dir),
        provider="Pexels",
        what="downloading B-roll",
        max_attempts=2,
        budget_s=45,
    )


def _download_clip_once(clip: BrollClip, cache_dir: Path) -> Path:
    """Download (and cache) a stock clip, enforcing a hard size limit."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / f"{clip.id}.mp4"
    if destination.is_file() and destination.stat().st_size > 0:
        return destination

    limit = settings.broll_download_max_mb * 1024 * 1024
    partial = cache_dir / f"{clip.id}.part-{uuid.uuid4().hex[:8]}.mp4"
    written = 0
    try:
        with (
            httpx.Client(timeout=httpx.Timeout(30, connect=10), follow_redirects=True) as client,
            client.stream("GET", clip.download_url) as response,
            partial.open("wb") as handle,
        ):
            if response.status_code == 429:
                raise RateLimitedError(
                    "Pexels rate limit reached while downloading B-roll.",
                    provider="Pexels",
                )
            response.raise_for_status()
            for block in response.iter_bytes(chunk_size=1 << 16):
                check_cancelled()
                written += len(block)
                if written > limit:
                    raise UpstreamAPIError(
                        "B-roll clip exceeds the configured size limit.",
                        provider="Pexels",
                        detail=f"Limit is {settings.broll_download_max_mb} MB.",
                    )
                handle.write(block)
    except httpx.HTTPError as exc:
        partial.unlink(missing_ok=True)
        raise UpstreamAPIError(
            "Failed to download the B-roll clip.",
            provider="Pexels",
            status=exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else 503,
        ) from exc
    except Exception:
        partial.unlink(missing_ok=True)
        raise

    if written == 0:
        partial.unlink(missing_ok=True)
        raise UpstreamAPIError(
            "Pexels returned an empty B-roll clip.",
            provider="Pexels",
            detail=clip.download_url[:200],
        )
    partial.replace(destination)
    return destination
