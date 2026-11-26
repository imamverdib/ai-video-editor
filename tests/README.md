# Tests

`test_core.py` is **network-free** and safe to run anywhere:

```bash
pip install pytest
pytest tests -q
```

It covers timeline arithmetic, silence trimming, B-roll re-timing, subtitle line
grouping / highlight states, `.ass` generation and escaping, filtergraph
escaping, and the `ffprobe` error paths. Video fixtures are synthesised with
FFmpeg inside a temp directory, so FFmpeg must be on `PATH`.

Integration coverage that needs live credentials (`GROQ_API_KEY`,
`PEXELS_API_KEY`) is intentionally *not* in this suite, because it spends money
and depends on third-party availability. To exercise it manually:

1. `./run.sh`
2. Upload a short vertical clip in the UI and let ingest finish.
3. Strike out some words, accept one B-roll suggestion, render, download.

The equivalent checks the pipeline was validated with:

* transcription of a 13 s clip -> 31 words with word-level timestamps,
* terminology normalisation rewriting only misspelled terms and leaving
  `start`/`end` untouched,
* LLM B-roll moment selection + Pexels portrait search returning 8 candidates,
* renders in both audio join modes reproducing the computed duration exactly,
* a 10-clip filtergraph, a landscape source centre-cropped to 1080x1920,
  and subtitles-off / B-roll-off variants.
