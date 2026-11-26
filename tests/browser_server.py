"""Isolated browser fixture server. Never reads/writes the user's projects."""

import os
import subprocess
import tempfile
from pathlib import Path

directory = tempfile.TemporaryDirectory(prefix="video-editor-browser-")
os.environ["DATA_DIR"] = directory.name
os.environ["NORMALIZER_PROVIDER"] = "off"

from app.config import settings  # noqa: E402
from app.schemas import MediaInfo, ProjectStatus, Sentence, Transcript, Word  # noqa: E402
from app.store import store  # noqa: E402

source = Path(directory.name) / "fixture.mp4"
subprocess.run(
    [
        settings.resolve_binary(settings.ffmpeg_binary),
        "-v",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=320x240:rate=30:duration=2",
        "-f",
        "lavfi",
        "-i",
        "sine=duration=2",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-c:a",
        "aac",
        str(source),
    ],
    check=True,
)
for name in ["Browser A.mp4", "Browser B.mp4"]:
    project = store.create(name)
    os.link(source, store.source_path(project))
    project.status = ProjectStatus.ready
    project.media = MediaInfo(duration=2, width=320, height=240, fps=30, has_audio=True)
    words = [
        Word(id=i, text=text, start=i * 0.5, end=(i + 1) * 0.5, sentence_id=0)
        for i, text in enumerate(["Bir", "İki", "Üç", "Dörd"])
    ]
    project.transcript = Transcript(
        language="az",
        text="Bir İki Üç Dörd",
        words=words,
        sentences=[Sentence(id=0, word_ids=[0, 1, 2, 3], text="Bir İki Üç Dörd", start=0, end=2)],
    )
    store.save(project)

if __name__ == "__main__":
    import uvicorn

    try:
        uvicorn.run("app.main:app", host="127.0.0.1", port=8877, log_level="warning")
    finally:
        directory.cleanup()
