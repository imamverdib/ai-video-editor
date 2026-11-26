"""Regression tests for the September 2026 audit findings."""

import json
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from app.agy_provider import AGYSession
from app.config import settings
from app.errors import EditorError, FFmpegError, UpstreamAPIError
from app.ffmpeg_utils import probe_media, run_ffmpeg
from app.jobs import JobManager
from app.main import app
from app.render_profiles import RenderProfile
from app.renderer import _build_audio_chain, render_timeline
from app.schemas import (
    Job,
    JobStatus,
    MediaInfo,
    ProjectStatus,
    RenderOptions,
    Sentence,
    Transcript,
    Word,
)
from app.store import ProjectStore
from app.subtitles import _ass_time
from app.timeline_manager import compute_timeline


@pytest.fixture
def project_api(tmp_path, monkeypatch):
    db = ProjectStore(tmp_path / "projects")
    project = db.create("Azərbaycan videosu.mp4")
    words = [
        Word(id=i, text=t, start=i * 0.5, end=(i + 1) * 0.5, sentence_id=0)
        for i, t in enumerate(["bir", "iki", "üç", "dörd"])
    ]
    project.status = ProjectStatus.ready
    project.media = MediaInfo(duration=2, width=320, height=240, fps=30, has_audio=True)
    project.transcript = Transcript(
        language="az",
        text="bir iki üç dörd",
        words=words,
        sentences=[Sentence(id=0, word_ids=[0, 1, 2, 3], text="bir iki üç dörd", start=0, end=2)],
    )
    db.save(project)
    monkeypatch.setattr("app.main.store", db)
    return TestClient(app), db, project


def test_reversed_merge_is_undoable_without_creating_cuts(project_api):
    client, db, p = project_api
    prefix = f"/api/projects/{p.id}"
    result = client.post(prefix + "/words/merge", json={"word_ids": [1, 0]}).json()
    assert [w["id"] for w in result["project"]["transcript"]["words"]] == [0, 2, 3]
    assert len(result["timeline"]["clips"]) == 1
    assert result["project"]["can_undo"]
    assert "undo_history" not in result["project"]
    undo = client.post(prefix + "/history/undo").json()
    assert len(undo["project"]["transcript"]["words"]) == 4
    assert undo["project"]["can_redo"]
    assert client.post(prefix + "/history/redo").status_code == 200
    assert len(ProjectStore(db.root).get(p.id).transcript.words) == 3


@pytest.mark.parametrize("ids", [[0, 2], [1, 1], [0, 999]])
def test_invalid_merge_is_atomic(project_api, ids):
    client, db, p = project_api
    before = db.get(p.id).model_dump()
    assert (
        client.post(f"/api/projects/{p.id}/words/merge", json={"word_ids": ids}).status_code == 400
    )
    assert db.get(p.id).model_dump() == before


def test_ordered_json_autosave_and_conflict(project_api):
    client, db, p = project_api
    url = f"/api/projects/{p.id}/edit"
    body = {"removed_word_ids": [1], "client_id": "tab", "sequence": 2, "expected_revision": 0}
    assert client.post(url, json=body).status_code == 200
    assert db.get(p.id).revision == 1
    body.update(sequence=1, removed_word_ids=[2])
    assert client.post(url, json=body).status_code == 200
    assert db.get(p.id).edit.removed_word_ids == [1]
    body.update(client_id="other", sequence=3)
    assert client.post(url, json=body).status_code == 409
    assert db.get(p.id).edit.removed_word_ids == [1]


def test_unicode_download_header(project_api):
    client, db, p = project_api
    (db.project_dir(p.id) / p.source_file).write_bytes(b"test")
    response = client.get(f"/api/projects/{p.id}/source")
    assert response.status_code == 200
    # Download helper is also used by the render endpoint.
    from starlette.requests import Request

    from app.main import _ranged_response

    response = _ranged_response(
        db.source_path(p), Request({"type": "http", "headers": []}), filename="Azərbaycan.mp4"
    )
    assert "filename*=UTF-8''Az%C9%99rbaycan.mp4" in response.headers["content-disposition"]


def test_delete_cannot_be_resurrected(project_api):
    _, db, p = project_api
    db.delete(p.id)
    for operation in [lambda: db.save(p), lambda: db.work_dir(p.id), lambda: db.output_dir(p.id)]:
        with pytest.raises(EditorError):
            operation()
    assert not db.project_dir(p.id).exists()


def test_restart_reconciles_incomplete_jobs(project_api):
    _, db, p = project_api
    job = Job(id="job_stale", project_id=p.id, kind="ingest", status=JobStatus.running)
    db.save_job(job)
    manager = JobManager(db)
    try:
        manager.reconcile()
        assert db.get_job(p.id, job.id).status == JobStatus.failed
        assert db.get(p.id).status == ProjectStatus.failed
    finally:
        manager.shutdown()


def test_ai_failure_is_visible_and_old_suggestions_clear(monkeypatch):
    import app.transcription as tx

    monkeypatch.setattr(settings, "normalizer_provider", "agy")

    def fail(*args, **kwargs):
        raise UpstreamAPIError("offline", provider="AGY")

    monkeypatch.setattr(tx, "_call_proofreader", fail)
    warnings = []
    words = [Word(id=0, text="salam", start=0, end=1, sentence_id=0, suggestion="stale")]
    result, count = tx.generate_spelling_suggestions(words, warnings=warnings)
    assert count == 0 and result[0].suggestion is None and warnings


def test_httpx_errors_classified_for_retry():
    from app.transcription import is_transient

    assert is_transient(httpx.ReadTimeout("timeout"))
    response = httpx.Response(429, request=httpx.Request("POST", "https://example.com"))
    assert is_transient(httpx.HTTPStatusError("quota", request=response.request, response=response))


@pytest.mark.parametrize(
    "response",
    [
        None,
        "",
        "{}",
        '{"fixes":[{}]}',
        '{"fixes":[{"i":true,"t":"wrong"}]}',
        '{"fixes":[{"i":0,"t":""}]}',
    ],
)
def test_agy_rejects_empty_or_invalid_success(response, monkeypatch):
    session = AGYSession()

    class Process:
        stdin = type("Input", (), {"write": lambda *_a: None, "flush": lambda *_a: None})()

        def poll(self):
            return None

    session.process = Process()
    monkeypatch.setattr(session, "close", lambda: None)
    session.events.put({"event": "result", "result": {"status": "SUCCESS", "response": response}})
    with pytest.raises(UpstreamAPIError):
        session.chat("test", "test", "fixes")


def test_agy_native_session_reuses_process_and_cleans_up(tmp_path, monkeypatch):
    fake = tmp_path / "agy"
    fake.write_text(
        f"#!{Path(sys.executable).resolve()}\n"
        + """import json,sys
print(json.dumps({"event":"init"}), flush=True)
for line in sys.stdin:
    value=json.loads(line)
    assert value["event"] == "user"
    print(json.dumps({"event":"step_update","step_update":{"step_type":"user_input"}}),flush=True)
    print(json.dumps({"event":"result","result":{"status":"SUCCESS","response":'{"fixes":[]}'}}),flush=True)
"""
    )
    fake.chmod(0o700)
    monkeypatch.setattr(settings, "agy_binary", str(fake))
    session = AGYSession()
    assert json.loads(session.chat("test", "one", "fixes")) == {"fixes": []}
    process = session.process
    directory = Path(session.directory.name)
    session.chat("test", "two", "fixes")
    assert session.process is process
    session.close()
    assert process.poll() is not None and not directory.exists()


def test_ffmpeg_timeout_works_with_silent_child(tmp_path, monkeypatch):
    fake = tmp_path / "ffmpeg"
    fake.write_text(f"#!{Path(sys.executable).resolve()}\nimport time\ntime.sleep(20)\n")
    fake.chmod(0o700)
    monkeypatch.setattr("app.ffmpeg_utils.ffmpeg_path", lambda: str(fake))
    start = time.monotonic()
    with pytest.raises(FFmpegError):
        run_ffmpeg([], timeout=0.3)
    assert time.monotonic() - start < 4


def test_hard_deletions_disable_audio_overlap(project_api):
    _, _, p = project_api
    timeline = compute_timeline(
        words=p.transcript.words,
        sentences=p.transcript.sentences,
        removed_word_ids=[1],
        source_duration=2,
        source_fps=30,
    )
    parts, _, mode = _build_audio_chain(timeline, 2)
    assert mode == "fade" and "acrossfade" not in ";".join(parts)
    assert all(c.source_end <= 0.5 or c.source_start >= 1 for c in timeline.clips)


def test_source_master_and_hdr_policy():
    media = MediaInfo(
        duration=2, width=3840, height=2160, fps=59.94, has_audio=True, color_transfer="smpte2084"
    )
    profile = RenderProfile.resolve(media, RenderOptions(profile="master"))
    assert (profile.width, profile.height, profile.fps, profile.crf) == (3840, 2160, 59.94, 16)
    assert profile.tonemap
    with pytest.raises(EditorError):
        RenderProfile.resolve(media, RenderOptions(hdr_policy="reject"))
    assert _ass_time(59.999) == "0:01:00.00"


@pytest.fixture(scope="module")
def source_media(tmp_path_factory):
    root = tmp_path_factory.mktemp("render-matrix")
    source = root / "source.mp4"
    subprocess.run(
        [
            settings.resolve_binary(settings.ffmpeg_binary),
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x240:rate=60:duration=2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=2",
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
    return root, source


@pytest.mark.parametrize(
    "engine,framing,fps",
    [(None, "crop", "source"), ("ass", "fit", "30"), ("pillow", "source", "source")],
)
def test_real_render_geometry_subtitles_and_sync(source_media, monkeypatch, engine, framing, fps):
    root, source = source_media
    monkeypatch.setattr(settings, "output_width", 180)
    monkeypatch.setattr(settings, "output_height", 320)
    words = [
        Word(id=i, text=t, start=i * 0.5, end=(i + 1) * 0.5, sentence_id=0)
        for i, t in enumerate(["İşıq", "silinən", "gözəl", "şəhər"])
    ]
    sentence = Sentence(id=0, word_ids=[0, 1, 2, 3], text="test", start=0, end=2)
    timeline = compute_timeline(
        words=words, sentences=[sentence], removed_word_ids=[1], source_duration=2, source_fps=60
    )
    output = root / f"{engine}-{framing}.mp4"
    result = render_timeline(
        source_path=source,
        timeline=timeline,
        work_dir=root / f"work-{engine}",
        output_path=output,
        options=RenderOptions(
            burn_subtitles=bool(engine),
            subtitle_engine=engine,
            framing=framing,
            fps=fps,
            include_broll=False,
        ),
    )
    media = probe_media(output)
    expected = (320, 240) if framing == "source" else (180, 320)
    assert (media.width, media.height) == expected
    assert media.fps == (60 if fps == "source" else 30)
    assert abs(media.duration - timeline.output_duration) < 0.07
    assert abs(media.audio_duration - media.duration) < 0.07
    assert result.audio_join_mode == "fade"


@pytest.mark.parametrize("variant", ["hdr", "vfr", "audio-offset"])
def test_real_nonstandard_sources(tmp_path, monkeypatch, variant):
    monkeypatch.setattr(settings, "output_width", 180)
    monkeypatch.setattr(settings, "output_height", 320)
    source = tmp_path / ("source.mp4" if variant == "hdr" else "source.mkv")
    command = [
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
        "sine=duration=2:sample_rate=48000",
    ]
    if variant == "hdr":
        command += [
            "-pix_fmt",
            "yuv420p10le",
            "-color_primaries",
            "bt2020",
            "-color_trc",
            "smpte2084",
            "-colorspace",
            "bt2020nc",
        ]
    elif variant == "vfr":
        command += ["-vf", "select='if(lt(t,1),1,not(mod(n,2)))'", "-fps_mode", "vfr"]
    else:
        command += ["-af", "asetpts=PTS+0.2/TB"]
    codec = (
        [
            "-c:v",
            "libx265",
            "-preset",
            "ultrafast",
            "-x265-params",
            "colorprim=9:transfer=16:colormatrix=9:log-level=error",
            "-c:a",
            "aac",
        ]
        if variant == "hdr"
        else ["-c:v", "ffv1", "-c:a", "pcm_s16le"]
    )
    subprocess.run(command + codec + [str(source)], check=True)
    media = probe_media(source)
    words = [Word(id=0, text="İşıq", start=0, end=2, sentence_id=0)]
    timeline = compute_timeline(
        words=words,
        sentences=[Sentence(id=0, word_ids=[0], text="İşıq", start=0, end=2)],
        removed_word_ids=[],
        source_duration=media.duration,
        source_fps=media.fps,
    )
    output = tmp_path / "result.mp4"
    render_timeline(
        source_path=source,
        timeline=timeline,
        work_dir=tmp_path / "work",
        output_path=output,
        options=RenderOptions(burn_subtitles=False),
    )
    result = probe_media(output)
    assert abs(result.duration - result.audio_duration) < 0.07
    if variant == "hdr":
        assert result.color_transfer == "bt709"
        assert result.pixel_format == "yuv420p"
    if variant == "audio-offset":
        pcm = subprocess.check_output(
            [
                settings.resolve_binary(settings.ffmpeg_binary),
                "-v",
                "error",
                "-i",
                str(output),
                "-t",
                "0.1",
                "-f",
                "s16le",
                "-ac",
                "1",
                "-",
            ]
        )
        import numpy as np

        assert np.abs(np.frombuffer(pcm, dtype="<i2")).max() < 100


def test_deleted_audio_tone_is_absent(tmp_path, monkeypatch):
    import numpy as np

    monkeypatch.setattr(settings, "output_width", 180)
    monkeypatch.setattr(settings, "output_height", 320)
    source = tmp_path / "tone.mkv"
    executable = settings.resolve_binary(settings.ffmpeg_binary)
    subprocess.run(
        [
            executable,
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=size=180x320:rate=30:duration=2",
            "-f",
            "lavfi",
            "-i",
            "aevalsrc=0.1*sin(2*PI*440*t)+if(between(t\\,0.5\\,1)\\,0.3*sin(2*PI*880*t)\\,0):s=48000:d=2",
            "-c:v",
            "ffv1",
            "-c:a",
            "pcm_s16le",
            str(source),
        ],
        check=True,
    )
    words = [
        Word(id=i, text="test", start=i * 0.5, end=(i + 1) * 0.5, sentence_id=0) for i in range(4)
    ]
    timeline = compute_timeline(
        words=words,
        sentences=[Sentence(id=0, word_ids=[0, 1, 2, 3], text="test", start=0, end=2)],
        removed_word_ids=[1],
        source_duration=2,
        source_fps=30,
    )
    output = tmp_path / "result.mp4"
    render_timeline(
        source_path=source,
        timeline=timeline,
        work_dir=tmp_path / "work",
        output_path=output,
        options=RenderOptions(burn_subtitles=False),
    )
    pcm = subprocess.check_output(
        [
            executable,
            "-v",
            "error",
            "-i",
            str(output),
            "-f",
            "f32le",
            "-ac",
            "1",
            "-ar",
            "48000",
            "-",
        ]
    )
    samples = np.frombuffer(pcm, dtype="<f4")
    power = np.abs(np.fft.rfft(samples * np.hanning(len(samples))))
    frequencies = np.fft.rfftfreq(len(samples), 1 / 48000)

    def peak(hz):
        return max(power[np.abs(frequencies - hz) < 4])

    assert peak(880) / peak(440) < 0.01
