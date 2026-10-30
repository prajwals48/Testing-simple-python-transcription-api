from fastapi.testclient import TestClient
from pytest import MonkeyPatch

import main


def test_transcribe_returns_text(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    class _Result:
        text = "hello"

    class _Transcriptions:
        async def create(self, *, model: str, file: object) -> _Result:
            assert model == main.WHISPER_MODEL
            assert getattr(file, "name", None) == "upload.wav"
            return _Result()

    class _Audio:
        transcriptions = _Transcriptions()

    class _Client:
        audio = _Audio()

    monkeypatch.setattr(main, "AsyncOpenAI", lambda: _Client())

    response = TestClient(main.app).post(
        "/transcribe",
        files={"file": ("clip.wav", b"not-a-real-wav", "audio/wav")},
    )

    assert response.status_code == 200
    assert response.json() == {"text": "hello"}


def test_transcribe_rejects_oversized_file_with_actual_limit(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(main, "MAX_AUDIO_BYTES", 1024 * 1024)

    response = TestClient(main.app).post(
        "/transcribe",
        files={"file": ("clip.wav", b"x" * (1024 * 1024 + 1), "audio/wav")},
    )

    assert response.status_code == 413
    assert response.json() == {"detail": "File is larger than 1 MB"}


def test_transcribe_accepts_file_at_limit(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(main, "MAX_AUDIO_BYTES", 1024)

    class _Result:
        text = "ok"

    class _Transcriptions:
        async def create(self, *, model: str, file: object) -> _Result:
            return _Result()

    class _Audio:
        transcriptions = _Transcriptions()

    class _Client:
        audio = _Audio()

    monkeypatch.setattr(main, "AsyncOpenAI", lambda: _Client())

    response = TestClient(main.app).post(
        "/transcribe",
        files={"file": ("clip.wav", b"x" * 1024, "audio/wav")},
    )

    assert response.status_code == 200
