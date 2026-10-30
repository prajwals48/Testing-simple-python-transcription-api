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
