import logging
from types import SimpleNamespace

import pytest

from withoutbg_comfyui import backends


class FakeModel:
    def __init__(self, fail_on=None):
        self.calls = []
        self.fail_on = fail_on
        self.model = SimpleNamespace(_download_from_hf=self._download)

    def _download(self, name, label):
        self.calls.append(name)
        if name == self.fail_on:
            raise RuntimeError("boom")


@pytest.fixture
def fake(monkeypatch):
    model = FakeModel()
    monkeypatch.setattr(backends, "_get_open_weights", lambda device="auto": model)
    return model


def test_nothing_missing_is_silent_and_free(monkeypatch, fake, caplog):
    monkeypatch.setattr(backends, "_missing_bundle_files", lambda: [])
    with caplog.at_level(logging.INFO, logger="withoutbg_comfyui"):
        assert backends.ensure_open_weights() == 0
    assert fake.calls == [] and caplog.records == []


def test_downloads_missing_files_logs_and_reports_progress(monkeypatch, fake, caplog):
    missing = ["a.onnx", "b.onnx"]
    monkeypatch.setattr(backends, "_missing_bundle_files", lambda: missing)
    seen = []
    with caplog.at_level(logging.INFO, logger="withoutbg_comfyui"):
        n = backends.ensure_open_weights(lambda d, t: seen.append((d, t)))
    assert n == 2 and fake.calls == missing
    assert seen == [(0, 2), (1, 2), (2, 2)]
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "1.5 GB" in text and "a.onnx (1/2)" in text and "model ready" in text


def test_download_failure_has_actionable_message(monkeypatch):
    model = FakeModel(fail_on="b.onnx")
    monkeypatch.setattr(backends, "_get_open_weights", lambda device="auto": model)
    monkeypatch.setattr(backends, "_missing_bundle_files", lambda: ["a.onnx", "b.onnx"])
    with pytest.raises(
        backends.BackendError, match=r"(?s)b\.onnx.*WITHOUTBG_MODEL_PATH"
    ):
        backends.ensure_open_weights()


def test_model_path_override_skips_download(monkeypatch):
    monkeypatch.setenv("WITHOUTBG_MODEL_PATH", "/tmp/x.onnx")
    assert backends._missing_bundle_files() == []


def test_missing_files_reads_hf_cache(monkeypatch):
    monkeypatch.delenv("WITHOUTBG_MODEL_PATH", raising=False)
    import huggingface_hub

    cached = {backends._BUNDLE_FILES[0], backends._BUNDLE_FILES[1]}

    def fake_lookup(repo, name, revision=None):
        return "/cache/" + name if name in cached else None

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", fake_lookup)
    assert backends._missing_bundle_files() == list(backends._BUNDLE_FILES[2:])
