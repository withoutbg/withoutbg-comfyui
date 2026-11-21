import sys
from types import SimpleNamespace

import pytest
from PIL import Image

from withoutbg_comfyui import backends

CPU = ["CPUExecutionProvider"]


def _fake_ort(available, preload=None):
    ns = SimpleNamespace(get_available_providers=lambda: available)
    if preload is not None:
        ns.preload_dlls = preload
    return ns


@pytest.fixture(autouse=True)
def reset(monkeypatch):
    monkeypatch.setattr(backends, "_gpu_failed", False)
    monkeypatch.setattr(backends, "_open_weights", {})


def _have_onnx(monkeypatch, present=True):
    monkeypatch.setattr(
        backends.importlib.util, "find_spec", lambda name: object() if present else None
    )


def test_cpu_device_forces_cpu(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "onnxruntime", _fake_ort(["CUDAExecutionProvider"])
    )
    _have_onnx(monkeypatch)
    assert backends.pick_providers("cpu") == CPU


def test_auto_prefers_cuda_and_preloads(monkeypatch):
    calls = []
    ort = _fake_ort(
        ["CUDAExecutionProvider", "CPUExecutionProvider"], lambda: calls.append(1)
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    _have_onnx(monkeypatch)
    assert backends.pick_providers("auto") == ["CUDAExecutionProvider"]
    assert calls == [1]


def test_auto_cpu_when_cuda_but_no_onnx_package(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "onnxruntime", _fake_ort(["CUDAExecutionProvider"])
    )
    _have_onnx(monkeypatch, present=False)
    assert backends.pick_providers("auto") == CPU


def test_auto_cpu_when_no_gpu(monkeypatch):
    monkeypatch.setitem(sys.modules, "onnxruntime", _fake_ort(["CPUExecutionProvider"]))
    _have_onnx(monkeypatch)
    assert backends.pick_providers("auto") == CPU


def test_unknown_device():
    with pytest.raises(backends.BackendError):
        backends.pick_providers("tpu")


def test_gpu_failure_falls_back_to_cpu_once(monkeypatch, caplog):
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        _fake_ort(["CUDAExecutionProvider", "CPUExecutionProvider"]),
    )
    _have_onnx(monkeypatch)

    class Gpu:
        def remove_background(self, image):
            raise RuntimeError("cuda boom")

    class Cpu:
        def remove_background(self, image):
            return image.convert("RGBA")

    def get(device="auto"):
        return Cpu() if backends.pick_providers(device) == CPU else Gpu()

    monkeypatch.setattr(backends, "_get_open_weights", get)
    img = Image.new("RGB", (2, 2))
    with caplog.at_level("WARNING", logger="withoutbg_comfyui"):
        assert backends.remove_open_weights(img).mode == "RGBA"
        assert backends._gpu_failed is True
        assert backends.remove_open_weights(img).mode == "RGBA"
    assert sum("falling back" in r.message for r in caplog.records) == 1


def test_cpu_failure_raises_backend_error(monkeypatch):
    class Bad:
        def remove_background(self, image):
            raise RuntimeError("nope")

    monkeypatch.setattr(backends, "_get_open_weights", lambda device="auto": Bad())
    monkeypatch.setattr(backends, "pick_providers", lambda d: CPU)
    with pytest.raises(backends.BackendError, match="nope"):
        backends.remove_open_weights(Image.new("RGB", (2, 2)), "cpu")
