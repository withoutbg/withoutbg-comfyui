import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from withoutbg_comfyui import backends
from withoutbg_comfyui.nodes import WithoutBGRemoveBackground


def _fake_result(pil, device="auto"):
    out = pil.convert("RGBA")
    out.putalpha(Image.new("L", pil.size, 128))
    return out


@pytest.mark.parametrize("batch", [1, 3])
def test_open_weights_shapes_and_ranges(monkeypatch, batch):
    monkeypatch.setattr(backends, "remove_open_weights", _fake_result)
    monkeypatch.setattr(
        backends, "ensure_open_weights", lambda progress=None, device="auto": 0
    )
    img = torch.rand(batch, 10, 12, 3)
    out, mask = WithoutBGRemoveBackground().run(img, "open_weights")
    assert out.shape == (batch, 10, 12, 4)
    assert mask.shape == (batch, 10, 12)
    assert 0.0 <= float(mask.min()) and float(mask.max()) <= 1.0
    assert abs(float(mask[0, 0, 0]) - 128 / 255) < 1e-3


def test_pro_uses_client(monkeypatch):
    calls = []
    monkeypatch.setattr(backends, "make_pro_client", lambda: "client")
    monkeypatch.setattr(
        backends, "remove_pro", lambda c, pil: calls.append(c) or _fake_result(pil)
    )
    WithoutBGRemoveBackground().run(torch.rand(2, 4, 4, 3), "pro")
    assert calls == ["client"] * 2


def test_pro_without_key_errors(monkeypatch):
    monkeypatch.delenv("WITHOUTBG_API_KEY", raising=False)
    with pytest.raises(backends.BackendError, match="API key"):
        WithoutBGRemoveBackground().run(torch.rand(1, 4, 4, 3), "pro")


def test_unknown_engine():
    with pytest.raises(backends.BackendError):
        WithoutBGRemoveBackground().run(torch.rand(1, 4, 4, 3), "nope")


def test_user_agent_format():
    ua = backends.user_agent("0.3.0")
    assert ua.startswith("withoutbg-comfyui/0.2.0 (ComfyUI 0.3.0; ")


def test_api_message_maps_codes():
    assert "credits" in backends._api_message(Exception("HTTP 402"))


def test_no_api_key_widget():
    inputs = WithoutBGRemoveBackground.INPUT_TYPES()
    assert "optional" not in inputs
    assert "api_key" not in inputs["required"]


def test_old_workflow_widgets_still_load():
    # 0.1.0 workflows store widgets_values ["open_weights"]; new inputs follow engine
    assert list(WithoutBGRemoveBackground.INPUT_TYPES()["required"])[:2] == [
        "image",
        "engine",
    ]


def test_background_white_is_three_channels(monkeypatch):
    monkeypatch.setattr(backends, "remove_open_weights", _fake_result)
    monkeypatch.setattr(backends, "ensure_open_weights", lambda *a, **k: 0)
    img = torch.zeros(1, 4, 4, 3)
    out, mask = WithoutBGRemoveBackground().run(img, "open_weights", "cpu", "white")
    assert out.shape == (1, 4, 4, 3)
    assert mask.shape == (1, 4, 4)
    expected = 128 / 255  # black pixel at alpha .5 over white
    assert abs(float(out[0, 0, 0, 0]) - (1 - expected)) < 1e-3


def test_background_color_and_bad_hex(monkeypatch):
    monkeypatch.setattr(backends, "remove_open_weights", _fake_result)
    monkeypatch.setattr(backends, "ensure_open_weights", lambda *a, **k: 0)
    img = torch.zeros(1, 2, 2, 3)
    out, _ = WithoutBGRemoveBackground().run(
        img, "open_weights", "cpu", "color", "#00ff00"
    )
    assert abs(float(out[0, 0, 0, 1]) - (1 - 128 / 255)) < 1e-3
    with pytest.raises(backends.BackendError, match="hex"):
        WithoutBGRemoveBackground().run(img, "open_weights", "cpu", "color", "nope")


def test_unknown_device():
    with pytest.raises(backends.BackendError):
        WithoutBGRemoveBackground().run(torch.rand(1, 4, 4, 3), "open_weights", "tpu")


def test_open_weights_downloads_before_first_frame(monkeypatch):
    order = []
    monkeypatch.setattr(
        backends,
        "ensure_open_weights",
        lambda progress=None, device="auto": order.append("dl"),
    )
    monkeypatch.setattr(
        backends,
        "remove_open_weights",
        lambda pil, device="auto": order.append("run") or _fake_result(pil),
    )
    WithoutBGRemoveBackground().run(torch.rand(2, 4, 4, 3), "open_weights")
    assert order == ["dl", "run", "run"]


def test_pro_does_not_download_model(monkeypatch):
    monkeypatch.setattr(
        backends,
        "ensure_open_weights",
        lambda progress=None, device="auto": pytest.fail("pro must not fetch"),
    )
    monkeypatch.setattr(backends, "make_pro_client", lambda: "client")
    monkeypatch.setattr(backends, "remove_pro", lambda c, pil: _fake_result(pil))
    WithoutBGRemoveBackground().run(torch.rand(1, 4, 4, 3), "pro")
