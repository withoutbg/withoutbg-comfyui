import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from withoutbg_comfyui import backends
from withoutbg_comfyui.nodes import WithoutBGRemoveBackground


def _fake_result(pil):
    out = pil.convert("RGBA")
    out.putalpha(Image.new("L", pil.size, 128))
    return out


@pytest.mark.parametrize("batch", [1, 3])
def test_open_weights_shapes_and_ranges(monkeypatch, batch):
    monkeypatch.setattr(backends, "remove_open_weights", _fake_result)
    monkeypatch.setattr(backends, "ensure_open_weights", lambda progress=None: 0)
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
    assert ua.startswith("withoutbg-comfyui/0.1.0 (ComfyUI 0.3.0; ")


def test_api_message_maps_codes():
    assert "credits" in backends._api_message(Exception("HTTP 402"))


def test_no_api_key_widget():
    inputs = WithoutBGRemoveBackground.INPUT_TYPES()
    assert "optional" not in inputs
    assert "api_key" not in inputs["required"]


def test_open_weights_downloads_before_first_frame(monkeypatch):
    order = []
    monkeypatch.setattr(
        backends, "ensure_open_weights", lambda progress=None: order.append("dl")
    )
    monkeypatch.setattr(
        backends, "remove_open_weights", lambda pil: order.append("run") or _fake_result(pil)
    )
    WithoutBGRemoveBackground().run(torch.rand(2, 4, 4, 3), "open_weights")
    assert order == ["dl", "run", "run"]


def test_pro_does_not_download_model(monkeypatch):
    monkeypatch.setattr(
        backends,
        "ensure_open_weights",
        lambda progress=None: pytest.fail("pro must not fetch open weights"),
    )
    monkeypatch.setattr(backends, "make_pro_client", lambda: "client")
    monkeypatch.setattr(backends, "remove_pro", lambda c, pil: _fake_result(pil))
    WithoutBGRemoveBackground().run(torch.rand(1, 4, 4, 3), "pro")
