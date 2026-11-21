import numpy as np
import pytest
from PIL import Image

from withoutbg_comfyui.convert import alpha_to_array, compose_rgba, frame_to_pil

torch = pytest.importorskip("torch")


def test_frame_to_pil_rgb_roundtrip():
    frame = torch.rand(8, 6, 3)
    pil = frame_to_pil(frame)
    assert pil.mode == "RGB" and pil.size == (6, 8)
    back = np.asarray(pil, dtype=np.float32) / 255.0
    assert np.abs(back - frame.numpy()).max() <= 1 / 255 + 1e-6


def test_frame_to_pil_drops_alpha_and_expands_gray():
    assert frame_to_pil(torch.rand(4, 4, 4)).mode == "RGB"
    assert frame_to_pil(torch.rand(4, 4, 1)).size == (4, 4)


def test_alpha_to_array_from_rgba_and_resize():
    rgba = Image.new("RGBA", (4, 4), (10, 20, 30, 255))
    a = alpha_to_array(rgba, (4, 4))
    assert a.shape == (4, 4) and a.dtype == np.float32 and a.min() == 1.0
    small = Image.new("L", (2, 2), 0)
    assert alpha_to_array(small, (4, 4)).shape == (4, 4)


def test_compose_rgba_keeps_input_colours():
    frame = torch.rand(5, 7, 3)
    out = compose_rgba(frame, np.full((5, 7), 0.25, np.float32))
    assert out.shape == (5, 7, 4)
    np.testing.assert_array_equal(out[..., :3], frame.numpy())
    assert np.all(out[..., 3] == 0.25)
