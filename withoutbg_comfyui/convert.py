"""ComfyUI tensor <-> PIL helpers.

ComfyUI images are float32 tensors shaped ``[B, H, W, C]`` in 0..1, masks are
``[B, H, W]`` in 0..1. Kept free of ComfyUI imports so tests run without it.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def frame_to_pil(frame) -> Image.Image:
    """One ``[H, W, C]`` tensor (C = 1, 3 or 4) to an RGB PIL image.

    Alpha is dropped: the node cuts out the colour data it is given.
    """
    arr = (
        frame.detach().cpu().numpy() if hasattr(frame, "detach") else np.asarray(frame)
    )
    arr = np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return Image.fromarray(arr[..., :3], "RGB")


def alpha_to_array(alpha: Image.Image, size: tuple[int, int]) -> np.ndarray:
    """Alpha matte (PIL ``L`` or RGBA) to float32 ``[H, W]`` in 0..1 at ``size`` (w, h)."""
    if alpha.mode == "RGBA":
        alpha = alpha.getchannel("A")
    elif alpha.mode != "L":
        alpha = alpha.convert("L")
    if alpha.size != size:
        alpha = alpha.resize(size, Image.Resampling.LANCZOS)
    return np.asarray(alpha, dtype=np.float32) / 255.0


def compose_rgba(frame, alpha: np.ndarray) -> np.ndarray:
    """Original RGB of ``frame`` plus ``alpha`` as float32 ``[H, W, 4]``.

    The colour channels come from the input tensor, not the model output, so
    they are bit-exact with what the user fed in.
    """
    arr = (
        frame.detach().cpu().numpy() if hasattr(frame, "detach") else np.asarray(frame)
    )
    arr = arr.astype(np.float32, copy=False)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return np.concatenate([arr[..., :3], alpha[..., None]], axis=-1)
