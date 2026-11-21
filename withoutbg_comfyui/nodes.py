"""withoutBG Remove Background node."""

from __future__ import annotations

import numpy as np

from . import backends
from .convert import alpha_to_array, compose_rgba, frame_to_pil


def _progress_bar(total: int):
    try:
        import comfy.utils

        return comfy.utils.ProgressBar(total)
    except ImportError:
        return None


def _check_interrupt() -> None:
    try:
        import comfy.model_management as mm

        mm.throw_exception_if_processing_interrupted()
    except ImportError:
        pass


class WithoutBGRemoveBackground:
    CATEGORY = "image/withoutBG"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    DESCRIPTION = (
        "Remove the background with withoutBG. Open weights run locally and are "
        "free; Pro uses the Cloud API (opt-in, key from WITHOUTBG_API_KEY, "
        "1 credit per image)."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "engine": (list(backends._ENGINES), {"default": "open_weights"}),
            },
        }

    @staticmethod
    def _ensure_model() -> None:
        """First-run download with a progress bar (one step per model file)."""
        state: dict = {"bar": None, "done": 0}

        def progress(done: int, total: int) -> None:
            if state["bar"] is None:
                state["bar"] = _progress_bar(total)
            elif state["bar"] is not None and done > state["done"]:
                state["bar"].update(done - state["done"])
            state["done"] = done

        backends.ensure_open_weights(progress)

    def run(self, image, engine):
        import torch

        backends.validate_engine(engine)
        client = backends.make_pro_client() if engine == "pro" else None
        if client is None:
            self._ensure_model()

        count = image.shape[0]
        bar = _progress_bar(count)
        rgba, masks = [], []
        for i in range(count):
            _check_interrupt()
            frame = image[i]
            pil = frame_to_pil(frame)
            result = (
                backends.remove_pro(client, pil)
                if client is not None
                else backends.remove_open_weights(pil)
            )
            alpha = alpha_to_array(result, pil.size)
            rgba.append(compose_rgba(frame, alpha))
            masks.append(alpha)
            if bar is not None:
                bar.update(1)

        out = torch.from_numpy(np.stack(rgba)).to(dtype=image.dtype)
        mask = torch.from_numpy(np.stack(masks)).to(dtype=image.dtype)
        return (out, mask)
