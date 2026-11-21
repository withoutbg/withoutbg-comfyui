"""Backends: open weights (in-process ONNX) and Pro (Cloud API)."""

from __future__ import annotations

import logging
import os
import platform
import sys
import threading
import time

from PIL import Image

from .__version__ import __version__

_ENGINES = ("open_weights", "pro")
_DEVICES = ("auto", "cpu")
BACKGROUNDS = ("alpha", "white", "black", "color")
_CPU = "CPUExecutionProvider"
_KEY_URL = "https://withoutbg.com/account?utm_source=comfyui&utm_medium=plugin&utm_campaign=open-weights-v4&utm_content=withoutbg-comfyui-api-key"

log = logging.getLogger("withoutbg_comfyui")

# Files of the open weights bundle (HF repo withoutbg/withoutbg-openweights-onnx,
# pinned by the SDK). The SDK fetches the routed branches lazily, so a hard image
# could trigger a ~1 GB download mid-run. We fetch them all up front instead.
_BUNDLE_FILES = (
    "withoutbg-open-weights.onnx.json",
    "withoutbg-open-weights.onnx",
    "withoutbg-open-weights-backbone.onnx",
    "birefnet-general.onnx",
)

_open_weights: dict[tuple[str, ...], object] = {}
_open_weights_lock = threading.Lock()
_download_lock = threading.Lock()
_gpu_failed = False  # set once a GPU session fails; later runs stay on CPU


class BackendError(RuntimeError):
    """Error with a message that is safe and useful to show in ComfyUI."""


def user_agent(comfy_version: str | None = None) -> str:
    """``withoutbg-comfyui/<ver> (ComfyUI <ver>; <platform>)`` product token.

    The SDK appends its own ``withoutbg-python/...`` token after this one.
    Holds no user or machine identifiers.
    """
    host = f"{sys.platform}-{platform.machine().lower()}"
    comfy = f"ComfyUI {comfy_version}" if comfy_version else "ComfyUI"
    return f"withoutbg-comfyui/{__version__} ({comfy}; {host})"


def _comfy_version() -> str | None:
    try:
        from comfyui_version import __version__ as v  # type: ignore[import-not-found]

        return str(v)
    except ImportError:
        return None


def pick_providers(device: str = "auto") -> list[str]:
    """ONNX Runtime providers for ``device``: GPU first on ``auto``, CPU last."""
    if device not in _DEVICES:
        raise BackendError(f"Unknown device {device!r}; use one of {_DEVICES}.")
    if device == "cpu" or _gpu_failed:
        return [_CPU]
    try:
        import onnxruntime as ort
    except ImportError:
        return [_CPU]
    available = set(ort.get_available_providers())
    chosen = [
        p for p in ("CUDAExecutionProvider", "DmlExecutionProvider") if p in available
    ]
    if "CUDAExecutionProvider" in chosen and hasattr(ort, "preload_dlls"):
        try:  # reuse the CUDA / cuDNN libraries that ComfyUI's torch ships
            ort.preload_dlls()
        except Exception as e:  # best effort
            log.debug("withoutBG: onnxruntime.preload_dlls failed: %s", e)
    return chosen[:1] or [_CPU]


def _get_open_weights(device: str = "auto"):
    providers = pick_providers(device)
    key = tuple(providers)
    with _open_weights_lock:
        if key not in _open_weights:
            from withoutbg import WithoutBG

            _open_weights[key] = WithoutBG.open_weights(providers=providers)
            log.info("withoutBG: open weights running on %s", providers[0])
        return _open_weights[key]


def _missing_bundle_files() -> list[str]:
    """Bundle files not yet in the Hugging Face cache (none if a path override is set)."""
    if os.getenv("WITHOUTBG_MODEL_PATH"):
        return []
    from huggingface_hub import try_to_load_from_cache
    from withoutbg.models import HF_REPO_ID, HF_REVISION

    return [
        name
        for name in _BUNDLE_FILES
        if not isinstance(
            try_to_load_from_cache(HF_REPO_ID, name, revision=HF_REVISION), str
        )
    ]


def ensure_open_weights(progress=None, device: str = "auto") -> int:
    """Download any missing model files up front, logging each one.

    ``progress(done, total)`` is called with ``done=0`` before the first file and
    after each file. Returns the number of files downloaded (0 when cached).
    Nothing is loaded into memory here.
    """
    with _download_lock:
        missing = _missing_bundle_files()
        if not missing:
            return 0
        model = _get_open_weights(device).model
        total = len(missing)
        log.info(
            "withoutBG: downloading the open weights model from Hugging Face "
            "(%d files, about 1.5 GB, first run only). ComfyUI looks idle until "
            "this finishes.",
            total,
        )
        started = time.monotonic()
        if progress:
            progress(0, total)
        for done, name in enumerate(missing, 1):
            log.info("withoutBG: downloading %s (%d/%d)", name, done, total)
            try:
                model._download_from_hf(name, name)
            except Exception as e:  # SDK raises ModelNotFoundError; keep message
                raise BackendError(
                    f"withoutBG could not download {name}: {e}\n"
                    "Check your connection, or download the files from the model "
                    "page and set WITHOUTBG_MODEL_PATH to the folder's .onnx file."
                ) from e
            if progress:
                progress(done, total)
        log.info(
            "withoutBG: model ready (%.0f s). Later runs use the local copy.",
            time.monotonic() - started,
        )
        return total


def remove_open_weights(image: Image.Image, device: str = "auto") -> Image.Image:
    """Alpha matte for ``image`` from the open weights model (RGBA out).

    If a GPU provider fails, warn once and rerun on CPU for the rest of the session.
    """
    global _gpu_failed
    try:
        return _get_open_weights(device).remove_background(image)
    except Exception as e:  # SDK raises WithoutBGError; keep message
        if pick_providers(device) == [_CPU]:
            raise BackendError(f"withoutBG open weights failed: {e}") from e
        log.warning(
            "withoutBG: GPU run failed (%s); falling back to CPU for this session. "
            "Set device to cpu to silence this.",
            e,
        )
        _gpu_failed = True
    try:
        return _get_open_weights(device).remove_background(image)
    except Exception as e:
        raise BackendError(f"withoutBG open weights failed: {e}") from e


def _api_message(err: Exception) -> str:
    text = str(err)
    for code, msg in (
        ("401", "API key rejected (401). Check the key."),
        ("402", "Out of credits (402). Top up at withoutbg.com."),
        ("403", "API key not allowed for this request (403)."),
        ("429", "Rate limited (429). Wait a moment and retry."),
    ):
        if code in text:
            return f"withoutBG Pro: {msg}"
    return f"withoutBG Pro failed: {text}"


def make_pro_client():
    """Cloud API client. Pro is opt-in: key comes from the environment only.

    There is deliberately no key field on the node: ComfyUI saves widget values
    in workflow files and in every saved PNG, so a pasted key would leak.
    """
    key = os.getenv("WITHOUTBG_API_KEY", "").strip()
    if not key:
        raise BackendError(
            "withoutBG Pro needs an API key in the WITHOUTBG_API_KEY environment "
            f"variable, set before starting ComfyUI. Get a key: {_KEY_URL}"
        )
    from withoutbg import WithoutBGAPIClient

    return WithoutBGAPIClient(api_key=key, user_agent=user_agent(_comfy_version()))


def remove_pro(client, image: Image.Image) -> Image.Image:
    try:
        return client.remove_background(image)
    except Exception as e:
        raise BackendError(_api_message(e)) from e


def validate_device(device: str) -> str:
    if device not in _DEVICES:
        raise BackendError(f"Unknown device {device!r}; use one of {_DEVICES}.")
    return device


def validate_engine(engine: str) -> str:
    if engine not in _ENGINES:
        raise BackendError(f"Unknown engine {engine!r}; use one of {_ENGINES}.")
    return engine
