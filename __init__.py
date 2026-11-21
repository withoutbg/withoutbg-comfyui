"""ComfyUI entry point: ComfyUI imports this folder as a custom node pack."""

try:
    from .withoutbg_comfyui import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ImportError:  # imported without a parent package (e.g. pytest rootdir)
    from withoutbg_comfyui import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
