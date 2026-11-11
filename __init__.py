"""comfyui-seedvc-voice-swap: zero-shot voice replacement for ComfyUI, built on Seed-VC. GPL-3.0."""
if __package__:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
else:  # imported as a plain file (pytest collecting this folder), not by ComfyUI
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
