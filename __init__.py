"""ComfyUI-AnimaArtistBlend — private test build.

Blends multiple artist tags in the hidden-state space of the Qwen3 0.6B
text encoder used by Anima, instead of stacking them in one prompt
string where they compete and a single artist wins.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
