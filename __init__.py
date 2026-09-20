"""ComfyUI-H3-Chainloop.

Seamless-loop nodes for MiniMax H3. The headline node, **H3 Chain Loop**,
chains N Ref2VA clips into one video that loops perfectly back to its own
start; the two building-block nodes it uses internally (**H3 Loop Close**
and **H3 Loop Trim**) are also exposed for hand-built graphs.

Registers the nodes without changing ComfyUI's runtime behavior at import
time. They activate the suite's marker-gated H3 keyframe layout patch inline
on first execution (see nodes._ensure_suite_patches), so this pack is inert
until an H3 loop node actually runs.

RUNTIME DEPENDENCY: ComfyUI-H3-Project-Suite must be installed and loaded --
it provides the arbitrary-index H3 keyframe patch the tail keyframes need.
This pack loads BEFORE the suite (custom_nodes load alphabetically), which is
why nodes.py imports nothing from the suite at module load and reaches it
only at node-execution time.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
