# =============================================================================
# 04_ca_gltf_python.py — glTF Animation Curve Editor  (Seminar 04, Demo 4)
# =============================================================================
# Author   : Renato Fera P.
# LinkedIn : https://www.linkedin.com/in/renato-profeta/
# Course   : Computeranimation — Chapter 04: glTF & Python Animation Tools
# Institute: Institut für Medientechnik, Technische Universität Ilmenau
# =============================================================================
"""glTF Animation Curve Editor — Seminar 04, Demo 4.

Overview
--------
This application lets you open any binary glTF file (.glb), inspect its
skeleton hierarchy, visualise and edit the animation F-curves, and export
the modified result back to disk.  It is intentionally built from first
principles so every layer of a real production animation tool is visible.

Computer Animation Concepts
---------------------------
**F-curves (Function Curves)**
    Every animated property in a 3-D package — position X, rotation Y,
    scale Z — is stored as a *function curve*: a list of (time, value) pairs
    called *keyframes*, with an interpolation rule between them.  Maya calls
    them "animation curves"; Blender calls them "F-curves"; glTF calls them
    "animation samplers".  They are all the same idea: a 1-D spline that maps
    time → value.

**Skeleton / Rig**
    A skeleton is a tree of *joints* (or *bones*).  Each joint has a local
    transform (translation, rotation, scale — TRS) relative to its parent.
    The root joint's world transform is just its own TRS; every other joint's
    world transform is the chain product: world_parent × local_self.

**Skinning (Skeletal Mesh Deformation)**
    The mesh vertices are not rigidly attached to one bone.  Each vertex
    stores up to four (joint_index, weight) pairs.  At runtime the GPU blends
    the four joint matrices weighted by those values to produce a smooth
    deformation — the same formula used in every real-time game engine.

**glTF 2.0**
    The GL Transmission Format stores geometry as typed binary arrays
    (accessors), hierarchy as a node graph, and animation as channels that
    map a sampler (time input → value output) to a target node property
    (translation / rotation / scale / morph weights).

Five-panel layout
-----------------
1. **3-D Viewport** — OpenGL 3.3 Core Profile; Blinn-Phong shading;
   GPU skeletal skinning; skeleton overlay; orbit / pan / zoom camera.
2. **Scene Tree** — glTF node hierarchy (🦴 joints, ○ transform nodes).
3. **Curve Editor** — PyQtGraph F-curve display; click-to-select,
   drag-to-edit keyframes; Escape to revert.
4. **Keyframe Table** — exact numeric values per keyframe; double-click
   a cell to type a new number.
5. **Transport bar** — play / pause / stop / loop, speed slider,
   timeline scrubber.

Workflow
--------
1. File → Open GLB  (Ctrl+O) — load any .glb with animations.
2. Click a node in the Scene Tree — its animation channels appear below.
3. Click a channel (translation / rotation / scale) — curves are drawn.
4. Press Space (or ▶) to play — watch the model animate.
5. Click a keyframe dot on the curve to select it, then drag up/down.
6. File → Export GLB  (Ctrl+S) — save the modified file.

Viewport controls
-----------------
- Left-drag    → orbit the camera around the model.
- Middle-drag  → pan the camera target point.
- Scroll wheel → zoom in / out.
- Double-click → reframe the camera to the model's bounding box.
- Click a joint dot → select that bone in the Scene Tree.

Technology stack
----------------
- **PySide6**   — Qt6 Python binding (used in Maya, Blender, Houdini).
- **PyOpenGL**  — OpenGL 3.3 Core Profile bindings.
- **PyQtGraph** — GPU-accelerated 2-D plot library for the curve editor.
- **pygltflib** — Pure-Python glTF 2.0 reader/writer.
- **NumPy**     — All animation and geometry data lives as typed arrays.

How to run
----------
::

    cd 04_gltf
    uv sync                          # first time — creates .venv, installs deps
    uv run python 04_ca_gltf_python.py

Ubuntu system dependencies (once)::

    sudo apt install libgl1 libegl1 libxcb-cursor0 libxcb-xinerama0

Windows / macOS: no extra steps — PySide6 bundles everything.
"""
# =============================================================================

from __future__ import annotations

import math
import struct
import sys
import traceback
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from pygltflib import GLTF2
from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtGui import (
    QAction, QColor, QFont, QMatrix4x4, QPixmap, QVector3D,
)
from PySide6.QtOpenGL import QOpenGLShader, QOpenGLShaderProgram
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QToolBar,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtGui import QSurfaceFormat
from OpenGL import GL

# =============================================================================
# STEP 1 — Module-level constants
# =============================================================================
#
# glTF encodes every data type as a numeric constant from the WebGL / OpenGL ES
# specification.  For example, 5126 = GL_FLOAT (a 32-bit IEEE 754 float).
# These exact same constants appear in WebGL shaders, Three.js BufferAttribute,
# and every other GL-family API, so learning them here transfers directly to
# browser and game-engine work.

# Maps glTF componentType integer → (NumPy dtype, bytes-per-element).
# Used by read_accessor() to interpret raw binary data from the GLB chunk.
COMPONENT_DTYPE: dict[int, tuple[np.dtype, int]] = {
    5120: (np.dtype("int8"),    1),   # GL_BYTE
    5121: (np.dtype("uint8"),   1),   # GL_UNSIGNED_BYTE
    5122: (np.dtype("int16"),   2),   # GL_SHORT
    5123: (np.dtype("uint16"),  2),   # GL_UNSIGNED_SHORT
    5125: (np.dtype("uint32"),  4),   # GL_UNSIGNED_INT
    5126: (np.dtype("float32"), 4),   # GL_FLOAT  ← most common in animations
}

# Maps glTF accessor type string → number of scalar components per element.
# A "VEC3" position has 3 components (x, y, z); a "VEC4" quaternion has 4.
ACCESSOR_NUM_COMPONENTS: dict[str, int] = {
    "SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4,
    "MAT2":   4, "MAT3": 9, "MAT4": 16,
}

# Curve colours per component: X=red  Y=green  Z=blue  W=yellow.
# This matches the convention used in Maya's Graph Editor and Blender's
# F-Curve Editor, so students who switch between tools see the same colours.
CURVE_COLORS: list[tuple[int, int, int, int]] = [
    (220,  60,  60, 220),   # X — red
    ( 80, 200,  80, 220),   # Y — green
    ( 60, 120, 220, 220),   # Z — blue
    (220, 200,  50, 220),   # W — yellow
]

# Human-readable axis labels for each glTF animation target path.
# "rotation" uses XYZW because glTF stores quaternions in (x, y, z, w) order,
# where W is the scalar part.  "weights" labels are for morph-target blending.
COMPONENT_LABELS: dict[str, list[str]] = {
    "translation": ["X", "Y", "Z"],
    "rotation":    ["X", "Y", "Z", "W"],
    "scale":       ["X", "Y", "Z"],
    "weights":     ["W0", "W1", "W2", "W3"],
}

# Target frame rate for playback.  16 ms ≈ 62.5 fps, close enough to 60 fps.
# A QTimer fires this often; the actual animation time is computed from elapsed
# wall-clock milliseconds so playback speed is independent of frame drops.
PLAYBACK_INTERVAL_MS: int = 16

# AMS group logo — expected one directory above the script (the repo root).
LOGO_PATH: Path = Path(__file__).resolve().parent.parent / "ams_group_tui.png"

# Dark-theme colour palette.  All colours are hex strings consumed by Qt
# stylesheets.  Keeping them in one dict makes global retheming trivial.
PALETTE = {
    "bg":      "#0d0f14",   # darkest background — main window
    "surface": "#161a22",   # slightly lighter — panels, headers
    "border":  "#252b38",   # subtle dividers
    "text":    "#d8dce6",   # primary text
    "muted":   "#5a6175",   # secondary / placeholder text
    "accent":  "#4f9cf9",   # interactive highlights (blue)
    "accent2": "#a78bfa",   # bone nodes in the scene tree (purple)
    "green":   "#34d399",   # positive indicators
    "orange":  "#fb923c",   # warnings / negative values
    "red":     "#f87171",   # errors
}

# Qt stylesheet applied to the entire application.  Python f-string interpolation
# embeds the palette values so the stylesheet reads from one source of truth.
APP_STYLESHEET = f"""
    QMainWindow, QWidget {{
        background-color: {PALETTE['bg']};
        color: {PALETTE['text']};
        font-family: 'IBM Plex Sans', 'Segoe UI', Arial, sans-serif;
        font-size: 13px;
    }}
    QToolBar {{
        background-color: {PALETTE['surface']};
        border-bottom: 1px solid {PALETTE['border']};
        spacing: 6px;
        padding: 4px 8px;
    }}
    QToolBar QLabel {{
        color: {PALETTE['muted']};
        font-size: 12px;
        padding-left: 8px;
    }}
    QPushButton {{
        background-color: {PALETTE['surface']};
        color: {PALETTE['text']};
        border: 1px solid {PALETTE['border']};
        border-radius: 4px;
        padding: 5px 14px;
        font-size: 12px;
    }}
    QPushButton:hover {{
        background-color: {PALETTE['border']};
        border-color: {PALETTE['accent']};
    }}
    QPushButton:pressed {{
        background-color: {PALETTE['accent']};
        color: #ffffff;
    }}
    QPushButton:checked {{
        background-color: {PALETTE['accent']};
        color: #ffffff;
        border-color: {PALETTE['accent']};
    }}
    QPushButton#accent_btn {{
        background-color: {PALETTE['accent']};
        color: #ffffff;
        border-color: {PALETTE['accent']};
        font-weight: 600;
    }}
    QPushButton#accent_btn:hover {{ background-color: #6aabff; }}
    QPushButton#help_btn {{
        background-color: transparent;
        color: {PALETTE['muted']};
        border: 1px solid {PALETTE['border']};
        border-radius: 4px;
        padding: 4px 10px;
        font-size: 12px;
    }}
    QPushButton#help_btn:hover {{
        color: {PALETTE['accent']};
        border-color: {PALETTE['accent']};
    }}
    QTreeWidget {{
        background-color: {PALETTE['surface']};
        border: 1px solid {PALETTE['border']};
        border-radius: 4px;
        color: {PALETTE['text']};
        font-size: 12px;
    }}
    QTreeWidget::item:selected {{
        background-color: {PALETTE['accent']};
        color: #ffffff;
    }}
    QTreeWidget::item:hover {{ background-color: {PALETTE['border']}; }}
    QTableWidget {{
        background-color: {PALETTE['surface']};
        border: 1px solid {PALETTE['border']};
        border-radius: 4px;
        color: {PALETTE['text']};
        gridline-color: {PALETTE['border']};
        font-size: 12px;
        font-family: 'IBM Plex Mono', 'Courier New', monospace;
    }}
    QTableWidget::item:selected {{
        background-color: {PALETTE['accent']};
        color: #ffffff;
    }}
    QHeaderView::section {{
        background-color: {PALETTE['border']};
        color: {PALETTE['muted']};
        border: none;
        padding: 4px 8px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.05em;
    }}
    QSlider::groove:horizontal {{
        height: 4px;
        background-color: {PALETTE['border']};
        border-radius: 2px;
    }}
    QSlider::handle:horizontal {{
        width: 14px; height: 14px; margin: -5px 0;
        background-color: {PALETTE['accent']};
        border-radius: 7px;
    }}
    QSlider::sub-page:horizontal {{
        background-color: {PALETTE['accent']};
        border-radius: 2px;
    }}
    QSplitter::handle {{
        background-color: {PALETTE['border']};
        width: 2px; height: 2px;
    }}
    QStatusBar {{
        background-color: {PALETTE['surface']};
        color: {PALETTE['muted']};
        border-top: 1px solid {PALETTE['border']};
        font-size: 11px;
    }}
    QLabel#section_header {{
        color: {PALETTE['muted']};
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 0.08em;
        padding: 8px 0 4px 0;
    }}
    QLabel#time_display {{
        color: {PALETTE['accent']};
        font-family: 'IBM Plex Mono', 'Courier New', monospace;
        font-size: 14px;
        font-weight: 700;
        padding: 0 8px;
    }}
    QCheckBox {{
        color: {PALETTE['text']};
        spacing: 6px;
        font-size: 12px;
    }}
    QCheckBox::indicator {{
        width: 14px; height: 14px;
        border: 1px solid {PALETTE['border']};
        border-radius: 3px;
        background-color: {PALETTE['surface']};
    }}
    QCheckBox::indicator:checked {{
        background-color: {PALETTE['accent']};
        border-color: {PALETTE['accent']};
    }}
    QScrollBar:vertical {{
        background: {PALETTE['surface']};
        width: 8px; margin: 0;
    }}
    QScrollBar::handle:vertical {{
        background: {PALETTE['border']};
        border-radius: 4px;
        min-height: 20px;
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
    QFrame#divider {{
        background-color: {PALETTE['border']};
        max-height: 1px;
    }}
    QWidget#help_panel {{
        background-color: {PALETTE['surface']};
        border: 1px solid {PALETTE['border']};
        border-radius: 6px;
    }}
"""

# =============================================================================
# STEP 2 — glTF data-access helpers
# =============================================================================
#
# A binary .glb file has two sections:
#   • A JSON chunk — the scene graph, materials, accessor descriptors.
#   • A binary chunk — the raw float / int / byte data referenced by accessors.
#
# pygltflib parses the JSON automatically but hands us the binary chunk as a
# raw bytes object.  The helpers below navigate from accessor → bufferView →
# binary blob → NumPy array, which is the fundamental read operation for all
# animation data and mesh geometry.

def load_gltf(path: Path) -> GLTF2:
    """Load a binary glTF (.glb) file from disk and return the parsed object.

    Args:
        path: Filesystem path to the .glb file.

    Returns:
        A ``pygltflib.GLTF2`` object with the JSON hierarchy parsed and the
        binary blob attached but not yet decoded into NumPy arrays.

    Raises:
        FileNotFoundError: If the path does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"GLB file not found: {path}")
    return GLTF2().load(str(path))


def _get_blob(gltf: GLTF2) -> bytes:
    """Return the GLB binary chunk, decoding and caching it on the first call.

    Why caching is necessary:
        ``pygltflib.binary_blob()`` re-decodes the full binary chunk from
        scratch on every call — there is no internal cache inside pygltflib.
        For large files (e.g. Barbarian.glb with 1.6 MB of geometry and
        animation data) this means 1.6 MB of Python byte-work for every
        single accessor read.  The animation runtime calls read_accessor()
        for every channel on every frame tick, so without a cache the app
        hangs permanently before the first frame is drawn.

        We attach the decoded bytes to the GLTF2 object's ``__dict__``
        directly (bypassing any pygltflib property logic) so the decode
        happens exactly once per loaded model.

    Args:
        gltf: The loaded glTF object.

    Returns:
        The raw binary chunk as an immutable ``bytes`` object.
    """
    # Access the instance dict directly so we do not accidentally trigger
    # any pygltflib property getter.
    cache = object.__getattribute__(gltf, "__dict__")
    if "_blob_cache" not in cache:
        cache["_blob_cache"] = gltf.binary_blob()
    return cache["_blob_cache"]


def read_accessor(gltf: GLTF2, accessor_index: int) -> np.ndarray:
    """Read one glTF accessor as a writable NumPy array.

    An *accessor* in glTF is a typed view into binary data, described by:
        - ``componentType`` — the scalar type (float32, uint16, …)
        - ``type``          — the shape ("SCALAR", "VEC3", "VEC4", "MAT4", …)
        - ``count``         — number of elements
        - ``bufferView``    — which slice of the binary chunk to read
        - ``byteOffset``    — offset within that slice

    This function assembles those fields into a NumPy array so the rest of
    the code can work with clean typed arrays rather than raw bytes.

    Args:
        gltf: The loaded glTF object.
        accessor_index: Index into ``gltf.accessors``.

    Returns:
        A writable float32 (or matching dtype) NumPy array of shape
        ``(count,)`` for SCALAR or ``(count, n_comp)`` for vector/matrix
        types.  Returns an empty array on any error rather than raising, so
        callers can safely continue rendering with defaults.
    """
    if not gltf.accessors or accessor_index >= len(gltf.accessors):
        return np.zeros(0, dtype=np.float32)

    accessor = gltf.accessors[accessor_index]
    dtype, item_size = COMPONENT_DTYPE.get(
        accessor.componentType, (np.dtype("float32"), 4)
    )
    n_comp = ACCESSOR_NUM_COMPONENTS.get(accessor.type, 1)
    count  = accessor.count or 0

    # A bufferView of None means the accessor is "sparse" and its base values
    # are all zeros.  Sparse accessors are uncommon but valid in glTF.
    if accessor.bufferView is None:
        arr = np.zeros((count, n_comp) if n_comp > 1 else (count,), dtype=dtype)
        return arr.copy()

    if not gltf.bufferViews or accessor.bufferView >= len(gltf.bufferViews):
        arr = np.zeros((count, n_comp) if n_comp > 1 else (count,), dtype=dtype)
        return arr.copy()

    buffer_view = gltf.bufferViews[accessor.bufferView]

    # Use the cached blob — not binary_blob() which re-decodes every time.
    blob: bytes = _get_blob(gltf)

    # Byte offsets stack: bufferView offset + accessor offset within the view.
    view_start  = buffer_view.byteOffset or 0
    acc_start   = view_start + (accessor.byteOffset or 0)

    # byteStride is the gap between the start of one element and the next.
    # When omitted (None or 0) the data is tightly packed, which is the
    # common case.  Interleaved buffers (e.g. position + normal + uv in one
    # VBO) use a stride larger than the element size.
    byte_stride   = buffer_view.byteStride or (item_size * n_comp)
    element_bytes = item_size * n_comp

    if byte_stride == element_bytes:
        # Tight layout — a single slice, zero extra copies.
        end = acc_start + count * element_bytes
        raw: bytes = blob[acc_start:end]
    else:
        # Interleaved layout — stride-gather each element individually.
        # This is slower but correct and rare in practice.
        raw = b"".join(
            blob[acc_start + i * byte_stride: acc_start + i * byte_stride + element_bytes]
            for i in range(count)
        )

    # np.frombuffer returns a read-only view; .copy() makes it writable so
    # the caller can safely modify values (e.g. during keyframe editing).
    arr = np.frombuffer(raw, dtype=dtype).copy()
    if n_comp > 1:
        arr = arr.reshape(-1, n_comp)
    return arr


def build_channel_cache(
    gltf: GLTF2,
) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    """Pre-read every animation channel into NumPy arrays at load time.

    Why a cache is necessary:
        The animation runtime calls ``read_accessor()`` for every channel
        on every frame tick.  Even with the binary blob cached, 123 channels
        (Barbarian.glb) means 123 NumPy slice + frombuffer + reshape
        operations per frame at 60 fps — far too expensive for real-time use.

        By reading all accessors once at load time and storing the resulting
        ``(times, values)`` arrays in a dict, every subsequent per-frame
        lookup is a single O(1) dict access with no I/O or allocation.

        The total work is identical to what the original code paid
        continuously at runtime — it is simply amortised to load time, which
        is acceptable (a few seconds) whereas a per-frame hang is not.

    Args:
        gltf: The loaded glTF object.

    Returns:
        A dict mapping ``(anim_index, channel_index)`` to
        ``(times_array, values_array)``.  Attach the result to the GLTF2
        object under ``_channel_cache`` so ``read_channel_data()`` finds it.
    """
    cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    # Warm the blob cache before the loop so binary_blob() is called only once
    # even if _get_blob() is not yet cached.
    _get_blob(gltf)

    for ai, anim in enumerate(gltf.animations or []):
        for ci, channel in enumerate(anim.channels or []):
            sampler = anim.samplers[channel.sampler]
            # sampler.input  → accessor holding keyframe times  (float32, SCALAR)
            # sampler.output → accessor holding keyframe values  (float32, VEC3 or VEC4)
            times  = read_accessor(gltf, sampler.input).flatten().astype(np.float32)
            values = read_accessor(gltf, sampler.output).astype(np.float32)
            cache[(ai, ci)] = (times, values)
    return cache


def read_channel_data(
    gltf: GLTF2, anim_index: int, channel_index: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(times, values)`` for one animation channel.

    Checks the pre-built channel cache first (populated by
    ``build_channel_cache()`` at load time).  Falls back to reading the
    accessor directly when no cache is present, so this function is usable
    in isolation (unit tests, export path) without the full app context.

    Args:
        gltf: The loaded glTF object.
        anim_index: Index into ``gltf.animations``.
        channel_index: Index into ``gltf.animations[anim_index].channels``.

    Returns:
        A tuple ``(times, values)`` where ``times`` is shape ``(n_keys,)``
        and ``values`` is shape ``(n_keys,)`` or ``(n_keys, n_components)``.
    """
    cache = object.__getattribute__(gltf, "__dict__").get("_channel_cache")
    if cache is not None:
        entry = cache.get((anim_index, channel_index))
        if entry is not None:
            return entry

    # Fallback: direct accessor read (no cache present).
    anim    = gltf.animations[anim_index]
    channel = anim.channels[channel_index]
    sampler = anim.samplers[channel.sampler]
    times   = read_accessor(gltf, sampler.input).flatten().astype(np.float32)
    values  = read_accessor(gltf, sampler.output).astype(np.float32)
    return times, values


def get_node_name(gltf: GLTF2, node_index: int | None) -> str:
    """Return the display name for a glTF node, falling back to an index label.

    Args:
        gltf: The loaded glTF object.
        node_index: Index into ``gltf.nodes``, or ``None``.

    Returns:
        The node's ``name`` field, or ``"Node_N"`` if the name is empty,
        or ``"(none)"`` if ``node_index`` is ``None``.
    """
    if node_index is None:
        return "(none)"
    node = gltf.nodes[node_index]
    return node.name or f"Node_{node_index}"


def save_gltf(gltf: GLTF2, path: Path) -> int:
    """Write the modified glTF object to disk and return the file size in bytes.

    Args:
        gltf: The glTF object, potentially with modified binary data.
        path: Destination path for the .glb file.

    Returns:
        The size of the written file in bytes.
    """
    gltf.save(str(path))
    return path.stat().st_size


def write_channel_values(
    gltf: GLTF2, anim_index: int, channel_index: int,
    new_values: np.ndarray,
) -> None:
    """Patch keyframe values directly into the GLB binary chunk in memory.

    How glTF binary editing works:
        The animation values live as raw IEEE 754 floats in the binary chunk.
        The accessor and bufferView tell us exactly which byte range to
        overwrite.  We read the full binary blob as a mutable ``bytearray``,
        patch the relevant bytes with ``struct.pack``, then write the blob
        back to the GLTF2 object via ``set_binary_blob()``.

        No structural changes to the JSON are needed — only the bytes change.
        This is why glTF is a good interchange format: values can be edited
        without rebuilding the whole document.

    Args:
        gltf: The loaded glTF object to patch in place.
        anim_index: Index into ``gltf.animations``.
        channel_index: Index into the animation's channels list.
        new_values: Array of shape ``(n_keys, n_components)`` containing the
            replacement values in float32.
    """
    anim        = gltf.animations[anim_index]
    channel     = anim.channels[channel_index]
    sampler     = anim.samplers[channel.sampler]
    accessor    = gltf.accessors[sampler.output]
    buffer_view = gltf.bufferViews[accessor.bufferView]

    # Mutable copy of the binary chunk.
    blob = bytearray(gltf.binary_blob())

    # Stack the two byte offsets: bufferView start + accessor offset within it.
    byte_offset = (buffer_view.byteOffset or 0) + (accessor.byteOffset or 0)

    # Flatten to 1-D float32 and pack as little-endian binary.
    flat      = new_values.astype(np.float32).flatten()
    new_bytes = struct.pack(f"{len(flat)}f", *flat)

    # Overwrite the exact byte range — no other data is touched.
    blob[byte_offset: byte_offset + len(new_bytes)] = new_bytes
    gltf.set_binary_blob(bytes(blob))


# =============================================================================
# STEP 3 — Scene tree builder
# =============================================================================

def build_scene_tree(
    gltf: GLTF2,
    parent_item: QTreeWidgetItem | None,
    node_index: int,
    skin_joint_set: set[int],
) -> QTreeWidgetItem | None:
    """Recursively build a QTreeWidgetItem for one glTF node.

    What the scene tree represents:
        A glTF scene is a directed acyclic graph of *nodes*.  Each node has
        an optional name, TRS transform, optional mesh, optional skin, and a
        list of children.  Joints (bones) are simply nodes that appear in a
        skin's ``joints`` list — there is no separate "bone" type in glTF.

    Which nodes are shown:
        Nodes whose only role is "this geometry references skin[N]" (skinned-
        mesh container nodes) carry no animation channels and are never
        selected by an animator.  Showing them clutters the tree and confuses
        students.  They are skipped; their children are still traversed and
        re-parented to the nearest visible ancestor so the rest of the
        hierarchy is preserved.  This matches Maya / Blender behaviour, where
        the mesh shape node is hidden and only the rig is shown in the
        outliner.

    Args:
        gltf: The loaded glTF object.
        parent_item: The parent ``QTreeWidgetItem``, or ``None`` for roots.
        node_index: Index into ``gltf.nodes`` for this node.
        skin_joint_set: Set of all node indices that appear in any skin's
            ``joints`` list — used to distinguish bones from plain nodes.

    Returns:
        The created ``QTreeWidgetItem``, or ``None`` when the node is a
        pure skinned-mesh container and is therefore skipped.
    """
    node     = gltf.nodes[node_index]
    name     = node.name or f"Node_{node_index}"
    is_bone  = node_index in skin_joint_set
    has_mesh = node.mesh is not None
    has_skin = node.skin is not None   # node references a skin → skinned-mesh container

    # A node with both a mesh and a skin is a skinned-mesh container: its
    # job is "draw mesh[N] with skin[M]".  It contributes nothing to the
    # animation rig, so we skip it but still recurse into its children.
    if has_mesh and has_skin and not is_bone:
        for child_index in (node.children or []):
            build_scene_tree(gltf, parent_item, child_index, skin_joint_set)
        return None

    # Joints (bones) get a 🦴 icon and purple colour to match the convention
    # used in Maya's Outliner.  Plain transform nodes get a grey ○.
    if is_bone:
        label, badge = f"🦴  {name}", "bone"
    else:
        label, badge = f"○  {name}", "node"

    if parent_item is not None:
        item = QTreeWidgetItem(parent_item, [label, badge, str(node_index)])
    else:
        item = QTreeWidgetItem([label, badge, str(node_index)])

    # Store the glTF node index in the Qt UserRole so selection handlers can
    # retrieve it without parsing the label text.
    item.setData(0, Qt.ItemDataRole.UserRole, node_index)
    if is_bone:
        item.setForeground(0, QColor(PALETTE["accent2"]))

    for child_index in (node.children or []):
        build_scene_tree(gltf, item, child_index, skin_joint_set)
    return item


# =============================================================================
# STEP 4 — Animation channel query
# =============================================================================

def get_animation_channels_for_node(
    gltf: GLTF2, node_index: int
) -> list[tuple[int, int, str, str]]:
    """Find all animation channels that target a specific node.

    In glTF, an animation channel links a *sampler* (which holds the
    keyframe data) to a *target* (which specifies which node and which
    property — translation, rotation, scale, or morph weights — to drive).

    This function scans all animations and returns every channel whose
    target node matches the given index, so the UI can show the full list
    of animatable properties for the selected bone.

    Args:
        gltf: The loaded glTF object.
        node_index: The glTF node index to query.

    Returns:
        A list of tuples ``(anim_index, channel_index, anim_name, path)``
        where ``path`` is one of ``"translation"``, ``"rotation"``,
        ``"scale"``, or ``"weights"``.
    """
    results: list[tuple[int, int, str, str]] = []
    for anim_idx, anim in enumerate(gltf.animations or []):
        anim_name = anim.name or f"Animation_{anim_idx}"
        for ch_idx, channel in enumerate(anim.channels or []):
            if channel.target.node == node_index:
                results.append((anim_idx, ch_idx, anim_name, channel.target.path))
    return results


# =============================================================================
# STEP 5 — Animation runtime helpers (quaternion slerp, matrix math)
# =============================================================================
#
# To animate a skeleton we evaluate the F-curves at the current time, compose
# each joint's local TRS into a world matrix, and then compute the joint
# matrices that the GPU skinning shader needs.  The same math drives
# Three.js AnimationMixer, Unreal's AnimGraph, and Maya's DAG evaluation.

def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two unit quaternions.

    Why SLERP instead of LERP for rotations:
        Linear interpolation of quaternions (LERP) produces the right end
        points but travels along a *chord* of the unit 4-sphere, so the
        result is not a unit quaternion and the speed of rotation is not
        constant.  SLERP travels along the *great arc* of the sphere,
        giving constant angular velocity — the same behaviour you see when
        you interpolate between two key poses in Maya or Blender.

    The negation check (``dot < 0``):
        A unit quaternion q and its negation −q represent the **same**
        3-D rotation (the rotation matrices are identical).  When the dot
        product of q0 and q1 is negative, the two quaternions are on
        opposite hemispheres of the 4-sphere and SLERP would take the
        *long way around* (more than 180° of arc).  Negating q1 flips it
        to the same hemisphere so we always interpolate the short way.

    Args:
        q0: Starting unit quaternion, shape ``(4,)``, order (x, y, z, w).
        q1: Ending unit quaternion, same shape.
        t:  Interpolation parameter in [0, 1].

    Returns:
        Interpolated unit quaternion, shape ``(4,)``.
    """
    dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))

    # If dot < 0, negate q1 to ensure we take the shortest arc.
    if dot < 0.0:
        q1  = -q1
        dot = -dot

    # When the quaternions are very close (dot ≈ 1) the arc is tiny and
    # sin(theta) ≈ 0 — dividing by it would produce numerical garbage.
    # Fall back to normalised LERP, which is accurate for small angles.
    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return result / (np.linalg.norm(result) + 1e-12)

    # General SLERP formula:
    #   slerp(q0, q1, t) = q0 * sin((1-t)*θ) / sin(θ)
    #                     + q1 * sin(t*θ)     / sin(θ)
    # where θ = acos(dot).
    theta0 = math.acos(dot)         # angle between q0 and q1 on the sphere
    theta  = theta0 * t             # angle we want to travel
    sin_t  = math.sin(theta)
    sin_t0 = math.sin(theta0)
    s0     = math.cos(theta) - dot * sin_t / sin_t0
    s1     = sin_t / sin_t0
    return (s0 * q0 + s1 * q1)


def _lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Linear interpolation between two arrays.

    Args:
        a: Start value.
        b: End value.
        t: Interpolation parameter in [0, 1].

    Returns:
        ``a + t * (b - a)``.
    """
    return a + t * (b - a)


def _sample_channel(
    times: np.ndarray, values: np.ndarray, t: float, path: str
) -> np.ndarray:
    """Evaluate an animation channel at time ``t`` using linear interpolation.

    How keyframe evaluation works:
        The sampler holds a sorted array of keyframe times and a parallel
        array of values.  To find the value at an arbitrary time ``t``:
        1. Binary-search to find the two surrounding keyframes i and i+1.
        2. Compute ``alpha = (t - times[i]) / (times[i+1] - times[i])``.
        3. Interpolate: SLERP for rotations (quaternions), LERP for
           translation / scale / morph weights.

        glTF also defines STEP and CUBICSPLINE interpolation; this demo
        implements LINEAR only, which covers the majority of exported rigs.

    Args:
        times:  Keyframe times, shape ``(n_keys,)``, sorted ascending.
        values: Keyframe values, shape ``(n_keys, n_components)``.
        t:      Current playback time in seconds.
        path:   glTF target path: ``"rotation"``, ``"translation"``,
                ``"scale"``, or ``"weights"``.

    Returns:
        Interpolated value array, shape ``(n_components,)``.
    """
    if len(times) == 0:
        return values[0] if len(values) > 0 else np.zeros(3)
    if t <= times[0]:
        return values[0]
    if t >= times[-1]:
        return values[-1]

    # np.searchsorted returns the index where t would be inserted to keep
    # the array sorted.  Subtracting 1 gives the left keyframe index.
    i = int(np.searchsorted(times, t, side="right")) - 1
    i = max(0, min(i, len(times) - 2))

    # Normalise t into [0, 1] between keyframe i and i+1.
    alpha = float((t - times[i]) / (times[i + 1] - times[i] + 1e-12))

    if path == "rotation":
        # Quaternion interpolation must use SLERP to maintain unit length
        # and produce constant angular velocity between poses.
        return _slerp(values[i], values[i + 1], alpha)
    return _lerp(values[i], values[i + 1], alpha)


def _trs_to_mat4(
    translation: np.ndarray | None,
    rotation: np.ndarray | None,
    scale: np.ndarray | None,
) -> np.ndarray:
    """Build a column-major 4×4 float32 matrix from TRS components.

    Why TRS and not a single matrix:
        glTF stores transforms as separate Translation, Rotation (quaternion),
        and Scale components rather than a 4×4 matrix.  This makes
        interpolation straightforward — you can LERP translations, SLERP
        rotations, and LERP scales independently, then compose them.  If you
        stored a 4×4 matrix you would have to decompose it back to TRS before
        interpolating, losing numerical precision.

    Quaternion-to-rotation matrix conversion (glTF order: x, y, z, w):
        The formula comes from expanding q * v * q^{-1} for a pure vector v.
        Each entry of the 3×3 rotation block is a quadratic in {x, y, z, w}.
        For example, R[0,0] = 1 - 2(y² + z²), R[0,1] = 2(xy - zw), etc.

    Composition order:
        The final matrix is T × R × S, applied right-to-left to a column
        vector: M * v = T * (R * (S * v)).  This scales first, then rotates
        in the scaled space, then translates — the standard "SRT" convention
        used by every major 3-D package.

    Args:
        translation: 3-vector (x, y, z) or ``None`` for identity.
        rotation:    4-vector quaternion (x, y, z, w) or ``None``.
        scale:       3-vector (sx, sy, sz) or ``None``.

    Returns:
        Row-major 4×4 float32 transformation matrix.
    """
    # Translation matrix — only the last column differs from identity.
    T = np.eye(4, dtype=np.float32)
    if translation is not None:
        T[:3, 3] = translation[:3]

    # Rotation matrix from unit quaternion (x, y, z, w).
    R = np.eye(4, dtype=np.float32)
    if rotation is not None:
        x, y, z, w = rotation
        R[0, 0] = 1 - 2*(y*y + z*z)
        R[0, 1] = 2*(x*y - z*w)
        R[0, 2] = 2*(x*z + y*w)
        R[1, 0] = 2*(x*y + z*w)
        R[1, 1] = 1 - 2*(x*x + z*z)
        R[1, 2] = 2*(y*z - x*w)
        R[2, 0] = 2*(x*z - y*w)
        R[2, 1] = 2*(y*z + x*w)
        R[2, 2] = 1 - 2*(x*x + y*y)

    # Scale matrix — only the diagonal differs from identity.
    S = np.eye(4, dtype=np.float32)
    if scale is not None:
        S[0, 0], S[1, 1], S[2, 2] = scale[0], scale[1], scale[2]

    # Compose: T × R × S  (@ is NumPy matrix multiplication)
    return T @ R @ S


def _node_rest_mat(node) -> np.ndarray:
    """Return the rest-pose local matrix for a glTF node.

    A node's rest pose is the transform it has when no animation is playing.
    It is stored either as a precomputed 4×4 matrix or as separate TRS
    components — never both simultaneously (glTF spec §5.3.2).

    Args:
        node: A ``pygltflib.Node`` object.

    Returns:
        4×4 float32 matrix in row-major order.
    """
    if node.matrix is not None:
        # glTF stores matrices in column-major (OpenGL) order.
        # NumPy's reshape fills row-first, so we transpose after reshape.
        return np.array(node.matrix, dtype=np.float32).reshape(4, 4).T
    t = np.array(node.translation or [0, 0, 0], dtype=np.float32)
    r = np.array(node.rotation    or [0, 0, 0, 1], dtype=np.float32)
    s = np.array(node.scale       or [1, 1, 1], dtype=np.float32)
    return _trs_to_mat4(t, r, s)


def _build_anim_node_lookup(
    gltf: GLTF2, anim_index: int
) -> dict[int, dict[str, tuple[int, np.ndarray, np.ndarray]]]:
    """Build a reverse index from node → channel data for one animation.

    Why a reverse index:
        The natural glTF structure is a list of channels, each containing a
        target node index.  To evaluate the skeleton we need the inverse:
        given a node, what channels (and thus what keyframe data) target it?
        A linear scan over all channels for each node is O(n_channels) per
        node per frame.  For Barbarian.glb (123 channels, 41 joints) that is
        5 043 iterations per frame at 60 fps — over 300 000 per second.

        Building the reverse index once and storing it reduces per-frame work
        to a single O(1) dict lookup per joint, dropping total frame cost
        from O(n_channels × n_joints) to O(n_joints).

        The result is cached on the GLTF2 object under ``_anim_node_lookup``
        so it is built at most once per animation per loaded model.

    Args:
        gltf: The loaded glTF object.
        anim_index: Index into ``gltf.animations``.

    Returns:
        Dict mapping ``node_index → {path → (channel_index, times, values)}``.
    """
    gltf_dict = object.__getattribute__(gltf, "__dict__")
    top = gltf_dict.setdefault("_anim_node_lookup", {})
    if anim_index in top:
        return top[anim_index]

    result: dict[int, dict[str, tuple[int, np.ndarray, np.ndarray]]] = {}
    if anim_index >= 0 and gltf.animations and anim_index < len(gltf.animations):
        anim = gltf.animations[anim_index]
        for ch_idx, channel in enumerate(anim.channels or []):
            ni = channel.target.node
            if ni is None:
                continue
            path = channel.target.path
            times, values = read_channel_data(gltf, anim_index, ch_idx)
            # Ensure values is always 2-D for uniform indexing downstream.
            if values.ndim == 1:
                values = values[:, np.newaxis]
            result.setdefault(ni, {})[path] = (ch_idx, times, values)

    top[anim_index] = result
    return result


def compute_joint_matrices(
    gltf: GLTF2,
    skin_index: int,
    anim_index: int,
    t: float,
) -> np.ndarray:
    """Compute the joint matrix palette for one skin at playback time ``t``.

    The GPU skeletal skinning formula:
        Each vertex position is displaced by a weighted sum of joint matrices:

            skinned_pos = Σ_i  weight_i * J_i * rest_pos

        where J_i is the joint matrix for joint i:

            J_i  =  world_i  ×  inverse_bind_i

        - ``world_i`` is the joint's animated world-space transform at time t,
          computed by walking the scene graph from the root down to joint i
          and multiplying the animated local matrices along the way.
        - ``inverse_bind_i`` is a fixed matrix stored in the skin's
          ``inverseBindMatrices`` accessor.  It transforms a vertex from
          *world space at rest pose* into *joint-local space*, so the
          subsequent ``world_i`` brings it back into the correct animated
          world position.  Without the inverse bind, applying ``world_i``
          would double-count the rest-pose offset.

        This formula is identical in every GPU skeletal animation system:
        GLSL skinning shaders, DirectX HLSL, Metal shading language, Unreal's
        GPU Skin vertex factory.

    Args:
        gltf: The loaded glTF object.
        skin_index: Index into ``gltf.skins``.
        anim_index: Index into ``gltf.animations`` (-1 for rest pose).
        t: Current playback time in seconds.

    Returns:
        Float32 array of shape ``(n_joints, 4, 4)``.
    """
    skin      = gltf.skins[skin_index]
    joint_ids = skin.joints or []
    n_joints  = len(joint_ids)
    n_nodes   = len(gltf.nodes or [])

    # Start from the rest pose for every node in the scene.
    local_mats = [_node_rest_mat(gltf.nodes[i]) for i in range(n_nodes)]

    # Override rest-pose matrices with animated values using the reverse index.
    # This is an O(n_animated_nodes) pass, not O(n_channels).
    node_lookup = _build_anim_node_lookup(gltf, anim_index)
    for ni, path_map in node_lookup.items():
        node   = gltf.nodes[ni]
        rest_t = np.array(node.translation or [0, 0, 0], dtype=np.float32)
        rest_r = np.array(node.rotation    or [0, 0, 0, 1], dtype=np.float32)
        rest_s = np.array(node.scale       or [1, 1, 1], dtype=np.float32)
        for path, (_ci, times, values) in path_map.items():
            val = _sample_channel(times, values, t, path)
            if path == "translation":
                rest_t = val.flatten()[:3]
            elif path == "rotation":
                rest_r = val.flatten()[:4]
            elif path == "scale":
                rest_s = val.flatten()[:3]
        local_mats[ni] = _trs_to_mat4(rest_t, rest_r, rest_s)

    # Accumulate world matrices with an iterative stack traversal.
    # A recursive implementation would be simpler but Python's default
    # recursion limit (~1000) is too shallow for deep skeletons.
    world_mats = [None] * n_nodes
    root_ids: list[int] = []
    for scene in (gltf.scenes or []):
        root_ids.extend(scene.nodes or [])
    if not root_ids:
        root_ids = list(range(n_nodes))

    # Each stack entry is (node_index, parent_world_matrix).
    stack = [(ri, np.eye(4, dtype=np.float32)) for ri in root_ids]
    while stack:
        ni, parent_world = stack.pop()
        # World matrix = parent's world matrix × this node's local matrix.
        world_mats[ni] = parent_world @ local_mats[ni]
        for ci in (gltf.nodes[ni].children or []):
            stack.append((ci, world_mats[ni]))

    # Read inverse bind matrices from the skin's accessor.
    # They are stored in column-major order (glTF spec §4.3), so we
    # transpose each one after reshaping from the flat accessor.
    ibm_arr = np.eye(4, dtype=np.float32)[np.newaxis].repeat(n_joints, axis=0)
    if skin.inverseBindMatrices is not None:
        raw_ibm = read_accessor(gltf, skin.inverseBindMatrices)
        if raw_ibm.ndim == 1:
            raw_ibm = raw_ibm.reshape(-1, 16)
        for i, row in enumerate(raw_ibm[:n_joints]):
            ibm_arr[i] = row.reshape(4, 4).T

    # Final joint matrix: world × inverse_bind.
    joint_mats = np.zeros((n_joints, 4, 4), dtype=np.float32)
    for i, jid in enumerate(joint_ids):
        wm = world_mats[jid]
        if wm is None:
            wm = np.eye(4, dtype=np.float32)
        joint_mats[i] = wm @ ibm_arr[i]

    return joint_mats


def compute_world_positions(
    gltf: GLTF2,
    anim_index: int,
    t: float,
) -> dict[int, np.ndarray]:
    """Return the world-space origin (vec3) for every node at playback time ``t``.

    This is used to draw the skeleton overlay as lines between parent and
    child joint positions.  It evaluates the same animated world matrices as
    ``compute_joint_matrices()`` but returns only the translation column
    (column 3 of the 4×4 matrix) for each node — the world-space origin.

    Args:
        gltf: The loaded glTF object.
        anim_index: Index into ``gltf.animations``.
        t: Current playback time in seconds.

    Returns:
        Dict mapping ``node_index → world_position (shape (3,) float32)``.
    """
    n_nodes    = len(gltf.nodes or [])
    local_mats = [_node_rest_mat(gltf.nodes[i]) for i in range(n_nodes)]

    node_lookup = _build_anim_node_lookup(gltf, anim_index)
    for ni, path_map in node_lookup.items():
        node   = gltf.nodes[ni]
        rest_t = np.array(node.translation or [0, 0, 0], dtype=np.float32)
        rest_r = np.array(node.rotation    or [0, 0, 0, 1], dtype=np.float32)
        rest_s = np.array(node.scale       or [1, 1, 1], dtype=np.float32)
        for path, (_ci, times, values) in path_map.items():
            val = _sample_channel(times, values, t, path)
            if path == "translation":
                rest_t = val.flatten()[:3]
            elif path == "rotation":
                rest_r = val.flatten()[:4]
            elif path == "scale":
                rest_s = val.flatten()[:3]
        local_mats[ni] = _trs_to_mat4(rest_t, rest_r, rest_s)

    world_mats: dict[int, np.ndarray] = {}
    roots: list[int] = []
    for scene in (gltf.scenes or []):
        roots.extend(scene.nodes or [])
    if not roots:
        roots = list(range(n_nodes))

    stack2 = [(r, np.eye(4, dtype=np.float32)) for r in roots]
    while stack2:
        ni, pw = stack2.pop()
        world_mats[ni] = pw @ local_mats[ni]
        for ci in (gltf.nodes[ni].children or []):
            stack2.append((ci, world_mats[ni]))

    # Extract the translation column (column 3, rows 0–2) from each 4×4 matrix.
    # This gives the world-space position of each node's origin.
    return {ni: wm[:3, 3] for ni, wm in world_mats.items()}


def _build_node_parent_map(gltf: GLTF2) -> dict[int, int]:
    """Build a child → parent lookup for the entire scene graph.

    glTF stores the graph as parent → children lists, which is efficient
    for top-down traversal.  For prop rendering we need the inverse: given a
    prop node, walk upward to the root accumulating animated world transforms.
    This O(n_nodes) pre-pass builds that mapping once and caches it on the
    GLTF2 object.

    Args:
        gltf: The loaded glTF object.

    Returns:
        Dict mapping ``child_node_index → parent_node_index``.
        Root nodes (no parent) have no entry in the dict.
    """
    gltf_dict = object.__getattribute__(gltf, "__dict__")
    if "_node_parent_map" in gltf_dict:
        return gltf_dict["_node_parent_map"]
    parent_map: dict[int, int] = {}
    for ni, node in enumerate(gltf.nodes or []):
        for ci in (node.children or []):
            parent_map[ci] = ni
    gltf_dict["_node_parent_map"] = parent_map
    return parent_map


def _get_node_world_mat(
    gltf: GLTF2,
    node_index: int,
    anim_index: int,
    t: float,
    parent_map: dict[int, int],
) -> np.ndarray:
    """Return the animated world-space 4×4 matrix for any single node at time ``t``.

    How prop attachment works in a game engine:
        A weapon, shield, or hat is a mesh parented to a bone node in the
        scene graph.  To draw it at the right position, the engine computes
        the full animated world transform of that node by walking up the
        parent chain and multiplying animated local TRS matrices along the
        way — no skin / inverse-bind step is needed because the mesh moves
        rigidly with its parent bone.

        This function replicates that chain exactly:
        1. Collect the ancestor chain: node → parent → … → root.
        2. Sample each ancestor's animated local TRS at time ``t``
           (falling back to the rest pose if no channel targets that node).
        3. Multiply root-first to accumulate the world matrix.

        Result: ``M_world = M_root @ M_child1 @ … @ M_node``

    Args:
        gltf: The loaded glTF object.
        node_index: The node whose world matrix we want.
        anim_index: Which animation is currently playing.
        t: Current playback time in seconds.
        parent_map: Pre-built child → parent lookup from ``_build_node_parent_map()``.

    Returns:
        4×4 float32 world matrix for the node at time ``t``.
    """
    # Collect ancestor chain bottom-up, then reverse to get root-first order.
    chain: list[int] = []
    cur = node_index
    while cur is not None:
        chain.append(cur)
        cur = parent_map.get(cur)
    chain.reverse()   # now root → … → node

    node_lookup = _build_anim_node_lookup(gltf, anim_index)

    world = np.eye(4, dtype=np.float32)
    for ni in chain:
        node     = gltf.nodes[ni]
        path_map = node_lookup.get(ni, {})
        # Start from rest pose and override any animated channels.
        rest_t = np.array(node.translation or [0, 0, 0], dtype=np.float32)
        rest_r = np.array(node.rotation    or [0, 0, 0, 1], dtype=np.float32)
        rest_s = np.array(node.scale       or [1, 1, 1], dtype=np.float32)
        for path, (_ci, times, values) in path_map.items():
            val = _sample_channel(times, values, t, path)
            if path == "translation":
                rest_t = val.flatten()[:3]
            elif path == "rotation":
                rest_r = val.flatten()[:4]
            elif path == "scale":
                rest_s = val.flatten()[:3]
        world = world @ _trs_to_mat4(rest_t, rest_r, rest_s)
    return world


# =============================================================================
# STEP 6 — 3-D OpenGL viewport
# =============================================================================
#
# The 3-D viewport uses OpenGL 3.3 Core Profile — the version that removed all
# deprecated fixed-function features (glBegin/glEnd, glMatrixMode, etc.) and
# requires explicit shader programs for every draw call.  This is also the
# minimum version that runs on macOS (which supports 3.3 Core and nothing older
# on modern hardware).
#
# Two shader programs are compiled at startup:
#   _VERT_SRC / _FRAG_SRC  — mesh shader (Blinn-Phong + GPU skinning)
#   _LINE_VERT / _LINE_FRAG — line shader (flat colour, no lighting)

# -----------------------------------------------------------------------------
# Mesh vertex shader
# -----------------------------------------------------------------------------
# Input attributes (set by _make_vao via glVertexAttribPointer):
#   location 0 — aPos       vec3  vertex position in model space
#   location 1 — aNormal    vec3  vertex normal in model space
#   location 2 — aJoints    vec4  indices of up to 4 influencing joints
#   location 3 — aWeights   vec4  blend weights for those joints (sum ≈ 1)
#   location 4 — aTexCoord  vec2  UV coordinates
#
# GPU Skeletal Skinning:
#   When uSkinned is true, the shader blends four joint matrices weighted by
#   aWeights to form a single "skin matrix", then transforms the vertex and
#   normal through it.  This is the standard linear blend skinning (LBS)
#   algorithm used in virtually every real-time game engine.
#
#   skin = Σ_i  aWeights[i] * uJoints[aJoints[i]]
#
#   LBS is not physically perfect (it produces the "candy-wrapper" artefact
#   near twisting joints) but it is fast enough to run at 60 fps on the GPU
#   with hundreds of thousands of vertices.

_VERT_SRC = """
#version 330 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec4 aJoints;
layout(location = 3) in vec4 aWeights;
layout(location = 4) in vec2 aTexCoord;

uniform mat4 uModel;        // model → world (identity for skinned meshes)
uniform mat4 uView;         // world → camera space
uniform mat4 uProj;         // camera → clip space (perspective)
uniform mat4 uJoints[64];   // joint matrix palette — up to 64 bones
uniform bool uSkinned;      // true → apply GPU skinning

out vec3 vNormal;           // interpolated world-space normal → fragment shader
out vec3 vFragPos;          // interpolated world-space position → fragment shader
out vec2 vTexCoord;

void main() {
    vec4 pos4 = vec4(aPos, 1.0);
    vec3 norm = aNormal;

    if (uSkinned) {
        // Build the blended skin matrix from up to four weighted joint matrices.
        // aJoints stores joint INDICES (as floats); int() converts for indexing.
        mat4 skin =
            aWeights.x * uJoints[int(aJoints.x)] +
            aWeights.y * uJoints[int(aJoints.y)] +
            aWeights.z * uJoints[int(aJoints.z)] +
            aWeights.w * uJoints[int(aJoints.w)];

        // Apply the skin matrix to position and normal.
        // The normal uses only the upper-left 3×3 of the skin matrix.
        pos4 = skin * pos4;
        norm = mat3(skin) * aNormal;
    }

    // Transform from model space → world space → clip space.
    vec4 worldPos = uModel * pos4;
    gl_Position   = uProj * uView * worldPos;

    // Pass world-space values to the fragment shader for lighting.
    vFragPos  = worldPos.xyz;
    // The normal transform must use the inverse-transpose of the model matrix
    // so non-uniform scale does not distort the normals.
    vNormal   = normalize(mat3(transpose(inverse(uModel))) * norm);
    vTexCoord = aTexCoord;
}
"""

# -----------------------------------------------------------------------------
# Mesh fragment shader — Blinn-Phong lighting + gamma correction
# -----------------------------------------------------------------------------
# Blinn-Phong is a classic real-time lighting model with three components:
#   Ambient  — constant low-level illumination so shadows are never pure black.
#   Diffuse  — Lambertian shading: brightness ∝ cos(angle between N and L).
#   Specular — Blinn's halfway-vector approximation to Phong specular.
#
# Blinn-Phong vs Phong:
#   Original Phong specular uses reflect(L, N) · V.
#   Blinn's version uses H = normalize(L + V) and computes N · H instead.
#   The halfway vector H stays between L and V, so it is cheaper to compute
#   and behaves better at grazing angles.
#
# Gamma correction:
#   Monitors apply a ~2.2 gamma curve when displaying pixels.  All lighting
#   must be done in linear light space; at the end we raise the result to
#   1/2.2 to encode it back into the sRGB-ish display space.  Without this,
#   shadows are too dark and highlights are too bright.

_FRAG_SRC = """
#version 330 core

in vec3 vNormal;
in vec3 vFragPos;
in vec2 vTexCoord;

uniform vec3      uLightPos;    // world-space position of the key light
uniform vec3      uLightColor;  // light colour (white = (1,1,1))
uniform vec4      uBaseColor;   // per-material base colour factor (RGBA, linear)
uniform float     uAmbient;     // ambient intensity [0, 1]
uniform bool      uHasTexture;  // true → sample albedo from uAlbedo
uniform sampler2D uAlbedo;      // base colour texture (uploaded as GL_SRGB8_ALPHA8)

out vec4 FragColor;

void main() {
    // ── Albedo (all computation in linear space) ──────────────────────────────
    vec4 albedo = uBaseColor;
    if (uHasTexture) {
        // GL_SRGB8_ALPHA8 tells OpenGL to auto-convert sRGB texels → linear
        // during sampling, so this multiplication stays in linear space.
        albedo *= texture(uAlbedo, vTexCoord);
    }

    // ── Blinn-Phong lighting ──────────────────────────────────────────────────
    vec3 N      = normalize(vNormal);
    vec3 L      = normalize(uLightPos - vFragPos);   // surface → light
    float NdotL = max(dot(N, L), 0.0);              // Lambertian term

    // Halfway vector H for Blinn specular.
    // -vFragPos gives the view direction when vFragPos is in view space;
    // here it approximates it since vFragPos is in world space and the
    // camera origin varies, but it is good enough for a demo.
    vec3  V     = normalize(-vFragPos);
    vec3  H     = normalize(L + V);
    float NdotH = max(dot(N, H), 0.0);
    float spec  = pow(NdotH, 32.0) * 0.25;   // 32 = shininess exponent

    vec3 ambient  = uAmbient * albedo.rgb;
    vec3 diffuse  = NdotL   * albedo.rgb * uLightColor;
    vec3 specular = spec     * uLightColor;

    vec3 result = ambient + diffuse + specular;

    // ── Gamma-encode for display ──────────────────────────────────────────────
    // Most framebuffers on student machines are NOT sRGB-linear, so we apply
    // the gamma curve manually: out = clamp(linear, 0,1) ^ (1/2.2).
    result    = pow(clamp(result, 0.0, 1.0), vec3(1.0 / 2.2));
    FragColor = vec4(result, albedo.a);
}
"""

# -----------------------------------------------------------------------------
# Skeleton line shader — flat colour, no lighting
# -----------------------------------------------------------------------------
# The skeleton is drawn as GL_LINES and GL_POINTS directly in world space.
# The MVP (model × view × projection) matrix is uploaded as a single uniform
# since the skeleton does not need per-vertex lighting.

_LINE_VERT = """
#version 330 core
layout(location = 0) in vec3 aPos;
uniform mat4 uMVP;
void main() { gl_Position = uMVP * vec4(aPos, 1.0); }
"""

_LINE_FRAG = """
#version 330 core
uniform vec3 uColor;
out vec4 FragColor;
void main() { FragColor = vec4(uColor, 1.0); }
"""


class ModelViewportWidget(QOpenGLWidget):
    """OpenGL 3.3 Core Profile 3-D viewport for the glTF model.

    Renders the loaded mesh with Blinn-Phong shading and overlays the
    skeleton as coloured bone lines and joint dots.  The model updates live
    during playback via ``set_playback_time()``.

    Camera model — orbit / pan / zoom:
        The camera orbits around a *target point* (``_bbox_center``) at a
        fixed distance (``_orbit_dist``), with the angle controlled by yaw
        and pitch.  This is the same "tumble" model used in Maya (Alt+LMB)
        and Blender (middle-drag).

        - **Left-drag**    → orbit: change yaw and pitch.
        - **Middle-drag**  → pan: translate ``_bbox_center`` in the camera's
          local X/Y plane so the model follows the cursor.
        - **Scroll wheel** → zoom: scale ``_orbit_dist``.
        - **Double-click** → reframe: restore ``_bbox_center`` and
          ``_orbit_dist`` to the values computed from the model's bounding box.

    OpenGL pipeline summary:
        CPU numpy arrays → VBO upload → vertex shader (skinning)
        → fragment shader (Blinn-Phong) → framebuffer → screen.

    Attributes:
        bone_selected: Qt Signal emitted with the glTF node index when the
            user clicks a joint dot in the skeleton overlay.
    """

    # Signal emitted when the user clicks a joint dot in the skeleton overlay.
    # The int payload is the glTF node index of the nearest joint.
    bone_selected = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # glTF data
        self._gltf:       GLTF2 | None = None
        self._anim_index: int          = -1
        self._playback_t: float        = 0.0

        # GPU drawables — one entry per renderable mesh primitive.
        # Each dict holds: vao, vbo, ebo, index_count, skin_index, node_index,
        # is_skinned, has_tex, tex_id, base_color.
        self._drawables: list[dict] = []

        # Compiled shader programs (created in initializeGL).
        self._prog:      QOpenGLShaderProgram | None = None
        self._line_prog: QOpenGLShaderProgram | None = None

        # Skeleton line geometry — one VAO/VBO pair, refilled each frame.
        self._line_vao: int = 0
        self._line_vbo: int = 0

        # Camera state — orbit camera around _bbox_center.
        self._orbit_yaw:   float = 30.0    # horizontal angle in degrees
        self._orbit_pitch: float = 15.0    # vertical angle in degrees
        self._orbit_dist:  float = 3.0     # distance from target to eye

        # Mouse tracking for orbit (left-button) and pan (middle-button).
        self._last_mouse     = None   # QPointF while left button is held
        self._pan_last_mouse = None   # QPointF while middle button is held

        # World-space centre and scale of the loaded model's bounding box.
        # _bbox_center is also the camera target point for orbit / pan.
        self._bbox_center = np.zeros(3, dtype=np.float32)
        self._bbox_scale: float = 1.0

        # Saved camera defaults for double-click reframe.
        # Stored when the model is loaded so the camera can be reset at any time.
        self._bbox_center_default  = np.zeros(3, dtype=np.float32)
        self._orbit_dist_default:  float = 3.0
        self._orbit_yaw_default:   float = 30.0
        self._orbit_pitch_default: float = 15.0

        # Skeleton display options.
        self._show_skeleton:  bool = True   # toggled by toolbar button
        self._selected_joint: int  = -1     # highlighted joint node index

        # Path hint for external texture resolution (rare in .glb files).
        self._gltf_path_hint: "Path | None" = None

        # World-space joint positions populated each frame for bone picking.
        self._joint_world_pos: dict[int, np.ndarray] = {}

        self.setMinimumSize(320, 240)
        # ClickFocus: the viewport acquires keyboard focus on mouse click,
        # which is needed for any key handlers placed at the widget level.
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_model(self, gltf: GLTF2, anim_index: int = 0,
                   path: "Path | None" = None) -> None:
        """Upload all mesh primitives from ``gltf`` to the GPU.

        Args:
            gltf: The loaded glTF object.
            anim_index: Which animation to play (-1 for rest pose).
            path: Optional filesystem path to the .glb, used to resolve
                external texture URIs (rare in self-contained .glb files).
        """
        self._gltf           = gltf
        self._anim_index     = anim_index
        self._playback_t     = 0.0
        self._gltf_path_hint = path

        # makeCurrent / doneCurrent bracket all OpenGL calls made outside
        # the paintGL / initializeGL callbacks to ensure the correct context
        # is active on the calling thread.
        self.makeCurrent()
        self._upload_mesh()
        self._upload_skeleton_vbo()
        self.doneCurrent()
        self.update()

    def set_playback_time(self, t: float) -> None:
        """Set the current playback time and schedule a redraw.

        Args:
            t: Time in seconds.
        """
        self._playback_t = t
        self.update()

    def set_anim_index(self, idx: int) -> None:
        """Switch to a different animation clip.

        Args:
            idx: Index into ``gltf.animations``.
        """
        self._anim_index = idx
        self.update()

    def set_skeleton_visible(self, visible: bool) -> None:
        """Show or hide the skeleton overlay.

        Args:
            visible: ``True`` to show, ``False`` to hide.
        """
        self._show_skeleton = visible
        self.update()

    def set_selected_joint(self, node_index: int) -> None:
        """Highlight a joint by its glTF node index.

        Called when the user selects a bone in the Scene Tree so the
        corresponding joint dot turns orange in the viewport.

        Args:
            node_index: glTF node index of the joint to highlight.
        """
        self._selected_joint = node_index
        self.update()

    # ------------------------------------------------------------------
    # Qt OpenGL lifecycle callbacks
    # ------------------------------------------------------------------

    def initializeGL(self) -> None:
        """Set up OpenGL state and compile shader programs.

        Called once by Qt after the OpenGL context is created.  All one-time
        GPU setup (depth test, face culling, shader compilation, VAO
        allocation) lives here.
        """
        # Depth testing: fragments closer to the camera overwrite farther ones.
        GL.glEnable(GL.GL_DEPTH_TEST)

        # Back-face culling: skip triangles whose normal points away from the
        # camera.  This halves the fragment shader work on closed meshes.
        GL.glEnable(GL.GL_CULL_FACE)

        # Set the background clear colour from the dark-theme palette.
        GL.glClearColor(
            int(PALETTE["bg"][1:3], 16) / 255,
            int(PALETTE["bg"][3:5], 16) / 255,
            int(PALETTE["bg"][5:7], 16) / 255,
            1.0,
        )

        # Compile and link the mesh shader.  Any compile/link error is printed
        # to the console so a blank viewport is diagnosable.
        self._prog = QOpenGLShaderProgram(self)
        ok = self._prog.addShaderFromSourceCode(
            QOpenGLShader.ShaderTypeBit.Vertex, _VERT_SRC)
        if not ok:
            print(f"[Viewport] Vertex shader compile error:\n{self._prog.log()}")
        ok = self._prog.addShaderFromSourceCode(
            QOpenGLShader.ShaderTypeBit.Fragment, _FRAG_SRC)
        if not ok:
            print(f"[Viewport] Fragment shader compile error:\n{self._prog.log()}")
        if not self._prog.link():
            print(f"[Viewport] Mesh shader link error:\n{self._prog.log()}")

        # Compile and link the skeleton line shader.
        self._line_prog = QOpenGLShaderProgram(self)
        ok = self._line_prog.addShaderFromSourceCode(
            QOpenGLShader.ShaderTypeBit.Vertex, _LINE_VERT)
        if not ok:
            print(f"[Viewport] Line vertex shader error:\n{self._line_prog.log()}")
        ok = self._line_prog.addShaderFromSourceCode(
            QOpenGLShader.ShaderTypeBit.Fragment, _LINE_FRAG)
        if not ok:
            print(f"[Viewport] Line fragment shader error:\n{self._line_prog.log()}")
        if not self._line_prog.link():
            print(f"[Viewport] Line shader link error:\n{self._line_prog.log()}")

        # Allocate the skeleton VAO/VBO here.  Mesh VAOs are created per
        # primitive inside _upload_mesh() since we don't know the count yet.
        self._line_vao = int(GL.glGenVertexArrays(1))
        self._line_vbo = int(GL.glGenBuffers(1))

    def resizeGL(self, w: int, h: int) -> None:
        """Update the OpenGL viewport when the widget is resized.

        Args:
            w: New width in pixels.
            h: New height in pixels.
        """
        GL.glViewport(0, 0, w, h)

    def paintGL(self) -> None:
        """Render one frame.

        Qt calls this automatically after every ``update()`` call.  The
        sequence is: clear → draw mesh → draw skeleton.
        """
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        if self._gltf is None or not self._drawables:
            self._draw_empty_hint()
            return
        self._draw_mesh()
        if self._show_skeleton:
            self._draw_skeleton()

    # ------------------------------------------------------------------
    # Mouse event handlers — orbit, pan, zoom, bone picking
    # ------------------------------------------------------------------

    def mouseDoubleClickEvent(self, event) -> None:
        """Reframe the camera to the model's bounding-box fit on double-click.

        Restores ``_bbox_center``, ``_orbit_dist``, yaw, and pitch to the
        values saved when the model was loaded.  This gives a one-click
        escape from an accidental pan or extreme zoom.

        The event is ``accept()``ed so it does not propagate to the window
        title bar, which would toggle the maximised state on some platforms.

        Args:
            event: QMouseEvent from Qt.
        """
        self._bbox_center  = self._bbox_center_default.copy()
        self._orbit_dist   = self._orbit_dist_default
        self._orbit_yaw    = self._orbit_yaw_default
        self._orbit_pitch  = self._orbit_pitch_default
        self.update()
        event.accept()

    def mousePressEvent(self, event) -> None:
        """Begin orbit (left button) or pan (middle button) on mouse press.

        Args:
            event: QMouseEvent from Qt.
        """
        pos = event.position()
        if event.button() == Qt.MouseButton.LeftButton:
            # Store the press position so mouseMoveEvent can compute a delta.
            # _try_pick_bone fires immediately on press; if the user then
            # drags, the pick is suppressed (see mouseMoveEvent).
            self._last_mouse = pos
            self._try_pick_bone(pos)
        elif event.button() == Qt.MouseButton.MiddleButton:
            # Middle button (scroll wheel click) starts a pan operation.
            # We track it separately from _last_mouse so orbit and pan can
            # coexist without interfering with each other.
            self._pan_last_mouse = pos

    def mouseReleaseEvent(self, event) -> None:
        """End orbit or pan tracking on mouse release.

        Args:
            event: QMouseEvent from Qt.
        """
        if event.button() == Qt.MouseButton.LeftButton:
            self._last_mouse = None
        elif event.button() == Qt.MouseButton.MiddleButton:
            self._pan_last_mouse = None

    def mouseMoveEvent(self, event) -> None:
        """Update orbit or pan while the mouse is dragged.

        Orbit (left-drag):
            Accumulates yaw (horizontal) and pitch (vertical) from the pixel
            delta.  Pitch is clamped to ±89° so the camera never flips.

        Pan (middle-drag):
            Translates ``_bbox_center`` (the orbit target) in the camera's
            local X/Y plane.  The pixel delta is converted to world-space
            units using the camera's right vector (``r_vec``) and up vector
            (``u_vec``), scaled by ``_orbit_dist / viewport_height`` so one
            pixel of pan always moves the same fraction of the visible scene
            regardless of zoom level.

            We recompute ``r_vec`` and ``u_vec`` from the current yaw/pitch
            here rather than reading them from ``_mvp()`` because ``_mvp()``
            does not cache those vectors — it would allocate the full 4×4
            view matrix just to discard it, which is wasteful inside a
            per-pixel mouse handler.

        Args:
            event: QMouseEvent from Qt.
        """
        pos = event.position()

        # ── Left-drag orbit ───────────────────────────────────────────────────
        if self._last_mouse is not None:
            dx = pos.x() - self._last_mouse.x()
            dy = pos.y() - self._last_mouse.y()
            if abs(dx) > 2 or abs(dy) > 2:
                # A real drag started — suppress the bone pick that fired on
                # press so clicking and dragging don't both select a bone.
                self._last_mouse = None
            self._orbit_yaw  += dx * 0.4
            self._orbit_pitch = max(-89.0, min(89.0, self._orbit_pitch + dy * 0.4))
            self._last_mouse  = pos
            self.update()

        # ── Middle-drag pan ───────────────────────────────────────────────────
        if self._pan_last_mouse is not None:
            dx_px = pos.x() - self._pan_last_mouse.x()
            dy_px = pos.y() - self._pan_last_mouse.y()

            # Reconstruct the camera's right (r_vec) and up (u_vec) vectors
            # from yaw and pitch — the same vectors _mvp() computes for the
            # view matrix.  We need them to know which world-space directions
            # correspond to screen-space left/right and up/down.
            yaw   = math.radians(self._orbit_yaw)
            pitch = math.radians(self._orbit_pitch)
            cx    = math.cos(pitch) * math.sin(yaw)
            cy    = math.sin(pitch)
            cz    = math.cos(pitch) * math.cos(yaw)
            f_vec = -np.array([cx, cy, cz], dtype=np.float32)   # forward (toward target)
            up    = np.array([0.0, 1.0, 0.0], dtype=np.float32)

            # Right vector: perpendicular to forward and world-up.
            r_vec = np.cross(f_vec, up)
            norm  = np.linalg.norm(r_vec)
            if norm > 1e-6:
                r_vec /= norm

            # Camera up vector: perpendicular to right and forward.
            u_vec = np.cross(r_vec, f_vec)
            norm  = np.linalg.norm(u_vec)
            if norm > 1e-6:
                u_vec /= norm

            # Scale factor: orbit_dist / viewport_height gives world units per
            # pixel, keeping pan speed proportional to the visible scene size.
            h_px  = max(1, self.height())
            scale = self._orbit_dist / h_px

            # Moving right on screen (dx > 0) should shift the target leftward
            # in world space, so we subtract r_vec; moving down (dy > 0) should
            # shift the target downward, so we subtract u_vec.
            self._bbox_center -= r_vec * (dx_px * scale)
            self._bbox_center += u_vec * (dy_px * scale)
            self._pan_last_mouse = pos
            self.update()

    def wheelEvent(self, event) -> None:
        """Zoom in / out proportionally on scroll wheel.

        Each standard scroll notch (angleDelta = 120) multiplies
        ``_orbit_dist`` by ``0.999^120 ≈ 0.887`` (zoom in by ~11%).
        Proportional zoom feels linear in log-space, matching Blender
        and Maya behaviour.

        Args:
            event: QWheelEvent from Qt.
        """
        delta = event.angleDelta().y()
        # 0.999^120 ≈ 0.887 per notch — feel-tuned for typical trackpads.
        self._orbit_dist *= 0.999 ** delta
        self._orbit_dist  = max(0.01, self._orbit_dist)
        self.update()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _mvp(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the model, view, and projection matrices for the current frame.

        All three matrices are returned as row-major 4×4 float32 arrays.
        They are uploaded to the GPU with ``GL_TRUE`` (transpose flag),
        which makes OpenGL interpret each one as row-major — matching the
        standard mathematical convention M @ v rather than v @ M.

        Perspective projection matrix (derivation):
            Starting from a symmetric frustum with half-FOV angle α, aspect
            ratio a, near plane n and far plane f, the projection matrix maps
            the frustum to the OpenGL NDC cube [-1,1]^3.  In row-major form:

            Row 0: [ f/(a·tan(α))   0           0               0        ]
            Row 1: [      0       f/tan(α)      0               0        ]
            Row 2: [      0          0    (f+n)/(n-f)   2·f·n/(n-f)      ]
            Row 3: [      0          0        -1               0        ]

            The ``-1`` in row 3, column 2 (``proj[3,2]``) is the *perspective
            divide* — it copies the negated Z into the W component of the
            clip-space vector, so the GPU can divide XYZ by W to produce NDC
            coordinates.  Without it every vertex maps to W=0 and NDC becomes
            undefined (all vertices appear at the horizon).

        Near/far scaling:
            The near plane is set to 0.5% of the orbit distance so the camera
            can zoom in without clipping the mesh.  The far plane is 200×
            the orbit distance so large meshes (Fox.glb is ~150 units tall)
            are never clipped on the far side.

        Returns:
            Tuple ``(model, view, proj)`` of 4×4 float32 ndarrays.
        """
        w, h   = max(1, self.width()), max(1, self.height())
        aspect = w / h

        # Scale near/far with orbit distance so clipping adapts to zoom level.
        near = max(0.001, self._orbit_dist * 0.005)
        far  = self._orbit_dist * 200.0

        # Perspective projection matrix — derived from the frustum equations.
        fov    = math.radians(45.0)
        f      = 1.0 / math.tan(fov / 2)   # cotangent of half-FOV
        proj   = np.zeros((4, 4), dtype=np.float32)
        proj[0, 0] = f / aspect            # X scale (accounts for aspect ratio)
        proj[1, 1] = f                     # Y scale
        proj[2, 2] = (far + near) / (near - far)
        proj[2, 3] = (2 * far * near) / (near - far)   # Z translation term
        proj[3, 2] = -1.0                              # perspective divide trigger

        # Orbit camera — eye position computed from yaw, pitch, and distance.
        yaw   = math.radians(self._orbit_yaw)
        pitch = math.radians(self._orbit_pitch)
        cx    = self._orbit_dist * math.cos(pitch) * math.sin(yaw)
        cy    = self._orbit_dist * math.sin(pitch)
        cz    = self._orbit_dist * math.cos(pitch) * math.cos(yaw)
        eye   = self._bbox_center + np.array([cx, cy, cz], dtype=np.float32)
        ctr   = self._bbox_center.copy()
        up    = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        # Orthonormal camera basis from the standard look-at construction.
        f_vec = (ctr - eye); f_vec /= (np.linalg.norm(f_vec) + 1e-9)   # forward
        r_vec = np.cross(f_vec, up); r_vec /= (np.linalg.norm(r_vec) + 1e-9)   # right
        u_vec = np.cross(r_vec, f_vec)                                            # up

        # Row-major view matrix: dot each basis vector with the eye position
        # to build the camera-space translation.
        view  = np.eye(4, dtype=np.float32)
        view[0, :3] = r_vec
        view[1, :3] = u_vec
        view[2, :3] = -f_vec
        view[0, 3]  = -np.dot(r_vec, eye)
        view[1, 3]  = -np.dot(u_vec, eye)
        view[2, 3]  =  np.dot(f_vec, eye)

        # Model matrix is identity: all world-space transforms are baked into
        # the joint matrices for skinned meshes, or into the per-drawable
        # model matrix for unskinned props.
        model = np.eye(4, dtype=np.float32)
        return model, view, proj

    # ------------------------------------------------------------------
    # Mesh upload helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_positions(gltf: "GLTF2", attrs) -> "np.ndarray | None":
        """Read the POSITION accessor from a mesh primitive's attributes.

        Args:
            gltf: The loaded glTF object.
            attrs: The ``primitive.attributes`` object from pygltflib.

        Returns:
            Float32 array of shape ``(n_verts, 3)``, or ``None`` on failure.
        """
        pos_idx = getattr(attrs, "POSITION", None)
        if pos_idx is None:
            return None
        try:
            arr = read_accessor(gltf, pos_idx).astype(np.float32)
        except Exception:
            traceback.print_exc()
            return None
        if arr.size == 0:
            return None
        if arr.ndim == 1:
            if arr.size % 3 != 0:
                return None
            arr = arr.reshape(-1, 3)
        if arr.shape[1] != 3:
            return None
        return arr

    @staticmethod
    def _make_vao(
        vertex_data: "np.ndarray", indices: "np.ndarray"
    ) -> "tuple[int, int, int]":
        """Upload interleaved vertex + index data to a new VAO.

        Interleaved vertex layout (stride = 64 bytes = 16 floats):
            bytes  0–11:  position    (3 × float32)
            bytes 12–23:  normal      (3 × float32)
            bytes 24–39:  joints      (4 × float32, stored as float for GLSL)
            bytes 40–55:  weights     (4 × float32)
            bytes 56–63:  texcoord    (2 × float32)

        Interleaving all attributes in one VBO is a GPU-cache-friendly layout
        because every attribute for a given vertex lives in one contiguous
        64-byte block — a single cache line on most GPUs.

        Args:
            vertex_data: Flat float32 array of shape ``(n_verts * 16,)``.
            indices:     Uint32 index array for indexed drawing.

        Returns:
            Tuple ``(vao, vbo, ebo)`` of OpenGL object IDs.
        """
        stride = 16 * 4   # 16 floats × 4 bytes = 64 bytes per vertex
        vao = int(GL.glGenVertexArrays(1))
        vbo = int(GL.glGenBuffers(1))
        ebo = int(GL.glGenBuffers(1))

        GL.glBindVertexArray(vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
        # GL_DYNAMIC_DRAW: data may be updated each frame (skinning can
        # update the VBO; we also re-use the VBO for skeleton drawing).
        GL.glBufferData(GL.GL_ARRAY_BUFFER, vertex_data.nbytes,
                        vertex_data, GL.GL_DYNAMIC_DRAW)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, ebo)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes,
                        indices, GL.GL_STATIC_DRAW)

        # Tell the VAO how to interpret each attribute within the interleaved
        # buffer.  Arguments: (location, components, type, normalise, stride, offset).
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, stride,
                                 GL.ctypes.c_void_p(0))    # position at byte 0
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, stride,
                                 GL.ctypes.c_void_p(12))   # normal at byte 12
        GL.glEnableVertexAttribArray(2)
        GL.glVertexAttribPointer(2, 4, GL.GL_FLOAT, GL.GL_FALSE, stride,
                                 GL.ctypes.c_void_p(24))   # joints at byte 24
        GL.glEnableVertexAttribArray(3)
        GL.glVertexAttribPointer(3, 4, GL.GL_FLOAT, GL.GL_FALSE, stride,
                                 GL.ctypes.c_void_p(40))   # weights at byte 40
        GL.glEnableVertexAttribArray(4)
        GL.glVertexAttribPointer(4, 2, GL.GL_FLOAT, GL.GL_FALSE, stride,
                                 GL.ctypes.c_void_p(56))   # texcoord at byte 56
        GL.glBindVertexArray(0)
        return vao, vbo, ebo

    def _upload_mesh(self) -> None:
        """Upload all mesh primitives that have POSITION data to the GPU.

        Why we iterate ALL primitives:
            A glTF model often splits its geometry across multiple mesh
            primitives, each with its own material.  For example,
            Barbarian.glb has 13 meshes (body, head, two arms, two legs,
            plus several prop meshes) and 6 skin-referenced nodes.  An
            implementation that stops at the first primitive with POSITION
            data would draw only one part — typically whichever mesh comes
            first in the file.

            This method iterates every primitive of every mesh, allocates a
            separate VAO/VBO/EBO per part into ``self._drawables``, and
            computes the unified bounding box across all parts for correct
            camera fit.

        Skinned vs. unskinned drawables:
            If a primitive's attributes include JOINTS_0 and WEIGHTS_0 AND
            its owner node references a skin, the drawable is *skinned*: the
            vertex shader blends the joint matrix palette.

            If not (a prop mesh — weapon, hat, cape, etc.), the drawable is
            *unskinned*: at draw time we compute the full animated world
            matrix of the owner node and pass it as ``uModel`` so the prop
            follows its parent bone exactly as a game engine would.
        """
        gltf = self._gltf
        if not gltf or not gltf.meshes:
            return

        # Free previously allocated GPU objects before uploading new ones.
        for d in self._drawables:
            try:
                GL.glDeleteVertexArrays(1, [d["vao"]])
                GL.glDeleteBuffers(2, [d["vbo"], d["ebo"]])
                if d["tex_id"]:
                    GL.glDeleteTextures(1, [d["tex_id"]])
            except Exception:
                pass
        self._drawables = []

        # Build lookup tables: mesh_index → skin_index and mesh_index → node_index.
        # We need these to know which skin (if any) each mesh primitive belongs to,
        # and which node owns the mesh (for prop world-transform computation).
        node_mesh_skin: dict[int, int] = {}   # mesh_index → skin_index
        mesh_to_node:   dict[int, int] = {}   # mesh_index → node_index
        for ni, node in enumerate(gltf.nodes or []):
            if node.mesh is not None:
                mesh_to_node[node.mesh] = ni
                if node.skin is not None:
                    node_mesh_skin[node.mesh] = node.skin

        # Pre-build the node parent map for prop world-transform evaluation.
        _build_node_parent_map(gltf)

        def _pad4(a: np.ndarray) -> np.ndarray:
            """Pad an array to 4 columns, filling missing columns with zero."""
            if a.ndim < 2 or a.shape[1] < 4:
                p = np.zeros((len(a), 4), dtype=np.float32)
                if a.ndim > 1:
                    p[:, :a.shape[1]] = a
                return p
            return a[:, :4]

        # Accumulate all vertex positions across all primitives for unified BBox.
        all_positions: list[np.ndarray] = []

        for mesh_idx, mesh in enumerate(gltf.meshes):
            for prim in (mesh.primitives or []):
                attrs = prim.attributes

                positions = self._read_positions(gltf, attrs)
                if positions is None:
                    continue
                n_verts = len(positions)
                all_positions.append(positions)

                # ── Normals ───────────────────────────────────────────────────
                # If the primitive has no normals, fill with (0, 0, 1) so the
                # Phong shader still works (the surface will appear flat-lit).
                norm_idx = getattr(attrs, "NORMAL", None)
                if norm_idx is not None:
                    try:
                        normals = read_accessor(gltf, norm_idx).astype(np.float32)
                        if normals.ndim == 1:
                            normals = normals.reshape(-1, 3)
                        if normals.shape != (n_verts, 3):
                            raise ValueError
                    except Exception:
                        normals = np.zeros((n_verts, 3), dtype=np.float32)
                        normals[:, 2] = 1.0
                else:
                    normals = np.zeros((n_verts, 3), dtype=np.float32)
                    normals[:, 2] = 1.0

                # ── Texture coordinates ───────────────────────────────────────
                tc_idx = getattr(attrs, "TEXCOORD_0", None)
                if tc_idx is not None:
                    try:
                        texcoords = read_accessor(gltf, tc_idx).astype(np.float32)
                        if texcoords.ndim == 1:
                            texcoords = texcoords.reshape(-1, 2)
                        if texcoords.shape[0] != n_verts or texcoords.shape[1] < 2:
                            raise ValueError
                        texcoords = texcoords[:, :2]
                    except Exception:
                        texcoords = np.zeros((n_verts, 2), dtype=np.float32)
                else:
                    texcoords = np.zeros((n_verts, 2), dtype=np.float32)

                # ── Joints / weights ──────────────────────────────────────────
                # JOINTS_0: up to 4 joint indices per vertex (stored as integers
                #   but we keep them as float32 for the shader).
                # WEIGHTS_0: corresponding blend weights — must sum to ≈ 1.
                j0_idx = getattr(attrs, "JOINTS_0",  None)
                w0_idx = getattr(attrs, "WEIGHTS_0", None)
                is_skinned = False
                if j0_idx is not None and w0_idx is not None:
                    try:
                        joints  = _pad4(read_accessor(gltf, j0_idx).astype(np.float32))
                        weights = _pad4(read_accessor(gltf, w0_idx).astype(np.float32))
                        if joints.shape[0] == n_verts and weights.shape[0] == n_verts:
                            is_skinned = True
                        else:
                            raise ValueError("shape mismatch")
                    except Exception:
                        traceback.print_exc()
                        joints  = np.zeros((n_verts, 4), dtype=np.float32)
                        weights = np.zeros((n_verts, 4), dtype=np.float32)
                        weights[:, 0] = 1.0   # bind fully to joint 0 as fallback
                else:
                    # No skinning data — vertex binds entirely to joint 0.
                    joints  = np.zeros((n_verts, 4), dtype=np.float32)
                    weights = np.zeros((n_verts, 4), dtype=np.float32)
                    weights[:, 0] = 1.0

                # ── Indices ───────────────────────────────────────────────────
                # Indexed drawing lets multiple triangles share vertices,
                # reducing data sent to the GPU.  If no index buffer exists,
                # generate a trivial sequential index array.
                if prim.indices is not None:
                    try:
                        indices = read_accessor(gltf, prim.indices).flatten().astype(np.uint32)
                    except Exception:
                        traceback.print_exc()
                        indices = np.arange(n_verts, dtype=np.uint32)
                else:
                    indices = np.arange(n_verts, dtype=np.uint32)

                # ── Material base colour and texture ──────────────────────────
                # glTF PBR materials store a base colour factor (RGBA, linear)
                # and an optional base colour texture.  We use both for the
                # Blinn-Phong shader; the PBR metallic-roughness model is out
                # of scope for this demo but the structure is the same.
                base_color = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
                has_tex    = False
                tex_id     = 0
                mat_idx    = prim.material
                if mat_idx is not None and gltf.materials and mat_idx < len(gltf.materials):
                    mat = gltf.materials[mat_idx]
                    pbr = getattr(mat, "pbrMetallicRoughness", None)
                    if pbr is not None:
                        bcf = getattr(pbr, "baseColorFactor", None)
                        if bcf is not None and len(bcf) >= 3:
                            base_color = np.array(
                                [bcf[0], bcf[1], bcf[2], bcf[3] if len(bcf) > 3 else 1.0],
                                dtype=np.float32,
                            )
                        tex_info = getattr(pbr, "baseColorTexture", None)
                        if tex_info is not None:
                            tex_src_idx = getattr(tex_info, "index", None)
                            if tex_src_idx is not None:
                                result = self._upload_texture(gltf, tex_src_idx)
                                if result:
                                    tex_id, has_tex = result

                # ── Interleave all attributes and upload ──────────────────────
                # Pack position, normal, joints, weights, and texcoords into
                # a single array.  Shape is (n_verts, 16) which we flatten
                # before passing to glBufferData.
                interleaved           = np.zeros((n_verts, 16), dtype=np.float32)
                interleaved[:, 0:3]   = positions
                interleaved[:, 3:6]   = normals
                interleaved[:, 6:10]  = joints
                interleaved[:, 10:14] = weights
                interleaved[:, 14:16] = texcoords
                vao, vbo, ebo = self._make_vao(interleaved.flatten(), indices)

                skin_index = node_mesh_skin.get(mesh_idx, -1)
                node_index = mesh_to_node.get(mesh_idx, -1)

                self._drawables.append({
                    "vao":         vao,
                    "vbo":         vbo,
                    "ebo":         ebo,
                    "index_count": len(indices),
                    "skin_index":  skin_index,
                    "node_index":  node_index,
                    "is_skinned":  is_skinned and skin_index >= 0,
                    "has_tex":     has_tex,
                    "tex_id":      tex_id,
                    "base_color":  base_color,
                })
                print(f"[Viewport] mesh[{mesh_idx}] '{mesh.name}': "
                      f"{n_verts} verts, {len(indices)} idx, "
                      f"skinned={is_skinned}, skin={skin_index}, tex={has_tex}")

        if not self._drawables:
            print("[Viewport] No usable mesh primitives found in this GLB.")
            return

        # ── Unified bounding box across ALL parts ─────────────────────────────
        # Concatenate all vertex positions and find the global min/max.
        # This gives a bounding box that encompasses the entire model,
        # which we use to position the camera so the whole model is in view.
        all_pts = np.concatenate(all_positions, axis=0)
        mn, mx  = all_pts.min(axis=0), all_pts.max(axis=0)
        self._bbox_center = ((mn + mx) / 2.0).astype(np.float32)
        extent            = float(np.max(mx - mn)) or 1.0
        self._bbox_scale  = 1.0 / extent
        # Place the camera at 1.4× the longest extent — close enough to see
        # detail, far enough to see the whole model.
        self._orbit_dist  = extent * 1.4

        # Save the initial camera state so double-click can restore it.
        self._bbox_center_default  = self._bbox_center.copy()
        self._orbit_dist_default   = self._orbit_dist
        self._orbit_yaw_default    = 30.0
        self._orbit_pitch_default  = 15.0
        print(f"[Viewport] Total drawables: {len(self._drawables)}, "
              f"bbox extent={extent:.3f}")

    def _upload_texture(self, gltf: "GLTF2",
                        tex_index: int) -> "tuple[int, bool] | None":
        """Decode and upload the glTF texture at ``tex_index`` to the GPU.

        glTF stores images in one of three ways:
            1. ``bufferView`` — bytes embedded in the GLB binary chunk (most common).
            2. ``data:`` URI  — base-64 encoded inline image in the JSON.
            3. External URI   — a separate file on disk (uncommon in .glb files).

        The texture is uploaded as ``GL_SRGB8_ALPHA8``, which tells OpenGL to
        automatically linearise sRGB-encoded texel values when sampled by the
        fragment shader.  This keeps all Blinn-Phong lighting in linear space
        without any manual conversion in the shader.

        Args:
            gltf: The loaded glTF object.
            tex_index: Index into ``gltf.textures``.

        Returns:
            ``(tex_id, True)`` on success, or ``None`` on failure.
        """
        import io, base64 as _b64
        try:
            from PIL import Image as PILImage
        except ImportError:
            print("[Viewport] Pillow not installed — textures disabled. "
                  "pip install pillow")
            return None
        try:
            if not gltf.textures or tex_index >= len(gltf.textures):
                return None
            tex   = gltf.textures[tex_index]
            img_i = tex.source
            if img_i is None:
                return None
            if not gltf.images or img_i >= len(gltf.images):
                return None
            img_def = gltf.images[img_i]

            raw_bytes: bytes | None = None

            if img_def.bufferView is not None:
                # Case 1: image embedded in the GLB binary chunk.
                if gltf.bufferViews and img_def.bufferView < len(gltf.bufferViews):
                    bv        = gltf.bufferViews[img_def.bufferView]
                    blob      = _get_blob(gltf)
                    start     = bv.byteOffset or 0
                    raw_bytes = bytes(blob[start: start + bv.byteLength])
            elif img_def.uri:
                if img_def.uri.startswith("data:"):
                    # Case 2: data URI — "data:image/png;base64,iVBOR…"
                    try:
                        _, b64 = img_def.uri.split(",", 1)
                        raw_bytes = _b64.b64decode(b64)
                    except Exception:
                        traceback.print_exc()
                        return None
                else:
                    # Case 3: external file — only attempt if path is known.
                    if self._gltf_path_hint is not None:
                        candidate = self._gltf_path_hint.parent / img_def.uri
                        if candidate.exists():
                            raw_bytes = candidate.read_bytes()
                        else:
                            print(f"[Viewport] External texture '{img_def.uri}' "
                                  "not found next to the .glb — skipped.")
                            return None
                    else:
                        return None

            if not raw_bytes:
                return None

            pil      = PILImage.open(io.BytesIO(raw_bytes)).convert("RGBA")
            img_data = np.array(pil, dtype=np.uint8)
            h, w     = img_data.shape[:2]

            tex_id = int(GL.glGenTextures(1))
            GL.glBindTexture(GL.GL_TEXTURE_2D, tex_id)
            # GL_SRGB8_ALPHA8: GPU linearises sRGB-encoded texels before
            # the fragment shader reads them, so lighting stays in linear space.
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0,
                            GL.GL_SRGB8_ALPHA8,
                            w, h, 0,
                            GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, img_data)
            # Mipmaps improve appearance when the texture is viewed at a small
            # size on screen — each mip level is a half-resolution version.
            GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER,
                               GL.GL_LINEAR_MIPMAP_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER,
                               GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S,
                               GL.GL_REPEAT)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T,
                               GL.GL_REPEAT)
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
            print(f"[Viewport] Texture {tex_index} uploaded ({w}×{h} px, sRGB)")
            return tex_id, True
        except Exception:
            traceback.print_exc()
            return None

    # Maximum number of bone line segments the skeleton VBO can hold.
    # 2048 segments × 2 endpoints × 3 floats × 4 bytes = 192 KB.
    _SKEL_VBO_SEGMENTS: int = 2048

    def _upload_skeleton_vbo(self) -> None:
        """Allocate the skeleton line VBO with enough space for all bone segments.

        The VBO is filled dynamically each frame in ``_draw_skeleton()``
        using ``GL_DYNAMIC_DRAW`` so the driver keeps the data in fast
        write-combined memory.
        """
        GL.glBindVertexArray(self._line_vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._line_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER,
                        self._SKEL_VBO_SEGMENTS * 2 * 3 * 4,
                        None, GL.GL_DYNAMIC_DRAW)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 12,
                                 GL.ctypes.c_void_p(0))
        GL.glBindVertexArray(0)

    def _draw_mesh(self) -> None:
        """Draw all loaded mesh primitives for the current frame.

        Shared uniforms (view, projection, light) are set once before the
        per-drawable loop so the driver does not re-upload them redundantly.

        Skinned drawables:
            Use the identity model matrix — the joint matrices already move
            vertices into world space.  Joint matrices are cached by skin
            index so parts that share a skeleton (e.g. body + head on the
            same skin) compute the expensive skinning only once per frame.

        Unskinned drawables (props):
            Compute the full animated world matrix of the owner node by
            walking the scene graph, and pass it as ``uModel``.  This is
            cached by node index within the frame.
        """
        if self._prog is None or not self._drawables:
            return
        model, view, proj = self._mvp()
        self._prog.bind()
        pid = self._prog.programId()

        # Helper to look up uniform locations by name.
        def _u(name: str) -> int:
            return GL.glGetUniformLocation(pid, name)

        # ── Uniforms shared by every drawable ─────────────────────────────────
        GL.glUniformMatrix4fv(_u("uView"),  1, GL.GL_TRUE, view)
        GL.glUniformMatrix4fv(_u("uProj"),  1, GL.GL_TRUE, proj)
        # Place the key light slightly above and to the right of the camera
        # so the model is well-lit from most viewing angles.
        light = self._bbox_center + self._orbit_dist * np.array(
            [0.5, 0.8, 0.6], dtype=np.float32)
        GL.glUniform3f(_u("uLightPos"),   *light)
        GL.glUniform3f(_u("uLightColor"), 1.0, 1.0, 1.0)
        GL.glUniform1f(_u("uAmbient"),    0.15)

        # Per-frame caches to avoid recomputing shared data per drawable.
        jmat_cache:    dict[int, np.ndarray] = {}   # skin_index → joint matrices
        prop_mat_cache: dict[int, np.ndarray] = {}  # node_index → world matrix
        parent_map = (_build_node_parent_map(self._gltf)
                      if self._gltf is not None else {})

        # ── Per-drawable draw loop ─────────────────────────────────────────────
        for d in self._drawables:
            bc = d["base_color"]
            GL.glUniform4f(_u("uBaseColor"),
                           float(bc[0]), float(bc[1]), float(bc[2]), float(bc[3]))

            # Bind texture if this drawable has one.
            if d["has_tex"]:
                GL.glActiveTexture(GL.GL_TEXTURE0)
                GL.glBindTexture(GL.GL_TEXTURE_2D, d["tex_id"])
                GL.glUniform1i(_u("uAlbedo"),     0)
                GL.glUniform1i(_u("uHasTexture"), 1)
            else:
                GL.glUniform1i(_u("uHasTexture"), 0)

            skin_idx   = d["skin_index"]
            node_idx   = d["node_index"]
            is_skinned = d["is_skinned"]
            GL.glUniform1i(_u("uSkinned"), 1 if is_skinned else 0)

            if is_skinned and self._gltf is not None:
                # Skinned mesh: identity model matrix — vertices are already
                # in world space after the joint matrices are applied in the
                # vertex shader.
                GL.glUniformMatrix4fv(_u("uModel"), 1, GL.GL_TRUE, model)

                # Compute joint matrices for this skin, using the cache to
                # avoid recomputing for parts that share the same skeleton.
                if skin_idx not in jmat_cache:
                    try:
                        jmats = compute_joint_matrices(
                            self._gltf, skin_idx,
                            self._anim_index, self._playback_t,
                        )
                        jmat_cache[skin_idx] = jmats
                    except Exception:
                        traceback.print_exc()
                        jmat_cache[skin_idx] = np.eye(4, dtype=np.float32)[np.newaxis]
                jmats = jmat_cache[skin_idx]
                n    = min(len(jmats), 64)   # shader declares uJoints[64]
                flat = jmats[:n].flatten().astype(np.float32)
                loc  = GL.glGetUniformLocation(pid, "uJoints[0]")
                GL.glUniformMatrix4fv(loc, n, GL.GL_TRUE, flat)
            else:
                # Unskinned prop: compute the animated world matrix for this
                # node by walking up the scene graph parent chain, composing
                # animated TRS matrices along the way.
                if node_idx >= 0 and self._gltf is not None:
                    if node_idx not in prop_mat_cache:
                        try:
                            prop_world = _get_node_world_mat(
                                self._gltf, node_idx,
                                self._anim_index, self._playback_t,
                                parent_map,
                            )
                        except Exception:
                            traceback.print_exc()
                            prop_world = model   # fallback: identity
                        prop_mat_cache[node_idx] = prop_world
                    GL.glUniformMatrix4fv(
                        _u("uModel"), 1, GL.GL_TRUE, prop_mat_cache[node_idx]
                    )
                else:
                    GL.glUniformMatrix4fv(_u("uModel"), 1, GL.GL_TRUE, model)

            # Issue the actual draw call: render indexed triangles.
            GL.glBindVertexArray(d["vao"])
            GL.glDrawElements(GL.GL_TRIANGLES, d["index_count"],
                              GL.GL_UNSIGNED_INT, None)
            GL.glBindVertexArray(0)

            if d["has_tex"]:
                GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

        self._prog.release()

    def _draw_skeleton(self) -> None:
        """Draw parent→child bone lines and joint dots in the skeleton overlay.

        Line width constraint in Core Profile:
            OpenGL 3.3 Core Profile only permits ``glLineWidth(1.0)`` — any
            other value raises ``GL_INVALID_VALUE``.  We compensate with a
            bright blue colour and high-contrast background.

        Joint positions are stored in ``self._joint_world_pos`` after each
        skeleton draw so the bone-picking code in ``_try_pick_bone()`` can
        project them to screen space without rerunning the animation.
        """
        if self._line_prog is None or self._gltf is None:
            return
        gltf = self._gltf
        try:
            world_pos = compute_world_positions(gltf, self._anim_index, self._playback_t)
        except Exception:
            return

        # Cache world positions for the bone-picking hit test.
        self._joint_world_pos = world_pos

        # Build a flat list of line segment endpoints: for each parent node,
        # emit two positions (parent → child) for each child it has.
        segments: list[float] = []
        for ni, node in enumerate(gltf.nodes or []):
            p_pos = world_pos.get(ni)
            if p_pos is None:
                continue
            for ci in (node.children or []):
                c_pos = world_pos.get(ci)
                if c_pos is None:
                    continue
                segments.extend(p_pos.tolist())
                segments.extend(c_pos.tolist())

        if not segments:
            return

        arr       = np.array(segments, dtype=np.float32)
        max_bytes = self._SKEL_VBO_SEGMENTS * 2 * 3 * 4

        # Upload line data to the shared skeleton VBO.
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._line_vbo)
        upload = arr[:max_bytes // 4]
        GL.glBufferSubData(GL.GL_ARRAY_BUFFER, 0, upload.nbytes, upload)

        # Build the combined MVP matrix for the skeleton (no per-vertex transform).
        model, view, proj = self._mvp()
        mvp = proj @ view @ model

        self._line_prog.bind()
        loc_mvp   = GL.glGetUniformLocation(self._line_prog.programId(), "uMVP")
        loc_color = GL.glGetUniformLocation(self._line_prog.programId(), "uColor")
        GL.glUniformMatrix4fv(loc_mvp, 1, GL.GL_TRUE, mvp.flatten())

        # Width 1.0 is the only legal line width in OpenGL 3.3 Core Profile.
        GL.glLineWidth(1.0)

        # Draw bone lines in accent blue.
        GL.glUniform3f(loc_color, 0.310, 0.780, 0.976)
        GL.glBindVertexArray(self._line_vao)
        n_verts = min(len(segments) // 3, self._SKEL_VBO_SEGMENTS * 2)
        GL.glDrawArrays(GL.GL_LINES, 0, n_verts)

        # Draw joint dots: regular joints white, the selected joint orange.
        joint_pts:   list[float] = []
        selected_pt: list[float] = []
        for ni, pos in world_pos.items():
            if ni == self._selected_joint:
                selected_pt.extend(pos.tolist())
            else:
                joint_pts.extend(pos.tolist())

        if joint_pts:
            pts_arr = np.array(joint_pts, dtype=np.float32)
            cap     = min(pts_arr.nbytes, max_bytes)
            GL.glBufferSubData(GL.GL_ARRAY_BUFFER, 0, cap, pts_arr.flatten()[:cap // 4])
            GL.glUniform3f(loc_color, 0.9, 0.9, 0.9)
            GL.glPointSize(5.0)
            GL.glDrawArrays(GL.GL_POINTS, 0, min(len(joint_pts) // 3,
                                                   self._SKEL_VBO_SEGMENTS * 2))

        if selected_pt:
            sel_arr = np.array(selected_pt[:3], dtype=np.float32)
            GL.glBufferSubData(GL.GL_ARRAY_BUFFER, 0, sel_arr.nbytes, sel_arr)
            GL.glUniform3f(loc_color, 1.0, 0.576, 0.235)   # orange
            GL.glPointSize(10.0)
            GL.glDrawArrays(GL.GL_POINTS, 0, 1)

        GL.glPointSize(1.0)
        GL.glBindVertexArray(0)
        self._line_prog.release()

    def _try_pick_bone(self, screen_pos) -> None:
        """Select the nearest joint to the given screen position.

        How bone picking works:
            For each joint we know its world-space position from the previous
            skeleton draw.  We project it through the MVP matrix into NDC
            (Normalised Device Coordinates, the [-1,1]^3 cube), then map to
            pixel coordinates.  Whichever joint's projected pixel position is
            within the pick radius of the click position wins.

        Args:
            screen_pos: QPointF click position in viewport pixel coordinates.
        """
        if not self._joint_world_pos or self._gltf is None:
            return
        model, view, proj = self._mvp()
        mvp = proj @ view @ model
        w, h = max(1, self.width()), max(1, self.height())
        sx, sy = float(screen_pos.x()), float(screen_pos.y())

        best_ni   = -1
        best_dist = 25.0   # maximum pixel radius for a valid pick

        for ni, wpos in self._joint_world_pos.items():
            # Project world position to clip space: p4 = MVP * [x, y, z, 1].
            p4 = mvp @ np.array([*wpos, 1.0], dtype=np.float32)
            if p4[3] <= 0:
                continue   # behind the camera — skip
            # Divide by W to get NDC, then map to pixel coordinates.
            ndc_x =  p4[0] / p4[3]
            ndc_y = -p4[1] / p4[3]   # flip Y: NDC +Y = screen top
            px = (ndc_x * 0.5 + 0.5) * w
            py = (ndc_y * 0.5 + 0.5) * h
            d  = math.hypot(px - sx, py - sy)
            if d < best_dist:
                best_dist = d
                best_ni   = ni

        if best_ni >= 0:
            self._selected_joint = best_ni
            self.bone_selected.emit(best_ni)
            self.update()

    def _draw_empty_hint(self) -> None:
        """Draw a centred instruction text when no model is loaded."""
        from PySide6.QtGui import QPainter, QFont as _QFont, QColor as _QColor
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(),
                         _QColor(int(PALETTE["bg"][1:3], 16),
                                 int(PALETTE["bg"][3:5], 16),
                                 int(PALETTE["bg"][5:7], 16)))
        font = _QFont("Segoe UI", 14)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(_QColor(PALETTE["muted"]))
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                         "3-D Viewport\n\nOpen a .glb file to see the model here\n"
                         "(Ctrl+O or the blue button)")


# =============================================================================
# STEP 7 — Help / instructions panel
# =============================================================================

class HelpPanel(QWidget):
    """Collapsible instruction panel displayed above the 3-D viewport.

    Summarises the four-step workflow and viewport controls in a compact
    HTML table.  Toggled by the ``? Help`` toolbar button or the ``?`` key.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("help_panel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 10, 16, 10)
        layout.setSpacing(0)

        html = f"""
        <table cellspacing="0" cellpadding="0" width="100%">
          <tr><td colspan="2" style="padding-bottom:8px;">
            <span style="color:{PALETTE['accent']};font-size:13px;font-weight:bold;">
              How to use this demo
            </span>
          </td></tr>
          <tr>
            <td style="color:{PALETTE['text']};font-size:12px;font-weight:bold;
                       padding:3px 12px 1px 0;white-space:nowrap;vertical-align:top;">
              1 &nbsp;Open a model
            </td>
            <td style="color:{PALETTE['muted']};font-size:11px;padding:3px 0 1px 0;">
              Ctrl+O or the blue button — load any .glb file that contains animations.
              The first animated bone is selected automatically.
            </td>
          </tr>
          <tr>
            <td style="color:{PALETTE['text']};font-size:12px;font-weight:bold;
                       padding:3px 12px 1px 0;white-space:nowrap;vertical-align:top;">
              2 &nbsp;Browse the scene
            </td>
            <td style="color:{PALETTE['muted']};font-size:11px;padding:3px 0 1px 0;">
              Click any bone in the Scene Tree (left). Its animation channels appear
              below, and the first channel is selected automatically.
            </td>
          </tr>
          <tr>
            <td style="color:{PALETTE['text']};font-size:12px;font-weight:bold;
                       padding:3px 12px 1px 0;white-space:nowrap;vertical-align:top;">
              3 &nbsp;Inspect curves
            </td>
            <td style="color:{PALETTE['muted']};font-size:11px;padding:3px 0 1px 0;">
              Click a channel (translation / rotation / scale) to draw its F-curves.
              X=red &nbsp;Y=green &nbsp;Z=blue &nbsp;W=yellow. Toggle with checkboxes.
            </td>
          </tr>
          <tr>
            <td style="color:{PALETTE['text']};font-size:12px;font-weight:bold;
                       padding:3px 12px 1px 0;white-space:nowrap;vertical-align:top;">
              4 &nbsp;Play &amp; edit
            </td>
            <td style="color:{PALETTE['muted']};font-size:11px;padding:3px 0 1px 0;">
              Space or &#9654; to play — model animates in the viewport above.<br>
              <b>Click a coloured dot</b> on any curve to select it, then
              <b>drag up/down</b> to edit the value. Ctrl+S exports the modified .glb.
            </td>
          </tr>
          <tr>
            <td style="color:{PALETTE['text']};font-size:12px;font-weight:bold;
                       padding:3px 12px 1px 0;white-space:nowrap;vertical-align:top;">
              Viewport
            </td>
            <td style="color:{PALETTE['muted']};font-size:11px;padding:3px 0 1px 0;">
              <b>Left-drag</b> to orbit &middot; <b>Middle-drag</b> to pan &middot; <b>Scroll</b> to zoom.<br>
              <b>Double-click</b> to reframe the camera to the model's bounding box.<br>
              <b>Click a joint dot</b> on the skeleton to select that bone in the Scene Tree.<br>
              <b>🦴 Skeleton</b> button (or <b>K</b>) toggles the skeleton overlay on/off.
            </td>
          </tr>
          <tr><td colspan="2" style="padding-top:8px;">
            <span style="color:{PALETTE['muted']};font-size:10px;font-style:italic;">
              Press ? or click &#10005; Help to hide this panel.
            </span>
          </td></tr>
        </table>
        """
        lbl = QLabel(html)
        lbl.setWordWrap(True)
        lbl.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(lbl)


# =============================================================================
# STEP 8 — Curve editor widget (PyQtGraph)
# =============================================================================

class CurveEditorWidget(QWidget):
    """PyQtGraph-backed animation F-curve editor with drag-to-edit keyframes.

    Animation Concept — the Graph Editor:
        Maya's Graph Editor and Blender's F-Curve Editor both show time on the
        X axis and animated value on the Y axis, with one coloured curve per
        channel component (X=red, Y=green, Z=blue, W=yellow).  The *shape* of
        the curve — ease-in, ease-out, overshoot — directly controls the
        *feel* of the animation.  This editor reproduces that core interaction
        in Python using PyQtGraph for the 2-D plot.

    Editing interaction (matches Maya Graph Editor / Blender F-Curve Editor):
        - Click a coloured dot  → select that keyframe (turns white).
        - Drag selected dot up/down → value changes live; the 3-D viewport
          updates on every mouse-move tick so you see the pose change in
          real time before committing.
        - Release the mouse → value is committed to the glTF binary blob.
        - Press Escape while dragging → revert to the value at drag-start.

    Why an application-level event filter for Escape:
        Qt routes key events to whichever widget has keyboard focus.  After
        the user interacts with the PyQtGraph plot, focus belongs to the
        ViewBox or its internal widgets — not to our ``CurveEditorWidget``.
        A viewport-level ``KeyPress`` filter never receives the event because
        PyQtGraph's own key handler consumes it first.

        Installing the filter on ``QApplication.instance()`` instead puts it
        *above* all widget routing in the process.  Every key event passes
        through it before any widget sees it, regardless of which widget
        currently has focus.  We guard the handler tightly so it only acts
        when a drag is active, to avoid interfering with normal key input
        anywhere else in the application.

    Attributes:
        _drag_active: True while a drag is in progress.  Decoupled from
            ``_drag_start_y`` so Escape works even if the mouse was
            momentarily released between press and key press.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        pg.setConfigOptions(antialias=True, background=None)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Header bar ────────────────────────────────────────────────────────
        header = QWidget()
        header.setStyleSheet(f"background-color: {PALETTE['surface']};")
        hl = QHBoxLayout(header)
        hl.setContentsMargins(12, 6, 12, 6)
        self._channel_label = QLabel("No channel selected — click a bone, then a channel")
        self._channel_label.setStyleSheet(f"color: {PALETTE['muted']}; font-size: 12px;")
        hl.addWidget(self._channel_label)
        hl.addStretch()

        self._edit_tip = QLabel("Click a dot to select  ·  Drag up/down to edit  ·  Scroll to zoom")
        self._edit_tip.setStyleSheet(
            f"color: {PALETTE['muted']}; font-size: 10px; font-style: italic; padding-right: 8px;"
        )
        hl.addWidget(self._edit_tip)

        # Component toggles (X/Y/Z/W checkboxes) let the user hide individual
        # curves.  Each checkbox is coloured to match its curve.
        self._toggles: list[QCheckBox] = []
        for i, (lbl, col) in enumerate(zip(["X", "Y", "Z", "W"], CURVE_COLORS)):
            cb = QCheckBox(lbl)
            cb.setChecked(True)
            r, g, b, _ = col
            cb.setStyleSheet(
                f"QCheckBox {{ color: rgb({r},{g},{b}); font-size: 12px; }}"
                f"QCheckBox::indicator:checked {{ background-color: rgb({r},{g},{b}); "
                f"border-color: rgb({r},{g},{b}); }}"
            )
            cb.stateChanged.connect(lambda _, idx=i: self._toggle_curve(idx))
            self._toggles.append(cb)
            hl.addWidget(cb)
        layout.addWidget(header)

        div = QFrame()
        div.setFrameShape(QFrame.Shape.HLine)
        div.setStyleSheet(f"background-color: {PALETTE['border']}; max-height: 1px;")
        layout.addWidget(div)

        # ── PyQtGraph plot ────────────────────────────────────────────────────
        self._plot = pg.PlotWidget()
        self._plot.setMenuEnabled(False)
        self._plot.showGrid(x=True, y=True, alpha=0.15)
        self._plot.setLabel("bottom", "Time", units="s",  color=PALETTE["muted"], size="10pt")
        self._plot.setLabel("left",   "Value",             color=PALETTE["muted"], size="10pt")
        self._plot.getAxis("bottom").setPen(pg.mkPen(color=PALETTE["border"]))
        self._plot.getAxis("left").setPen(pg.mkPen(color=PALETTE["border"]))
        self._plot.getAxis("bottom").setTextPen(pg.mkPen(color=PALETTE["muted"]))
        self._plot.getAxis("left").setTextPen(pg.mkPen(color=PALETTE["muted"]))
        self._plot.setBackground(PALETTE["surface"])

        # Install a mouse event filter on the PyQtGraph viewport widget so we
        # can intercept press / move / release without fighting PyQtGraph's
        # built-in pan and zoom handlers (which also listen on the viewport).
        # We only act when the click is near the selected keyframe dot.
        self._plot.viewport().installEventFilter(self)

        # Install a second, application-level event filter to catch Escape.
        # This filter sits above all widget routing so it fires regardless of
        # which widget has keyboard focus.  We guard it with _drag_active so
        # it only consumes the key event while a drag is genuinely in progress.
        QApplication.instance().installEventFilter(self)
        layout.addWidget(self._plot, stretch=1)

        # ── Internal state ────────────────────────────────────────────────────
        self._curves:        list[pg.PlotDataItem]       = []
        self._scatter_items: list[pg.ScatterPlotItem]    = []
        self._playhead:      pg.InfiniteLine | None      = None
        self._times:         np.ndarray | None           = None
        self._values:        np.ndarray | None           = None
        self._anim_index:    int                         = -1
        self._channel_index: int                         = -1
        self._gltf_ref:      GLTF2 | None               = None
        self._on_modified_cb: callable | None            = None
        self._on_live_cb:     callable | None            = None
        self._current_target_path_labels: list[str]     = ["X", "Y", "Z", "W"]

        # Drag state — all reset when a drag completes or is reverted.
        self._drag_comp:          int        = -1     # component index being dragged
        self._drag_key:           int        = -1     # keyframe index being dragged
        self._drag_orig:          float      = 0.0    # value at drag-start (for Escape revert)
        self._drag_start_y:       float | None = None # screen-Y pixel at mouse-down
        self._drag_start_scene_y: float      = 0.0    # plot-space Y at mouse-down
        # _drag_active is set True on mouse-down and False on release or revert.
        # It is separate from _drag_start_y so Escape works even if the mouse
        # was momentarily released between press and pressing Escape.
        self._drag_active: bool = False

        # Selection highlight — a white dot drawn over the selected keyframe.
        self._sel_scatter: pg.ScatterPlotItem | None = None
        self._sel_comp:    int                       = -1
        self._sel_key:     int                       = -1

    def closeEvent(self, event) -> None:
        """Uninstall the application-level event filter on widget destruction.

        Qt does not automatically uninstall event filters installed on objects
        with a different lifecycle (here: QApplication vs. this widget).
        Failing to remove the filter would leave a dangling reference and
        trigger a crash or error after the widget is destroyed.

        Args:
            event: QCloseEvent from Qt.
        """
        try:
            QApplication.instance().removeEventFilter(self)
        except Exception:
            pass
        super().closeEvent(event)

    # ── Public API ────────────────────────────────────────────────────────────

    def set_modified_callback(self, cb: callable) -> None:
        """Register a callback fired on mouse-release (keyframe commit).

        Args:
            cb: Callable with signature ``(anim_index, channel_index, values)``.
        """
        self._on_modified_cb = cb

    def set_live_callback(self, cb: callable) -> None:
        """Register a callback fired on every mouse-move during a drag.

        Use this to drive the 3-D viewport live while the user is editing,
        matching Maya's behaviour where the pose updates in real time.

        Args:
            cb: Callable with signature ``(anim_index, channel_index, values)``.
        """
        self._on_live_cb = cb

    def load_channel(
        self,
        gltf: GLTF2,
        anim_index: int,
        channel_index: int,
        node_name: str,
        target_path: str,
        anim_name: str,
    ) -> None:
        """Display the F-curves for one animation channel.

        Args:
            gltf: The loaded glTF object.
            anim_index: Index into ``gltf.animations``.
            channel_index: Index into the animation's channels.
            node_name: Display name of the target node (for the header label).
            target_path: glTF path: ``"translation"``, ``"rotation"``, etc.
            anim_name: Display name of the animation clip.
        """
        self._gltf_ref      = gltf
        self._anim_index    = anim_index
        self._channel_index = channel_index

        times, values = read_channel_data(gltf, anim_index, channel_index)
        # Ensure values is 2-D: (n_keys, n_components).
        if values.ndim == 1:
            values = values[:, np.newaxis]
        self._times  = times
        self._values = values

        labels = COMPONENT_LABELS.get(target_path, ["V0", "V1", "V2", "V3"])
        self._current_target_path_labels = labels

        duration = float(times[-1]) if len(times) > 0 else 0.0
        n_keys   = len(times)
        self._channel_label.setText(
            f"{node_name}  ·  {target_path}  ·  {anim_name}  ·  "
            f"{n_keys} keyframes  ·  {duration:.3f}s"
        )

        self._clear_plot()
        n_components = self._values.shape[1]
        for i in range(n_components):
            color = CURVE_COLORS[i % len(CURVE_COLORS)]
            pen   = pg.mkPen(color=color[:3], width=2)
            curve = self._plot.plot(
                times, self._values[:, i],
                pen=pen, name=labels[i] if i < len(labels) else f"V{i}"
            )
            self._curves.append(curve)

            # ScatterPlotItem draws the keyframe dots on top of the curve line.
            # Each dot stores (component_index, keyframe_index) in its data
            # payload so the click handler can identify which keyframe was hit.
            scatter = pg.ScatterPlotItem(
                x=times, y=self._values[:, i],
                size=10,
                pen=pg.mkPen(color=color[:3], width=1),
                brush=pg.mkBrush(color=(*color[:3], 200)),
                hoverable=True,
                hoverPen=pg.mkPen("w", width=2),
                hoverBrush=pg.mkBrush(color=(*color[:3], 255)),
            )
            scatter.setData(
                x=times, y=self._values[:, i],
                data=[(i, j) for j in range(len(times))]
            )
            # sigClicked handles selection only — editing is done via the
            # eventFilter drag, which gives sub-pixel precision.
            scatter.sigClicked.connect(self._on_scatter_clicked)
            self._plot.addItem(scatter)
            self._scatter_items.append(scatter)

            visible = self._toggles[i].isChecked() if i < len(self._toggles) else True
            curve.setVisible(visible)
            scatter.setVisible(visible)

        # Add the orange dashed playhead line at t=0.
        self._playhead = pg.InfiniteLine(
            pos=0.0, angle=90,
            pen=pg.mkPen(color=PALETTE["orange"], width=1,
                         style=Qt.PenStyle.DashLine),
            movable=False,
        )
        self._plot.addItem(self._playhead)

        # Apply autoRange() to fit all curves in view, then enforce a minimum
        # Y window height so near-constant channels (scale ≈ 1.0, quaternion
        # W ≈ 1.0) don't zoom into a hairline that makes hit-testing fail.
        self._plot.autoRange()
        self._apply_sane_y_range()

    # Minimum Y window height in data units.  Channels whose values barely
    # deviate from a constant auto-range to a window of a few thousandths,
    # making the hit-test inaccurate and drag sensitivity imperceptibly small.
    _MIN_Y_WINDOW: float = 0.5

    def _apply_sane_y_range(self) -> None:
        """Enforce a minimum Y window after ``autoRange()``.

        Why this matters:
            Channels with near-constant values (e.g. a scale channel that
            stays at exactly 1.0, or a quaternion W channel near 1.0) produce
            a Y data range of only a few thousandths of a unit.  At that
            zoom level:

            - One pixel of mouse drag corresponds to only ~0.0001 value change
              — edits feel completely frozen and unresponsive.
            - The coordinate round-trip in ``_is_near_selected_dot()``
              accumulates enough floating-point error to miss the hit radius,
              so clicks on the dot never start a drag at all.

            Widening the window to at least ``_MIN_Y_WINDOW`` centred on the
            data midpoint keeps edits responsive without hiding the curve shape.
        """
        if self._values is None or self._values.size == 0:
            return
        vb  = self._plot.getPlotItem().getViewBox()
        rng = vb.viewRange()     # [[xmin, xmax], [ymin, ymax]]
        y_lo, y_hi = rng[1]
        window = y_hi - y_lo
        if window < self._MIN_Y_WINDOW:
            mid  = (y_lo + y_hi) / 2.0
            half = self._MIN_Y_WINDOW / 2.0
            vb.setYRange(mid - half, mid + half, padding=0)

    def set_playhead(self, t: float) -> None:
        """Move the vertical playhead line to time ``t``.

        Args:
            t: Time in seconds.
        """
        if self._playhead is not None:
            self._playhead.setPos(t)

    def refresh_values(self) -> None:
        """Re-read ``_values`` into all curves and scatter items.

        Called by the main window after a table edit so the curve display
        and keyframe dots reflect the new values immediately.
        """
        if self._values is None or self._times is None:
            return
        for i, (curve, scatter) in enumerate(
                zip(self._curves, self._scatter_items)):
            if i < self._values.shape[1]:
                curve.setData(self._times, self._values[:, i])
                scatter.setData(
                    x=self._times, y=self._values[:, i],
                    data=[(i, j) for j in range(len(self._times))]
                )
        self._refresh_selection_dot()

    def _clear_plot(self) -> None:
        """Remove all plot items and reset the curve/scatter lists."""
        self._plot.clear()
        self._curves.clear()
        self._scatter_items.clear()
        self._playhead    = None
        self._sel_scatter = None

    def clear(self) -> None:
        """Clear the plot and reset the header label to the default text."""
        self._clear_plot()
        self._channel_label.setText("No channel selected — click a bone, then a channel")
        self._times  = None
        self._values = None

    def _toggle_curve(self, component_index: int) -> None:
        """Show or hide the curve and dots for one component.

        Args:
            component_index: 0=X, 1=Y, 2=Z, 3=W.
        """
        visible = self._toggles[component_index].isChecked()
        if component_index < len(self._curves):
            self._curves[component_index].setVisible(visible)
        if component_index < len(self._scatter_items):
            self._scatter_items[component_index].setVisible(visible)

    # ── Selection ─────────────────────────────────────────────────────────────

    def _on_scatter_clicked(self, *args) -> None:
        """Select the clicked keyframe dot.

        Clicking a dot selects it (the dot turns white) and shows its
        coordinates in the status tip.  Actual value editing happens via
        the drag mechanic in ``eventFilter()``, not here.

        PyQtGraph's ``sigClicked`` signature varies between versions:
        it may pass ``(scatter, points, event)`` or ``(scatter, points)``.
        Accepting ``*args`` handles both.
        """
        points = None
        for arg in args:
            if isinstance(arg, (list, np.ndarray)) and len(arg) > 0:
                points = arg
                break
        if (points is None
                or (hasattr(points, "__len__") and len(points) == 0)
                or self._values is None):
            return

        point = points[0]
        data  = point.data()
        if data is None:
            return
        comp, key = data
        self._sel_comp = comp
        self._sel_key  = key
        self._refresh_selection_dot()

        val = float(self._values[key, comp])
        t   = float(self._times[key])
        lbl = (self._current_target_path_labels[comp]
               if comp < len(self._current_target_path_labels) else f"V{comp}")
        self._edit_tip.setText(
            f"Selected  t={t:.4f}s  {lbl}={val:.5f}  ·  Drag UP/DOWN to edit  ·  Scroll to zoom"
        )
        self._edit_tip.setStyleSheet(
            f"color: {PALETTE['accent']}; font-size: 10px; font-style: italic; padding-right: 8px;"
        )

    def _refresh_selection_dot(self) -> None:
        """Redraw the white selection highlight dot over the selected keyframe."""
        if self._sel_scatter is not None:
            try:
                self._plot.removeItem(self._sel_scatter)
            except Exception:
                pass
            self._sel_scatter = None

        if (self._sel_comp < 0 or self._sel_key < 0
                or self._values is None or self._times is None):
            return
        if self._sel_comp >= self._values.shape[1]:
            return
        if self._sel_key >= len(self._times):
            return

        x = float(self._times[self._sel_key])
        y = float(self._values[self._sel_key, self._sel_comp])
        self._sel_scatter = pg.ScatterPlotItem(
            x=[x], y=[y],
            size=16,
            pen=pg.mkPen("w", width=2),
            brush=pg.mkBrush(255, 255, 255, 60),
            symbol="o",
        )
        self._plot.addItem(self._sel_scatter)

    # ── Event filter — drag editing and Escape revert ──────────────────────────

    def eventFilter(self, obj, event) -> bool:  # type: ignore[override]
        """Intercept mouse events for drag editing and key events for Escape revert.

        This single method handles two distinct event sources:

        1. ``self._plot.viewport()`` — mouse press / move / release for the
           drag-to-edit mechanic.  We intercept these before PyQtGraph's own
           pan/zoom handlers see them, but only act when the click is near
           the selected keyframe dot.

        2. ``QApplication.instance()`` — key press events for Escape.
           An application-level filter sits above all widget routing in the
           process, so it fires regardless of which widget has keyboard focus.
           We guard it tightly: only consume Escape when ``_drag_active`` is
           True, so it does not interfere with Escape pressed in dialogs or
           other panels.

        Args:
            obj: The object the event was delivered to.
            event: The Qt event.

        Returns:
            ``True`` to consume the event (stop further processing),
            ``False`` to pass it through.
        """
        from PySide6.QtCore import QEvent

        t = event.type()

        # ── Branch 1: Application-level Escape revert ─────────────────────────
        if (obj is QApplication.instance()
                and t == QEvent.Type.KeyPress
                and self._drag_active
                and self._values is not None):
            if event.key() == Qt.Key.Key_Escape:
                # Restore the original value that was stored at drag-start.
                self._values[self._drag_key, self._drag_comp] = np.float32(
                    self._drag_orig)
                self._update_curve_display(self._drag_comp)
                self._refresh_selection_dot()
                # Notify the live callback so the 3-D viewport shows the
                # reverted pose immediately.
                if self._on_live_cb is not None:
                    self._on_live_cb(
                        self._anim_index, self._channel_index, self._values)
                # Reset all drag state.
                self._drag_active  = False
                self._drag_start_y = None
                self._drag_comp    = -1
                self._drag_key     = -1
                self._edit_tip.setText(
                    "Drag reverted  ·  Click a dot to select  ·  Drag up/down to edit"
                )
                self._edit_tip.setStyleSheet(
                    f"color: {PALETTE['muted']}; font-size: 10px;"
                    "font-style: italic; padding-right: 8px;"
                )
                return True   # Consume the event — don't let Escape close dialogs.
            return False      # Other keys pass through normally.

        # ── Branch 2: Viewport mouse events ───────────────────────────────────
        # Ignore events from any object other than the plot's viewport widget.
        if obj is not self._plot.viewport():
            return False

        if t == QEvent.Type.MouseButtonPress:
            # Only begin a drag if the left button is pressed near the selected dot.
            if event.button() == Qt.MouseButton.LeftButton and self._sel_comp >= 0:
                if self._is_near_selected_dot(event.position()):
                    # Record the drag start state: which keyframe, what value.
                    self._drag_comp          = self._sel_comp
                    self._drag_key           = self._sel_key
                    self._drag_orig          = float(
                        self._values[self._drag_key, self._drag_comp])
                    self._drag_start_y       = float(event.position().y())
                    self._drag_start_scene_y = self._drag_orig
                    self._drag_active        = True
                    return True   # Consume — prevent PyQtGraph from starting a pan.
            return False

        if t == QEvent.Type.MouseMove:
            if self._drag_start_y is not None and self._values is not None:
                # Compute how far the mouse has moved in Y since drag-start.
                dy_px     = float(event.position().y()) - self._drag_start_y
                view_rect = self._plot.viewRect()
                h_px      = max(1, self._plot.viewport().height())
                # Convert pixel delta to data-space delta using the current
                # view scale (data units per pixel in Y).
                px_per_unit = h_px / max(1e-9, view_rect.height())
                delta_val   = -dy_px / px_per_unit   # negative: drag up = increase value
                new_val     = np.float32(self._drag_start_scene_y + delta_val)
                # Write the new value directly into the shared array so the
                # live callback, curve display, and inspector all see it.
                self._values[self._drag_key, self._drag_comp] = new_val
                self._update_curve_display(self._drag_comp)
                self._refresh_selection_dot()
                lbl = (self._current_target_path_labels[self._drag_comp]
                       if self._drag_comp < len(self._current_target_path_labels)
                       else f"V{self._drag_comp}")
                self._edit_tip.setText(
                    f"Editing  t={float(self._times[self._drag_key]):.4f}s  "
                    f"{lbl}={float(new_val):.5f}  ·  Release to commit  ·  Esc to revert"
                )
                if self._on_live_cb is not None:
                    self._on_live_cb(
                        self._anim_index, self._channel_index, self._values)
                return True
            return False

        if t == QEvent.Type.MouseButtonRelease:
            if self._drag_start_y is not None:
                # Commit the edited value to the glTF binary blob.
                if self._on_modified_cb is not None and self._values is not None:
                    self._on_modified_cb(
                        self._anim_index, self._channel_index, self._values)
                # Reset drag state.
                self._drag_active  = False
                self._drag_start_y = None
                self._drag_comp    = -1
                self._drag_key     = -1
                self._edit_tip.setStyleSheet(
                    f"color: {PALETTE['accent']}; font-size: 10px;"
                    "font-style: italic; padding-right: 8px;"
                )
                return True
            return False

        return False

    # Pick radius in pixels.  Larger than the dot radius (5 px) so a click
    # slightly off-centre still starts the drag.
    _HIT_RADIUS_PX: float = 28.0

    def _is_near_selected_dot(self, screen_pos) -> bool:
        """Return True if ``screen_pos`` is within ``_HIT_RADIUS_PX`` of the selected dot.

        Why we work in data space rather than pixel space:
            The naive approach is to project the dot's data coordinates to
            screen pixels and compare with the click position directly.
            This breaks catastrophically when the Y view range is very narrow
            (e.g. a scale channel at exactly 1.0 ± 0.001): a one-pixel
            quantisation error in the scene→widget projection step maps to
            30+ pixels of error in the final comparison, so the threshold is
            never met and the drag never starts.

            Instead, we invert the chain — convert the click position from
            viewport pixels to data coordinates — and compare in data space
            using a hit radius derived from the current pixel-per-data-unit
            scale.  This is immune to narrow Y windows because we work in
            data units throughout.

        Coordinate chain::

            viewport pixel
              → PlotWidget.mapToScene()    → QGraphicsScene point
              → ViewBox.mapSceneToView()   → data coordinates
              data radius ← _HIT_RADIUS_PX / px_per_data_unit

        Args:
            screen_pos: QPointF click position in viewport pixel coordinates.

        Returns:
            ``True`` if the click is within the pick radius of the selected dot.
        """
        if self._values is None or self._times is None:
            return False
        if self._sel_key  < 0 or self._sel_key  >= len(self._times):
            return False
        if self._sel_comp < 0 or self._sel_comp >= self._values.shape[1]:
            return False

        from PySide6.QtCore import QPointF

        vb = self._plot.getPlotItem().getViewBox()

        # The viewport is a child widget of the PlotWidget (QGraphicsView).
        # We must add the viewport's offset to convert from viewport-local
        # pixels to PlotWidget-local pixels before calling mapToScene().
        vp_offset = self._plot.viewport().pos()
        widget_pt = QPointF(
            screen_pos.x() + vp_offset.x(),
            screen_pos.y() + vp_offset.y(),
        )
        scene_click = self._plot.mapToScene(widget_pt.toPoint())
        data_click  = vb.mapSceneToView(scene_click)

        # Dot position in data coordinates.
        dot_x = float(self._times[self._sel_key])
        dot_y = float(self._values[self._sel_key, self._sel_comp])

        # Convert _HIT_RADIUS_PX to data units using the current view transform.
        # We compute separate X and Y pixel-per-unit scales to handle
        # non-square aspect ratios correctly.
        view_rect = vb.viewRect()
        vp_w = max(1, self._plot.viewport().width())
        vp_h = max(1, self._plot.viewport().height())
        px_per_x = vp_w / max(1e-12, view_rect.width())
        px_per_y = vp_h / max(1e-12, view_rect.height())

        # Distance in pixel-equivalent units (Euclidean, mixing X and Y scales).
        dx_px = (data_click.x() - dot_x) * px_per_x
        dy_px = (data_click.y() - dot_y) * px_per_y
        return math.hypot(dx_px, dy_px) < self._HIT_RADIUS_PX

    def _update_curve_display(self, comp: int) -> None:
        """Redraw one curve line and its scatter dots after a value change.

        Args:
            comp: Component index (0=X, 1=Y, 2=Z, 3=W).
        """
        if self._values is None or self._times is None:
            return
        if comp < len(self._curves):
            self._curves[comp].setData(self._times, self._values[:, comp])
        if comp < len(self._scatter_items):
            self._scatter_items[comp].setData(
                x=self._times, y=self._values[:, comp],
                data=[(comp, j) for j in range(len(self._times))]
            )


# =============================================================================
# STEP 9 — Keyframe inspector table
# =============================================================================

class KeyframeInspectorWidget(QWidget):
    """Spreadsheet view of exact numeric keyframe values for one channel.

    Each row is one keyframe; columns are Time, X, Y, Z, W (or fewer
    depending on the channel type).  The Time column is read-only; the value
    columns are double-click editable.

    When a cell is edited and committed (Enter / Tab / click away), the new
    value is validated, written directly into the shared ``_values`` array,
    and the ``modified_callback`` is fired so the curve editor and 3-D
    viewport update immediately — no reload required.

    The ``_loading`` flag suppresses the ``itemChanged`` signal during bulk
    table fills so individual cell edits are not mistakenly treated as user
    input while data is being loaded.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        hdr = QLabel("KEYFRAME TABLE  —  double-click a value cell to edit")
        hdr.setObjectName("section_header")
        layout.addWidget(hdr)

        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(["Time", "X / W0", "Y / W1", "Z / W2", "W / W3"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.verticalHeader().setDefaultSectionSize(22)
        self._table.verticalHeader().setVisible(False)
        # Only the value columns (1–4) are editable; Time (column 0) is read-only
        # because glTF stores time stamps in a separate input accessor and they
        # are not trivially rewritten in this demo.
        self._table.setEditTriggers(QTableWidget.EditTrigger.DoubleClicked)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self._table, stretch=1)

        self._stats_label = QLabel("")
        self._stats_label.setStyleSheet(
            f"color: {PALETTE['muted']}; font-size: 11px;"
            "font-family: 'IBM Plex Mono', monospace; padding: 4px 0;"
        )
        self._stats_label.setWordWrap(True)
        layout.addWidget(self._stats_label)

        # Shared data references.  _values points to the same numpy array as
        # CurveEditorWidget._values so edits here are immediately visible there.
        self._times:        np.ndarray | None = None
        self._values:       np.ndarray | None = None
        self._n_comp:       int               = 0
        self._target_path:  str               = ""
        self._loading:      bool              = False   # suppresses itemChanged during fill
        self._modified_cb:  callable | None   = None
        self._live_cb:      callable | None   = None
        self._anim_index:   int               = -1
        self._channel_index: int              = -1

    def set_modified_callback(self, cb: callable) -> None:
        """Register a callback fired when a cell edit is committed.

        Args:
            cb: Callable with signature ``(anim_index, channel_index, values)``.
        """
        self._modified_cb = cb

    def set_live_callback(self, cb: callable) -> None:
        """Register a callback fired immediately when a cell value changes.

        Args:
            cb: Callable with signature ``(anim_index, channel_index, values)``.
        """
        self._live_cb = cb

    def set_channel_indices(self, anim_index: int, channel_index: int) -> None:
        """Store the current channel indices for use in callback payloads.

        Args:
            anim_index: Index into ``gltf.animations``.
            channel_index: Index into the animation's channels.
        """
        self._anim_index    = anim_index
        self._channel_index = channel_index

    def load_channel(
        self, times: np.ndarray, values: np.ndarray, target_path: str
    ) -> None:
        """Populate the table from a channel's keyframe data.

        Args:
            times: Keyframe times, shape ``(n_keys,)``.
            values: Keyframe values, shape ``(n_keys, n_components)``.
            target_path: glTF target path for column labelling.
        """
        self._loading = True
        try:
            if values.ndim == 1:
                values = values[:, np.newaxis]
            self._times       = times
            self._values      = values   # shared reference — edits go here directly
            self._n_comp      = values.shape[1]
            self._target_path = target_path

            n_keys, n_comp = values.shape
            self._table.setRowCount(n_keys)
            labels = COMPONENT_LABELS.get(target_path, ["V0", "V1", "V2", "V3"])
            col_headers = ["Time"] + labels[:n_comp]
            while len(col_headers) < 5:
                col_headers.append("—")
            self._table.setHorizontalHeaderLabels(col_headers)

            self._fill_table(times, values, n_comp, labels)
            self._update_stats(times, values, n_comp, labels)
        finally:
            self._loading = False

    def _fill_table(self, times, values, n_comp, labels) -> None:
        """Fill all table cells from the given arrays.

        Args:
            times: Keyframe times array.
            values: Keyframe values array.
            n_comp: Number of components (columns).
            labels: Display labels for each component column.
        """
        n_keys = len(times)
        for row in range(n_keys):
            # Time column — read-only, muted colour.
            t_item = QTableWidgetItem(f"{times[row]:.4f}")
            t_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            t_item.setForeground(QColor(PALETTE["muted"]))
            t_item.setFlags(t_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(row, 0, t_item)

            # Value columns — editable; negative values shown in orange.
            for col in range(n_comp):
                item = QTableWidgetItem(f"{values[row, col]:.6f}")
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                )
                if values[row, col] < 0:
                    item.setForeground(QColor(PALETTE["orange"]))
                else:
                    item.setForeground(QColor(PALETTE["text"]))
                self._table.setItem(row, col + 1, item)
            # Fill unused columns with a non-editable placeholder.
            for col in range(n_comp, 4):
                placeholder = QTableWidgetItem("—")
                placeholder.setFlags(placeholder.flags() & ~Qt.ItemFlag.ItemIsEditable)
                placeholder.setForeground(QColor(PALETTE["muted"]))
                self._table.setItem(row, col + 1, placeholder)

    def _update_stats(self, times, values, n_comp, labels) -> None:
        """Update the statistics label below the table.

        Shows the keyframe count, duration, and [min → max] range per component.

        Args:
            times: Keyframe times array.
            values: Keyframe values array.
            n_comp: Number of components.
            labels: Display labels for each component.
        """
        n_keys = len(times)
        stats_parts = []
        for col in range(n_comp):
            lbl  = labels[col] if col < len(labels) else f"V{col}"
            lo, hi = float(values[:, col].min()), float(values[:, col].max())
            stats_parts.append(f"{lbl}: [{lo:.3f} → {hi:.3f}]")
        duration = float(times[-1]) if n_keys > 0 else 0.0
        self._stats_label.setText(
            f"{n_keys} keyframes  ·  {duration:.3f}s\n" + "  ".join(stats_parts)
        )

    def refresh_from_values(self) -> None:
        """Repopulate all cells from the shared ``_values`` array.

        Called by the main window after a drag-edit in the curve editor so
        the table reflects the in-progress or committed value immediately.
        """
        if self._values is None or self._times is None:
            return
        self._loading = True
        try:
            labels = COMPONENT_LABELS.get(self._target_path, ["V0", "V1", "V2", "V3"])
            self._fill_table(self._times, self._values, self._n_comp, labels)
            self._update_stats(self._times, self._values, self._n_comp, labels)
        finally:
            self._loading = False

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        """Validate and commit a table cell edit.

        Called by Qt whenever any cell's text changes.  The ``_loading``
        guard prevents this from firing during bulk table fills.

        Args:
            item: The cell that changed.
        """
        if self._loading or self._values is None or self._times is None:
            return
        row = item.row()
        col = item.column()
        if col == 0 or col > self._n_comp:
            return   # Time column or empty placeholder — ignore.
        comp = col - 1
        try:
            # Accept both "." and "," as decimal separator.
            new_val = float(item.text().replace(",", "."))
        except ValueError:
            # Invalid text — revert the cell to the current stored value.
            self._loading = True
            item.setText(f"{float(self._values[row, comp]):.6f}")
            self._loading = False
            return

        self._values[row, comp] = np.float32(new_val)

        # Recolour the cell: negative values in orange, positive in white.
        self._loading = True
        if new_val < 0:
            item.setForeground(QColor(PALETTE["orange"]))
        else:
            item.setForeground(QColor(PALETTE["text"]))
        self._loading = False

        # Fire the live callback first so the curve editor and viewport update
        # immediately, then fire the modified callback to write to the blob.
        if self._live_cb is not None:
            self._live_cb(self._anim_index, self._channel_index, self._values)
        if self._modified_cb is not None:
            self._modified_cb(self._anim_index, self._channel_index, self._values)

        labels = COMPONENT_LABELS.get(self._target_path, ["V0", "V1", "V2", "V3"])
        self._update_stats(self._times, self._values, self._n_comp, labels)

    def highlight_row(self, keyframe_index: int) -> None:
        """Select and scroll to the row for a given keyframe index.

        Called during playback so the table scrolls to show the nearest
        keyframe to the current playhead position.

        Args:
            keyframe_index: Row index to highlight.
        """
        if 0 <= keyframe_index < self._table.rowCount():
            self._table.selectRow(keyframe_index)
            self._table.scrollToItem(self._table.item(keyframe_index, 0))

    def clear(self) -> None:
        """Remove all rows and reset data references."""
        self._loading = True
        self._table.setRowCount(0)
        self._loading = False
        self._stats_label.setText("")
        self._times  = None
        self._values = None


# =============================================================================
# STEP 10 — Animation channel list panel
# =============================================================================

class ChannelListWidget(QWidget):
    """Left-side panel listing all animation channels for the selected node.

    Groups channels by animation clip name and shows a path icon (→ for
    translation, ↻ for rotation, ⊡ for scale) for quick identification.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        hdr = QLabel("ANIMATION CHANNELS")
        hdr.setObjectName("section_header")
        layout.addWidget(hdr)

        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.setColumnCount(1)
        self._tree.setIndentation(16)
        layout.addWidget(self._tree, stretch=1)

        self._on_select_cb: callable | None = None
        self._tree.itemClicked.connect(self._on_item_clicked)

    def set_select_callback(self, cb: callable) -> None:
        """Register a callback fired when a channel item is clicked.

        Args:
            cb: Callable with signature
                ``(anim_index, channel_index, anim_name, path)``.
        """
        self._on_select_cb = cb

    def load_channels(self, channels: list[tuple[int, int, str, str]]) -> None:
        """Populate the tree from a list of channel tuples.

        Args:
            channels: List of ``(anim_index, channel_index, anim_name, path)``
                tuples as returned by ``get_animation_channels_for_node()``.
        """
        self._tree.clear()
        # Group by animation name so channels from the same clip are together.
        anim_groups: dict[str, list[tuple[int, int, str, str]]] = {}
        for entry in channels:
            anim_groups.setdefault(entry[2], []).append(entry)
        for anim_name, entries in anim_groups.items():
            group = QTreeWidgetItem([f"▸ {anim_name}"])
            group.setForeground(0, QColor(PALETTE["accent"]))
            # Animation group headers are not selectable — only leaf items are.
            group.setFlags(group.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            self._tree.addTopLevelItem(group)
            for anim_idx, ch_idx, _, path in entries:
                # Icon visually encodes the channel type.
                icon  = {"translation": "→", "rotation": "↻",
                         "scale": "⊡", "weights": "≈"}.get(path, "·")
                child = QTreeWidgetItem([f"  {icon}  {path}"])
                # Store the full tuple in UserRole so the click handler can
                # retrieve it without parsing the label text.
                child.setData(0, Qt.ItemDataRole.UserRole, (anim_idx, ch_idx, anim_name, path))
                child.setForeground(0, QColor(PALETTE["text"]))
                group.addChild(child)
            group.setExpanded(True)

    def clear(self) -> None:
        """Remove all items from the tree."""
        self._tree.clear()

    def _on_item_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        """Fire the select callback when a channel item is clicked.

        Args:
            item: The clicked tree item.
            _column: Column index (unused — single-column tree).
        """
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if data and self._on_select_cb:
            self._on_select_cb(*data)


# =============================================================================
# STEP 11 — Transport / playback controls
# =============================================================================

class TransportWidget(QWidget):
    """Bottom transport bar: play / pause / stop / loop, speed slider, scrubber.

    Mirrors the transport controls of any professional animation package:
    Maya's Transport Controls, Blender's Timeline header, or any NLE timeline.
    The speed slider maps values 1–30 to 0.1×–3.0× speed in 0.1× steps.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(52)
        self.setStyleSheet(
            f"background-color: {PALETTE['surface']}; "
            f"border-top: 1px solid {PALETTE['border']};"
        )
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(8)

        # Play / pause — checkable so it stays depressed while playing.
        self._play_btn = QPushButton("▶")
        self._play_btn.setCheckable(True)
        self._play_btn.setFixedWidth(36)
        self._play_btn.setToolTip("Play / Pause  (Space)")
        layout.addWidget(self._play_btn)

        # Stop rewinds to t=0.
        self._stop_btn = QPushButton("■")
        self._stop_btn.setFixedWidth(36)
        self._stop_btn.setToolTip("Stop — rewind to t=0")
        layout.addWidget(self._stop_btn)

        # Loop — when checked, playback wraps from the end back to t=0.
        self._loop_btn = QPushButton("⟳")
        self._loop_btn.setCheckable(True)
        self._loop_btn.setChecked(True)
        self._loop_btn.setFixedWidth(36)
        self._loop_btn.setToolTip("Toggle loop")
        layout.addWidget(self._loop_btn)

        layout.addSpacing(8)

        speed_lbl = QLabel("Speed:")
        speed_lbl.setStyleSheet(f"color: {PALETTE['muted']}; font-size: 11px;")
        layout.addWidget(speed_lbl)

        # Speed slider: integer range 1–30 mapped to 0.1×–3.0× by dividing by 10.
        self._speed_slider = QSlider(Qt.Orientation.Horizontal)
        self._speed_slider.setRange(1, 30)
        self._speed_slider.setValue(10)
        self._speed_slider.setFixedWidth(90)
        self._speed_slider.setToolTip("Playback speed (0.1× – 3.0×)")
        layout.addWidget(self._speed_slider)

        self._speed_label = QLabel("1.0×")
        self._speed_label.setStyleSheet(
            f"color: {PALETTE['accent']}; font-size: 11px; "
            "font-family: 'IBM Plex Mono', monospace; min-width: 32px;"
        )
        layout.addWidget(self._speed_label)
        self._speed_slider.valueChanged.connect(
            lambda v: self._speed_label.setText(f"{v / 10:.1f}×")
        )

        layout.addSpacing(8)

        # Scrubber: maps [0, 10000] to [0, duration] with 0.01% precision.
        self._scrubber = QSlider(Qt.Orientation.Horizontal)
        self._scrubber.setRange(0, 10000)
        self._scrubber.setValue(0)
        layout.addWidget(self._scrubber, stretch=1)

        self._time_display = QLabel("0.000 s")
        self._time_display.setObjectName("time_display")
        layout.addWidget(self._time_display)

        # Expose controls as public attributes for the main window to connect.
        self.play_btn     = self._play_btn
        self.stop_btn     = self._stop_btn
        self.loop_btn     = self._loop_btn
        self.speed_slider = self._speed_slider
        self.scrubber     = self._scrubber

    def set_duration(self, duration: float) -> None:
        """Reset the scrubber to the start when a new channel is loaded.

        Args:
            duration: Total animation duration in seconds.
        """
        self._scrubber.setValue(0)
        self._duration = duration

    def update_time(self, t: float, duration: float) -> None:
        """Update the time display and scrubber position during playback.

        ``blockSignals`` prevents the scrubber's ``sliderMoved`` signal from
        firing while we set its position programmatically, which would cause
        a feedback loop with the scrubber-moved callback in the main window.

        Args:
            t: Current playback time in seconds.
            duration: Total animation duration in seconds.
        """
        self._time_display.setText(f"{t:.3f} s")
        if duration > 0:
            self._scrubber.blockSignals(True)
            self._scrubber.setValue(int(t / duration * 10000))
            self._scrubber.blockSignals(False)


# =============================================================================
# STEP 12 — Main application window
# =============================================================================

class GltfCurveEditor(QMainWindow):
    """Main window — glTF Animation Curve Editor.

    Coordinates the five panels (3-D viewport, scene tree, curve editor,
    keyframe inspector, transport bar) and routes all inter-panel signals:
    bone selection → channel list → curve editor → keyframe table → viewport.

    Layout::

        ┌──────────────────────────────────────────────────────────────────────┐
        │  Toolbar: [logo] [author]  Open GLB  [filename]      ? Help  Export │
        ├───────────────┬──────────────────────────────────────────────────────┤
        │  Scene Tree   │  [Help panel — collapsible]                          │
        │  (hierarchy)  │  3-D Viewport  (OpenGL — mesh + skeleton)            │
        ├───────────────┤  ────────────────────────────────────────────────────┤
        │  Anim Channels│  Curve Editor (PyQtGraph)     │  Keyframe Table      │
        │  (per node)   │                               │  (QTableWidget)      │
        ├───────────────┴───────────────────────────────┴──────────────────────┤
        │  Transport:  ▶  ■  ⟳   Speed ───  Scrubber ──────────  0.000 s      │
        └──────────────────────────────────────────────────────────────────────┘
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("glTF Animation Curve Editor — Computeranimation Seminar 04")
        self.setMinimumSize(1100, 700)

        # Application state.
        self._gltf:                  GLTF2 | None = None
        self._gltf_path:             Path  | None = None
        self._current_anim_index:    int          = -1
        self._current_channel_index: int          = -1
        self._current_duration:      float        = 0.0
        self._playback_time:         float        = 0.0
        self._is_playing:            bool         = False
        self._dirty:                 bool         = False   # True if unsaved edits exist
        self._help_visible:          bool         = False
        self._skel_btn:              QPushButton  = QPushButton()

        self._build_toolbar()
        self._build_central()
        self._build_transport()
        self._build_status_bar()
        self._build_timer()

        self.setStyleSheet(APP_STYLESHEET)
        self._show_empty_state()
        # Keep the help panel hidden by default so the 3-D viewport is
        # immediately visible when the app opens.
        self._help_panel.setVisible(False)
        self._help_btn.setChecked(False)
        self._help_btn.setText("? Help")

    # -------------------------------------------------------------------------
    # UI construction
    # -------------------------------------------------------------------------

    def _build_toolbar(self) -> None:
        """Create the top toolbar with logo, file actions, and skeleton toggle."""
        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        # ── Logo + author ─────────────────────────────────────────────────────
        logo_widget = QWidget()
        logo_layout = QVBoxLayout(logo_widget)
        logo_layout.setContentsMargins(0, 0, 8, 0)
        logo_layout.setSpacing(1)

        logo_lbl = QLabel()
        logo_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if LOGO_PATH.exists():
            pixmap = QPixmap(str(LOGO_PATH)).scaledToWidth(
                60, Qt.TransformationMode.SmoothTransformation
            )
            logo_lbl.setPixmap(pixmap)
        else:
            logo_lbl.setText("AMS Group – TU Ilmenau")
            logo_lbl.setStyleSheet(
                f"color: {PALETTE['accent']}; font-size: 10px; font-weight: 600;"
            )

        author_lbl = QLabel("Renato Fera P.")
        author_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        author_lbl.setStyleSheet(
            f"color: {PALETTE['muted']}; font-size: 9px; letter-spacing: 0.5px;"
        )

        logo_layout.addWidget(logo_lbl)
        logo_layout.addWidget(author_lbl)
        toolbar.addWidget(logo_widget)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet(f"color: {PALETTE['border']}; margin: 4px 6px;")
        toolbar.addWidget(sep)

        # ── File actions ──────────────────────────────────────────────────────
        open_btn = QPushButton("  Open GLB…")
        open_btn.setObjectName("accent_btn")
        open_btn.setToolTip("Open a .glb file  (Ctrl+O)")
        open_btn.clicked.connect(self._on_open)
        toolbar.addWidget(open_btn)

        toolbar.addSeparator()

        self._file_label = QLabel("No file loaded")
        toolbar.addWidget(self._file_label)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)

        help_btn = QPushButton("? Help")
        help_btn.setObjectName("help_btn")
        help_btn.setToolTip("Toggle usage instructions  (?)")
        help_btn.setCheckable(True)
        help_btn.setChecked(True)
        help_btn.clicked.connect(self._toggle_help)
        toolbar.addWidget(help_btn)
        self._help_btn = help_btn

        toolbar.addSeparator()

        export_btn = QPushButton("  Export GLB…")
        export_btn.setToolTip("Export modified file  (Ctrl+S)")
        export_btn.clicked.connect(self._on_export)
        toolbar.addWidget(export_btn)

        toolbar.addSeparator()

        # ── Skeleton overlay toggle ───────────────────────────────────────────
        skel_btn = QPushButton("🦴 Skeleton")
        skel_btn.setCheckable(True)
        skel_btn.setChecked(True)
        skel_btn.setToolTip("Toggle skeleton overlay  (K)")
        skel_btn.clicked.connect(self._toggle_skeleton)
        toolbar.addWidget(skel_btn)
        self._skel_btn = skel_btn

        # Register keyboard shortcuts via QAction so they work even when the
        # toolbar buttons don't have focus.
        for shortcut, slot in [("Ctrl+O", self._on_open),
                                ("Ctrl+S", self._on_export),
                                ("Space",  self._toggle_playback),
                                ("?",      self._toggle_help),
                                ("K",      self._toggle_skeleton)]:
            act = QAction(self)
            act.setShortcut(shortcut)
            act.triggered.connect(slot)
            self.addAction(act)

    def _build_central(self) -> None:
        """Build the three-column splitter layout in the central area."""
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        h_split = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(h_split, stretch=1)
        self._h_splitter = h_split

        # ── Left column: scene tree + channel list ────────────────────────────
        left_panel  = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        tree_hdr = QLabel("SCENE TREE")
        tree_hdr.setObjectName("section_header")
        tree_hdr.setContentsMargins(8, 8, 8, 4)
        left_layout.addWidget(tree_hdr)

        self._scene_tree = QTreeWidget()
        self._scene_tree.setHeaderHidden(True)
        self._scene_tree.setColumnCount(1)
        self._scene_tree.setIndentation(14)
        self._scene_tree.itemClicked.connect(self._on_node_selected)
        left_layout.addWidget(self._scene_tree, stretch=3)

        div = QFrame(); div.setObjectName("divider")
        div.setFrameShape(QFrame.Shape.HLine)
        left_layout.addWidget(div)

        self._channel_list = ChannelListWidget()
        self._channel_list.set_select_callback(self._on_channel_selected)
        left_layout.addWidget(self._channel_list, stretch=2)

        left_panel.setMinimumWidth(220)
        left_panel.setMaximumWidth(340)
        h_split.addWidget(left_panel)

        # ── Centre column: help panel + 3-D viewport + curve editor ──────────
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)

        self._help_panel = HelpPanel()
        center_layout.addWidget(self._help_panel)

        self._viewport = ModelViewportWidget()
        self._viewport.bone_selected.connect(self._on_bone_selected_in_viewport)
        center_layout.addWidget(self._viewport, stretch=2)

        vp_div = QFrame(); vp_div.setObjectName("divider")
        vp_div.setFrameShape(QFrame.Shape.HLine)
        center_layout.addWidget(vp_div)

        self._curve_editor = CurveEditorWidget()
        self._curve_editor.set_modified_callback(self._on_keyframe_modified)
        self._curve_editor.set_live_callback(self._on_keyframe_live)
        center_layout.addWidget(self._curve_editor, stretch=3)

        h_split.addWidget(center_widget)

        # ── Right column: keyframe inspector (scrollable) ─────────────────────
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setMinimumWidth(220)
        right_scroll.setMaximumWidth(340)
        right_scroll.setStyleSheet(f"border: none; background: {PALETTE['bg']};")

        self._inspector = KeyframeInspectorWidget()
        self._inspector.set_modified_callback(self._on_keyframe_modified)
        self._inspector.set_live_callback(self._on_keyframe_live)
        right_scroll.setWidget(self._inspector)
        h_split.addWidget(right_scroll)

        # Distribute horizontal space: 18% left, 62% centre, 20% right.
        h_split.setStretchFactor(0, 18)
        h_split.setStretchFactor(1, 62)
        h_split.setStretchFactor(2, 20)

    def _build_transport(self) -> None:
        """Add the TransportWidget below the central splitter."""
        self._transport = TransportWidget()
        self.centralWidget().layout().addWidget(self._transport)
        self._transport.play_btn.clicked.connect(self._toggle_playback)
        self._transport.stop_btn.clicked.connect(self._stop_playback)
        self._transport.scrubber.sliderMoved.connect(self._on_scrubber_moved)

    def _build_status_bar(self) -> None:
        """Add the status bar with an initial hint message."""
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage(
            "Step 1: Open a .glb file (Ctrl+O or blue button)  ·  "
            "Step 2: Click a bone in the Scene Tree (or click a joint dot in the viewport)  ·  "
            "Step 3: Click a channel (rotation / translation / scale)  ·  "
            "Step 4: Press Space to play  ·  "
            "Viewport: Left-drag orbit  ·  Middle-drag pan  ·  Scroll zoom  ·  "
            "Double-click reframe  ·  🦴 Skeleton / K to toggle skeleton"
        )

    def _build_timer(self) -> None:
        """Create the playback timer.  Fires ``_on_tick`` at ~60 fps."""
        self._timer = QTimer(self)
        self._timer.setInterval(PLAYBACK_INTERVAL_MS)
        self._timer.timeout.connect(self._on_tick)

    # -------------------------------------------------------------------------
    # Empty / help state
    # -------------------------------------------------------------------------

    def _show_empty_state(self) -> None:
        """Clear all panels to their initial empty state."""
        self._scene_tree.clear()
        self._channel_list.clear()
        self._curve_editor.clear()
        self._inspector.clear()

    def _toggle_help(self) -> None:
        """Show or hide the help panel."""
        self._help_visible = not self._help_visible
        self._help_panel.setVisible(self._help_visible)
        self._help_btn.setChecked(self._help_visible)
        self._help_btn.setText("✕ Help" if self._help_visible else "? Help")

    def _toggle_skeleton(self) -> None:
        """Toggle the skeleton overlay in the 3-D viewport."""
        visible = self._skel_btn.isChecked()
        self._viewport.set_skeleton_visible(visible)
        self._skel_btn.setText("🦴 Skeleton" if visible else "○ Skeleton")

    def _on_bone_selected_in_viewport(self, node_index: int) -> None:
        """Handle a joint click in the 3-D viewport.

        Finds the matching item in the Scene Tree, selects it, and fires
        ``_on_node_selected`` so the channel list and curve editor update
        as if the user had clicked the bone in the tree directly.

        Args:
            node_index: glTF node index of the clicked joint.
        """
        def _find(parent) -> "QTreeWidgetItem | None":
            count = (self._scene_tree.topLevelItemCount()
                     if parent is None else parent.childCount())
            for i in range(count):
                item = (self._scene_tree.topLevelItem(i)
                        if parent is None else parent.child(i))
                if item.data(0, Qt.ItemDataRole.UserRole) == node_index:
                    return item
                found = _find(item)
                if found is not None:
                    return found
            return None

        target = _find(None)
        if target is not None:
            self._scene_tree.setCurrentItem(target)
            self._scene_tree.scrollToItem(target)
            self._on_node_selected(target, 0)

    # -------------------------------------------------------------------------
    # File I/O
    # -------------------------------------------------------------------------

    def _on_open(self) -> None:
        """Open a file dialog and load the selected .glb file."""
        start_dir = str(self._gltf_path.parent) if self._gltf_path else str(Path.home())
        path_str, _ = QFileDialog.getOpenFileName(
            self, "Open glTF Binary (.glb)", start_dir,
            "glTF Binary (*.glb);;All Files (*)"
        )
        if not path_str:
            return
        path = Path(path_str)
        try:
            gltf = load_gltf(path)
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", f"Could not load {path.name}:\n{exc}")
            return

        self._gltf      = gltf
        self._gltf_path = path
        self._dirty     = False
        self._file_label.setText(f"  {path.name}")

        # Pre-read all accessor data and build reverse-index lookups immediately
        # after loading.  Amortising this cost to load time means every
        # subsequent per-frame animation evaluation is just dict lookups
        # and fast NumPy array sampling — no I/O, no allocation.
        self._status.showMessage(f"Building animation cache for {path.name}…")
        QApplication.processEvents()   # let the status bar repaint
        channel_cache = build_channel_cache(gltf)
        object.__getattribute__(gltf, "__dict__")["_channel_cache"] = channel_cache
        for ai in range(len(gltf.animations or [])):
            _build_anim_node_lookup(gltf, ai)

        first_anim = 0 if (gltf.animations and len(gltf.animations) > 0) else -1
        self._viewport.load_model(gltf, anim_index=first_anim, path=path)
        self._current_anim_index = first_anim

        self._populate_scene_tree()
        self._channel_list.clear()
        self._curve_editor.clear()
        self._inspector.clear()

        n_nodes  = len(gltf.nodes or [])
        n_anims  = len(gltf.animations or [])
        n_meshes = len(gltf.meshes or [])
        n_skins  = len(gltf.skins or [])
        self._status.showMessage(
            f"{path.name}  ·  {n_nodes} nodes  ·  {n_meshes} meshes  ·  "
            f"{n_skins} skins  ·  {n_anims} animations  ·  "
            f"Model loaded — auto-selecting first animated bone…"
        )

        # Auto-select the first bone that has animation channels so the user
        # immediately sees curves without having to click through the tree.
        self._auto_select_first_animated_node()

    def _on_export(self) -> None:
        """Open a save dialog and write the modified .glb to disk."""
        if self._gltf is None:
            QMessageBox.information(self, "Export", "No file loaded.")
            return
        default_name = (self._gltf_path.stem + "_modified.glb"
                        if self._gltf_path else "export.glb")
        start_dir = str(self._gltf_path.parent) if self._gltf_path else str(Path.home())
        path_str, _ = QFileDialog.getSaveFileName(
            self, "Export glTF Binary (.glb)",
            str(Path(start_dir) / default_name),
            "glTF Binary (*.glb);;All Files (*)"
        )
        if not path_str:
            return
        out_path = Path(path_str)
        try:
            size = save_gltf(self._gltf, out_path)
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", f"Could not save:\n{exc}")
            return
        self._dirty = False
        self._status.showMessage(
            f"Exported → {out_path.name}  ({size / 1024:.1f} KB)"
        )

    # -------------------------------------------------------------------------
    # Scene tree
    # -------------------------------------------------------------------------

    def _populate_scene_tree(self) -> None:
        """Rebuild the Scene Tree from the loaded glTF node hierarchy."""
        self._scene_tree.clear()
        if self._gltf is None:
            return
        # Collect all node indices that appear in any skin's joint list.
        # These are the "bones" — they get a special icon and colour.
        skin_joint_set: set[int] = set()
        for skin in (self._gltf.skins or []):
            skin_joint_set.update(skin.joints or [])
        for scene in (self._gltf.scenes or []):
            for root_idx in (scene.nodes or []):
                item = build_scene_tree(self._gltf, None, root_idx, skin_joint_set)
                if item is not None:
                    self._scene_tree.addTopLevelItem(item)
        self._scene_tree.expandAll()

    # -------------------------------------------------------------------------
    # Auto-selection helper
    # -------------------------------------------------------------------------

    def _auto_select_first_animated_node(self) -> None:
        """Select the first node that has animation channels automatically.

        This runs after every file load so the user immediately sees curves
        in the editor without having to navigate the tree themselves.
        """
        if self._gltf is None:
            return

        first_node_idx: int | None = None
        for anim in (self._gltf.animations or []):
            for channel in (anim.channels or []):
                if channel.target.node is not None:
                    first_node_idx = channel.target.node
                    break
            if first_node_idx is not None:
                break

        if first_node_idx is None:
            return

        # Walk the QTreeWidget to find the item whose UserRole matches.
        def _find(parent: QTreeWidgetItem | None) -> QTreeWidgetItem | None:
            count = (self._scene_tree.topLevelItemCount()
                     if parent is None else parent.childCount())
            for i in range(count):
                item = (self._scene_tree.topLevelItem(i)
                        if parent is None else parent.child(i))
                if item.data(0, Qt.ItemDataRole.UserRole) == first_node_idx:
                    return item
                found = _find(item)
                if found is not None:
                    return found
            return None

        target_item = _find(None)
        if target_item is None:
            return

        self._scene_tree.setCurrentItem(target_item)
        self._scene_tree.scrollToItem(target_item)
        self._on_node_selected(target_item, 0)

    # -------------------------------------------------------------------------
    # Selection handlers
    # -------------------------------------------------------------------------

    def _on_node_selected(self, item: QTreeWidgetItem, _column: int) -> None:
        """Handle a node click in the Scene Tree.

        Highlights the joint in the 3-D viewport, loads its animation
        channels into the channel list, and auto-selects the first channel.

        Args:
            item: The clicked tree item.
            _column: Column index (unused).
        """
        node_index: int | None = item.data(0, Qt.ItemDataRole.UserRole)
        if node_index is None or self._gltf is None:
            return
        self._viewport.set_selected_joint(node_index)
        channels = get_animation_channels_for_node(self._gltf, node_index)
        self._channel_list.load_channels(channels)
        if not channels:
            self._curve_editor.clear()
            self._inspector.clear()
            self._status.showMessage(
                f"Node '{get_node_name(self._gltf, node_index)}' has no animation channels."
            )
        else:
            # Auto-select the first channel so curves are always visible after
            # clicking a bone — no second click required.
            self._on_channel_selected(*channels[0])

    def _on_channel_selected(self, anim_index: int, channel_index: int,
                              anim_name: str, target_path: str) -> None:
        """Handle a channel selection from the ChannelListWidget.

        Loads the selected channel into the curve editor, keyframe inspector,
        and synchronises the viewport and transport bar to this channel's
        animation.

        Args:
            anim_index: Index into ``gltf.animations``.
            channel_index: Index into the animation's channels list.
            anim_name: Display name of the animation clip.
            target_path: glTF target path (``"translation"``, etc.).
        """
        if self._gltf is None:
            return
        self._current_anim_index    = anim_index
        self._current_channel_index = channel_index

        self._viewport.set_anim_index(anim_index)

        channel   = self._gltf.animations[anim_index].channels[channel_index]
        node_name = get_node_name(self._gltf, channel.target.node)

        self._curve_editor.load_channel(
            self._gltf, anim_index, channel_index,
            node_name, target_path, anim_name,
        )

        times, values = read_channel_data(self._gltf, anim_index, channel_index)
        if values.ndim == 1:
            values = values[:, np.newaxis]
        # Pass the channel indices to the inspector so its callbacks fire
        # with the correct (anim_index, channel_index) payload.
        self._inspector.set_channel_indices(anim_index, channel_index)
        self._inspector.load_channel(times, values, target_path)

        duration = float(times[-1]) if len(times) > 0 else 0.0
        self._current_duration = duration
        self._transport.set_duration(duration)
        self._playback_time = 0.0
        self._transport.update_time(0.0, duration)

        self._status.showMessage(
            f"'{node_name}'  ·  {target_path}  ·  {anim_name}  ·  "
            f"{len(times)} keyframes  ·  {duration:.3f}s  ·  "
            f"Space to play  |  Click a coloured keyframe dot on the curve to edit its value"
        )

    # -------------------------------------------------------------------------
    # Keyframe modification callbacks
    # -------------------------------------------------------------------------

    def _on_keyframe_modified(self, anim_index: int, channel_index: int,
                               new_values: np.ndarray) -> None:
        """Commit a keyframe edit to the glTF binary blob.

        Called on mouse-release (from the curve editor) or on table cell
        commit (from the keyframe inspector).  Writes the new values to the
        in-memory binary blob, marks the document dirty, and refreshes the
        inspector table.

        Args:
            anim_index: Index into ``gltf.animations``.
            channel_index: Index into the animation's channels.
            new_values: Updated values array, shape ``(n_keys, n_components)``.
        """
        if self._gltf is None:
            return
        write_channel_values(self._gltf, anim_index, channel_index, new_values)
        self._dirty = True
        times, values = read_channel_data(self._gltf, anim_index, channel_index)
        channel = self._gltf.animations[anim_index].channels[channel_index]
        self._inspector.load_channel(times, values, channel.target.path)
        self._status.showMessage(
            "✎  Keyframe modified — unsaved changes  ·  Ctrl+S to export the modified .glb"
        )

    def _on_keyframe_live(self, anim_index: int, channel_index: int,
                          new_values: np.ndarray) -> None:
        """Drive the UI on every mouse-move during a drag (before commit).

        Does NOT write to the glTF blob — that only happens in
        ``_on_keyframe_modified`` on mouse-release.  This refreshes the curve
        display and drives the 3-D viewport so the user sees the pose change
        live while dragging, matching Maya / Blender Graph Editor behaviour.

        Args:
            anim_index: Index into ``gltf.animations``.
            channel_index: Index into the animation's channels.
            new_values: Current (uncommitted) values array.
        """
        if self._gltf is None:
            return
        # Refresh the curve display so the dragged dot and line move live.
        self._curve_editor.refresh_values()
        # Refresh the inspector table cells to reflect the in-progress value.
        self._inspector.refresh_from_values()
        # Drive the 3-D viewport with the current uncommitted values.
        self._viewport.set_playback_time(self._playback_time)

    # -------------------------------------------------------------------------
    # Playback
    # -------------------------------------------------------------------------

    def _toggle_playback(self) -> None:
        """Start or pause playback."""
        if self._current_duration <= 0:
            return
        self._is_playing = not self._is_playing
        self._transport.play_btn.setChecked(self._is_playing)
        self._transport.play_btn.setText("⏸" if self._is_playing else "▶")
        if self._is_playing:
            self._timer.start()
        else:
            self._timer.stop()

    def _stop_playback(self) -> None:
        """Stop playback and rewind to t=0."""
        self._is_playing = False
        self._timer.stop()
        self._transport.play_btn.setChecked(False)
        self._transport.play_btn.setText("▶")
        self._playback_time = 0.0
        self._transport.update_time(0.0, self._current_duration)
        self._curve_editor.set_playhead(0.0)
        self._viewport.set_playback_time(0.0)

    def _on_tick(self) -> None:
        """Advance playback by one frame interval.

        Called by the QTimer every 16 ms (~60 fps).  Computes the new
        playback time from the speed slider value, handles loop or stop at
        the end of the clip, and drives all three real-time displays
        (transport scrubber, curve playhead, 3-D viewport).
        """
        speed = self._transport.speed_slider.value() / 10.0
        dt    = PLAYBACK_INTERVAL_MS / 1000.0 * speed
        self._playback_time += dt

        if self._playback_time > self._current_duration:
            if self._transport.loop_btn.isChecked():
                self._playback_time = 0.0
            else:
                self._playback_time = self._current_duration
                self._stop_playback()
                return

        self._transport.update_time(self._playback_time, self._current_duration)
        self._curve_editor.set_playhead(self._playback_time)
        self._viewport.set_playback_time(self._playback_time)

        # Scroll the keyframe table to the nearest keyframe so students can
        # watch the values change as the animation plays.
        if (self._gltf is not None
                and self._current_anim_index >= 0
                and self._current_channel_index >= 0):
            times, _ = read_channel_data(
                self._gltf, self._current_anim_index, self._current_channel_index
            )
            nearest = int(np.argmin(np.abs(times - self._playback_time)))
            self._inspector.highlight_row(nearest)

    def _on_scrubber_moved(self, value: int) -> None:
        """Jump to a specific time when the scrubber is dragged.

        Args:
            value: Slider integer value in [0, 10000]; mapped to [0, duration].
        """
        if self._current_duration <= 0:
            return
        t = value / 10000.0 * self._current_duration
        self._playback_time = t
        self._transport.update_time(t, self._current_duration)
        self._curve_editor.set_playhead(t)
        self._viewport.set_playback_time(t)


# =============================================================================
# STEP 13 — Entry point
# =============================================================================

def main() -> None:
    """Create the QApplication and enter the Qt event loop.

    OpenGL surface format:
        The surface format must be set **before** creating the QApplication
        on all platforms.  Setting it after QApplication construction is
        ignored on most drivers.

        We request OpenGL 3.3 Core Profile because:
        - Core Profile removes all deprecated fixed-function features and
          forces the use of explicit VAOs, VBOs, and shader programs —
          the same API surface that WebGL 2, Metal, and Vulkan share
          conceptually.
        - 3.3 is the minimum version guaranteed on macOS (Apple dropped
          older OpenGL versions on modern hardware).
        - The 24-bit depth buffer ensures correct depth testing at all zoom
          levels without Z-fighting artefacts.
        - 4× MSAA (multi-sample anti-aliasing) smooths the jagged skeleton
          line edges without requiring manual supersampling in the shader.
    """
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setDepthBufferSize(24)
    fmt.setSamples(4)   # 4× MSAA for smoother skeleton lines and mesh edges
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setApplicationName("glTF Animation Curve Editor")
    app.setApplicationVersion("2.3.0")
    app.setOrganizationName("TU Ilmenau — AMS Group")

    window = GltfCurveEditor()
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

# =============================================================================
# End of 04_ca_gltf_python.py
# Author   : Renato Fera P.
# LinkedIn : https://www.linkedin.com/in/renato-profeta/
# Course   : Computeranimation — Chapter 04: glTF & Python Animation Tools
# Institute: Institut für Medientechnik, Technische Universität Ilmenau
# =============================================================================
