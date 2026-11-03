# =============================================================================
# 04_ca_gltf_python.py — glTF Animation Curve Editor  (Seminar 04, Demo 5)
# =============================================================================
# Author   : Renato Fera P.
# LinkedIn : https://www.linkedin.com/in/renato-profeta/
# Course   : Computeranimation — Chapter 04: glTF & Python Animation Tools
# Institute: Institut für Medientechnik, Technische Universität Ilmenau
# -----------------------------------------------------------------------------
# Description:
"""04_ca_gltf_python.py — glTF Animation Curve Editor.

A professional Qt application for inspecting, visualising and editing
animation data inside binary glTF (.glb) files.  Five panels work together:

  1. 3-D Viewport   — renders the mesh with Phong shading and draws the
                       skeleton as coloured bone lines; the model animates
                       live as the transport plays.
  2. Scene Tree     — glTF node hierarchy (bones in purple, meshes in green).
  3. Curve Editor   — PyQtGraph F-curve editor; one coloured curve per
                       animated component (X=red Y=green Z=blue W=yellow).
  4. Keyframe Table — spreadsheet view of the selected channel; rows highlight
                       as the playhead moves.
  5. Transport bar  — ▶ pause stop loop, speed slider, timeline scrubber.

Workflow
--------
  1. File → Open GLB  (Ctrl+O) — load any .glb with animations.
  2. Click a node in the Scene Tree — its animation channels appear below.
  3. Click a channel (translation / rotation / scale) — curves are drawn.
  4. Press Space (or ▶) to play — watch the model animate.
  5. Click a keyframe dot on the curve to edit its value in-place.
  6. File → Export GLB  (Ctrl+S) — save the modified file.

Stack
-----
  PySide6    — Qt6 Python binding (industry standard: Maya, Blender, Houdini).
  PyOpenGL   — OpenGL bindings for the 3-D viewport.
  PyQtGraph  — GPU-accelerated plot library for the curve editor.
  pygltflib  — Pure Python glTF 2.0 reader / writer.
  NumPy      — All keyframe and geometry data live as float32 typed arrays.

How to run
----------
  cd 04_gltf
  uv sync                          # first time — creates .venv, installs deps
  uv run python 04_ca_gltf_python.py

Ubuntu system dependencies (once):
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
# ANIMATION CONCEPT: glTF encodes data types as numeric constants defined in
# the WebGL / OpenGL ES specification. 5126 = GL_FLOAT (32-bit IEEE 754).
# These same constants appear in WebGL shaders, Three.js BufferAttribute,
# and every other GL-family API.

COMPONENT_DTYPE: dict[int, tuple[np.dtype, int]] = {
    5120: (np.dtype("int8"),    1),
    5121: (np.dtype("uint8"),   1),
    5122: (np.dtype("int16"),   2),
    5123: (np.dtype("uint16"),  2),
    5125: (np.dtype("uint32"),  4),
    5126: (np.dtype("float32"), 4),
}

ACCESSOR_NUM_COMPONENTS: dict[str, int] = {
    "SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4,
    "MAT2":   4, "MAT3": 9, "MAT4": 16,
}

# X=red  Y=green  Z=blue  W=yellow — matches Maya / Blender convention
CURVE_COLORS: list[tuple[int, int, int, int]] = [
    (220,  60,  60, 220),
    ( 80, 200,  80, 220),
    ( 60, 120, 220, 220),
    (220, 200,  50, 220),
]

COMPONENT_LABELS: dict[str, list[str]] = {
    "translation": ["X", "Y", "Z"],
    "rotation":    ["X", "Y", "Z", "W"],
    "scale":       ["X", "Y", "Z"],
    "weights":     ["W0", "W1", "W2", "W3"],
}

PLAYBACK_INTERVAL_MS: int = 16   # ≈60 fps

# AMS logo — one level up from the script (repo root)
LOGO_PATH: Path = Path(__file__).resolve().parent.parent / "ams_group_tui.png"

PALETTE = {
    "bg":      "#0d0f14",
    "surface": "#161a22",
    "border":  "#252b38",
    "text":    "#d8dce6",
    "muted":   "#5a6175",
    "accent":  "#4f9cf9",
    "accent2": "#a78bfa",
    "green":   "#34d399",
    "orange":  "#fb923c",
    "red":     "#f87171",
}

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

def load_gltf(path: Path) -> GLTF2:
    if not path.exists():
        raise FileNotFoundError(f"GLB file not found: {path}")
    return GLTF2().load(str(path))


def read_accessor(gltf: GLTF2, accessor_index: int) -> np.ndarray:
    """Read one glTF accessor as a NumPy array (copy, writeable)."""
    accessor    = gltf.accessors[accessor_index]
    buffer_view = gltf.bufferViews[accessor.bufferView]
    blob: bytes = gltf.binary_blob()
    view_start  = buffer_view.byteOffset or 0
    acc_start   = view_start + (accessor.byteOffset or 0)
    dtype, item_size = COMPONENT_DTYPE.get(
        accessor.componentType, (np.dtype("float32"), 4)
    )
    n_comp      = ACCESSOR_NUM_COMPONENTS.get(accessor.type, 1)
    byte_stride = buffer_view.byteStride or (item_size * n_comp)
    raw: bytes  = b"".join(
        blob[acc_start + i * byte_stride: acc_start + i * byte_stride + item_size * n_comp]
        for i in range(accessor.count)
    )
    dtype, _ = COMPONENT_DTYPE.get(accessor.componentType, (np.dtype("float32"), 4))
    arr = np.frombuffer(raw, dtype=dtype).copy()
    if n_comp > 1:
        arr = arr.reshape(-1, n_comp)
    return arr


def read_channel_data(
    gltf: GLTF2, anim_index: int, channel_index: int
) -> tuple[np.ndarray, np.ndarray]:
    anim    = gltf.animations[anim_index]
    channel = anim.channels[channel_index]
    sampler = anim.samplers[channel.sampler]
    times   = read_accessor(gltf, sampler.input).flatten().astype(np.float32)
    values  = read_accessor(gltf, sampler.output).astype(np.float32)
    return times, values


def get_node_name(gltf: GLTF2, node_index: int | None) -> str:
    if node_index is None:
        return "(none)"
    node = gltf.nodes[node_index]
    return node.name or f"Node_{node_index}"


def save_gltf(gltf: GLTF2, path: Path) -> int:
    gltf.save(str(path))
    return path.stat().st_size


def write_channel_values(
    gltf: GLTF2, anim_index: int, channel_index: int,
    new_values: np.ndarray,
) -> None:
    anim        = gltf.animations[anim_index]
    channel     = anim.channels[channel_index]
    sampler     = anim.samplers[channel.sampler]
    accessor    = gltf.accessors[sampler.output]
    buffer_view = gltf.bufferViews[accessor.bufferView]
    blob        = bytearray(gltf.binary_blob())
    byte_offset = (buffer_view.byteOffset or 0) + (accessor.byteOffset or 0)
    flat        = new_values.astype(np.float32).flatten()
    new_bytes   = struct.pack(f"{len(flat)}f", *flat)
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
) -> QTreeWidgetItem:
    node     = gltf.nodes[node_index]
    name     = node.name or f"Node_{node_index}"
    is_bone  = node_index in skin_joint_set
    has_mesh = node.mesh is not None

    if is_bone:
        label, badge = f"🦴  {name}", "bone"
    elif has_mesh:
        label, badge = f"▣  {name}", "mesh"
    else:
        label, badge = f"○  {name}", "node"

    if parent_item is not None:
        item = QTreeWidgetItem(parent_item, [label, badge, str(node_index)])
    else:
        item = QTreeWidgetItem([label, badge, str(node_index)])

    item.setData(0, Qt.ItemDataRole.UserRole, node_index)
    if is_bone:
        item.setForeground(0, QColor(PALETTE["accent2"]))
    elif has_mesh:
        item.setForeground(0, QColor(PALETTE["green"]))

    for child_index in (node.children or []):
        build_scene_tree(gltf, item, child_index, skin_joint_set)
    return item


# =============================================================================
# STEP 4 — Animation channel query
# =============================================================================

def get_animation_channels_for_node(
    gltf: GLTF2, node_index: int
) -> list[tuple[int, int, str, str]]:
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
# ANIMATION CONCEPT: To animate a skeleton we need to evaluate the animation
# curves at the current time, then compose each joint's local transform into
# a world-space matrix.  The same math drives Three.js's AnimationMixer and
# Maya's DAG evaluation.

def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two unit quaternions."""
    dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        return (q0 + t * (q1 - q0)) / np.linalg.norm(q0 + t * (q1 - q0) + 1e-12)
    theta0   = math.acos(dot)
    theta    = theta0 * t
    sin_t    = math.sin(theta)
    sin_t0   = math.sin(theta0)
    s0       = math.cos(theta) - dot * sin_t / sin_t0
    s1       = sin_t / sin_t0
    return (s0 * q0 + s1 * q1)


def _lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return a + t * (b - a)


def _sample_channel(times: np.ndarray, values: np.ndarray,
                    t: float, path: str) -> np.ndarray:
    """Evaluate an animation channel at time t (linear interpolation)."""
    if len(times) == 0:
        return values[0] if len(values) > 0 else np.zeros(3)
    if t <= times[0]:
        return values[0]
    if t >= times[-1]:
        return values[-1]
    i = int(np.searchsorted(times, t, side="right")) - 1
    i = max(0, min(i, len(times) - 2))
    alpha = float((t - times[i]) / (times[i + 1] - times[i] + 1e-12))
    if path == "rotation":
        return _slerp(values[i], values[i + 1], alpha)
    return _lerp(values[i], values[i + 1], alpha)


def _trs_to_mat4(
    translation: np.ndarray | None,
    rotation: np.ndarray | None,
    scale: np.ndarray | None,
) -> np.ndarray:
    """Build a column-major 4×4 float32 matrix from TRS components."""
    T = np.eye(4, dtype=np.float32)
    if translation is not None:
        T[:3, 3] = translation[:3]

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

    S = np.eye(4, dtype=np.float32)
    if scale is not None:
        S[0, 0], S[1, 1], S[2, 2] = scale[0], scale[1], scale[2]

    return T @ R @ S


def _node_rest_mat(node) -> np.ndarray:
    """Return the rest-pose local matrix for a glTF node."""
    if node.matrix is not None:
        return np.array(node.matrix, dtype=np.float32).reshape(4, 4).T
    t = np.array(node.translation or [0, 0, 0], dtype=np.float32)
    r = np.array(node.rotation    or [0, 0, 0, 1], dtype=np.float32)
    s = np.array(node.scale       or [1, 1, 1], dtype=np.float32)
    return _trs_to_mat4(t, r, s)


def compute_joint_matrices(
    gltf: GLTF2,
    skin_index: int,
    anim_index: int,
    t: float,
) -> np.ndarray:
    """
    Compute the joint matrix palette for one skin at time t.

    ANIMATION CONCEPT: Each joint matrix is:
        J_i = global_world_i  ×  inverse_bind_i
    global_world_i is the joint's animated world-space transform; the inverse
    bind matrix undoes the rest pose so vertices are displaced correctly.
    This is identical to the skinning formula in every GPU skeletal animation
    shader (GLSL, HLSL, Metal).

    Returns float32 array of shape (n_joints, 4, 4).
    """
    skin        = gltf.skins[skin_index]
    joint_ids   = skin.joints or []
    n_joints    = len(joint_ids)

    # Build animated local matrices for ALL nodes in the scene.
    n_nodes     = len(gltf.nodes or [])
    local_mats  = [_node_rest_mat(gltf.nodes[i]) for i in range(n_nodes)]

    # For each channel in the target animation, override the rest pose.
    if anim_index >= 0 and gltf.animations:
        anim = gltf.animations[anim_index]
        for ch_idx, channel in enumerate(anim.channels or []):
            ni = channel.target.node
            if ni is None:
                continue
            path = channel.target.path
            times, values = read_channel_data(gltf, anim_index, ch_idx)
            if values.ndim == 1:
                values = values[:, np.newaxis]
            val = _sample_channel(times, values, t, path)

            node = gltf.nodes[ni]
            rest_t = np.array(node.translation or [0, 0, 0], dtype=np.float32)
            rest_r = np.array(node.rotation    or [0, 0, 0, 1], dtype=np.float32)
            rest_s = np.array(node.scale       or [1, 1, 1], dtype=np.float32)
            if path == "translation":
                rest_t = val.flatten()[:3]
            elif path == "rotation":
                rest_r = val.flatten()[:4]
            elif path == "scale":
                rest_s = val.flatten()[:3]
            local_mats[ni] = _trs_to_mat4(rest_t, rest_r, rest_s)

    # Accumulate world matrices via scene-graph traversal.
    world_mats = [None] * n_nodes

    def _visit(ni: int, parent_world: np.ndarray) -> None:
        world_mats[ni] = parent_world @ local_mats[ni]
        for ci in (gltf.nodes[ni].children or []):
            _visit(ci, world_mats[ni])

    root_ids: list[int] = []
    for scene in (gltf.scenes or []):
        root_ids.extend(scene.nodes or [])
    if not root_ids:
        root_ids = list(range(n_nodes))
    for ri in root_ids:
        _visit(ri, np.eye(4, dtype=np.float32))

    # Inverse bind matrices
    ibm_arr = np.eye(4, dtype=np.float32)[np.newaxis].repeat(n_joints, axis=0)
    if skin.inverseBindMatrices is not None:
        raw_ibm = read_accessor(gltf, skin.inverseBindMatrices)
        if raw_ibm.ndim == 1:
            raw_ibm = raw_ibm.reshape(-1, 16)
        for i, row in enumerate(raw_ibm[:n_joints]):
            ibm_arr[i] = row.reshape(4, 4).T   # glTF is column-major

    # Joint matrix = world × inverse_bind
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
    """
    Return world-space origin (vec3) for every node at time t.
    Used to draw the skeleton as lines in the viewport.
    """
    n_nodes    = len(gltf.nodes or [])
    local_mats = [_node_rest_mat(gltf.nodes[i]) for i in range(n_nodes)]

    if anim_index >= 0 and gltf.animations:
        anim = gltf.animations[anim_index]
        for ch_idx, channel in enumerate(anim.channels or []):
            ni   = channel.target.node
            if ni is None:
                continue
            path = channel.target.path
            times, values = read_channel_data(gltf, anim_index, ch_idx)
            if values.ndim == 1:
                values = values[:, np.newaxis]
            val  = _sample_channel(times, values, t, path)
            node = gltf.nodes[ni]
            rest_t = np.array(node.translation or [0, 0, 0], dtype=np.float32)
            rest_r = np.array(node.rotation    or [0, 0, 0, 1], dtype=np.float32)
            rest_s = np.array(node.scale       or [1, 1, 1], dtype=np.float32)
            if path == "translation":
                rest_t = val.flatten()[:3]
            elif path == "rotation":
                rest_r = val.flatten()[:4]
            elif path == "scale":
                rest_s = val.flatten()[:3]
            local_mats[ni] = _trs_to_mat4(rest_t, rest_r, rest_s)

    world_mats: dict[int, np.ndarray] = {}

    def _visit(ni: int, pw: np.ndarray) -> None:
        world_mats[ni] = pw @ local_mats[ni]
        for ci in (gltf.nodes[ni].children or []):
            _visit(ci, world_mats[ni])

    roots: list[int] = []
    for scene in (gltf.scenes or []):
        roots.extend(scene.nodes or [])
    if not roots:
        roots = list(range(n_nodes))
    for r in roots:
        _visit(r, np.eye(4, dtype=np.float32))

    return {ni: wm[:3, 3] for ni, wm in world_mats.items()}


# =============================================================================
# STEP 6 — 3-D OpenGL viewport
# =============================================================================

_VERT_SRC = """
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec4 aJoints;
layout(location = 3) in vec4 aWeights;
layout(location = 4) in vec2 aTexCoord;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProj;
uniform mat4 uJoints[64];
uniform bool uSkinned;

out vec3 vNormal;
out vec3 vFragPos;
out vec2 vTexCoord;

void main() {
    vec4 pos4 = vec4(aPos, 1.0);
    vec3 norm = aNormal;

    if (uSkinned) {
        mat4 skin =
            aWeights.x * uJoints[int(aJoints.x)] +
            aWeights.y * uJoints[int(aJoints.y)] +
            aWeights.z * uJoints[int(aJoints.z)] +
            aWeights.w * uJoints[int(aJoints.w)];
        pos4  = skin * pos4;
        norm  = mat3(skin) * aNormal;
    }

    vec4 worldPos = uModel * pos4;
    gl_Position   = uProj * uView * worldPos;
    vFragPos      = worldPos.xyz;
    vNormal       = normalize(mat3(transpose(inverse(uModel))) * norm);
    vTexCoord     = aTexCoord;
}
"""

_FRAG_SRC = """
#version 330 core
in vec3 vNormal;
in vec3 vFragPos;
in vec2 vTexCoord;

uniform vec3  uLightPos;
uniform vec3  uLightColor;
uniform vec4  uBaseColor;      // glTF pbrMetallicRoughness.baseColorFactor (linear)
uniform float uAmbient;
uniform bool  uHasTexture;
uniform sampler2D uAlbedo;     // bound as GL_SRGB8_ALPHA8 — GPU linearises automatically

out vec4 FragColor;

void main() {
    // ── Albedo ────────────────────────────────────────────────────────────────
    // uBaseColor is already linear (glTF spec §3.9.3).
    // The texture sampler is sRGB-encoded; because we uploaded it as
    // GL_SRGB8_ALPHA8 the hardware linearises it on read — no manual pow(2.2).
    vec4 albedo = uBaseColor;
    if (uHasTexture) {
        albedo *= texture(uAlbedo, vTexCoord);   // already linear after sRGB decode
    }

    // ── Blinn-Phong lighting (all in linear space) ────────────────────────────
    vec3 norm     = normalize(vNormal);
    vec3 lightDir = normalize(uLightPos - vFragPos);
    float NdotL   = max(dot(norm, lightDir), 0.0);

    // View direction: vFragPos is in world space and camera is at the view origin
    // after the view transform, so we approximate view dir from the fragment pos.
    vec3 viewDir  = normalize(-vFragPos);
    vec3 halfDir  = normalize(lightDir + viewDir);
    float NdotH   = max(dot(norm, halfDir), 0.0);
    // Specular: attenuated so it doesn't blow out coloured surfaces
    float spec    = pow(NdotH, 32.0) * 0.25;

    vec3 ambient  = uAmbient * albedo.rgb;
    vec3 diffuse  = NdotL   * albedo.rgb * uLightColor;
    vec3 specular = spec     * uLightColor;          // white highlight on top

    vec3 result   = ambient + diffuse + specular;

    // ── Approximate gamma encode for display (no sRGB framebuffer guaranteed) ─
    result        = pow(clamp(result, 0.0, 1.0), vec3(1.0 / 2.2));
    FragColor     = vec4(result, albedo.a);
}
"""

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
    """
    OpenGL 3-D viewport.

    Renders the first mesh found in the loaded glTF with Phong shading,
    and overlays the skeleton as coloured bone lines.  The model is
    updated every frame by the main-window playback timer via
    set_playback_time().

    OPENGL CONCEPT: The pipeline is:
      CPU numpy arrays  →  VBO upload  →  vertex shader (skinning)
      →  fragment shader (Phong)  →  framebuffer  →  screen.
    The vertex shader performs GPU skeletal skinning exactly as a real
    game engine does: each vertex samples up to 4 joint matrices weighted
    by the JOINTS_0 / WEIGHTS_0 accessors in the glTF primitive.
    """

    # Signal emitted when user clicks a bone joint in the viewport.
    # Carries the glTF node index of the nearest joint.
    bone_selected = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._gltf:          GLTF2 | None = None
        self._anim_index:    int          = -1
        self._playback_t:    float        = 0.0
        self._mesh_vao:      int          = 0
        self._mesh_vbo:      int          = 0
        self._mesh_ebo:      int          = 0
        self._mesh_count:    int          = 0
        self._skin_index:    int          = -1
        self._is_skinned:    bool         = False
        self._prog:          QOpenGLShaderProgram | None = None
        self._line_prog:     QOpenGLShaderProgram | None = None
        self._line_vao:      int          = 0
        self._line_vbo:      int          = 0
        self._orbit_yaw:     float        = 30.0
        self._orbit_pitch:   float        = 15.0
        self._orbit_dist:    float        = 3.0
        self._last_mouse                  = None
        self._bbox_center    = np.zeros(3, dtype=np.float32)
        self._bbox_scale:    float        = 1.0
        self._show_skeleton: bool         = True       # toggled by toolbar button
        self._selected_joint: int         = -1         # highlighted joint node index
        self._tex_id:        int          = 0          # OpenGL texture object (0 = none)
        self._has_tex:       bool         = False
        self._base_color     = np.array([0.8, 0.8, 0.8, 1.0], dtype=np.float32)
        # World-space joint positions for picking (populated each frame)
        self._joint_world_pos: dict[int, np.ndarray] = {}
        self.setMinimumSize(320, 240)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_model(self, gltf: GLTF2, anim_index: int = 0) -> None:
        """Upload the first mesh primitive from gltf to the GPU."""
        self._gltf       = gltf
        self._anim_index = anim_index
        self._playback_t = 0.0
        self.makeCurrent()
        self._upload_mesh()
        self._upload_skeleton_vbo()
        self.doneCurrent()
        self.update()

    def set_playback_time(self, t: float) -> None:
        self._playback_t = t
        self.update()

    def set_anim_index(self, idx: int) -> None:
        self._anim_index = idx
        self.update()

    def set_skeleton_visible(self, visible: bool) -> None:
        """Toggle skeleton overlay on/off."""
        self._show_skeleton = visible
        self.update()

    def set_selected_joint(self, node_index: int) -> None:
        """Highlight a joint by node index (called when tree selection changes)."""
        self._selected_joint = node_index
        self.update()

    # ------------------------------------------------------------------
    # Qt OpenGL overrides
    # ------------------------------------------------------------------

    def initializeGL(self) -> None:
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_CULL_FACE)
        GL.glClearColor(
            int(PALETTE["bg"][1:3], 16) / 255,
            int(PALETTE["bg"][3:5], 16) / 255,
            int(PALETTE["bg"][5:7], 16) / 255,
            1.0,
        )

        # Mesh shader — log compile/link errors so blank viewport is diagnosable
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

        # Skeleton line shader
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

        # Allocate VAO / VBO placeholders
        vaos = GL.glGenVertexArrays(2)
        self._mesh_vao, self._line_vao = int(vaos[0]), int(vaos[1])
        vbos = GL.glGenBuffers(3)
        self._mesh_vbo = int(vbos[0])
        self._mesh_ebo = int(vbos[1])
        self._line_vbo = int(vbos[2])

        # Allocate one texture slot; filled by _upload_texture()
        self._tex_id = int(GL.glGenTextures(1))

    def resizeGL(self, w: int, h: int) -> None:
        GL.glViewport(0, 0, w, h)

    def paintGL(self) -> None:
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        if self._gltf is None or self._mesh_count == 0:
            self._draw_empty_hint()
            return
        self._draw_mesh()
        if self._show_skeleton:
            self._draw_skeleton()

    # ------------------------------------------------------------------
    # Mouse orbit
    # ------------------------------------------------------------------

    def mouseDoubleClickEvent(self, event) -> None:
        """Consume double-clicks so they don't reach the window title bar
        and accidentally toggle the maximized state."""
        event.accept()

    def mousePressEvent(self, event) -> None:
        self._last_mouse = event.position()
        # Bone picking: only on left-click with no drag
        if event.button() == Qt.MouseButton.LeftButton:
            self._try_pick_bone(event.position())

    def mouseMoveEvent(self, event) -> None:
        if self._last_mouse is None:
            return
        dx = event.position().x() - self._last_mouse.x()
        dy = event.position().y() - self._last_mouse.y()
        # If there's real drag, cancel any pending pick so orbit takes priority
        if abs(dx) > 2 or abs(dy) > 2:
            self._last_mouse = None   # will be reset below
        self._orbit_yaw   += dx * 0.4
        self._orbit_pitch  = max(-89, min(89, self._orbit_pitch + dy * 0.4))
        self._last_mouse   = event.position()
        self.update()

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        # FIX: proportional zoom so large and small models both feel responsive.
        # One standard wheel notch (delta = 120) scales distance by ~0.9 (zoom in)
        # or ~1.11 (zoom out).
        self._orbit_dist *= 0.999 ** delta
        self._orbit_dist = max(0.01, self._orbit_dist)
        self.update()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _mvp(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (model, view, proj) as row-major 4×4 float32 ndarrays.

        All three are uploaded with GL_TRUE (transpose), which makes OpenGL
        treat each numpy array as row-major — matching standard math notation
        M @ v rather than v @ M.
        """
        w, h   = max(1, self.width()), max(1, self.height())
        aspect = w / h

        # FIX: scale near/far with the current orbit distance so that large
        # meshes (Fox.glb is ~150 units tall) are never clipped.
        near = max(0.001, self._orbit_dist * 0.005)
        far  = self._orbit_dist * 200.0

        # Projection (perspective) — row-major form:
        #   Row 0: [f/a   0    0              0            ]
        #   Row 1: [  0   f    0              0            ]
        #   Row 2: [  0   0   (f+n)/(n-f)   2fn/(n-f)    ]  ← proj[2,3]
        #   Row 3: [  0   0   -1             0            ]  ← proj[3,2]
        fov    = math.radians(45.0)
        f      = 1.0 / math.tan(fov / 2)
        proj   = np.zeros((4, 4), dtype=np.float32)
        proj[0, 0] = f / aspect
        proj[1, 1] = f
        proj[2, 2] = (far + near) / (near - far)
        # FIX: these two entries were swapped in the original, sending every
        # vertex outside the clip volume (nothing visible).
        proj[2, 3] = (2 * far * near) / (near - far)   # was -1.0  ← BUG
        proj[3, 2] = -1.0                               # was 2fn/(n-f) ← BUG

        # View (orbit camera)
        yaw   = math.radians(self._orbit_yaw)
        pitch = math.radians(self._orbit_pitch)
        cx    = self._orbit_dist * math.cos(pitch) * math.sin(yaw)
        cy    = self._orbit_dist * math.sin(pitch)
        cz    = self._orbit_dist * math.cos(pitch) * math.cos(yaw)
        eye   = self._bbox_center + np.array([cx, cy, cz], dtype=np.float32)
        ctr   = self._bbox_center.copy()
        up    = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        f_vec = (ctr - eye); f_vec /= (np.linalg.norm(f_vec) + 1e-9)
        r_vec = np.cross(f_vec, up); r_vec /= (np.linalg.norm(r_vec) + 1e-9)
        u_vec = np.cross(r_vec, f_vec)
        view  = np.eye(4, dtype=np.float32)
        view[0, :3] = r_vec
        view[1, :3] = u_vec
        view[2, :3] = -f_vec
        view[0, 3]  = -np.dot(r_vec, eye)
        view[1, 3]  = -np.dot(u_vec, eye)
        view[2, 3]  =  np.dot(f_vec, eye)

        model = np.eye(4, dtype=np.float32)
        return model, view, proj

    def _upload_mesh(self) -> None:
        """Extract position / normal / joints / weights / texcoord from the first mesh
        primitive, read the material base colour and optional albedo texture, then
        upload everything to the GPU."""
        gltf = self._gltf
        if not gltf or not gltf.meshes:
            return

        mesh  = gltf.meshes[0]
        prim  = mesh.primitives[0]
        attrs = prim.attributes

        # Positions (required)
        pos_idx = getattr(attrs, "POSITION", None)
        if pos_idx is None:
            return
        positions = read_accessor(gltf, pos_idx).astype(np.float32)
        if positions.ndim == 1:
            positions = positions.reshape(-1, 3)

        # Normals (compute flat normals if absent)
        norm_idx = getattr(attrs, "NORMAL", None)
        if norm_idx is not None:
            normals = read_accessor(gltf, norm_idx).astype(np.float32)
            if normals.ndim == 1:
                normals = normals.reshape(-1, 3)
        else:
            normals = np.zeros_like(positions)
            normals[:, 2] = 1.0

        # Texture coordinates
        tc_idx = getattr(attrs, "TEXCOORD_0", None)
        if tc_idx is not None:
            texcoords = read_accessor(gltf, tc_idx).astype(np.float32)
            if texcoords.ndim == 1:
                texcoords = texcoords.reshape(-1, 2)
        else:
            texcoords = np.zeros((len(positions), 2), dtype=np.float32)

        # Joints / weights for skinning
        j0_idx = getattr(attrs, "JOINTS_0",  None)
        w0_idx = getattr(attrs, "WEIGHTS_0", None)
        if j0_idx is not None and w0_idx is not None:
            joints  = read_accessor(gltf, j0_idx).astype(np.float32)
            weights = read_accessor(gltf, w0_idx).astype(np.float32)
            if joints.ndim  == 1: joints  = joints.reshape(-1, 4)
            if weights.ndim == 1: weights = weights.reshape(-1, 4)
            self._is_skinned = True
        else:
            joints  = np.zeros((len(positions), 4), dtype=np.float32)
            weights = np.tile([1, 0, 0, 0], (len(positions), 1)).astype(np.float32)
            self._is_skinned = False

        # Indices
        if prim.indices is not None:
            indices = read_accessor(gltf, prim.indices).flatten().astype(np.uint32)
        else:
            indices = np.arange(len(positions), dtype=np.uint32)
        self._mesh_count = len(indices)

        # ── Material: base colour + optional texture ──────────────────────────
        # glTF default baseColorFactor is [1, 1, 1, 1] (white, fully opaque).
        self._base_color = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self._has_tex    = False
        mat_idx = prim.material
        if mat_idx is not None and gltf.materials and mat_idx < len(gltf.materials):
            mat = gltf.materials[mat_idx]
            pbr = getattr(mat, "pbrMetallicRoughness", None)
            if pbr is not None:
                bcf = getattr(pbr, "baseColorFactor", None)
                if bcf is not None and len(bcf) >= 3:
                    # glTF spec: baseColorFactor is in linear space
                    self._base_color = np.array(
                        [bcf[0], bcf[1], bcf[2], bcf[3] if len(bcf) > 3 else 1.0],
                        dtype=np.float32
                    )
                tex_info = getattr(pbr, "baseColorTexture", None)
                if tex_info is not None:
                    tex_src_idx = getattr(tex_info, "index", None)
                    if tex_src_idx is not None:
                        self._upload_texture(gltf, tex_src_idx)

        # ── Camera / bbox auto-fit ────────────────────────────────────────────
        mn, mx = positions.min(axis=0), positions.max(axis=0)
        self._bbox_center = ((mn + mx) / 2.0).astype(np.float32)
        extent = float(np.max(mx - mn)) or 1.0
        self._bbox_scale  = 1.0 / extent
        # Place camera 1.4× the mesh extent away for comfortable framing.
        self._orbit_dist  = extent * 1.4

        # ── Interleave: pos(3)|normal(3)|joints(4)|weights(4)|texcoord(2) = 16 floats ──
        n_verts     = len(positions)
        interleaved = np.zeros((n_verts, 16), dtype=np.float32)
        interleaved[:, 0:3]   = positions
        interleaved[:, 3:6]   = normals
        interleaved[:, 6:10]  = joints
        interleaved[:, 10:14] = weights
        interleaved[:, 14:16] = texcoords
        vertex_data = interleaved.flatten()

        stride = 16 * 4   # bytes

        GL.glBindVertexArray(self._mesh_vao)

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._mesh_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, vertex_data.nbytes,
                        vertex_data, GL.GL_DYNAMIC_DRAW)

        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self._mesh_ebo)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes,
                        indices, GL.GL_STATIC_DRAW)

        # Attribute pointers
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, stride,
                                 GL.ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, stride,
                                 GL.ctypes.c_void_p(12))
        GL.glEnableVertexAttribArray(2)
        GL.glVertexAttribPointer(2, 4, GL.GL_FLOAT, GL.GL_FALSE, stride,
                                 GL.ctypes.c_void_p(24))
        GL.glEnableVertexAttribArray(3)
        GL.glVertexAttribPointer(3, 4, GL.GL_FLOAT, GL.GL_FALSE, stride,
                                 GL.ctypes.c_void_p(40))
        GL.glEnableVertexAttribArray(4)
        GL.glVertexAttribPointer(4, 2, GL.GL_FLOAT, GL.GL_FALSE, stride,
                                 GL.ctypes.c_void_p(56))

        GL.glBindVertexArray(0)

        # Pick skin index for this mesh
        self._skin_index = -1
        for ni, node in enumerate(gltf.nodes or []):
            if node.mesh == 0 and node.skin is not None:
                self._skin_index = node.skin
                break

    def _upload_texture(self, gltf: "GLTF2", tex_index: int) -> None:
        """Decode and upload the glTF texture at tex_index to self._tex_id.

        Handles three image storage cases:
          1. bufferView  — image bytes embedded in the GLB binary chunk (most common)
          2. data: URI   — base-64 encoded image in the JSON (some exporters)
          3. external URI — file on disk relative to the .glb (skipped for .glb)

        Uses GL_SRGB8_ALPHA8 as the internal format so OpenGL linearises the
        sRGB texture samples before Phong lighting — without this, albedo colours
        look too dark and desaturated because the shader multiplies already-gamma-
        encoded values in linear space.
        """
        import io, base64
        try:
            from PIL import Image as PILImage
        except ImportError:
            print("[Viewport] Pillow not installed — textures disabled. "
                  "Run: pip install pillow")
            return
        try:
            if not gltf.textures or tex_index >= len(gltf.textures):
                return
            tex   = gltf.textures[tex_index]
            img_i = tex.source
            if img_i is None:
                return
            if not gltf.images or img_i >= len(gltf.images):
                return
            img_def = gltf.images[img_i]

            raw_bytes: bytes | None = None

            # Case 1: image bytes live in the GLB binary chunk
            if img_def.bufferView is not None:
                bv        = gltf.bufferViews[img_def.bufferView]
                blob      = gltf.binary_blob()
                start     = bv.byteOffset or 0
                raw_bytes = bytes(blob[start: start + bv.byteLength])

            # Case 2: data: URI  (e.g. "data:image/png;base64,iVBOR…")
            elif img_def.uri and img_def.uri.startswith("data:"):
                header, b64 = img_def.uri.split(",", 1)
                raw_bytes   = base64.b64decode(b64)

            # Case 3: external file reference — not useful inside a .glb
            else:
                print(f"[Viewport] External image URI '{img_def.uri}' "
                      "— skipped (only embedded textures supported).")
                return

            if raw_bytes is None:
                return

            pil      = PILImage.open(io.BytesIO(raw_bytes)).convert("RGBA")
            img_data = np.array(pil, dtype=np.uint8)
            h, w     = img_data.shape[:2]

            GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex_id)
            # GL_SRGB8_ALPHA8: GPU linearises sRGB→linear before texturing,
            # matching glTF's requirement that baseColorTexture is sRGB-encoded.
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0,
                            GL.GL_SRGB8_ALPHA8,        # internal (linearise)
                            w, h, 0,
                            GL.GL_RGBA,                # source format
                            GL.GL_UNSIGNED_BYTE, img_data)
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
            self._has_tex = True
            print(f"[Viewport] Texture {tex_index} uploaded ({w}×{h} px, sRGB)")
        except Exception:
            traceback.print_exc()

    def _upload_skeleton_vbo(self) -> None:
        """Allocate skeleton line VBO (filled dynamically each frame)."""
        GL.glBindVertexArray(self._line_vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._line_vbo)
        # Allocate space for up to 256 bone segments × 2 endpoints × 3 floats
        GL.glBufferData(GL.GL_ARRAY_BUFFER, 256 * 2 * 3 * 4,
                        None, GL.GL_DYNAMIC_DRAW)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 12,
                                 GL.ctypes.c_void_p(0))
        GL.glBindVertexArray(0)

    def _draw_mesh(self) -> None:
        if self._prog is None:
            return
        model, view, proj = self._mvp()
        self._prog.bind()

        def _u(name: str) -> int:
            return GL.glGetUniformLocation(self._prog.programId(), name)

        GL.glUniformMatrix4fv(_u("uModel"), 1, GL.GL_TRUE, model)
        GL.glUniformMatrix4fv(_u("uView"),  1, GL.GL_TRUE, view)
        GL.glUniformMatrix4fv(_u("uProj"),  1, GL.GL_TRUE, proj)

        # Light: scale position with orbit distance so it always illuminates
        # the model from a sensible angle regardless of mesh size.
        light = self._bbox_center + self._orbit_dist * np.array(
            [0.5, 0.8, 0.6], dtype=np.float32)
        GL.glUniform3f(_u("uLightPos"),   *light)
        GL.glUniform3f(_u("uLightColor"), 1.0, 1.0, 1.0)
        # Ambient: 0.15 preserves colour saturation — 0.30 washed everything out
        GL.glUniform1f(_u("uAmbient"),    0.15)

        # Material: pass the glTF base colour factor (linear, as per glTF spec)
        bc = self._base_color
        r  = float(bc[0]) if len(bc) > 0 else 0.8
        g  = float(bc[1]) if len(bc) > 1 else 0.8
        b  = float(bc[2]) if len(bc) > 2 else 0.8
        a  = float(bc[3]) if len(bc) > 3 else 1.0
        GL.glUniform4f(_u("uBaseColor"), r, g, b, a)

        # Albedo texture (slot 0)
        if self._has_tex:
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex_id)
            GL.glUniform1i(_u("uAlbedo"),     0)
            GL.glUniform1i(_u("uHasTexture"), 1)
        else:
            GL.glUniform1i(_u("uHasTexture"), 0)

        GL.glUniform1i(_u("uSkinned"),
                       1 if self._is_skinned and self._skin_index >= 0 else 0)

        # Upload joint matrices
        if self._is_skinned and self._skin_index >= 0 and self._gltf is not None:
            try:
                jmats = compute_joint_matrices(
                    self._gltf, self._skin_index,
                    self._anim_index, self._playback_t,
                )
                n = min(len(jmats), 64)
                flat = jmats[:n].flatten().astype(np.float32)
                loc  = GL.glGetUniformLocation(self._prog.programId(), "uJoints[0]")
                GL.glUniformMatrix4fv(loc, n, GL.GL_TRUE, flat)
            except Exception:
                # Print once so shader / skinning problems are diagnosable,
                # then continue — the mesh will render in rest pose.
                traceback.print_exc()

        GL.glBindVertexArray(self._mesh_vao)
        GL.glDrawElements(GL.GL_TRIANGLES, self._mesh_count,
                          GL.GL_UNSIGNED_INT, None)
        GL.glBindVertexArray(0)
        if self._has_tex:
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        self._prog.release()

    def _draw_skeleton(self) -> None:
        """Draw parent→child bone lines and joint dots in the skeleton overlay.

        FIX: glLineWidth(2.0) is illegal in OpenGL Core Profile 3.3 and raises
        GL_INVALID_VALUE (err 1281).  Core Profile only permits width 1.0.
        We therefore use width 1.0 and compensate with bright colour + alpha.
        Joint positions are stored in self._joint_world_pos for bone picking.
        """
        if self._line_prog is None or self._gltf is None:
            return
        gltf = self._gltf
        try:
            world_pos = compute_world_positions(gltf, self._anim_index, self._playback_t)
        except Exception:
            return

        # Store for bone-picking in mouse events
        self._joint_world_pos = world_pos

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

        arr = np.array(segments, dtype=np.float32)
        max_bytes = 256 * 2 * 3 * 4
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._line_vbo)
        GL.glBufferSubData(GL.GL_ARRAY_BUFFER, 0,
                           min(arr.nbytes, max_bytes), arr)

        model, view, proj = self._mvp()
        mvp = proj @ view @ model

        self._line_prog.bind()
        loc_mvp   = GL.glGetUniformLocation(self._line_prog.programId(), "uMVP")
        loc_color = GL.glGetUniformLocation(self._line_prog.programId(), "uColor")
        GL.glUniformMatrix4fv(loc_mvp, 1, GL.GL_TRUE, mvp.flatten())

        # FIX: glLineWidth > 1.0 is not allowed in Core Profile (raises GL error 1281).
        # Use exactly 1.0 — the only guaranteed value.
        GL.glLineWidth(1.0)

        # Draw bone lines in accent blue
        GL.glUniform3f(loc_color, 0.310, 0.780, 0.976)
        GL.glBindVertexArray(self._line_vao)
        n_verts = min(len(segments) // 3, 256 * 2)
        GL.glDrawArrays(GL.GL_LINES, 0, n_verts)

        # Draw joint dots: upload each joint position as a 1-vertex point
        # and draw it as GL_POINTS (size set by gl_PointSize in the shader is
        # not available without setting it in GLSL, so we draw a tiny cross
        # instead by pushing 6 extra vertices per joint — 3 axis segments).
        # Simpler approach: collect all joint world-pos and draw as GL_POINTS
        # using a separate small upload. GL_POINTS size is set to 6 via
        # glPointSize (allowed in Core Profile unlike glLineWidth).
        joint_pts: list[float] = []
        selected_pt: list[float] = []
        for ni, pos in world_pos.items():
            if ni == self._selected_joint:
                selected_pt.extend(pos.tolist())
            else:
                joint_pts.extend(pos.tolist())

        # Regular joints — white dots, size 5
        if joint_pts:
            pts_arr = np.array(joint_pts, dtype=np.float32)
            GL.glBufferSubData(GL.GL_ARRAY_BUFFER, 0,
                               min(pts_arr.nbytes, max_bytes), pts_arr)
            GL.glUniform3f(loc_color, 0.9, 0.9, 0.9)
            GL.glPointSize(5.0)
            GL.glDrawArrays(GL.GL_POINTS, 0, len(joint_pts) // 3)

        # Selected joint — orange dot, larger
        if selected_pt:
            sel_arr = np.array(selected_pt, dtype=np.float32)
            GL.glBufferSubData(GL.GL_ARRAY_BUFFER, 0,
                               min(sel_arr.nbytes, max_bytes), sel_arr)
            GL.glUniform3f(loc_color, 1.0, 0.576, 0.235)   # orange
            GL.glPointSize(10.0)
            GL.glDrawArrays(GL.GL_POINTS, 0, 1)

        GL.glPointSize(1.0)
        GL.glBindVertexArray(0)
        self._line_prog.release()

    def _try_pick_bone(self, screen_pos) -> None:
        """Ray-sphere bone picking: find the nearest joint to the click position.

        Projects each joint world position to NDC, then picks the closest one
        within a pixel-radius threshold and emits bone_selected(node_index).
        """
        if not self._joint_world_pos or self._gltf is None:
            return
        model, view, proj = self._mvp()
        mvp = proj @ view @ model
        w, h = max(1, self.width()), max(1, self.height())
        sx, sy = float(screen_pos.x()), float(screen_pos.y())

        best_ni   = -1
        best_dist = 20.0   # max pixel radius for a hit

        for ni, wpos in self._joint_world_pos.items():
            p4  = mvp @ np.array([*wpos, 1.0], dtype=np.float32)
            if p4[3] <= 0:
                continue
            ndc_x =  p4[0] / p4[3]
            ndc_y = -p4[1] / p4[3]   # flip Y: NDC up = screen up
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
        """Draw a centred hint when no model is loaded yet."""
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
    """
    Collapsible panel shown at the top of the viewport column.
    Explains the four-step workflow to the student.
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
              <b>Click a coloured dot</b> on any curve to edit that keyframe value.
              A dialog will ask for the new number. Ctrl+S exports the modified .glb.
            </td>
          </tr>
          <tr>
            <td style="color:{PALETTE['text']};font-size:12px;font-weight:bold;
                       padding:3px 12px 1px 0;white-space:nowrap;vertical-align:top;">
              Viewport
            </td>
            <td style="color:{PALETTE['muted']};font-size:11px;padding:3px 0 1px 0;">
              Drag to orbit &middot; Scroll to zoom (works at any scale).<br>
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
    """
    PyQtGraph-backed animation curve editor panel with drag-to-edit keyframes.

    EDITING UX (matches Maya Graph Editor / Blender F-Curve Editor):
      • Click a coloured dot  →  select it (turns white, status bar shows value).
      • Drag selected dot UP/DOWN  →  value changes live; curve and 3-D viewport
        update on every mouse-move so you see the animation change in real time.
      • Release mouse  →  value is committed to the glTF binary blob.
      • Escape while dragging  →  reverts to the value at drag-start.

    ANIMATION CONCEPT: Maya's Graph Editor and Blender's F-Curve Editor show
    exactly this: time on X, animated value on Y, one coloured line per channel
    component.  The curve shape — ease-in, ease-out, overshoot — is core animation
    literacy.
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

        # Install event filter on the viewport so we can intercept mouse events
        # for drag-editing without fighting PyQtGraph's own pan/zoom handlers.
        self._plot.viewport().installEventFilter(self)
        layout.addWidget(self._plot, stretch=1)

        # ── State ─────────────────────────────────────────────────────────────
        self._curves: list[pg.PlotDataItem]          = []
        self._scatter_items: list[pg.ScatterPlotItem] = []
        self._playhead: pg.InfiniteLine | None        = None
        self._times:  np.ndarray | None               = None
        self._values: np.ndarray | None               = None
        self._anim_index:    int                      = -1
        self._channel_index: int                      = -1
        self._gltf_ref: GLTF2 | None                 = None
        self._on_modified_cb: callable | None         = None
        self._on_live_cb:     callable | None         = None   # fires during drag
        self._current_target_path_labels: list[str]  = ["X", "Y", "Z", "W"]

        # Drag state
        self._drag_comp:   int   = -1    # component index being dragged
        self._drag_key:    int   = -1    # keyframe index being dragged
        self._drag_orig:   float = 0.0   # value at drag-start (for Escape revert)
        self._drag_start_y: float | None = None   # screen Y at mouse-down
        self._drag_start_scene_y: float = 0.0     # plot-space Y at mouse-down

        # Selection highlight: one selected dot drawn as an extra scatter item
        self._sel_scatter: pg.ScatterPlotItem | None = None
        self._sel_comp:    int   = -1
        self._sel_key:     int   = -1

    # ── Public API ────────────────────────────────────────────────────────────

    def set_modified_callback(self, cb: callable) -> None:
        """Called on mouse-release (commit) with (anim_index, channel_index, values)."""
        self._on_modified_cb = cb

    def set_live_callback(self, cb: callable) -> None:
        """Called on every mouse-move during a drag with the same signature.
        Use this to drive the 3-D viewport live while the user drags."""
        self._on_live_cb = cb

    def load_channel(self, gltf: GLTF2, anim_index: int, channel_index: int,
                     node_name: str, target_path: str, anim_name: str) -> None:
        self._gltf_ref      = gltf
        self._anim_index    = anim_index
        self._channel_index = channel_index
        self._sel_comp      = -1
        self._sel_key       = -1

        times, values = read_channel_data(gltf, anim_index, channel_index)
        self._times  = times
        self._values = values if values.ndim > 1 else values[:, np.newaxis]

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
            # sigClicked handles selection (not editing — editing is via eventFilter drag)
            scatter.sigClicked.connect(self._on_scatter_clicked)
            self._plot.addItem(scatter)
            self._scatter_items.append(scatter)

            visible = self._toggles[i].isChecked() if i < len(self._toggles) else True
            curve.setVisible(visible)
            scatter.setVisible(visible)

        self._playhead = pg.InfiniteLine(
            pos=0.0, angle=90,
            pen=pg.mkPen(color=PALETTE["orange"], width=1,
                         style=Qt.PenStyle.DashLine),
            movable=False,
        )
        self._plot.addItem(self._playhead)
        self._plot.autoRange()

    def set_playhead(self, t: float) -> None:
        if self._playhead is not None:
            self._playhead.setPos(t)

    def refresh_values(self) -> None:
        """Re-read _values into all curves and scatter items (called after table edit)."""
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
        self._plot.clear()
        self._curves.clear()
        self._scatter_items.clear()
        self._playhead    = None
        self._sel_scatter = None

    def clear(self) -> None:
        self._clear_plot()
        self._channel_label.setText("No channel selected — click a bone, then a channel")
        self._times  = None
        self._values = None

    def _toggle_curve(self, component_index: int) -> None:
        visible = self._toggles[component_index].isChecked()
        if component_index < len(self._curves):
            self._curves[component_index].setVisible(visible)
        if component_index < len(self._scatter_items):
            self._scatter_items[component_index].setVisible(visible)

    # ── Selection ─────────────────────────────────────────────────────────────

    def _on_scatter_clicked(self, *args) -> None:
        """Click on a keyframe dot → select it.  No dialog, no edit — just select.
        Dragging (via eventFilter) is what actually edits the value.

        PyQtGraph sigClicked: (scatter, points[, ev]) — accept *args.
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

    # ── Drag-to-edit via event filter ─────────────────────────────────────────

    def eventFilter(self, obj, event) -> bool:  # type: ignore[override]
        """Intercept mouse press/move/release on the plot viewport for drag editing.

        We only handle the drag when a keyframe is selected AND the user presses
        left-button on the plot.  Pan/zoom still works because we only consume
        the event when a drag has actually started on a selected dot.
        """
        from PySide6.QtCore import QEvent
        from PySide6.QtGui  import QMouseEvent

        if obj is not self._plot.viewport():
            return False

        t = event.type()

        if t == QEvent.Type.MouseButtonPress:
            if event.button() == Qt.MouseButton.LeftButton and self._sel_comp >= 0:
                # Check whether the press is close enough to the selected dot
                if self._is_near_selected_dot(event.position()):
                    self._drag_comp   = self._sel_comp
                    self._drag_key    = self._sel_key
                    self._drag_orig   = float(
                        self._values[self._drag_key, self._drag_comp])
                    self._drag_start_y = float(event.position().y())
                    self._drag_start_scene_y = self._drag_orig
                    return True   # consume — prevent pan
            return False

        if t == QEvent.Type.MouseMove:
            if self._drag_start_y is not None and self._values is not None:
                dy_px = float(event.position().y()) - self._drag_start_y
                # Map pixel delta to value delta using the current plot Y range
                view_rect = self._plot.viewRect()
                h_px = max(1, self._plot.viewport().height())
                px_per_unit = h_px / max(1e-9, view_rect.height())
                # Dragging UP (negative dy_px) should increase value
                delta_val = -dy_px / px_per_unit
                new_val   = np.float32(self._drag_start_scene_y + delta_val)
                self._values[self._drag_key, self._drag_comp] = new_val
                self._update_curve_display(self._drag_comp)
                self._refresh_selection_dot()
                # Update tip label live
                lbl = (self._current_target_path_labels[self._drag_comp]
                       if self._drag_comp < len(self._current_target_path_labels)
                       else f"V{self._drag_comp}")
                self._edit_tip.setText(
                    f"Editing  t={float(self._times[self._drag_key]):.4f}s  "
                    f"{lbl}={float(new_val):.5f}  ·  Release to commit  ·  Esc to revert"
                )
                # Fire live callback so the viewport updates
                if self._on_live_cb is not None:
                    self._on_live_cb(
                        self._anim_index, self._channel_index, self._values)
                return True
            return False

        if t == QEvent.Type.MouseButtonRelease:
            if self._drag_start_y is not None:
                # Commit the edit to the glTF blob
                if self._on_modified_cb is not None and self._values is not None:
                    self._on_modified_cb(
                        self._anim_index, self._channel_index, self._values)
                self._drag_start_y = None
                self._drag_comp    = -1
                self._drag_key     = -1
                self._edit_tip.setStyleSheet(
                    f"color: {PALETTE['accent']}; font-size: 10px;"
                    "font-style: italic; padding-right: 8px;"
                )
                return True
            return False

        if t == QEvent.Type.KeyPress:
            if (event.key() == Qt.Key.Key_Escape
                    and self._drag_start_y is not None
                    and self._values is not None):
                # Revert to the value at drag-start
                self._values[self._drag_key, self._drag_comp] = np.float32(
                    self._drag_orig)
                self._update_curve_display(self._drag_comp)
                self._refresh_selection_dot()
                if self._on_live_cb is not None:
                    self._on_live_cb(
                        self._anim_index, self._channel_index, self._values)
                self._drag_start_y = None
                self._edit_tip.setText(
                    "Drag reverted  ·  Click a dot to select  ·  Drag up/down to edit"
                )
                return True
            return False

        return False

    def _is_near_selected_dot(self, screen_pos) -> bool:
        """Return True if screen_pos is within 20 px of the selected dot.

        Coordinate chain:
          data coords  →  ViewBox.mapViewToScene()  →  scene coords
          scene coords →  PlotWidget.mapFromScene() →  viewport pixel coords

        self._plot is a PlotWidget (a QGraphicsView subclass), so it has
        mapFromScene(). The .viewport() child is a plain QWidget and does NOT
        have that method — that was the original crash.
        """
        if self._values is None or self._times is None:
            return False
        if self._sel_key >= len(self._times):
            return False
        if self._sel_comp >= self._values.shape[1]:
            return False

        from PySide6.QtCore import QPointF
        vb       = self._plot.getPlotItem().getViewBox()
        scene_pt = vb.mapViewToScene(pg.Point(
            float(self._times[self._sel_key]),
            float(self._values[self._sel_key, self._sel_comp])
        ))
        # mapFromScene lives on the QGraphicsView (PlotWidget), not on viewport()
        widget_pt = self._plot.mapFromScene(QPointF(scene_pt.x(), scene_pt.y()))

        dx = screen_pos.x() - widget_pt.x()
        dy = screen_pos.y() - widget_pt.y()
        return math.hypot(dx, dy) < 20.0

    def _update_curve_display(self, comp: int) -> None:
        """Redraw one curve and its scatter after a value change."""
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
    """Spreadsheet view: exact numeric values per keyframe for the selected channel.

    Value cells (columns 1-4) are directly editable — double-click a cell to
    type a new number.  On commit (Enter / Tab / click away) the value is
    validated, written into the shared _values array, and the modified_callback
    is fired so the curve editor and 3-D viewport update immediately.
    Time column (column 0) is read-only.
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
        # Allow double-click to open an editor on value columns (not Time)
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

        # Shared data references (set by load_channel)
        self._times:   np.ndarray | None = None
        self._values:  np.ndarray | None = None
        self._n_comp:  int               = 0
        self._target_path: str           = ""
        self._loading: bool              = False   # suppress itemChanged during fill
        self._modified_cb: callable | None = None
        self._live_cb:     callable | None = None
        self._anim_index:    int = -1
        self._channel_index: int = -1

    def set_modified_callback(self, cb: callable) -> None:
        self._modified_cb = cb

    def set_live_callback(self, cb: callable) -> None:
        self._live_cb = cb

    def set_channel_indices(self, anim_index: int, channel_index: int) -> None:
        self._anim_index    = anim_index
        self._channel_index = channel_index

    def load_channel(self, times: np.ndarray, values: np.ndarray,
                     target_path: str) -> None:
        self._loading = True
        try:
            if values.ndim == 1:
                values = values[:, np.newaxis]
            self._times       = times
            self._values      = values   # shared array — edits go directly here
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
        n_keys = len(times)
        for row in range(n_keys):
            # Time — read-only
            t_item = QTableWidgetItem(f"{times[row]:.4f}")
            t_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            t_item.setForeground(QColor(PALETTE["muted"]))
            t_item.setFlags(t_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(row, 0, t_item)

            # Value cells — editable
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
            for col in range(n_comp, 4):
                placeholder = QTableWidgetItem("—")
                placeholder.setFlags(placeholder.flags() & ~Qt.ItemFlag.ItemIsEditable)
                placeholder.setForeground(QColor(PALETTE["muted"]))
                self._table.setItem(row, col + 1, placeholder)

    def _update_stats(self, times, values, n_comp, labels) -> None:
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
        """Repopulate cells from the shared _values array (called after drag edit)."""
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
        """Validate and commit a cell edit from the table."""
        if self._loading or self._values is None or self._times is None:
            return
        row = item.row()
        col = item.column()
        if col == 0 or col > self._n_comp:
            return   # time column or empty placeholder — ignore
        comp = col - 1
        try:
            new_val = float(item.text().replace(",", "."))
        except ValueError:
            # Invalid text — revert the cell to the current value
            self._loading = True
            item.setText(f"{float(self._values[row, comp]):.6f}")
            self._loading = False
            return

        self._values[row, comp] = np.float32(new_val)

        # Recolour the cell
        self._loading = True
        if new_val < 0:
            item.setForeground(QColor(PALETTE["orange"]))
        else:
            item.setForeground(QColor(PALETTE["text"]))
        self._loading = False

        # Fire live first (updates curve editor and viewport immediately)
        if self._live_cb is not None:
            self._live_cb(self._anim_index, self._channel_index, self._values)
        # Then commit to glTF blob
        if self._modified_cb is not None:
            self._modified_cb(self._anim_index, self._channel_index, self._values)

        labels = COMPONENT_LABELS.get(self._target_path, ["V0", "V1", "V2", "V3"])
        self._update_stats(self._times, self._values, self._n_comp, labels)

    def highlight_row(self, keyframe_index: int) -> None:
        if 0 <= keyframe_index < self._table.rowCount():
            self._table.selectRow(keyframe_index)
            self._table.scrollToItem(self._table.item(keyframe_index, 0))

    def clear(self) -> None:
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
    """Left-side channel list for the node selected in the scene tree."""

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
        self._on_select_cb = cb

    def load_channels(self, channels: list[tuple[int, int, str, str]]) -> None:
        self._tree.clear()
        anim_groups: dict[str, list[tuple[int, int, str, str]]] = {}
        for entry in channels:
            anim_groups.setdefault(entry[2], []).append(entry)
        for anim_name, entries in anim_groups.items():
            group = QTreeWidgetItem([f"▸ {anim_name}"])
            group.setForeground(0, QColor(PALETTE["accent"]))
            group.setFlags(group.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            self._tree.addTopLevelItem(group)
            for anim_idx, ch_idx, _, path in entries:
                icon  = {"translation": "→", "rotation": "↻",
                         "scale": "⊡", "weights": "≈"}.get(path, "·")
                child = QTreeWidgetItem([f"  {icon}  {path}"])
                child.setData(0, Qt.ItemDataRole.UserRole, (anim_idx, ch_idx, anim_name, path))
                child.setForeground(0, QColor(PALETTE["text"]))
                group.addChild(child)
            group.setExpanded(True)

    def clear(self) -> None:
        self._tree.clear()

    def _on_item_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if data and self._on_select_cb:
            self._on_select_cb(*data)


# =============================================================================
# STEP 11 — Transport / playback controls
# =============================================================================

class TransportWidget(QWidget):
    """Bottom transport bar: play/pause, stop, loop, speed, timeline scrubber."""

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

        self._play_btn = QPushButton("▶")
        self._play_btn.setCheckable(True)
        self._play_btn.setFixedWidth(36)
        self._play_btn.setToolTip("Play / Pause  (Space)")
        layout.addWidget(self._play_btn)

        self._stop_btn = QPushButton("■")
        self._stop_btn.setFixedWidth(36)
        self._stop_btn.setToolTip("Stop — rewind to t=0")
        layout.addWidget(self._stop_btn)

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

        self._scrubber = QSlider(Qt.Orientation.Horizontal)
        self._scrubber.setRange(0, 10000)
        self._scrubber.setValue(0)
        layout.addWidget(self._scrubber, stretch=1)

        self._time_display = QLabel("0.000 s")
        self._time_display.setObjectName("time_display")
        layout.addWidget(self._time_display)

        self.play_btn    = self._play_btn
        self.stop_btn    = self._stop_btn
        self.loop_btn    = self._loop_btn
        self.speed_slider = self._speed_slider
        self.scrubber    = self._scrubber

    def set_duration(self, duration: float) -> None:
        self._scrubber.setValue(0)
        self._duration = duration

    def update_time(self, t: float, duration: float) -> None:
        self._time_display.setText(f"{t:.3f} s")
        if duration > 0:
            self._scrubber.blockSignals(True)
            self._scrubber.setValue(int(t / duration * 10000))
            self._scrubber.blockSignals(False)


# =============================================================================
# STEP 12 — Main application window
# =============================================================================

class GltfCurveEditor(QMainWindow):
    """
    Main window — glTF Animation Curve Editor.

    Layout:
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
        # Window is shown maximized via showMaximized() in main().
        # Set a sensible minimum so the layout doesn't collapse if restored.
        self.setMinimumSize(1100, 700)

        self._gltf:                 GLTF2 | None = None
        self._gltf_path:            Path  | None = None
        self._current_anim_index:   int          = -1
        self._current_channel_index: int         = -1
        self._current_duration:     float        = 0.0
        self._playback_time:        float        = 0.0
        self._is_playing:           bool         = False
        self._dirty:                bool         = False
        self._help_visible:         bool         = False
        self._skel_btn:             QPushButton  = QPushButton()   # replaced in _build_toolbar

        self._build_toolbar()
        self._build_central()
        self._build_transport()
        self._build_status_bar()
        self._build_timer()

        self.setStyleSheet(APP_STYLESHEET)
        self._show_empty_state()
        # Keep help hidden by default so the 3-D viewport is immediately visible
        self._help_panel.setVisible(False)
        self._help_btn.setChecked(False)
        self._help_btn.setText("? Help")

    # -------------------------------------------------------------------------
    # UI construction
    # -------------------------------------------------------------------------

    def _build_toolbar(self) -> None:
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

        # ── Actions ───────────────────────────────────────────────────────────
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

        # ── Skeleton toggle ───────────────────────────────────────────────────
        skel_btn = QPushButton("🦴 Skeleton")
        skel_btn.setCheckable(True)
        skel_btn.setChecked(True)
        skel_btn.setToolTip("Toggle skeleton overlay  (K)")
        skel_btn.clicked.connect(self._toggle_skeleton)
        toolbar.addWidget(skel_btn)
        self._skel_btn = skel_btn

        # Keyboard shortcuts
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
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        h_split = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(h_split, stretch=1)
        self._h_splitter = h_split

        # ── Left panel: scene tree + channel list ─────────────────────────────
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

        # ── Center: viewport + help + curve editor stacked vertically ─────────
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)

        # Help panel (collapsible)
        self._help_panel = HelpPanel()
        center_layout.addWidget(self._help_panel)

        # 3-D viewport
        self._viewport = ModelViewportWidget()
        self._viewport.bone_selected.connect(self._on_bone_selected_in_viewport)
        center_layout.addWidget(self._viewport, stretch=2)

        # Thin divider
        vp_div = QFrame(); vp_div.setObjectName("divider")
        vp_div.setFrameShape(QFrame.Shape.HLine)
        center_layout.addWidget(vp_div)

        # Curve editor below the viewport
        self._curve_editor = CurveEditorWidget()
        self._curve_editor.set_modified_callback(self._on_keyframe_modified)
        self._curve_editor.set_live_callback(self._on_keyframe_live)
        center_layout.addWidget(self._curve_editor, stretch=3)

        h_split.addWidget(center_widget)

        # ── Right panel: keyframe inspector ───────────────────────────────────
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

        h_split.setStretchFactor(0, 18)
        h_split.setStretchFactor(1, 62)
        h_split.setStretchFactor(2, 20)

    def _build_transport(self) -> None:
        self._transport = TransportWidget()
        self.centralWidget().layout().addWidget(self._transport)
        self._transport.play_btn.clicked.connect(self._toggle_playback)
        self._transport.stop_btn.clicked.connect(self._stop_playback)
        self._transport.scrubber.sliderMoved.connect(self._on_scrubber_moved)

    def _build_status_bar(self) -> None:
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage(
            "Step 1: Open a .glb file (Ctrl+O or blue button)  ·  "
            "Step 2: Click a bone in the Scene Tree (or click a joint dot in the viewport)  ·  "
            "Step 3: Click a channel (rotation / translation / scale)  ·  "
            "Step 4: Press Space to play  ·  🦴 Skeleton button or K to toggle skeleton"
        )

    def _build_timer(self) -> None:
        self._timer = QTimer(self)
        self._timer.setInterval(PLAYBACK_INTERVAL_MS)
        self._timer.timeout.connect(self._on_tick)

    # -------------------------------------------------------------------------
    # Empty / help state
    # -------------------------------------------------------------------------

    def _show_empty_state(self) -> None:
        self._scene_tree.clear()
        self._channel_list.clear()
        self._curve_editor.clear()
        self._inspector.clear()

    def _toggle_help(self) -> None:
        self._help_visible = not self._help_visible
        self._help_panel.setVisible(self._help_visible)
        self._help_btn.setChecked(self._help_visible)
        self._help_btn.setText("✕ Help" if self._help_visible else "? Help")

    def _toggle_skeleton(self) -> None:
        """Toggle the skeleton overlay in the 3-D viewport (toolbar button or K)."""
        visible = self._skel_btn.isChecked()
        self._viewport.set_skeleton_visible(visible)
        self._skel_btn.setText("🦴 Skeleton" if visible else "○ Skeleton")

    def _on_bone_selected_in_viewport(self, node_index: int) -> None:
        """Called when the user clicks a joint dot in the 3-D viewport.

        Finds the matching item in the Scene Tree and selects it, which also
        fires _on_node_selected so the channel list and curve editor update.
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

        self._gltf       = gltf
        self._gltf_path  = path
        self._dirty      = False
        self._file_label.setText(f"  {path.name}")

        # Load into 3-D viewport
        first_anim = 0 if (gltf.animations and len(gltf.animations) > 0) else -1
        self._viewport.load_model(gltf, anim_index=first_anim)
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
        self._scene_tree.clear()
        if self._gltf is None:
            return
        skin_joint_set: set[int] = set()
        for skin in (self._gltf.skins or []):
            skin_joint_set.update(skin.joints or [])
        for scene in (self._gltf.scenes or []):
            for root_idx in (scene.nodes or []):
                item = build_scene_tree(self._gltf, None, root_idx, skin_joint_set)
                self._scene_tree.addTopLevelItem(item)
        self._scene_tree.expandAll()

    # -------------------------------------------------------------------------
    # Auto-selection helper
    # -------------------------------------------------------------------------

    def _auto_select_first_animated_node(self) -> None:
        """Find the first node that has animation channels and select it.

        This runs after loading so the user immediately sees curves and the
        keyframe table filled in, without having to click through the tree.
        """
        if self._gltf is None:
            return

        # Find the node index of the first animated channel.
        first_node_idx: int | None = None
        for anim in (self._gltf.animations or []):
            for channel in (anim.channels or []):
                if channel.target.node is not None:
                    first_node_idx = channel.target.node
                    break
            if first_node_idx is not None:
                break

        if first_node_idx is None:
            return  # No animated nodes at all

        # Walk the QTreeWidget looking for the item whose UserRole == first_node_idx.
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
        node_index: int | None = item.data(0, Qt.ItemDataRole.UserRole)
        if node_index is None or self._gltf is None:
            return
        # Highlight the matching joint in the 3-D viewport
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
            # Auto-select the first channel so the curve editor fills immediately.
            self._on_channel_selected(*channels[0])

    def _on_channel_selected(self, anim_index: int, channel_index: int,
                              anim_name: str, target_path: str) -> None:
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

    def _on_keyframe_live(self, anim_index: int, channel_index: int,
                          new_values: np.ndarray) -> None:
        """Called on every mouse-move during a curve-editor drag or a table cell edit.

        Writes the new values straight into the glTF blob so the viewport's
        set_playback_time() re-samples the animation with the updated data.
        Does NOT mark the file dirty — that only happens on final commit.
        """
        if self._gltf is None:
            return
        write_channel_values(self._gltf, anim_index, channel_index, new_values)
        self._curve_editor.refresh_values()
        self._viewport.set_playback_time(self._playback_time)

    def _on_keyframe_modified(self, anim_index: int, channel_index: int,
                               new_values: np.ndarray) -> None:
        if self._gltf is None:
            return
        write_channel_values(self._gltf, anim_index, channel_index, new_values)
        self._dirty = True
        times, values = read_channel_data(self._gltf, anim_index, channel_index)
        if values.ndim == 1:
            values = values[:, np.newaxis]
        channel = self._gltf.animations[anim_index].channels[channel_index]
        self._inspector.set_channel_indices(anim_index, channel_index)
        self._inspector.load_channel(times, values, channel.target.path)
        self._curve_editor.refresh_values()
        self._viewport.set_playback_time(self._playback_time)
        self._status.showMessage(
            "✎  Keyframe modified — unsaved changes  ·  Ctrl+S to export the modified .glb"
        )

    # -------------------------------------------------------------------------
    # Playback
    # -------------------------------------------------------------------------

    def _toggle_playback(self) -> None:
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
        self._is_playing = False
        self._timer.stop()
        self._transport.play_btn.setChecked(False)
        self._transport.play_btn.setText("▶")
        self._playback_time = 0.0
        self._transport.update_time(0.0, self._current_duration)
        self._curve_editor.set_playhead(0.0)
        self._viewport.set_playback_time(0.0)

    def _on_tick(self) -> None:
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

        # Drive the 3-D viewport
        self._viewport.set_playback_time(self._playback_time)

        # Highlight nearest keyframe in the inspector table
        if (self._gltf is not None
                and self._current_anim_index >= 0
                and self._current_channel_index >= 0):
            times, _ = read_channel_data(
                self._gltf, self._current_anim_index, self._current_channel_index
            )
            nearest = int(np.argmin(np.abs(times - self._playback_time)))
            self._inspector.highlight_row(nearest)

    def _on_scrubber_moved(self, value: int) -> None:
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
    """Create the QApplication, apply the dark stylesheet, and enter the event loop."""
    # FIX: Set the OpenGL surface format BEFORE creating QApplication.
    # Without this, on macOS the context defaults to 2.1 (no Core Profile),
    # and on some Linux drivers the depth buffer may be missing — both produce
    # a completely blank viewport with no error message.
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setDepthBufferSize(24)
    fmt.setSamples(4)  # 4× MSAA for smoother edges
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setApplicationName("glTF Animation Curve Editor")
    app.setApplicationVersion("2.0.0")
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
