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
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from pygltflib import GLTF2
from PySide6.QtCore import QTimer, Qt
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
    QWidget
)
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

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProj;
uniform mat4 uJoints[64];
uniform bool uSkinned;

out vec3 vNormal;
out vec3 vFragPos;

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
}
"""

_FRAG_SRC = """
#version 330 core
in vec3 vNormal;
in vec3 vFragPos;

uniform vec3 uLightPos;
uniform vec3 uLightColor;
uniform vec3 uObjectColor;
uniform float uAmbient;

out vec4 FragColor;

void main() {
    // Phong shading
    vec3 norm     = normalize(vNormal);
    vec3 lightDir = normalize(uLightPos - vFragPos);
    float diff    = max(dot(norm, lightDir), 0.0);
    vec3 diffuse  = diff * uLightColor;
    vec3 ambient  = uAmbient * uLightColor;
    vec3 result   = (ambient + diffuse) * uObjectColor;
    FragColor     = vec4(result, 1.0);
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

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._gltf:         GLTF2 | None = None
        self._anim_index:   int          = -1
        self._playback_t:   float        = 0.0
        self._mesh_vao:     int          = 0
        self._mesh_vbo:     int          = 0
        self._mesh_ebo:     int          = 0
        self._mesh_count:   int          = 0
        self._skin_index:   int          = -1
        self._prog:         QOpenGLShaderProgram | None = None
        self._line_prog:    QOpenGLShaderProgram | None = None
        self._line_vao:     int          = 0
        self._line_vbo:     int          = 0
        self._orbit_yaw:    float        = 30.0
        self._orbit_pitch:  float        = 15.0
        self._orbit_dist:   float        = 3.0
        self._last_mouse    = None
        self._bbox_center   = np.zeros(3, dtype=np.float32)
        self._bbox_scale    = 1.0
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

        # Mesh shader
        self._prog = QOpenGLShaderProgram(self)
        self._prog.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Vertex,   _VERT_SRC)
        self._prog.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Fragment, _FRAG_SRC)
        self._prog.link()

        # Skeleton line shader
        self._line_prog = QOpenGLShaderProgram(self)
        self._line_prog.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Vertex,   _LINE_VERT)
        self._line_prog.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Fragment, _LINE_FRAG)
        self._line_prog.link()

        # Allocate VAO / VBO placeholders
        vaos = GL.glGenVertexArrays(2)
        self._mesh_vao, self._line_vao = int(vaos[0]), int(vaos[1])
        vbos = GL.glGenBuffers(3)
        self._mesh_vbo = int(vbos[0])
        self._mesh_ebo = int(vbos[1])
        self._line_vbo = int(vbos[2])

    def resizeGL(self, w: int, h: int) -> None:
        GL.glViewport(0, 0, w, h)

    def paintGL(self) -> None:
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        if self._gltf is None or self._mesh_count == 0:
            self._draw_empty_hint()
            return
        self._draw_mesh()
        self._draw_skeleton()

    # ------------------------------------------------------------------
    # Mouse orbit
    # ------------------------------------------------------------------

    def mousePressEvent(self, event) -> None:
        self._last_mouse = event.position()

    def mouseMoveEvent(self, event) -> None:
        if self._last_mouse is None:
            return
        dx = event.position().x() - self._last_mouse.x()
        dy = event.position().y() - self._last_mouse.y()
        self._orbit_yaw   += dx * 0.4
        self._orbit_pitch  = max(-89, min(89, self._orbit_pitch + dy * 0.4))
        self._last_mouse   = event.position()
        self.update()

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        self._orbit_dist = max(0.5, self._orbit_dist - delta * 0.005)
        self.update()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _mvp(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (model, view, proj) as 4×4 float32 ndarrays."""
        w, h   = max(1, self.width()), max(1, self.height())
        aspect = w / h

        # Projection (perspective)
        fov    = math.radians(45.0)
        near, far = 0.01, 200.0
        f      = 1.0 / math.tan(fov / 2)
        proj   = np.zeros((4, 4), dtype=np.float32)
        proj[0, 0] = f / aspect
        proj[1, 1] = f
        proj[2, 2] = (far + near) / (near - far)
        proj[2, 3] = -1.0
        proj[3, 2] = (2 * far * near) / (near - far)

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
        """Extract position / normal / joints / weights from the first mesh primitive."""
        gltf = self._gltf
        if not gltf or not gltf.meshes:
            return

        mesh      = gltf.meshes[0]
        prim      = mesh.primitives[0]
        attrs     = prim.attributes

        # Positions (required)
        pos_idx   = getattr(attrs, "POSITION", None)
        if pos_idx is None:
            return
        positions = read_accessor(gltf, pos_idx).astype(np.float32)   # (N,3)
        if positions.ndim == 1:
            positions = positions.reshape(-1, 3)

        # Normals (compute flat normals if absent)
        norm_idx  = getattr(attrs, "NORMAL", None)
        if norm_idx is not None:
            normals = read_accessor(gltf, norm_idx).astype(np.float32)
            if normals.ndim == 1:
                normals = normals.reshape(-1, 3)
        else:
            normals = np.zeros_like(positions)
            normals[:, 2] = 1.0

        # Joints / weights for skinning
        j0_idx  = getattr(attrs, "JOINTS_0",  None)
        w0_idx  = getattr(attrs, "WEIGHTS_0", None)
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

        # Normalise mesh to unit-box centred at origin
        mn, mx         = positions.min(axis=0), positions.max(axis=0)
        self._bbox_center = ((mn + mx) / 2.0).astype(np.float32)
        extent         = float(np.max(mx - mn)) or 1.0
        self._bbox_scale = 1.0 / extent
        self._orbit_dist = 2.0 / self._bbox_scale * 0.6  # auto-fit

        # Interleave: pos(3) | normal(3) | joints(4) | weights(4) = 14 floats / vertex
        n_verts   = len(positions)
        interleaved = np.zeros((n_verts, 14), dtype=np.float32)
        interleaved[:, 0:3]  = positions
        interleaved[:, 3:6]  = normals
        interleaved[:, 6:10] = joints
        interleaved[:, 10:14] = weights
        vertex_data = interleaved.flatten()

        # Upload to GPU
        stride = 14 * 4   # bytes

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

        GL.glBindVertexArray(0)

        # Pick skin index for this mesh
        self._skin_index = -1
        node_idx = None
        for ni, node in enumerate(gltf.nodes or []):
            if node.mesh == 0 and node.skin is not None:
                self._skin_index = node.skin
                node_idx = ni
                break

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

        # Light slightly above and to the right
        light = self._bbox_center + np.array([2, 4, 3], dtype=np.float32)
        GL.glUniform3f(_u("uLightPos"),    *light)
        GL.glUniform3f(_u("uLightColor"),  1.0, 1.0, 1.0)
        GL.glUniform3f(_u("uObjectColor"), 0.55, 0.65, 0.75)
        GL.glUniform1f(_u("uAmbient"),     0.35)
        GL.glUniform1i(_u("uSkinned"), 1 if self._is_skinned and self._skin_index >= 0 else 0)

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
                pass

        GL.glBindVertexArray(self._mesh_vao)
        GL.glDrawElements(GL.GL_TRIANGLES, self._mesh_count,
                          GL.GL_UNSIGNED_INT, None)
        GL.glBindVertexArray(0)
        self._prog.release()

    def _draw_skeleton(self) -> None:
        """Draw parent→child bone lines in accent colour."""
        if self._line_prog is None or self._gltf is None:
            return
        gltf = self._gltf
        try:
            world_pos = compute_world_positions(gltf, self._anim_index, self._playback_t)
        except Exception:
            return

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
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._line_vbo)
        GL.glBufferSubData(GL.GL_ARRAY_BUFFER, 0,
                           min(arr.nbytes, 256 * 2 * 3 * 4), arr)

        model, view, proj = self._mvp()
        mvp = proj @ view @ model

        self._line_prog.bind()
        loc_mvp   = GL.glGetUniformLocation(self._line_prog.programId(), "uMVP")
        loc_color = GL.glGetUniformLocation(self._line_prog.programId(), "uColor")
        GL.glUniformMatrix4fv(loc_mvp, 1, GL.GL_TRUE, mvp.flatten())
        GL.glUniform3f(loc_color, 0.310, 0.612, 0.976)   # accent blue
        GL.glLineWidth(2.0)

        GL.glBindVertexArray(self._line_vao)
        n_verts = min(len(segments) // 3, 256 * 2)
        GL.glDrawArrays(GL.GL_LINES, 0, n_verts)
        GL.glBindVertexArray(0)
        self._line_prog.release()

    def _draw_empty_hint(self) -> None:
        """When no model is loaded, the viewport just stays dark (hint in status bar)."""
        pass


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
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(6)

        title = QLabel("How to use this demo")
        title.setStyleSheet(
            f"color: {PALETTE['accent']}; font-weight: 700; font-size: 13px;"
        )
        layout.addWidget(title)

        steps = [
            ("1  Open a model",
             "Ctrl+O or the blue button — load any .glb file that contains animations."),
            ("2  Browse the scene",
             "Click a node in the Scene Tree (left). Bone nodes 🦴 hold most channels."),
            ("3  Inspect curves",
             "Click a channel (translation / rotation / scale) to draw its F-curves.\n"
             "X=red  Y=green  Z=blue  W=yellow.  Toggle components with the checkboxes."),
            ("4  Play & edit",
             "Space or ▶ to play — the 3-D model animates in the viewport above.\n"
             "Click a dot on any curve to edit that keyframe value directly.\n"
             "Ctrl+S exports the modified .glb."),
            ("Viewport controls",
             "Drag to orbit · Scroll to zoom."),
        ]

        for hdr, body in steps:
            h = QLabel(hdr)
            h.setStyleSheet(
                f"color: {PALETTE['text']}; font-weight: 600; font-size: 12px; "
                "margin-top: 4px;"
            )
            layout.addWidget(h)
            b = QLabel(body)
            b.setStyleSheet(
                f"color: {PALETTE['muted']}; font-size: 11px; padding-left: 12px;"
            )
            b.setWordWrap(True)
            layout.addWidget(b)

        div = QFrame()
        div.setFrameShape(QFrame.Shape.HLine)
        div.setStyleSheet(f"color: {PALETTE['border']};")
        layout.addWidget(div)

        close_lbl = QLabel("Press ? or click  ✕ Help  to hide this panel.")
        close_lbl.setStyleSheet(
            f"color: {PALETTE['muted']}; font-size: 10px; font-style: italic;"
        )
        layout.addWidget(close_lbl)


# =============================================================================
# STEP 8 — Curve editor widget (PyQtGraph)
# =============================================================================

class CurveEditorWidget(QWidget):
    """
    PyQtGraph-backed animation curve editor panel.

    THREE.JS CONCEPT: This is the Python equivalent of visualising Three.js's
    AnimationMixer internals.  In Three.js you call mixer.clipAction(clip).play()
    and the engine interpolates curves invisibly.  Here we expose those same
    curves (time → value per component) so you can see and edit them.

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

        # Header
        header = QWidget()
        header.setStyleSheet(f"background-color: {PALETTE['surface']};")
        hl = QHBoxLayout(header)
        hl.setContentsMargins(12, 6, 12, 6)
        self._channel_label = QLabel("No node selected")
        self._channel_label.setStyleSheet(f"color: {PALETTE['muted']}; font-size: 12px;")
        hl.addWidget(self._channel_label)
        hl.addStretch()

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
        layout.addWidget(self._plot, stretch=1)

        self._curves: list[pg.PlotDataItem]     = []
        self._scatter_items: list[pg.ScatterPlotItem] = []
        self._playhead: pg.InfiniteLine | None   = None
        self._times:  np.ndarray | None          = None
        self._values: np.ndarray | None          = None
        self._anim_index:    int                 = -1
        self._channel_index: int                 = -1
        self._gltf_ref: GLTF2 | None             = None
        self._on_modified_cb: callable | None    = None
        self._current_target_path_labels: list[str] = ["X", "Y", "Z", "W"]

    def set_modified_callback(self, cb: callable) -> None:
        self._on_modified_cb = cb

    def load_channel(self, gltf: GLTF2, anim_index: int, channel_index: int,
                     node_name: str, target_path: str, anim_name: str) -> None:
        self._gltf_ref      = gltf
        self._anim_index    = anim_index
        self._channel_index = channel_index

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
                size=9,
                pen=pg.mkPen(color=color[:3], width=1),
                brush=pg.mkBrush(color=(*color[:3], 180)),
                hoverable=True,
                hoverPen=pg.mkPen("w", width=2),
                hoverBrush=pg.mkBrush("w"),
                tip=lambda x, y, data: f"t={x:.3f}  v={y:.4f}",
            )
            scatter.setData(
                x=times, y=self._values[:, i],
                data=[(i, j) for j in range(len(times))]
            )
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

    def _clear_plot(self) -> None:
        self._plot.clear()
        self._curves.clear()
        self._scatter_items.clear()
        self._playhead = None

    def clear(self) -> None:
        self._clear_plot()
        self._channel_label.setText("No node selected")
        self._times  = None
        self._values = None

    def _toggle_curve(self, component_index: int) -> None:
        visible = self._toggles[component_index].isChecked()
        if component_index < len(self._curves):
            self._curves[component_index].setVisible(visible)
        if component_index < len(self._scatter_items):
            self._scatter_items[component_index].setVisible(visible)

    def _on_scatter_clicked(self, scatter: pg.ScatterPlotItem,
                            points: list) -> None:
        from PySide6.QtWidgets import QInputDialog
        if not points or self._values is None or self._gltf_ref is None:
            return
        point                    = points[0]
        component_index, keyframe_index = point.data()
        original                 = float(self._values[keyframe_index, component_index])
        t                        = float(self._times[keyframe_index])
        comp_label               = self._current_target_path_labels
        label                    = (comp_label[component_index]
                                    if component_index < len(comp_label)
                                    else f"V{component_index}")
        new_val, ok = QInputDialog.getDouble(
            self,
            "Edit Keyframe",
            f"t = {t:.4f} s   |   component {label}\n\nNew value:",
            value=original, min=-1e6, max=1e6, decimals=6,
        )
        if not ok:
            return
        self._values[keyframe_index, component_index] = np.float32(new_val)
        if self._on_modified_cb:
            self._on_modified_cb(self._anim_index, self._channel_index,
                                 self._values)
        comp = component_index
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
    """Spreadsheet view: exact numeric values per keyframe for the selected channel."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        hdr = QLabel("KEYFRAME TABLE")
        hdr.setObjectName("section_header")
        layout.addWidget(hdr)

        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(["Time", "X / W0", "Y / W1", "Z / W2", "W / W3"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.verticalHeader().setDefaultSectionSize(22)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        layout.addWidget(self._table, stretch=1)

        self._stats_label = QLabel("")
        self._stats_label.setStyleSheet(
            f"color: {PALETTE['muted']}; font-size: 11px;"
            "font-family: 'IBM Plex Mono', monospace; padding: 4px 0;"
        )
        self._stats_label.setWordWrap(True)
        layout.addWidget(self._stats_label)

    def load_channel(self, times: np.ndarray, values: np.ndarray,
                     target_path: str) -> None:
        if values.ndim == 1:
            values = values[:, np.newaxis]
        n_keys, n_comp = values.shape
        self._table.setRowCount(n_keys)
        labels     = COMPONENT_LABELS.get(target_path, ["V0", "V1", "V2", "V3"])
        col_headers = ["Time"] + labels[:n_comp]
        while len(col_headers) < 5:
            col_headers.append("—")
        self._table.setHorizontalHeaderLabels(col_headers)

        for row in range(n_keys):
            t_item = QTableWidgetItem(f"{times[row]:.4f}")
            t_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            t_item.setForeground(QColor(PALETTE["muted"]))
            self._table.setItem(row, 0, t_item)
            for col in range(n_comp):
                item = QTableWidgetItem(f"{values[row, col]:.5f}")
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                )
                if values[row, col] < 0:
                    item.setForeground(QColor(PALETTE["orange"]))
                self._table.setItem(row, col + 1, item)
            for col in range(n_comp, 4):
                self._table.setItem(row, col + 1, QTableWidgetItem(""))

        stats_parts = []
        for col in range(n_comp):
            lbl  = labels[col] if col < len(labels) else f"V{col}"
            lo, hi = float(values[:, col].min()), float(values[:, col].max())
            stats_parts.append(f"{lbl}: [{lo:.3f} → {hi:.3f}]")
        duration = float(times[-1]) if n_keys > 0 else 0.0
        self._stats_label.setText(
            f"{n_keys} keyframes  ·  {duration:.3f}s\n" + "  ".join(stats_parts)
        )

    def highlight_row(self, keyframe_index: int) -> None:
        if 0 <= keyframe_index < self._table.rowCount():
            self._table.selectRow(keyframe_index)
            self._table.scrollToItem(self._table.item(keyframe_index, 0))

    def clear(self) -> None:
        self._table.setRowCount(0)
        self._stats_label.setText("")


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

    PIPELINE CONCEPT: This is a minimal professional animation pipeline tool.
    Real equivalents: Maya Graph Editor + Outliner, Blender NLA + Dope Sheet,
    MotionBuilder FCurve Editor.  All share the same layout: hierarchy on the
    left, curve editor centre, inspector right, transport at the bottom.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("glTF Animation Curve Editor — Computeranimation Seminar 04")
        self.resize(1500, 920)

        self._gltf:                 GLTF2 | None = None
        self._gltf_path:            Path  | None = None
        self._current_anim_index:   int          = -1
        self._current_channel_index: int         = -1
        self._current_duration:     float        = 0.0
        self._playback_time:        float        = 0.0
        self._is_playing:           bool         = False
        self._dirty:                bool         = False
        self._help_visible:         bool         = True

        self._build_toolbar()
        self._build_central()
        self._build_transport()
        self._build_status_bar()
        self._build_timer()

        self.setStyleSheet(APP_STYLESHEET)
        self._show_empty_state()

    # -------------------------------------------------------------------------
    # UI construction
    # -------------------------------------------------------------------------

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        # ── Logo + author (matches 03_ca_hello_pyside6 template) ──────────────
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

        # Keyboard shortcuts
        for shortcut, slot in [("Ctrl+O", self._on_open),
                                ("Ctrl+S", self._on_export),
                                ("Space",  self._toggle_playback),
                                ("?",      self._toggle_help)]:
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
        center_layout.addWidget(self._viewport, stretch=2)

        # Thin divider
        vp_div = QFrame(); vp_div.setObjectName("divider")
        vp_div.setFrameShape(QFrame.Shape.HLine)
        center_layout.addWidget(vp_div)

        # Curve editor below the viewport
        self._curve_editor = CurveEditorWidget()
        self._curve_editor.set_modified_callback(self._on_keyframe_modified)
        center_layout.addWidget(self._curve_editor, stretch=3)

        h_split.addWidget(center_widget)

        # ── Right panel: keyframe inspector ───────────────────────────────────
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setMinimumWidth(220)
        right_scroll.setMaximumWidth(340)
        right_scroll.setStyleSheet(f"border: none; background: {PALETTE['bg']};")

        self._inspector = KeyframeInspectorWidget()
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
            "Open a .glb file to begin — Ctrl+O  ·  "
            "Then click a node → pick a channel → Space to play  ·  "
            "Drag viewport to orbit, scroll to zoom"
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
            f"Click a node in the Scene Tree to inspect its channels  ·  "
            f"Drag viewport to orbit"
        )

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
    # Selection handlers
    # -------------------------------------------------------------------------

    def _on_node_selected(self, item: QTreeWidgetItem, _column: int) -> None:
        node_index: int | None = item.data(0, Qt.ItemDataRole.UserRole)
        if node_index is None or self._gltf is None:
            return
        channels = get_animation_channels_for_node(self._gltf, node_index)
        self._channel_list.load_channels(channels)
        if not channels:
            self._curve_editor.clear()
            self._inspector.clear()
            self._status.showMessage(
                f"Node '{get_node_name(self._gltf, node_index)}' has no animation channels."
            )
        else:
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
        self._inspector.load_channel(times, values, target_path)

        duration = float(times[-1]) if len(times) > 0 else 0.0
        self._current_duration = duration
        self._transport.set_duration(duration)
        self._playback_time = 0.0
        self._transport.update_time(0.0, duration)

        self._status.showMessage(
            f"'{node_name}'  ·  {target_path}  ·  {anim_name}  ·  "
            f"{len(times)} keyframes  ·  {duration:.3f}s  ·  "
            f"Space to play  |  Click a keyframe dot to edit its value"
        )

    # -------------------------------------------------------------------------
    # Keyframe modification callback
    # -------------------------------------------------------------------------

    def _on_keyframe_modified(self, anim_index: int, channel_index: int,
                               new_values: np.ndarray) -> None:
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

        # ── Drive the 3-D viewport ──────────────────────────────────────────
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
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setApplicationName("glTF Animation Curve Editor")
    app.setApplicationVersion("2.0.0")
    app.setOrganizationName("TU Ilmenau — AMS Group")

    window = GltfCurveEditor()
    window.show()
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
