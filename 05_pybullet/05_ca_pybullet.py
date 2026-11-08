# =============================================================================
# 05_ca_pybullet.py — Physics-Based Skeleton Animation
# =============================================================================
# Author   : Renato Fera P.
# LinkedIn : https://www.linkedin.com/in/renato-profeta/
# Course   : Computeranimation — Chapter 05: Physics-based Animation
# Institute: Institut für Medientechnik, Technische Universität Ilmenau
# -----------------------------------------------------------------------------
# Description:
"""05_ca_pybullet.py — Physics-Based Skeleton Animation.

A professional real-time 3D skeleton driven by CMU motion capture data.
The human skeleton GLB mesh is rendered using GPU skinning via PyOpenGL.
BVH mocap files retarget onto the GLB rig every frame.

Pipeline:
  BVH file → parse joint rotations → retarget onto the GLB bone hierarchy
  (per-bone rest-pose correction, A-pose rig vs T-pose mocap)
  → compute skinning matrices → GPU vertex transform → OpenGL render

Required files (place in the same directory as this script):
  human-skeleton-rigged.glb  → grab3d.com/free-rigged-3d-models/
  08_01.bvh                  → github.com/una-dinosauria/cmu-mocap  data/008/08_01.bvh   (walk)
  09_01.bvh                  → github.com/una-dinosauria/cmu-mocap  data/009/09_01.bvh   (run)
  60_01.bvh                  → github.com/una-dinosauria/cmu-mocap  data/060/60_01.bvh   (salsa dance)

How to run
----------
First time (creates .venv and installs all dependencies):
    uv sync

Every time after that:
    uv run python 05_ca_pybullet.py

Ubuntu system dependencies (run once if not already installed):
    sudo apt install libgl1 libegl1 libxcb-cursor0 libxcb-xinerama0

Connections to other seminars
------------------------------
- The BVH → joint transform pipeline is the Python equivalent of the
  AnimationMixer / bone skinning from seminar 04.
- GPU skinning (joint matrices uploaded as uniforms) is the same
  technique used in every real-time game engine.
- The fixed-timestep accumulator inside the render loop mirrors the
  pattern used in every browser physics demo (Demos 1-4).
"""
# =============================================================================

from __future__ import annotations

import json
import math
import struct
import sys
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
from OpenGL.GL import (
    GL_AMBIENT,
    GL_AMBIENT_AND_DIFFUSE,
    GL_ARRAY_BUFFER,
    GL_BLEND,
    GL_COLOR_BUFFER_BIT,
    GL_COLOR_MATERIAL,
    GL_CULL_FACE,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_DIFFUSE,
    GL_ELEMENT_ARRAY_BUFFER,
    GL_FLOAT,
    GL_FRONT_AND_BACK,
    GL_LIGHT0,
    GL_LIGHTING,
    GL_LINE_SMOOTH,
    GL_LINES,
    GL_MODELVIEW,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_POSITION,
    GL_PROJECTION,
    GL_SHININESS,
    GL_SMOOTH,
    GL_SPECULAR,
    GL_SRC_ALPHA,
    GL_STATIC_DRAW,
    GL_TRIANGLES,
    GL_UNSIGNED_INT,
    GL_UNSIGNED_SHORT,
    GL_VERTEX_SHADER,
    GL_FRAGMENT_SHADER,
    GL_LINK_STATUS,
    GL_COMPILE_STATUS,
    GL_TRUE,
    glBegin,
    glBlendFunc,
    glClear,
    glClearColor,
    glColorMaterial,
    glDisable,
    glEnable,
    glEnd,
    glLightfv,
    glLineWidth,
    glLoadIdentity,
    glMaterialf,
    glMaterialfv,
    glMatrixMode,
    glNormal3f,
    glPopMatrix,
    glPushMatrix,
    glRotatef,
    glColor4f,
    glShadeModel,
    glTranslatef,
    glVertex3f,
    glViewport,
    glGenBuffers,
    glBindBuffer,
    glBufferData,
    glGenVertexArrays,
    glBindVertexArray,
    glEnableVertexAttribArray,
    glVertexAttribPointer,
    glVertexAttribIPointer,
    glDrawElements,
    glUseProgram,
    glGetUniformLocation,
    glUniformMatrix4fv,
    glUniform3f,
    glUniform1i,
    glCreateShader,
    glShaderSource,
    glCompileShader,
    glGetShaderiv,
    glGetShaderInfoLog,
    glCreateProgram,
    glAttachShader,
    glLinkProgram,
    glGetProgramiv,
    glGetProgramInfoLog,
    GL_INT,
    GL_UNSIGNED_BYTE,
)
from OpenGL.GLU import gluLookAt, gluPerspective
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap, QSurfaceFormat
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDockWidget,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
    QFrame,
    QSizePolicy,
)

# ── Constants ─────────────────────────────────────────────────────────────────

WINDOW_TITLE: str = "Physics-Based Skeleton Animation — Computeranimation 05"
WINDOW_MIN_WIDTH: int = 1280
WINDOW_MIN_HEIGHT: int = 800

SCRIPT_DIR: Path = Path(__file__).resolve().parent
LOGO_PATH: Path = SCRIPT_DIR.parent / "ams_group_tui.png"

# Asset filenames (must be present in SCRIPT_DIR)
SKELETON_GLB: Path = SCRIPT_DIR / "human-skeleton-rigged.glb"
BVH_WALK: Path = SCRIPT_DIR / "08_01.bvh"
BVH_RUN: Path = SCRIPT_DIR / "09_01.bvh"
BVH_DANCE: Path = SCRIPT_DIR / "60_01.bvh"

# Rendering
RENDER_FPS: int = 60

# BVH scale — CMU BVH uses its own unit (~0.056m per unit), GLB positions are in metres
BVH_SCALE: float = 0.056448

# Ground grid
GRID_SIZE: int = 10
GRID_STEP: float = 0.5
GRID_COLOUR: tuple[float, float, float] = (0.18, 0.22, 0.30)

# Camera defaults
CAM_DISTANCE_DEFAULT: float = 3.2
CAM_AZIMUTH_DEFAULT: float = 25.0
CAM_ELEVATION_DEFAULT: float = 10.0
CAM_TARGET_DEFAULT: tuple[float, float, float] = (0.0, 0.9, 0.0)

# GLB bone → CMU BVH joint name mapping
# GLB rig has 22 bones that map directly onto CMU joints.
# LHipJoint / RHipJoint exist in BVH but not in GLB — we skip them.
GLB_TO_BVH: dict[str, str] = {
    "Hips":          "Hips",
    "LeftUpLeg":     "LeftUpLeg",
    "LeftLeg":       "LeftLeg",
    "LeftFoot":      "LeftFoot",
    "LeftToeBase":   "LeftToeBase",
    "RightUpLeg":    "RightUpLeg",
    "RightLeg":      "RightLeg",
    "RightFoot":     "RightFoot",
    "RightToeBase":  "RightToeBase",
    "Spine02":       "LowerBack",
    "Spine01":       "Spine",
    "Spine":         "Spine1",
    "LeftShoulder":  "LeftShoulder",
    "LeftArm":       "LeftArm",
    "LeftForeArm":   "LeftForeArm",
    "LeftHand":      "LeftHand",
    "neck":          "Neck",
    "Head":          "Head",
    "RightShoulder": "RightShoulder",
    "RightArm":      "RightArm",
    "RightForeArm":  "RightForeArm",
    "RightHand":     "RightHand",
}

# ── GLSL Shaders ──────────────────────────────────────────────────────────────

VERT_SRC = """
#version 330 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in uvec4 aJoints;
layout(location = 3) in vec4  aWeights;

uniform mat4 uProjection;
uniform mat4 uView;
uniform mat4 uJointMatrices[24];

out vec3 vNormal;
out vec3 vFragPos;

void main() {
    // GPU skinning: blend up to 4 joint transforms
    mat4 skin =
        aWeights.x * uJointMatrices[aJoints.x] +
        aWeights.y * uJointMatrices[aJoints.y] +
        aWeights.z * uJointMatrices[aJoints.z] +
        aWeights.w * uJointMatrices[aJoints.w];

    vec4 worldPos = skin * vec4(aPos, 1.0);
    vFragPos  = worldPos.xyz;
    // Joint matrices are rotation + translation only, so mat3(skin) is a
    // valid normal matrix (and far cheaper than inverse-transpose on 178k verts).
    vNormal   = mat3(skin) * aNormal;
    gl_Position = uProjection * uView * worldPos;
}
"""

FRAG_SRC = """
#version 330 core

in vec3 vNormal;
in vec3 vFragPos;

uniform vec3 uLightPos;
uniform vec3 uViewPos;

out vec4 FragColor;

void main() {
    // Ivory bone colour
    vec3 boneColor = vec3(0.95, 0.92, 0.85);

    vec3 N = normalize(vNormal);
    vec3 L = normalize(uLightPos - vFragPos);
    vec3 V = normalize(uViewPos  - vFragPos);
    if (dot(N, V) < 0.0) N = -N;          // two-sided: don't depend on winding
    vec3 H = normalize(L + V);

    float ambient  = mix(0.14, 0.32, N.y * 0.5 + 0.5);   // simple sky/ground fill
    float diffuse  = max(dot(N, L), 0.0) * 0.75;
    float specular = pow(max(dot(N, H), 0.0), 60.0) * 0.4;

    vec3 col = boneColor * (ambient + diffuse) + vec3(specular);
    FragColor = vec4(col, 1.0);
}
"""


# ── BVH Parser ────────────────────────────────────────────────────────────────

class BVHJoint(NamedTuple):
    name: str
    parent: int
    offset: np.ndarray
    channels: list[str]
    channel_start: int


class BVHData(NamedTuple):
    joints: list[BVHJoint]
    frames: np.ndarray
    frame_time: float


def parse_bvh(path: Path) -> BVHData:
    """Parse a BVH file into joints and frame data."""
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = [l.strip() for l in text.splitlines()]

    joints: list[BVHJoint] = []
    stack: list[int] = []
    channel_index: int = 0
    i: int = 0

    while i < len(lines):
        line = lines[i]
        if line == "MOTION":
            break
        tokens = line.split()
        if not tokens:
            i += 1
            continue
        keyword = tokens[0].upper()

        if keyword in ("ROOT", "JOINT"):
            name = tokens[1] if len(tokens) > 1 else f"Joint_{len(joints)}"
            parent = stack[-1] if stack else -1
            joints.append(BVHJoint(
                name=name, parent=parent,
                offset=np.zeros(3, dtype=np.float32),
                channels=[], channel_start=channel_index,
            ))
            stack.append(len(joints) - 1)
            i += 1
            continue

        if keyword == "OFFSET" and stack:
            idx = stack[-1]
            offset = np.array([float(tokens[1]), float(tokens[2]), float(tokens[3])],
                               dtype=np.float32) * BVH_SCALE
            joints[idx] = joints[idx]._replace(offset=offset)
            i += 1
            continue

        if keyword == "CHANNELS" and stack:
            idx = stack[-1]
            n = int(tokens[1])
            channels = tokens[2: 2 + n]
            joints[idx] = joints[idx]._replace(
                channels=channels, channel_start=channel_index)
            channel_index += n
            i += 1
            continue

        if keyword == "END":
            i += 1
            depth = 1
            while i < len(lines) and depth > 0:
                if "{" in lines[i]: depth += 1
                if "}" in lines[i]: depth -= 1
                i += 1
            continue

        if line == "}":
            if stack: stack.pop()
            i += 1
            continue

        i += 1

    # MOTION section
    frame_time: float = 1.0 / 120.0
    while i < len(lines):
        line = lines[i]
        tokens = line.split()
        if not tokens:
            i += 1
            continue
        if tokens[0] == "Frame" and len(tokens) > 2 and tokens[1] == "Time:":
            frame_time = float(tokens[2])
        elif len(tokens) >= channel_index > 0:
            break
        i += 1

    frame_lines = []
    while i < len(lines):
        tokens = lines[i].split()
        if len(tokens) >= channel_index > 0:
            frame_lines.append([float(t) for t in tokens[:channel_index]])
        i += 1

    frames = np.array(frame_lines, dtype=np.float32)
    if frames.shape[0] == 0:
        frames = np.zeros((1, channel_index), dtype=np.float32)

    return BVHData(joints=joints, frames=frames, frame_time=frame_time)


def _rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


_AXIS_ROT = {"X": _rot_x, "Y": _rot_y, "Z": _rot_z}


def bvh_pose(bvh: BVHData, frame_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Forward kinematics of the BVH rotations at one frame.

    Returns
    -------
    world_rot : (n_joints, 3, 3)  world-space rotation of every BVH joint
    root_pos  : (3,)              root translation in metres
    """
    data = bvh.frames[frame_idx % len(bvh.frames)]
    n = len(bvh.joints)
    world_rot = np.tile(np.eye(3), (n, 1, 1))
    root_pos = np.zeros(3)

    for j, joint in enumerate(bvh.joints):
        local = np.eye(3)
        for k, ch in enumerate(joint.channels):
            val = float(data[joint.channel_start + k])
            if ch.endswith("rotation"):
                # channels are listed in application order: R = R_first @ R_second @ ...
                local = local @ _AXIS_ROT[ch[0]](math.radians(val))
            elif ch.endswith("position") and joint.parent < 0:
                root_pos["XYZ".index(ch[0])] = val * BVH_SCALE
        world_rot[j] = world_rot[joint.parent] @ local if joint.parent >= 0 else local

    return world_rot, root_pos


def bvh_rest_positions(bvh: BVHData) -> np.ndarray:
    """World position of every joint in the BVH rest pose (all angles zero)."""
    pos = np.zeros((len(bvh.joints), 3))
    for j, joint in enumerate(bvh.joints):
        pos[j] = (pos[joint.parent] if joint.parent >= 0 else 0.0) + joint.offset
    return pos


# ── GLB Loader ────────────────────────────────────────────────────────────────

class GLBMesh(NamedTuple):
    positions:   np.ndarray   # (N,3) float32  vertices in bind-pose world space (metres)
    normals:     np.ndarray   # (N,3) float32
    joints:      np.ndarray   # (N,4) uint8/uint16
    weights:     np.ndarray   # (N,4) float32
    indices:     np.ndarray   # (M,)  uint32
    joint_names: list[str]    # name of each joint slot (order of JOINTS_0)
    parent:      np.ndarray   # (J,)  parent joint slot, -1 for the root
    bind_pos:    np.ndarray   # (J,3) joint position in bind-pose world space
    bind_rot:    np.ndarray   # (J,3,3) joint rotation in bind-pose world space


_COMP_DTYPE = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16,
               5125: np.uint32, 5126: np.float32}
_TYPE_COUNT = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def _acc_data(data: dict, bin_data: bytes, acc_idx: int) -> np.ndarray:
    """Read a glTF accessor into a numpy array (handles interleaved buffers)."""
    acc = data["accessors"][acc_idx]
    bv = data["bufferViews"][acc["bufferView"]]
    offset = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
    count = acc["count"]
    dtype = np.dtype(_COMP_DTYPE[acc["componentType"]])
    n_comp = _TYPE_COUNT[acc["type"]]
    stride = bv.get("byteStride", 0) or dtype.itemsize * n_comp

    # A strided view straight onto the buffer — no Python loop, works for
    # both tightly-packed and interleaved data.
    arr = np.ndarray(shape=(count, n_comp), dtype=dtype, buffer=bin_data,
                     offset=offset, strides=(stride, dtype.itemsize))
    arr = np.ascontiguousarray(arr)
    if acc["type"] == "MAT4":
        arr = arr.reshape(count, 4, 4)
    elif n_comp == 1:
        arr = arr.reshape(count)
    return arr


def _quat_to_mat3(q) -> np.ndarray:
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _read_glb_chunks(path: Path) -> tuple[dict, bytes]:
    raw = path.read_bytes()
    magic, _version, _length = struct.unpack_from("<4sII", raw, 0)
    if magic != b"glTF":
        raise ValueError(f"{path.name} is not a binary glTF (.glb) file")
    pos, js, binary = 12, None, b""
    while pos + 8 <= len(raw):
        clen, ctype = struct.unpack_from("<II", raw, pos)
        chunk = raw[pos + 8: pos + 8 + clen]
        if ctype == 0x4E4F534A:      # 'JSON'
            js = json.loads(chunk)
        elif ctype == 0x004E4942:    # 'BIN\0'
            binary = chunk
        pos += 8 + clen
    if js is None:
        raise ValueError("GLB has no JSON chunk")
    return js, binary


def load_glb(path: Path) -> GLBMesh:
    """Load the skinned mesh and its bind pose from a GLB file."""
    data, bin_data = _read_glb_chunks(path)

    # ── Mesh geometry ──
    prim = data["meshes"][0]["primitives"][0]
    attrs = prim["attributes"]
    positions = _acc_data(data, bin_data, attrs["POSITION"]).astype(np.float32)
    normals = _acc_data(data, bin_data, attrs["NORMAL"]).astype(np.float32)
    joints_arr = _acc_data(data, bin_data, attrs["JOINTS_0"])
    if joints_arr.dtype not in (np.uint8, np.uint16):
        joints_arr = joints_arr.astype(np.uint16)
    weights = _acc_data(data, bin_data, attrs["WEIGHTS_0"]).astype(np.float32)
    indices = _acc_data(data, bin_data, prim["indices"]).astype(np.uint32)

    # ── Node hierarchy → bind-pose world transforms ──
    # NOTE: the world transform includes *every* ancestor — in this file the
    # "Armature" node carries a 0.01 scale (bones are authored in cm), which
    # brings the bind pose into metres, the same space as the vertices.
    nodes = data["nodes"]
    parent_of: dict[int, int] = {}
    for ni, node in enumerate(nodes):
        for child in node.get("children", []):
            parent_of[child] = ni

    def local_matrix(node: dict) -> np.ndarray:
        if "matrix" in node:
            return np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T
        M = np.eye(4)
        M[:3, :3] = _quat_to_mat3(node.get("rotation", [0, 0, 0, 1])) * np.array(
            node.get("scale", [1, 1, 1]), dtype=np.float64)
        M[:3, 3] = node.get("translation", [0, 0, 0])
        return M

    world_cache: dict[int, np.ndarray] = {}

    def world(ni: int) -> np.ndarray:
        if ni not in world_cache:
            local = local_matrix(nodes[ni])
            world_cache[ni] = world(parent_of[ni]) @ local if ni in parent_of else local
        return world_cache[ni]

    skin = data["skins"][0]
    joint_nodes = skin["joints"]
    joint_names = [nodes[i].get("name", f"joint_{i}") for i in joint_nodes]
    slot_of_node = {ni: s for s, ni in enumerate(joint_nodes)}

    n_j = len(joint_nodes)
    parent = np.full(n_j, -1, dtype=np.int32)
    bind_pos = np.zeros((n_j, 3))
    bind_rot = np.zeros((n_j, 3, 3))
    for s, ni in enumerate(joint_nodes):
        W = world(ni)
        bind_pos[s] = W[:3, 3]
        rot = W[:3, :3].copy()
        rot /= np.linalg.norm(rot, axis=0, keepdims=True)   # strip scale
        bind_rot[s] = rot
        # nearest ancestor that is also a joint
        anc = parent_of.get(ni)
        while anc is not None and anc not in slot_of_node:
            anc = parent_of.get(anc)
        parent[s] = slot_of_node[anc] if anc is not None else -1

    return GLBMesh(
        positions=np.ascontiguousarray(positions),
        normals=np.ascontiguousarray(normals),
        joints=np.ascontiguousarray(joints_arr),
        weights=np.ascontiguousarray(weights),
        indices=np.ascontiguousarray(indices),
        joint_names=joint_names,
        parent=parent,
        bind_pos=bind_pos,
        bind_rot=bind_rot,
    )


# ── Retargeting + Skinning ────────────────────────────────────────────────────

# Direction of each GLB bone is measured towards this child (GLB names).
# Bones not listed (hands, toes, head, end-effectors) inherit their parent's
# correction.
GLB_MAIN_CHILD: dict[str, str] = {
    "Hips": "Spine02", "Spine02": "Spine01", "Spine01": "Spine", "Spine": "neck",
    "neck": "Head",
    "LeftUpLeg": "LeftLeg", "LeftLeg": "LeftFoot", "LeftFoot": "LeftToeBase",
    "RightUpLeg": "RightLeg", "RightLeg": "RightFoot", "RightFoot": "RightToeBase",
    "LeftShoulder": "LeftArm", "LeftArm": "LeftForeArm", "LeftForeArm": "LeftHand",
    "RightShoulder": "RightArm", "RightArm": "RightForeArm", "RightForeArm": "RightHand",
}


def _rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Shortest-arc rotation matrix taking direction a onto direction b."""
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -0.999999:                         # opposite directions
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        axis /= np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        return np.eye(3) + 2.0 * (K @ K)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + (K @ K) / (1.0 + c)


class Retargeter:
    """Drive the GLB skeleton with BVH motion.

    The GLB bind pose (A-pose) and the BVH rest pose (T-pose) differ, so BVH
    world matrices cannot be used directly. Per bone we compute a constant
    correction C that rotates the BVH rest bone direction onto the GLB bind
    bone direction, then at every frame

        D_j(t) = R_bvh_world(t) @ C_j^T        (world-space rotation of bone j)
        p_j(t) = p_parent(t) + D_parent @ (bind_j - bind_parent)

    so the GLB keeps its own proportions and only receives the BVH rotations.
    The skinning matrix is  M_j = T(p_j(t)) @ D_j @ T(-bind_j)  which maps the
    bind-pose vertices straight to the posed ones.
    """

    def __init__(self, mesh: GLBMesh, bvh: BVHData) -> None:
        self.mesh = mesh
        self.bvh = bvh
        names = mesh.joint_names
        n_j = len(names)
        glb_idx = {n: i for i, n in enumerate(names)}
        bvh_idx = {j.name: i for i, j in enumerate(bvh.joints)}
        self.bvh_of = np.array(
            [bvh_idx.get(GLB_TO_BVH.get(n, ""), -1) for n in names], dtype=np.int32)

        # Parents-first processing order
        order: list[int] = []
        seen = [False] * n_j
        def visit(j: int) -> None:
            if seen[j]:
                return
            seen[j] = True
            if mesh.parent[j] >= 0:
                visit(int(mesh.parent[j]))
            order.append(j)
        for j in range(n_j):
            visit(j)
        self.order = order

        # Constant rest-pose corrections
        rest = bvh_rest_positions(bvh)
        P = mesh.bind_pos
        self.C = np.tile(np.eye(3), (n_j, 1, 1))
        for j in order:
            name, b = names[j], self.bvh_of[j]
            par = int(mesh.parent[j])
            child = glb_idx.get(GLB_MAIN_CHILD.get(name, ""), -1)
            if b >= 0 and child >= 0 and self.bvh_of[child] >= 0:
                d_bvh = rest[self.bvh_of[child]] - rest[b]
                d_glb = P[child] - P[j]
                self.C[j] = _rotation_between(d_bvh, d_glb)
            elif par >= 0:
                self.C[j] = self.C[par]

        # Uniform scale so the BVH root height/stride fits the GLB legs
        self.scale = 1.0
        hips, foot = glb_idx.get("Hips"), glb_idx.get("LeftFoot")
        if hips is not None and foot is not None:
            bh, bf = self.bvh_of[hips], self.bvh_of[foot]
            if bh >= 0 and bf >= 0:
                bvh_leg = abs(rest[bh][1] - rest[bf][1])
                glb_leg = abs(P[hips][1] - P[foot][1])
                if bvh_leg > 1e-6:
                    self.scale = glb_leg / bvh_leg

        _, root0 = bvh_pose(bvh, 0)
        self.root0 = root0

    def pose(self, frame: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ((J,4,4) skinning matrices, root world position)."""
        mesh = self.mesh
        n_j = len(mesh.joint_names)
        P = mesh.bind_pos
        rot, root = bvh_pose(self.bvh, frame)

        root_world = np.array([
            (root[0] - self.root0[0]) * self.scale,
            root[1] * self.scale,
            (root[2] - self.root0[2]) * self.scale,
        ])

        D = np.tile(np.eye(3), (n_j, 1, 1))
        p = np.zeros((n_j, 3))
        mats = np.tile(np.eye(4), (n_j, 1, 1))
        for j in self.order:
            par = int(mesh.parent[j])
            b = self.bvh_of[j]
            if b >= 0:
                D[j] = rot[b] @ self.C[j].T
            elif par >= 0:
                D[j] = D[par]
            if par >= 0:
                p[j] = p[par] + D[par] @ (P[j] - P[par])
            else:
                p[j] = root_world if b == 0 else P[j]
            mats[j, :3, :3] = D[j]
            mats[j, :3, 3] = p[j] - D[j] @ P[j]
        return mats.astype(np.float32), root_world


# ── OpenGL helpers ────────────────────────────────────────────────────────────

def _compile_shader(src: str, kind: int) -> int:
    sid = glCreateShader(kind)
    glShaderSource(sid, src)
    glCompileShader(sid)
    if glGetShaderiv(sid, GL_COMPILE_STATUS) != GL_TRUE:
        raise RuntimeError(glGetShaderInfoLog(sid).decode())
    return sid


def _link_program(vert_src: str, frag_src: str) -> int:
    vs = _compile_shader(vert_src, GL_VERTEX_SHADER)
    fs = _compile_shader(frag_src, GL_FRAGMENT_SHADER)
    prog = glCreateProgram()
    glAttachShader(prog, vs)
    glAttachShader(prog, fs)
    glLinkProgram(prog)
    if glGetProgramiv(prog, GL_LINK_STATUS) != GL_TRUE:
        raise RuntimeError(glGetProgramInfoLog(prog).decode())
    return prog


# ── Ground Grid ───────────────────────────────────────────────────────────────

def draw_ground_grid(cx: float = 0.0, cz: float = 0.0) -> None:
    glPushMatrix()
    # snap to the grid step so the floor appears endless while the camera follows
    glTranslatef(round(cx / GRID_STEP) * GRID_STEP, 0.0, round(cz / GRID_STEP) * GRID_STEP)
    glDisable(GL_LIGHTING)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_LINE_SMOOTH)
    glLineWidth(1.0)

    half = GRID_SIZE * GRID_STEP
    r, g, b = GRID_COLOUR
    glColor4f(r, g, b, 0.6)
    glBegin(GL_LINES)
    x = -half
    while x <= half + 1e-5:
        glVertex3f(x, 0.0, -half); glVertex3f(x, 0.0,  half)
        glVertex3f(-half, 0.0, x); glVertex3f( half, 0.0, x)
        x += GRID_STEP
    glEnd()

    glLineWidth(1.5)
    glColor4f(r+0.15, g+0.15, b+0.15, 0.9)
    glBegin(GL_LINES)
    glVertex3f(0.0, 0.0, -half); glVertex3f(0.0, 0.0,  half)
    glVertex3f(-half, 0.0, 0.0); glVertex3f( half, 0.0, 0.0)
    glEnd()

    glDisable(GL_LINE_SMOOTH)
    glDisable(GL_BLEND)
    glPopMatrix()


# ── OpenGL Viewport ───────────────────────────────────────────────────────────

class SkeletonViewport(QOpenGLWidget):
    """QOpenGLWidget: renders the GLB skeleton driven by BVH mocap."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(800, 600)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Camera
        self._cam_azimuth:   float = CAM_AZIMUTH_DEFAULT
        self._cam_elevation: float = CAM_ELEVATION_DEFAULT
        self._cam_distance:  float = CAM_DISTANCE_DEFAULT
        self._cam_target:    list[float] = list(CAM_TARGET_DEFAULT)
        self._last_mouse_pos = None

        # Animation state
        self._bvh:         BVHData | None = None
        self._mesh:        GLBMesh | None = None
        self._retarget:    Retargeter | None = None
        self._follow:      np.ndarray = np.zeros(3)   # smoothed root position the camera tracks
        self._frame:       int   = 0
        self._frame_timer: float = 0.0
        self._playing:     bool  = False
        self._speed:       float = 1.0
        self._motion_name: str   = "—"
        self._frame_count: int   = 0

        # Auto-orbit
        self._auto_orbit:       bool  = True
        self._auto_orbit_speed: float = 8.0

        # FPS
        self._fps_display:  float = 0.0
        self._last_fps_time: float = time.perf_counter()
        self._fps_frames:   int   = 0

        # GL objects (created in initializeGL)
        self._prog:    int = 0
        self._vao:     int = 0
        self._n_idx:   int = 0
        self._idx_type = GL_UNSIGNED_INT

    # ── Public API ──────────────────────────────────────────────────────────

    def load_bvh(self, path: Path, motion_name: str) -> None:
        self._bvh         = parse_bvh(path)
        self._frame       = 0
        self._frame_timer = 0.0
        self._motion_name = motion_name
        self._frame_count = len(self._bvh.frames)
        self._follow[:] = 0.0
        self._rebuild_retarget()
        self.update()

    def _rebuild_retarget(self) -> None:
        """(Re)build the BVH→GLB retargeter once both assets are available."""
        if self._mesh is None or self._bvh is None:
            return
        try:
            self._retarget = Retargeter(self._mesh, self._bvh)
        except Exception as e:
            import traceback
            print(f"[ERROR] Retargeting setup failed: {e}")
            traceback.print_exc()
            self._retarget = None

    def set_playing(self, playing: bool) -> None:
        self._playing = playing

    def set_speed(self, speed: float) -> None:
        self._speed = speed

    def set_auto_orbit(self, enabled: bool) -> None:
        self._auto_orbit = enabled

    def advance_frame(self, dt: float) -> None:
        if self._bvh is None:
            return

        if self._playing:
            self._frame_timer += dt * self._speed
            frame_dur = self._bvh.frame_time
            while self._frame_timer >= frame_dur:
                self._frame_timer -= frame_dur
                self._frame = (self._frame + 1) % self._frame_count

        if self._auto_orbit and not self._playing:
            self._cam_azimuth += self._auto_orbit_speed * dt

        self._fps_frames += 1
        now = time.perf_counter()
        elapsed = now - self._last_fps_time
        if elapsed >= 1.0:
            self._fps_display    = self._fps_frames / elapsed
            self._fps_frames     = 0
            self._last_fps_time  = now

        self.update()

    def reset_camera(self) -> None:
        self._cam_azimuth   = CAM_AZIMUTH_DEFAULT
        self._cam_elevation = CAM_ELEVATION_DEFAULT
        self._cam_distance  = CAM_DISTANCE_DEFAULT
        self._cam_target    = list(CAM_TARGET_DEFAULT)
        self.update()

    # ── QOpenGLWidget overrides ─────────────────────────────────────────────

    def initializeGL(self) -> None:
        import traceback
        from OpenGL.GL import glEnable, glClearColor
        glClearColor(0.05, 0.07, 0.11, 1.0)
        glEnable(GL_DEPTH_TEST)

        # Compile shader
        try:
            self._prog = _link_program(VERT_SRC, FRAG_SRC)
        except Exception as e:
            print(f"[ERROR] Shader compilation failed: {e}")
            traceback.print_exc()
            return

        # Load GLB if present
        if SKELETON_GLB.exists():
            try:
                self._load_mesh()
            except Exception as e:
                print(f"[ERROR] GLB mesh load failed: {e}")
                traceback.print_exc()
        else:
            print(f"[ERROR] GLB not found at: {SKELETON_GLB}")

    def _load_mesh(self) -> None:
        """Upload GLB geometry to the GPU."""
        self._mesh = load_glb(SKELETON_GLB)
        m = self._mesh

        self._vao = glGenVertexArrays(1)
        glBindVertexArray(self._vao)

        # Positions — location 0
        pos_buf = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, pos_buf)
        glBufferData(GL_ARRAY_BUFFER, m.positions.nbytes, m.positions, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, False, 0, None)

        # Normals — location 1
        nrm_buf = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, nrm_buf)
        glBufferData(GL_ARRAY_BUFFER, m.normals.nbytes, m.normals, GL_STATIC_DRAW)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, False, 0, None)

        # Joints — location 2 (integer attribute; dtype may be uint8 or uint16)
        j_buf = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, j_buf)
        glBufferData(GL_ARRAY_BUFFER, m.joints.nbytes, m.joints, GL_STATIC_DRAW)
        glEnableVertexAttribArray(2)
        joint_gl_type = GL_UNSIGNED_SHORT if m.joints.dtype == np.uint16 else GL_UNSIGNED_BYTE
        glVertexAttribIPointer(2, 4, joint_gl_type, 0, None)

        # Weights — location 3
        w_buf = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, w_buf)
        glBufferData(GL_ARRAY_BUFFER, m.weights.nbytes, m.weights, GL_STATIC_DRAW)
        glEnableVertexAttribArray(3)
        glVertexAttribPointer(3, 4, GL_FLOAT, False, 0, None)

        # Index buffer
        idx_buf = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, idx_buf)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, m.indices.nbytes, m.indices, GL_STATIC_DRAW)
        self._n_idx = len(m.indices)

        glBindVertexArray(0)
        self._rebuild_retarget()

    def resizeGL(self, w: int, h: int) -> None:
        pass   # viewport is set (in device pixels) at the start of paintGL

    def paintGL(self) -> None:
        # QOpenGLWidget renders into a framebuffer in *device* pixels, which
        # differs from widget size on scaled (HiDPI) displays.
        dpr = self.devicePixelRatioF()
        w, h = self.width(), max(self.height(), 1)
        glViewport(0, 0, int(w * dpr), int(h * dpr))
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # ── Pose first, so the camera can follow the character ──
        joint_mats = None
        if self._mesh is not None and self._retarget is not None and self._prog != 0:
            try:
                joint_mats, root = self._retarget.pose(self._frame)
                # smooth follow on the ground plane only
                self._follow[0] += (root[0] - self._follow[0]) * 0.15
                self._follow[2] += (root[2] - self._follow[2]) * 0.15
            except Exception as e:
                import traceback
                if not getattr(self, "_pose_error_shown", False):
                    self._pose_error_shown = True
                    print(f"[ERROR] Pose computation failed: {e}")
                    traceback.print_exc()

        # ── Camera matrices ──
        tx = self._cam_target[0] + float(self._follow[0])
        ty = self._cam_target[1]
        tz = self._cam_target[2] + float(self._follow[2])
        az = math.radians(self._cam_azimuth)
        el = math.radians(self._cam_elevation)
        cx = tx + self._cam_distance * math.cos(el) * math.sin(az)
        cy = ty + self._cam_distance * math.sin(el)
        cz = tz + self._cam_distance * math.cos(el) * math.cos(az)
        eye = np.array([cx, cy, cz], dtype=np.float32)
        aspect = w / h

        # Projection
        fov, near, far = 45.0, 0.05, 100.0
        f = 1.0 / math.tan(math.radians(fov) / 2)
        proj = np.array([
            [f/aspect, 0,  0,                        0],
            [0,        f,  0,                        0],
            [0,        0,  (far+near)/(near-far),   (2*far*near)/(near-far)],
            [0,        0,  -1,                       0],
        ], dtype=np.float32)

        # View (lookAt)
        tgt = np.array([tx, ty, tz], dtype=np.float32)
        fwd = tgt - eye; fwd /= np.linalg.norm(fwd)
        up  = np.array([0, 1, 0], dtype=np.float32)
        rgt = np.cross(fwd, up); rgt /= np.linalg.norm(rgt)
        up2 = np.cross(rgt, fwd)
        view = np.array([
            [rgt[0],  rgt[1],  rgt[2],  -np.dot(rgt, eye)],
            [up2[0],  up2[1],  up2[2],  -np.dot(up2, eye)],
            [-fwd[0],-fwd[1], -fwd[2],   np.dot(fwd, eye)],
            [0,       0,       0,         1               ],
        ], dtype=np.float32)

        # ── Draw ground grid (immediate mode, no shader) ──
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(fov, aspect, near, far)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(cx, cy, cz, tx, ty, tz, 0.0, 1.0, 0.0)
        draw_ground_grid(float(self._follow[0]), float(self._follow[2]))

        # ── Draw skeleton mesh ──
        if joint_mats is None:
            if not getattr(self, "_warned_once", False):
                self._warned_once = True
                print(f"[WARN] Cannot draw skeleton: mesh={self._mesh is not None}, "
                      f"bvh={self._bvh is not None}, "
                      f"retarget={self._retarget is not None}, prog={self._prog}")
            return

        try:
            glUseProgram(self._prog)

            # row-major numpy matrices → transpose=True
            glUniformMatrix4fv(
                glGetUniformLocation(self._prog, "uProjection"), 1, True, proj)
            glUniformMatrix4fv(
                glGetUniformLocation(self._prog, "uView"), 1, True, view)
            glUniformMatrix4fv(
                glGetUniformLocation(self._prog, "uJointMatrices[0]"),
                len(joint_mats), True, np.ascontiguousarray(joint_mats))

            glUniform3f(glGetUniformLocation(self._prog, "uLightPos"),
                        tx + 2.0, 5.0, tz + 3.0)
            glUniform3f(glGetUniformLocation(self._prog, "uViewPos"),
                        eye[0], eye[1], eye[2])

            glEnable(GL_DEPTH_TEST)
            glDisable(GL_CULL_FACE)      # mesh winding isn't guaranteed; shader is two-sided
            glBindVertexArray(self._vao)
            glDrawElements(GL_TRIANGLES, self._n_idx, GL_UNSIGNED_INT, None)
            glBindVertexArray(0)
            glUseProgram(0)
        except Exception as e:
            import traceback
            if not getattr(self, "_draw_error_shown", False):
                self._draw_error_shown = True
                print(f"[ERROR] Skeleton draw failed: {e}")
                traceback.print_exc()

    # ── Mouse interaction ───────────────────────────────────────────────────

    def mousePressEvent(self, event) -> None:
        self._last_mouse_pos = event.position()
        self._auto_orbit = False

    def mouseMoveEvent(self, event) -> None:
        if self._last_mouse_pos is None:
            return
        dx = event.position().x() - self._last_mouse_pos.x()
        dy = event.position().y() - self._last_mouse_pos.y()
        self._last_mouse_pos = event.position()

        if event.buttons() & Qt.LeftButton:
            self._cam_azimuth   += dx * 0.5
            self._cam_elevation -= dy * 0.5
            self._cam_elevation  = max(-89.0, min(89.0, self._cam_elevation))
        elif event.buttons() & Qt.RightButton:
            scale = self._cam_distance * 0.001
            az = math.radians(self._cam_azimuth)
            self._cam_target[0] -= dx * scale * math.cos(az)
            self._cam_target[2] -= dx * scale * math.sin(az)
            self._cam_target[1] += dy * scale
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        self._last_mouse_pos = None

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        self._cam_distance *= 0.9 if delta > 0 else 1.1
        self._cam_distance = max(0.3, min(20.0, self._cam_distance))
        self.update()


# ── Control Panel ─────────────────────────────────────────────────────────────

class ControlPanel(QWidget):
    def __init__(self, viewport: SkeletonViewport,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._viewport = viewport
        self.setMinimumWidth(280)
        self.setMaximumWidth(320)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(14)

        # ── Header ────────────────────────────────────────────────────────
        header = QWidget()
        h_layout = QVBoxLayout(header)
        h_layout.setContentsMargins(0, 0, 0, 0)
        h_layout.setSpacing(4)

        logo_label = QLabel()
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if LOGO_PATH.exists():
            pixmap = QPixmap(str(LOGO_PATH)).scaledToWidth(
                72, Qt.TransformationMode.SmoothTransformation)
            logo_label.setPixmap(pixmap)
        else:
            logo_label.setText("AMS Group – TU Ilmenau")
            logo_label.setStyleSheet("color: #4f9cf9; font-size: 10px;")

        author_label = QLabel("Renato Fera P.")
        author_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        author_label.setStyleSheet("color: #5a6175; font-size: 10px;")

        chapter_label = QLabel("Chapter 05 · Physics-based Animation")
        chapter_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        chapter_label.setStyleSheet("color: #4f9cf9; font-size: 9px;")
        chapter_label.setWordWrap(True)

        h_layout.addWidget(logo_label)
        h_layout.addWidget(author_label)
        h_layout.addWidget(chapter_label)
        layout.addWidget(header)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color: #252b38;")
        layout.addWidget(line)

        # ── Motion preset ─────────────────────────────────────────────────
        motion_group = QGroupBox("Motion Preset")
        motion_group.setStyleSheet(self._group_style())
        m_layout = QVBoxLayout(motion_group)
        m_layout.setSpacing(6)

        self._motion_combo = QComboBox()
        self._motion_combo.setStyleSheet(self._combo_style())
        for label, path in self._motion_options():
            self._motion_combo.addItem(label, path)
        self._motion_combo.currentIndexChanged.connect(self._on_motion_changed)
        m_layout.addWidget(self._motion_combo)

        self._status_label = QLabel("No motion loaded")
        self._status_label.setStyleSheet("color: #5a6175; font-size: 10px;")
        self._status_label.setWordWrap(True)
        m_layout.addWidget(self._status_label)
        layout.addWidget(motion_group)

        # ── Playback ──────────────────────────────────────────────────────
        playback_group = QGroupBox("Playback")
        playback_group.setStyleSheet(self._group_style())
        p_layout = QVBoxLayout(playback_group)
        p_layout.setSpacing(8)

        btn_row = QHBoxLayout()
        self._play_btn = QPushButton("▶  Play")
        self._play_btn.setCheckable(True)
        self._play_btn.setStyleSheet(self._button_style(accent=True))
        self._play_btn.toggled.connect(self._on_play_toggled)
        btn_row.addWidget(self._play_btn)

        reset_btn = QPushButton("↺  Reset")
        reset_btn.setStyleSheet(self._button_style())
        reset_btn.clicked.connect(self._on_reset)
        btn_row.addWidget(reset_btn)
        p_layout.addLayout(btn_row)

        speed_label = QLabel("Playback Speed")
        speed_label.setStyleSheet("color: #d8dce6; font-size: 11px;")
        p_layout.addWidget(speed_label)

        speed_row = QHBoxLayout()
        self._speed_slider = QSlider(Qt.Horizontal)
        self._speed_slider.setRange(10, 300)
        self._speed_slider.setValue(100)
        self._speed_slider.setStyleSheet(self._slider_style())
        self._speed_value = QLabel("1.0×")
        self._speed_value.setStyleSheet("color: #4f9cf9; font-size: 11px;")
        self._speed_value.setMinimumWidth(36)
        self._speed_slider.valueChanged.connect(self._on_speed_changed)
        speed_row.addWidget(self._speed_slider)
        speed_row.addWidget(self._speed_value)
        p_layout.addLayout(speed_row)
        layout.addWidget(playback_group)

        # ── Camera ────────────────────────────────────────────────────────
        camera_group = QGroupBox("Camera")
        camera_group.setStyleSheet(self._group_style())
        c_layout = QVBoxLayout(camera_group)
        c_layout.setSpacing(8)

        cam_hint = QLabel("Left drag: orbit   Right drag: pan\nScroll: zoom")
        cam_hint.setStyleSheet("color: #5a6175; font-size: 10px;")
        c_layout.addWidget(cam_hint)

        cam_btn_row = QHBoxLayout()
        reset_cam_btn = QPushButton("Reset Camera")
        reset_cam_btn.setStyleSheet(self._button_style())
        reset_cam_btn.clicked.connect(self._viewport.reset_camera)
        cam_btn_row.addWidget(reset_cam_btn)

        self._orbit_btn = QPushButton("Auto-Orbit: ON")
        self._orbit_btn.setCheckable(True)
        self._orbit_btn.setChecked(True)
        self._orbit_btn.setStyleSheet(self._button_style())
        self._orbit_btn.toggled.connect(self._on_orbit_toggled)
        cam_btn_row.addWidget(self._orbit_btn)
        c_layout.addLayout(cam_btn_row)
        layout.addWidget(camera_group)

        # ── Info ──────────────────────────────────────────────────────────
        info_group = QGroupBox("Info")
        info_group.setStyleSheet(self._group_style())
        i_layout = QFormLayout(info_group)
        i_layout.setSpacing(6)

        style = "color: #4f9cf9; font-size: 11px; font-family: monospace;"
        self._info_motion  = QLabel("—")
        self._info_frames  = QLabel("—")
        self._info_fps_bvh = QLabel("—")
        self._info_fps_gl  = QLabel("—")
        for lbl in (self._info_motion, self._info_frames,
                    self._info_fps_bvh, self._info_fps_gl):
            lbl.setStyleSheet(style)

        i_layout.addRow(QLabel("Motion:"),  self._info_motion)
        i_layout.addRow(QLabel("Frames:"),  self._info_frames)
        i_layout.addRow(QLabel("BVH fps:"), self._info_fps_bvh)
        i_layout.addRow(QLabel("GL fps:"),  self._info_fps_gl)
        for row in range(i_layout.rowCount()):
            lbl = i_layout.itemAt(row, QFormLayout.LabelRole)
            if lbl and lbl.widget():
                lbl.widget().setStyleSheet("color: #5a6175; font-size: 11px;")

        layout.addWidget(info_group)
        layout.addStretch()

        self._on_motion_changed(0)

    def _motion_options(self) -> list[tuple[str, Path | None]]:
        options = []
        for label, path in [
            ("Walk  (Subject 08)", BVH_WALK),
            ("Run   (Subject 09)", BVH_RUN),
            ("Salsa Dance (Subject 60)", BVH_DANCE),
        ]:
            if path.exists():
                options.append((label, path))
        if not options:
            options.append(("No BVH files found", None))
        return options

    def _on_motion_changed(self, index: int) -> None:
        path = self._motion_combo.itemData(index)
        label = self._motion_combo.itemText(index)
        if path is None or not path.exists():
            self._status_label.setText(f"⚠  File not found:\n{path}")
            return
        self._viewport.load_bvh(path, label.strip())
        self._status_label.setText(f"Loaded: {path.name}")
        self._update_info()

    def _on_play_toggled(self, checked: bool) -> None:
        self._viewport.set_playing(checked)
        self._play_btn.setText("⏸  Pause" if checked else "▶  Play")
        if checked:
            self._viewport.set_auto_orbit(False)
            self._orbit_btn.setChecked(False)
            self._orbit_btn.setText("Auto-Orbit: OFF")

    def _on_reset(self) -> None:
        self._viewport.set_playing(False)
        self._play_btn.setChecked(False)
        self._play_btn.setText("▶  Play")
        self._viewport._frame       = 0
        self._viewport._frame_timer = 0.0
        self._viewport.update()

    def _on_speed_changed(self, value: int) -> None:
        speed = value / 100.0
        self._viewport.set_speed(speed)
        self._speed_value.setText(f"{speed:.1f}×")

    def _on_orbit_toggled(self, checked: bool) -> None:
        self._viewport.set_auto_orbit(checked)
        self._orbit_btn.setText("Auto-Orbit: ON" if checked else "Auto-Orbit: OFF")

    def update_info(self) -> None:
        self._update_info()

    def _update_info(self) -> None:
        vp = self._viewport
        if vp._bvh:
            fps_bvh = 1.0 / vp._bvh.frame_time if vp._bvh.frame_time > 0 else 0
            self._info_motion.setText(vp._motion_name[:24])
            self._info_frames.setText(f"{vp._frame + 1} / {vp._frame_count}")
            self._info_fps_bvh.setText(f"{fps_bvh:.0f} fps")
        self._info_fps_gl.setText(f"{vp._fps_display:.1f} fps")

    # ── Style helpers ────────────────────────────────────────────────────

    def _group_style(self) -> str:
        return """
            QGroupBox {
                color: #d8dce6; font-size: 11px; font-weight: 500;
                border: 1px solid #252b38; border-radius: 6px;
                margin-top: 8px; padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin; subcontrol-position: top left;
                padding: 0 6px; color: #a78bfa;
            }
        """

    def _button_style(self, accent: bool = False) -> str:
        bg     = "#1e3a5f" if accent else "#1a1f2b"
        hover  = "#2a4a7f" if accent else "#252b38"
        colour = "#4f9cf9" if accent else "#d8dce6"
        return f"""
            QPushButton {{
                background: {bg}; color: {colour};
                border: 1px solid #252b38; border-radius: 5px;
                padding: 5px 10px; font-size: 11px;
            }}
            QPushButton:hover {{ background: {hover}; border-color: #4f9cf9; }}
            QPushButton:checked {{ background: #2a4a7f; border-color: #4f9cf9; }}
        """

    def _slider_style(self) -> str:
        return """
            QSlider::groove:horizontal { height: 4px; background: #252b38; border-radius: 2px; }
            QSlider::handle:horizontal {
                background: #4f9cf9; border: none;
                width: 12px; height: 12px; margin: -4px 0; border-radius: 6px;
            }
            QSlider::sub-page:horizontal { background: #1e3a5f; border-radius: 2px; }
        """

    def _combo_style(self) -> str:
        return """
            QComboBox {
                background: #1a1f2b; color: #d8dce6;
                border: 1px solid #252b38; border-radius: 5px;
                padding: 4px 8px; font-size: 11px;
            }
            QComboBox:hover { border-color: #4f9cf9; }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background: #161a22; color: #d8dce6;
                selection-background-color: #1e3a5f;
                border: 1px solid #252b38;
            }
        """


# ── Main Window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)
        self._apply_dark_theme()
        self._build_ui()
        self._start_timer()
        self._check_assets()

    def _build_ui(self) -> None:
        self._viewport = SkeletonViewport()
        self.setCentralWidget(self._viewport)

        dock = QDockWidget("Controls", self)
        dock.setFeatures(
            QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        self._panel = ControlPanel(self._viewport)
        dock.setWidget(self._panel)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

        dock.setStyleSheet("""
            QDockWidget { color: #d8dce6; font-size: 11px; }
            QDockWidget::title {
                background: #161a22; padding: 6px 10px;
                border-bottom: 1px solid #252b38;
            }
        """)

    def _start_timer(self) -> None:
        self._last_tick = time.perf_counter()
        self._timer = QTimer(self)
        self._timer.setInterval(1000 // RENDER_FPS)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    def _tick(self) -> None:
        now = time.perf_counter()
        dt  = min(now - self._last_tick, 0.1)
        self._last_tick = now
        self._viewport.advance_frame(dt)
        self._panel.update_info()

    def _apply_dark_theme(self) -> None:
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background: #0d0f14; color: #d8dce6;
                font-family: 'IBM Plex Sans', 'Segoe UI', sans-serif;
            }
            QScrollBar:vertical { background: #161a22; width: 8px; border-radius: 4px; }
            QScrollBar::handle:vertical { background: #252b38; border-radius: 4px; }
        """)

    def _check_assets(self) -> None:
        missing = []
        if not SKELETON_GLB.exists():
            missing.append(SKELETON_GLB.name)
        for path in (BVH_WALK, BVH_RUN, BVH_DANCE):
            if not path.exists():
                missing.append(path.name)
        if missing:
            names = ", ".join(missing)
            self.setWindowTitle(f"{WINDOW_TITLE}  ⚠  Missing: {names}")
            print(f"\n⚠  Missing asset files in {SCRIPT_DIR}")
            print("   human-skeleton-rigged.glb → grab3d.com/free-rigged-3d-models/")
            print("   08_01.bvh → github.com/una-dinosauria/cmu-mocap  data/008/08_01.bvh")
            print("   09_01.bvh → github.com/una-dinosauria/cmu-mocap  data/009/09_01.bvh")
            print("   60_01.bvh → github.com/una-dinosauria/cmu-mocap  data/060/60_01.bvh")


# ── Entry Point ───────────────────────────────────────────────────────────────

def main() -> None:
    """Create the QApplication, show the window, and start the event loop.

    ANIMATION CONCEPT — the event loop:
    app.exec() blocks here and runs Qt's event loop. The QTimer fires at
    RENDER_FPS and calls _tick(), which advances the BVH frame counter and
    triggers a repaint. Same event-loop-plus-timer pattern as the browser
    demos in seminars 01-04.

    SKINNING CONCEPT — GPU skinning:
    Each vertex has up to 4 joint indices and blend weights. Each frame we
    compute one 4x4 matrix per joint (BVH world pose * inverse bind matrix)
    and upload them to the vertex shader as uniforms. The shader blends the
    joint matrices by the per-vertex weights — identical to what Three.js
    SkinnedMesh does in seminar 04, but here in raw GLSL.
    """
    fmt = QSurfaceFormat()
    fmt.setDepthBufferSize(24)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)
    app.setApplicationName("Computeranimation 05 — Skeleton Physics")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

# =============================================================================
# End of 05_ca_pybullet.py
# Author   : Renato Fera P.
# LinkedIn : https://www.linkedin.com/in/renato-profeta/
# Course   : Computeranimation — Chapter 05: Physics-based Animation
# Institute: Institut für Medientechnik, Technische Universität Ilmenau
# =============================================================================
