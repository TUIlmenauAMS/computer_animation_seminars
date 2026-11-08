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

A professional real-time 3D skeleton ragdoll driven by CMU motion capture
data. The skeleton mesh (human-skeleton-rigged.glb) is rendered using
PyOpenGL inside a QOpenGLWidget. BVH mocap files drive the skeleton joints
via a kinematic-to-ragdoll blend: at blend=0 the skeleton follows mocap
exactly; at blend=1 it becomes a full ragdoll under physics.

Pipeline:
  BVH file → parse joint transforms → retarget to skeleton rig
  → PyBullet kinematic/dynamic bodies → OpenGL render

Required files (place in the same directory as this script):
  human-skeleton-rigged.glb  → https://grab3d.com/dl/human-skeleton-rigged/human-skeleton-rigged.glb
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
- The kinematic-to-ragdoll blend slider demonstrates the professional
  "driven + simulated" animation pattern discussed in seminar 05 theory.
- The fixed-timestep accumulator inside the render loop mirrors the
  pattern used in every browser physics demo (Demos 1–4).
- Exported JSON (simulate → record → export) bridges to Three.js seminar 02.
"""
# =============================================================================

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
from OpenGL.GL import (
    GL_AMBIENT,
    GL_AMBIENT_AND_DIFFUSE,
    GL_BLEND,
    GL_COLOR_BUFFER_BIT,
    GL_COLOR_MATERIAL,
    GL_CULL_FACE,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_DIFFUSE,
    GL_FRONT_AND_BACK,
    GL_LIGHT0,
    GL_LIGHTING,
    GL_LINE_SMOOTH,
    GL_LINES,
    GL_MODELVIEW,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_POSITION,
    GL_PROJECTION,
    GL_QUADS,
    GL_SHININESS,
    GL_SMOOTH,
    GL_SPECULAR,
    GL_SRC_ALPHA,
    GL_TRIANGLES,
    glBegin,
    glBlendFunc,
    glClear,
    glClearColor,
    glColor4f,
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
    glShadeModel,
    glTranslatef,
    glVertex3f,
    glViewport,
)
from OpenGL.GLU import gluLookAt, gluNewQuadric, gluPerspective, gluSphere
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDockWidget,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
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

# Physics
PHYSICS_TIMESTEP: float = 1.0 / 120.0  # 120 Hz simulation
RENDER_FPS: int = 60
GRAVITY: float = -9.81

# Skeleton scale — CMU BVH is in cm, we work in metres
BVH_SCALE: float = 0.01

# Bone colours — warm ivory with slight variation per bone group
COLOUR_SPINE: tuple[float, float, float] = (0.95, 0.92, 0.85)
COLOUR_LIMB: tuple[float, float, float] = (0.90, 0.87, 0.80)
COLOUR_SKULL: tuple[float, float, float] = (0.97, 0.94, 0.88)
COLOUR_PELVIS: tuple[float, float, float] = (0.88, 0.85, 0.78)
COLOUR_EXTREMITY: tuple[float, float, float] = (0.85, 0.82, 0.75)

# Ground grid
GRID_SIZE: int = 10
GRID_STEP: float = 0.5
GRID_COLOUR: tuple[float, float, float] = (0.18, 0.22, 0.30)

# Camera defaults
CAM_DISTANCE_DEFAULT: float = 3.5
CAM_AZIMUTH_DEFAULT: float = 25.0
CAM_ELEVATION_DEFAULT: float = 15.0
CAM_TARGET_DEFAULT: tuple[float, float, float] = (0.0, 0.9, 0.0)


# ── BVH Parser ────────────────────────────────────────────────────────────────


class BVHJoint(NamedTuple):
    name: str
    parent: int  # index into joint list, -1 for root
    offset: np.ndarray  # rest-pose offset (metres)
    channels: list[str]  # channel names e.g. ['Xrotation', 'Yrotation', 'Zrotation']
    channel_start: int  # index into one frame's flat channel array


class BVHData(NamedTuple):
    joints: list[BVHJoint]
    frames: np.ndarray  # shape (n_frames, n_channels)
    frame_time: float  # seconds per frame


def parse_bvh(path: Path) -> BVHData:
    """Parse a BVH file into joints and frame data.

    Returns a BVHData with joints in hierarchy order, frames as a
    (n_frames, n_channels) float32 array, and frame_time in seconds.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = [l.strip() for l in text.splitlines()]

    joints: list[BVHJoint] = []
    stack: list[int] = []  # stack of parent indices
    channel_index: int = 0
    i: int = 0

    # ── HIERARCHY section ──
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
            joints.append(
                BVHJoint(
                    name=name,
                    parent=parent,
                    offset=np.zeros(3, dtype=np.float32),
                    channels=[],
                    channel_start=channel_index,
                )
            )
            stack.append(len(joints) - 1)
            i += 1
            continue

        if keyword == "OFFSET" and stack:
            idx = stack[-1]
            offset = (
                np.array(
                    [float(tokens[1]), float(tokens[2]), float(tokens[3])],
                    dtype=np.float32,
                )
                * BVH_SCALE
            )
            joints[idx] = joints[idx]._replace(offset=offset)
            i += 1
            continue

        if keyword == "CHANNELS" and stack:
            idx = stack[-1]
            n = int(tokens[1])
            channels = tokens[2 : 2 + n]
            joints[idx] = joints[idx]._replace(
                channels=channels,
                channel_start=channel_index,
            )
            channel_index += n
            i += 1
            continue

        if keyword == "END":  # End Site
            # skip until closing brace
            i += 1
            depth = 1
            while i < len(lines) and depth > 0:
                if "{" in lines[i]:
                    depth += 1
                if "}" in lines[i]:
                    depth -= 1
                i += 1
            continue

        if line == "}":
            if stack:
                stack.pop()
            i += 1
            continue

        i += 1

    # ── MOTION section ──
    n_frames: int = 0
    frame_time: float = 1.0 / 120.0
    while i < len(lines):
        line = lines[i]
        tokens = line.split()
        if not tokens:
            i += 1
            continue
        if tokens[0] == "Frames:":
            n_frames = int(tokens[1])
        elif tokens[0] == "Frame" and tokens[1] == "Time:":
            frame_time = float(tokens[2])
        elif len(tokens) == channel_index and channel_index > 0:
            break
        i += 1

    # Read frame data
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


def euler_to_matrix(rx: float, ry: float, rz: float, order: str = "ZXY") -> np.ndarray:
    """Convert Euler angles (degrees) to 4×4 rotation matrix.

    BVH files typically use ZXY order (Zrotation Xrotation Yrotation).
    """
    rx_r, ry_r, rz_r = math.radians(rx), math.radians(ry), math.radians(rz)

    cx, sx = math.cos(rx_r), math.sin(rx_r)
    cy, sy = math.cos(ry_r), math.sin(ry_r)
    cz, sz = math.cos(rz_r), math.sin(rz_r)

    Rx = np.array(
        [[1, 0, 0, 0], [0, cx, -sx, 0], [0, sx, cx, 0], [0, 0, 0, 1]], dtype=np.float32
    )
    Ry = np.array(
        [[cy, 0, sy, 0], [0, 1, 0, 0], [-sy, 0, cy, 0], [0, 0, 0, 1]], dtype=np.float32
    )
    Rz = np.array(
        [[cz, -sz, 0, 0], [sz, cz, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
    )

    matrices = {"X": Rx, "Y": Ry, "Z": Rz}
    result = np.eye(4, dtype=np.float32)
    for axis in order:
        result = result @ matrices[axis]
    return result


def bvh_frame_transforms(bvh: BVHData, frame_idx: int) -> list[np.ndarray]:
    """Compute world-space 4×4 transform for every joint at frame_idx.

    Returns a list of 4×4 numpy arrays, one per joint, in joint order.
    """
    frame_idx = frame_idx % len(bvh.frames)
    data = bvh.frames[frame_idx]
    n = len(bvh.joints)
    local: list[np.ndarray] = [np.eye(4, dtype=np.float32)] * n
    world: list[np.ndarray] = [np.eye(4, dtype=np.float32)] * n

    for j, joint in enumerate(bvh.joints):
        # Build local transform: offset then channels
        T = np.eye(4, dtype=np.float32)
        T[0, 3] = joint.offset[0]
        T[1, 3] = joint.offset[1]
        T[2, 3] = joint.offset[2]

        R = np.eye(4, dtype=np.float32)
        tx_x = tx_y = tx_z = 0.0
        rots = {}

        for ch in joint.channels:
            val = data[joint.channel_start + joint.channels.index(ch)]
            if ch == "Xposition":
                tx_x = val * BVH_SCALE
            elif ch == "Yposition":
                tx_y = val * BVH_SCALE
            elif ch == "Zposition":
                tx_z = val * BVH_SCALE
            elif ch in ("Xrotation", "Yrotation", "Zrotation"):
                rots[ch[0]] = val

        if tx_x or tx_y or tx_z:
            T[0, 3] += tx_x
            T[1, 3] += tx_y
            T[2, 3] += tx_z

        if rots:
            order = "".join(ch[0] for ch in joint.channels if ch.endswith("rotation"))
            R = euler_to_matrix(
                rots.get("X", 0.0),
                rots.get("Y", 0.0),
                rots.get("Z", 0.0),
                order if order else "ZXY",
            )

        local[j] = T @ R

        if joint.parent == -1:
            world[j] = local[j]
        else:
            world[j] = world[joint.parent] @ local[j]

    return world


# ── Skeleton Definition ───────────────────────────────────────────────────────
# Mapping from CMU BVH joint names to body-segment groups and radii.
# CMU uses MotionBuilder-compatible names (after Bruce Hahne's conversion).

BONE_SEGMENTS: list[tuple[str, str, float, tuple[float, float, float]]] = [
    # (parent_joint, child_joint, radius, colour)
    # Spine
    ("Hips", "Spine", 0.055, COLOUR_PELVIS),
    ("Spine", "Spine1", 0.050, COLOUR_SPINE),
    ("Spine1", "Spine2", 0.048, COLOUR_SPINE),
    ("Spine2", "Spine3", 0.045, COLOUR_SPINE),
    ("Spine3", "Neck", 0.040, COLOUR_SPINE),
    ("Neck", "Head", 0.038, COLOUR_SKULL),
    # Left arm
    ("Spine3", "LeftShoulder", 0.030, COLOUR_LIMB),
    ("LeftShoulder", "LeftArm", 0.028, COLOUR_LIMB),
    ("LeftArm", "LeftForeArm", 0.025, COLOUR_LIMB),
    ("LeftForeArm", "LeftHand", 0.022, COLOUR_EXTREMITY),
    # Right arm
    ("Spine3", "RightShoulder", 0.030, COLOUR_LIMB),
    ("RightShoulder", "RightArm", 0.028, COLOUR_LIMB),
    ("RightArm", "RightForeArm", 0.025, COLOUR_LIMB),
    ("RightForeArm", "RightHand", 0.022, COLOUR_EXTREMITY),
    # Left leg
    ("Hips", "LeftUpLeg", 0.045, COLOUR_LIMB),
    ("LeftUpLeg", "LeftLeg", 0.040, COLOUR_LIMB),
    ("LeftLeg", "LeftFoot", 0.030, COLOUR_LIMB),
    ("LeftFoot", "LeftToeBase", 0.022, COLOUR_EXTREMITY),
    # Right leg
    ("Hips", "RightUpLeg", 0.045, COLOUR_LIMB),
    ("RightUpLeg", "RightLeg", 0.040, COLOUR_LIMB),
    ("RightLeg", "RightFoot", 0.030, COLOUR_LIMB),
    ("RightFoot", "RightToeBase", 0.022, COLOUR_EXTREMITY),
]

# Joint names that get a sphere rendered (joint ball)
JOINT_SPHERES: dict[str, tuple[float, tuple[float, float, float]]] = {
    "Hips": (0.070, COLOUR_PELVIS),
    "Head": (0.090, COLOUR_SKULL),
    "LeftHand": (0.030, COLOUR_EXTREMITY),
    "RightHand": (0.030, COLOUR_EXTREMITY),
    "LeftFoot": (0.035, COLOUR_EXTREMITY),
    "RightFoot": (0.035, COLOUR_EXTREMITY),
}


# ── OpenGL Primitives ─────────────────────────────────────────────────────────


def _draw_cylinder(
    p0: np.ndarray, p1: np.ndarray, radius: float, slices: int = 12
) -> None:
    """Draw a capped cylinder from p0 to p1 using immediate-mode GL."""
    vec = p1 - p0
    length = float(np.linalg.norm(vec))
    if length < 1e-5:
        return

    # Build rotation from Z-axis to vec
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z, vec / length)
    ax_len = float(np.linalg.norm(axis))
    angle = math.degrees(math.acos(float(np.clip(np.dot(z, vec / length), -1, 1))))

    glPushMatrix()
    glTranslatef(float(p0[0]), float(p0[1]), float(p0[2]))
    if ax_len > 1e-6:
        glRotatef(
            angle,
            float(axis[0] / ax_len),
            float(axis[1] / ax_len),
            float(axis[2] / ax_len),
        )
    elif angle > 90:
        glRotatef(180.0, 1.0, 0.0, 0.0)

    # Tube faces
    glBegin(GL_QUADS)
    for i in range(slices):
        a0 = 2 * math.pi * i / slices
        a1 = 2 * math.pi * (i + 1) / slices
        x0, y0 = radius * math.cos(a0), radius * math.sin(a0)
        x1, y1 = radius * math.cos(a1), radius * math.sin(a1)
        nx0, ny0 = math.cos(a0), math.sin(a0)
        nx1, ny1 = math.cos(a1), math.sin(a1)
        glNormal3f(nx0, ny0, 0.0)
        glVertex3f(x0, y0, 0.0)
        glNormal3f(nx1, ny1, 0.0)
        glVertex3f(x1, y1, 0.0)
        glNormal3f(nx1, ny1, 0.0)
        glVertex3f(x1, y1, length)
        glNormal3f(nx0, ny0, 0.0)
        glVertex3f(x0, y0, length)
    glEnd()

    # End caps
    for z_cap in (0.0, length):
        normal_z = -1.0 if z_cap == 0.0 else 1.0
        glBegin(GL_TRIANGLES)
        glNormal3f(0.0, 0.0, normal_z)
        for i in range(slices):
            a0 = 2 * math.pi * i / slices
            a1 = 2 * math.pi * (i + 1) / slices
            glVertex3f(0.0, 0.0, z_cap)
            glVertex3f(radius * math.cos(a0), radius * math.sin(a0), z_cap)
            glVertex3f(radius * math.cos(a1), radius * math.sin(a1), z_cap)
        glEnd()

    glPopMatrix()


def _draw_sphere_at(pos: np.ndarray, radius: float) -> None:
    """Draw a sphere at world position pos."""
    quad = gluNewQuadric()
    glPushMatrix()
    glTranslatef(float(pos[0]), float(pos[1]), float(pos[2]))
    gluSphere(quad, radius, 16, 12)
    glPopMatrix()


def _set_bone_material(colour: tuple[float, float, float]) -> None:
    """Set OpenGL material for ivory bone rendering."""
    r, g, b = colour
    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, [r * 0.4, g * 0.4, b * 0.4, 1.0])
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, [r, g, b, 1.0])
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [0.6, 0.6, 0.55, 1.0])
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 60.0)


# ── Skeleton Renderer ─────────────────────────────────────────────────────────


class SkeletonRenderer:
    """Renders BVH joint transforms as anatomical bone geometry.

    For each frame, bvh_frame_transforms() provides a list of 4×4 world-space
    matrices — one per BVH joint. We extract the translation of each joint
    to get its 3D position, then draw cylinders between connected joints.
    """

    def __init__(self, bvh: BVHData) -> None:
        self._bvh = bvh
        self._joint_index: dict[str, int] = {
            j.name: i for i, j in enumerate(bvh.joints)
        }
        self._frame: int = 0
        self._world: list[np.ndarray] = bvh_frame_transforms(bvh, 0)

    def set_frame(self, frame: int) -> None:
        self._frame = frame % len(self._bvh.frames)
        self._world = bvh_frame_transforms(self._bvh, self._frame)

    def joint_position(self, name: str) -> np.ndarray | None:
        idx = self._joint_index.get(name)
        if idx is None:
            return None
        m = self._world[idx]
        return np.array([m[0, 3], m[1, 3], m[2, 3]], dtype=np.float32)

    def render(self) -> None:
        """Draw all bone segments and joint spheres."""
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glShadeModel(GL_SMOOTH)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)

        for parent_name, child_name, radius, colour in BONE_SEGMENTS:
            p0 = self.joint_position(parent_name)
            p1 = self.joint_position(child_name)
            if p0 is None or p1 is None:
                continue
            _set_bone_material(colour)
            _draw_cylinder(p0, p1, radius)

        for joint_name, (radius, colour) in JOINT_SPHERES.items():
            pos = self.joint_position(joint_name)
            if pos is None:
                continue
            _set_bone_material(colour)
            _draw_sphere_at(pos, radius)

        glDisable(GL_CULL_FACE)


# ── Ground Grid ───────────────────────────────────────────────────────────────


def draw_ground_grid() -> None:
    """Draw a subtle dark grid on y=0."""
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
        glVertex3f(x, 0.0, -half)
        glVertex3f(x, 0.0, half)
        glVertex3f(-half, 0.0, x)
        glVertex3f(half, 0.0, x)
        x += GRID_STEP
    glEnd()

    # Brighter centre lines
    glLineWidth(1.5)
    glColor4f(r + 0.15, g + 0.15, b + 0.15, 0.9)
    glBegin(GL_LINES)
    glVertex3f(0.0, 0.0, -half)
    glVertex3f(0.0, 0.0, half)
    glVertex3f(-half, 0.0, 0.0)
    glVertex3f(half, 0.0, 0.0)
    glEnd()

    glDisable(GL_LINE_SMOOTH)
    glDisable(GL_BLEND)
    glEnable(GL_LIGHTING)


# ── OpenGL Viewport ───────────────────────────────────────────────────────────


class SkeletonViewport(QOpenGLWidget):
    """QOpenGLWidget that renders the skeleton scene.

    Camera is controlled by mouse drag (orbit) and scroll (zoom).
    Physics timestep accumulator drives animation at a fixed rate
    regardless of render frame rate — the same pattern as the browser
    demos in seminars 01–04.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(800, 600)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Camera state
        self._cam_azimuth: float = CAM_AZIMUTH_DEFAULT
        self._cam_elevation: float = CAM_ELEVATION_DEFAULT
        self._cam_distance: float = CAM_DISTANCE_DEFAULT
        self._cam_target: list[float] = list(CAM_TARGET_DEFAULT)
        self._last_mouse_pos = None

        # Animation state
        self._renderer: SkeletonRenderer | None = None
        self._bvh: BVHData | None = None
        self._frame: int = 0
        self._frame_timer: float = 0.0
        self._playing: bool = False
        self._speed: float = 1.0

        # HUD info
        self._motion_name: str = "—"
        self._frame_count: int = 0
        self._fps_display: float = 0.0
        self._last_fps_time: float = time.perf_counter()
        self._fps_frames: int = 0

        # Auto-orbit when paused
        self._auto_orbit: bool = True
        self._auto_orbit_speed: float = 8.0  # degrees/sec

    # ── Public API ─────────────────────────────────────────────────────────

    def load_bvh(self, path: Path, motion_name: str) -> None:
        """Parse and load a BVH file, replacing the current animation."""
        bvh = parse_bvh(path)
        self._bvh = bvh
        self._renderer = SkeletonRenderer(bvh)
        self._frame = 0
        self._frame_timer = 0.0
        self._motion_name = motion_name
        self._frame_count = len(bvh.frames)
        self.update()

    def set_playing(self, playing: bool) -> None:
        self._playing = playing

    def set_speed(self, speed: float) -> None:
        self._speed = speed

    def set_auto_orbit(self, enabled: bool) -> None:
        self._auto_orbit = enabled

    def advance_frame(self, dt: float) -> None:
        """Advance animation by dt seconds (called by the main timer)."""
        if self._bvh is None or self._renderer is None:
            return

        if self._playing:
            self._frame_timer += dt * self._speed
            frame_duration = self._bvh.frame_time
            while self._frame_timer >= frame_duration:
                self._frame_timer -= frame_duration
                self._frame = (self._frame + 1) % self._frame_count
            self._renderer.set_frame(self._frame)

        if self._auto_orbit and not self._playing:
            self._cam_azimuth += self._auto_orbit_speed * dt

        self._fps_frames += 1
        now = time.perf_counter()
        elapsed = now - self._last_fps_time
        if elapsed >= 1.0:
            self._fps_display = self._fps_frames / elapsed
            self._fps_frames = 0
            self._last_fps_time = now

        self.update()

    def reset_camera(self) -> None:
        self._cam_azimuth = CAM_AZIMUTH_DEFAULT
        self._cam_elevation = CAM_ELEVATION_DEFAULT
        self._cam_distance = CAM_DISTANCE_DEFAULT
        self._cam_target = list(CAM_TARGET_DEFAULT)
        self.update()

    # ── QOpenGLWidget overrides ─────────────────────────────────────────────

    def initializeGL(self) -> None:
        glClearColor(0.05, 0.07, 0.11, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glShadeModel(GL_SMOOTH)

        # Key light — warm white from upper front
        glLightfv(GL_LIGHT0, GL_POSITION, [2.0, 5.0, 3.0, 0.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.0, 0.98, 0.95, 1.0])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [0.8, 0.8, 0.75, 1.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.12, 0.13, 0.15, 1.0])

    def resizeGL(self, w: int, h: int) -> None:
        h = max(h, 1)
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, w / h, 0.05, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self) -> None:
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # Camera: spherical coordinates around target
        az = math.radians(self._cam_azimuth)
        el = math.radians(self._cam_elevation)
        cx = self._cam_target[0] + self._cam_distance * math.cos(el) * math.sin(az)
        cy = self._cam_target[1] + self._cam_distance * math.sin(el)
        cz = self._cam_target[2] + self._cam_distance * math.cos(el) * math.cos(az)
        gluLookAt(
            cx,
            cy,
            cz,
            self._cam_target[0],
            self._cam_target[1],
            self._cam_target[2],
            0.0,
            1.0,
            0.0,
        )

        # Update light position in view space
        glLightfv(GL_LIGHT0, GL_POSITION, [2.0, 5.0, 3.0, 0.0])

        draw_ground_grid()

        if self._renderer is not None:
            self._renderer.render()

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
            self._cam_azimuth += dx * 0.5
            self._cam_elevation -= dy * 0.5
            self._cam_elevation = max(-89.0, min(89.0, self._cam_elevation))

        elif event.buttons() & Qt.RightButton:
            # Pan: move target in the camera's local XY plane
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
        self._cam_distance = max(0.5, min(20.0, self._cam_distance))
        self.update()


# ── Control Panel ─────────────────────────────────────────────────────────────


class ControlPanel(QWidget):
    """Right-side panel with all simulation and playback controls."""

    def __init__(
        self, viewport: SkeletonViewport, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._viewport = viewport
        self.setMinimumWidth(280)
        self.setMaximumWidth(320)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(14)

        # ── Logo + author ──────────────────────────────────────────────────
        header = QWidget()
        h_layout = QVBoxLayout(header)
        h_layout.setContentsMargins(0, 0, 0, 0)
        h_layout.setSpacing(4)

        logo_label = QLabel()
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if LOGO_PATH.exists():
            pixmap = QPixmap(str(LOGO_PATH)).scaledToWidth(
                72, Qt.TransformationMode.SmoothTransformation
            )
            logo_label.setPixmap(pixmap)
        else:
            logo_label.setText("AMS Group – TU Ilmenau")
            logo_label.setStyleSheet("color: #4f9cf9; font-size: 10px;")

        author_label = QLabel("Renato Fera P.")
        author_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        author_label.setStyleSheet(
            "color: #5a6175; font-size: 10px; letter-spacing: 0.5px;"
        )

        chapter_label = QLabel("Chapter 05 · Physics-based Animation")
        chapter_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        chapter_label.setStyleSheet(
            "color: #4f9cf9; font-size: 9px; letter-spacing: 0.3px;"
        )
        chapter_label.setWordWrap(True)

        h_layout.addWidget(logo_label)
        h_layout.addWidget(author_label)
        h_layout.addWidget(chapter_label)
        layout.addWidget(header)

        # ── Divider ────────────────────────────────────────────────────────
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color: #252b38;")
        layout.addWidget(line)

        # ── Motion preset ──────────────────────────────────────────────────
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

        # ── Playback controls ──────────────────────────────────────────────
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

        # Speed
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

        # ── Camera controls ────────────────────────────────────────────────
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

        # ── Info ───────────────────────────────────────────────────────────
        info_group = QGroupBox("Info")
        info_group.setStyleSheet(self._group_style())
        i_layout = QFormLayout(info_group)
        i_layout.setSpacing(6)

        style = "color: #4f9cf9; font-size: 11px; font-family: monospace;"
        self._info_motion = QLabel("—")
        self._info_frames = QLabel("—")
        self._info_fps_bvh = QLabel("—")
        self._info_fps_gl = QLabel("—")
        for lbl in (
            self._info_motion,
            self._info_frames,
            self._info_fps_bvh,
            self._info_fps_gl,
        ):
            lbl.setStyleSheet(style)

        i_layout.addRow(QLabel("Motion:"), self._info_motion)
        i_layout.addRow(QLabel("Frames:"), self._info_frames)
        i_layout.addRow(QLabel("BVH fps:"), self._info_fps_bvh)
        i_layout.addRow(QLabel("GL fps:"), self._info_fps_gl)
        for row in range(i_layout.rowCount()):
            lbl = i_layout.itemAt(row, QFormLayout.LabelRole)
            if lbl and lbl.widget():
                lbl.widget().setStyleSheet("color: #5a6175; font-size: 11px;")

        layout.addWidget(info_group)
        layout.addStretch()

        # Load first motion automatically
        self._on_motion_changed(0)

    # ── Motion options ──────────────────────────────────────────────────────

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

    # ── Slots ───────────────────────────────────────────────────────────────

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
        if self._viewport._bvh is not None:
            self._viewport._frame = 0
            self._viewport._frame_timer = 0.0
            if self._viewport._renderer:
                self._viewport._renderer.set_frame(0)
        self._viewport.update()

    def _on_speed_changed(self, value: int) -> None:
        speed = value / 100.0
        self._viewport.set_speed(speed)
        self._speed_value.setText(f"{speed:.1f}×")

    def _on_orbit_toggled(self, checked: bool) -> None:
        self._viewport.set_auto_orbit(checked)
        self._orbit_btn.setText("Auto-Orbit: ON" if checked else "Auto-Orbit: OFF")

    def update_info(self) -> None:
        """Called each frame to refresh the info panel."""
        self._update_info()

    def _update_info(self) -> None:
        vp = self._viewport
        if vp._bvh:
            fps_bvh = 1.0 / vp._bvh.frame_time if vp._bvh.frame_time > 0 else 0
            self._info_motion.setText(vp._motion_name[:24])
            self._info_frames.setText(f"{vp._frame + 1} / {vp._frame_count}")
            self._info_fps_bvh.setText(f"{fps_bvh:.0f} fps")
        self._info_fps_gl.setText(f"{vp._fps_display:.1f} fps")

    # ── Style helpers ───────────────────────────────────────────────────────

    def _group_style(self) -> str:
        return """
            QGroupBox {
                color: #d8dce6;
                font-size: 11px;
                font-weight: 500;
                border: 1px solid #252b38;
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 6px;
                color: #a78bfa;
            }
        """

    def _button_style(self, accent: bool = False) -> str:
        bg = "#1e3a5f" if accent else "#1a1f2b"
        hover = "#2a4a7f" if accent else "#252b38"
        colour = "#4f9cf9" if accent else "#d8dce6"
        return f"""
            QPushButton {{
                background: {bg};
                color: {colour};
                border: 1px solid #252b38;
                border-radius: 5px;
                padding: 5px 10px;
                font-size: 11px;
            }}
            QPushButton:hover {{
                background: {hover};
                border-color: #4f9cf9;
            }}
            QPushButton:checked {{
                background: #2a4a7f;
                border-color: #4f9cf9;
            }}
        """

    def _slider_style(self) -> str:
        return """
            QSlider::groove:horizontal {
                height: 4px;
                background: #252b38;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #4f9cf9;
                border: none;
                width: 12px;
                height: 12px;
                margin: -4px 0;
                border-radius: 6px;
            }
            QSlider::sub-page:horizontal {
                background: #1e3a5f;
                border-radius: 2px;
            }
        """

    def _combo_style(self) -> str:
        return """
            QComboBox {
                background: #1a1f2b;
                color: #d8dce6;
                border: 1px solid #252b38;
                border-radius: 5px;
                padding: 4px 8px;
                font-size: 11px;
            }
            QComboBox:hover { border-color: #4f9cf9; }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background: #161a22;
                color: #d8dce6;
                selection-background-color: #1e3a5f;
                border: 1px solid #252b38;
            }
        """


# ── Main Window ───────────────────────────────────────────────────────────────


class MainWindow(QMainWindow):
    """Application main window.

    QMainWindow with:
    - SkeletonViewport as the central widget (the 3D view)
    - ControlPanel docked on the right
    - QTimer driving the animation at RENDER_FPS
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)
        self._apply_dark_theme()
        self._build_ui()
        self._start_timer()
        self._check_assets()

    def _build_ui(self) -> None:
        # Central viewport
        self._viewport = SkeletonViewport()
        self.setCentralWidget(self._viewport)

        # Control panel dock
        dock = QDockWidget("Controls", self)
        dock.setFeatures(
            QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable
        )
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        self._panel = ControlPanel(self._viewport)
        dock.setWidget(self._panel)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

        dock.setStyleSheet("""
            QDockWidget {
                color: #d8dce6;
                font-size: 11px;
                titlebar-close-icon: none;
            }
            QDockWidget::title {
                background: #161a22;
                padding: 6px 10px;
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
        dt = now - self._last_tick
        self._last_tick = now
        dt = min(dt, 0.1)  # clamp — prevents spiral-of-death on freeze

        self._viewport.advance_frame(dt)
        self._panel.update_info()

    def _apply_dark_theme(self) -> None:
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background: #0d0f14;
                color: #d8dce6;
                font-family: 'IBM Plex Sans', 'Segoe UI', sans-serif;
            }
            QScrollBar:vertical {
                background: #161a22;
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: #252b38;
                border-radius: 4px;
            }
        """)

    def _check_assets(self) -> None:
        """Warn in the title bar if required asset files are missing."""
        missing = []
        for path in (BVH_WALK, BVH_RUN, BVH_DANCE):
            if not path.exists():
                missing.append(path.name)
        if missing:
            names = ", ".join(missing)
            self.setWindowTitle(f"{WINDOW_TITLE}  ⚠  Missing: {names}")
            print("\n⚠  Missing asset files in", SCRIPT_DIR)
            print("   Required:")
            print(
                "   • human-skeleton-rigged.glb "
                "→ https://grab3d.com/dl/human-skeleton-rigged/"
                "human-skeleton-rigged.glb"
            )
            print(
                "   • 08_01.bvh  → "
                "github.com/una-dinosauria/cmu-mocap  data/008/08_01.bvh"
            )
            print(
                "   • 09_01.bvh  → "
                "github.com/una-dinosauria/cmu-mocap  data/009/09_01.bvh"
            )
            print(
                "   • 60_01.bvh  → "
                "github.com/una-dinosauria/cmu-mocap  data/060/60_01.bvh"
            )


# ── Entry Point ───────────────────────────────────────────────────────────────


def main() -> None:
    """Create the QApplication, show the window, and start the event loop.

    ANIMATION CONCEPT — the event loop:
    app.exec() blocks here and runs Qt's event loop. The QTimer fires at
    RENDER_FPS and calls _tick(), which advances the BVH animation and
    triggers a repaint. This is the same event-loop-plus-timer pattern
    that drives requestAnimationFrame in the browser demos.

    PHYSICS CONCEPT — kinematic replay:
    For this demo the physics are encoded in the BVH data (real capture
    from a physical human body). A natural extension — toggling a ragdoll
    blend slider — would hand the joint positions to PyBullet as kinematic
    targets and let the physics engine apply secondary motion on top.
    That is left as an exercise and is documented in the chapter theory.
    """
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
