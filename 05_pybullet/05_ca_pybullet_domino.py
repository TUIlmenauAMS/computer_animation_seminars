# =============================================================================
# 05_ca_pybullet_domino.py — Domino Cascade  ·  PyBullet Physics + OpenGL 3.3
# =============================================================================
# Author   : Renato Fera P.
# LinkedIn : https://www.linkedin.com/in/renato-profeta/
# Course   : Computeranimation — Chapter 05: Physics-based Animation
# Institute: Institut für Medientechnik, Technische Universität Ilmenau
# -----------------------------------------------------------------------------
"""05_ca_pybullet_domino.py — Professional Domino Cascade Demo.

Architecture — physics and rendering are fully decoupled:
  · PyBullet DIRECT  — headless rigid-body simulation, no rendering
  · OpenGL 3.3 Core  — QOpenGLWidget, Blinn-Phong, two-light setup,
                        checkerboard floor, 4× MSAA
  · Interactive camera — orbit (left-drag), pan (mid-drag), zoom (scroll),
                          reframe (double-click), optional wave-front auto-track

Layout strategies:
  · Staircase Canyon (C): dominoes descend and ascend elevated platforms,
    converting gravitational PE → KE → PE.
  · Branching Tree   (D): one falling domino physically contacts two targets
    → wave front splits → 4 simultaneous final arms.

PHYSICS CONCEPTS TAUGHT
------------------------
1. Rigid body dynamics       — mass, inertia tensor, restitution, friction
2. Collision detection       — AABB broadphase + GJK/EPA narrowphase
3. Impulse propagation       — angular momentum transferred on contact
4. Gravitational PE → KE     — staircase height drives cascade speed
5. Branching impulse chains  — fork geometry lets one tile reach two targets
6. Fixed-timestep loop       — stepSimulation at 1/240 s, decoupled from fps
7. Physics/render separation — PyBullet for simulation, OpenGL for display
8. GPU instanced transforms  — per-domino model matrix from physics pose

How to run
----------
    uv sync --extra arm64
    uv run python 05_ca_pybullet_domino.py

Ubuntu:  sudo apt install libgl1 libegl1 libxcb-cursor0 libxcb-xinerama0
"""
# =============================================================================

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pybullet
import pybullet_data
from OpenGL import GL
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QPixmap, QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLShader, QOpenGLShaderProgram
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

# =============================================================================
# ── Constants ─────────────────────────────────────────────────────────────────
# =============================================================================

WINDOW_TITLE = (
    "Domino Cascade — Computeranimation Chapter 05  ·  PyBullet + OpenGL 3.3"
)
LOGO_PATH: Path = Path(__file__).resolve().parent.parent / "ams_group_tui.png"

# ── Physics ───────────────────────────────────────────────────────────────────
PHYSICS_DT: float      = 1.0 / 240.0   # 240 Hz — stable for thin bodies

# ── Domino geometry (metres) — real proportions H:W:D ≈ 2:1:0.25 ─────────────
DOM_H: float           = 0.30
DOM_W: float           = 0.14
DOM_D: float           = 0.030
DOM_MASS: float        = 0.090   # kg
DOM_RESTITUTION: float = 0.15
DOM_FRICTION: float    = 0.70

# ── Layout geometry ───────────────────────────────────────────────────────────
DOMINO_SPACING: float  = 0.155   # slightly tighter for denser chains
STEP_HEIGHT: float     = 0.045   # shallower steps → more steps per ramp
STEP_DEPTH: float      = 0.155
STEPS_PER_RAMP: int    = 10      # deeper staircases
FORK_ANGLE_DEG: float  = 28.0

# ── Speed control ─────────────────────────────────────────────────────────────
# 4 substeps × (1/240 s) × 60 Hz timer = exactly 1× real-time.
# The speed table maps human-readable labels to substep counts.
REAL_TIME_SUBSTEPS: int = 4
SPEED_TABLE: list[tuple[str, int]] = [
    ("½×",  2),
    ("1×",  4),   # default — real-time
    ("2×",  8),
    ("4×", 16),
    ("8×", 32),
]
SPEED_DEFAULT_IDX: int = 1   # index into SPEED_TABLE → "1×" real-time

# ── Domino body gradient (blue → cyan → green → gold → red) ──────────────────
BODY_GRADIENT: list[tuple[float, float, float]] = [
    (0.08, 0.20, 0.72),
    (0.04, 0.72, 0.82),
    (0.12, 0.76, 0.22),
    (0.88, 0.72, 0.04),
    (0.82, 0.12, 0.08),
]

# ── UI palette — identical to demo 04 ────────────────────────────────────────
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
    background-color: {PALETTE["bg"]};
    color: {PALETTE["text"]};
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 13px;
}}

/* ── Generic button base ─────────────────────────────────────────────────── */
QPushButton {{
    background-color: {PALETTE["surface"]};
    color: {PALETTE["text"]};
    border: 1px solid {PALETTE["border"]};
    border-radius: 5px;
    padding: 6px 16px;
    font-weight: 600;
    font-size: 13px;
}}
QPushButton:hover  {{
    background-color: {PALETTE["border"]};
    color: {PALETTE["text"]};
    border-color: {PALETTE["accent"]};
}}
QPushButton:pressed {{ background-color: {PALETTE["bg"]}; }}

/* ── Start / Reset buttons: rely on per-widget setStyleSheet instead ─────── */
/* (Fusion style ignores app-level #id selectors on some platforms)          */

/* ── Stat cards ─────────────────────────────────────────────────────────── */
QWidget#stat_card {{
    background-color: {PALETTE["surface"]};
    border: 1px solid {PALETTE["border"]};
    border-radius: 6px;
}}

/* ── Section headers ────────────────────────────────────────────────────── */
QLabel#section_header {{
    color: {PALETTE["muted"]};
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.10em;
    padding: 10px 0 4px 0;
}}

/* ── Dividers ───────────────────────────────────────────────────────────── */
QFrame#divider {{
    background-color: {PALETTE["border"]};
    max-height: 1px;
}}

/* ── Status bar ─────────────────────────────────────────────────────────── */
QStatusBar {{
    background-color: {PALETTE["surface"]};
    color: {PALETTE["muted"]};
    border-top: 1px solid {PALETTE["border"]};
    font-size: 11px;
}}

/* ── Checkbox ───────────────────────────────────────────────────────────── */
QCheckBox {{
    color: {PALETTE["text"]};
    spacing: 7px;
    font-size: 12px;
}}
QCheckBox::indicator {{
    width: 14px; height: 14px;
    border: 1px solid {PALETTE["border"]};
    border-radius: 3px;
    background-color: {PALETTE["surface"]};
}}
QCheckBox::indicator:checked {{
    background-color: {PALETTE["accent"]};
    border-color: {PALETTE["accent"]};
}}

/* ── Splitter ───────────────────────────────────────────────────────────── */
QSplitter::handle {{
    background-color: {PALETTE["border"]};
    width: 1px;
}}
"""

# Inline styles for speed chips — set directly via setStyleSheet() on each
# button so we never depend on Qt property selectors, which are unreliable
# on Windows with the Fusion style.
_CHIP_INACTIVE = (
    f"QPushButton {{ background-color: {PALETTE['surface']};"
    f" color: {PALETTE['muted']};"
    f" border: 1px solid {PALETTE['border']};"
    f" border-radius: 5px; font-size: 12px; font-weight: 600;"
    f" padding: 4px 12px; min-width: 38px; }}"
    f"QPushButton:hover {{ background-color: {PALETTE['border']};"
    f" color: {PALETTE['text']}; border-color: {PALETTE['accent']}; }}"
)
_CHIP_ACTIVE = (
    f"QPushButton {{ background-color: {PALETTE['accent']};"
    f" color: #03101e;"
    f" border: 1px solid {PALETTE['accent']};"
    f" border-radius: 5px; font-size: 12px; font-weight: 700;"
    f" padding: 4px 12px; min-width: 38px; }}"
    f"QPushButton:hover {{ background-color: #72b4ff;"
    f" border-color: #72b4ff; color: #03101e; }}"
)

# =============================================================================
# ── GLSL shaders ──────────────────────────────────────────────────────────────
# =============================================================================

# Vertex shader — standard MVP transform, world-space normal propagation.
_VERT_SRC = """
#version 330 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProj;

out vec3 vNormal;
out vec3 vFragPos;

void main() {
    vec4 worldPos = uModel * vec4(aPos, 1.0);
    gl_Position   = uProj * uView * worldPos;
    vFragPos      = worldPos.xyz;
    // Inverse-transpose handles non-uniform scale without distorting normals.
    vNormal = normalize(mat3(transpose(inverse(uModel))) * aNormal);
}
"""

# Fragment shader — two-light Blinn-Phong + gamma correction.
# Key light (warm, strong) + fill light (cool, dim) gives cinematic depth.
# uFloor activates a checkerboard pattern computed from world-space XY
# position, giving spatial scale reference without extra geometry.
_FRAG_SRC = """
#version 330 core

in vec3 vNormal;
in vec3 vFragPos;

uniform vec3  uLight1Pos;    // key light — warm, upper-left
uniform vec3  uLight2Pos;    // fill light — cool, opposite side
uniform vec3  uLight1Color;
uniform vec3  uLight2Color;
uniform vec4  uBaseColor;
uniform vec3  uCamPos;       // correct specular for any viewpoint
uniform float uAmbient;
uniform float uSpecPower;
uniform bool  uFloor;

out vec4 FragColor;

void main() {
    vec4 col = uBaseColor;

    // ── Checkerboard for the floor ────────────────────────────────────────
    if (uFloor) {
        float cx = floor(vFragPos.x * 2.0);   // 0.5-metre squares
        float cy = floor(vFragPos.y * 2.0);
        float c  = mod(cx + cy, 2.0);
        col.rgb  = mix(col.rgb * 0.48, col.rgb * 1.35, c);
        // Fade toward the edges so there is no hard border
        float dist = length(vFragPos.xy);
        col.rgb *= 1.0 - smoothstep(14.0, 20.0, dist) * 0.6;
    }

    vec3 N = normalize(vNormal);
    vec3 V = normalize(uCamPos - vFragPos);

    // ── Key light (warm) ──────────────────────────────────────────────────
    vec3  L1   = normalize(uLight1Pos - vFragPos);
    float d1   = max(dot(N, L1), 0.0);
    vec3  H1   = normalize(L1 + V);
    float s1   = pow(max(dot(N, H1), 0.0), uSpecPower);

    // ── Fill light (cool, 35% intensity) ─────────────────────────────────
    vec3  L2   = normalize(uLight2Pos - vFragPos);
    float d2   = max(dot(N, L2), 0.0) * 0.35;
    vec3  H2   = normalize(L2 + V);
    float s2   = pow(max(dot(N, H2), 0.0), uSpecPower * 0.4) * 0.12;

    vec3 ambient  = uAmbient * col.rgb;
    vec3 diffuse  = d1 * col.rgb * uLight1Color
                  + d2 * col.rgb * uLight2Color;
    vec3 specular = s1 * uLight1Color * 0.55
                  + s2 * uLight2Color;

    vec3 result = ambient + diffuse + specular;

    // ── Gamma-encode for sRGB display (γ = 1/2.2) ────────────────────────
    result    = pow(clamp(result, 0.0, 1.0), vec3(1.0 / 2.2));
    FragColor = vec4(result, col.a);
}
"""

# =============================================================================
# ── Maths helpers ─────────────────────────────────────────────────────────────
# =============================================================================

def _lerp3(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    t: float,
) -> tuple[float, float, float]:
    return (a[0]+(b[0]-a[0])*t, a[1]+(b[1]-a[1])*t, a[2]+(b[2]-a[2])*t)


def body_color(index: int, total: int) -> tuple[float, float, float]:
    """Map a domino index to a colour along the five-stop gradient."""
    t  = index / max(total - 1, 1)
    s  = t * (len(BODY_GRADIENT) - 1)
    lo = int(s); hi = min(lo + 1, len(BODY_GRADIENT) - 1)
    return _lerp3(BODY_GRADIENT[lo], BODY_GRADIENT[hi], s - lo)


def _quat_to_mat4(quat: tuple | list) -> np.ndarray:
    """
    Unit quaternion (x, y, z, w) → row-major 4×4 rotation matrix.

    This is the standard formula from expanding q * v * q⁻¹:
    each entry in the 3×3 block is a quadratic in {x, y, z, w}.
    PyBullet returns quaternions in (x, y, z, w) order.
    """
    x, y, z, w = quat
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-z*w),   2*(x*z+y*w),  0],
        [  2*(x*y+z*w), 1-2*(x*x+z*z),   2*(y*z-x*w),  0],
        [  2*(x*z-y*w),   2*(y*z+x*w), 1-2*(x*x+y*y),  0],
        [            0,             0,             0,   1],
    ], dtype=np.float32)


def _build_model_mat(
    pos: tuple | list,
    orn: tuple | list,
    sx: float, sy: float, sz: float,
) -> np.ndarray:
    """
    Build a row-major 4×4 model matrix from a physics body pose + scale.

    Composition order: T × R × S
    (scale first, then rotate in scaled space, then translate)
    This is the standard SRT convention used by every major 3-D engine.
    """
    T = np.eye(4, dtype=np.float32)
    T[0, 3] = pos[0]; T[1, 3] = pos[1]; T[2, 3] = pos[2]
    R = _quat_to_mat4(orn)
    S = np.diag([sx, sy, sz, 1.0]).astype(np.float32)
    return T @ R @ S


# =============================================================================
# ── Domino layout ─────────────────────────────────────────────────────────────
# =============================================================================

class DP(NamedTuple):
    """Domino placement: world-space bottom-centre + yaw (radians)."""
    x: float; y: float; z: float; yaw: float


def _run(ox: float, oy: float, oz: float, yaw: float, n: int,
         sp: float = DOMINO_SPACING) -> list[DP]:
    dx, dy = math.cos(yaw) * sp, math.sin(yaw) * sp
    return [DP(ox + dx*i, oy + dy*i, oz, yaw) for i in range(n)]


def _descend(ox: float, oy: float, oz: float, yaw: float, n: int) -> list[DP]:
    """
    Staircase descent — PHYSICS: PE = m·g·h accumulated over n×STEP_HEIGHT.
    Each step converts potential energy → kinetic; the cascade accelerates
    downhill, arriving at the fork with extra momentum.
    """
    pl: list[DP] = []
    dx, dy = math.cos(yaw), math.sin(yaw)
    x, y, z = ox, oy, oz
    for _ in range(n):
        pl.append(DP(x, y, z, yaw))
        x += dx * STEP_DEPTH; y += dy * STEP_DEPTH; z -= STEP_HEIGHT
    return pl


def _ascend(ox: float, oy: float, oz: float, yaw: float, n: int) -> list[DP]:
    """
    Staircase ascent — demonstrates KE→PE: the branch must arrive with
    enough kinetic energy at the base to push dominoes uphill.
    If spacing is too large, the chain stalls — a vivid teaching moment.
    """
    pl: list[DP] = []
    dx, dy = math.cos(yaw), math.sin(yaw)
    x, y, z = ox, oy, oz
    for _ in range(n):
        pl.append(DP(x, y, z, yaw))
        x += dx * STEP_DEPTH; y += dy * STEP_DEPTH; z += STEP_HEIGHT
    return pl


def _arc(
    cx: float, cy: float, cz: float,
    radius: float, start_ang: float, end_ang: float,
    n: int,
) -> list[DP]:
    """
    Circular arc of n dominoes.  Each domino is tangent to the arc so it
    naturally faces the next one in the chain.
    """
    pl: list[DP] = []
    for i in range(n):
        t   = i / max(n - 1, 1)
        ang = start_ang + (end_ang - start_ang) * t
        x   = cx + radius * math.cos(ang)
        y   = cy + radius * math.sin(ang)
        # tangent direction = perpendicular to radial direction
        yaw = ang + math.pi * 0.5 * (1 if end_ang > start_ang else -1)
        pl.append(DP(x, y, cz, yaw))
    return pl


def build_layout() -> tuple[list[DP], list[int]]:
    """
    Cascade layout — ~260 dominoes in five acts, all on the ground plane
    or on proper staircase steps.  Every segment connects flush to the
    previous one so the wave propagates without gaps.

    ACT 1 · Single straight entry (8 dominoes) — one tile is pushed.
    ACT 2 · S-curve on the ground — adds visual interest.
    ACT 3 · Staircase descent (8 steps) → PE→KE acceleration.
    ACT 4 · Ground run then staircase ascent (6 steps) → KE→PE.
    ACT 5 · Fork: one domino fans two branches; each branch forks again
              → 4 final straight arms of 20 dominoes each.

    Returns
    -------
    pl  : ordered list of DP (bottom-centre position + yaw)
    fi  : indices in pl of the two fork dominoes (for InfoPanel display)
    """
    pl: list[DP] = []
    fi: list[int] = []

    SP  = DOMINO_SPACING          # 0.155 m — standard spacing
    fr  = math.radians(FORK_ANGLE_DEG)   # fork half-angle

    # ── ACT 1 — Straight entry, heading east (+X) ─────────────────────────
    # First domino is at the origin facing east.  The player pushes domino 0.
    entry = _run(0.0, 0.0, 0.0, 0.0, 10)
    pl.extend(entry)

    # ── ACT 2 — Staircase ascent to plateau (8 steps) ─────────────────────
    # The first domino is on the ground.  We build steps rising upward so
    # the descent in ACT 3 has real height → PE→KE conversion.
    N_STEPS = 8
    plateau_z = N_STEPS * STEP_HEIGHT

    p = entry[-1]
    rise = _ascend(
        p.x + math.cos(p.yaw) * STEP_DEPTH,
        p.y + math.sin(p.yaw) * STEP_DEPTH,
        STEP_HEIGHT, p.yaw, N_STEPS,
    )
    pl.extend(rise)

    # S-curve on the plateau
    p2 = rise[-1]
    # Arc 1: sweep left (CCW) ~80° with radius 1.6 m
    cx1 = p2.x + math.cos(p2.yaw + math.pi * 0.5) * 1.6
    cy1 = p2.y + math.sin(p2.yaw + math.pi * 0.5) * 1.6
    sa1 = p2.yaw - math.pi * 0.5
    ea1 = sa1 + math.radians(80)
    arc1 = _arc(cx1, cy1, plateau_z, 1.6, sa1, ea1, 10)
    pl.extend(arc1)

    # Arc 2: sweep right (CW) ~80° — completes the S
    p3 = arc1[-1]
    cx2 = p3.x + math.cos(p3.yaw - math.pi * 0.5) * 1.6
    cy2 = p3.y + math.sin(p3.yaw - math.pi * 0.5) * 1.6
    sa2 = p3.yaw + math.pi * 0.5
    ea2 = sa2 - math.radians(80)
    arc2 = _arc(cx2, cy2, plateau_z, 1.6, sa2, ea2, 10)
    pl.extend(arc2)

    # Short straight on plateau after S
    p4 = arc2[-1]
    link1 = _run(
        p4.x + math.cos(p4.yaw) * SP,
        p4.y + math.sin(p4.yaw) * SP,
        plateau_z, p4.yaw, 6,
    )
    pl.extend(link1)

    # ── ACT 3 — Staircase descent back to ground ───────────────────────────
    p5 = link1[-1]
    descent = _descend(
        p5.x + math.cos(p5.yaw) * STEP_DEPTH,
        p5.y + math.sin(p5.yaw) * STEP_DEPTH,
        plateau_z - STEP_HEIGHT, p5.yaw, N_STEPS,
    )
    pl.extend(descent)

    # Ground run at the bottom
    p6 = descent[-1]
    ground1 = _run(
        p6.x + math.cos(p6.yaw) * SP,
        p6.y + math.sin(p6.yaw) * SP,
        0.0, p6.yaw, 10,
    )
    pl.extend(ground1)

    # ── ACT 4 — Second ascent (6 steps) → elevated fork platform ──────────
    p7 = ground1[-1]
    N_STEPS2 = 6
    fork_z = N_STEPS2 * STEP_HEIGHT

    ascent = _ascend(
        p7.x + math.cos(p7.yaw) * STEP_DEPTH,
        p7.y + math.sin(p7.yaw) * STEP_DEPTH,
        STEP_HEIGHT, p7.yaw, N_STEPS2,
    )
    pl.extend(ascent)

    # Short plateau run before the fork domino
    p8 = ascent[-1]
    pre_fork = _run(
        p8.x + math.cos(p8.yaw) * SP,
        p8.y + math.sin(p8.yaw) * SP,
        fork_z, p8.yaw, 6,
    )
    pl.extend(pre_fork)

    # ── ACT 5 — Fork 1 → two branches, each descends then forks → 4 arms ──
    # Fork geometry rule: the first domino of each branch must be within
    # DOM_H * sin(fork_angle) ≈ 0.14 m of the fork domino so the tip of the
    # falling fork tile physically contacts it.  We use a reduced fork angle
    # and place the first branch domino at exactly SP along the branch heading.
    fi.append(len(pl) - 1)   # record first fork domino index
    pf = pre_fork[-1]
    fork_half = math.radians(22)   # tighter angle → first branch tile in reach
    sub_fr    = math.radians(18)   # sub-fork angle

    for branch_yaw in [pf.yaw + fork_half, pf.yaw - fork_half]:
        # First branch domino placed at SP along branch heading from fork
        bx = pf.x + math.cos(branch_yaw) * SP
        by = pf.y + math.sin(branch_yaw) * SP
        branch_top = _run(bx, by, fork_z, branch_yaw, 6)
        pl.extend(branch_top)

        # Descend from elevated fork to ground
        pb_top = branch_top[-1]
        branch_desc = _descend(
            pb_top.x + math.cos(branch_yaw) * STEP_DEPTH,
            pb_top.y + math.sin(branch_yaw) * STEP_DEPTH,
            fork_z - STEP_HEIGHT, branch_yaw, N_STEPS2,
        )
        pl.extend(branch_desc)

        # Ground run before the second fork
        pb_gnd = branch_desc[-1]
        branch_ground = _run(
            pb_gnd.x + math.cos(branch_yaw) * SP,
            pb_gnd.y + math.sin(branch_yaw) * SP,
            0.0, branch_yaw, 8,
        )
        pl.extend(branch_ground)

        # Second fork (at ground level) — same tight placement rule
        fi.append(len(pl) - 1)
        pb2 = branch_ground[-1]

        for arm_yaw in [pb2.yaw + sub_fr, pb2.yaw - sub_fr]:
            ax = pb2.x + math.cos(arm_yaw) * SP
            ay = pb2.y + math.sin(arm_yaw) * SP
            arm = _run(ax, ay, 0.0, arm_yaw, 20)
            pl.extend(arm)

    return pl, fi


# =============================================================================
# ── Interactive Camera ────────────────────────────────────────────────────────
# =============================================================================

class InteractiveCamera:
    """
    Orbit / pan / zoom camera with optional wave-front auto-track.

    Coordinate system: Z-up (matches PyBullet).
      · Eye position is computed in spherical coords from yaw + pitch + dist.
      · Left-drag  → orbit (yaw + pitch)
      · Middle-drag → pan (translate target in camera's local XY)
      · Scroll      → zoom (scale distance)
      · Double-click → reframe (restore saved defaults)

    Auto-track mode: smoothly interpolates the target toward the physics
    wave-front position (a moving average of the most recently fallen dominoes).
    The smooth interpolation factor (dt * 2.0) keeps the camera slightly behind
    the cascade for a natural following feel — identical to the lag in a real
    camera truck following a practitioner.
    """

    def __init__(self) -> None:
        self._yaw   =  28.0   # horizontal angle (degrees)
        self._pitch =  24.0   # vertical tilt (degrees)
        self._dist  =  10.0   # distance from target (metres)
        self._tgt   = np.array([0.0, 0.0, 0.5], dtype=np.float32)
        self.auto_track: bool = False

        # Defaults for double-click reframe
        self._def_yaw   =  28.0
        self._def_pitch =  24.0
        self._def_dist  =  10.0
        self._def_tgt   = np.array([0.0, 0.0, 0.5], dtype=np.float32)

    @property
    def eye(self) -> np.ndarray:
        yr = math.radians(self._yaw)
        pr = math.radians(self._pitch)
        return self._tgt + np.array([
            self._dist * math.cos(pr) * math.cos(yr),
            self._dist * math.cos(pr) * math.sin(yr),
            self._dist * math.sin(pr),
        ], dtype=np.float32)

    def view_matrix(self) -> np.ndarray:
        """
        Standard lookAt construction — identical to glm::lookAt.
        Builds an orthonormal camera frame from forward/right/up vectors.
        """
        e  = self.eye
        t  = self._tgt
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # Z-up world
        f  = t - e;  f /= np.linalg.norm(f) + 1e-9  # forward
        r  = np.cross(f, up); r /= np.linalg.norm(r) + 1e-9  # right
        u  = np.cross(r, f)                                    # camera-up
        V  = np.eye(4, dtype=np.float32)
        V[0, :3] = r;  V[0, 3] = -np.dot(r, e)
        V[1, :3] = u;  V[1, 3] = -np.dot(u, e)
        V[2, :3] = -f; V[2, 3] =  np.dot(f, e)
        return V

    def proj_matrix(self, aspect: float) -> np.ndarray:
        """
        Perspective projection derived from frustum geometry.
        near/far scale with orbit distance so clipping adapts to zoom level.
        The −1 at [3,2] is the perspective divide that maps depth to W.
        """
        near = max(0.01, self._dist * 0.004)
        far  = max(100.0, self._dist * 160.0)
        fov  = math.radians(50.0)
        f    = 1.0 / math.tan(fov * 0.5)   # cot(fov/2)
        P    = np.zeros((4, 4), dtype=np.float32)
        P[0, 0] = f / aspect
        P[1, 1] = f
        P[2, 2] = (far + near) / (near - far)
        P[2, 3] = 2 * far * near / (near - far)
        P[3, 2] = -1.0  # perspective divide
        return P

    def orbit(self, dx: float, dy: float) -> None:
        self._yaw  += dx * 0.4
        self._pitch = max(-89.0, min(89.0, self._pitch + dy * 0.4))

    def pan(self, dx_px: float, dy_px: float, vp_h: int) -> None:
        """
        Translate the orbit target in the camera's local XY plane.
        Scale by orbit_dist/viewport_height so panning feels the same
        regardless of zoom level — matching Maya / Blender behaviour.
        """
        yr = math.radians(self._yaw); pr = math.radians(self._pitch)
        f  = -np.array([
            math.cos(pr)*math.cos(yr),
            math.cos(pr)*math.sin(yr),
            math.sin(pr),
        ])
        up = np.array([0., 0., 1.])
        r  = np.cross(f, up); r /= np.linalg.norm(r) + 1e-9
        u  = np.cross(r, f)
        s  = self._dist / max(1, vp_h)
        self._tgt -= r.astype(np.float32) * (dx_px * s)
        self._tgt += u.astype(np.float32) * (dy_px * s)

    def zoom(self, delta: float) -> None:
        """Proportional zoom — 0.999^120 ≈ 0.887 per scroll notch."""
        self._dist = max(0.3, min(80.0, self._dist * (0.999 ** delta)))

    def reframe(self) -> None:
        self._yaw = self._def_yaw; self._pitch = self._def_pitch
        self._dist = self._def_dist; self._tgt = self._def_tgt.copy()

    def update_wave(self, wx: float, wy: float, wz: float, dt: float) -> None:
        """Smooth-follow the wave-front when auto-track is enabled."""
        if self.auto_track:
            goal       = np.array([wx, wy, wz + 0.3], dtype=np.float32)
            self._tgt += (goal - self._tgt) * min(1.0, dt * 2.0)


# =============================================================================
# ── Physics World (PyBullet DIRECT — headless, no rendering) ──────────────────
# =============================================================================

class DominoPhysics:
    """
    Rigid-body simulation via PyBullet DIRECT (headless).

    This class is responsible for physics only.  It exposes three arrays
    that the OpenGL viewport reads each frame:
      · body_states    — list[(pos, orn)] per domino, refreshed after each step
      · platform_data  — static list[(pos, orn, hx, hy, hz)] for step platforms
      · body_colors    — gradient RGBA pre-computed once at build time

    PIPELINE CONCEPT
    ----------------
    connect(DIRECT)             → no GUI, no renderer
    setGravity                  → 9.81 m/s² downward (−Z)
    loadURDF("plane.urdf")      → infinite static ground plane (mass = 0)
    createCollisionShape(BOX)   → AABB for collision detection
    createMultiBody             → rigid body = shape + mass + pose
    changeDynamics              → restitution, friction, damping
    stepSimulation()            → integrate forces + resolve contacts
    getBasePositionAndOrientation() → query pose for rendering
    """

    def __init__(self) -> None:
        self._client     = -1
        self._dom_ids:   list[int] = []
        self._placements: list[DP] = []
        self._n_fallen   = 0
        self._sim_time   = 0.0
        self._running    = False
        self._wave: tuple[float, float, float] = (0.0, 0.0, 0.5)

        # Rendering data exposed to DominoViewport
        self.body_states:   list[tuple] = []
        self.platform_data: list[tuple] = []   # (pos, orn, hx, hy, hz)
        self.body_colors:   list[tuple] = []

    # ── Build ─────────────────────────────────────────────────────────────────

    def connect(self) -> None:
        self._client = pybullet.connect(pybullet.DIRECT)
        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        pybullet.setGravity(0, 0, -9.81)
        pybullet.setTimeStep(PHYSICS_DT)
        self._build_ground()
        self._build_scene()

    def _build_ground(self) -> None:
        """
        Infinite static ground plane via URDF.
        mass = 0 in PyBullet = infinite mass = unmovable by any force.
        """
        gid = pybullet.loadURDF("plane.urdf", useFixedBase=True)
        pybullet.changeDynamics(gid, -1, lateralFriction=DOM_FRICTION)

    def _build_scene(self) -> None:
        """
        Build step platforms and domino rigid bodies.

        STEP PLATFORMS
        --------------
        Each step is a GEOM_BOX slab under exactly one domino (mass = 0 →
        static).  We store their pose and half-extents in platform_data
        for the renderer to draw — no per-frame PyBullet query needed.

        DOMINOES
        --------
        GEOM_BOX (DOM_D × DOM_W × DOM_H), mass = DOM_MASS.
        PyBullet auto-computes the inertia tensor from mass + geometry.
        The domino stands with its centre of mass at p.z + DOM_H/2.
        """
        self._placements, _ = build_layout()

        # ── Step platforms ────────────────────────────────────────────────────
        # Each domino that is NOT at z ≈ 0 sits on a solid step slab.
        # The slab half-extents span the full step depth and are tall enough
        # to reach from z down to the ground (or the next step surface).
        # This creates a proper staircase the physics engine can reason about.
        step_hw = STEP_DEPTH * 0.6     # half-depth along travel direction
        step_ww = DOM_W * 1.1          # slightly wider than domino
        self.platform_data = []

        for p in self._placements:
            if abs(p.z) < 0.004:        # at ground level — plane.urdf handles it
                continue
            # Slab top surface = p.z; slab bottom = ground (z=0) or negative floor.
            # Half-height = |p.z| / 2; slab centre at p.z / 2.
            step_hh = abs(p.z) * 0.5 + 0.002   # tiny extra to avoid gap
            pos = (p.x, p.y, p.z * 0.5)
            q   = pybullet.getQuaternionFromEuler([0, 0, p.yaw])
            self.platform_data.append((pos, q, step_hw, step_ww, step_hh))
            sh  = pybullet.createCollisionShape(
                pybullet.GEOM_BOX,
                halfExtents=[step_hw, step_ww, step_hh],
            )
            bid = pybullet.createMultiBody(0, sh, -1, list(pos), list(q))
            pybullet.changeDynamics(bid, -1, lateralFriction=DOM_FRICTION)

        # ── Domino rigid bodies ───────────────────────────────────────────────
        dom_col = pybullet.createCollisionShape(
            pybullet.GEOM_BOX,
            halfExtents=[DOM_D*0.5, DOM_W*0.5, DOM_H*0.5],
        )
        total = len(self._placements)
        for i, p in enumerate(self._placements):
            q   = pybullet.getQuaternionFromEuler([0, 0, p.yaw])
            pos = [p.x, p.y, p.z + DOM_H * 0.5]
            bid = pybullet.createMultiBody(DOM_MASS, dom_col, -1, pos, list(q))
            pybullet.changeDynamics(
                bid, -1,
                restitution   = DOM_RESTITUTION,
                lateralFriction = DOM_FRICTION,
                linearDamping = 0.02,
                angularDamping = 0.02,
            )
            self._dom_ids.append(bid)
            self.body_colors.append(body_color(i, total))

        self._sync_states()

    def _sync_states(self) -> None:
        """Query every domino's current pose from PyBullet."""
        self.body_states = [
            pybullet.getBasePositionAndOrientation(bid)
            for bid in self._dom_ids
        ]

    # ── Control ───────────────────────────────────────────────────────────────

    def start(self) -> None:
        """
        Tip ONLY the first domino (index 0) with a torque + linear impulse.

        CONCEPT: applyExternalTorque adds an angular impulse (N·m).
        Once tilt exceeds arctan(DOM_D/DOM_H) ≈ 5.7°, gravity takes over.
        Torque axis is perpendicular to the domino heading so it falls
        forward (into the chain), not sideways.

        A complementary linear velocity nudge along the domino's facing
        direction guarantees the tip even if the torque is absorbed by the
        ground contact point, ensuring one — and only one — tile is pushed.
        """
        if not self._dom_ids:
            return
        self._running = True
        p0  = self._placements[0]
        bid = self._dom_ids[0]

        # Angular impulse — perpendicular to heading, tilts it forward
        pybullet.applyExternalTorque(
            bid, -1,
            torqueObj=[-math.sin(p0.yaw) * 12.0,
                        math.cos(p0.yaw) * 12.0,
                        0.0],
            flags=pybullet.WORLD_FRAME,
        )
        # Small linear nudge along the facing direction for extra reliability
        pybullet.resetBaseVelocity(
            bid,
            linearVelocity =[math.cos(p0.yaw) * 0.4,
                              math.sin(p0.yaw) * 0.4,
                              0.0],
            angularVelocity=[0.0, 0.0, 0.0],
        )

    def step(self, n: int) -> None:
        """
        Advance n physics sub-steps.

        ANIMATION CONCEPT — fixed-timestep loop:
        QTimer fires at ~60 Hz. Each tick advances n×PHYSICS_DT simulation.
        n = REAL_TIME_SUBSTEPS (4) → 4×(1/240) ≈ 16.7 ms/frame = 1× real-time.
        n = 8  → 2× real-time.  n = 32 → 8× real-time.
        Physics accuracy is unchanged — only temporal compression varies.
        """
        if not self._running:
            return
        for _ in range(n):
            pybullet.stepSimulation()
            self._sim_time += PHYSICS_DT
        self._sync_states()
        self._update_wave()

    def _update_wave(self) -> None:
        """
        Track wave front = average position of the last 5 fallen dominoes.
        A domino is 'fallen' when its total tilt exceeds 40°.
        """
        fallen: list[tuple] = []
        cnt = 0
        for bid in self._dom_ids:
            pos, orn = pybullet.getBasePositionAndOrientation(bid)
            e = pybullet.getEulerFromQuaternion(orn)
            if abs(e[0]) + abs(e[1]) > math.radians(40):
                cnt += 1; fallen.append(pos)
        self._n_fallen = cnt
        if fallen:
            rec = fallen[-5:]
            self._wave = (
                sum(v[0] for v in rec) / len(rec),
                sum(v[1] for v in rec) / len(rec),
                sum(v[2] for v in rec) / len(rec),
            )

    def reset(self) -> None:
        if self._client >= 0:
            pybullet.disconnect(self._client)
        self._client = -1; self._dom_ids = []; self._placements = []
        self._n_fallen = 0; self._sim_time = 0.0; self._running = False
        self._wave = (0.0, 0.0, 0.5)
        self.body_states = []; self.platform_data = []; self.body_colors = []
        self.connect()

    def disconnect(self) -> None:
        if self._client >= 0:
            pybullet.disconnect(self._client); self._client = -1

    @property
    def n_fallen(self)   -> int:   return self._n_fallen
    @property
    def total(self)      -> int:   return len(self._dom_ids)
    @property
    def sim_time(self)   -> float: return self._sim_time
    @property
    def is_running(self) -> bool:  return self._running
    @property
    def wave(self)       -> tuple: return self._wave


# =============================================================================
# ── OpenGL Viewport ───────────────────────────────────────────────────────────
# =============================================================================

class DominoViewport(QOpenGLWidget):
    """
    OpenGL 3.3 Core Profile rendering of the domino scene.

    RENDERING PIPELINE (per frame)
    --------------------------------
    1. DominoPhysics.body_states provides (pos, orn) per domino.
    2. For each body:
         model = T × R × S   (translation × rotation × scale)
         → upload as uModel uniform
         → draw body (gradient colour)
         → shift model along local X → draw ivory face stripe
    3. Same box VAO for all draws — only the model matrix and colour change.
    4. Floor: a single large quad at z = 0, checkerboard in fragment shader.

    CAMERA CONTROLS
    ---------------
    Left-drag    → orbit  (yaw + pitch around target)
    Middle-drag  → pan    (translate target in camera's local XY)
    Scroll wheel → zoom   (proportional distance scale)
    Double-click → reframe (restore saved defaults)
    """

    def __init__(self, physics: DominoPhysics, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._ph   = physics
        self.cam   = InteractiveCamera()
        self._prog: QOpenGLShaderProgram | None = None
        self._box_vao = self._box_vbo = self._box_ebo = 0
        self._box_idx_count = 0
        self._floor_vao = self._floor_vbo = self._floor_ebo = 0
        self._floor_idx_count = 0
        self._last_mouse = None
        self._pan_last   = None
        self._last_wall  = time.perf_counter()

        self.setMinimumSize(640, 400)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)

    # ── Qt OpenGL lifecycle ────────────────────────────────────────────────────

    def initializeGL(self) -> None:
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_CULL_FACE)   # dominoes show all faces while tumbling
        GL.glEnable(GL.GL_MULTISAMPLE)  # 4× MSAA from QSurfaceFormat

        # Clear colour from the UI palette
        GL.glClearColor(
            int(PALETTE["bg"][1:3], 16) / 255,
            int(PALETTE["bg"][3:5], 16) / 255,
            int(PALETTE["bg"][5:7], 16) / 255,
            1.0,
        )

        # Compile and link shaders
        self._prog = QOpenGLShaderProgram(self)
        ok = self._prog.addShaderFromSourceCode(
            QOpenGLShader.ShaderTypeBit.Vertex, _VERT_SRC)
        if not ok:
            print(f"[VP] Vertex shader error:\n{self._prog.log()}")
        ok = self._prog.addShaderFromSourceCode(
            QOpenGLShader.ShaderTypeBit.Fragment, _FRAG_SRC)
        if not ok:
            print(f"[VP] Fragment shader error:\n{self._prog.log()}")
        if not self._prog.link():
            print(f"[VP] Shader link error:\n{self._prog.log()}")

        self._make_box_vao()
        self._make_floor_vao()

    def resizeGL(self, w: int, h: int) -> None:
        GL.glViewport(0, 0, w, h)

    def paintGL(self) -> None:
        # Advance auto-track camera
        now = time.perf_counter()
        dt  = now - self._last_wall
        self._last_wall = now
        self.cam.update_wave(*self._ph.wave, dt)

        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        if self._prog is None:
            return

        self._prog.bind()
        pid = self._prog.programId()

        w, h = max(1, self.width()), max(1, self.height())
        view = self.cam.view_matrix()
        proj = self.cam.proj_matrix(w / h)
        eye  = self.cam.eye

        def _u(name: str) -> int:
            return GL.glGetUniformLocation(pid, name)

        # ── Shared uniforms ───────────────────────────────────────────────────
        GL.glUniformMatrix4fv(_u("uView"), 1, GL.GL_TRUE, view)
        GL.glUniformMatrix4fv(_u("uProj"), 1, GL.GL_TRUE, proj)
        # Key light — warm upper-left
        GL.glUniform3fv(_u("uLight1Pos"),   1, [5.0, -8.0, 12.0])
        GL.glUniform3fv(_u("uLight1Color"), 1, [1.0, 0.95, 0.88])
        # Fill light — cool opposite side
        GL.glUniform3fv(_u("uLight2Pos"),   1, [-8.0, 5.0, 5.0])
        GL.glUniform3fv(_u("uLight2Color"), 1, [0.5, 0.65, 1.0])
        GL.glUniform3fv(_u("uCamPos"),      1, eye.tolist())

        identity = np.eye(4, dtype=np.float32)

        # ── Floor (large quad, checkerboard) ─────────────────────────────────
        GL.glUniform1f(_u("uAmbient"),   0.35)
        GL.glUniform1f(_u("uSpecPower"), 6.0)
        GL.glUniform1i(_u("uFloor"),     1)
        GL.glUniformMatrix4fv(_u("uModel"), 1, GL.GL_TRUE, identity)
        GL.glUniform4f(_u("uBaseColor"), 0.11, 0.13, 0.17, 1.0)
        GL.glBindVertexArray(self._floor_vao)
        GL.glDrawElements(GL.GL_TRIANGLES, self._floor_idx_count,
                          GL.GL_UNSIGNED_INT, None)
        GL.glUniform1i(_u("uFloor"), 0)

        GL.glBindVertexArray(self._box_vao)

        # ── Step platforms (dark concrete slate) ─────────────────────────────
        GL.glUniform1f(_u("uAmbient"),   0.20)
        GL.glUniform1f(_u("uSpecPower"), 10.0)
        for pos, orn, hx, hy, hz in self._ph.platform_data:
            m = _build_model_mat(pos, orn, hx * 2, hy * 2, hz * 2)
            GL.glUniformMatrix4fv(_u("uModel"), 1, GL.GL_TRUE, m)
            GL.glUniform4f(_u("uBaseColor"), 0.20, 0.22, 0.30, 1.0)
            GL.glDrawElements(GL.GL_TRIANGLES, self._box_idx_count,
                              GL.GL_UNSIGNED_INT, None)

        # ── Domino bodies + ivory face stripes ────────────────────────────────
        # FACE GEOMETRY CONCEPT:
        # The ivory face stripe is a separate thin box placed on the +X face
        # of each domino.  Instead of baking it into the mesh, we compute
        # a second model matrix by offsetting the domino's centre along its
        # animated local X axis:
        #   local_x_unit = model_col0 / DOM_D   (rotation-only X direction)
        #   face_centre  = body_pos + local_x_unit × (DOM_D/2 + FACE_D/2)
        # This works correctly for any orientation, including falling dominoes.
        FACE_D = DOM_D * 0.07   # 2.1 mm — slightly raised from the face
        GL.glUniform1f(_u("uAmbient"),   0.14)

        for i, (pos, orn) in enumerate(self._ph.body_states):
            r, g, b = self._ph.body_colors[i]
            m = _build_model_mat(pos, orn, DOM_D, DOM_W, DOM_H)

            # Body
            GL.glUniformMatrix4fv(_u("uModel"), 1, GL.GL_TRUE, m)
            GL.glUniform4f(_u("uBaseColor"), r, g, b, 1.0)
            GL.glUniform1f(_u("uSpecPower"), 32.0)
            GL.glDrawElements(GL.GL_TRIANGLES, self._box_idx_count,
                              GL.GL_UNSIGNED_INT, None)

            # Ivory face stripe — offset along domino's local X axis
            # m[:3,0] = R @ [DOM_D, 0, 0] → divide by DOM_D to get unit vector
            local_x  = m[:3, 0] / DOM_D
            face_pos = np.array(pos, dtype=np.float32) + local_x * (DOM_D * 0.5 + FACE_D * 0.5)
            mf = _build_model_mat(face_pos.tolist(), orn,
                                   FACE_D, DOM_W * 0.84, DOM_H * 0.88)
            GL.glUniformMatrix4fv(_u("uModel"), 1, GL.GL_TRUE, mf)
            GL.glUniform4f(_u("uBaseColor"), 0.96, 0.94, 0.86, 1.0)
            GL.glUniform1f(_u("uSpecPower"), 110.0)   # shinier ivory surface
            GL.glDrawElements(GL.GL_TRIANGLES, self._box_idx_count,
                              GL.GL_UNSIGNED_INT, None)

        GL.glBindVertexArray(0)
        self._prog.release()

    # ── Mesh generation ───────────────────────────────────────────────────────

    def _make_box_vao(self) -> None:
        """
        Upload a unit cube (−0.5 to +0.5 in each axis) with flat per-face normals.

        Each face gets its own 4 vertices (cannot share between faces because
        normals differ).  Six faces × 4 vertices × 6 floats = 144 floats.
        Six faces × 2 triangles × 3 indices = 36 uint32 indices.

        Interleaved layout (stride = 24 bytes):
            bytes 0–11  : position  (3 × float32)
            bytes 12–23 : normal    (3 × float32)
        """
        # (normal_xyz, [[v0], [v1], [v2], [v3]]) for each face, CCW winding
        FACES = [
            (( 1, 0, 0), [[ .5,-.5,-.5],[ .5, .5,-.5],[ .5, .5, .5],[ .5,-.5, .5]]),
            ((-1, 0, 0), [[-.5, .5,-.5],[-.5,-.5,-.5],[-.5,-.5, .5],[-.5, .5, .5]]),
            (( 0, 1, 0), [[ .5, .5,-.5],[-.5, .5,-.5],[-.5, .5, .5],[ .5, .5, .5]]),
            (( 0,-1, 0), [[-.5,-.5,-.5],[ .5,-.5,-.5],[ .5,-.5, .5],[-.5,-.5, .5]]),
            (( 0, 0, 1), [[-.5,-.5, .5],[ .5,-.5, .5],[ .5, .5, .5],[-.5, .5, .5]]),
            (( 0, 0,-1), [[ .5,-.5,-.5],[-.5,-.5,-.5],[-.5, .5,-.5],[ .5, .5,-.5]]),
        ]
        verts: list[float] = []
        idx:   list[int]   = []
        base = 0
        for (nx, ny, nz), corners in FACES:
            for cx, cy, cz in corners:
                verts.extend([cx, cy, cz, nx, ny, nz])
            idx.extend([base, base+1, base+2, base, base+2, base+3])
            base += 4

        v = np.array(verts, dtype=np.float32)
        i = np.array(idx,   dtype=np.uint32)

        self._box_vao = int(GL.glGenVertexArrays(1))
        self._box_vbo = int(GL.glGenBuffers(1))
        self._box_ebo = int(GL.glGenBuffers(1))

        GL.glBindVertexArray(self._box_vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._box_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, v.nbytes, v, GL.GL_STATIC_DRAW)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self._box_ebo)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, i.nbytes, i, GL.GL_STATIC_DRAW)
        stride = 6 * 4
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE,
                                  stride, GL.ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE,
                                  stride, GL.ctypes.c_void_p(12))
        GL.glBindVertexArray(0)
        self._box_idx_count = len(i)

    def _make_floor_vao(self) -> None:
        """Large flat quad at z = 0 for the floor plane."""
        S = 20.0   # half-extent (40 × 40 m floor)
        v = np.array([
            -S, -S, 0.0,  0, 0, 1,
             S, -S, 0.0,  0, 0, 1,
             S,  S, 0.0,  0, 0, 1,
            -S,  S, 0.0,  0, 0, 1,
        ], dtype=np.float32)
        i = np.array([0, 1, 2,  0, 2, 3], dtype=np.uint32)

        self._floor_vao = int(GL.glGenVertexArrays(1))
        self._floor_vbo = int(GL.glGenBuffers(1))
        self._floor_ebo = int(GL.glGenBuffers(1))

        GL.glBindVertexArray(self._floor_vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._floor_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, v.nbytes, v, GL.GL_STATIC_DRAW)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self._floor_ebo)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, i.nbytes, i, GL.GL_STATIC_DRAW)
        stride = 6 * 4
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE,
                                  stride, GL.ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE,
                                  stride, GL.ctypes.c_void_p(12))
        GL.glBindVertexArray(0)
        self._floor_idx_count = len(i)

    # ── Mouse event handlers ──────────────────────────────────────────────────

    def mouseDoubleClickEvent(self, event) -> None:
        self.cam.reframe()
        self.update()
        event.accept()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._last_mouse = event.position()
        elif event.button() == Qt.MouseButton.MiddleButton:
            self._pan_last = event.position()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._last_mouse = None
        elif event.button() == Qt.MouseButton.MiddleButton:
            self._pan_last = None

    def mouseMoveEvent(self, event) -> None:
        pos = event.position()
        if self._last_mouse is not None:
            dx = pos.x() - self._last_mouse.x()
            dy = pos.y() - self._last_mouse.y()
            self.cam.orbit(dx, dy)
            self._last_mouse = pos
            self.update()
        if self._pan_last is not None:
            dx = pos.x() - self._pan_last.x()
            dy = pos.y() - self._pan_last.y()
            self.cam.pan(dx, dy, self.height())
            self._pan_last = pos
            self.update()

    def wheelEvent(self, event) -> None:
        self.cam.zoom(event.angleDelta().y())
        self.update()


# =============================================================================
# ── Left Info Panel ───────────────────────────────────────────────────────────
# =============================================================================

class InfoPanel(QWidget):
    """
    Left sidebar — identical structure to demo 04's scene tree panel.
    Sections: layout info · live stats · camera controls · physics parameters.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(230)
        self.setMaximumWidth(300)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 12)
        layout.setSpacing(0)

        MONO = ("font-family: 'IBM Plex Mono', 'Courier New', monospace;"
                " font-size: 12px;")

        def _hdr(text: str) -> QLabel:
            lbl = QLabel(text)
            lbl.setObjectName("section_header")
            return lbl

        def _div() -> QFrame:
            f = QFrame(); f.setObjectName("divider")
            f.setFrameShape(QFrame.Shape.HLine)
            return f

        def _stat_card(value_lbl: QLabel, caption: str) -> QWidget:
            """Mini stat card: large value + small caption below."""
            card = QWidget(); card.setObjectName("stat_card")
            cv = QVBoxLayout(card)
            cv.setContentsMargins(10, 8, 10, 8); cv.setSpacing(1)
            value_lbl.setStyleSheet(
                f"color:{PALETTE['text']}; font-size:20px; font-weight:700;")
            value_lbl.setAlignment(Qt.AlignmentFlag.AlignLeft)
            cap = QLabel(caption)
            cap.setStyleSheet(
                f"color:{PALETTE['muted']}; font-size:10px; font-weight:600;"
                "letter-spacing:0.08em;")
            cv.addWidget(value_lbl)
            cv.addWidget(cap)
            return card

        # ── Layout info ───────────────────────────────────────────────────────
        layout.addWidget(_hdr("LAYOUT"))
        self._layout_info = QLabel()
        self._layout_info.setWordWrap(True)
        self._layout_info.setStyleSheet(f"color:{PALETTE['text']}; font-size:12px;")
        layout.addWidget(self._layout_info)
        layout.addSpacing(8)
        layout.addWidget(_div())

        # ── Live stats — two stat cards side-by-side ──────────────────────────
        layout.addWidget(_hdr("LIVE STATS"))
        layout.addSpacing(4)

        self._fallen_val = QLabel("0")
        self._simtime_val = QLabel("0.00 s")
        cards_row = QHBoxLayout()
        cards_row.setSpacing(6)
        cards_row.addWidget(_stat_card(self._fallen_val, "FALLEN"))
        cards_row.addWidget(_stat_card(self._simtime_val, "SIM TIME"))
        layout.addLayout(cards_row)
        layout.addSpacing(8)

        self._wave_lbl = QLabel("Wave front: waiting…")
        self._wave_lbl.setStyleSheet(
            f"color:{PALETTE['muted']}; font-size:11px; {MONO}")
        layout.addWidget(self._wave_lbl)

        # Progress bar
        layout.addSpacing(8)
        self._prog_outer = QWidget()
        self._prog_outer.setFixedHeight(4)
        self._prog_outer.setStyleSheet(
            f"background:{PALETTE['border']}; border-radius:2px;")
        self._prog_inner = QWidget(self._prog_outer)
        self._prog_inner.setFixedHeight(4)
        self._prog_inner.setStyleSheet(
            f"background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            f" stop:0 {PALETTE['accent']}, stop:1 {PALETTE['green']});"
            "border-radius:2px;")
        self._prog_inner.setFixedWidth(0)
        layout.addWidget(self._prog_outer)
        layout.addSpacing(8)
        layout.addWidget(_div())

        # ── Camera controls ───────────────────────────────────────────────────
        layout.addWidget(_hdr("CAMERA"))
        self._auto_cb = QCheckBox("Auto-track wave front")
        layout.addWidget(self._auto_cb)
        layout.addSpacing(6)
        controls = [
            ("Left-drag",  "Orbit"),
            ("Mid-drag",   "Pan"),
            ("Scroll",     "Zoom"),
            ("Dbl-click",  "Reframe"),
        ]
        for key, action in controls:
            row = QHBoxLayout()
            row.setSpacing(0)
            k = QLabel(key)
            k.setStyleSheet(
                f"color:{PALETTE['muted']}; font-size:11px; {MONO}")
            k.setFixedWidth(72)
            a = QLabel(action)
            a.setStyleSheet(f"color:{PALETTE['text']}; font-size:11px;")
            row.addWidget(k); row.addWidget(a); row.addStretch()
            layout.addLayout(row)

        layout.addSpacing(8)
        layout.addWidget(_div())

        # ── Physics parameters ────────────────────────────────────────────────
        layout.addWidget(_hdr("PHYSICS PARAMETERS"))
        phys_rows = [
            ("Mass",        f"{DOM_MASS*1000:.0f} g"),
            ("Restitution", f"{DOM_RESTITUTION:.2f}"),
            ("Friction",    f"{DOM_FRICTION:.2f}"),
            ("Step height", f"{STEP_HEIGHT*100:.0f} mm"),
            ("Timestep",    f"1/{int(1/PHYSICS_DT)} s"),
            ("1× speed",    f"{REAL_TIME_SUBSTEPS} substeps/frame"),
        ]
        for label, value in phys_rows:
            row = QHBoxLayout(); row.setSpacing(0)
            lk = QLabel(label)
            lk.setStyleSheet(
                f"color:{PALETTE['muted']}; font-size:11px; {MONO}")
            lk.setFixedWidth(88)
            lv = QLabel(value)
            lv.setStyleSheet(f"color:{PALETTE['text']}; font-size:11px; {MONO}")
            row.addWidget(lk); row.addWidget(lv); row.addStretch()
            layout.addLayout(row)

        layout.addStretch()

    def set_layout_info(self, total: int) -> None:
        self._layout_info.setText(
            f"<b>Staircase Canyon</b> (C)<br>"
            f"+ <b>Branching Tree</b> (D)<br>"
            f"<br>"
            f"<span style='color:{PALETTE['muted']}'>"
            f"{total} dominoes · 4 final arms</span>"
        )

    def update_stats(
        self,
        n_fallen: int,
        total: int,
        sim_time: float,
        wave: tuple,
        running: bool,
    ) -> None:
        self._fallen_val.setText(f"{n_fallen} / {total}")
        self._simtime_val.setText(f"{sim_time:.2f} s")
        if running and n_fallen > 0:
            self._wave_lbl.setText(
                f"Wave front: ({wave[0]:+.1f}, {wave[1]:+.1f})")
        else:
            self._wave_lbl.setText("Wave front: waiting…")

        # Update progress bar width
        if total > 0:
            frac = n_fallen / total
            w = int(self._prog_outer.width() * frac)
            self._prog_inner.setFixedWidth(max(0, w))


# =============================================================================
# ── Main Window ───────────────────────────────────────────────────────────────
# =============================================================================

class MainWindow(QMainWindow):
    """
    Application main window — mirrors demo 04's structure exactly:
      · Logo + author header widget (not a QToolBar)
      · Horizontal splitter: InfoPanel (left) + DominoViewport (centre)
      · Bottom control bar: START · RESET · speed slider · HUD readout
      · Status bar

    Signal/slot wiring:
      QTimer.timeout  → _on_tick()
      start_btn       → _on_start()
      reset_btn       → _on_reset()
      speed_slider    → _on_speed()
      auto_cb         → camera.auto_track
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.setMinimumSize(1200, 760)

        self._physics  = DominoPhysics()
        self._physics.connect()
        self._substeps = SPEED_TABLE[SPEED_DEFAULT_IDX][1]
        self._speed_idx = SPEED_DEFAULT_IDX
        self._fps      = 60.0
        self._last_t   = time.perf_counter()

        self._build_ui()
        self._start_loop()

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # ── Header (matches demo 04 exactly) ─────────────────────────────────
        header = QWidget()
        header.setStyleSheet(
            f"background-color:{PALETTE['surface']};"
            f"border-bottom: 1px solid {PALETTE['border']};"
        )
        hrow = QHBoxLayout(header)
        hrow.setContentsMargins(14, 8, 14, 8); hrow.setSpacing(14)

        logo = QLabel()
        logo.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        if LOGO_PATH.exists():
            px = QPixmap(str(LOGO_PATH)).scaledToHeight(
                36, Qt.TransformationMode.SmoothTransformation)
            logo.setPixmap(px)
        else:
            logo.setText("AMS · TU Ilmenau")
            logo.setStyleSheet(
                f"color:{PALETTE['accent']}; font-size:11px; font-weight:600;")

        title = QLabel(
            "<b>Demo 05 — Domino Cascade</b>"
            f"&nbsp;&nbsp;<span style='color:{PALETTE['muted']}; font-weight:400;'>"
            "PyBullet Physics · OpenGL 3.3 · Staircase Canyon + Branching Tree"
            "</span>"
        )
        title.setStyleSheet("font-size:14px;")

        author = QLabel("Renato Fera P.")
        author.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight)
        author.setStyleSheet(
            f"color:{PALETTE['muted']}; font-size:10px; letter-spacing:0.5px;")

        hrow.addWidget(logo)
        hrow.addWidget(title, stretch=1)
        hrow.addWidget(author)

        # ── Viewport + info panel ─────────────────────────────────────────────
        self._info = InfoPanel()
        self._info.set_layout_info(self._physics.total)

        self._vp = DominoViewport(self._physics)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._info)
        splitter.addWidget(self._vp)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([250, 1200])

        # Wire auto-track toggle
        self._info._auto_cb.stateChanged.connect(
            lambda s: setattr(self._vp.cam, "auto_track", bool(s))
        )

        # ── Control bar ───────────────────────────────────────────────────────
        ctrl = QWidget()
        ctrl.setStyleSheet(
            f"background-color:{PALETTE['surface']};"
            f"border-top: 1px solid {PALETTE['border']};"
        )
        ctrl.setFixedHeight(68)
        crow = QHBoxLayout(ctrl)
        crow.setContentsMargins(20, 12, 20, 12); crow.setSpacing(14)

        # ── START button — green pill, high-contrast, works under Fusion ────
        self._start_btn = QPushButton("▶  Start Cascade")
        self._start_btn.setObjectName("start_btn")
        self._start_btn.setFixedHeight(42)
        self._start_btn.setMinimumWidth(180)
        self._start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._start_btn.setStyleSheet("""
            QPushButton {
                background-color: #1db96a;
                color: #ffffff;
                border: none;
                border-radius: 21px;
                padding: 0px 28px;
                font-size: 14px;
                font-weight: 700;
                letter-spacing: 0.5px;
            }
            QPushButton:hover {
                background-color: #25e07f;
                color: #ffffff;
            }
            QPushButton:pressed {
                background-color: #12944f;
                color: #ffffff;
            }
            QPushButton:disabled {
                background-color: #1a2630;
                color: #3a5040;
                border: 1px solid #253530;
            }
        """)

        # ── RESET button — slate-blue pill, clearly readable ──────────────
        self._reset_btn = QPushButton("↺  Reset")
        self._reset_btn.setObjectName("reset_btn")
        self._reset_btn.setFixedHeight(42)
        self._reset_btn.setMinimumWidth(120)
        self._reset_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #2d3a52;
                color: #e0e6f8;
                border: 1px solid #4a5878;
                border-radius: 21px;
                padding: 0px 24px;
                font-size: 14px;
                font-weight: 600;
                letter-spacing: 0.3px;
            }
            QPushButton:hover {
                background-color: #3d4f70;
                color: #ffffff;
                border-color: #6a7ea8;
            }
            QPushButton:pressed {
                background-color: #1e2a3e;
                color: #ffffff;
            }
        """)

        # ── Speed chips ───────────────────────────────────────────────────────
        # 1× = real-time (REAL_TIME_SUBSTEPS substeps/frame at 60 Hz)
        spd_cap = QLabel("SPEED")
        spd_cap.setStyleSheet(
            f"color:{PALETTE['muted']}; font-size:10px; font-weight:700;"
            "letter-spacing:0.08em;")
        spd_cap.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        self._speed_chips: list[QPushButton] = []
        chips_w = QWidget()
        chips_row = QHBoxLayout(chips_w)
        chips_row.setContentsMargins(0, 0, 0, 0)
        chips_row.setSpacing(4)
        for i, (label, _substeps) in enumerate(SPEED_TABLE):
            btn = QPushButton(label)
            btn.setFixedHeight(30)
            btn.setStyleSheet(_CHIP_INACTIVE)
            btn.clicked.connect(lambda _checked, idx=i: self._on_speed(idx))
            chips_row.addWidget(btn)
            self._speed_chips.append(btn)
        self._update_speed_chips()

        spd_w = QWidget(); sv = QHBoxLayout(spd_w)
        sv.setContentsMargins(0, 0, 0, 0); sv.setSpacing(8)
        sv.addWidget(spd_cap)
        sv.addWidget(chips_w)

        # ── HUD readout ───────────────────────────────────────────────────────
        self._hud = QLabel()
        self._hud.setAlignment(
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight)
        self._hud.setStyleSheet(
            f"color:{PALETTE['muted']}; font-size:11px;"
            "font-family:'IBM Plex Mono','Courier New',monospace;")

        crow.addWidget(self._start_btn)
        crow.addWidget(self._reset_btn)
        crow.addSpacing(8)
        crow.addWidget(spd_w)
        crow.addStretch()
        crow.addWidget(self._hud)

        # ── Assemble central widget ───────────────────────────────────────────
        c = QWidget(); self.setCentralWidget(c)
        cv = QVBoxLayout(c); cv.setContentsMargins(0, 0, 0, 0); cv.setSpacing(0)
        cv.addWidget(header)
        cv.addWidget(splitter, stretch=1)
        cv.addWidget(ctrl)

        # ── Status bar ────────────────────────────────────────────────────────
        sb = QStatusBar(); self.setStatusBar(sb)
        sb.showMessage(
            f"Ready · {self._physics.total} dominoes · "
            "Camera: left-drag orbit · middle-drag pan · scroll zoom · "
            "double-click reframe · press START to release the cascade"
        )

        # ── Signals & slots ───────────────────────────────────────────────────
        self._start_btn.clicked.connect(self._on_start)
        self._reset_btn.clicked.connect(self._on_reset)

        self._refresh_hud()

    def _start_loop(self) -> None:
        """QTimer at 16 ms ≈ 60 Hz — the main animation loop."""
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._on_tick)
        self._timer.start()

    # ── Main loop ─────────────────────────────────────────────────────────────

    def _on_tick(self) -> None:
        """Physics step → schedule repaint → update HUD stats.  Called ~60 ×/s."""
        now = time.perf_counter()
        dt  = now - self._last_t
        self._last_t = now
        self._fps = 0.1 * (1.0 / max(dt, 1e-6)) + 0.9 * self._fps

        self._physics.step(self._substeps)
        self._vp.update()

        ph = self._physics
        self._info.update_stats(
            ph.n_fallen, ph.total, ph.sim_time, ph.wave, ph.is_running)
        self._refresh_hud()

    def _refresh_hud(self) -> None:
        ph = self._physics
        spd_label = SPEED_TABLE[self._speed_idx][0]
        self._hud.setText(
            f"Speed {spd_label}"
            f"   Fallen {ph.n_fallen:3d}/{ph.total}"
            f"   Sim {ph.sim_time:6.2f} s"
            f"   {self._fps:.0f} fps"
        )

    # ── Slots ──────────────────────────────────────────────────────────────────

    def _on_start(self) -> None:
        self._physics.start()
        self._start_btn.setText("▶  Running…")
        self._start_btn.setEnabled(False)
        self.statusBar().showMessage(
            "Cascade released — left-drag to orbit · scroll to zoom · "
            "enable Auto-track in the left panel to follow the wave front"
        )

    def _on_reset(self) -> None:
        self._physics.reset()
        self._start_btn.setText("▶  Start Cascade")
        self._start_btn.setEnabled(True)
        self._info.set_layout_info(self._physics.total)
        self.statusBar().showMessage(
            f"Reset · {self._physics.total} dominoes · press START"
        )
        self._refresh_hud()

    def _update_speed_chips(self) -> None:
        """Highlight the active speed chip and un-highlight all others."""
        for i, btn in enumerate(self._speed_chips):
            btn.setStyleSheet(_CHIP_ACTIVE if i == self._speed_idx else _CHIP_INACTIVE)

    def _on_speed(self, idx: int) -> None:
        self._speed_idx = idx
        label, substeps = SPEED_TABLE[idx]
        self._substeps = substeps
        # Also scale the timer interval so ½× truly halves the update rate
        # and higher speeds don't just pile substeps into the same 16 ms slot.
        # Base interval = 16 ms (≈60 Hz) at 1× real-time (4 substeps).
        # ½× → 32 ms (≈30 Hz); 2× → 8 ms; 4× → 4 ms; 8× → 2 ms.
        base_interval = 16  # ms at 1× (REAL_TIME_SUBSTEPS substeps)
        ratio = substeps / REAL_TIME_SUBSTEPS   # 0.5 / 1 / 2 / 4 / 8
        new_interval = max(2, round(base_interval / ratio))
        self._timer.setInterval(new_interval)
        self._update_speed_chips()

    def closeEvent(self, event) -> None:   # type: ignore[override]
        self._timer.stop()
        self._physics.disconnect()
        event.accept()


# =============================================================================
# ── Entry point ───────────────────────────────────────────────────────────────
# =============================================================================

def main() -> None:
    """
    Set up OpenGL 3.3 Core Profile surface format before QApplication.

    The surface format MUST be configured before QApplication is constructed
    on all platforms.  Setting it afterwards is silently ignored on most drivers.

    OpenGL 3.3 Core Profile:
      · Removes all deprecated fixed-function features (glBegin/End etc.)
      · Requires explicit VAOs, VBOs, and shader programs for every draw call
      · Minimum version guaranteed on macOS (Apple dropped older versions)
      · 24-bit depth buffer prevents Z-fighting at all zoom levels
      · 4× MSAA smooths domino edges without manual supersampling in the shader
    """
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setDepthBufferSize(24)
    fmt.setSamples(4)   # 4× MSAA
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(APP_STYLESHEET)
    app.setApplicationName("Domino Cascade")
    app.setApplicationVersion("2.0.0")
    app.setOrganizationName("TU Ilmenau — AMS Group")

    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

# =============================================================================
# End of 05_ca_pybullet_domino.py
# Author   : Renato Fera P.
# LinkedIn : https://www.linkedin.com/in/renato-profeta/
# Course   : Computeranimation — Chapter 05: Physics-based Animation
# Institute: Institut für Medientechnik, Technische Universität Ilmenau
# =============================================================================
