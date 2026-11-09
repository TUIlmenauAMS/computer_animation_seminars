# =============================================================================
# 05_ca_pybullet_domino_simple.py — Domino Cascade
# =============================================================================
# Author   : Renato Fera P.
# LinkedIn : https://www.linkedin.com/in/renato-profeta/
# Course   : Computeranimation — Chapter 05: Physics-based Animation
# Institute: Institut für Medientechnik, Technische Universität Ilmenau
# -----------------------------------------------------------------------------
"""Domino cascade simulation using PyBullet for physics and OpenGL for rendering.

This demo illustrates how rigid-body physics can drive a visual animation.
The physics engine (PyBullet) runs in headless mode and owns all collision
detection and dynamics.  The renderer (OpenGL 3.3) reads the resulting body
poses each frame and draws the scene independently.  Neither side knows about
the internals of the other — this separation makes it straightforward to swap
either subsystem without touching the other.

Layout:
    A single cascade wave travels through five acts in sequence.  Domino 0 is
    the only tile that ever receives an external force; every subsequent fall
    is caused by physical contact between neighbours.

    Entry platform → stair ramp down → flat ground run →
    stair ramp up → elevated plateau → fork into two arms (±35°).

Physics concepts demonstrated:
    1. Rigid body dynamics    — mass, inertia tensor, restitution, friction.
    2. Collision detection    — AABB broadphase followed by GJK/EPA narrowphase.
    3. Impulse propagation    — angular momentum transferred on contact.
    4. Gravitational PE ↔ KE  — the ramp down converts height into speed;
                                 the ramp up converts that speed back into
                                 height (half-pipe principle).
    5. Branching              — one falling domino can reach two neighbours
                                 placed at diverging angles, splitting the wave.
    6. Fixed-timestep loop    — stepSimulation advances at 1/240 s regardless
                                 of the display frame rate.
    7. Physics/render split   — PyBullet computes motion; OpenGL draws it.
    8. Per-body model matrix  — each domino's physics pose becomes a 4×4
                                 transform uploaded to the GPU each frame.

How to run:
    uv sync --extra arm64
    uv run python 05_ca_pybullet_domino_simple.py

Ubuntu dependencies:
    sudo apt install libgl1 libegl1 libxcb-cursor0 libxcb-xinerama0
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
    "Domino Cascade (Simple) — Computeranimation Chapter 05  ·  PyBullet + OpenGL 3.3"
)
LOGO_PATH: Path = Path(__file__).resolve().parent.parent / "ams_group_tui.png"

# ── Physics timestep ──────────────────────────────────────────────────────────
# 240 Hz gives enough temporal resolution for thin, fast-tipping bodies.
# A lower rate causes dominoes to interpenetrate before the solver can
# separate them, which produces visible jitter or missed contacts.
PHYSICS_DT: float      = 1.0 / 240.0

# ── Domino geometry (metres) ──────────────────────────────────────────────────
# Standard domino proportions: height : width : depth ≈ 2 : 1 : 0.25.
DOM_H: float           = 0.30
DOM_W: float           = 0.14
DOM_D: float           = 0.030
DOM_MASS: float        = 0.090   # kg

# Restitution controls how much energy a collision returns to the bodies.
# Setting it to zero means a domino that hits the floor or another tile does
# not bounce; all kinetic energy on the collision normal is absorbed.
# This ensures only a domino that is physically struck by its neighbour falls —
# a distant vibration or floor bounce cannot trigger the cascade on its own.
DOM_RESTITUTION: float = 0.0

# Friction determines how well dominoes grip the platform surface.
# A value of 0.80 is high enough to prevent sliding while still allowing
# the tile to pivot cleanly about its bottom edge when it tips.
DOM_FRICTION: float    = 0.80

# ── Layout geometry ───────────────────────────────────────────────────────────
# Spacing on flat ground: centre-to-centre distance between adjacent dominoes.
DOMINO_SPACING: float    = 0.155

# On a downhill ramp, gravity accelerates the falling tile forward, so it can
# reach a neighbour that is slightly farther away than on flat ground.
DESCEND_SPACING: float   = 0.165

# On an uphill ramp, the tile decelerates as it falls, so neighbours must be
# placed closer together to stay within reach before the tile loses momentum.
ASCEND_SPACING: float    = 0.130

# Vertical rise (or drop) per stair step on the descending ramp.
DESCEND_STEP_RISE: float = 0.050

# Vertical rise per stair step on the ascending ramp.
# Kept smaller than DESCEND_STEP_RISE so the kinetic energy accumulated on
# the way down is enough to carry the wave up the other side — the same
# energy trade that makes a skate half-pipe work.
ASCEND_STEP_RISE: float  = 0.032

# Thickness of each static platform slab placed under elevated dominoes.
PLATFORM_THICKNESS: float = 0.05

# ── Simulation speed ──────────────────────────────────────────────────────────
# The QTimer fires at ~60 Hz.  Each tick advances the simulation by
# (substeps × PHYSICS_DT) seconds.  At 4 substeps that equals 4 × 1/240 ≈ 16.7 ms,
# matching wall-clock time and giving 1× real-time playback.
# Higher substep counts compress time without changing physics accuracy.
REAL_TIME_SUBSTEPS: int = 4
SPEED_TABLE: list[tuple[str, int]] = [
    ("½×",  2),
    ("1×",  4),   # default — 1× real-time
    ("2×",  8),
    ("4×", 16),
    ("8×", 32),
]
SPEED_DEFAULT_IDX: int = 1   # index into SPEED_TABLE selecting "1×"

# ── Colour gradient for domino bodies (blue → cyan → green → gold → red) ──────
# Each domino is assigned a colour by interpolating this gradient according to
# its position in the sequence.  The colour change makes the cascade wave
# visually traceable from the first tile to the last.
BODY_GRADIENT: list[tuple[float, float, float]] = [
    (0.08, 0.20, 0.72),
    (0.04, 0.72, 0.82),
    (0.12, 0.76, 0.22),
    (0.88, 0.72, 0.04),
    (0.82, 0.12, 0.08),
]

# ── UI colour palette ─────────────────────────────────────────────────────────
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

# Speed chip button styles are set directly via setStyleSheet() on each widget
# rather than through the application stylesheet, because Qt property selectors
# (e.g. QPushButton#active) are not reliably applied by the Fusion style on all
# platforms.  Two string constants — one for the selected state, one for the
# unselected state — are toggled each time the user changes the speed.
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

# ---------------------------------------------------------------------------
# Vertex shader
# ---------------------------------------------------------------------------
# Transforms each vertex from object space to clip space via the standard
# Model-View-Projection (MVP) chain:
#
#   clip position  = Projection × View × Model × object position
#
# The world-space position (after Model but before View/Projection) is passed
# to the fragment shader so that lighting can be computed relative to actual
# 3-D positions rather than screen positions.
#
# Normals are transformed by the inverse-transpose of the model matrix.
# This is necessary because a plain matrix multiplication distorts normal
# directions when the model matrix contains non-uniform scaling — the
# inverse-transpose corrects for that distortion.
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
    // Transform the normal by the inverse-transpose of the model matrix so
    // that non-uniform scaling does not distort the normal direction.
    vNormal = normalize(mat3(transpose(inverse(uModel))) * aNormal);
}
"""

# ---------------------------------------------------------------------------
# Fragment shader
# ---------------------------------------------------------------------------
# Implements Blinn-Phong shading with two lights and gamma correction.
#
# Two-light setup:
#   - Key light: warm colour, high intensity, positioned upper-left.
#     This is the primary source of diffuse and specular highlights.
#   - Fill light: cool colour, 35% the intensity of the key light,
#     positioned opposite.  It softens the shadows on the unlit side.
#
# The uFloor flag activates a checkerboard pattern on the floor quad.
# The pattern is computed in the fragment shader from the world-space XY
# position rather than from a texture, so no image file is needed and the
# tile size stays constant regardless of the mesh's UV coordinates.
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

    // Checkerboard pattern for the floor quad.
    // Multiplying the world-space XY coordinate by 2 makes each tile 0.5 m wide.
    // mod(cx + cy, 2.0) alternates between 0 and 1 in a checkerboard arrangement.
    // The edge fade reduces the colour toward the perimeter so the floor blends
    // into the background rather than ending with a sharp square border.
    if (uFloor) {
        float cx = floor(vFragPos.x * 2.0);
        float cy = floor(vFragPos.y * 2.0);
        float c  = mod(cx + cy, 2.0);
        col.rgb  = mix(col.rgb * 0.48, col.rgb * 1.35, c);
        float dist = length(vFragPos.xy);
        col.rgb *= 1.0 - smoothstep(14.0, 20.0, dist) * 0.6;
    }

    vec3 N = normalize(vNormal);
    vec3 V = normalize(uCamPos - vFragPos);  // direction from fragment to camera

    // Key light contribution.
    // L1 is the direction from the fragment to the key light.
    // d1 is the Lambertian (diffuse) term: cos of the angle between N and L1.
    // H1 is the half-vector used by Blinn-Phong for the specular term.
    vec3  L1   = normalize(uLight1Pos - vFragPos);
    float d1   = max(dot(N, L1), 0.0);
    vec3  H1   = normalize(L1 + V);
    float s1   = pow(max(dot(N, H1), 0.0), uSpecPower);

    // Fill light contribution — same calculation at reduced (35%) intensity.
    // A lower specular exponent (uSpecPower * 0.4) gives broader, softer highlights.
    vec3  L2   = normalize(uLight2Pos - vFragPos);
    float d2   = max(dot(N, L2), 0.0) * 0.35;
    vec3  H2   = normalize(L2 + V);
    float s2   = pow(max(dot(N, H2), 0.0), uSpecPower * 0.4) * 0.12;

    // Combine ambient, diffuse, and specular contributions from both lights.
    vec3 ambient  = uAmbient * col.rgb;
    vec3 diffuse  = d1 * col.rgb * uLight1Color
                  + d2 * col.rgb * uLight2Color;
    vec3 specular = s1 * uLight1Color * 0.55
                  + s2 * uLight2Color;

    vec3 result = ambient + diffuse + specular;

    // Apply gamma correction (γ = 1/2.2) before writing to the sRGB framebuffer.
    // Without this step, colours computed in linear light space appear too dark
    // on a standard monitor because the display applies its own gamma on top.
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
    """Linearly interpolate between two RGB colour tuples.

    Args:
        a: Starting colour as (r, g, b) floats in [0, 1].
        b: Ending colour as (r, g, b) floats in [0, 1].
        t: Interpolation parameter in [0, 1].  t=0 returns a; t=1 returns b.

    Returns:
        Interpolated colour as (r, g, b).
    """
    return (a[0]+(b[0]-a[0])*t, a[1]+(b[1]-a[1])*t, a[2]+(b[2]-a[2])*t)


def body_color(index: int, total: int) -> tuple[float, float, float]:
    """Map a domino index to a colour sampled from the five-stop gradient.

    The gradient runs from the first domino (blue) to the last (red), passing
    through cyan, green, and gold.  This makes the cascade wave easy to follow
    visually: the colour changes continuously from start to finish.

    Args:
        index: Zero-based index of the domino in the placement sequence.
        total: Total number of dominoes in the scene.

    Returns:
        RGB colour tuple with components in [0, 1].
    """
    t  = index / max(total - 1, 1)        # normalise index to [0, 1]
    s  = t * (len(BODY_GRADIENT) - 1)     # scale to gradient segment space
    lo = int(s)                            # index of the lower colour stop
    hi = min(lo + 1, len(BODY_GRADIENT) - 1)
    return _lerp3(BODY_GRADIENT[lo], BODY_GRADIENT[hi], s - lo)


def _quat_to_mat4(quat: tuple | list) -> np.ndarray:
    """Convert a unit quaternion to a row-major 4×4 rotation matrix.

    A quaternion (x, y, z, w) encodes a rotation in 3-D space.  Expanding
    the sandwich product q * v * q⁻¹ gives the equivalent 3×3 rotation matrix
    whose entries are quadratic in the quaternion components.  The result is
    embedded in a 4×4 identity matrix so it can be multiplied with translation
    and scale matrices.

    PyBullet returns quaternions in (x, y, z, w) order, matching this function's
    input convention.

    Args:
        quat: Unit quaternion as (x, y, z, w).

    Returns:
        Row-major 4×4 rotation matrix as a float32 NumPy array.
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
    """Build a row-major 4×4 model matrix from a physics body pose and scale.

    The matrix is composed as T × R × S, meaning scale is applied first (in
    object space), then rotation, then translation.  This is the standard
    Scale-Rotate-Translate order used in 3-D graphics.

    Args:
        pos: World-space position as (x, y, z).
        orn: Orientation as a unit quaternion (x, y, z, w).
        sx:  Scale along the object's local X axis.
        sy:  Scale along the object's local Y axis.
        sz:  Scale along the object's local Z axis.

    Returns:
        Row-major 4×4 model matrix as a float32 NumPy array.
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
    """World-space placement of a single domino.

    Attributes:
        x:   World-space X coordinate of the domino's bottom-centre (metres).
        y:   World-space Y coordinate of the domino's bottom-centre (metres).
        z:   Elevation of the surface the domino stands on (metres above z=0).
        yaw: Heading angle in radians.  The domino faces in the direction
             (cos(yaw), sin(yaw), 0) and will fall toward the next tile.
    """
    x: float; y: float; z: float; yaw: float


def _run(ox: float, oy: float, oz: float, yaw: float, n: int,
         sp: float = DOMINO_SPACING) -> list[DP]:
    """Place n dominoes in a straight line at a constant elevation.

    Args:
        ox:  X coordinate of the first domino's bottom-centre.
        oy:  Y coordinate of the first domino's bottom-centre.
        oz:  Elevation (z coordinate) shared by all dominoes in this run.
        yaw: Heading direction in radians.
        n:   Number of dominoes to place.
        sp:  Centre-to-centre spacing between dominoes.

    Returns:
        List of DP placements ordered from first to last in the cascade.
    """
    dx, dy = math.cos(yaw) * sp, math.sin(yaw) * sp
    return [DP(ox + dx*i, oy + dy*i, oz, yaw) for i in range(n)]


def _arc(
    cx: float, cy: float, cz: float,
    radius: float, start_ang: float, end_ang: float,
    n: int,
) -> list[DP]:
    """Place n dominoes along a circular arc at a constant elevation.

    Each domino is oriented tangent to the arc so that it naturally faces the
    next tile in the chain.  The tangent direction is perpendicular to the
    radius at that point, with the sign chosen according to the sweep direction.

    Args:
        cx:        X coordinate of the arc centre.
        cy:        Y coordinate of the arc centre.
        cz:        Elevation shared by all dominoes on this arc.
        radius:    Distance from the centre to each domino's bottom-centre.
        start_ang: Starting angle of the arc in radians.
        end_ang:   Ending angle of the arc in radians.
        n:         Number of dominoes to place along the arc.

    Returns:
        List of DP placements ordered from start_ang to end_ang.
    """
    pl: list[DP] = []
    for i in range(n):
        t   = i / max(n - 1, 1)
        ang = start_ang + (end_ang - start_ang) * t
        x   = cx + radius * math.cos(ang)
        y   = cy + radius * math.sin(ang)
        # The tangent direction is perpendicular to the radius.
        # The sign selects the tangent that points in the sweep direction.
        yaw = ang + math.pi * 0.5 * (1 if end_ang > start_ang else -1)
        pl.append(DP(x, y, cz, yaw))
    return pl


def _platform_for(dp: DP, length: float) -> tuple:
    """Create a static platform box sized to support a single elevated domino.

    Used for dominoes placed on a curved elevated path where the rectangular
    slab produced by _stair does not apply.  The box top face aligns with
    dp.z so the domino stands flush on its surface.

    Args:
        dp:     Placement of the domino that needs support.
        length: Desired length of the platform slab along the domino's heading.

    Returns:
        Platform specification tuple (pos_xyz, yaw, half_x, half_y, half_z)
        ready to be passed to DominoPhysics as a static collision box.
    """
    hz = PLATFORM_THICKNESS * 0.5
    return (
        (dp.x, dp.y, dp.z - hz), dp.yaw,
        length * 0.5, (DOM_W + 0.10) * 0.5, hz,
    )


def _stair(
    ox: float, oy: float, oz: float, yaw: float, n: int,
    rise: float, run: float = DOMINO_SPACING,
) -> tuple[list[DP], list[tuple]]:
    """Place n dominoes in a straight run with a constant rise per step.

    When rise is negative, successive dominoes step downward (a descending
    ramp).  When rise is positive, they step upward (an ascending ramp).
    When rise is zero, all dominoes share the same elevation (a flat run).

    For every domino whose elevation is above z=0, this function also
    generates a static platform box whose top face aligns with that domino's
    footing elevation.  Dominoes at z=0 are supported by the global ground
    plane and do not need an additional box.

    The platform half-length along the heading direction is set slightly larger
    than half the step spacing so that consecutive slabs overlap by a small
    margin.  This prevents a narrow gap from forming between adjacent slabs
    at the step edge — a gap that a falling tile's corner could catch on.

    Args:
        ox:   X coordinate of the first domino's bottom-centre.
        oy:   Y coordinate of the first domino's bottom-centre.
        oz:   Starting elevation (z of the first domino's footing).
        yaw:  Heading direction in radians.
        n:    Number of dominoes to place.
        rise: Elevation change per step (metres).  Negative for descent.
        run:  Horizontal spacing between successive dominoes (metres).

    Returns:
        A tuple of (placements, platforms) where:
            placements: list of DP, one per domino, in cascade order.
            platforms:  list of (pos_xyz, yaw, hx, hy, hz) tuples for each
                        elevated step, ready for DominoPhysics.
    """
    dx, dy = math.cos(yaw), math.sin(yaw)
    placements: list[DP] = []
    platforms:  list[tuple] = []
    for i in range(n):
        z = oz + rise * i
        x = ox + dx * run * i
        y = oy + dy * run * i
        placements.append(DP(x, y, z, yaw))
        # Only create a platform box when the step is elevated above the ground
        # plane.  At z=0 the global plane.urdf already provides the surface.
        if abs(z) > 1e-6:
            hz = PLATFORM_THICKNESS * 0.5
            platforms.append((
                (x, y, z - hz), yaw,
                run * 0.55, (DOM_W + 0.10) * 0.5, hz,
            ))
    return placements, platforms


def build_layout() -> tuple[list[DP], list[int], list[tuple]]:
    """Construct the full domino placement for the single-wave cascade.

    The cascade is organised into five acts that follow one continuous
    path.  Domino 0 is the only tile that ever receives an external push;
    every subsequent fall is caused by a neighbouring domino striking it.

    Acts:
        1. Entry platform: a flat elevated starting section heading in the
           +X direction.  The starting height is chosen so that the descent
           in Act 2 lands exactly at z=0.
        2. Descent: a stair-stepped ramp down to z=0.  Each step drops the
           domino by STEP_RISE metres.  As the wave travels downward, the
           falling tiles accelerate because gravity adds kinetic energy to
           each step — potential energy is being converted to kinetic energy.
        3. Ground run: a short flat section at z=0 where the cascade settles
           onto the global ground plane before beginning the climb.
        4. Ascent: a stair-stepped ramp back up to PLATEAU_H.  The ascent
           uses shallower steps and tighter spacing than the descent.  The
           kinetic energy accumulated on the way down carries the wave up the
           ramp — the same energy exchange that occurs on a skate half-pipe.
        5. Plateau: a flat section at PLATEAU_H that ends at the fork domino.
        6. Fork: the last plateau domino falls toward two neighbours placed at
           ±FORK_DEG from the heading direction.  Each neighbour starts its
           own arm of the cascade.  The angle must be wide enough that the
           two first arm dominoes do not overlap when initially placed;
           a 2-D oriented-box check shows the minimum safe angle is about
           28° at this spacing, so 35° provides a clear margin.
        7. Arm A and Arm B: two straight flat runs on the plateau surface,
           one for each branch of the fork.

    Returns:
        A tuple of (placements, fork_idx, platforms) where:
            placements: Ordered list of DP, one per domino in cascade order.
            fork_idx:   List containing the index of the fork domino in
                        placements.  Can be used to highlight or annotate
                        the branching point.
            platforms:  List of (pos_xyz, yaw, hx, hy, hz) tuples describing
                        the static support boxes for all elevated tiles.
    """
    pl: list[DP] = []
    fork_idx: list[int] = []
    platforms: list[tuple] = []

    # Derive the starting height so that DESCEND_STEPS steps of STEP_RISE
    # each bring the ramp exactly to z=0 on the last step.
    DESCEND_STEPS = 10
    STEP_RISE     = DESCEND_STEP_RISE          # 0.050 m per step
    START_H       = STEP_RISE * DESCEND_STEPS  # 0.50 m total starting height

    # The ascending ramp uses a smaller rise per step than the descent so
    # that the kinetic energy from the downhill run can carry the wave up.
    ASCEND_STEPS  = 8
    A_RISE        = STEP_RISE * 0.6            # 0.030 m per step
    PLATEAU_H     = A_RISE * ASCEND_STEPS      # ~0.24 m plateau height

    # Fork angle: each arm deviates ±FORK_DEG from the plateau heading.
    FORK_DEG      = 35.0
    FORK_HALF     = math.radians(FORK_DEG)
    # First spacing in each arm is slightly larger than DOMINO_SPACING to
    # ensure the two first-arm dominoes do not touch each other at placement.
    FORK_SP       = 0.20

    # Act 1: flat entry section at START_H, heading in the +X direction (yaw=0).
    entry, plats = _stair(0.0, 0.0, START_H, 0.0, 8, rise=0.0)
    pl.extend(entry); platforms.extend(plats)

    # Act 2: descending ramp from START_H down to z=0.
    # The next domino is placed one DESCEND_SPACING ahead of the last entry tile.
    p = entry[-1]
    descend, plats = _stair(
        p.x + math.cos(p.yaw) * DESCEND_SPACING,
        p.y + math.sin(p.yaw) * DESCEND_SPACING,
        START_H, p.yaw,
        DESCEND_STEPS,
        rise=-STEP_RISE, run=DESCEND_SPACING,
    )
    pl.extend(descend); platforms.extend(plats)

    # Act 3: flat ground run at z=0.
    # Dominoes here rest on the global ground plane (no platform boxes needed).
    p = descend[-1]
    floor_run, plats = _stair(
        p.x + math.cos(p.yaw) * DOMINO_SPACING,
        p.y + math.sin(p.yaw) * DOMINO_SPACING,
        0.0, p.yaw, 6, rise=0.0,
    )
    pl.extend(floor_run); platforms.extend(plats)

    # Act 4: ascending ramp from z=0 up to PLATEAU_H.
    # Tighter spacing (ASCEND_SPACING) keeps each tile within reach as the
    # cascade decelerates while climbing.
    p = floor_run[-1]
    ascend, plats = _stair(
        p.x + math.cos(p.yaw) * ASCEND_SPACING,
        p.y + math.sin(p.yaw) * ASCEND_SPACING,
        0.0, p.yaw,
        ASCEND_STEPS,
        rise=A_RISE, run=ASCEND_SPACING,
    )
    pl.extend(ascend); platforms.extend(plats)

    # Act 5: flat plateau section at PLATEAU_H.
    p = ascend[-1]
    plateau, plats = _stair(
        p.x + math.cos(p.yaw) * DOMINO_SPACING,
        p.y + math.sin(p.yaw) * DOMINO_SPACING,
        PLATEAU_H, p.yaw, 5, rise=0.0,
    )
    pl.extend(plateau); platforms.extend(plats)

    # Act 6 and 7: fork and arms.
    # Record the index of the last plateau tile — this is the fork domino.
    # When it falls, it tips toward two neighbours simultaneously.
    fork_idx.append(len(pl) - 1)
    pf = plateau[-1]

    # Place one arm on each side of the heading direction.
    for sign in (+1.0, -1.0):
        arm_yaw = pf.yaw + sign * FORK_HALF
        ax = pf.x + math.cos(arm_yaw) * FORK_SP
        ay = pf.y + math.sin(arm_yaw) * FORK_SP
        arm, plats = _stair(ax, ay, PLATEAU_H, arm_yaw, 12, rise=0.0)
        pl.extend(arm); platforms.extend(plats)

    return pl, fork_idx, platforms


# =============================================================================
# ── Interactive Camera ────────────────────────────────────────────────────────
# =============================================================================

class InteractiveCamera:
    """Orbit/pan/zoom camera with optional wave-front auto-tracking.

    The camera is defined in spherical coordinates around a target point:
    yaw (horizontal angle), pitch (vertical angle), and distance.  The
    eye position is derived from these three values each frame.

    The coordinate system is Z-up, matching PyBullet's convention.

    Controls:
        Left-drag:    Orbit — change yaw and pitch around the target.
        Middle-drag:  Pan — translate the target in the camera's local XY plane.
        Scroll wheel: Zoom — scale the distance from eye to target.
        Double-click: Reframe — restore the saved default view.

    Auto-track mode:
        When enabled, the target smoothly follows the wave-front position
        reported by DominoPhysics.  The target is interpolated toward the
        wave position each frame using a factor of (dt × 2.0), which keeps
        the camera slightly behind the leading edge of the cascade.
    """

    def __init__(self) -> None:
        self._yaw   =  28.0   # horizontal orbit angle in degrees
        self._pitch =  24.0   # vertical orbit angle in degrees
        self._dist  =  10.0   # distance from eye to target in metres
        self._tgt   = np.array([0.0, 0.0, 0.5], dtype=np.float32)
        self.auto_track: bool = False

        # Store default values so the camera can be restored on double-click.
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
        """Compute the view matrix from the current eye position and target.

        Constructs an orthonormal camera frame from three vectors:
            forward (f): direction from eye to target.
            right (r):   perpendicular to forward and world up.
            camera-up (u): perpendicular to both forward and right.

        The resulting 4×4 matrix transforms world-space coordinates into
        camera space, where the camera looks along its negative Z axis.

        Returns:
            Row-major 4×4 view matrix as a float32 NumPy array.
        """
        e  = self.eye
        t  = self._tgt
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # world Z-up
        f  = t - e;  f /= np.linalg.norm(f) + 1e-9        # forward vector
        r  = np.cross(f, up); r /= np.linalg.norm(r) + 1e-9  # right vector
        u  = np.cross(r, f)                                    # camera-up vector
        V  = np.eye(4, dtype=np.float32)
        V[0, :3] = r;  V[0, 3] = -np.dot(r, e)
        V[1, :3] = u;  V[1, 3] = -np.dot(u, e)
        V[2, :3] = -f; V[2, 3] =  np.dot(f, e)
        return V

    def proj_matrix(self, aspect: float) -> np.ndarray:
        """Compute a perspective projection matrix for the current zoom level.

        Near and far clip planes are derived from the orbit distance so that
        the depth range stays well-conditioned at any zoom level: zooming in
        (small distance) tightens the clip planes, reducing depth-buffer waste.

        The entry P[3,2] = -1 performs the perspective divide, mapping the
        z component into the W component that OpenGL uses to divide clip-space
        coordinates into normalised device coordinates.

        Args:
            aspect: Viewport width divided by viewport height.

        Returns:
            Row-major 4×4 projection matrix as a float32 NumPy array.
        """
        near = max(0.01, self._dist * 0.004)
        far  = max(100.0, self._dist * 160.0)
        fov  = math.radians(50.0)
        f    = 1.0 / math.tan(fov * 0.5)   # cotangent of the half field-of-view
        P    = np.zeros((4, 4), dtype=np.float32)
        P[0, 0] = f / aspect
        P[1, 1] = f
        P[2, 2] = (far + near) / (near - far)
        P[2, 3] = 2 * far * near / (near - far)
        P[3, 2] = -1.0  # perspective divide
        return P

    def orbit(self, dx: float, dy: float) -> None:
        """Adjust the orbit angles by mouse delta values.

        Args:
            dx: Horizontal mouse movement in pixels (positive = right).
            dy: Vertical mouse movement in pixels (positive = down).
        """
        self._yaw  += dx * 0.4
        self._pitch = max(-89.0, min(89.0, self._pitch + dy * 0.4))

    def pan(self, dx_px: float, dy_px: float, vp_h: int) -> None:
        """Translate the orbit target in the camera's local XY plane.

        The translation is scaled by (orbit distance / viewport height) so
        that the same pixel movement corresponds to the same scene movement
        regardless of the current zoom level.

        Args:
            dx_px: Horizontal mouse movement in pixels.
            dy_px: Vertical mouse movement in pixels.
            vp_h:  Current viewport height in pixels.
        """
        yr = math.radians(self._yaw); pr = math.radians(self._pitch)
        # Reconstruct the forward direction from spherical coordinates.
        f  = -np.array([
            math.cos(pr)*math.cos(yr),
            math.cos(pr)*math.sin(yr),
            math.sin(pr),
        ])
        up = np.array([0., 0., 1.])
        r  = np.cross(f, up); r /= np.linalg.norm(r) + 1e-9
        u  = np.cross(r, f)
        # Scale factor: pixels → metres, proportional to distance and inversely
        # proportional to viewport height.
        s  = self._dist / max(1, vp_h)
        self._tgt -= r.astype(np.float32) * (dx_px * s)
        self._tgt += u.astype(np.float32) * (dy_px * s)

    def zoom(self, delta: float) -> None:
        """Scale the orbit distance proportionally to the scroll delta.

        A multiplicative scale (0.999 per unit of delta) keeps the zoom speed
        proportional to the current distance — the same number of scroll
        notches halves the distance regardless of whether the camera is close
        or far from the target.

        Args:
            delta: Raw scroll delta from the wheel event (positive = zoom in).
        """
        self._dist = max(0.3, min(80.0, self._dist * (0.999 ** delta)))

    def reframe(self) -> None:
        """Restore the camera to its saved default position and orientation."""
        self._yaw = self._def_yaw; self._pitch = self._def_pitch
        self._dist = self._def_dist; self._tgt = self._def_tgt.copy()

    def update_wave(self, wx: float, wy: float, wz: float, dt: float) -> None:
        """Smoothly move the camera target toward the wave-front position.

        Called every frame when auto-track is enabled.  The target is
        interpolated toward the goal using a factor of (dt × 2.0), which
        introduces a small lag so the camera follows the cascade smoothly
        rather than snapping to it instantly.

        Args:
            wx: Wave-front X position in world space.
            wy: Wave-front Y position in world space.
            wz: Wave-front Z position in world space.
            dt: Elapsed time since the last frame in seconds.
        """
        if self.auto_track:
            goal       = np.array([wx, wy, wz + 0.3], dtype=np.float32)
            self._tgt += (goal - self._tgt) * min(1.0, dt * 2.0)


# =============================================================================
# ── Physics World (PyBullet DIRECT — headless, no rendering) ──────────────────
# =============================================================================

class DominoPhysics:
    """Headless rigid-body simulation of the domino cascade.

    This class owns the PyBullet physics world and is responsible for
    building the scene, advancing the simulation, and reporting body
    poses to the renderer.  It has no knowledge of OpenGL or Qt.

    PyBullet is used in DIRECT mode, which runs the physics engine without
    launching any graphical window.  Poses are queried after each step and
    stored in public attributes that DominoViewport reads every frame.

    Physics pipeline per simulation step:
        stepSimulation() integrates forces and resolves contact constraints.
        getBasePositionAndOrientation() reads the resulting pose of each body.

    Attributes:
        body_states:   List of (pos, orn) tuples, one per domino, updated
                       after every call to step().
        platform_data: List of (pos, orn, hx, hy, hz) tuples describing the
                       static support boxes.  Set once during connect().
        body_colors:   List of (r, g, b) tuples, one per domino, computed
                       once from the colour gradient during connect().
    """

    def __init__(self) -> None:
        self._client     = -1
        self._dom_ids:   list[int] = []
        self._placements: list[DP] = []
        self._n_fallen   = 0
        self._sim_time   = 0.0
        self._running    = False
        self._wave: tuple[float, float, float] = (0.0, 0.0, 0.5)

        # Public arrays read by DominoViewport each frame.
        self.body_states:   list[tuple] = []
        self.platform_data: list[tuple] = []
        self.body_colors:   list[tuple] = []

    def connect(self) -> None:
        """Initialise the PyBullet world and populate the scene.

        Connects to PyBullet in DIRECT (headless) mode, sets gravity and the
        fixed timestep, then calls the internal build methods to place the
        ground plane, support platforms, and domino rigid bodies.

        The solver iteration count is raised from the default of 10 to 20.
        More iterations mean the contact solver converges closer to the true
        solution each step, which reduces penetration artefacts in stacks of
        thin bodies like dominoes.
        """
        self._client = pybullet.connect(pybullet.DIRECT)
        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        pybullet.setGravity(0, 0, -9.81)
        pybullet.setTimeStep(PHYSICS_DT)

        # Increase solver iterations to improve contact resolution for the
        # thin, closely spaced domino bodies.
        pybullet.setPhysicsEngineParameter(
            numSolverIterations       = 20,
            numSubSteps               = 1,
            contactBreakingThreshold  = 0.001,  # contact detection threshold (m)
        )

        self._build_ground()
        self._build_scene()

    def _build_ground(self) -> None:
        """Load the infinite static ground plane and configure its surface.

        PyBullet's plane.urdf provides an infinite rigid slab at z=0 with
        mass=0, meaning it has infinite effective mass and cannot be moved
        by any force.  All dominoes and platforms at z=0 rest on this plane.

        Restitution is set to zero on the ground plane so that a domino
        landing on it does not bounce.  A bounce could transfer energy through
        the floor surface to a neighbouring domino that was never struck,
        which would propagate the cascade by a mechanism other than direct
        physical contact.

        contactStiffness and contactDamping control how PyBullet models
        resting contact.  High stiffness prevents the bodies from sinking
        into the floor; the damping suppresses the residual oscillation that
        would otherwise cause resting bodies to jitter.
        """
        gid = pybullet.loadURDF("plane.urdf", useFixedBase=True)
        pybullet.changeDynamics(
            gid, -1,
            lateralFriction   = DOM_FRICTION,
            restitution       = 0.0,
            contactStiffness  = 1e6,
            contactDamping    = 1e4,
        )

    def _build_scene(self) -> None:
        """Create the static support platforms and all domino rigid bodies.

        Static platforms (mass=0):
            Every domino that sits above z=0 needs a solid surface under it.
            build_layout() returns a platform specification for each such
            domino — a box whose top face aligns with the domino's footing
            elevation.  These boxes are created with mass=0 so they cannot
            be moved by collisions.

        Domino rigid bodies:
            Each domino is represented by a GEOM_BOX collision shape with
            half-extents (D/2, W/2, H/2).  The body's centre of mass is
            positioned at p.z + DOM_H/2 — half the domino's height above
            the footing surface it stands on.

            Linear and angular damping of 0.05 suppress the small numerical
            drift that accumulates over many time steps without meaningfully
            slowing the cascade.

        After all bodies are created, their poses and velocities are reset
        to the exact upright initial state so that any numerical noise
        introduced during body creation does not affect the simulation.
        """
        self._placements, _fork_idx, platform_specs = build_layout()
        self.platform_data = []

        # Create one static box for each elevated support platform.
        # mass=0 makes the body static — it has infinite effective mass
        # and is unaffected by any collision force.
        for pos, yaw, hx, hy, hz in platform_specs:
            q = pybullet.getQuaternionFromEuler([0, 0, yaw])
            plat_col = pybullet.createCollisionShape(
                pybullet.GEOM_BOX, halfExtents=[hx, hy, hz])
            pid = pybullet.createMultiBody(0, plat_col, -1, list(pos), list(q))
            pybullet.changeDynamics(
                pid, -1,
                lateralFriction  = DOM_FRICTION,
                restitution      = 0.0,
                contactStiffness = 1e6,
                contactDamping   = 1e4,
            )
            self.platform_data.append((pos, q, hx, hy, hz))

        # Create the domino collision shape once and reuse it for all bodies.
        # Half-extents correspond to half the domino's depth, width, and height.
        dom_col = pybullet.createCollisionShape(
            pybullet.GEOM_BOX,
            halfExtents=[DOM_D * 0.5, DOM_W * 0.5, DOM_H * 0.5],
        )
        total = len(self._placements)
        for i, p in enumerate(self._placements):
            q   = pybullet.getQuaternionFromEuler([0, 0, p.yaw])
            # p.z is the footing elevation; the body's centre of mass is
            # half the domino height above that surface.
            pos = [p.x, p.y, p.z + DOM_H * 0.5]
            bid = pybullet.createMultiBody(DOM_MASS, dom_col, -1, pos, list(q))
            pybullet.changeDynamics(
                bid, -1,
                restitution     = DOM_RESTITUTION,
                lateralFriction = DOM_FRICTION,
                linearDamping   = 0.05,
                angularDamping  = 0.05,
            )
            self._dom_ids.append(bid)
            self.body_colors.append(body_color(i, total))

        # Reset every domino to its exact upright pose with zero velocity.
        # This eliminates any numerical noise from the body creation process.
        for bid, p in zip(self._dom_ids, self._placements):
            q   = pybullet.getQuaternionFromEuler([0, 0, p.yaw])
            pos = [p.x, p.y, p.z + DOM_H * 0.5]
            pybullet.resetBasePositionAndOrientation(bid, pos, q)
            pybullet.resetBaseVelocity(
                bid,
                linearVelocity =[0.0, 0.0, 0.0],
                angularVelocity=[0.0, 0.0, 0.0],
            )

        self._sync_states()

    def _sync_states(self) -> None:
        """Read the current pose of every domino body from PyBullet.

        Populates body_states with (position, orientation) tuples so that
        DominoViewport can build the model matrices for rendering without
        needing direct access to the physics world.
        """
        self.body_states = [
            pybullet.getBasePositionAndOrientation(bid)
            for bid in self._dom_ids
        ]

    # ── Control ───────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Apply a single angular impulse to domino 0 to begin the cascade.

        Only domino 0 ever receives an external force.  Every subsequent fall
        is caused by a neighbouring domino physically striking the next one.

        A pure torque (angular impulse) is used rather than a linear force.
        A torque rotates the domino forward about its own base edge.  Once
        the tile tilts past the tipping angle — arctan(D/H) ≈ 5.7° — its
        centre of mass passes beyond the pivot edge and gravity takes over.
        A linear force would slide the tile into its neighbour instead of
        tipping it, which does not produce a realistic fall.

        The torque axis is perpendicular to the domino's heading direction
        in the XY plane, so the tile rotates forward rather than sideways.
        The torque magnitude of 8 N·m is enough to rotate the tile ~6°
        (past the tipping threshold) in a single physics step without
        launching the tile off the platform.
        """
        if not self._dom_ids:
            return
        self._running = True
        p0  = self._placements[0]
        bid = self._dom_ids[0]

        # The torque axis is perpendicular to the heading (rotated 90° in XY).
        # A vector perpendicular to (cos θ, sin θ, 0) is (-sin θ, cos θ, 0).
        pybullet.applyExternalTorque(
            bid, -1,
            torqueObj=[-math.sin(p0.yaw) * 8.0,
                        math.cos(p0.yaw) * 8.0,
                        0.0],
            flags=pybullet.WORLD_FRAME,
        )

    def step(self, n: int) -> None:
        """Advance the simulation by n physics steps.

        This method implements the fixed-timestep loop.  The QTimer fires at
        approximately 60 Hz, and each call to step() advances the simulation
        by (n × PHYSICS_DT) seconds of simulated time.

        At n=4 (1× real-time), the simulation advances by 4 × 1/240 ≈ 16.7 ms
        per frame, matching wall-clock time.  Increasing n compresses time
        without changing physics accuracy — the same number of solver
        iterations runs per step, only the number of steps per frame changes.

        Args:
            n: Number of physics steps to advance this tick.
        """
        if not self._running:
            return
        for _ in range(n):
            pybullet.stepSimulation()
            self._sim_time += PHYSICS_DT
        self._sync_states()
        self._update_wave()

    def _update_wave(self) -> None:
        """Update the wave-front position from the most recently fallen tiles.

        A domino is considered fallen when the sum of its absolute Euler
        angles around X and Y exceeds 40°.  This threshold is well past the
        tipping angle (~5.7°) but well below 90° (fully flat on the floor),
        so it captures tiles that are actively falling rather than those that
        have already come to rest.

        The wave position is the average of the last five fallen dominoes'
        positions.  Averaging a small window smooths out the jitter that
        would arise from tracking a single tile.
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
            # Average the last five fallen positions to get a smooth wave front.
            rec = fallen[-5:]
            self._wave = (
                sum(v[0] for v in rec) / len(rec),
                sum(v[1] for v in rec) / len(rec),
                sum(v[2] for v in rec) / len(rec),
            )

    def reset(self) -> None:
        """Disconnect the current physics world and rebuild it from scratch.

        Disconnecting destroys all bodies and frees PyBullet's internal state.
        connect() then creates a fresh world and repopulates the scene,
        returning all dominoes to their upright initial poses.
        """
        if self._client >= 0:
            pybullet.disconnect(self._client)
        self._client = -1; self._dom_ids = []; self._placements = []
        self._n_fallen = 0; self._sim_time = 0.0; self._running = False
        self._wave = (0.0, 0.0, 0.5)
        self.body_states = []; self.platform_data = []; self.body_colors = []
        self.connect()

    def disconnect(self) -> None:
        """Shut down the PyBullet client cleanly.

        Called when the application window is closed.  Disconnecting releases
        the shared memory and sockets that PyBullet uses internally.
        """
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
    """OpenGL 3.3 Core Profile renderer for the domino scene.

    This widget reads physics body poses from DominoPhysics each frame and
    draws the scene using a Blinn-Phong shader.  It has no knowledge of
    the physics internals — the two subsystems communicate only through the
    public arrays on DominoPhysics (body_states, platform_data, body_colors).

    Rendering pipeline per frame:
        1. body_states provides (pos, orn) for each domino.
        2. For each body, _build_model_mat() constructs the 4×4 model matrix
           and it is uploaded as the uModel shader uniform.
        3. The body is drawn in its gradient colour.
        4. A second, smaller box is drawn on the domino's front face using an
           offset along the body's local X axis, giving the ivory stripe detail.
        5. Static platforms are drawn using their stored (pos, orn, hx, hy, hz).
        6. The floor is a single large quad rendered with a procedural
           checkerboard pattern computed in the fragment shader.

    All geometry uses the same unit-cube VAO.  Only the model matrix, colour,
    and specular exponent change between draw calls.
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
        """Set up OpenGL state, compile shaders, and upload geometry.

        Called once by Qt when the OpenGL context becomes available.
        """
        # Depth testing ensures closer surfaces hide surfaces behind them.
        GL.glEnable(GL.GL_DEPTH_TEST)
        # Back-face culling is disabled so that domino faces remain visible
        # from any angle, including while the tile is falling or upside-down.
        GL.glDisable(GL.GL_CULL_FACE)
        # Enable multisample anti-aliasing (4× MSAA configured in main()).
        GL.glEnable(GL.GL_MULTISAMPLE)

        # Set the background colour by decoding the hex string from PALETTE.
        GL.glClearColor(
            int(PALETTE["bg"][1:3], 16) / 255,
            int(PALETTE["bg"][3:5], 16) / 255,
            int(PALETTE["bg"][5:7], 16) / 255,
            1.0,
        )

        # Compile the GLSL source strings and link them into a shader program.
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

        # Upload the two geometry meshes to the GPU.
        self._make_box_vao()
        self._make_floor_vao()

    def resizeGL(self, w: int, h: int) -> None:
        """Update the OpenGL viewport when the widget is resized."""
        GL.glViewport(0, 0, w, h)

    def paintGL(self) -> None:
        """Draw the full scene for the current frame.

        Called by Qt whenever the widget needs to be repainted.  The method
        reads the latest body poses from DominoPhysics and draws the floor,
        support platforms, domino bodies, and face stripes in that order.
        """
        # Update the auto-tracking camera before drawing.
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
            """Return the uniform location for the named shader variable."""
            return GL.glGetUniformLocation(pid, name)

        # Upload uniforms that are shared across all draw calls this frame.
        GL.glUniformMatrix4fv(_u("uView"), 1, GL.GL_TRUE, view)
        GL.glUniformMatrix4fv(_u("uProj"), 1, GL.GL_TRUE, proj)
        # Key light: warm colour, positioned upper-left of the scene.
        GL.glUniform3fv(_u("uLight1Pos"),   1, [5.0, -8.0, 12.0])
        GL.glUniform3fv(_u("uLight1Color"), 1, [1.0, 0.95, 0.88])
        # Fill light: cool colour, opposite side of the scene at lower intensity.
        GL.glUniform3fv(_u("uLight2Pos"),   1, [-8.0, 5.0, 5.0])
        GL.glUniform3fv(_u("uLight2Color"), 1, [0.5, 0.65, 1.0])
        GL.glUniform3fv(_u("uCamPos"),      1, eye.tolist())

        identity = np.eye(4, dtype=np.float32)

        # Draw the floor quad.  The uFloor flag activates the checkerboard
        # pattern in the fragment shader.  The model matrix is the identity
        # because the floor quad is already defined in world space.
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

        # Draw the static support platforms as dark-coloured boxes.
        # The model matrix is built from each platform's stored pose and
        # half-extents (hx, hy, hz): the box is scaled by 2× the half-extents
        # so that the unit cube spans the full platform dimensions.
        GL.glUniform1f(_u("uAmbient"),   0.20)
        GL.glUniform1f(_u("uSpecPower"), 10.0)
        for pos, orn, hx, hy, hz in self._ph.platform_data:
            m = _build_model_mat(pos, orn, hx * 2, hy * 2, hz * 2)
            GL.glUniformMatrix4fv(_u("uModel"), 1, GL.GL_TRUE, m)
            GL.glUniform4f(_u("uBaseColor"), 0.20, 0.22, 0.30, 1.0)
            GL.glDrawElements(GL.GL_TRIANGLES, self._box_idx_count,
                              GL.GL_UNSIGNED_INT, None)

        # Draw each domino as two boxes: the coloured body and the ivory face stripe.
        #
        # Face stripe geometry:
        #   The stripe is a thin box placed on the front face of the domino.
        #   Its centre is offset from the body centre along the domino's local
        #   X axis (the direction the domino faces).  The first column of the
        #   model matrix, m[:3, 0], is the scaled local X axis in world space.
        #   Dividing by DOM_D gives the unit direction vector, which is then
        #   used to compute the stripe centre position.  This approach works
        #   correctly for any body orientation, including mid-fall.
        FACE_D = DOM_D * 0.07   # stripe thickness in metres (~2.1 mm)
        GL.glUniform1f(_u("uAmbient"),   0.14)

        for i, (pos, orn) in enumerate(self._ph.body_states):
            r, g, b = self._ph.body_colors[i]
            m = _build_model_mat(pos, orn, DOM_D, DOM_W, DOM_H)

            # Draw the domino body in its gradient colour.
            GL.glUniformMatrix4fv(_u("uModel"), 1, GL.GL_TRUE, m)
            GL.glUniform4f(_u("uBaseColor"), r, g, b, 1.0)
            GL.glUniform1f(_u("uSpecPower"), 32.0)
            GL.glDrawElements(GL.GL_TRIANGLES, self._box_idx_count,
                              GL.GL_UNSIGNED_INT, None)

            # Compute the face stripe model matrix.
            # m[:3, 0] is the local X axis scaled by DOM_D; dividing gives the
            # unit direction vector in world space.
            local_x  = m[:3, 0] / DOM_D
            face_pos = np.array(pos, dtype=np.float32) + local_x * (DOM_D * 0.5 + FACE_D * 0.5)
            mf = _build_model_mat(face_pos.tolist(), orn,
                                   FACE_D, DOM_W * 0.84, DOM_H * 0.88)
            GL.glUniformMatrix4fv(_u("uModel"), 1, GL.GL_TRUE, mf)
            GL.glUniform4f(_u("uBaseColor"), 0.96, 0.94, 0.86, 1.0)
            GL.glUniform1f(_u("uSpecPower"), 110.0)
            GL.glDrawElements(GL.GL_TRIANGLES, self._box_idx_count,
                              GL.GL_UNSIGNED_INT, None)

        GL.glBindVertexArray(0)
        self._prog.release()

    # ── Mesh generation ───────────────────────────────────────────────────────

    def _make_box_vao(self) -> None:
        """Upload a unit cube mesh to the GPU.

        The cube spans from -0.5 to +0.5 on each axis.  Each face has its own
        four vertices so that each vertex can carry the outward-facing normal
        for that face.  Sharing vertices between faces would require averaging
        normals, which would produce incorrect flat shading.

        Vertex buffer layout (stride = 24 bytes):
            bytes  0–11:  position  (3 × float32)
            bytes 12–23:  normal    (3 × float32)

        Index buffer: 6 faces × 2 triangles × 3 indices = 36 uint32 values.
        Vertices are wound counter-clockwise when viewed from outside the cube.
        """
        # Each face is defined by its outward normal and four corner positions.
        # Corners are listed counter-clockwise when viewed from outside the face.
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
            ("Spacing",     f"{DOMINO_SPACING*100:.0f} mm"),
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
            f"<b>Canyon Cascade + Branches</b><br>"
            f"<br>"
            f"<span style='color:{PALETTE['muted']}'>"
            f"{total} dominoes · descend · climb · 3 final arms</span>"
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
            "PyBullet Physics · OpenGL 3.3 · Canyon Cascade + Branches"
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
