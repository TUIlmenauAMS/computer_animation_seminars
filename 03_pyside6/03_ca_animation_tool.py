"""03_ca_animation_tool.py — Mini Animation Tool.

A complete, professional-feeling desktop animation application that combines
all four previous seminar 03 components into one multi-panel window:

  - GLViewport     — QOpenGLWidget rendering an animated sphere with Phong lighting
  - TimelineWidget — frame scrubber, play/pause/stop, transport controls
  - CurveEditor    — QPainter graph of the bounce easing curve, draggable Bézier handles
  - ParameterPanel — sliders and colour pickers for all animation properties

All panels communicate exclusively through Qt signals — no panel holds a reference
to another panel.  MainWindow is the only place that knows all panels exist, and it
wires them together.  This architecture mirrors (at a smaller scale) Maya, Blender,
and Nuke.

How to run
----------
    uv sync                                  # first time only
    uv run python 03_ca_animation_tool.py

Ubuntu system dependencies (once):
    sudo apt install libgl1 libegl1 libxcb-cursor0 libxcb-xinerama0

    libgl1           — Mesa OpenGL (required by PyOpenGL)
    libegl1          — EGL interface (required by PySide6's Qt platform plugin)
    libxcb-cursor0   — XCB cursor library (required by Qt's xcb plugin on Ubuntu 22.04+)
    libxcb-xinerama0 — multi-monitor support for Qt

Windows: no extra steps needed — PySide6 bundles everything.

Connections to other seminars
------------------------------
- GLViewport uses the same OpenGL 3.3 pipeline as demo 4 (03_ca_opengl_widget.py),
  now rendering a UV sphere instead of a cube, with Phong lighting in GLSL.
- TimelineWidget is the same component as demo 2 (03_ca_timeline.py), extended
  with an FPS selector and a total-frames spinbox.
- CurveEditor draws the same cubic Bézier easing curve as demo 3 (03_ca_curves.py),
  adapted to show the full bounce arc over all frames rather than t in [0,1].
- BounceAnimation.evaluate() is a pure Python function — no Qt, no OpenGL.
  The same function could drive a web animation, an audio envelope, or a render
  farm script without modification.  Separating the math from the UI is the most
  important architectural principle in animation tooling.
"""

from __future__ import annotations

import ctypes
import math
import sys

import numpy as np
from OpenGL import GL
from OpenGL.GL import shaders as gl_shaders
from PySide6.QtCore import (
    Qt,
    QElapsedTimer,
    QSettings,
    QTimer,
    Signal,
)
from PySide6.QtGui import (
    QColor,
    QFont,
    QKeySequence,
    QPainter,
    QPen,
    QBrush,
    QShortcut,
    QSurfaceFormat,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMenuBar,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
    QOpenGLWidget,
    QMessageBox,
)

# ── QSurfaceFormat — MUST be set before QApplication ─────────────────────────
#
# QT CONCEPT — surface format ordering:
# This configures the OpenGL context that QOpenGLWidget will create.
# It must be called before QApplication() constructs the platform plugin.
# See demo 4 for the full explanation.

def _configure_surface_format() -> None:
    """Request OpenGL 3.3 Core Profile, 24-bit depth, 4× MSAA."""
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setDepthBufferSize(24)
    fmt.setSamples(4)
    QSurfaceFormat.setDefaultFormat(fmt)


_configure_surface_format()


# ── Constants ─────────────────────────────────────────────────────────────────

WINDOW_TITLE: str = "Animation Tool — Computeranimation Seminar 03"
WINDOW_MIN_WIDTH: int = 1100
WINDOW_MIN_HEIGHT: int = 680

# Animation defaults
DEFAULT_TOTAL_FRAMES: int = 120      # 4 s at 30 fps
DEFAULT_FPS: int = 30
DEFAULT_BOUNCE_HEIGHT: float = 2.0   # world-space units
DEFAULT_SPHERE_RADIUS: float = 0.25  # world-space units
DEFAULT_EASING_P1: tuple[float, float] = (0.42, 0.0)   # CSS ease-in equivalent
DEFAULT_EASING_P2: tuple[float, float] = (0.58, 1.0)   # CSS ease-out equivalent

# OpenGL / rendering
CLEAR_COLOR: tuple[float, float, float, float] = (0.051, 0.059, 0.078, 1.0)  # #0d0f14
SPHERE_STACKS: int = 24             # UV sphere latitude subdivisions
SPHERE_SECTORS: int = 36            # UV sphere longitude subdivisions
GROUND_Y: float = -0.5              # world-space Y position of the ground plane
GROUND_HALF_SIZE: float = 2.5       # half-width of the ground quad

# Timer
RENDER_INTERVAL_MS: int = 16        # ~60 fps render loop

# Curve editor
HANDLE_RADIUS: int = 7              # px — hit-test radius for Bézier handles
BEZIER_SAMPLES: int = 200           # number of points sampled to draw the curve

# Colours (design system)
COLOR_BG: str = "#0d0f14"
COLOR_SURFACE: str = "#161a22"
COLOR_BORDER: str = "#252b38"
COLOR_TEXT: str = "#d8dce6"
COLOR_MUTED: str = "#5a6175"
COLOR_ACCENT: str = "#4f9cf9"
COLOR_ACCENT2: str = "#a78bfa"
COLOR_GREEN: str = "#34d399"
COLOR_ORANGE: str = "#fb923c"

PANEL_STYLESHEET: str = (
    f"QWidget {{ background: {COLOR_SURFACE}; color: {COLOR_TEXT};"
    f"          font-family: 'IBM Plex Mono'; font-size: 12px; }}"
    f"QLabel  {{ color: {COLOR_MUTED}; }}"
    f"QSlider::groove:horizontal {{ height: 4px; background: {COLOR_BORDER};"
    f"                              border-radius: 2px; }}"
    f"QSlider::handle:horizontal {{ width: 12px; height: 12px;"
    f"                              background: {COLOR_ACCENT}; border-radius: 6px;"
    f"                              margin: -4px 0; }}"
    f"QSlider::sub-page:horizontal {{ background: {COLOR_ACCENT}; border-radius: 2px; }}"
    f"QPushButton {{ background: #1e2330; border: 1px solid {COLOR_BORDER};"
    f"               border-radius: 4px; padding: 5px 10px; color: {COLOR_TEXT}; }}"
    f"QPushButton:hover {{ background: {COLOR_BORDER}; }}"
    f"QCheckBox {{ color: {COLOR_TEXT}; spacing: 8px; }}"
    f"QCheckBox::indicator {{ width: 14px; height: 14px;"
    f"                        border: 1px solid {COLOR_BORDER}; border-radius: 3px;"
    f"                        background: #1e2330; }}"
    f"QCheckBox::indicator:checked {{ background: {COLOR_ACCENT}; }}"
    f"QComboBox {{ background: #1e2330; border: 1px solid {COLOR_BORDER};"
    f"             border-radius: 4px; padding: 4px 8px; color: {COLOR_TEXT}; }}"
    f"QSpinBox  {{ background: #1e2330; border: 1px solid {COLOR_BORDER};"
    f"             border-radius: 4px; padding: 4px 8px; color: {COLOR_TEXT}; }}"
)


# ── STEP 1: BounceAnimation — the pure model layer ────────────────────────────

class BounceAnimation:
    """Pure animation model — no Qt, no OpenGL dependencies.

    ANIMATION CONCEPT — model / view separation:
    BounceAnimation holds all animation data and the evaluate() method that
    computes the sphere's Y position for any given frame.  It has zero imports
    from PySide6 or OpenGL.  This means:
      - It is trivially unit-testable with pytest.
      - The same object could drive a web canvas, an audio envelope, or a
        render-farm batch script without any changes.
      - Each UI panel (GLViewport, CurveEditor, ParameterPanel) reads from
        this model; none of them modifies another panel directly.

    This separation is the most important architectural principle in animation
    tooling.  Maya's animation engine, Blender's depsgraph, and Houdini's
    parameter expressions all embody the same idea: pure math in one place,
    UI in another.

    Attributes:
        total_frames:  Total frame count of the clip.
        fps:           Playback frame rate.
        bounce_height: Peak Y position of the sphere (world units).
        sphere_radius: Radius of the sphere (world units, for rendering).
        sphere_color:  RGB tuple in [0, 1] — sphere diffuse colour.
        ground_color:  RGB tuple in [0, 1] — ground plane colour.
        wireframe:     Render in wireframe mode if True.
        easing_p1:     First cubic Bézier control point (x, y) in [0, 1].
        easing_p2:     Second cubic Bézier control point (x, y) in [0, 1].
    """

    def __init__(self) -> None:
        """Initialise with default animation parameters."""
        self.total_frames: int = DEFAULT_TOTAL_FRAMES
        self.fps: int = DEFAULT_FPS
        self.bounce_height: float = DEFAULT_BOUNCE_HEIGHT
        self.sphere_radius: float = DEFAULT_SPHERE_RADIUS
        self.sphere_color: tuple[float, float, float] = (0.31, 0.61, 0.98)
        self.ground_color: tuple[float, float, float] = (0.145, 0.102, 0.220)
        self.wireframe: bool = False
        self.easing_p1: tuple[float, float] = DEFAULT_EASING_P1
        self.easing_p2: tuple[float, float] = DEFAULT_EASING_P2

    @property
    def duration_seconds(self) -> float:
        """Total clip duration in seconds."""
        return self.total_frames / self.fps

    def evaluate(self, frame: int) -> float:
        """Compute the sphere's Y world-position at the given frame.

        The bounce is a parabolic arc (sin²) shaped by the Bézier easing
        curve.  The arc rises from GROUND_Y, peaks at bounce_height,
        and returns to GROUND_Y.

        ANIMATION CONCEPT — easing applied to a physical arc:
        The raw arc is sin²(π·t), which is already smooth.  We then remap t
        through the cubic Bézier easing function — the same algorithm used by
        CSS cubic-bezier() and Three.js InterpolateSmooth.  Remapping t before
        evaluating the arc changes the *feel* of the motion (ease-in launches
        slowly, ease-out lands softly) without changing the arc's shape.

        This is exactly what professional animators do: they design the curve
        in a graph editor (like the CurveEditor panel below), and the easing
        curve modulates how quickly the animation moves through its keyframes.

        Args:
            frame: Frame number (clamped to [0, total_frames - 1]).

        Returns:
            Y world-position of the sphere centre.
        """
        frame = max(0, min(frame, self.total_frames - 1))
        # Normalise frame to [0, 1].
        t_raw = frame / max(self.total_frames - 1, 1)
        # Apply Bézier easing to get the eased t in [0, 1].
        t_eased = _cubic_bezier(
            t_raw,
            self.easing_p1[0], self.easing_p1[1],
            self.easing_p2[0], self.easing_p2[1],
        )
        # Map to a parabolic arc: 0 at t=0, peak at t=0.5, 0 at t=1.
        arc = math.sin(math.pi * t_eased) ** 2
        return GROUND_Y + self.sphere_radius + arc * self.bounce_height


def _cubic_bezier(t: float, p1x: float, p1y: float, p2x: float, p2y: float) -> float:
    """Evaluate a CSS-style cubic Bézier easing at time t in [0, 1].

    ANIMATION CONCEPT — the CSS cubic-bezier() algorithm:
    A cubic Bézier with fixed endpoints P0=(0,0) and P3=(1,1) is
    parameterised by u ∈ [0,1]:
        x(u) = 3·p1x·u(1-u)² + 3·p2x·u²(1-u) + u³
        y(u) = 3·p1y·u(1-u)² + 3·p2y·u²(1-u) + u³

    We want y(t), but t is given as an x-value.  Because x(u) is
    monotonically increasing, we binary-search for the u where x(u)=t,
    then return y(u).  This is the same algorithm that browsers use
    internally to implement CSS transition-timing-function: cubic-bezier().

    Args:
        t:   Time parameter in [0, 1] (the x-axis value we want to invert).
        p1x: X coordinate of first control point.
        p1y: Y coordinate of first control point.
        p2x: X coordinate of second control point.
        p2y: Y coordinate of second control point.

    Returns:
        Eased output value in [0, 1] (the y-axis value at x=t).
    """
    # Edge cases: t=0 always maps to 0, t=1 always maps to 1.
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0

    def _bx(u: float) -> float:
        """Cubic Bézier x(u)."""
        return 3 * p1x * u * (1 - u) ** 2 + 3 * p2x * u**2 * (1 - u) + u**3

    def _by(u: float) -> float:
        """Cubic Bézier y(u)."""
        return 3 * p1y * u * (1 - u) ** 2 + 3 * p2y * u**2 * (1 - u) + u**3

    # Binary search for u such that x(u) ≈ t.
    lo, hi = 0.0, 1.0
    for _ in range(32):          # 32 iterations → precision ≈ 2⁻³² ≈ 2.3e-10
        mid = (lo + hi) * 0.5
        if _bx(mid) < t:
            lo = mid
        else:
            hi = mid
    return _by((lo + hi) * 0.5)


# ── STEP 2: GLViewport — OpenGL 3.3 sphere renderer ──────────────────────────

# ── GLSL shaders ──────────────────────────────────────────────────────────────

_VERT_SRC: str = """
#version 330 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProjection;
uniform mat3 uNormalMatrix;   // transpose(inverse(uModel)) for correct normal transform

out vec3 vNormal;
out vec3 vFragPos;

void main() {
    vec4 worldPos   = uModel * vec4(aPos, 1.0);
    vFragPos        = worldPos.xyz;
    vNormal         = normalize(uNormalMatrix * aNormal);
    gl_Position     = uProjection * uView * worldPos;
}
"""

_FRAG_SRC: str = """
#version 330 core

in vec3 vNormal;
in vec3 vFragPos;

uniform vec3 uLightPos;      // world-space light position
uniform vec3 uCameraPos;     // world-space camera position
uniform vec3 uDiffuseColor;  // material diffuse colour
uniform float uAmbient;      // ambient strength

out vec4 fragColor;

void main() {
    // Phong lighting — ambient + diffuse + specular.
    //
    // ANIMATION CONCEPT — lighting separates shape from colour:
    // The Phong model is the same lighting used in seminar 02's Three.js
    // MeshPhongMaterial, just written by hand.  Understanding it here lets
    // students see what Three.js is doing internally.

    vec3 N = normalize(vNormal);
    vec3 L = normalize(uLightPos - vFragPos);   // light direction
    vec3 V = normalize(uCameraPos - vFragPos);  // view direction
    vec3 R = reflect(-L, N);                    // reflection direction

    float diff = max(dot(N, L), 0.0);
    float spec = pow(max(dot(V, R), 0.0), 32.0);

    vec3 ambient  = uAmbient * uDiffuseColor;
    vec3 diffuse  = diff * uDiffuseColor;
    vec3 specular = spec * vec3(0.4);

    fragColor = vec4(ambient + diffuse + specular, 1.0);
}
"""

_FLAT_VERT_SRC: str = """
#version 330 core
layout(location = 0) in vec3 aPos;
uniform mat4 uMVP;
void main() { gl_Position = uMVP * vec4(aPos, 1.0); }
"""

_FLAT_FRAG_SRC: str = """
#version 330 core
uniform vec3 uColor;
out vec4 fragColor;
void main() { fragColor = vec4(uColor, 1.0); }
"""


def _make_uv_sphere(
    stacks: int, sectors: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build a UV sphere centred at the origin with radius 1.

    Returns interleaved (position xyz, normal xyz) vertex data and
    an index array for GL_TRIANGLES.

    Args:
        stacks:  Number of horizontal rings (latitude).
        sectors: Number of vertical slices (longitude).

    Returns:
        Tuple of (vertices float32 array, indices uint32 array).
    """
    verts = []
    for i in range(stacks + 1):
        phi = math.pi / 2 - i * math.pi / stacks     # from +π/2 to -π/2
        y = math.sin(phi)
        xz = math.cos(phi)
        for j in range(sectors + 1):
            theta = j * 2 * math.pi / sectors
            x = xz * math.cos(theta)
            z = xz * math.sin(theta)
            # For a unit sphere: normal == position.
            verts.extend([x, y, z, x, y, z])

    indices = []
    for i in range(stacks):
        k1 = i * (sectors + 1)
        k2 = k1 + sectors + 1
        for j in range(sectors):
            if i != 0:
                indices.extend([k1 + j, k2 + j, k1 + j + 1])
            if i != stacks - 1:
                indices.extend([k1 + j + 1, k2 + j, k2 + j + 1])

    return (
        np.array(verts, dtype=np.float32),
        np.array(indices, dtype=np.uint32),
    )


def _make_quad(
    y: float, half: float
) -> tuple[np.ndarray, np.ndarray]:
    """Build a flat horizontal quad (the ground plane) at height y.

    Args:
        y:    World-space Y position of the quad.
        half: Half-size (quad spans [-half, +half] on X and Z).

    Returns:
        Tuple of (vertices float32, indices uint32).
    """
    verts = np.array([
        -half, y, -half,  0, 1, 0,
         half, y, -half,  0, 1, 0,
         half, y,  half,  0, 1, 0,
        -half, y,  half,  0, 1, 0,
    ], dtype=np.float32)
    indices = np.array([0, 1, 2,  2, 3, 0], dtype=np.uint32)
    return verts, indices


def _mat4_perspective(fovy_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    """Perspective projection matrix (NumPy, row-major)."""
    f = 1.0 / math.tan(math.radians(fovy_deg) / 2.0)
    nf = 1.0 / (near - far)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) * nf
    m[2, 3] = 2.0 * far * near * nf
    m[3, 2] = -1.0
    return m


def _mat4_look_at(eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
    """View matrix (NumPy, row-major)."""
    f = center - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u2 = np.cross(r, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3] = r
    m[1, :3] = u2
    m[2, :3] = -f
    m[0, 3] = -np.dot(r, eye)
    m[1, 3] = -np.dot(u2, eye)
    m[2, 3] = np.dot(f, eye)
    return m


def _mat4_translate_scale(tx: float, ty: float, tz: float, s: float) -> np.ndarray:
    """Translation + uniform scale matrix (row-major)."""
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = s
    m[1, 1] = s
    m[2, 2] = s
    m[0, 3] = tx
    m[1, 3] = ty
    m[2, 3] = tz
    return m


_EYE = np.array([0.0, 2.0, 5.0], dtype=np.float32)
_CENTER = np.array([0.0, 0.5, 0.0], dtype=np.float32)
_UP = np.array([0.0, 1.0, 0.0], dtype=np.float32)
_LIGHT_POS = np.array([3.0, 5.0, 3.0], dtype=np.float32)


class GLViewport(QOpenGLWidget):
    """QOpenGLWidget that renders the bouncing sphere with Phong lighting.

    ARCHITECTURE CONCEPT — viewport as a pure view:
    GLViewport has no knowledge of the TimelineWidget, CurveEditor, or
    ParameterPanel.  It exposes one public method: set_frame(frame).
    When called, it reads the animation position from BounceAnimation.evaluate()
    and schedules a repaint.  No other coupling exists.

    The fps property is polled by the status bar update slot in MainWindow.
    """

    def __init__(self, anim: BounceAnimation, parent: QWidget | None = None) -> None:
        """Initialise with a reference to the shared animation model.

        Args:
            anim:   The shared BounceAnimation model.
            parent: Optional Qt parent widget.
        """
        super().__init__(parent)
        self._anim = anim
        self.setMinimumSize(300, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._sphere_y: float = GROUND_Y + anim.sphere_radius
        self.fps: float = 0.0
        self._fps_frames: int = 0
        self._fps_ms: int = 0

        self._elapsed = QElapsedTimer()
        self._elapsed.start()
        self._last_ms: int = 0

        # GL handles — set in initializeGL
        self._prog_phong: int = 0
        self._prog_flat: int = 0
        self._sphere_vao: int = 0
        self._sphere_vbo: int = 0
        self._sphere_ebo: int = 0
        self._sphere_index_count: int = 0
        self._ground_vao: int = 0
        self._ground_vbo: int = 0
        self._ground_ebo: int = 0

        self._proj: np.ndarray = np.eye(4, dtype=np.float32)
        self._view: np.ndarray = _mat4_look_at(_EYE, _CENTER, _UP)

    # ── Public API ─────────────────────────────────────────────────────────────

    def set_frame(self, frame: int) -> None:
        """Update the sphere position for the given frame and schedule repaint.

        Args:
            frame: The current animation frame number.
        """
        self._sphere_y = self._anim.evaluate(frame)
        self.update()

    # ── QOpenGLWidget overrides ────────────────────────────────────────────────

    def initializeGL(self) -> None:  # noqa: N802
        """Set up the OpenGL pipeline.

        QT CONCEPT — initializeGL:
        This is the first point where OpenGL calls are safe.  Qt guarantees
        the context is current here and in resizeGL / paintGL only.
        Calling any gl* function before this (e.g. in __init__) will crash
        or silently fail because no context exists yet.
        """
        GL.glClearColor(*CLEAR_COLOR)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_MULTISAMPLE)

        # ── Compile shaders ────────────────────────────────────────────────────
        try:
            vs_phong = gl_shaders.compileShader(_VERT_SRC, GL.GL_VERTEX_SHADER)
            fs_phong = gl_shaders.compileShader(_FRAG_SRC, GL.GL_FRAGMENT_SHADER)
            self._prog_phong = gl_shaders.compileProgram(vs_phong, fs_phong)

            vs_flat = gl_shaders.compileShader(_FLAT_VERT_SRC, GL.GL_VERTEX_SHADER)
            fs_flat = gl_shaders.compileShader(_FLAT_FRAG_SRC, GL.GL_FRAGMENT_SHADER)
            self._prog_flat = gl_shaders.compileProgram(vs_flat, fs_flat)
        except Exception as exc:  # noqa: BLE001
            print(f"[GLViewport] Shader error: {exc}")
            return

        for prog, label in ((self._prog_phong, "phong"), (self._prog_flat, "flat")):
            if not GL.glGetProgramiv(prog, GL.GL_LINK_STATUS):
                log = GL.glGetProgramInfoLog(prog).decode()
                print(f"[GLViewport] {label} link error: {log}")

        # ── Upload sphere geometry ─────────────────────────────────────────────
        sphere_verts, sphere_idx = _make_uv_sphere(SPHERE_STACKS, SPHERE_SECTORS)
        self._sphere_index_count = len(sphere_idx)
        self._sphere_vao, self._sphere_vbo, self._sphere_ebo = self._upload_mesh(
            sphere_verts, sphere_idx
        )

        # ── Upload ground geometry ─────────────────────────────────────────────
        ground_verts, ground_idx = _make_quad(0.0, GROUND_HALF_SIZE)
        self._ground_vao, self._ground_vbo, self._ground_ebo = self._upload_mesh(
            ground_verts, ground_idx
        )

        self._last_ms = self._elapsed.elapsed()

    def _upload_mesh(
        self, verts: np.ndarray, indices: np.ndarray
    ) -> tuple[int, int, int]:
        """Upload a mesh to a VAO/VBO/EBO and return the handles.

        Vertex layout: [x, y, z, nx, ny, nz] — position + normal, 6 floats/vertex.

        Args:
            verts:   Float32 interleaved vertex+normal array.
            indices: Uint32 index array.

        Returns:
            Tuple of (vao, vbo, ebo) integer handles.
        """
        vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(vao)

        vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, verts.nbytes, verts, GL.GL_STATIC_DRAW)

        ebo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, ebo)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL.GL_STATIC_DRAW)

        stride = 6 * 4  # 6 floats × 4 bytes
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(12))
        GL.glEnableVertexAttribArray(1)

        GL.glBindVertexArray(0)
        return int(vao), int(vbo), int(ebo)

    def resizeGL(self, w: int, h: int) -> None:  # noqa: N802
        """Update viewport and projection on resize.

        Args:
            w: New width in pixels.
            h: New height in pixels.
        """
        GL.glViewport(0, 0, w, h)
        self._proj = _mat4_perspective(45.0, w / max(h, 1), 0.1, 50.0)

    def paintGL(self) -> None:  # noqa: N802
        """Render one frame: ground plane + animated sphere."""
        # FPS tracking
        now_ms = self._elapsed.elapsed()
        delta_ms = now_ms - self._last_ms
        self._last_ms = now_ms
        self._fps_frames += 1
        self._fps_ms += delta_ms
        if self._fps_ms >= 500:
            self.fps = self._fps_frames / (self._fps_ms / 1000.0)
            self._fps_frames = 0
            self._fps_ms = 0

        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        if self._anim.wireframe:
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
        else:
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)

        # ── Draw ground with Phong ─────────────────────────────────────────────
        self._draw_phong(
            vao=self._ground_vao,
            index_count=6,
            model=_mat4_translate_scale(0, GROUND_Y, 0, 1.0),
            color=self._anim.ground_color,
        )

        # ── Draw sphere with Phong ─────────────────────────────────────────────
        self._draw_phong(
            vao=self._sphere_vao,
            index_count=self._sphere_index_count,
            model=_mat4_translate_scale(0.0, self._sphere_y, 0.0, self._anim.sphere_radius),
            color=self._anim.sphere_color,
        )

        if self._anim.wireframe:
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)

    def _draw_phong(
        self,
        vao: int,
        index_count: int,
        model: np.ndarray,
        color: tuple[float, float, float],
    ) -> None:
        """Draw a mesh with the Phong shader.

        Args:
            vao:         Vertex array object handle.
            index_count: Number of indices to draw.
            model:       Model matrix (row-major float32).
            color:       RGB diffuse colour in [0, 1].
        """
        GL.glUseProgram(self._prog_phong)

        # Normal matrix = transpose(inverse(model)) upper-left 3×3.
        # Required for correct normal transformation when the model matrix
        # contains non-uniform scaling.  (Our matrix only has uniform scale,
        # so the normal matrix equals the model's rotation part — but we
        # compute it correctly for generality.)
        normal_matrix = np.linalg.inv(model[:3, :3]).T.astype(np.float32)

        loc = lambda name: GL.glGetUniformLocation(self._prog_phong, name)  # noqa: E731
        GL.glUniformMatrix4fv(loc("uModel"),      1, GL.GL_TRUE, model)
        GL.glUniformMatrix4fv(loc("uView"),       1, GL.GL_TRUE, self._view)
        GL.glUniformMatrix4fv(loc("uProjection"), 1, GL.GL_TRUE, self._proj)
        GL.glUniformMatrix3fv(loc("uNormalMatrix"), 1, GL.GL_TRUE, normal_matrix)
        GL.glUniform3fv(loc("uLightPos"),   1, _LIGHT_POS)
        GL.glUniform3fv(loc("uCameraPos"),  1, _EYE)
        GL.glUniform3f(loc("uDiffuseColor"), *color)
        GL.glUniform1f(loc("uAmbient"), 0.18)

        GL.glBindVertexArray(vao)
        GL.glDrawElements(GL.GL_TRIANGLES, index_count, GL.GL_UNSIGNED_INT, None)
        GL.glBindVertexArray(0)
        GL.glUseProgram(0)


# ── STEP 3: CurveEditor — QPainter easing curve with draggable handles ────────

class CurveEditor(QWidget):
    """QPainter widget that shows and edits the bounce easing curve.

    Displays the full Y-position curve over all frames (X = frame, Y = height).
    An orange vertical playhead marks the current frame.
    Two draggable Bézier handles control the easing shape.
    Emits easing_changed(p1x, p1y, p2x, p2y) when handles are moved.

    ARCHITECTURE CONCEPT — CurveEditor as both view and controller:
    This widget both displays state (from BounceAnimation) and modifies it
    (by emitting easing_changed when the user drags handles).  The signal
    bubbles up to MainWindow, which updates BounceAnimation and triggers
    redraws on GLViewport.  CurveEditor itself never calls viewport.update().
    """

    easing_changed = Signal(float, float, float, float)
    """Emitted when Bézier handles are dragged.

    Args: p1x, p1y, p2x, p2y — new control point coordinates in [0, 1].
    """

    def __init__(self, anim: BounceAnimation, parent: QWidget | None = None) -> None:
        """Initialise with a reference to the shared animation model.

        Args:
            anim:   The shared BounceAnimation model.
            parent: Optional Qt parent widget.
        """
        super().__init__(parent)
        self._anim = anim
        self._playhead_frame: int = 0
        self._dragging: int = 0   # 0 = none, 1 = P1, 2 = P2
        self.setMinimumHeight(160)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)

    # ── Public API ─────────────────────────────────────────────────────────────

    def set_playhead(self, frame: int) -> None:
        """Move the orange playhead to the given frame and repaint.

        Args:
            frame: Current frame number.
        """
        self._playhead_frame = frame
        self.update()

    # ── Coordinate helpers ─────────────────────────────────────────────────────

    def _frame_to_x(self, frame: float) -> float:
        """Map a frame number to a canvas X pixel."""
        pad = 32
        return pad + (frame / max(self._anim.total_frames - 1, 1)) * (self.width() - 2 * pad)

    def _height_to_y(self, h: float) -> float:
        """Map a world-space height to a canvas Y pixel (inverted — top = high)."""
        pad = 16
        y_min = GROUND_Y
        y_max = GROUND_Y + DEFAULT_BOUNCE_HEIGHT * 1.2
        norm = (h - y_min) / (y_max - y_min)
        return (self.height() - pad) - norm * (self.height() - 2 * pad)

    def _bezier_handle_pixel(self, which: int) -> tuple[float, float]:
        """Return the pixel position of a Bézier handle.

        Args:
            which: 1 for P1, 2 for P2.

        Returns:
            (px, py) pixel coordinates.
        """
        total = self._anim.total_frames - 1
        if which == 1:
            px, py = self._anim.easing_p1
        else:
            px, py = self._anim.easing_p2
        # The handle x maps to frame space, handle y maps to a fraction of bounce height.
        frame_x = self._frame_to_x(px * total)
        world_y = GROUND_Y + self._anim.sphere_radius + py * self._anim.bounce_height
        return frame_x, self._height_to_y(world_y)

    def _pixel_to_bezier(self, px: float, py: float) -> tuple[float, float]:
        """Convert a pixel coordinate to a normalised Bézier handle coordinate.

        Args:
            px: Pixel X.
            py: Pixel Y.

        Returns:
            (t, v) normalised coordinates clamped to [0, 1].
        """
        pad = 32
        t = (px - pad) / max(self.width() - 2 * pad, 1)

        pad_y = 16
        y_min = GROUND_Y
        y_max = GROUND_Y + DEFAULT_BOUNCE_HEIGHT * 1.2
        norm = 1.0 - (py - pad_y) / max(self.height() - 2 * pad_y, 1)
        # Map norm back to [0, 1] where 0 = ground, 1 = peak
        world_y = y_min + norm * (y_max - y_min)
        v = (world_y - GROUND_Y - self._anim.sphere_radius) / max(self._anim.bounce_height, 0.01)

        return max(0.0, min(1.0, t)), max(0.0, min(1.0, v))

    # ── Mouse events ───────────────────────────────────────────────────────────

    def mousePressEvent(self, event) -> None:  # noqa: ANN001
        """Begin dragging the nearest Bézier handle if click is within radius.

        QT CONCEPT — mouse drag pattern:
        Qt drag handling requires overriding three events:
          mousePressEvent  → start drag if hit-test succeeds
          mouseMoveEvent   → update while button held
          mouseReleaseEvent→ end drag
        This mirrors mousedown/mousemove/mouseup in the browser.
        """
        mx, my = event.position().x(), event.position().y()
        for handle_id in (1, 2):
            hx, hy = self._bezier_handle_pixel(handle_id)
            if math.hypot(mx - hx, my - hy) <= HANDLE_RADIUS + 4:
                self._dragging = handle_id
                return

    def mouseMoveEvent(self, event) -> None:  # noqa: ANN001
        """Update the dragged handle and emit easing_changed."""
        if not self._dragging:
            return
        t, v = self._pixel_to_bezier(event.position().x(), event.position().y())
        if self._dragging == 1:
            self._anim.easing_p1 = (t, v)
        else:
            self._anim.easing_p2 = (t, v)
        self.easing_changed.emit(
            self._anim.easing_p1[0], self._anim.easing_p1[1],
            self._anim.easing_p2[0], self._anim.easing_p2[1],
        )
        self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: ANN001
        """End the drag."""
        self._dragging = 0

    # ── Painting ───────────────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:  # noqa: ANN001
        """Draw the easing curve, handles, playhead, and axis labels."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()

        # Background
        painter.fillRect(0, 0, w, h, QColor(COLOR_BG))

        # Grid (horizontal lines at 0%, 50%, 100% height)
        grid_pen = QPen(QColor(COLOR_BORDER), 1, Qt.PenStyle.DotLine)
        painter.setPen(grid_pen)
        for frac in (0.0, 0.5, 1.0):
            world_h = GROUND_Y + self._anim.sphere_radius + frac * self._anim.bounce_height
            gy = int(self._height_to_y(world_h))
            painter.drawLine(0, gy, w, gy)

        # ── Sample and draw the full bounce curve ──────────────────────────────
        frames = self._anim.total_frames
        pts = [
            (
                self._frame_to_x(f),
                self._height_to_y(self._anim.evaluate(f)),
            )
            for f in np.linspace(0, frames - 1, BEZIER_SAMPLES)
        ]
        curve_pen = QPen(QColor(COLOR_ACCENT), 2)
        painter.setPen(curve_pen)
        for i in range(len(pts) - 1):
            x0, y0 = pts[i]
            x1, y1 = pts[i + 1]
            painter.drawLine(int(x0), int(y0), int(x1), int(y1))

        # ── Playhead ───────────────────────────────────────────────────────────
        ph_x = int(self._frame_to_x(self._playhead_frame))
        painter.setPen(QPen(QColor(COLOR_ORANGE), 2))
        painter.drawLine(ph_x, 0, ph_x, h)

        # ── Bézier handles + tangent lines ────────────────────────────────────
        anchor_start = (self._frame_to_x(0), self._height_to_y(GROUND_Y + self._anim.sphere_radius))
        anchor_end   = (self._frame_to_x(frames - 1), self._height_to_y(GROUND_Y + self._anim.sphere_radius))

        tangent_pen = QPen(QColor(COLOR_MUTED), 1, Qt.PenStyle.DashLine)
        painter.setPen(tangent_pen)

        h1x, h1y = self._bezier_handle_pixel(1)
        h2x, h2y = self._bezier_handle_pixel(2)
        painter.drawLine(int(anchor_start[0]), int(anchor_start[1]), int(h1x), int(h1y))
        painter.drawLine(int(anchor_end[0]),   int(anchor_end[1]),   int(h2x), int(h2y))

        for (hx, hy), color in ((h1x, h1y), COLOR_GREEN), ((h2x, h2y), COLOR_ACCENT2):
            painter.setBrush(QBrush(QColor(color)))
            painter.setPen(QPen(QColor(COLOR_TEXT), 1))
            painter.drawEllipse(int(hx) - HANDLE_RADIUS, int(hy) - HANDLE_RADIUS,
                                HANDLE_RADIUS * 2, HANDLE_RADIUS * 2)

        # Label
        painter.setPen(QPen(QColor(COLOR_MUTED)))
        painter.setFont(QFont("IBM Plex Mono", 9))
        painter.drawText(6, h - 6, "Easing Curve")


# ── STEP 4: TimelineWidget ────────────────────────────────────────────────────

class TimelineWidget(QWidget):
    """Animation timeline: scrubber, transport controls, FPS selector.

    Reused from demo 2 (03_ca_timeline.py) with two additions:
      - FPS QComboBox (12, 24, 30, 60)
      - Total-frames QSpinBox

    Emits frame_changed(int) on every frame change.  All other panels
    react to this signal; none of them call into TimelineWidget directly.

    ARCHITECTURE CONCEPT — TimelineWidget as the animation clock:
    In professional DCC tools (Maya, Nuke, Houdini) the timeline is the
    master clock.  Every other panel subscribes to the current-time change.
    No panel reaches into the timeline to ask what frame it is on.  This is
    the observer pattern, implemented here with Qt signals.
    """

    frame_changed = Signal(int)
    """Emitted whenever the current frame changes."""

    def __init__(self, anim: BounceAnimation, parent: QWidget | None = None) -> None:
        """Build the timeline controls and connect internal signals.

        Args:
            anim:   The shared BounceAnimation model.
            parent: Optional Qt parent widget.
        """
        super().__init__(parent)
        self._anim = anim
        self._current_frame: int = 0
        self._is_playing: bool = False
        self.setMaximumHeight(110)
        self.setStyleSheet(PANEL_STYLESHEET)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)

        # ── Scrubber ──────────────────────────────────────────────────────────
        self._scrubber = QSlider(Qt.Orientation.Horizontal)
        self._scrubber.setRange(0, anim.total_frames - 1)
        layout.addWidget(self._scrubber)

        # ── Info labels ───────────────────────────────────────────────────────
        info_row = QHBoxLayout()
        self._frame_label = QLabel("Frame   0")
        self._frame_label.setStyleSheet(f"color: {COLOR_TEXT};")
        self._time_label  = QLabel("0.00 s")
        self._time_label.setStyleSheet(f"color: {COLOR_MUTED};")
        info_row.addWidget(self._frame_label)
        info_row.addWidget(self._time_label)
        info_row.addStretch()

        # FPS selector
        info_row.addWidget(QLabel("FPS:"))
        self._fps_combo = QComboBox()
        self._fps_combo.addItems(["12", "24", "30", "60"])
        self._fps_combo.setCurrentText(str(anim.fps))
        self._fps_combo.setFixedWidth(56)
        info_row.addWidget(self._fps_combo)

        # Total frames
        info_row.addWidget(QLabel("Frames:"))
        self._frames_spin = QSpinBox()
        self._frames_spin.setRange(12, 600)
        self._frames_spin.setValue(anim.total_frames)
        self._frames_spin.setFixedWidth(64)
        info_row.addWidget(self._frames_spin)

        layout.addLayout(info_row)

        # ── Transport buttons ─────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        back_btn = QPushButton("⏮")
        self._play_btn = QPushButton("▶  Play")
        stop_btn = QPushButton("⏹")
        fwd_btn  = QPushButton("⏭")
        for btn in (back_btn, self._play_btn, stop_btn, fwd_btn):
            btn.setFixedHeight(28)
        btn_row.addWidget(back_btn)
        btn_row.addWidget(self._play_btn)
        btn_row.addWidget(stop_btn)
        btn_row.addWidget(fwd_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # ── QTimer ────────────────────────────────────────────────────────────
        self._timer = QTimer(self)
        self._timer.setInterval(1000 // anim.fps)
        self._timer.timeout.connect(self._on_tick)

        # ── Signal connections ─────────────────────────────────────────────────
        self._scrubber.valueChanged.connect(self._on_scrubber)
        self._play_btn.clicked.connect(self._toggle_play)
        stop_btn.clicked.connect(self._stop)
        back_btn.clicked.connect(self.step_back)
        fwd_btn.clicked.connect(self.step_forward)
        self._fps_combo.currentTextChanged.connect(self._on_fps_changed)
        self._frames_spin.valueChanged.connect(self._on_total_frames_changed)

    # ── Public API ─────────────────────────────────────────────────────────────

    def step_back(self) -> None:
        """Step one frame backward and pause."""
        self._pause()
        self._set_frame(max(0, self._current_frame - 1))

    def step_forward(self) -> None:
        """Step one frame forward and pause."""
        self._pause()
        self._set_frame(min(self._anim.total_frames - 1, self._current_frame + 1))

    def go_to_first(self) -> None:
        """Jump to frame 0."""
        self._pause()
        self._set_frame(0)

    def go_to_last(self) -> None:
        """Jump to the last frame."""
        self._pause()
        self._set_frame(self._anim.total_frames - 1)

    def toggle_play(self) -> None:
        """Public toggle play/pause (called from keyboard shortcuts)."""
        self._toggle_play()

    @property
    def current_frame(self) -> int:
        """The currently displayed frame."""
        return self._current_frame

    # ── Internal ───────────────────────────────────────────────────────────────

    def _on_tick(self) -> None:
        """Advance one frame; loop back to 0 at the end."""
        self._set_frame((self._current_frame + 1) % self._anim.total_frames)

    def _toggle_play(self) -> None:
        if self._is_playing:
            self._pause()
        else:
            self._play()

    def _play(self) -> None:
        self._is_playing = True
        self._play_btn.setText("⏸  Pause")
        self._timer.start()

    def _pause(self) -> None:
        self._is_playing = False
        self._play_btn.setText("▶  Play")
        self._timer.stop()

    def _stop(self) -> None:
        self._pause()
        self._set_frame(0)

    def _on_scrubber(self, value: int) -> None:
        if self._is_playing:
            self._pause()
        self._set_frame(value)

    def _set_frame(self, frame: int) -> None:
        """Single point through which all frame changes flow.

        Updates state, syncs the scrubber without re-emitting, updates
        labels, and emits frame_changed.  All downstream panels react
        to frame_changed; none is called directly.

        Args:
            frame: New frame number.
        """
        self._current_frame = frame
        self._scrubber.blockSignals(True)
        self._scrubber.setValue(frame)
        self._scrubber.blockSignals(False)
        self._frame_label.setText(f"Frame {frame:4d}")
        t = frame / max(self._anim.fps, 1)
        self._time_label.setText(f"{t:.2f} s")
        self.frame_changed.emit(frame)

    def _on_fps_changed(self, text: str) -> None:
        """Update the playback rate when the FPS combo changes.

        Args:
            text: The selected FPS string (e.g. "30").
        """
        try:
            fps = int(text)
        except ValueError:
            return
        self._anim.fps = fps
        self._timer.setInterval(1000 // fps)

    def _on_total_frames_changed(self, value: int) -> None:
        """Update total frame count and resize the scrubber.

        Args:
            value: New total frame count.
        """
        self._anim.total_frames = value
        self._scrubber.setRange(0, value - 1)
        if self._current_frame >= value:
            self._set_frame(value - 1)


# ── STEP 5: ParameterPanel ────────────────────────────────────────────────────

class ParameterPanel(QWidget):
    """QFormLayout panel exposing all animation parameters as controls.

    Emits signals when values change; never touches other panels directly.

    ARCHITECTURE CONCEPT — ParameterPanel as the model setter:
    This panel writes to BounceAnimation and emits signals.  MainWindow
    catches those signals and tells the affected panels to repaint.
    The panel does not know whether a GLViewport or CurveEditor even exists.
    """

    bounce_height_changed = Signal(float)
    """Emitted when the bounce height slider changes."""

    sphere_radius_changed = Signal(float)
    """Emitted when the sphere radius slider changes."""

    color_changed = Signal()
    """Emitted when any colour is changed."""

    wireframe_changed = Signal(bool)
    """Emitted when the wireframe checkbox is toggled."""

    def __init__(self, anim: BounceAnimation, parent: QWidget | None = None) -> None:
        """Build the form and connect controls.

        Args:
            anim:   The shared BounceAnimation model.
            parent: Optional Qt parent widget.
        """
        super().__init__(parent)
        self._anim = anim
        self.setStyleSheet(PANEL_STYLESHEET)

        form = QFormLayout(self)
        form.setContentsMargins(12, 12, 12, 12)
        form.setSpacing(8)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        # Bounce height
        self._height_slider = QSlider(Qt.Orientation.Horizontal)
        self._height_slider.setRange(10, 500)   # internal: ×0.01 → 0.10 … 5.00
        self._height_slider.setValue(int(anim.bounce_height * 100))
        self._height_val = QLabel(f"{anim.bounce_height:.2f}")
        h_row = QHBoxLayout()
        h_row.addWidget(self._height_slider)
        h_row.addWidget(self._height_val)
        form.addRow("Bounce height:", h_row)

        # Sphere radius
        self._radius_slider = QSlider(Qt.Orientation.Horizontal)
        self._radius_slider.setRange(5, 100)    # internal: ×0.01 → 0.05 … 1.00
        self._radius_slider.setValue(int(anim.sphere_radius * 100))
        self._radius_val = QLabel(f"{anim.sphere_radius:.2f}")
        r_row = QHBoxLayout()
        r_row.addWidget(self._radius_slider)
        r_row.addWidget(self._radius_val)
        form.addRow("Sphere radius:", r_row)

        # Sphere colour
        self._sphere_color_btn = QPushButton("Choose…")
        self._sphere_color_btn.clicked.connect(self._pick_sphere_color)
        form.addRow("Sphere colour:", self._sphere_color_btn)

        # Ground colour
        self._ground_color_btn = QPushButton("Choose…")
        self._ground_color_btn.clicked.connect(self._pick_ground_color)
        form.addRow("Ground colour:", self._ground_color_btn)

        # Wireframe
        self._wireframe_cb = QCheckBox()
        self._wireframe_cb.setChecked(anim.wireframe)
        form.addRow("Wireframe:", self._wireframe_cb)

        # Current frame (read-only)
        self._frame_display = QLabel("0")
        self._frame_display.setStyleSheet(f"color: {COLOR_TEXT};")
        form.addRow("Current frame:", self._frame_display)

        # ── Connections ────────────────────────────────────────────────────────
        self._height_slider.valueChanged.connect(self._on_height)
        self._radius_slider.valueChanged.connect(self._on_radius)
        self._wireframe_cb.stateChanged.connect(self._on_wireframe)

    # ── Public API ─────────────────────────────────────────────────────────────

    def set_frame(self, frame: int) -> None:
        """Update the read-only current-frame display.

        Args:
            frame: Current frame number.
        """
        self._frame_display.setText(str(frame))

    def reset_to_defaults(self) -> None:
        """Reset all controls and the animation model to default values."""
        self._anim.bounce_height = DEFAULT_BOUNCE_HEIGHT
        self._anim.sphere_radius = DEFAULT_SPHERE_RADIUS
        self._anim.easing_p1 = DEFAULT_EASING_P1
        self._anim.easing_p2 = DEFAULT_EASING_P2
        self._height_slider.setValue(int(DEFAULT_BOUNCE_HEIGHT * 100))
        self._radius_slider.setValue(int(DEFAULT_SPHERE_RADIUS * 100))
        self._wireframe_cb.setChecked(False)

    # ── Slots ──────────────────────────────────────────────────────────────────

    def _on_height(self, value: int) -> None:
        h = value / 100.0
        self._anim.bounce_height = h
        self._height_val.setText(f"{h:.2f}")
        self.bounce_height_changed.emit(h)

    def _on_radius(self, value: int) -> None:
        r = value / 100.0
        self._anim.sphere_radius = r
        self._radius_val.setText(f"{r:.2f}")
        self.sphere_radius_changed.emit(r)

    def _pick_sphere_color(self) -> None:
        r, g, b = self._anim.sphere_color
        initial = QColor(int(r * 255), int(g * 255), int(b * 255))
        color = QColorDialog.getColor(initial, self, "Sphere Colour")
        if color.isValid():
            self._anim.sphere_color = (color.redF(), color.greenF(), color.blueF())
            self.color_changed.emit()

    def _pick_ground_color(self) -> None:
        r, g, b = self._anim.ground_color
        initial = QColor(int(r * 255), int(g * 255), int(b * 255))
        color = QColorDialog.getColor(initial, self, "Ground Colour")
        if color.isValid():
            self._anim.ground_color = (color.redF(), color.greenF(), color.blueF())
            self.color_changed.emit()

    def _on_wireframe(self, state: int) -> None:
        enabled = bool(state)
        self._anim.wireframe = enabled
        self.wireframe_changed.emit(enabled)


# ── STEP 6: MainWindow — assembles everything ─────────────────────────────────

class MainWindow(QMainWindow):
    """Application main window — the wiring hub for all panels.

    Layout:
    ┌──────────────────────────────────────────────┐
    │  Menu bar                                    │
    ├───────────────────────────┬──────────────────┤
    │  QSplitter (Horizontal)   │                  │
    │  ┌────────────────────┐   │  QSplitter (V)   │
    │  │   GLViewport       │   │  ┌────────────┐  │
    │  │   (OpenGL sphere)  │   │  │ CurveEditor│  │
    │  ├────────────────────┤   │  ├────────────┤  │
    │  │   TimelineWidget   │   │  │ Parameters │  │
    │  └────────────────────┘   │  └────────────┘  │
    └───────────────────────────┴──────────────────┤
    │  Status bar: Frame N / M  |  T s  |  FPS     │
    └──────────────────────────────────────────────┘

    ARCHITECTURE CONCEPT — MainWindow as the controller:
    MainWindow is the only object that holds references to all four panels
    and to the BounceAnimation model.  All inter-panel wiring happens here,
    in _connect_signals().  No panel imports any other panel.

    The signal flow is:
        TimelineWidget.frame_changed(int)
            → MainWindow._on_frame_changed(frame)
                → GLViewport.set_frame(frame)
                → CurveEditor.set_playhead(frame)
                → ParameterPanel.set_frame(frame)
                → (status bar update)

        ParameterPanel.bounce_height_changed(float)
            → CurveEditor.update()   [redraws the curve]
            → GLViewport.update()    [re-renders with new height]

        ParameterPanel.color_changed()
            → GLViewport.update()

        ParameterPanel.wireframe_changed(bool)
            → GLViewport.update()

        CurveEditor.easing_changed(p1x, p1y, p2x, p2y)
            → MainWindow._on_easing_changed(...)
                → (updates BounceAnimation.easing_p1/p2)
                → CurveEditor.update()
                → GLViewport.update()

    QT CONCEPT — QSplitter state persistence:
    splitter.saveState() returns a QByteArray that QSettings can store.
    splitter.restoreState(bytes) restores the panel sizes on next launch.
    This is how professional applications persist panel layouts.
    """

    def __init__(self) -> None:
        """Initialise the window, create all panels, wire signals."""
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)
        self.setStyleSheet(f"QMainWindow {{ background: {COLOR_BG}; }}")

        # ── Shared model ──────────────────────────────────────────────────────
        # ANIMATION CONCEPT — single source of truth:
        # One BounceAnimation instance is created here and passed by reference
        # to every panel.  Every panel reads from it; only ParameterPanel and
        # CurveEditor write to it (via their signals, handled here).
        self._anim = BounceAnimation()

        # ── Panels ────────────────────────────────────────────────────────────
        self._viewport  = GLViewport(self._anim)
        self._timeline  = TimelineWidget(self._anim)
        self._curve     = CurveEditor(self._anim)
        self._params    = ParameterPanel(self._anim)

        # ── Layout ────────────────────────────────────────────────────────────
        left_splitter = QSplitter(Qt.Orientation.Vertical)
        left_splitter.addWidget(self._viewport)
        left_splitter.addWidget(self._timeline)
        left_splitter.setStretchFactor(0, 1)
        left_splitter.setStretchFactor(1, 0)
        left_splitter.setSizes([420, 110])

        right_splitter = QSplitter(Qt.Orientation.Vertical)
        right_splitter.addWidget(self._curve)
        right_splitter.addWidget(self._params)
        right_splitter.setStretchFactor(0, 1)
        right_splitter.setStretchFactor(1, 0)
        right_splitter.setSizes([220, 220])
        right_splitter.setMinimumWidth(260)
        right_splitter.setMaximumWidth(380)

        self._h_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._h_splitter.addWidget(left_splitter)
        self._h_splitter.addWidget(right_splitter)
        self._h_splitter.setStretchFactor(0, 1)
        self._h_splitter.setStretchFactor(1, 0)

        self.setCentralWidget(self._h_splitter)

        # ── Menu bar ──────────────────────────────────────────────────────────
        self._build_menu()

        # ── Status bar ────────────────────────────────────────────────────────
        self._status_label = QLabel("Frame 0 / 120  |  0.00 s  |  -- FPS")
        self._status_label.setStyleSheet(
            f"color: {COLOR_MUTED}; font-family: 'IBM Plex Mono'; font-size: 11px;"
        )
        status_bar = QStatusBar()
        status_bar.addWidget(self._status_label)
        status_bar.setStyleSheet(f"background: {COLOR_SURFACE}; border-top: 1px solid {COLOR_BORDER};")
        self.setStatusBar(status_bar)

        # ── Render timer ──────────────────────────────────────────────────────
        # ARCHITECTURE CONCEPT — render timer lives in MainWindow:
        # The timer drives viewport repaints independent of whether the
        # animation is playing.  This ensures the sphere stays sharp after
        # a colour or parameter change even when the timeline is paused.
        self._render_timer = QTimer(self)
        self._render_timer.setInterval(RENDER_INTERVAL_MS)
        self._render_timer.timeout.connect(self._viewport.update)
        self._render_timer.start()

        # Status bar update timer (every 500 ms is plenty)
        self._status_timer = QTimer(self)
        self._status_timer.setInterval(500)
        self._status_timer.timeout.connect(self._update_status)
        self._status_timer.start()

        # ── Wire signals ──────────────────────────────────────────────────────
        self._connect_signals()

        # ── Keyboard shortcuts ─────────────────────────────────────────────────
        self._build_shortcuts()

        # ── Restore splitter state ─────────────────────────────────────────────
        self._restore_state()

    # ── Menu ───────────────────────────────────────────────────────────────────

    def _build_menu(self) -> None:
        """Build File / View / Help menu bar."""
        bar = self.menuBar()
        bar.setStyleSheet(
            f"QMenuBar {{ background: {COLOR_SURFACE}; color: {COLOR_TEXT}; }}"
            f"QMenuBar::item:selected {{ background: {COLOR_BORDER}; }}"
            f"QMenu {{ background: {COLOR_SURFACE}; color: {COLOR_TEXT};"
            f"         border: 1px solid {COLOR_BORDER}; }}"
            f"QMenu::item:selected {{ background: {COLOR_BORDER}; }}"
        )

        # File
        file_menu: QMenu = bar.addMenu("File")
        file_menu.addAction("Quit", self.close, QKeySequence("Ctrl+Q"))

        # View
        view_menu: QMenu = bar.addMenu("View")
        view_menu.addAction("Reset Panels", self._reset_panels)

        # Help
        help_menu: QMenu = bar.addMenu("Help")
        help_menu.addAction("About", self._show_about)

    # ── Signal wiring ──────────────────────────────────────────────────────────

    def _connect_signals(self) -> None:
        """Wire all inter-panel signals.

        ARCHITECTURE CONCEPT — every connection is documented here.
        Students can read this method as a complete map of the data flow.
        In a larger application, this wiring would live in a controller class
        separate from the window, but for a teaching demo one clear method
        is cleaner.
        """
        # TimelineWidget → all consumers
        # When the timeline advances (or is scrubbed), tell every panel.
        self._timeline.frame_changed.connect(self._on_frame_changed)

        # ParameterPanel → CurveEditor and GLViewport
        # When bounce height changes the curve shape and sphere peak change.
        self._params.bounce_height_changed.connect(lambda _: self._curve.update())
        self._params.bounce_height_changed.connect(lambda _: self._viewport.update())

        # Radius change affects GLViewport rendering only.
        self._params.sphere_radius_changed.connect(lambda _: self._viewport.update())

        # Colour and wireframe changes affect GLViewport only.
        self._params.color_changed.connect(self._viewport.update)
        self._params.wireframe_changed.connect(lambda _: self._viewport.update())

        # CurveEditor → BounceAnimation + downstream panels
        # When handles are dragged, update the model and repaint dependents.
        self._curve.easing_changed.connect(self._on_easing_changed)

    # ── Slots ──────────────────────────────────────────────────────────────────

    def _on_frame_changed(self, frame: int) -> None:
        """Fan out a timeline frame change to all dependent panels.

        ARCHITECTURE CONCEPT — fan-out in the controller:
        MainWindow routes the single frame_changed signal to every panel that
        cares about it.  No panel knows about the others.

        Args:
            frame: The new current frame.
        """
        self._viewport.set_frame(frame)
        self._curve.set_playhead(frame)
        self._params.set_frame(frame)

    def _on_easing_changed(
        self, p1x: float, p1y: float, p2x: float, p2y: float
    ) -> None:
        """Update BounceAnimation easing and trigger repaints.

        Args:
            p1x, p1y: New P1 control point.
            p2x, p2y: New P2 control point.
        """
        self._anim.easing_p1 = (p1x, p1y)
        self._anim.easing_p2 = (p2x, p2y)
        # Force a viewport update at the current frame so the sphere jumps
        # immediately to its new position under the new easing.
        self._viewport.set_frame(self._timeline.current_frame)
        self._curve.update()

    def _update_status(self) -> None:
        """Refresh the status bar label."""
        f = self._timeline.current_frame
        total = self._anim.total_frames
        t = f / max(self._anim.fps, 1)
        fps = self._viewport.fps
        self._status_label.setText(
            f"Frame {f:4d} / {total}  |  {t:.2f} s  |  {fps:.0f} FPS"
        )

    # ── Keyboard shortcuts ─────────────────────────────────────────────────────

    def _build_shortcuts(self) -> None:
        """Register keyboard shortcuts.  Mirrors Maya / Blender / Nuke conventions."""
        # Space: play/pause
        QShortcut(QKeySequence("Space"), self, activated=self._timeline.toggle_play)
        # Left / Right: step one frame
        QShortcut(QKeySequence(Qt.Key.Key_Left),  self, activated=self._timeline.step_back)
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, activated=self._timeline.step_forward)
        # Home / End: first / last frame
        QShortcut(QKeySequence(Qt.Key.Key_Home), self, activated=self._timeline.go_to_first)
        QShortcut(QKeySequence(Qt.Key.Key_End),  self, activated=self._timeline.go_to_last)
        # Ctrl+R: reset animation to defaults
        QShortcut(QKeySequence("Ctrl+R"), self, activated=self._reset_animation)

    # ── View helpers ───────────────────────────────────────────────────────────

    def _reset_panels(self) -> None:
        """Reset QSplitter sizes to their defaults."""
        self._h_splitter.setSizes([self.width() - 310, 310])

    def _reset_animation(self) -> None:
        """Reset animation parameters and easing to defaults (Ctrl+R)."""
        self._params.reset_to_defaults()
        self._curve.update()
        self._viewport.update()

    # ── Splitter state persistence ─────────────────────────────────────────────

    def _restore_state(self) -> None:
        """Restore splitter sizes from the previous session using QSettings.

        QT CONCEPT — QSettings for persistent layout:
        QSettings stores key/value pairs in a platform-appropriate location
        (registry on Windows, ~/.config on Linux, ~/Library on macOS).
        splitter.saveState() / restoreState() serialise the panel sizes so
        the user's preferred layout survives between launches.
        Professional applications always do this — it is one of the small
        things that makes a tool feel polished.
        """
        settings = QSettings("TUIlmenauAMS", "ComputeranimationSeminar03Demo5")
        state = settings.value("h_splitter_state")
        if state:
            self._h_splitter.restoreState(state)

    def closeEvent(self, event) -> None:  # noqa: ANN001, N802
        """Save splitter state on close.

        Args:
            event: The Qt close event.
        """
        settings = QSettings("TUIlmenauAMS", "ComputeranimationSeminar03Demo5")
        settings.setValue("h_splitter_state", self._h_splitter.saveState())
        super().closeEvent(event)

    def _show_about(self) -> None:
        """Display the About dialog."""
        QMessageBox.about(
            self,
            "About",
            "<b>Animation Tool</b><br>"
            "Computeranimation Seminar 03 — Demo 5<br><br>"
            "Institut für Medientechnik · AMS Group<br>"
            "Technische Universität Ilmenau<br><br>"
            "<a href='https://www.linkedin.com/in/renato-profeta/'>Renato Fera P.</a>",
        )


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    """Create QApplication, show the main window, start the event loop.

    QT CONCEPT — QApplication:
    QApplication must be the first Qt object created.  It initialises the
    platform plugin and event loop.  _configure_surface_format() was already
    called at module import time (before this function runs), satisfying the
    ordering requirement: surface format → QApplication → QWidgets.

    sys.exit(app.exec()) starts the event loop and propagates the exit code
    back to the OS when the last window closes.
    """
    app = QApplication(sys.argv)
    app.setApplicationName("ComputeranimationSeminar03Demo5")
    app.setOrganizationName("TUIlmenauAMS")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
