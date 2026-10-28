# =============================================================================
# 03_ca_opengl_widget.py — QOpenGLWidget with Modern OpenGL 3.3
# =============================================================================
# Author   : Renato Fera P.
# LinkedIn : https://www.linkedin.com/in/renato-profeta/
# Course   : Computeranimation — Chapter 03: PySide6 & Python Animation
# Institute: Institut für Medientechnik, Technische Universität Ilmenau
# -----------------------------------------------------------------------------
# Description:
#   A QMainWindow containing a QOpenGLWidget that renders a rotating coloured
#   cube using the modern OpenGL 3.3 core profile pipeline: VAO, VBO, EBO,
#   GLSL vertex and fragment shaders, and a full MVP matrix pipeline computed
#   with NumPy.
#
#   Every OpenGL call is annotated with its WebGL 2 equivalent from seminar 01,
#   making this demo the bridge between browser-based and desktop 3D rendering.
#
# How to run:
#   uv sync                            # first time only
#   uv run python 03_ca_opengl_widget.py
#
# Ubuntu system dependencies (once):
#   sudo apt install libgl1 libegl1 libxcb-cursor0 libxcb-xinerama0
#
# Windows: no extra steps needed — PySide6 bundles everything.
# =============================================================================

from __future__ import annotations

import ctypes
import sys
import math
from pathlib import Path

import numpy as np
from OpenGL import GL
from OpenGL.GL import shaders as gl_shaders
from PySide6.QtCore import Qt, QElapsedTimer, QTimer
from PySide6.QtGui import (
    QColor,
    QKeySequence,
    QPixmap,
    QShortcut,
    QSurfaceFormat,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QColorDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtOpenGLWidgets import QOpenGLWidget  # fix: moved module in PySide6 6.x

# ── STEP 1: QSurfaceFormat — MUST be called before QApplication ───────────────
#
# QT CONCEPT — surface format:
# QSurfaceFormat tells Qt which OpenGL version and features to request from
# the OS driver when it creates the rendering context.  It MUST be set before
# QApplication is constructed — after that point the context is already being
# negotiated and any format change is ignored.
#
# We request:
#   - OpenGL 3.3 Core Profile: the minimum version that supports VAOs and the
#     layout(location=N) attribute syntax in shaders.  "Core" means all
#     deprecated fixed-function OpenGL (glBegin/glEnd, glColor3f, etc.) is
#     removed — exactly what WebGL 2 also drops.
#   - 24-bit depth buffer: needed for depth testing (painter's algorithm
#     replacement).  Without it, faces draw in submission order, not depth order.
#   - 4× MSAA (multisampling): hardware anti-aliasing — smooths jagged edges
#     on the cube's silhouette.  Enable with glEnable(GL_MULTISAMPLE) in init.

def _configure_surface_format() -> None:
    """Request OpenGL 3.3 Core Profile, depth buffer, and 4× MSAA.

    Must be called before QApplication() is constructed.
    """
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setDepthBufferSize(24)
    fmt.setSamples(4)
    QSurfaceFormat.setDefaultFormat(fmt)


_configure_surface_format()   # ← runs at import time, before any Qt object


# ── Constants ─────────────────────────────────────────────────────────────────

WINDOW_TITLE: str = "OpenGL Widget — Computeranimation Seminar 03"
WINDOW_MIN_WIDTH: int = 800
WINDOW_MIN_HEIGHT: int = 520

CONTROL_PANEL_WIDTH: int = 220       # fixed right-panel width in px
TIMER_INTERVAL_MS: int = 16          # ~60 fps (1000 ms / 60 ≈ 16.67 ms)

CLEAR_COLOR_R: float = 0.051         # background colour ≈ #0d0f14 (matches design system)
CLEAR_COLOR_G: float = 0.059
CLEAR_COLOR_B: float = 0.078
CLEAR_COLOR_A: float = 1.0

CAMERA_EYE: tuple[float, float, float] = (0.0, 1.5, 4.0)   # camera position
CAMERA_CENTER: tuple[float, float, float] = (0.0, 0.0, 0.0) # look-at target
CAMERA_UP: tuple[float, float, float] = (0.0, 1.0, 0.0)     # world up vector

FOV_DEG: float = 45.0                # vertical field of view in degrees
Z_NEAR: float = 0.1                  # near clip plane
Z_FAR: float = 100.0                 # far clip plane

DEFAULT_SPEED_X: int = 20            # slider default: degrees/second on X axis
DEFAULT_SPEED_Y: int = 45            # slider default: degrees/second on Y axis
DEFAULT_SPEED_Z: int = 0             # slider default: degrees/second on Z axis
MAX_SPEED: int = 180                 # slider maximum: degrees/second

# AMS logo path — one level up from the script's directory (repo root)
LOGO_PATH: Path = Path(__file__).resolve().parent.parent / "ams_group_tui.png"


# ── STEP 2: GLSL shader source strings ────────────────────────────────────────
#
# WEBGL EQUIVALENT — shaders:
# These GLSL strings are nearly identical to the WebGL shaders in seminar 01.
# The only differences are the version directive (#version 330 core vs
# #version 300 es) and the "precision mediump float" that WebGL ES requires.
# The layout(location=N) syntax, uniforms, varyings (in/out), and gl_Position
# are identical across OpenGL 3.3 and WebGL 2.

VERTEX_SHADER_SRC: str = """
#version 330 core

// Input vertex attributes — bound to locations 0 and 1 in initializeGL.
// WEBGL EQUIVALENT: attribute variables declared in the shader, bound with
//   gl.bindAttribLocation() or queried with gl.getAttribLocation().
layout(location = 0) in vec3 aPos;    // position (x, y, z)
layout(location = 1) in vec3 aColor;  // per-vertex RGB colour

// MVP matrix uniforms — uploaded every frame from paintGL.
// WEBGL EQUIVALENT: uniform mat4 uModel; etc. — identical syntax.
uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProjection;

// Output to fragment shader (interpolated across the triangle).
// WEBGL EQUIVALENT: varying vec3 vColor; (WebGL 1) / out vec3 vColor; (WebGL 2)
out vec3 vColor;

void main() {
    // Transform the vertex through the MVP pipeline:
    //   Model space → World space (uModel)
    //   World space → Camera/View space (uView)
    //   View space  → Clip space (uProjection)
    // WEBGL EQUIVALENT: gl_Position = uProjection * uView * uModel * vec4(aPos, 1.0);
    gl_Position = uProjection * uView * uModel * vec4(aPos, 1.0);
    vColor = aColor;
}
"""

FRAGMENT_SHADER_SRC: str = """
#version 330 core

// Received from vertex shader (interpolated).
// WEBGL EQUIVALENT: in vec3 vColor; (WebGL 2)
in vec3 vColor;

// The output colour written to the framebuffer.
// WEBGL EQUIVALENT: gl_FragColor = vec4(vColor, 1.0); (WebGL 1)
//                   or: out vec4 fragColor; (WebGL 2 — identical to this)
out vec4 fragColor;

// Uniform tint applied on top of per-vertex colour.
// Set by the colour picker in the control panel.
uniform vec3 uTint;

void main() {
    // Multiply per-vertex colour by the tint.  Default tint is (1,1,1) → no change.
    fragColor = vec4(vColor * uTint, 1.0);
}
"""


# ── STEP 4: mat4_* helper functions — pure NumPy matrix utilities ──────────────
#
# In production code, use PyGLM or pyrr for these.  Here we implement them
# with NumPy so the underlying math is visible to students.
#
# IMPORTANT — row-major vs column-major:
# NumPy stores matrices in row-major order (C order).
# OpenGL/GLSL expects column-major matrices.
# When uploading with glUniformMatrix4fv we pass transpose=GL.GL_TRUE,
# which tells OpenGL to transpose on upload — turning our row-major NumPy
# matrices into the column-major form GLSL expects.
# WEBGL EQUIVALENT: gl.uniformMatrix4fv(loc, false, matrix)
#   WebGL also expects column-major, but glMatrix/Three.js build matrices
#   column-major directly.  The end result in the shader is the same.

def mat4_identity() -> np.ndarray:
    """Return a 4×4 identity matrix as float32.

    WEBGL EQUIVALENT: mat4.identity(out) from the glMatrix library.
    """
    return np.eye(4, dtype=np.float32)


def mat4_perspective(fovy_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    """Return a perspective projection matrix.

    Implements the same formula as glm::perspective and mat4.perspective
    (glMatrix).  Maps the view frustum to the OpenGL NDC cube [-1, 1]³.

    WEBGL EQUIVALENT: mat4.perspective(out, fovy, aspect, near, far)

    Args:
        fovy_deg: Vertical field of view in degrees.
        aspect:   Viewport width / height.
        near:     Near clip distance (must be > 0).
        far:      Far clip distance.

    Returns:
        4×4 float32 projection matrix in row-major order.
    """
    fovy_rad = math.radians(fovy_deg)
    f = 1.0 / math.tan(fovy_rad / 2.0)
    nf = 1.0 / (near - far)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) * nf
    m[2, 3] = 2.0 * far * near * nf
    m[3, 2] = -1.0
    return m


def mat4_look_at(
    eye: tuple[float, float, float],
    center: tuple[float, float, float],
    up: tuple[float, float, float],
) -> np.ndarray:
    """Return a view matrix (look-at transform).

    Constructs an orthonormal basis (right, up, forward) from the eye
    position, the point to look at, and the world-up vector, then builds
    the view matrix that transforms world space into camera space.

    WEBGL EQUIVALENT: mat4.lookAt(out, eye, center, up) from glMatrix.
    Three.js: camera.lookAt(target) computes the same internally.

    Args:
        eye:    Camera position in world space.
        center: Point the camera looks at.
        up:     World up vector (usually (0, 1, 0)).

    Returns:
        4×4 float32 view matrix in row-major order.
    """
    e = np.array(eye, dtype=np.float32)
    c = np.array(center, dtype=np.float32)
    u = np.array(up, dtype=np.float32)

    f = c - e                          # forward vector
    f = f / np.linalg.norm(f)

    r = np.cross(f, u)                 # right vector
    r = r / np.linalg.norm(r)

    u2 = np.cross(r, f)                # corrected up vector

    m = np.eye(4, dtype=np.float32)
    m[0, :3] = r
    m[1, :3] = u2
    m[2, :3] = -f
    m[0, 3] = -np.dot(r, e)
    m[1, 3] = -np.dot(u2, e)
    m[2, 3] = np.dot(f, e)
    return m


def mat4_rotate_y(angle_deg: float) -> np.ndarray:
    """Return a rotation matrix around the Y axis.

    WEBGL EQUIVALENT: mat4.rotateY(out, a, rad) from glMatrix.

    Args:
        angle_deg: Rotation angle in degrees.

    Returns:
        4×4 float32 rotation matrix.
    """
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    m = np.eye(4, dtype=np.float32)
    m[0, 0] =  c;  m[0, 2] = s
    m[2, 0] = -s;  m[2, 2] = c
    return m


def mat4_rotate_x(angle_deg: float) -> np.ndarray:
    """Return a rotation matrix around the X axis.

    WEBGL EQUIVALENT: mat4.rotateX(out, a, rad) from glMatrix.

    Args:
        angle_deg: Rotation angle in degrees.

    Returns:
        4×4 float32 rotation matrix.
    """
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    m = np.eye(4, dtype=np.float32)
    m[1, 1] =  c;  m[1, 2] = -s
    m[2, 1] =  s;  m[2, 2] =  c
    return m


def mat4_rotate_z(angle_deg: float) -> np.ndarray:
    """Return a rotation matrix around the Z axis.

    WEBGL EQUIVALENT: mat4.rotateZ(out, a, rad) from glMatrix.

    Args:
        angle_deg: Rotation angle in degrees.

    Returns:
        4×4 float32 rotation matrix.
    """
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    m = np.eye(4, dtype=np.float32)
    m[0, 0] =  c;  m[0, 1] = -s
    m[1, 0] =  s;  m[1, 1] =  c
    return m


# ── STEP 3: CubeViewport — the QOpenGLWidget subclass ─────────────────────────

class CubeViewport(QOpenGLWidget):
    """QOpenGLWidget that renders a rotating cube with modern OpenGL 3.3.

    QT CONCEPT — QOpenGLWidget lifecycle:
    Qt calls three methods in order, each with a current OpenGL context:
      1. initializeGL()  — called once when the widget is first shown.
                           The only safe place to call glGen*, glCreate*, etc.
      2. resizeGL(w, h)  — called on creation and whenever the widget resizes.
                           Update glViewport and the projection matrix here.
      3. paintGL()       — called whenever the widget needs a new frame.
                           Build the scene, upload uniforms, draw.

    NEVER call OpenGL functions outside these three methods (e.g. in __init__).
    The OpenGL context is not current there and the calls will silently fail
    or crash.

    WEBGL EQUIVALENT — the three methods map to:
      initializeGL  → the setup code after gl = canvas.getContext('webgl2')
      resizeGL      → a ResizeObserver callback that calls gl.viewport(...)
      paintGL       → the drawScene() function called by requestAnimationFrame
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialise state variables.  No OpenGL calls here — context not yet current.

        Args:
            parent: Optional Qt parent widget.
        """
        super().__init__(parent)
        self.setMinimumSize(400, 400)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Rotation state — angles in degrees, advanced each frame.
        self._angle_x: float = 0.0
        self._angle_y: float = 0.0
        self._angle_z: float = 0.0

        # Rotation speeds in degrees per second — driven by control panel sliders.
        self.speed_x: float = float(DEFAULT_SPEED_X)
        self.speed_y: float = float(DEFAULT_SPEED_Y)
        self.speed_z: float = float(DEFAULT_SPEED_Z)

        # Tint colour sent to the fragment shader.  (1, 1, 1) = no tint.
        self._tint: tuple[float, float, float] = (1.0, 1.0, 1.0)

        # Wireframe mode toggle.
        self._wireframe: bool = False

        # Delta timer — measures real elapsed time between frames so rotation
        # speed is frame-rate independent.
        # ANIMATION CONCEPT — frame-rate independence:
        # Advancing by speed * delta_seconds means the cube rotates the same
        # number of degrees per second regardless of whether paintGL fires at
        # 60 fps or 30 fps.  This is identical to the pattern in seminar 01's
        # requestAnimationFrame loop and Three.js's clock.getDelta().
        self._elapsed_timer: QElapsedTimer = QElapsedTimer()
        self._elapsed_timer.start()
        self._last_ms: int = 0

        # FPS counter — updated each frame and read by the control panel.
        self.fps: float = 0.0
        self._fps_frame_count: int = 0
        self._fps_elapsed_ms: int = 0

        # OpenGL object handles — set in initializeGL.
        self._program: int = 0
        self._vao: int = 0
        self._vbo: int = 0
        self._ebo: int = 0
        self._index_count: int = 0

        # Cached uniform locations — set in initializeGL.
        self._loc_model: int = -1
        self._loc_view: int = -1
        self._loc_projection: int = -1
        self._loc_tint: int = -1

        # Cached projection matrix — rebuilt in resizeGL.
        self._proj_matrix: np.ndarray = mat4_identity()

        # Cached view matrix — constant (camera does not move).
        self._view_matrix: np.ndarray = mat4_look_at(
            CAMERA_EYE, CAMERA_CENTER, CAMERA_UP
        )

    # ── Public interface ───────────────────────────────────────────────────────

    def set_tint(self, r: float, g: float, b: float) -> None:
        """Set the colour tint sent to the fragment shader.

        Args:
            r: Red channel in [0, 1].
            g: Green channel in [0, 1].
            b: Blue channel in [0, 1].
        """
        self._tint = (r, g, b)
        self.update()

    def set_wireframe(self, enabled: bool) -> None:
        """Toggle wireframe rendering mode.

        Args:
            enabled: True for wireframe, False for solid fill.
        """
        self._wireframe = enabled
        self.update()

    # ── QOpenGLWidget overrides ────────────────────────────────────────────────

    def initializeGL(self) -> None:  # noqa: N802
        """Set up the OpenGL pipeline.  Called once, context is current here.

        QT CONCEPT — initializeGL:
        This is guaranteed to be called with a current OpenGL context.
        It is the only safe place to call glGenBuffers, glGenVertexArrays,
        compile shaders, etc.  Any GL call before this point will crash
        because there is no context to receive it.
        """
        # ── Global GL state ────────────────────────────────────────────────────
        GL.glClearColor(CLEAR_COLOR_R, CLEAR_COLOR_G, CLEAR_COLOR_B, CLEAR_COLOR_A)
        # WEBGL EQUIVALENT: gl.clearColor(r, g, b, a)

        GL.glEnable(GL.GL_DEPTH_TEST)
        # WEBGL EQUIVALENT: gl.enable(gl.DEPTH_TEST)
        # Depth test discards fragments that are behind already-drawn geometry.
        # Without it, the back faces of the cube would sometimes draw over the front.

        GL.glEnable(GL.GL_MULTISAMPLE)
        # WEBGL EQUIVALENT: enabled by default when antialias: true in getContext().
        # Activates the 4× MSAA we requested in QSurfaceFormat.

        # ── Compile and link shaders ───────────────────────────────────────────
        # WEBGL EQUIVALENT:
        #   const vs = gl.createShader(gl.VERTEX_SHADER);
        #   gl.shaderSource(vs, vsSource); gl.compileShader(vs);
        #   const prog = gl.createProgram();
        #   gl.attachShader(prog, vs); gl.linkProgram(prog);
        #
        # PyOpenGL's compileProgram() does all of the above and raises an
        # exception with the GLSL error log if compilation fails — essential
        # for debugging.  We also check manually below for clarity.
        try:
            vs = gl_shaders.compileShader(VERTEX_SHADER_SRC, GL.GL_VERTEX_SHADER)
            fs = gl_shaders.compileShader(FRAGMENT_SHADER_SRC, GL.GL_FRAGMENT_SHADER)
            self._program = gl_shaders.compileProgram(vs, fs)
        except Exception as exc:  # noqa: BLE001
            print(f"[CubeViewport] Shader compilation failed:\n{exc}")
            return

        # Manual compile-status check (belt-and-suspenders for teaching).
        for shader_obj, label in ((vs, "vertex"), (fs, "fragment")):
            status = GL.glGetShaderiv(shader_obj, GL.GL_COMPILE_STATUS)
            if not status:
                log = GL.glGetShaderInfoLog(shader_obj).decode()
                print(f"[CubeViewport] {label} shader error:\n{log}")

        link_status = GL.glGetProgramiv(self._program, GL.GL_LINK_STATUS)
        if not link_status:
            log = GL.glGetProgramInfoLog(self._program).decode()
            print(f"[CubeViewport] Program link error:\n{log}")

        # ── Cache uniform locations ────────────────────────────────────────────
        # WEBGL EQUIVALENT: gl.getUniformLocation(prog, "uModel") — identical.
        self._loc_model      = GL.glGetUniformLocation(self._program, "uModel")
        self._loc_view       = GL.glGetUniformLocation(self._program, "uView")
        self._loc_projection = GL.glGetUniformLocation(self._program, "uProjection")
        self._loc_tint       = GL.glGetUniformLocation(self._program, "uTint")

        # ── Cube geometry ──────────────────────────────────────────────────────
        # A cube has 8 unique corners.  We store position (xyz) and colour (rgb)
        # for each corner, interleaved in one array — the same interleaved layout
        # used in seminar 01's WebGL demos.
        #
        # Each face has a different hue so students can see all 6 faces rotate.
        #
        # Layout per vertex: [x, y, z, r, g, b] — 6 floats, 24 bytes.
        #
        # WEBGL EQUIVALENT: the same Float32Array passed to gl.bufferData().

        vertices = np.array([
            # x      y      z      r      g      b
            # Front face corners (z = +0.5)
            -0.5, -0.5,  0.5,   0.31,  0.61,  0.98,   # 0 bottom-left  (accent blue)
             0.5, -0.5,  0.5,   0.31,  0.61,  0.98,   # 1 bottom-right
             0.5,  0.5,  0.5,   0.31,  0.61,  0.98,   # 2 top-right
            -0.5,  0.5,  0.5,   0.31,  0.61,  0.98,   # 3 top-left

            # Back face corners (z = -0.5)
            -0.5, -0.5, -0.5,   0.65,  0.55,  0.98,   # 4  (accent purple)
             0.5, -0.5, -0.5,   0.65,  0.55,  0.98,   # 5
             0.5,  0.5, -0.5,   0.65,  0.55,  0.98,   # 6
            -0.5,  0.5, -0.5,   0.65,  0.55,  0.98,   # 7

            # Top face corners (y = +0.5)  — re-use positions with new colours
            -0.5,  0.5,  0.5,   0.20,  0.83,  0.60,   # 8  (green)
             0.5,  0.5,  0.5,   0.20,  0.83,  0.60,   # 9
             0.5,  0.5, -0.5,   0.20,  0.83,  0.60,   # 10
            -0.5,  0.5, -0.5,   0.20,  0.83,  0.60,   # 11

            # Bottom face (y = -0.5)
            -0.5, -0.5,  0.5,   0.98,  0.57,  0.24,   # 12 (orange)
             0.5, -0.5,  0.5,   0.98,  0.57,  0.24,   # 13
             0.5, -0.5, -0.5,   0.98,  0.57,  0.24,   # 14
            -0.5, -0.5, -0.5,   0.98,  0.57,  0.24,   # 15

            # Right face (x = +0.5)
             0.5, -0.5,  0.5,   0.96,  0.38,  0.38,   # 16 (red)
             0.5, -0.5, -0.5,   0.96,  0.38,  0.38,   # 17
             0.5,  0.5, -0.5,   0.96,  0.38,  0.38,   # 18
             0.5,  0.5,  0.5,   0.96,  0.38,  0.38,   # 19

            # Left face (x = -0.5)
            -0.5, -0.5,  0.5,   0.35,  0.76,  0.92,   # 20 (cyan)
            -0.5, -0.5, -0.5,   0.35,  0.76,  0.92,   # 21
            -0.5,  0.5, -0.5,   0.35,  0.76,  0.92,   # 22
            -0.5,  0.5,  0.5,   0.35,  0.76,  0.92,   # 23
        ], dtype=np.float32)

        # Index buffer: two triangles per face × 6 faces = 12 triangles = 36 indices.
        # WEBGL EQUIVALENT: new Uint32Array([...]) passed to gl.bufferData for the EBO.
        indices = np.array([
             0,  1,  2,   2,  3,  0,   # front
             4,  6,  5,   6,  4,  7,   # back  (winding reversed so normal faces out)
             8,  9, 10,  10, 11,  8,   # top
            12, 14, 13,  14, 12, 15,   # bottom
            16, 17, 18,  18, 19, 16,   # right
            20, 22, 21,  22, 20, 23,   # left
        ], dtype=np.uint32)
        self._index_count = len(indices)

        # ── VAO — Vertex Array Object ──────────────────────────────────────────
        # A VAO records the vertex attribute configuration so we don't have to
        # re-specify it every draw call.  Bind the VAO, configure attributes,
        # then just bind the VAO again at draw time.
        # WEBGL EQUIVALENT: this._vao = gl.createVertexArray(); gl.bindVertexArray(vao);
        self._vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self._vao)
        # WEBGL EQUIVALENT: gl.bindVertexArray(this._vao)

        # ── VBO — Vertex Buffer Object ─────────────────────────────────────────
        # Upload vertex data (positions + colours) to the GPU.
        # WEBGL EQUIVALENT: this._vbo = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
        self._vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo)
        # WEBGL EQUIVALENT: gl.bindBuffer(gl.ARRAY_BUFFER, this._vbo)

        GL.glBufferData(GL.GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL.GL_STATIC_DRAW)
        # WEBGL EQUIVALENT: gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW)
        # GL_STATIC_DRAW — hint to the driver that this data will not change.

        # ── EBO — Element Buffer Object (index buffer) ─────────────────────────
        # WEBGL EQUIVALENT: gl.createBuffer(); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ebo);
        self._ebo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self._ebo)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL.GL_STATIC_DRAW)
        # WEBGL EQUIVALENT: gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW)

        # ── Vertex attribute pointers ──────────────────────────────────────────
        # Tell OpenGL how to interpret the raw bytes in the VBO.
        # Each vertex is 6 floats × 4 bytes = 24 bytes (the "stride").
        #
        # Attribute 0: position — 3 floats starting at byte 0.
        # Attribute 1: colour   — 3 floats starting at byte 12 (3 floats × 4 bytes).
        #
        # WEBGL EQUIVALENT:
        #   gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 24, 0);
        #   gl.enableVertexAttribArray(0);
        #   gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 24, 12);
        #   gl.enableVertexAttribArray(1);

        stride = 6 * 4  # 6 components × 4 bytes per float

        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(0)

        GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(12))
        GL.glEnableVertexAttribArray(1)

        # Unbind the VAO — good practice to avoid accidental modification.
        GL.glBindVertexArray(0)

        # Reset the elapsed timer so delta starts from zero on the first frame.
        self._last_ms = self._elapsed_timer.elapsed()

    def resizeGL(self, w: int, h: int) -> None:  # noqa: N802
        """Handle viewport resize.  Called on creation and on every resize event.

        QT CONCEPT — resizeGL:
        Qt calls this whenever the widget changes size.  Always update the
        OpenGL viewport here, and rebuild any matrix that depends on aspect ratio.

        WEBGL EQUIVALENT:
            gl.viewport(0, 0, canvas.width, canvas.height);
            projMatrix = mat4.perspective(out, fovY, w / h, near, far);

        Args:
            w: New width in pixels.
            h: New height in pixels.
        """
        GL.glViewport(0, 0, w, h)
        # WEBGL EQUIVALENT: gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight)

        aspect = w / max(h, 1)
        self._proj_matrix = mat4_perspective(FOV_DEG, aspect, Z_NEAR, Z_FAR)

    def paintGL(self) -> None:  # noqa: N802
        """Render one frame.  Called by Qt whenever the widget needs repainting.

        QT CONCEPT — paintGL vs paintEvent:
        QOpenGLWidget uses paintGL (not paintEvent) for GPU rendering.
        Call self.update() to schedule the next paintGL; never call paintGL
        directly — Qt manages context binding around it.

        WEBGL EQUIVALENT: this is the drawScene() / render() function called
        inside requestAnimationFrame.
        """
        # ── Delta time ────────────────────────────────────────────────────────
        # Measure real elapsed time since the last frame.
        # ANIMATION CONCEPT — frame-rate-independent motion:
        # Multiplying by delta_sec means the cube rotates the same number of
        # degrees per second regardless of frame rate.  Compare with Three.js:
        #   const delta = clock.getDelta();
        #   mesh.rotation.y += speed * delta;
        now_ms = self._elapsed_timer.elapsed()
        delta_ms = now_ms - self._last_ms
        self._last_ms = now_ms
        delta_sec = delta_ms / 1000.0

        # ── Advance rotation angles ────────────────────────────────────────────
        self._angle_x = (self._angle_x + self.speed_x * delta_sec) % 360.0
        self._angle_y = (self._angle_y + self.speed_y * delta_sec) % 360.0
        self._angle_z = (self._angle_z + self.speed_z * delta_sec) % 360.0

        # ── FPS counter ────────────────────────────────────────────────────────
        self._fps_frame_count += 1
        self._fps_elapsed_ms += delta_ms
        if self._fps_elapsed_ms >= 500:   # update FPS display every 0.5 s
            self.fps = self._fps_frame_count / (self._fps_elapsed_ms / 1000.0)
            self._fps_frame_count = 0
            self._fps_elapsed_ms = 0

        # ── Clear ──────────────────────────────────────────────────────────────
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        # WEBGL EQUIVALENT: gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)

        # ── Wireframe toggle ───────────────────────────────────────────────────
        if self._wireframe:
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
        else:
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        # WEBGL EQUIVALENT: there is no direct WebGL equivalent for wireframe;
        # in WebGL you would draw with gl.LINES or use a geometry shader.
        # This is one area where desktop OpenGL is more convenient.

        # ── Bind shader program ────────────────────────────────────────────────
        GL.glUseProgram(self._program)
        # WEBGL EQUIVALENT: gl.useProgram(this._program)

        # ── Build model matrix ─────────────────────────────────────────────────
        # Combine X, Y, Z rotations: Rz × Rx × Ry (right-to-left application).
        # WEBGL EQUIVALENT:
        #   mat4.rotateY(model, identity, angleY);
        #   mat4.rotateX(model, model, angleX);
        model = mat4_rotate_y(self._angle_y) @ mat4_rotate_x(self._angle_x) @ mat4_rotate_z(self._angle_z)

        # ── Upload uniforms ────────────────────────────────────────────────────
        # GL_TRUE = transpose = True: NumPy is row-major, GLSL expects column-major.
        # WEBGL EQUIVALENT: gl.uniformMatrix4fv(loc, false, matrix)
        #   (WebGL matrices from glMatrix are already column-major, so transpose=false.)
        GL.glUniformMatrix4fv(self._loc_model,      1, GL.GL_TRUE, model)
        GL.glUniformMatrix4fv(self._loc_view,       1, GL.GL_TRUE, self._view_matrix)
        GL.glUniformMatrix4fv(self._loc_projection, 1, GL.GL_TRUE, self._proj_matrix)

        # Upload tint colour.
        # WEBGL EQUIVALENT: gl.uniform3fv(loc, [r, g, b])
        GL.glUniform3f(self._loc_tint, *self._tint)

        # ── Draw ───────────────────────────────────────────────────────────────
        GL.glBindVertexArray(self._vao)
        # WEBGL EQUIVALENT: gl.bindVertexArray(this._vao)

        GL.glDrawElements(GL.GL_TRIANGLES, self._index_count, GL.GL_UNSIGNED_INT, None)
        # WEBGL EQUIVALENT: gl.drawElements(gl.TRIANGLES, indexCount, gl.UNSIGNED_INT, 0)

        GL.glBindVertexArray(0)
        GL.glUseProgram(0)

        # ── Reset polygon mode ─────────────────────────────────────────────────
        # Always reset after wireframe draw so other widgets (e.g. Qt's own
        # painter overlay) are not affected.
        if self._wireframe:
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)


# ── STEP 5: ControlPanel ──────────────────────────────────────────────────────

class ControlPanel(QWidget):
    """Right-side control panel with sliders, colour picker, wireframe toggle.

    Communicates with CubeViewport by directly updating its public attributes
    and calling update().  In demo 5 (the full tool), every inter-panel
    communication goes through Qt signals instead — this simpler approach is
    used here to keep the focus on the OpenGL side.
    """

    def __init__(self, viewport: CubeViewport, parent: QWidget | None = None) -> None:
        """Build the panel and connect controls to the viewport.

        Args:
            viewport: The CubeViewport instance to control.
            parent:   Optional Qt parent widget.
        """
        super().__init__(parent)
        self._viewport = viewport
        self.setFixedWidth(CONTROL_PANEL_WIDTH)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 16, 12, 16)
        layout.setSpacing(8)

        # Stylesheet — matches the dark design system from demo 2.
        self.setStyleSheet(
            "QWidget { background: #161a22; color: #d8dce6;"
            "          font-family: 'IBM Plex Mono'; font-size: 12px; }"
            "QLabel  { color: #5a6175; }"
            "QSlider::groove:horizontal { height: 4px; background: #252b38;"
            "                             border-radius: 2px; }"
            "QSlider::handle:horizontal { width: 12px; height: 12px;"
            "                             background: #4f9cf9; border-radius: 6px;"
            "                             margin: -4px 0; }"
            "QSlider::sub-page:horizontal { background: #4f9cf9; border-radius: 2px; }"
            "QPushButton { background: #1e2330; border: 1px solid #252b38;"
            "              border-radius: 4px; padding: 6px 10px;"
            "              color: #d8dce6; }"
            "QPushButton:hover { background: #252b38; }"
            "QCheckBox { color: #d8dce6; spacing: 8px; }"
            "QCheckBox::indicator { width: 14px; height: 14px;"
            "                       border: 1px solid #252b38; border-radius: 3px;"
            "                       background: #1e2330; }"
            "QCheckBox::indicator:checked { background: #4f9cf9; }"
        )

        # ── AMS logo ───────────────────────────────────────────────────────────
        logo_label = QLabel()
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if LOGO_PATH.exists():
            pixmap = QPixmap(str(LOGO_PATH)).scaledToWidth(
                80,
                Qt.TransformationMode.SmoothTransformation,
            )
            logo_label.setPixmap(pixmap)
        else:
            logo_label.setText("AMS Group – TU Ilmenau")
            logo_label.setStyleSheet("color: #4f9cf9; font-size: 10px;")
        layout.addWidget(logo_label)

        layout.addSpacing(4)

        # ── Author credit ──────────────────────────────────────────────────────
        author_label = QLabel("Renato Fera P.")
        author_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        author_label.setStyleSheet("color: #5a6175; font-size: 10px; letter-spacing: 0.5px;")
        layout.addWidget(author_label)

        layout.addSpacing(12)

        # ── Section: Rotation speed ────────────────────────────────────────────
        layout.addWidget(self._section_label("Rotation Speed"))

        self._slider_x, self._label_x = self._make_slider(
            "X axis", DEFAULT_SPEED_X, 0, MAX_SPEED
        )
        self._slider_y, self._label_y = self._make_slider(
            "Y axis", DEFAULT_SPEED_Y, 0, MAX_SPEED
        )
        self._slider_z, self._label_z = self._make_slider(
            "Z axis", DEFAULT_SPEED_Z, 0, MAX_SPEED
        )

        for (slider, label), axis_label in (
            ((self._slider_x, self._label_x), "X"),
            ((self._slider_y, self._label_y), "Y"),
            ((self._slider_z, self._label_z), "Z"),
        ):
            row = QHBoxLayout()
            row.addWidget(QLabel(f"{axis_label}:"))
            row.addWidget(slider)
            row.addWidget(label)
            layout.addLayout(row)

        self._slider_x.valueChanged.connect(self._on_speed_x)
        self._slider_y.valueChanged.connect(self._on_speed_y)
        self._slider_z.valueChanged.connect(self._on_speed_z)

        layout.addSpacing(12)

        # ── Section: Appearance ────────────────────────────────────────────────
        layout.addWidget(self._section_label("Appearance"))

        self._colour_btn = QPushButton("Base Colour")
        self._colour_btn.clicked.connect(self._pick_colour)
        layout.addWidget(self._colour_btn)

        self._wireframe_cb = QCheckBox("Wireframe")
        self._wireframe_cb.setChecked(False)
        self._wireframe_cb.stateChanged.connect(self._on_wireframe)
        layout.addWidget(self._wireframe_cb)

        layout.addStretch()

        # ── FPS display ────────────────────────────────────────────────────────
        layout.addWidget(self._section_label("Performance"))
        self._fps_label = QLabel("-- FPS")
        self._fps_label.setStyleSheet("color: #34d399; font-size: 14px;")
        layout.addWidget(self._fps_label)

        # Update FPS label every 500 ms via a dedicated timer.
        fps_timer = QTimer(self)
        fps_timer.setInterval(500)
        fps_timer.timeout.connect(self._update_fps)
        fps_timer.start()

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _section_label(self, text: str) -> QLabel:
        """Return a styled section heading label.

        Args:
            text: The heading text.
        """
        lbl = QLabel(text.upper())
        lbl.setStyleSheet(
            "color: #5a6175; font-size: 10px; letter-spacing: 1px;"
            "padding-top: 4px;"
        )
        return lbl

    def _make_slider(
        self, name: str, default: int, minimum: int, maximum: int
    ) -> tuple[QSlider, QLabel]:
        """Create a horizontal slider and its value label.

        Args:
            name:    Accessibility name for the slider.
            default: Initial value.
            minimum: Minimum value.
            maximum: Maximum value.

        Returns:
            Tuple of (slider, value_label).
        """
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(minimum, maximum)
        slider.setValue(default)
        slider.setAccessibleName(name)
        label = QLabel(f"{default:3d}°/s")
        label.setFixedWidth(46)
        return slider, label

    # ── Slots ──────────────────────────────────────────────────────────────────

    def _on_speed_x(self, value: int) -> None:
        """Update X rotation speed on the viewport.

        Args:
            value: New speed in degrees/second.
        """
        self._viewport.speed_x = float(value)
        self._label_x.setText(f"{value:3d}°/s")

    def _on_speed_y(self, value: int) -> None:
        """Update Y rotation speed on the viewport.

        Args:
            value: New speed in degrees/second.
        """
        self._viewport.speed_y = float(value)
        self._label_y.setText(f"{value:3d}°/s")

    def _on_speed_z(self, value: int) -> None:
        """Update Z rotation speed on the viewport.

        Args:
            value: New speed in degrees/second.
        """
        self._viewport.speed_z = float(value)
        self._label_z.setText(f"{value:3d}°/s")

    def _pick_colour(self) -> None:
        """Open a QColorDialog and apply the chosen colour as a shader tint."""
        colour = QColorDialog.getColor(
            QColor(255, 255, 255), self, "Choose Base Colour"
        )
        if colour.isValid():
            self._viewport.set_tint(
                colour.redF(), colour.greenF(), colour.blueF()
            )

    def _on_wireframe(self, state: int) -> None:
        """Toggle wireframe mode.

        Args:
            state: Qt.CheckState integer (2 = checked, 0 = unchecked).
        """
        self._viewport.set_wireframe(bool(state))

    def _update_fps(self) -> None:
        """Read the viewport's fps counter and update the label."""
        self._fps_label.setText(f"{self._viewport.fps:.0f} FPS")


# ── STEP 6: MainWindow ────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    """Application main window.

    Layout (horizontal):
    ┌─────────────────────────────────┬─────────────┐
    │         CubeViewport            │ ControlPanel│
    │         (OpenGL, stretch)       │ (fixed 220) │
    └─────────────────────────────────┴─────────────┘

    The QTimer that drives the render loop lives here, not inside
    CubeViewport.  This keeps CubeViewport focused on rendering — it
    does not know about timing.  The timer calls viewport.update(),
    which schedules a paintGL call on the next event loop iteration.

    QT CONCEPT — separating the timer from the renderer:
    In a larger application, the same timer could drive multiple viewports
    or be replaced by a vsync signal.  Keeping the timer in the window
    (not embedded in the GL widget) makes this swap trivial.
    """

    def __init__(self) -> None:
        """Initialise the window, create panels, start the render timer."""
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)
        self.setStyleSheet("QMainWindow { background: #0d0f14; }")

        # ── Panels ────────────────────────────────────────────────────────────
        self._viewport = CubeViewport()
        self._panel    = ControlPanel(self._viewport)

        # ── Layout ────────────────────────────────────────────────────────────
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        row.addWidget(self._viewport, stretch=1)
        row.addWidget(self._panel)
        self.setCentralWidget(container)

        # ── STEP 7: QTimer — drives the render loop ────────────────────────────
        # QT CONCEPT — QTimer as render loop:
        # QTimer fires timeout every TIMER_INTERVAL_MS milliseconds in the main
        # event loop.  We call viewport.update() which schedules a paintGL call.
        # This is safe: Qt processes the paint in the same thread, context current.
        #
        # WEBGL EQUIVALENT:
        #   function loop() { drawScene(); requestAnimationFrame(loop); }
        #   requestAnimationFrame(loop);
        #
        # The difference: requestAnimationFrame is vsync-aware and fires at the
        # display refresh rate.  QTimer fires at a fixed interval.  For a proper
        # vsync loop in Qt, use QWindow with requestUpdate() — but QTimer at
        # 16 ms (≈60 fps) is fine for a teaching demo.
        self._render_timer = QTimer(self)
        self._render_timer.setInterval(TIMER_INTERVAL_MS)
        self._render_timer.timeout.connect(self._viewport.update)
        self._render_timer.start()

        # ── Keyboard shortcuts ─────────────────────────────────────────────────
        # W: toggle wireframe (mirrors the Maya / Blender shortcut)
        QShortcut(QKeySequence("W"), self, activated=self._toggle_wireframe)
        # R: reset all rotation angles to zero
        QShortcut(QKeySequence("R"), self, activated=self._reset_rotation)

    def _toggle_wireframe(self) -> None:
        """Keyboard shortcut: toggle the wireframe checkbox."""
        cb = self._panel._wireframe_cb  # noqa: SLF001
        cb.setChecked(not cb.isChecked())

    def _reset_rotation(self) -> None:
        """Keyboard shortcut: reset rotation angles to zero."""
        self._viewport._angle_x = 0.0  # noqa: SLF001
        self._viewport._angle_y = 0.0  # noqa: SLF001
        self._viewport._angle_z = 0.0  # noqa: SLF001


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    """Create the QApplication, build the window, and start the event loop.

    QT CONCEPT — QApplication:
    QApplication must be constructed before any QWidget.  It initialises
    the Qt platform plugin (xcb on Linux, win32 on Windows) and sets up the
    event loop.  sys.exit(app.exec()) starts the loop and ensures the process
    exits cleanly with the event loop's return code when the window is closed.

    NOTE: _configure_surface_format() is called at module import time (top of
    file), before QApplication is created — that ordering is mandatory.
    """
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

# =============================================================================
# End of 03_ca_opengl_widget.py
# Author   : Renato Fera P.
# LinkedIn : https://www.linkedin.com/in/renato-profeta/
# Course   : Computeranimation — Chapter 03: PySide6 & Python Animation
# Institute: Institut für Medientechnik, Technische Universität Ilmenau
# =============================================================================
