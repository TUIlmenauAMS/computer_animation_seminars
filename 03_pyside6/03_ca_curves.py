# =============================================================================
# 03_ca_curves.py — Curve Editor with QPainter
# =============================================================================
# Author   : Renato Fera P.
# LinkedIn : https://www.linkedin.com/in/renato-profeta/
# Course   : Computeranimation — Chapter 03: PySide6 & Python Animation
# Institute: Institut für Medientechnik, Technische Universität Ilmenau
# -----------------------------------------------------------------------------
# Description:
#   A live animation curve editor built entirely with QPainter and NumPy.
#   Three interpolation modes are selectable via a QComboBox:
#     - Linear
#     - Ease In-Out (cosine)
#     - Cubic Bézier (draggable handles)
#
#   In Bézier mode the two control points P1 and P2 are draggable directly
#   on the canvas. A QTimer animates a moving dot along the curve and drives
#   a second preview panel showing a ball moving left-to-right with the
#   selected easing applied.
#
# How to run:
#   uv sync                            # first time only
#   uv run python 03_ca_curves.py
#
# Ubuntu system dependencies (once):
#   sudo apt install libgl1 libegl1 libxcb-cursor0 libxcb-xinerama0
# =============================================================================

from __future__ import annotations

import sys
import math
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer, QPoint, Signal
from PySide6.QtGui import (
    QPainter,
    QPainterPath,
    QPen,
    QBrush,
    QColor,
    QFont,
    QCursor,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QSizePolicy,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

# ── Constants ─────────────────────────────────────────────────────────────────

WINDOW_TITLE: str = "Curve Editor — Computeranimation Seminar 03"
WINDOW_MIN_WIDTH: int = 820
WINDOW_MIN_HEIGHT: int = 500

# Animation
TARGET_FPS: int = 60
TIMER_INTERVAL_MS: int = 1000 // TARGET_FPS   # 16 ms ≈ 60 fps
ANIMATION_DURATION_S: float = 2.0             # seconds for one full t: 0→1→0 cycle

# Curve canvas colours
CANVAS_BG: str = "#0d0f14"
GRID_COLOR: str = "#1a1f2b"
AXIS_COLOR: str = "#252b38"
CURVE_COLOR: str = "#4f9cf9"
HANDLE_COLOR: str = "#a78bfa"
HANDLE_LINE_COLOR: str = "#3d3660"
DOT_COLOR: str = "#fb923c"
KEYFRAME_COLOR: str = "#34d399"

# Preview canvas colours
PREVIEW_BG: str = "#0d0f14"
BALL_COLOR: str = "#4f9cf9"
TRACK_COLOR: str = "#1a1f2b"

# Interaction
HANDLE_RADIUS: int = 7        # pixels — hit area for drag detection
CURVE_SAMPLES: int = 200      # number of points used to draw the curve

# Default Bézier control points (CSS ease-in-out equivalent)
DEFAULT_P1: tuple[float, float] = (0.42, 0.0)
DEFAULT_P2: tuple[float, float] = (0.58, 1.0)

# Interpolation mode names (must match QComboBox order)
MODE_LINEAR: int = 0
MODE_EASE_INOUT: int = 1
MODE_BEZIER: int = 2
MODE_NAMES: list[str] = ["Linear", "Ease In-Out (cosine)", "Cubic Bézier"]

# AMS logo path — one level up from the script's directory (repo root)
LOGO_PATH: Path = Path(__file__).resolve().parent.parent / "ams_group_tui.png"


# ── STEP 1: Easing functions (pure NumPy) ─────────────────────────────────────

def ease_linear(t: np.ndarray) -> np.ndarray:
    """Linear interpolation — no easing.

    Args:
        t: Array of time values in [0, 1].

    Returns:
        The same values unchanged.
    """
    return t.copy()


def ease_in_out_cosine(t: np.ndarray) -> np.ndarray:
    """Smooth ease-in-out using a cosine remap.

    Maps t ∈ [0,1] to a smooth S-curve: slow at both ends, fast in the middle.
    This is equivalent to CSS ease-in-out but computed analytically rather than
    approximated by a Bézier.

    ANIMATION CONCEPT:
    The formula (1 - cos(π·t)) / 2 maps the linear ramp to a cosine half-period.
    At t=0: cos(0)=1, result=0.  At t=0.5: cos(π/2)=0, result=0.5.
    At t=1: cos(π)=-1, result=1.  The derivative (speed) is zero at both ends —
    the motion eases in and out smoothly.

    Args:
        t: Array of time values in [0, 1].

    Returns:
        Eased values in [0, 1].
    """
    return (1.0 - np.cos(np.pi * t)) / 2.0


def bezier_x(u: np.ndarray, p1x: float, p2x: float) -> np.ndarray:
    """Evaluate the X component of a CSS-style cubic Bézier at parameter u.

    Args:
        u: Parameter values in [0, 1].
        p1x: X coordinate of control point 1.
        p2x: X coordinate of control point 2.

    Returns:
        X values of the Bézier curve at each u.
    """
    v = 1.0 - u
    return 3.0 * v * v * u * p1x + 3.0 * v * u * u * p2x + u * u * u


def bezier_y(u: np.ndarray, p1y: float, p2y: float) -> np.ndarray:
    """Evaluate the Y component of a CSS-style cubic Bézier at parameter u.

    Args:
        u: Parameter values in [0, 1].
        p1y: Y coordinate of control point 1.
        p2y: Y coordinate of control point 2.

    Returns:
        Y values of the Bézier curve at each u.
    """
    v = 1.0 - u
    return 3.0 * v * v * u * p1y + 3.0 * v * u * u * p2y + u * u * u


def ease_cubic_bezier(
    t: np.ndarray,
    p1x: float,
    p1y: float,
    p2x: float,
    p2y: float,
    iterations: int = 8,
) -> np.ndarray:
    """Evaluate the CSS cubic-bezier() easing function.

    ANIMATION CONCEPT — why this is non-trivial:
    The parametric Bézier form gives position as a function of curve
    parameter u, not time t.  To find the eased value at a given t,
    we must invert the X component: find the u where bezier_x(u) = t.
    This is the same inversion that browsers perform internally for
    every CSS transition-timing-function: cubic-bezier().

    We use a vectorised binary search over all t values simultaneously
    with NumPy, which is efficient even for large arrays.

    Args:
        t: Time values in [0, 1].
        p1x, p1y: First control point coordinates.
        p2x, p2y: Second control point coordinates.
        iterations: Binary search iterations (8 gives sub-pixel accuracy).

    Returns:
        Eased Y values in [0, 1] corresponding to each input t.
    """
    lo = np.zeros_like(t)
    hi = np.ones_like(t)

    for _ in range(iterations):
        mid = (lo + hi) / 2.0
        x_mid = bezier_x(mid, p1x, p2x)
        lo = np.where(x_mid < t, mid, lo)
        hi = np.where(x_mid >= t, mid, hi)

    u = (lo + hi) / 2.0
    return bezier_y(u, p1y, p2y)


# ── STEP 2: Curve canvas ───────────────────────────────────────────────────────

class CurveCanvas(QWidget):
    """A QPainter widget that draws an animation curve with draggable handles.

    Displays the selected easing function as a graph (X axis = time t,
    Y axis = eased value).  In Bézier mode, the two control point handles
    are draggable with the mouse.

    Signals:
        easing_changed: Emitted when Bézier handles are dragged.
                        Carries (p1x, p1y, p2x, p2y) as four floats.
    """

    easing_changed: Signal = Signal(float, float, float, float)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialise the curve canvas with default Bézier control points."""
        super().__init__(parent)
        self.setMinimumSize(300, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setCursor(QCursor(Qt.CursorShape.CrossCursor))

        self._mode: int = MODE_BEZIER
        self._p1x, self._p1y = DEFAULT_P1
        self._p2x, self._p2y = DEFAULT_P2
        self._t: float = 0.0
        self._dragging: int | None = None   # 1 = dragging P1, 2 = dragging P2

    # ── Public interface ───────────────────────────────────────────────────────

    def set_mode(self, mode: int) -> None:
        """Switch the interpolation mode and repaint.

        Args:
            mode: One of MODE_LINEAR, MODE_EASE_INOUT, MODE_BEZIER.
        """
        self._mode = mode
        self.update()

    def set_t(self, t: float) -> None:
        """Update the animated playhead position and repaint.

        Args:
            t: Current normalised time in [0, 1].
        """
        self._t = t
        self.update()

    def get_eased_value(self, t: float) -> float:
        """Return the eased value for a scalar t using the current mode.

        Args:
            t: Normalised time in [0, 1].

        Returns:
            Eased value in [0, 1].
        """
        t_arr = np.array([t], dtype=np.float64)
        if self._mode == MODE_LINEAR:
            return float(ease_linear(t_arr)[0])
        elif self._mode == MODE_EASE_INOUT:
            return float(ease_in_out_cosine(t_arr)[0])
        else:
            return float(ease_cubic_bezier(t_arr, self._p1x, self._p1y, self._p2x, self._p2y)[0])

    # ── Coordinate helpers ─────────────────────────────────────────────────────

    def _to_canvas(self, nx: float, ny: float) -> tuple[int, int]:
        """Convert normalised [0,1]×[0,1] coordinates to canvas pixels.

        Args:
            nx: Normalised X (time) in [0, 1].
            ny: Normalised Y (value) in [0, 1].

        Returns:
            (px, py) pixel coordinates.
        """
        pad = 32
        w, h = self.width() - pad * 2, self.height() - pad * 2
        px = int(pad + nx * w)
        py = int(pad + (1.0 - ny) * h)
        return px, py

    def _from_canvas(self, px: int, py: int) -> tuple[float, float]:
        """Convert canvas pixels to normalised coordinates.

        Args:
            px, py: Pixel coordinates.

        Returns:
            (nx, ny) normalised coordinates, clamped to [0,1]×[0,1].
        """
        pad = 32
        w, h = self.width() - pad * 2, self.height() - pad * 2
        nx = (px - pad) / max(w, 1)
        ny = 1.0 - (py - pad) / max(h, 1)
        return max(0.0, min(1.0, nx)), max(0.0, min(1.0, ny))

    def _handle_pos(self, handle: int) -> tuple[int, int]:
        """Return pixel position of handle 1 or 2.

        Args:
            handle: 1 for P1, 2 for P2.

        Returns:
            (px, py) pixel coordinates of the handle centre.
        """
        if handle == 1:
            return self._to_canvas(self._p1x, self._p1y)
        return self._to_canvas(self._p2x, self._p2y)

    def _hit_handle(self, pos: QPoint) -> int | None:
        """Return which handle (1 or 2) the mouse is over, or None.

        Args:
            pos: Mouse position in widget pixels.

        Returns:
            1, 2, or None.
        """
        for h in (1, 2):
            hx, hy = self._handle_pos(h)
            if math.hypot(pos.x() - hx, pos.y() - hy) <= HANDLE_RADIUS + 4:
                return h
        return None

    # ── Mouse interaction ──────────────────────────────────────────────────────

    def mousePressEvent(self, event) -> None:  # noqa: ANN001
        """Start dragging a handle if the click lands on one."""
        if self._mode != MODE_BEZIER:
            return
        self._dragging = self._hit_handle(event.pos())

    def mouseMoveEvent(self, event) -> None:  # noqa: ANN001
        """Update the dragged handle position and emit easing_changed."""
        if self._dragging is None or self._mode != MODE_BEZIER:
            if self._hit_handle(event.pos()):
                self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
            else:
                self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
            return

        self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
        nx, ny = self._from_canvas(event.pos().x(), event.pos().y())

        if self._dragging == 1:
            self._p1x, self._p1y = nx, ny
        else:
            self._p2x, self._p2y = nx, ny

        self.easing_changed.emit(self._p1x, self._p1y, self._p2x, self._p2y)
        self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: ANN001
        """End the drag operation."""
        self._dragging = None
        self.setCursor(QCursor(Qt.CursorShape.CrossCursor))

    # ── QPainter rendering ─────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:  # noqa: ANN001
        """Draw the graph: grid, axes, curve, handles, and the animated dot."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        pad = 32

        # Background
        painter.fillRect(0, 0, w, h, QColor(CANVAS_BG))

        # Grid
        grid_pen = QPen(QColor(GRID_COLOR), 1)
        painter.setPen(grid_pen)
        for i in range(1, 4):
            frac = i / 4.0
            gx1, gy1 = self._to_canvas(frac, 0.0)
            gx2, gy2 = self._to_canvas(frac, 1.0)
            painter.drawLine(gx1, gy1, gx2, gy2)
            gx1, gy1 = self._to_canvas(0.0, frac)
            gx2, gy2 = self._to_canvas(1.0, frac)
            painter.drawLine(gx1, gy1, gx2, gy2)

        # Axes
        axis_pen = QPen(QColor(AXIS_COLOR), 1)
        painter.setPen(axis_pen)
        ox0, oy0 = self._to_canvas(0.0, 0.0)
        ox1, oy1 = self._to_canvas(1.0, 0.0)
        painter.drawLine(ox0, oy0, ox1, oy1)
        ox0, oy0 = self._to_canvas(0.0, 0.0)
        ox1, oy1 = self._to_canvas(0.0, 1.0)
        painter.drawLine(ox0, oy0, ox1, oy1)

        painter.setPen(QPen(QColor("#5a6175")))
        painter.setFont(QFont("IBM Plex Mono", 9))
        lx, ly = self._to_canvas(1.0, 0.0)
        painter.drawText(lx - 8, ly + 20, "t")
        lx, ly = self._to_canvas(0.0, 1.0)
        painter.drawText(lx - 22, ly + 4, "val")

        # Bézier handle lines
        if self._mode == MODE_BEZIER:
            dash_pen = QPen(QColor(HANDLE_LINE_COLOR), 1, Qt.PenStyle.DashLine)
            painter.setPen(dash_pen)
            ax, ay = self._to_canvas(0.0, 0.0)
            p1x, p1y = self._handle_pos(1)
            p2x, p2y = self._handle_pos(2)
            ex, ey = self._to_canvas(1.0, 1.0)
            painter.drawLine(ax, ay, p1x, p1y)
            painter.drawLine(ex, ey, p2x, p2y)

        # Curve
        t_vals = np.linspace(0.0, 1.0, CURVE_SAMPLES)
        if self._mode == MODE_LINEAR:
            y_vals = ease_linear(t_vals)
        elif self._mode == MODE_EASE_INOUT:
            y_vals = ease_in_out_cosine(t_vals)
        else:
            y_vals = ease_cubic_bezier(t_vals, self._p1x, self._p1y, self._p2x, self._p2y)

        path = QPainterPath()
        px0, py0 = self._to_canvas(float(t_vals[0]), float(y_vals[0]))
        path.moveTo(px0, py0)
        for i in range(1, len(t_vals)):
            pxi, pyi = self._to_canvas(float(t_vals[i]), float(y_vals[i]))
            path.lineTo(pxi, pyi)

        painter.setPen(QPen(QColor(CURVE_COLOR), 2.5))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)

        # Keyframe dots
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(KEYFRAME_COLOR)))
        for kx, ky in [(0.0, 0.0), (1.0, 1.0)]:
            kpx, kpy = self._to_canvas(kx, ky)
            painter.drawEllipse(kpx - 5, kpy - 5, 10, 10)

        # Bézier handle dots
        if self._mode == MODE_BEZIER:
            painter.setPen(QPen(QColor(HANDLE_COLOR), 1.5))
            painter.setBrush(QBrush(QColor("#2a2540")))
            for hh in (1, 2):
                hpx, hpy = self._handle_pos(hh)
                r = HANDLE_RADIUS
                painter.drawEllipse(hpx - r, hpy - r, r * 2, r * 2)

        # Animated dot
        eased = self.get_eased_value(self._t)
        dpx, dpy = self._to_canvas(self._t, eased)

        tx, _ = self._to_canvas(self._t, 0.0)
        _, by = self._to_canvas(0.0, 0.0)
        dash_t_pen = QPen(QColor("#fb923c"), 1, Qt.PenStyle.DashLine)
        painter.setPen(dash_t_pen)
        painter.drawLine(tx, dpy, tx, by)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(DOT_COLOR)))
        painter.drawEllipse(dpx - 6, dpy - 6, 12, 12)

        painter.end()


# ── STEP 3: Ball preview ───────────────────────────────────────────────────────

class BallPreview(QWidget):
    """A QPainter preview showing a ball moving left-to-right.

    The ball's X position is driven by the eased value from the curve editor.
    This panel is a pure view — it receives (t, eased_value) and redraws.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialise the preview canvas."""
        super().__init__(parent)
        self.setMinimumHeight(80)
        self.setMaximumHeight(120)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._t: float = 0.0
        self._eased: float = 0.0

    def set_values(self, t: float, eased: float) -> None:
        """Update the current time and eased position and repaint.

        Args:
            t: Raw normalised time in [0, 1].
            eased: Eased normalised position in [0, 1].
        """
        self._t = t
        self._eased = eased
        self.update()

    def paintEvent(self, event) -> None:  # noqa: ANN001
        """Draw the track and ball."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        pad = 40
        cy = h // 2
        ball_r = 14

        painter.fillRect(0, 0, w, h, QColor(PREVIEW_BG))

        painter.setPen(QPen(QColor(TRACK_COLOR), 2))
        painter.drawLine(pad, cy, w - pad, cy)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor("#252b38")))
        for mx in (pad, w - pad):
            painter.drawEllipse(mx - 5, cy - 5, 10, 10)

        raw_x = int(pad + self._t * (w - pad * 2))
        painter.setBrush(QBrush(QColor(60, 80, 120, 80)))
        painter.drawEllipse(raw_x - ball_r, cy - ball_r, ball_r * 2, ball_r * 2)

        ball_x = int(pad + self._eased * (w - pad * 2))
        painter.setBrush(QBrush(QColor(BALL_COLOR)))
        painter.drawEllipse(ball_x - ball_r, cy - ball_r, ball_r * 2, ball_r * 2)

        painter.setPen(QPen(QColor("#5a6175")))
        painter.setFont(QFont("IBM Plex Mono", 9))
        painter.drawText(8, h - 6, f"t={self._t:.2f}  →  eased={self._eased:.2f}")

        painter.end()


# ── STEP 4: Control panel ──────────────────────────────────────────────────────

class ControlPanel(QWidget):
    """Right-side panel with mode selector and Bézier handle sliders.

    The sliders are only visible in Bézier mode.  In Linear and
    Ease In-Out mode they are hidden to avoid confusion.
    """

    params_changed: Signal = Signal(int, float, float, float, float)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialise the control panel."""
        super().__init__(parent)
        self.setFixedWidth(220)
        self._p1x, self._p1y = DEFAULT_P1
        self._p2x, self._p2y = DEFAULT_P2
        self._build_ui()

    def _build_ui(self) -> None:
        """Create mode selector, Bézier sliders, and info labels."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

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

        # ── Author credit ──────────────────────────────────────────────────────
        author_label = QLabel("Renato Fera P.")
        author_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        author_label.setStyleSheet("color: #5a6175; font-size: 10px; letter-spacing: 0.5px;")
        layout.addWidget(author_label)

        layout.addSpacing(4)

        # Mode selector
        mode_label = QLabel("Interpolation mode")
        mode_label.setStyleSheet("color: #5a6175; font-size: 12px;")
        self._combo = QComboBox()
        self._combo.addItems(MODE_NAMES)
        self._combo.setCurrentIndex(MODE_BEZIER)
        layout.addWidget(mode_label)
        layout.addWidget(self._combo)

        # Bézier control point sliders
        self._bezier_group = QGroupBox("Bézier control points")
        self._bezier_group.setStyleSheet(
            "QGroupBox { color: #a78bfa; font-size: 12px; border: 1px solid #252b38;"
            " border-radius: 4px; margin-top: 8px; padding-top: 8px; }"
        )
        form = QFormLayout(self._bezier_group)
        form.setSpacing(8)

        self._s_p1x = self._make_slider(int(DEFAULT_P1[0] * 100))
        self._s_p1y = self._make_slider(int(DEFAULT_P1[1] * 100))
        self._s_p2x = self._make_slider(int(DEFAULT_P2[0] * 100))
        self._s_p2y = self._make_slider(int(DEFAULT_P2[1] * 100))

        form.addRow("P1  x", self._s_p1x)
        form.addRow("P1  y", self._s_p1y)
        form.addRow("P2  x", self._s_p2x)
        form.addRow("P2  y", self._s_p2y)

        layout.addWidget(self._bezier_group)

        # Info box
        info = QLabel(
            "In Bézier mode:\n"
            "• Drag handles on the canvas\n"
            "• Or use the sliders above\n\n"
            "The faint ghost ball shows\n"
            "the raw linear position."
        )
        info.setStyleSheet(
            "color: #5a6175; font-size: 11px; font-family: 'IBM Plex Mono';"
            " padding: 8px; background: #161a22; border-radius: 4px;"
        )
        info.setWordWrap(True)
        layout.addWidget(info)
        layout.addStretch()

        # Connections
        self._combo.currentIndexChanged.connect(self._on_mode_changed)
        self._s_p1x.valueChanged.connect(self._on_slider_changed)
        self._s_p1y.valueChanged.connect(self._on_slider_changed)
        self._s_p2x.valueChanged.connect(self._on_slider_changed)
        self._s_p2y.valueChanged.connect(self._on_slider_changed)

        self._on_mode_changed(MODE_BEZIER)

    def _make_slider(self, default: int) -> QSlider:
        """Create a horizontal slider in range [0, 100].

        Args:
            default: Initial value (0–100, maps to 0.0–1.0).

        Returns:
            Configured QSlider.
        """
        s = QSlider(Qt.Orientation.Horizontal)
        s.setRange(0, 100)
        s.setValue(default)
        return s

    def _on_mode_changed(self, index: int) -> None:
        """Show/hide the Bézier group based on selected mode.

        Args:
            index: The newly selected QComboBox index.
        """
        self._bezier_group.setVisible(index == MODE_BEZIER)
        self._emit()

    def _on_slider_changed(self) -> None:
        """Read all sliders and emit params_changed."""
        self._p1x = self._s_p1x.value() / 100.0
        self._p1y = self._s_p1y.value() / 100.0
        self._p2x = self._s_p2x.value() / 100.0
        self._p2y = self._s_p2y.value() / 100.0
        self._emit()

    def _emit(self) -> None:
        """Emit params_changed with the current mode and control point values."""
        self.params_changed.emit(
            self._combo.currentIndex(),
            self._p1x, self._p1y,
            self._p2x, self._p2y,
        )

    def sync_handles(self, p1x: float, p1y: float, p2x: float, p2y: float) -> None:
        """Update sliders to reflect handle positions dragged on the canvas.

        Args:
            p1x, p1y: New P1 coordinates in [0, 1].
            p2x, p2y: New P2 coordinates in [0, 1].
        """
        for slider, value in (
            (self._s_p1x, p1x),
            (self._s_p1y, p1y),
            (self._s_p2x, p2x),
            (self._s_p2y, p2y),
        ):
            slider.blockSignals(True)
            slider.setValue(int(value * 100))
            slider.blockSignals(False)

        self._p1x, self._p1y = p1x, p1y
        self._p2x, self._p2y = p2x, p2y


# ── STEP 5: Main window ────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    """Application main window.

    Layout:
    ┌────────────────────────────┬──────────────┐
    │  CurveCanvas               │ ControlPanel │
    │  (graph editor)            │ (mode +      │
    ├────────────────────────────┤  sliders)    │
    │  BallPreview               │              │
    │  (eased ball)              │              │
    └────────────────────────────┴──────────────┘

    Signal flow:
    QTimer.timeout → _on_tick → CurveCanvas.set_t + BallPreview.set_values
    ControlPanel.params_changed → _on_params_changed → CurveCanvas + BallPreview
    CurveCanvas.easing_changed → _on_easing_dragged → ControlPanel.sync_handles
    """

    def __init__(self) -> None:
        """Initialise the window, create panels, start the animation timer."""
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)

        self._t: float = 0.0
        self._direction: int = 1
        self._elapsed = QTimer(self)
        self._elapsed.setInterval(TIMER_INTERVAL_MS)

        self._build_ui()
        self._connect_signals()
        self._elapsed.start()

    def _build_ui(self) -> None:
        """Create and arrange all panels."""
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        self._curve = CurveCanvas()
        self._preview = BallPreview()

        left_layout.addWidget(self._curve, stretch=1)
        left_layout.addWidget(self._preview)

        self._controls = ControlPanel()

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(self._controls)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([580, 220])

        self.setCentralWidget(splitter)

    def _connect_signals(self) -> None:
        """Connect all signals between panels and the timer."""
        self._elapsed.timeout.connect(self._on_tick)
        self._controls.params_changed.connect(self._on_params_changed)
        self._curve.easing_changed.connect(self._on_easing_dragged)

    # ── Timer slot ────────────────────────────────────────────────────────────

    def _on_tick(self) -> None:
        """Advance t, ping-pong at boundaries, and update both panels."""
        step = (1.0 / TARGET_FPS) / ANIMATION_DURATION_S
        self._t += self._direction * step

        if self._t >= 1.0:
            self._t = 1.0
            self._direction = -1
        elif self._t <= 0.0:
            self._t = 0.0
            self._direction = 1

        eased = self._curve.get_eased_value(self._t)
        self._curve.set_t(self._t)
        self._preview.set_values(self._t, eased)

    # ── Control panel slot ────────────────────────────────────────────────────

    def _on_params_changed(
        self,
        mode: int,
        p1x: float, p1y: float,
        p2x: float, p2y: float,
    ) -> None:
        """Propagate new mode and Bézier parameters to the curve canvas.

        Args:
            mode: Selected interpolation mode index.
            p1x, p1y: P1 control point in [0, 1].
            p2x, p2y: P2 control point in [0, 1].
        """
        self._curve.set_mode(mode)
        self._curve._p1x, self._curve._p1y = p1x, p1y
        self._curve._p2x, self._curve._p2y = p2x, p2y
        self._curve.update()

    # ── Canvas drag slot ──────────────────────────────────────────────────────

    def _on_easing_dragged(
        self,
        p1x: float, p1y: float,
        p2x: float, p2y: float,
    ) -> None:
        """Sync the control panel sliders when handles are dragged on the canvas.

        Args:
            p1x, p1y: New P1 position.
            p2x, p2y: New P2 position.
        """
        self._controls.sync_handles(p1x, p1y, p2x, p2y)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    """Create the QApplication, show the window, and start the event loop."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

# =============================================================================
# End of 03_ca_curves.py
# Author   : Renato Fera P.
# LinkedIn : https://www.linkedin.com/in/renato-profeta/
# Course   : Computeranimation — Chapter 03: PySide6 & Python Animation
# Institute: Institut für Medientechnik, Technische Universität Ilmenau
# =============================================================================
