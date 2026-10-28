"""03_ca_timeline.py — Animation Timeline Widget.

A reusable animation timeline component built with PySide6.
A QSlider acts as the frame scrubber, a QTimer drives playback,
Play/Pause/Stop buttons and Step controls give full transport control,
and a QPainter canvas shows a bouncing ball driven by the current frame.

The TimelineWidget class is designed as a drop-in component: it emits
a custom Signal(int) whenever the frame changes, so any downstream widget
can react without being coupled to the timeline's internals.  You will
reuse it directly in Demo 5.

How to run
----------
    uv sync                                  # first time only
    uv run python 03_ca_timeline.py

Ubuntu system dependencies (once):
    sudo apt install libgl1 libegl1 libxcb-cursor0 libxcb-xinerama0

Connections to other seminars
------------------------------
- The QTimer → slot → update() cycle is the desktop equivalent of
  requestAnimationFrame (seminar 01) and renderer.setAnimationLoop
  (seminar 02).  Same concept, different API.
- The bouncing ball's Y position uses a parabolic ease — the same
  curve math explored interactively in seminar 01's easing demo and
  drawn with QPainter in demo 3 of this seminar.
- Signal(int) makes TimelineWidget reusable: it is the observer pattern
  that connects animation controls to viewports in Maya and Houdini.
"""

from __future__ import annotations

import sys
import math

from PySide6.QtCore import Qt, QTimer, QElapsedTimer, Signal
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QFont
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

# ── Constants ─────────────────────────────────────────────────────────────────

WINDOW_TITLE: str = "Animation Timeline — Computeranimation Seminar 03"
WINDOW_MIN_WIDTH: int = 680
WINDOW_MIN_HEIGHT: int = 460

# Animation parameters
TARGET_FPS: int = 30
TOTAL_FRAMES: int = 90                        # 3 seconds at 30 fps
TIMER_INTERVAL_MS: int = 1000 // TARGET_FPS   # 33 ms ≈ 30 fps

# Ball canvas
CANVAS_BG_COLOR: str = "#0d0f14"
BALL_COLOR: str = "#4f9cf9"
GROUND_COLOR: str = "#252b38"
BALL_RADIUS: int = 18
GROUND_Y_FRACTION: float = 0.82              # ground line at 82% of canvas height

# Timeline widget
TIMELINE_BG_COLOR: str = "#161a22"
SCRUBBER_COLOR: str = "#fb923c"
PLAYHEAD_WIDTH: int = 2

# Layout
LAYOUT_MARGIN: int = 16
LAYOUT_SPACING: int = 10


# ── STEP 2: Animation state dataclass ─────────────────────────────────────────

class AnimationState:
    """Pure data model for the animation — no Qt dependencies.

    ANIMATION CONCEPT — model / view separation:
    Keeping animation state in a plain Python object (no Qt imports)
    means the math is testable, portable, and independent of the UI.
    The same AnimationState could drive a render farm script, a web
    export, or an audio envelope without any changes.  This is the
    same principle behind Three.js AnimationMixer (the model) being
    separate from the HTML controls (the view) in seminar 02.

    Attributes:
        current_frame: The frame currently displayed.
        is_playing: Whether the timeline is advancing automatically.
        total_frames: Total number of frames in the clip.
        fps: Playback frame rate.
    """

    def __init__(self, total_frames: int = TOTAL_FRAMES, fps: int = TARGET_FPS) -> None:
        """Initialise with default playback state."""
        self.current_frame: int = 0
        self.is_playing: bool = False
        self.total_frames: int = total_frames
        self.fps: int = fps

    @property
    def duration_seconds(self) -> float:
        """Total clip duration in seconds."""
        return self.total_frames / self.fps

    @property
    def current_time_seconds(self) -> float:
        """Current playback position in seconds."""
        return self.current_frame / self.fps


# ── STEP 3: Ball canvas ────────────────────────────────────────────────────────

class BallCanvas(QWidget):
    """A QPainter canvas that renders a bouncing ball.

    The ball's vertical position is computed from the current frame number
    using a parabolic bounce function.  The canvas is purely a view —
    it receives a frame number via set_frame() and redraws; it emits nothing.

    The bounce uses a simple parabolic arc: the ball rises from the ground,
    reaches its peak at the midpoint, then falls back.  A subtle squash
    effect compresses the ball vertically on impact — a classic animation
    principle (squash and stretch) that adds weight and life.
    """

    def __init__(self, state: AnimationState, parent: QWidget | None = None) -> None:
        """Initialise the canvas with a reference to the shared animation state.

        Args:
            state: The shared AnimationState model.
            parent: Optional Qt parent widget.
        """
        super().__init__(parent)
        self._state = state
        self.setMinimumHeight(260)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    # ── Public interface ───────────────────────────────────────────────────────

    def set_frame(self, frame: int) -> None:
        """Update the displayed frame and schedule a repaint.

        Args:
            frame: The new frame number to display.
        """
        self._state.current_frame = frame
        self.update()   # schedule repaint — Qt calls paintEvent when ready

    # ── Animation math ─────────────────────────────────────────────────────────

    def _ball_y(self, frame: int) -> float:
        """Compute normalised ball Y position [0.0 = ground, 1.0 = peak].

        Uses a parabolic arc that repeats every TOTAL_FRAMES frames.
        The arc is symmetric: the ball spends equal time rising and falling.

        ANIMATION CONCEPT — easing on a parabola:
        This is the same math as a quadratic ease-in-out, just mapped to
        a physical arc.  The sinusoidal variant used here (sin²) gives a
        smooth, physically plausible bounce without needing a full physics
        simulation.  Compare with the Bézier easing in demo 3.

        Args:
            frame: Current frame number.

        Returns:
            Normalised height in [0.0, 1.0].
        """
        # Normalise frame to [0, 1] over the full clip, then to [0, π].
        t = (frame % self._state.total_frames) / self._state.total_frames
        # sin² gives a smooth parabolic arc: 0 at t=0, peak at t=0.5, 0 at t=1.
        return math.sin(math.pi * t) ** 2

    # ── QPainter rendering ─────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:  # noqa: ANN001
        """Qt calls this whenever the widget needs to be redrawn.

        Never call this directly — use self.update() to request a repaint.
        """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()

        # ── Background ────────────────────────────────────────────────────────
        painter.fillRect(0, 0, w, h, QColor(CANVAS_BG_COLOR))

        # ── Ground line ───────────────────────────────────────────────────────
        ground_y = int(h * GROUND_Y_FRACTION)
        ground_pen = QPen(QColor(GROUND_COLOR), 2)
        painter.setPen(ground_pen)
        painter.drawLine(0, ground_y, w, ground_y)

        # ── Ball ──────────────────────────────────────────────────────────────
        normalised_height = self._ball_y(self._state.current_frame)

        # Map normalised height to pixel Y.
        # At height=0: ball sits on the ground (centre = ground_y - radius).
        # At height=1: ball is at the top of the arc.
        arc_range = ground_y - BALL_RADIUS - int(h * 0.12)  # usable vertical range
        ball_cy = ground_y - BALL_RADIUS - int(normalised_height * arc_range)
        ball_cx = w // 2

        # Squash and stretch: compress vertically near the ground,
        # stretch vertically at the peak.  Keeps the volume (area) roughly constant.
        # squash=1.0 at peak (full round), squash<1.0 near ground (flat ellipse).
        squash = 0.55 + 0.45 * normalised_height
        rx = int(BALL_RADIUS / squash)        # horizontal radius (wider when squashed)
        ry = int(BALL_RADIUS * squash)        # vertical radius (shorter when squashed)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(BALL_COLOR)))
        painter.drawEllipse(ball_cx - rx, ball_cy - ry, rx * 2, ry * 2)

        # ── Shadow ────────────────────────────────────────────────────────────
        # Shadow grows smaller and more opaque as the ball approaches the ground.
        shadow_alpha = int(180 * (1.0 - normalised_height))
        shadow_rx = int(BALL_RADIUS * (0.3 + 0.7 * (1.0 - normalised_height)))
        shadow_color = QColor(0, 0, 0, shadow_alpha)
        painter.setBrush(QBrush(shadow_color))
        painter.drawEllipse(ball_cx - shadow_rx, ground_y - 4, shadow_rx * 2, 8)

        # ── Frame / time overlay ──────────────────────────────────────────────
        painter.setPen(QPen(QColor("#5a6175")))
        painter.setFont(QFont("IBM Plex Mono", 10))
        t_str = f"{self._state.current_time_seconds:.2f} s"
        painter.drawText(8, h - 8, t_str)

        painter.end()


# ── STEP 4: Timeline widget ────────────────────────────────────────────────────

class TimelineWidget(QWidget):
    """A reusable animation timeline component.

    Contains: a QSlider scrubber, Play/Pause, Stop, Step Back, and Step
    Forward buttons, a frame counter label, and a time display in seconds.

    ANIMATION CONCEPT — reusability via signals:
    TimelineWidget emits frame_changed(int) whenever the current frame
    changes — whether from the timer, the scrubber, or the step buttons.
    Any widget that cares about the frame just connects to frame_changed.
    The timeline does not need to know what is downstream.  This is how
    professional DCC tool panels communicate: loose coupling through a
    well-defined signal interface.

    Signals:
        frame_changed: Emitted with the new frame number on every change.
    """

    # Custom signal — carries the new frame number as an int.
    # Any number of external widgets can connect to this.
    frame_changed: Signal = Signal(int)

    def __init__(
        self,
        state: AnimationState,
        parent: QWidget | None = None,
    ) -> None:
        """Initialise the timeline with a shared AnimationState.

        Args:
            state: The shared animation state model.
            parent: Optional Qt parent widget.
        """
        super().__init__(parent)
        self._state = state
        self._build_ui()
        self._setup_timer()

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        """Create and arrange all timeline widgets."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(LAYOUT_MARGIN, LAYOUT_MARGIN, LAYOUT_MARGIN, 8)
        layout.setSpacing(LAYOUT_SPACING)

        self.setStyleSheet(f"background: {TIMELINE_BG_COLOR};")

        # ── Scrubber row ──────────────────────────────────────────────────────
        self._scrubber = QSlider(Qt.Orientation.Horizontal)
        self._scrubber.setMinimum(0)
        self._scrubber.setMaximum(self._state.total_frames - 1)
        self._scrubber.setValue(0)
        self._scrubber.setToolTip("Drag to scrub")
        layout.addWidget(self._scrubber)

        # ── Info row: frame counter + time display ────────────────────────────
        info_row = QHBoxLayout()
        info_row.setSpacing(4)

        self._frame_label = QLabel("Frame  0")
        self._frame_label.setStyleSheet(
            "font-family: 'IBM Plex Mono'; font-size: 12px; color: #d8dce6;"
        )

        self._time_label = QLabel("0.00 s")
        self._time_label.setStyleSheet(
            "font-family: 'IBM Plex Mono'; font-size: 12px; color: #5a6175;"
        )

        total_label = QLabel(f"/ {self._state.total_frames - 1}  ({self._state.duration_seconds:.1f} s)")
        total_label.setStyleSheet(
            "font-family: 'IBM Plex Mono'; font-size: 12px; color: #5a6175;"
        )

        info_row.addWidget(self._frame_label)
        info_row.addWidget(self._time_label)
        info_row.addWidget(total_label)
        info_row.addStretch()
        layout.addLayout(info_row)

        # ── Transport buttons ─────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self._play_btn = QPushButton("▶  Play")
        stop_btn       = QPushButton("■  Stop")
        back_btn       = QPushButton("◀  Step")
        fwd_btn        = QPushButton("Step  ▶")

        for btn in (self._play_btn, stop_btn, back_btn, fwd_btn):
            btn.setFixedHeight(32)
            btn.setStyleSheet(
                "QPushButton {"
                "  background: #1e2330; border: 1px solid #252b38;"
                "  color: #d8dce6; border-radius: 4px;"
                "  font-family: 'IBM Plex Mono'; font-size: 12px; padding: 0 12px;"
                "}"
                "QPushButton:hover { background: #252b38; }"
                "QPushButton:pressed { background: #2e3548; }"
            )

        btn_row.addWidget(back_btn)
        btn_row.addWidget(self._play_btn)
        btn_row.addWidget(stop_btn)
        btn_row.addWidget(fwd_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # ── Signal connections ────────────────────────────────────────────────
        self._scrubber.valueChanged.connect(self._on_scrubber_moved)
        self._play_btn.clicked.connect(self._toggle_play)
        stop_btn.clicked.connect(self._stop)
        back_btn.clicked.connect(self._step_back)
        fwd_btn.clicked.connect(self._step_forward)

    # ── STEP 5: QTimer setup ──────────────────────────────────────────────────

    def _setup_timer(self) -> None:
        """Create and configure the QTimer that drives playback.

        QT CONCEPT — QTimer and the event loop:
        QTimer.timeout fires in the main thread's event loop, so it is
        safe to update Qt widgets directly from the connected slot.
        No locks, no thread marshalling.

        The timer fires every TIMER_INTERVAL_MS milliseconds.
        At 30 fps that is every 33 ms.  Qt may fire slightly late if the
        event loop is busy — always use QElapsedTimer for accurate delta
        in real-time rendering (see demo 4).  For a timeline scrubber,
        fixed-frame advance is fine.
        """
        self._timer = QTimer(self)
        self._timer.setInterval(TIMER_INTERVAL_MS)
        self._timer.timeout.connect(self._on_tick)

    # ── Timer slot ────────────────────────────────────────────────────────────

    def _on_tick(self) -> None:
        """Advance the frame counter by one and emit frame_changed.

        Called automatically by QTimer every TIMER_INTERVAL_MS milliseconds
        while playing.  Loops back to frame 0 at the end of the clip.
        """
        next_frame = (self._state.current_frame + 1) % self._state.total_frames
        self._set_frame(next_frame)

    # ── Transport actions ──────────────────────────────────────────────────────

    def _toggle_play(self) -> None:
        """Toggle between playing and paused states."""
        if self._state.is_playing:
            self._pause()
        else:
            self._play()

    def _play(self) -> None:
        """Start playback."""
        self._state.is_playing = True
        self._play_btn.setText("⏸  Pause")
        self._timer.start()

    def _pause(self) -> None:
        """Pause playback without resetting the frame."""
        self._state.is_playing = False
        self._play_btn.setText("▶  Play")
        self._timer.stop()

    def _stop(self) -> None:
        """Stop playback and return to frame 0."""
        self._pause()
        self._set_frame(0)

    def _step_back(self) -> None:
        """Step one frame backward (pauses if playing)."""
        self._pause()
        prev_frame = max(0, self._state.current_frame - 1)
        self._set_frame(prev_frame)

    def _step_forward(self) -> None:
        """Step one frame forward (pauses if playing)."""
        self._pause()
        next_frame = min(self._state.total_frames - 1, self._state.current_frame + 1)
        self._set_frame(next_frame)

    def _on_scrubber_moved(self, value: int) -> None:
        """Slot: called when the user drags the scrubber.

        Pauses playback (scrubbing is a manual override) and sets the frame.

        Args:
            value: The scrubber's new integer value (= frame number).
        """
        if self._state.is_playing:
            self._pause()
        self._set_frame(value)

    # ── Internal frame setter ──────────────────────────────────────────────────

    def _set_frame(self, frame: int) -> None:
        """Update state, sync the scrubber, update labels, and emit frame_changed.

        This is the single point through which all frame changes flow.
        Whether the change comes from the timer, the scrubber, or a step
        button, it always goes through here — keeping the UI consistent.

        Args:
            frame: The new frame number to display.
        """
        self._state.current_frame = frame

        # Block scrubber signals while we set its value programmatically.
        # Without this, setting the scrubber would re-emit valueChanged,
        # which would call _on_scrubber_moved, which would call _set_frame
        # again — an infinite signal loop.
        self._scrubber.blockSignals(True)
        self._scrubber.setValue(frame)
        self._scrubber.blockSignals(False)

        # Update info labels.
        self._frame_label.setText(f"Frame {frame:3d}")
        self._time_label.setText(f"{self._state.current_time_seconds:.2f} s")

        # Emit the public signal — all connected downstream widgets receive
        # the new frame number.  The timeline does not need to know who
        # is listening.
        self.frame_changed.emit(frame)


# ── STEP 6: Main window ────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    """Application main window.

    Layout (vertical QSplitter):
    ┌─────────────────────────┐
    │     BallCanvas          │  ← top: the animated ball viewport
    ├─────────────────────────┤
    │     TimelineWidget      │  ← bottom: scrubber + transport controls
    └─────────────────────────┘

    The two panels communicate only through signals:
    TimelineWidget.frame_changed(int) → BallCanvas.set_frame(int)

    Neither panel imports the other's class.  MainWindow is the only
    place that knows both exist — it creates them and wires them together.
    """

    def __init__(self) -> None:
        """Initialise the window, create panels, and connect signals."""
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)

        # ── Shared animation state (the model) ────────────────────────────────
        # Created once in MainWindow and passed to both panels by reference.
        # Both panels read from it; only TimelineWidget writes to it.
        self._state = AnimationState()

        # ── Panels (the views) ────────────────────────────────────────────────
        self._canvas   = BallCanvas(self._state)
        self._timeline = TimelineWidget(self._state)

        # ── Layout: vertical splitter ─────────────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self._canvas)
        splitter.addWidget(self._timeline)
        splitter.setStretchFactor(0, 1)   # canvas grows when window resizes
        splitter.setStretchFactor(1, 0)   # timeline stays a fixed height
        splitter.setSizes([320, 140])

        self.setCentralWidget(splitter)

        # ── Connect signals (the controller) ──────────────────────────────────
        # When the timeline emits frame_changed, tell the canvas to update.
        # This single line of code is the entire communication channel
        # between the two panels.
        self._timeline.frame_changed.connect(self._canvas.set_frame)

        # ── STEP 7: Keyboard shortcuts ─────────────────────────────────────────
        # These mirror the standard shortcuts in Maya, Blender, and Nuke.
        # QShortcut activates when the key is pressed while this window has focus.
        from PySide6.QtGui import QShortcut, QKeySequence  # noqa: PLC0415

        QShortcut(QKeySequence("Space"),          self, activated=self._timeline._toggle_play)
        QShortcut(QKeySequence(Qt.Key.Key_Left),  self, activated=self._timeline._step_back)
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, activated=self._timeline._step_forward)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    """Create the QApplication, show the window, and start the event loop."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
