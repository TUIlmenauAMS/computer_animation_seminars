"""03_ca_hello_pyside6.py — Hello PySide6.

The simplest complete PySide6 desktop application.
One QMainWindow with a title label, a slider, a live value display,
and a reset button. Signals and slots wire everything together.

This demo is the foundation for every subsequent demo in Seminar 03.
Read it line by line — every Qt concept introduced here reappears in
demos 2 through 5 at larger scale.

How to run
----------
First time (creates .venv and installs all dependencies):
    uv sync

Every time after that:
    uv run python 03_ca_hello_pyside6.py

Ubuntu system dependencies (run once if not already installed):
    sudo apt install libgl1 libegl1 libxcb-cursor0 libxcb-xinerama0

Windows: no extra steps needed — PySide6 bundles everything.

Connections to other seminars
------------------------------
- The event loop (app.exec) is the desktop equivalent of the browser's
  event loop that drives requestAnimationFrame in seminars 01 and 02.
- The slider → label signal connection is the same observer pattern
  that connects animation rig controls to their downstream solvers.
"""

from __future__ import annotations

# stdlib
import sys

# third-party — installed by uv sync from pyproject.toml
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

# ── Constants ─────────────────────────────────────────────────────────────────
# All magic numbers live here as named constants.
# This is required by ruff's style rules and makes the code self-documenting.

WINDOW_TITLE: str = "Hello PySide6 — Computeranimation Seminar 03"
WINDOW_MIN_WIDTH: int = 400
WINDOW_MIN_HEIGHT: int = 220

SLIDER_MIN: int = 0
SLIDER_MAX: int = 100
SLIDER_DEFAULT: int = 50

LAYOUT_MARGIN: int = 32       # pixels of padding around the layout edges
LAYOUT_SPACING: int = 16      # pixels between widgets in the layout


# ── STEP 2: QMainWindow subclass ──────────────────────────────────────────────

class MainWindow(QMainWindow):
    """The application's main window.

    Subclassing QMainWindow (rather than using a plain QWidget) gives us
    a menu bar, status bar, and dock widget areas for free. In this demo
    we do not use them, but the pattern is the same in demos 4 and 5
    where we do.

    Qt ownership: this window owns all child widgets created in __init__.
    When the window is closed and destroyed, Qt automatically destroys
    every child widget — no manual cleanup needed.
    """

    def __init__(self) -> None:
        """Initialise the main window and build the UI."""
        # Always call the parent __init__ first.
        # This initialises the Qt C++ object underneath the Python wrapper.
        super().__init__()

        self.setWindowTitle(WINDOW_TITLE)

        # setMinimumSize prevents the user from resizing the window smaller
        # than this. The window can still grow freely.
        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)

        # Build the UI — separated into its own method for clarity.
        self._build_ui()

    def _build_ui(self) -> None:
        """Create all widgets, assemble the layout, and connect signals."""

        # ── STEP 3: Central widget and layout ─────────────────────────────────
        #
        # QMainWindow requires a "central widget" — the main content area.
        # We use a plain QWidget as the container and give it a QVBoxLayout
        # (vertical box layout: widgets stacked top-to-bottom).
        #
        # Qt concept: a layout is attached to a widget, not the window.
        # The central widget owns the layout; the layout owns nothing itself —
        # it just manages the geometry of the widgets added to it.

        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)          # attach layout to central widget
        layout.setContentsMargins(             # padding around the edge
            LAYOUT_MARGIN, LAYOUT_MARGIN, LAYOUT_MARGIN, LAYOUT_MARGIN
        )
        layout.setSpacing(LAYOUT_SPACING)      # gap between each widget

        # ── STEP 4: Widgets ───────────────────────────────────────────────────

        # Title label — static text, no interaction.
        # HTML subset supported: <b>, <i>, <br>, basic inline styles.
        title_label = QLabel("<b>Seminar 03 — Hello PySide6</b>")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Value display label — updated live by the slider signal.
        # We store it as an instance attribute so the slot can access it.
        self._value_label = QLabel(f"Value: {SLIDER_DEFAULT}")
        self._value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # QSlider — a range input widget.
        # Qt.Orientation.Horizontal: the slider track runs left to right.
        # The equivalent in the browser is <input type="range">.
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(SLIDER_MIN)
        self._slider.setMaximum(SLIDER_MAX)
        self._slider.setValue(SLIDER_DEFAULT)
        self._slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._slider.setTickInterval(10)

        # Reset button — clicking it returns the slider to its default value.
        reset_btn = QPushButton("Reset")

        # Add widgets to the layout in top-to-bottom order.
        layout.addWidget(title_label)
        layout.addWidget(self._value_label)
        layout.addWidget(self._slider)
        layout.addWidget(reset_btn)

        # addStretch() inserts an invisible expanding spacer below the button.
        # It pushes all four widgets toward the top, leaving empty space below.
        # Equivalent to CSS flex-grow: 1 on an empty element.
        layout.addStretch()

        # ── STEP 5: Signals and slots ─────────────────────────────────────────
        #
        # ANIMATION CONCEPT — the observer pattern:
        # The slider does not know the label exists. The label does not know
        # the slider exists. They are decoupled. The MainWindow connects them:
        # "whenever the slider emits valueChanged, call _on_slider_changed."
        # This is the same pattern that connects animation rig controls to
        # their downstream solvers in Maya and Houdini.

        # QSlider.valueChanged(int) is emitted on every movement.
        # The int argument (the new value) is automatically passed to the slot.
        self._slider.valueChanged.connect(self._on_slider_changed)

        # QPushButton.clicked() is emitted on every press.
        # We use a lambda to inline the one-line reset action.
        reset_btn.clicked.connect(lambda: self._slider.setValue(SLIDER_DEFAULT))

    # ── Slots ──────────────────────────────────────────────────────────────────

    def _on_slider_changed(self, value: int) -> None:
        """Slot: called every time the slider value changes.

        Qt automatically passes the new value as the argument because
        QSlider.valueChanged is declared as Signal(int).

        Args:
            value: The slider's new integer value.
        """
        self._value_label.setText(f"Value: {value}")


# ── STEP 1 + STEP 6: QApplication and entry point ────────────────────────────

def main() -> None:
    """Create the QApplication, show the window, and start the event loop.

    Qt concept — QApplication:
    QApplication MUST be created before any QWidget. It initialises Qt
    internals, sets up the platform plugin (the OS window manager bridge),
    and owns the event loop. Only one QApplication can exist per process.

    ANIMATION CONCEPT — the event loop:
    app.exec() blocks here and runs Qt's event loop. The event loop is
    the same concept as the browser's event loop that drives
    requestAnimationFrame in seminars 01 and 02: it waits for events
    (user input, timer expiry, repaint requests) and dispatches them to
    the correct handlers (slots / rAF callbacks). Your code does not run
    continuously — it runs in response to events.

    Why sys.exit(app.exec()):
    app.exec() returns an integer exit code when the last window closes
    (0 = clean exit, non-zero = error). sys.exit() forwards that code to
    the operating system. Omitting sys.exit() works interactively but is
    technically incorrect — always use it.
    """
    # sys.argv is passed so Qt can consume its own command-line flags
    # (e.g. -style fusion, -platform xcb) before your application code runs.
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()   # show() makes the window visible; it does not start the loop

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
