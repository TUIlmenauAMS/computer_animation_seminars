# =============================================================================
# Computeranimation — Seminar 04
# Demo 5: glTF Animation Curve Editor
# =============================================================================
#
# A professional Qt application for inspecting, visualising and editing
# animation data inside binary glTF (.glb) files.
#
# Stack:
#   PySide6    — Qt6 Python binding, the UI framework behind Maya, Blender,
#                Houdini and Nuke. Industry standard for desktop tools.
#   PyQtGraph  — GPU-accelerated plot library built on Qt. Used in
#                scientific instruments, trading terminals and real-time
#                data dashboards. Provides the animation curve editor panel.
#   pygltflib  — Pure Python glTF 2.0 reader/writer.
#   NumPy      — All keyframe data lives as float32 typed arrays.
#
# Run this demo:
#   cd seminar_04
#   uv sync                              (first time — creates .venv, installs deps)
#   uv run python 04_ca_gltf_python.py
#
# =============================================================================

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from pygltflib import GLTF2
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QAction, QColor, QFont
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
    QWidget,
)

# =============================================================================
# STEP 1 — Module-level constants
# =============================================================================
#
# ANIMATION CONCEPT: glTF encodes data types as numeric constants defined in
# the WebGL / OpenGL ES specification. 5126 = GL_FLOAT (32-bit IEEE 754).
# These same constants appear in WebGL shaders, Three.js BufferAttribute,
# and every other GL-family API. Learning them here transfers directly.

# Maps glTF accessor componentType codes → (numpy dtype, byte size)
COMPONENT_DTYPE: dict[int, tuple[np.dtype, int]] = {
    5120: (np.dtype("int8"), 1),    # BYTE
    5121: (np.dtype("uint8"), 1),   # UNSIGNED_BYTE
    5122: (np.dtype("int16"), 2),   # SHORT
    5123: (np.dtype("uint16"), 2),  # UNSIGNED_SHORT
    5125: (np.dtype("uint32"), 4),  # UNSIGNED_INT
    5126: (np.dtype("float32"), 4), # FLOAT  ← animation data always uses this
}

# Maps glTF accessor type strings → number of components per element
# VEC3 translation → 3 floats per keyframe; VEC4 rotation → 4 floats (quaternion)
ACCESSOR_NUM_COMPONENTS: dict[str, int] = {
    "SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4,
    "MAT2": 4,   "MAT3": 9, "MAT4": 16,
}

# Per-channel colour palette (RGBA). Each animation component gets a distinct
# colour matching the convention used in Maya, Blender and Unreal:
#   X → red, Y → green, Z → blue, W → yellow (quaternion scalar)
CURVE_COLORS: list[tuple[int, int, int, int]] = [
    (220,  60,  60, 220),   # X — red
    ( 80, 200,  80, 220),   # Y — green
    ( 60, 120, 220, 220),   # Z — blue
    (220, 200,  50, 220),   # W — yellow (quaternion)
]

# Component labels per target path
COMPONENT_LABELS: dict[str, list[str]] = {
    "translation": ["X", "Y", "Z"],
    "rotation":    ["X", "Y", "Z", "W"],
    "scale":       ["X", "Y", "Z"],
    "weights":     ["W0", "W1", "W2", "W3"],
}

# Playback timer interval in milliseconds (≈60 fps)
PLAYBACK_INTERVAL_MS: int = 16

# Dark-theme palette matching the course design system
# (same CSS variables as the HTML seminars: --bg, --surface, --border, --accent)
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
    QPushButton#accent_btn:hover {{
        background-color: #6aabff;
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
    QTreeWidget::item:hover {{
        background-color: {PALETTE['border']};
    }}
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
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    QSlider::groove:horizontal {{
        height: 4px;
        background-color: {PALETTE['border']};
        border-radius: 2px;
    }}
    QSlider::handle:horizontal {{
        width: 14px;
        height: 14px;
        margin: -5px 0;
        background-color: {PALETTE['accent']};
        border-radius: 7px;
    }}
    QSlider::sub-page:horizontal {{
        background-color: {PALETTE['accent']};
        border-radius: 2px;
    }}
    QSplitter::handle {{
        background-color: {PALETTE['border']};
        width: 2px;
        height: 2px;
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
        text-transform: uppercase;
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
        width: 14px;
        height: 14px;
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
        width: 8px;
        margin: 0;
    }}
    QScrollBar::handle:vertical {{
        background: {PALETTE['border']};
        border-radius: 4px;
        min-height: 20px;
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0;
    }}
    QFrame#divider {{
        background-color: {PALETTE['border']};
        max-height: 1px;
    }}
"""


# =============================================================================
# STEP 2 — glTF data-access functions
# =============================================================================
#
# These functions are the core educational content of Demo 5.
# They directly access the binary blob inside a .glb file — the same binary
# layout that Three.js, WebGL, and every other GL-family runtime reads.

def load_gltf(path: Path) -> GLTF2:
    """Load a glTF 2.0 binary (.glb) file from disk.

    Args:
        path: Filesystem path to the .glb file.

    Returns:
        A pygltflib GLTF2 object with all JSON fields parsed and the
        binary blob loaded into memory.

    Raises:
        FileNotFoundError: If the file does not exist at the given path.
        ValueError: If the file is not a valid glTF 2.0 document.
    """
    if not path.exists():
        raise FileNotFoundError(f"GLB file not found: {path}")
    gltf = GLTF2().load(str(path))
    return gltf


def read_accessor(gltf: GLTF2, accessor_index: int) -> np.ndarray:
    """Read one glTF accessor as a NumPy array.

    ANIMATION CONCEPT: A glTF accessor is a typed pointer into the binary
    blob. It says: "starting at byteOffset X in bufferView Y, there are
    count elements of type VEC3 with componentType FLOAT". This is identical
    in concept to a WebGL VertexAttribPointer — both describe how to interpret
    a raw byte range as a structured array of numbers.

    PYTHON CONCEPT: np.frombuffer(raw, dtype=...) returns a read-only array
    backed by the original bytes object. Always call .copy() before modifying
    values in place — bytes objects in Python are immutable, and NumPy
    propagates this by marking the resulting array as read-only.

    Args:
        gltf: The loaded GLTF2 object.
        accessor_index: Index into gltf.accessors.

    Returns:
        NumPy array of shape (count,) for SCALAR or (count, n_components)
        for VEC2/VEC3/VEC4/MAT*.
    """
    accessor = gltf.accessors[accessor_index]
    # pygltflib's get_data_from_accessor handles the bufferView offset / stride
    # calculation for us and returns raw bytes.
    raw: bytes = gltf.get_data_from_accessor(accessor_index)

    _, _ = COMPONENT_DTYPE.get(accessor.componentType, (np.float32, 4))
    dtype, _ = COMPONENT_DTYPE.get(accessor.componentType, (np.dtype("float32"), 4))
    n_comp = ACCESSOR_NUM_COMPONENTS.get(accessor.type, 1)

    arr = np.frombuffer(raw, dtype=dtype).copy()  # .copy() makes it writeable

    if n_comp > 1:
        arr = arr.reshape(-1, n_comp)

    return arr


def read_channel_data(
    gltf: GLTF2, anim_index: int, channel_index: int
) -> tuple[np.ndarray, np.ndarray]:
    """Read the time and value arrays for one animation channel.

    ANIMATION CONCEPT: A glTF animation channel is a pair of accessors:
    INPUT (time keyframes) and OUTPUT (value keyframes at those times).
    The sampler.interpolation field tells us how to interpolate between them
    (LINEAR, STEP, or CUBICSPLINE). This is the same data model as Maya's
    animation curves, Blender's F-Curves, and Unreal's animation sequences —
    they all reduce to {times[], values[], interpolation}.

    Args:
        gltf: The loaded GLTF2 object.
        anim_index: Index into gltf.animations.
        channel_index: Index into gltf.animations[anim_index].channels.

    Returns:
        A tuple (times, values) where:
            times  — float32 array of shape (N,) in seconds.
            values — float32 array of shape (N,) for SCALAR or (N, K) for
                     VEC3/VEC4 (K components per keyframe).
    """
    anim = gltf.animations[anim_index]
    channel = anim.channels[channel_index]
    sampler = anim.samplers[channel.sampler]

    times = read_accessor(gltf, sampler.input).flatten().astype(np.float32)
    values = read_accessor(gltf, sampler.output).astype(np.float32)

    return times, values


def get_node_name(gltf: GLTF2, node_index: int | None) -> str:
    """Return the display name for a glTF node.

    Args:
        gltf: The loaded GLTF2 object.
        node_index: Index into gltf.nodes, or None.

    Returns:
        The node's name field, or a generated fallback if unnamed.
    """
    if node_index is None:
        return "(none)"
    node = gltf.nodes[node_index]
    return node.name or f"Node_{node_index}"


def save_gltf(gltf: GLTF2, path: Path) -> int:
    """Save a modified GLTF2 object to a .glb file.

    Args:
        gltf: The GLTF2 object to save (may have been modified in memory).
        path: Destination filesystem path.

    Returns:
        The size of the written file in bytes.
    """
    gltf.save(str(path))
    return path.stat().st_size


def write_channel_values(gltf: GLTF2, anim_index: int, channel_index: int,
                         new_values: np.ndarray) -> None:
    """Write modified keyframe values back into a GLTF2 object's binary blob.

    PIPELINE CONCEPT: This function is a minimal example of what a professional
    pipeline TD writes when batch-modifying animation data. The pattern is:
    1. Read the raw binary blob from the buffer.
    2. Locate the byte range for the target accessor using bufferView offsets.
    3. Overwrite those bytes with the new float32 data.
    4. Write the entire modified blob back to the gltf object.
    This same read-modify-write pattern applies to texture atlases, LOD meshes,
    collision geometry, and any other binary asset format.

    Args:
        gltf: The GLTF2 object to modify in place.
        anim_index: Index into gltf.animations.
        channel_index: Index into the animation's channels.
        new_values: float32 array of shape (N,) or (N, K) with the new keyframe
                    values. Must match the original shape exactly.
    """
    anim = gltf.animations[anim_index]
    channel = anim.channels[channel_index]
    sampler = anim.samplers[channel.sampler]

    accessor = gltf.accessors[sampler.output]
    buffer_view = gltf.bufferViews[accessor.bufferView]

    # The binary blob is the entire .bin chunk embedded in the .glb container.
    blob = bytearray(gltf.binary_blob())

    # Calculate byte range for the output accessor within the blob.
    byte_offset = buffer_view.byteOffset + (accessor.byteOffset or 0)
    flat = new_values.astype(np.float32).flatten()
    byte_count = flat.nbytes

    # Pack the modified float32 values into bytes and splice them into the blob.
    new_bytes = struct.pack(f"{len(flat)}f", *flat)
    blob[byte_offset: byte_offset + byte_count] = new_bytes

    # Write the modified blob back to the gltf object.
    gltf.set_binary_blob(bytes(blob))


# =============================================================================
# STEP 3 — Scene tree builder
# =============================================================================
#
# Walking the glTF node hierarchy mirrors exactly what Three.js GLTFLoader does
# internally when it reconstructs an Object3D scene graph from the JSON arrays.

def build_scene_tree(gltf: GLTF2, parent_item: QTreeWidgetItem | None,
                     node_index: int, skin_joint_set: set[int]) -> QTreeWidgetItem:
    """Recursively build a QTreeWidget subtree from glTF nodes.

    ANIMATION CONCEPT: In glTF the scene graph is represented as a flat array
    of nodes (gltf.nodes[]) with each node listing its children by index. This
    is the same design as Three.js's Object3D.children array — the hierarchy is
    explicit parent→child, not implicit parent-pointer. Walking it requires the
    same recursive descent that Three.js uses in GLTFLoader._loadNode().

    Args:
        gltf: The loaded GLTF2 object.
        parent_item: The QTreeWidgetItem to attach this node to, or None for
                     the root (attached directly to the QTreeWidget).
        node_index: Index into gltf.nodes for this node.
        skin_joint_set: Set of node indices that are skeleton joints (bones).

    Returns:
        The newly created QTreeWidgetItem for this node.
    """
    node = gltf.nodes[node_index]
    name = node.name or f"Node_{node_index}"
    is_bone = node_index in skin_joint_set
    has_mesh = node.mesh is not None

    # Build the label: name + type badge
    if is_bone:
        label = f"🦴  {name}"
        badge = "bone"
    elif has_mesh:
        label = f"▣  {name}"
        badge = "mesh"
    else:
        label = f"○  {name}"
        badge = "node"

    if parent_item is not None:
        item = QTreeWidgetItem(parent_item, [label, badge, str(node_index)])
    else:
        item = QTreeWidgetItem([label, badge, str(node_index)])

    item.setData(0, Qt.ItemDataRole.UserRole, node_index)

    # Color bones differently from mesh nodes
    if is_bone:
        item.setForeground(0, QColor(PALETTE["accent2"]))
    elif has_mesh:
        item.setForeground(0, QColor(PALETTE["green"]))

    # Recurse into children
    for child_index in (node.children or []):
        build_scene_tree(gltf, item, child_index, skin_joint_set)

    return item


# =============================================================================
# STEP 4 — Animation panel model
# =============================================================================

def get_animation_channels_for_node(
    gltf: GLTF2, node_index: int
) -> list[tuple[int, int, str, str]]:
    """Find all animation channels that target a specific node.

    Args:
        gltf: The loaded GLTF2 object.
        node_index: The node to search for.

    Returns:
        A list of tuples: (anim_index, channel_index, anim_name, target_path)
        sorted by animation index then channel index.
    """
    results: list[tuple[int, int, str, str]] = []
    for anim_idx, anim in enumerate(gltf.animations or []):
        anim_name = anim.name or f"Animation_{anim_idx}"
        for ch_idx, channel in enumerate(anim.channels or []):
            if channel.target.node == node_index:
                results.append((anim_idx, ch_idx, anim_name, channel.target.path))
    return results


# =============================================================================
# STEP 5 — PyQtGraph curve editor widget
# =============================================================================

class CurveEditorWidget(QWidget):
    """A PyQtGraph-backed animation curve editor panel.

    THREE.JS CONCEPT: This widget is the Python equivalent of Three.js's built-in
    animation system visualised. In Three.js you call mixer.clipAction(clip).play()
    and the engine interpolates curves internally. Here we expose the same curves
    (time → value per component) and let the user see and edit them directly.

    ANIMATION CONCEPT: The Graph Editor (Maya) and F-Curve Editor (Blender) are
    the primary instruments of professional keyframe animation. They show exactly
    what this widget shows: time on the X axis, animated value on the Y axis, one
    coloured curve per channel component. Understanding the curve shape — ease-in,
    ease-out, overshoot — is core animation literacy.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialise the curve editor with an empty plot."""
        super().__init__(parent)

        # PYQTGRAPH CONCEPT: pg.setConfigOptions must be called before any
        # PlotWidget is created. antialias=True enables subpixel line rendering
        # for smooth curves. background='default' defers to our stylesheet.
        pg.setConfigOptions(antialias=True, background=None)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header bar: channel toggles + stats label
        header = QWidget()
        header.setStyleSheet(f"background-color: {PALETTE['surface']};")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(12, 6, 12, 6)

        self._channel_label = QLabel("No node selected")
        self._channel_label.setStyleSheet(f"color: {PALETTE['muted']}; font-size: 12px;")
        header_layout.addWidget(self._channel_label)
        header_layout.addStretch()

        # Per-component visibility toggles (X Y Z W)
        self._toggles: list[QCheckBox] = []
        for i, (lbl, col) in enumerate(zip(["X", "Y", "Z", "W"], CURVE_COLORS)):
            cb = QCheckBox(lbl)
            cb.setChecked(True)
            r, g, b, _ = col
            cb.setStyleSheet(f"""
                QCheckBox {{ color: rgb({r},{g},{b}); font-size: 12px; }}
                QCheckBox::indicator:checked {{ background-color: rgb({r},{g},{b});
                                                border-color: rgb({r},{g},{b}); }}
            """)
            cb.stateChanged.connect(lambda _, idx=i: self._toggle_curve(idx))
            self._toggles.append(cb)
            header_layout.addWidget(cb)

        layout.addWidget(header)

        # Divider line
        div = QFrame()
        div.setFrameShape(QFrame.Shape.HLine)
        div.setStyleSheet(f"background-color: {PALETTE['border']}; max-height: 1px;")
        layout.addWidget(div)

        # PYQTGRAPH CONCEPT: PlotWidget wraps a PlotItem inside a QGraphicsView.
        # It handles all zoom/pan interaction via mouse (right-drag to zoom,
        # middle-drag or ctrl+drag to pan). setMenuEnabled(False) removes the
        # right-click context menu to keep the UI clean.
        self._plot = pg.PlotWidget()
        self._plot.setMenuEnabled(False)
        self._plot.showGrid(x=True, y=True, alpha=0.15)
        self._plot.setLabel("bottom", "Time", units="s",
                            color=PALETTE["muted"], size="10pt")
        self._plot.setLabel("left", "Value",
                            color=PALETTE["muted"], size="10pt")
        self._plot.getAxis("bottom").setPen(pg.mkPen(color=PALETTE["border"]))
        self._plot.getAxis("left").setPen(pg.mkPen(color=PALETTE["border"]))
        self._plot.getAxis("bottom").setTextPen(pg.mkPen(color=PALETTE["muted"]))
        self._plot.getAxis("left").setTextPen(pg.mkPen(color=PALETTE["muted"]))
        self._plot.setBackground(PALETTE["surface"])

        layout.addWidget(self._plot, stretch=1)

        # State
        self._curves: list[pg.PlotDataItem] = []
        self._scatter_items: list[pg.ScatterPlotItem] = []
        self._playhead: pg.InfiniteLine | None = None
        self._times: np.ndarray | None = None
        self._values: np.ndarray | None = None
        self._anim_index: int = -1
        self._channel_index: int = -1
        self._gltf_ref: GLTF2 | None = None
        self._on_modified_cb: callable | None = None

    def set_modified_callback(self, cb: callable) -> None:
        """Register a callback invoked when the user edits a keyframe point.

        Args:
            cb: Callable with signature cb(anim_index, channel_index, new_values).
        """
        self._on_modified_cb = cb

    def load_channel(self, gltf: GLTF2, anim_index: int, channel_index: int,
                     node_name: str, target_path: str, anim_name: str) -> None:
        """Plot the animation curves for one channel.

        PYQTGRAPH CONCEPT: PlotDataItem is the core curve primitive in PyQtGraph.
        Each component (X, Y, Z) gets its own PlotDataItem so we can toggle
        visibility, change colour, and update data independently. ScatterPlotItem
        overlays the keyframe points as draggable dots — one scatter item per
        component, with sigClicked for direct keyframe editing.

        Args:
            gltf: The loaded GLTF2 object.
            anim_index: Index into gltf.animations.
            channel_index: Index into the animation's channels.
            node_name: Display name of the target node (for the header label).
            target_path: The animated property ('translation', 'rotation', etc.).
            anim_name: Name of the parent animation (for the header label).
        """
        self._gltf_ref = gltf
        self._anim_index = anim_index
        self._channel_index = channel_index

        times, values = read_channel_data(gltf, anim_index, channel_index)
        self._times = times
        self._values = values if values.ndim > 1 else values[:, np.newaxis]

        # Header
        duration = float(times[-1]) if len(times) > 0 else 0.0
        n_keys = len(times)
        labels = COMPONENT_LABELS.get(target_path, ["V0", "V1", "V2", "V3"])
        self._channel_label.setText(
            f"{node_name}  ·  {target_path}  ·  {anim_name}  ·  "
            f"{n_keys} keyframes  ·  {duration:.3f}s"
        )

        self._clear_plot()

        n_components = self._values.shape[1]

        for i in range(n_components):
            color = CURVE_COLORS[i % len(CURVE_COLORS)]
            pen = pg.mkPen(color=color[:3], width=2)

            # Curve line
            curve = self._plot.plot(
                times, self._values[:, i],
                pen=pen, name=labels[i] if i < len(labels) else f"V{i}"
            )
            self._curves.append(curve)

            # Keyframe scatter points — clickable for direct editing
            scatter = pg.ScatterPlotItem(
                x=times, y=self._values[:, i],
                size=9, pen=pg.mkPen(color=color[:3], width=1),
                brush=pg.mkBrush(color=(*color[:3], 180)),
                hoverable=True, hoverPen=pg.mkPen("w", width=2),
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

            # Show/hide per toggle state
            visible = self._toggles[i].isChecked() if i < len(self._toggles) else True
            curve.setVisible(visible)
            scatter.setVisible(visible)

        # Playhead (vertical InfiniteLine at t=0)
        self._playhead = pg.InfiniteLine(
            pos=0.0, angle=90,
            pen=pg.mkPen(color=PALETTE["orange"], width=1, style=Qt.PenStyle.DashLine),
            movable=False
        )
        self._plot.addItem(self._playhead)

        # Fit view to data
        self._plot.autoRange()

    def set_playhead(self, t: float) -> None:
        """Move the playhead to time t.

        Args:
            t: Time in seconds.
        """
        if self._playhead is not None:
            self._playhead.setPos(t)

    def _clear_plot(self) -> None:
        """Remove all existing curves, scatter items and the playhead."""
        self._plot.clear()
        self._curves.clear()
        self._scatter_items.clear()
        self._playhead = None

    def clear(self) -> None:
        """Clear the plot and reset the header label."""
        self._clear_plot()
        self._channel_label.setText("No node selected")
        self._times = None
        self._values = None

    def _toggle_curve(self, component_index: int) -> None:
        """Show or hide the curve and scatter for one component.

        Args:
            component_index: 0=X, 1=Y, 2=Z, 3=W.
        """
        visible = self._toggles[component_index].isChecked()
        if component_index < len(self._curves):
            self._curves[component_index].setVisible(visible)
        if component_index < len(self._scatter_items):
            self._scatter_items[component_index].setVisible(visible)

    def _on_scatter_clicked(self, scatter: pg.ScatterPlotItem,
                            points: list[pg.SpotItem]) -> None:
        """Handle a click on a keyframe scatter point.

        ANIMATION CONCEPT: Direct keyframe editing on the curve is the primary
        workflow in Maya's Graph Editor and Blender's F-Curve Editor. The user
        selects a keyframe point and modifies its value — exactly what we do here.
        The change is written directly into the glTF binary blob in memory; calling
        File → Export saves the modified file to disk.

        Args:
            scatter: The scatter item that was clicked.
            points: List of SpotItem objects under the cursor (may be multiple
                    if keyframes overlap in time).
        """
        if not points or self._values is None or self._gltf_ref is None:
            return

        point = points[0]
        component_index, keyframe_index = point.data()

        # Prompt: we flip the sign of the value as a simple demonstration.
        # In a production tool this would open an inline value editor.
        original = float(self._values[keyframe_index, component_index])
        new_val = -original  # mirror the value

        # Mutate in-place
        self._values[keyframe_index, component_index] = new_val

        # Write back to glTF binary blob
        if self._on_modified_cb:
            self._on_modified_cb(self._anim_index, self._channel_index,
                                 self._values)

        # Redraw the affected curve and scatter
        comp = component_index
        if comp < len(self._curves):
            self._curves[comp].setData(self._times, self._values[:, comp])
        if comp < len(self._scatter_items):
            self._scatter_items[comp].setData(
                x=self._times, y=self._values[:, comp],
                data=[(comp, j) for j in range(len(self._times))]
            )


# =============================================================================
# STEP 6 — Keyframe inspector table
# =============================================================================

class KeyframeInspectorWidget(QWidget):
    """Right-panel inspector: shows the keyframe table for the selected channel.

    ANIMATION CONCEPT: Professional DCC tools always show two views of the same
    animation data simultaneously: a graph view (curves) and a spreadsheet view
    (exact numeric values per keyframe). The spreadsheet view is used for
    precision edits — nudging a value by 0.001 that would be invisible on the
    curve. This widget provides that second view.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialise the inspector with an empty keyframe table."""
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        hdr = QLabel("KEYFRAME TABLE")
        hdr.setObjectName("section_header")
        layout.addWidget(hdr)

        self._table = QTableWidget(0, 5)  # Time | X | Y | Z | W
        self._table.setHorizontalHeaderLabels(["Time", "X / W0", "Y / W1", "Z / W2", "W / W3"])
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self._table.verticalHeader().setDefaultSectionSize(22)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        layout.addWidget(self._table, stretch=1)

        # Stats below the table
        self._stats_label = QLabel("")
        self._stats_label.setStyleSheet(
            f"color: {PALETTE['muted']}; font-size: 11px;"
            "font-family: 'IBM Plex Mono', monospace; padding: 4px 0;"
        )
        self._stats_label.setWordWrap(True)
        layout.addWidget(self._stats_label)

    def load_channel(self, times: np.ndarray, values: np.ndarray,
                     target_path: str) -> None:
        """Populate the table with keyframe data.

        Args:
            times: float32 array of shape (N,) — time values in seconds.
            values: float32 array of shape (N,) or (N, K) — keyframe values.
            target_path: The animated property path for column labelling.
        """
        if values.ndim == 1:
            values = values[:, np.newaxis]

        n_keys, n_comp = values.shape
        self._table.setRowCount(n_keys)

        labels = COMPONENT_LABELS.get(target_path, ["V0", "V1", "V2", "V3"])
        col_headers = ["Time"] + labels[:n_comp]
        # Pad to 5 columns (Time + up to 4 components)
        while len(col_headers) < 5:
            col_headers.append("—")
        self._table.setHorizontalHeaderLabels(col_headers)

        for row in range(n_keys):
            t_item = QTableWidgetItem(f"{times[row]:.4f}")
            t_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            t_item.setForeground(QColor(PALETTE["muted"]))
            self._table.setItem(row, 0, t_item)

            for col in range(n_comp):
                val_str = f"{values[row, col]:.5f}"
                item = QTableWidgetItem(val_str)
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                )
                # Colour-code positive / negative values
                if values[row, col] < 0:
                    item.setForeground(QColor(PALETTE["orange"]))
                self._table.setItem(row, col + 1, item)

            # Fill unused columns
            for col in range(n_comp, 4):
                self._table.setItem(row, col + 1, QTableWidgetItem(""))

        # Compute per-component stats
        stats_parts = []
        for col in range(n_comp):
            lbl = labels[col] if col < len(labels) else f"V{col}"
            lo, hi = float(values[:, col].min()), float(values[:, col].max())
            stats_parts.append(f"{lbl}: [{lo:.3f} → {hi:.3f}]")

        duration = float(times[-1]) if n_keys > 0 else 0.0
        self._stats_label.setText(
            f"{n_keys} keyframes  ·  {duration:.3f}s\n" + "  ".join(stats_parts)
        )

    def highlight_row(self, keyframe_index: int) -> None:
        """Scroll to and select the row corresponding to a keyframe.

        Args:
            keyframe_index: Row index to select.
        """
        if 0 <= keyframe_index < self._table.rowCount():
            self._table.selectRow(keyframe_index)
            self._table.scrollToItem(self._table.item(keyframe_index, 0))

    def clear(self) -> None:
        """Remove all rows and reset the stats label."""
        self._table.setRowCount(0)
        self._stats_label.setText("")


# =============================================================================
# STEP 7 — Animation channel list panel
# =============================================================================

class ChannelListWidget(QWidget):
    """Left-side animation channel list for the selected node."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialise the channel list."""
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
        """Register a callback for channel selection.

        Args:
            cb: Called with (anim_index, channel_index, anim_name, target_path).
        """
        self._on_select_cb = cb

    def load_channels(
        self, channels: list[tuple[int, int, str, str]]
    ) -> None:
        """Populate the channel list for the selected node.

        Args:
            channels: List of (anim_index, channel_index, anim_name, target_path)
                      tuples as returned by get_animation_channels_for_node().
        """
        self._tree.clear()
        # Group by animation name
        anim_groups: dict[str, list[tuple[int, int, str, str]]] = {}
        for entry in channels:
            anim_name = entry[2]
            anim_groups.setdefault(anim_name, []).append(entry)

        for anim_name, entries in anim_groups.items():
            group = QTreeWidgetItem([f"▸ {anim_name}"])
            group.setForeground(0, QColor(PALETTE["accent"]))
            group.setFlags(group.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            self._tree.addTopLevelItem(group)

            for anim_idx, ch_idx, _, path in entries:
                icon = {"translation": "→", "rotation": "↻",
                        "scale": "⊡", "weights": "≈"}.get(path, "·")
                child = QTreeWidgetItem([f"  {icon}  {path}"])
                child.setData(0, Qt.ItemDataRole.UserRole, (anim_idx, ch_idx, anim_name, path))
                child.setForeground(0, QColor(PALETTE["text"]))
                group.addChild(child)

            group.setExpanded(True)

    def clear(self) -> None:
        """Remove all items from the channel list."""
        self._tree.clear()

    def _on_item_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        """Dispatch the selection callback when a channel item is clicked.

        Args:
            item: The clicked QTreeWidgetItem.
            _column: Column index (unused).
        """
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if data and self._on_select_cb:
            self._on_select_cb(*data)


# =============================================================================
# STEP 8 — Transport / playback controls
# =============================================================================

class TransportWidget(QWidget):
    """Bottom transport bar: play/pause, stop, loop, speed, timeline scrubber."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialise the transport bar."""
        super().__init__(parent)
        self.setFixedHeight(52)
        self.setStyleSheet(
            f"background-color: {PALETTE['surface']}; "
            f"border-top: 1px solid {PALETTE['border']};"
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(8)

        # Play / Pause toggle
        self._play_btn = QPushButton("▶")
        self._play_btn.setCheckable(True)
        self._play_btn.setFixedWidth(36)
        self._play_btn.setToolTip("Play / Pause  (Space)")
        layout.addWidget(self._play_btn)

        # Stop
        self._stop_btn = QPushButton("■")
        self._stop_btn.setFixedWidth(36)
        self._stop_btn.setToolTip("Stop — rewind to t=0")
        layout.addWidget(self._stop_btn)

        # Loop toggle
        self._loop_btn = QPushButton("⟳")
        self._loop_btn.setCheckable(True)
        self._loop_btn.setChecked(True)
        self._loop_btn.setFixedWidth(36)
        self._loop_btn.setToolTip("Toggle loop")
        layout.addWidget(self._loop_btn)

        layout.addSpacing(8)

        # Speed multiplier label
        speed_lbl = QLabel("Speed:")
        speed_lbl.setStyleSheet(f"color: {PALETTE['muted']}; font-size: 11px;")
        layout.addWidget(speed_lbl)

        # Speed slider: 0.1× to 3.0× in 0.1 steps (stored as int 1–30)
        self._speed_slider = QSlider(Qt.Orientation.Horizontal)
        self._speed_slider.setRange(1, 30)
        self._speed_slider.setValue(10)   # default 1.0×
        self._speed_slider.setFixedWidth(90)
        self._speed_slider.setToolTip("Playback speed multiplier")
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

        # Timeline scrubber
        self._scrubber = QSlider(Qt.Orientation.Horizontal)
        self._scrubber.setRange(0, 10000)   # mapped to [0, duration]
        self._scrubber.setValue(0)
        layout.addWidget(self._scrubber, stretch=1)

        # Time display
        self._time_display = QLabel("0.000 s")
        self._time_display.setObjectName("time_display")
        layout.addWidget(self._time_display)

        # Public signal references (connect in the main window)
        self.play_btn = self._play_btn
        self.stop_btn = self._stop_btn
        self.loop_btn = self._loop_btn
        self.speed_slider = self._speed_slider
        self.scrubber = self._scrubber

    def set_duration(self, duration: float) -> None:
        """Update the scrubber range for a new animation duration.

        Args:
            duration: Animation duration in seconds.
        """
        self._scrubber.setValue(0)
        self._time = 0.0
        self._duration = duration

    def update_time(self, t: float, duration: float) -> None:
        """Update the time display and scrubber position without triggering signals.

        Args:
            t: Current playback time in seconds.
            duration: Total animation duration in seconds.
        """
        self._time_display.setText(f"{t:.3f} s")
        if duration > 0:
            pos = int(t / duration * 10000)
            self._scrubber.blockSignals(True)
            self._scrubber.setValue(pos)
            self._scrubber.blockSignals(False)


# =============================================================================
# STEP 9 — Main application window
# =============================================================================

class GltfCurveEditor(QMainWindow):
    """Main window of the glTF Animation Curve Editor.

    Layout:
        ┌─────────────────────────────────────────────────────────────┐
        │  Toolbar: Open GLB  [filename]              Export GLB      │
        ├───────────────┬─────────────────────────┬───────────────────┤
        │  Scene Tree   │  Curve Editor (PyQtGraph)│  Keyframe Table  │
        │  (QTreeWidget)│  full-width curves panel │  (QTableWidget)  │
        │               │                          │                  │
        ├───────────────┤                          │                  │
        │  Anim Channels│                          │                  │
        │  (per node)   │                          │                  │
        ├───────────────┴─────────────────────────┴───────────────────┤
        │  Transport: ▶ ■ ⟳  Speed ─── Scrubber ──────── 0.000 s     │
        └─────────────────────────────────────────────────────────────┘

    PIPELINE CONCEPT: This window is a minimal version of a professional
    animation pipeline tool. Real-world equivalents include Maya's Graph
    Editor + Outliner, Blender's NLA Editor + Dope Sheet, and MotionBuilder's
    FCurve Editor. All share the same fundamental layout: hierarchy browser
    on the left, curve editor in the center, inspector on the right, transport
    at the bottom.
    """

    def __init__(self) -> None:
        """Initialise the main window with an empty state."""
        super().__init__()
        self.setWindowTitle("glTF Animation Curve Editor — Seminar 04")
        self.resize(1400, 860)

        self._gltf: GLTF2 | None = None
        self._gltf_path: Path | None = None
        self._current_anim_index: int = -1
        self._current_channel_index: int = -1
        self._current_duration: float = 0.0
        self._playback_time: float = 0.0
        self._is_playing: bool = False
        self._dirty: bool = False   # True when unsaved modifications exist

        self._build_toolbar()
        self._build_central()
        self._build_transport()
        self._build_status_bar()
        self._build_timer()

        self.setStyleSheet(APP_STYLESHEET)
        self._show_empty_state()

    # -------------------------------------------------------------------------
    # UI construction helpers
    # -------------------------------------------------------------------------

    def _build_toolbar(self) -> None:
        """Create the top toolbar with Open / Export actions."""
        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        # Open GLB
        open_btn = QPushButton("  Open GLB…")
        open_btn.setObjectName("accent_btn")
        open_btn.setToolTip("Open a .glb file  (Ctrl+O)")
        open_btn.clicked.connect(self._on_open)
        toolbar.addWidget(open_btn)

        toolbar.addSeparator()

        # Filename label (updated after load)
        self._file_label = QLabel("No file loaded")
        self._file_label.setObjectName("")
        toolbar.addWidget(self._file_label)

        # Spacer
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)

        # Export
        export_btn = QPushButton("  Export GLB…")
        export_btn.setToolTip("Export modified file  (Ctrl+S)")
        export_btn.clicked.connect(self._on_export)
        toolbar.addWidget(export_btn)

        # Keyboard shortcuts
        open_action = QAction(self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._on_open)
        self.addAction(open_action)

        save_action = QAction(self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self._on_export)
        self.addAction(save_action)

        space_action = QAction(self)
        space_action.setShortcut("Space")
        space_action.triggered.connect(self._toggle_playback)
        self.addAction(space_action)

    def _build_central(self) -> None:
        """Build the three-column central area using nested QSplitters."""
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Horizontal splitter: left panel | curve editor | right panel
        self._h_splitter = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(self._h_splitter, stretch=1)

        # -- Left panel: scene tree + channel list stacked vertically
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        tree_header = QLabel("SCENE TREE")
        tree_header.setObjectName("section_header")
        tree_header.setContentsMargins(8, 8, 8, 4)
        left_layout.addWidget(tree_header)

        self._scene_tree = QTreeWidget()
        self._scene_tree.setHeaderHidden(True)
        self._scene_tree.setColumnCount(1)
        self._scene_tree.setIndentation(14)
        self._scene_tree.itemClicked.connect(self._on_node_selected)
        left_layout.addWidget(self._scene_tree, stretch=3)

        div = QFrame()
        div.setObjectName("divider")
        div.setFrameShape(QFrame.Shape.HLine)
        left_layout.addWidget(div)

        self._channel_list = ChannelListWidget()
        self._channel_list.set_select_callback(self._on_channel_selected)
        left_layout.addWidget(self._channel_list, stretch=2)

        left_panel.setMinimumWidth(220)
        left_panel.setMaximumWidth(340)
        self._h_splitter.addWidget(left_panel)

        # -- Center: curve editor
        self._curve_editor = CurveEditorWidget()
        self._curve_editor.set_modified_callback(self._on_keyframe_modified)
        self._h_splitter.addWidget(self._curve_editor)

        # -- Right panel: keyframe inspector
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setMinimumWidth(220)
        right_scroll.setMaximumWidth(340)
        right_scroll.setStyleSheet(f"border: none; background: {PALETTE['bg']};")

        self._inspector = KeyframeInspectorWidget()
        right_scroll.setWidget(self._inspector)
        self._h_splitter.addWidget(right_scroll)

        # Splitter proportions: 22% | 56% | 22%
        self._h_splitter.setStretchFactor(0, 22)
        self._h_splitter.setStretchFactor(1, 56)
        self._h_splitter.setStretchFactor(2, 22)

    def _build_transport(self) -> None:
        """Create the bottom transport bar and wire up its controls."""
        self._transport = TransportWidget()
        self.centralWidget().layout().addWidget(self._transport)

        self._transport.play_btn.clicked.connect(self._toggle_playback)
        self._transport.stop_btn.clicked.connect(self._stop_playback)
        self._transport.scrubber.sliderMoved.connect(self._on_scrubber_moved)

    def _build_status_bar(self) -> None:
        """Create the bottom status bar."""
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage(
            "Open a .glb file to begin — Ctrl+O  |  Space: play/pause  |  "
            "Click a bone to inspect its animation curves"
        )

    def _build_timer(self) -> None:
        """Create the playback timer (fires every ~16 ms → ~60 fps)."""
        self._timer = QTimer(self)
        self._timer.setInterval(PLAYBACK_INTERVAL_MS)
        self._timer.timeout.connect(self._on_tick)

    # -------------------------------------------------------------------------
    # Empty / loaded state display
    # -------------------------------------------------------------------------

    def _show_empty_state(self) -> None:
        """Display a prompt in the status bar when no file is loaded."""
        self._scene_tree.clear()
        self._channel_list.clear()
        self._curve_editor.clear()
        self._inspector.clear()

    # -------------------------------------------------------------------------
    # File I/O
    # -------------------------------------------------------------------------

    def _on_open(self) -> None:
        """Show an open-file dialog and load the selected .glb file."""
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
        except (FileNotFoundError, Exception) as exc:
            QMessageBox.critical(self, "Load Error", f"Could not load {path.name}:\n{exc}")
            return

        self._gltf = gltf
        self._gltf_path = path
        self._dirty = False
        self._file_label.setText(f"  {path.name}")

        # Populate the scene tree
        self._populate_scene_tree()
        self._channel_list.clear()
        self._curve_editor.clear()
        self._inspector.clear()

        # Summarise the model in the status bar
        n_nodes = len(gltf.nodes or [])
        n_anims = len(gltf.animations or [])
        n_meshes = len(gltf.meshes or [])
        n_skins = len(gltf.skins or [])
        self._status.showMessage(
            f"{path.name}  ·  {n_nodes} nodes  ·  {n_meshes} meshes  ·  "
            f"{n_skins} skins  ·  {n_anims} animations  ·  "
            f"Click a node in the scene tree to inspect its animation channels"
        )

    def _on_export(self) -> None:
        """Show a save-file dialog and export the modified .glb."""
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
    # Scene tree population
    # -------------------------------------------------------------------------

    def _populate_scene_tree(self) -> None:
        """Fill the scene tree from the loaded gltf.nodes hierarchy."""
        self._scene_tree.clear()
        if self._gltf is None:
            return

        # Collect all node indices that are skeleton joints
        skin_joint_set: set[int] = set()
        for skin in (self._gltf.skins or []):
            skin_joint_set.update(skin.joints or [])

        # Walk each scene root
        for scene in (self._gltf.scenes or []):
            for root_index in (scene.nodes or []):
                item = build_scene_tree(
                    self._gltf, None, root_index, skin_joint_set
                )
                self._scene_tree.addTopLevelItem(item)

        self._scene_tree.expandAll()

    # -------------------------------------------------------------------------
    # Selection handlers
    # -------------------------------------------------------------------------

    def _on_node_selected(self, item: QTreeWidgetItem, _column: int) -> None:
        """Respond to a node being clicked in the scene tree.

        Queries the gltf for all animation channels targeting this node
        and populates the channel list panel.

        Args:
            item: The clicked QTreeWidgetItem.
            _column: Column index (unused).
        """
        node_index: int | None = item.data(0, Qt.ItemDataRole.UserRole)
        if node_index is None or self._gltf is None:
            return

        channels = get_animation_channels_for_node(self._gltf, node_index)
        self._channel_list.load_channels(channels)

        if not channels:
            self._curve_editor.clear()
            self._inspector.clear()
            node_name = get_node_name(self._gltf, node_index)
            self._status.showMessage(
                f"Node '{node_name}' has no animation channels."
            )
        else:
            # Auto-select the first channel
            self._on_channel_selected(*channels[0])

    def _on_channel_selected(self, anim_index: int, channel_index: int,
                              anim_name: str, target_path: str) -> None:
        """Load a channel into the curve editor and inspector.

        Args:
            anim_index: Index into gltf.animations.
            channel_index: Index into the animation's channels.
            anim_name: Name of the animation (for display).
            target_path: The animated property ('translation', etc.).
        """
        if self._gltf is None:
            return

        self._current_anim_index = anim_index
        self._current_channel_index = channel_index

        channel = self._gltf.animations[anim_index].channels[channel_index]
        node_name = get_node_name(self._gltf, channel.target.node)

        # Load curve editor
        self._curve_editor.load_channel(
            self._gltf, anim_index, channel_index,
            node_name, target_path, anim_name
        )

        # Load inspector table
        times, values = read_channel_data(self._gltf, anim_index, channel_index)
        if values.ndim == 1:
            values = values[:, np.newaxis]
        self._inspector.load_channel(times, values, target_path)

        # Set transport duration from this channel's time range
        duration = float(times[-1]) if len(times) > 0 else 0.0
        self._current_duration = duration
        self._transport.set_duration(duration)
        self._playback_time = 0.0
        self._transport.update_time(0.0, duration)

        self._status.showMessage(
            f"'{node_name}'  ·  {target_path}  ·  {anim_name}  ·  "
            f"{len(times)} keyframes  ·  {duration:.3f}s  ·  "
            f"Click ▶ to play  |  Click a keyframe point to mirror its value"
        )

    # -------------------------------------------------------------------------
    # Keyframe modification callback
    # -------------------------------------------------------------------------

    def _on_keyframe_modified(self, anim_index: int, channel_index: int,
                               new_values: np.ndarray) -> None:
        """Write modified keyframe values back to the gltf binary blob.

        Called by CurveEditorWidget when the user edits a keyframe point.

        Args:
            anim_index: Index into gltf.animations.
            channel_index: Index into the animation's channels.
            new_values: Modified float32 value array (N, K).
        """
        if self._gltf is None:
            return

        write_channel_values(self._gltf, anim_index, channel_index, new_values)
        self._dirty = True

        # Refresh the inspector table to show the updated values
        times, values = read_channel_data(self._gltf, anim_index, channel_index)
        channel = self._gltf.animations[anim_index].channels[channel_index]
        self._inspector.load_channel(times, values, channel.target.path)

        self._status.showMessage(
            "Keyframe modified — unsaved changes. Use Ctrl+S to export."
        )

    # -------------------------------------------------------------------------
    # Playback
    # -------------------------------------------------------------------------

    def _toggle_playback(self) -> None:
        """Toggle play / pause state."""
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
        """Stop playback and rewind to t=0."""
        self._is_playing = False
        self._timer.stop()
        self._transport.play_btn.setChecked(False)
        self._transport.play_btn.setText("▶")
        self._playback_time = 0.0
        self._transport.update_time(0.0, self._current_duration)
        self._curve_editor.set_playhead(0.0)

    def _on_tick(self) -> None:
        """Advance playback time by one tick (~16 ms) and update the UI."""
        speed = self._transport.speed_slider.value() / 10.0
        dt = PLAYBACK_INTERVAL_MS / 1000.0 * speed

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

        # Highlight the nearest keyframe in the inspector table
        if (self._gltf is not None and self._current_anim_index >= 0
                and self._current_channel_index >= 0):
            times, _ = read_channel_data(
                self._gltf, self._current_anim_index, self._current_channel_index
            )
            nearest = int(np.argmin(np.abs(times - self._playback_time)))
            self._inspector.highlight_row(nearest)

    def _on_scrubber_moved(self, value: int) -> None:
        """Respond to manual scrubber drag — seek to the dragged time.

        Args:
            value: Scrubber integer position in [0, 10000].
        """
        if self._current_duration <= 0:
            return
        t = value / 10000.0 * self._current_duration
        self._playback_time = t
        self._transport.update_time(t, self._current_duration)
        self._curve_editor.set_playhead(t)


# =============================================================================
# STEP 10 — Entry point
# =============================================================================

def main() -> None:
    """Application entry point.

    Creates the QApplication, applies the dark stylesheet, shows the main
    window and enters the Qt event loop.

    PYTHON CONCEPT: QApplication must be created before any QWidget. It owns
    the event loop (app.exec()) that drives all Qt signals, slots, timers and
    paint events. On macOS, setStyle('Fusion') forces the cross-platform Fusion
    style so the dark stylesheet renders identically on all platforms.
    """
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setApplicationName("glTF Animation Curve Editor")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("TU Ilmenau — AMS Group")

    window = GltfCurveEditor()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
