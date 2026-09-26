"""OCStudio window: sectors on the left, the O-C diagram over the point view and the table in the middle,
ephemeris, series and the shape of O-C on the right."""

from pathlib import Path

import numpy as np
from apps.approximation.logic import evaluate, segment
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QFileDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QProgressBar, QPushButton, QRadioButton, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget,
    QVBoxLayout, QWidget,
)
from splitter.constants import COLORS
from splitter.window import STYLE
from vispy import scene
from vispy.color import Color
from vispy.scene import AxisWidget, visuals

from ocstudio import extrema as X
from ocstudio import oc as O

# label, colour, marker; the four colours pass the colour-blind checks as a set, the markers differ too
SERIES = {
    "primary_min": ("Primary minimum", "#2a78d6", "disc", "●"),
    "secondary_min": ("Secondary minimum", "#eb6834", "square", "■"),
    "max_I": ("Maximum I", "#1baf7a", "triangle_up", "▲"),
    "max_II": ("Maximum II", "#4a3aa7", "diamond", "◆"),
}
DOUBT_RIM = "#fab219"  # status "warning": doubtful cycle number
SHAPES = {"": "none", "line": "line", "parabola": "parabola", "sine": "sine + line"}
METHOD_NAMES = {"poly": "polynomial", "brat": "Brat+", "sym": "symmetric polynomial", "wsl": "wall-supported line",
                "apar": "asymptotic parabola"}
COLS = ["JD", "O−C, d", "σ, d", "min/max", "type", "N", "[N]", "correction", "sector", "method", "state"]


def button(text, slot):
    b = QPushButton(text)
    b.clicked.connect(slot)
    return b


def box(layout, *items):
    for it in items:
        layout.addWidget(it) if isinstance(it, QWidget) else layout.addLayout(it)
    return layout


class Plot(QWidget):
    """VisPy axes with pan and zoom; clicked(x, y) in screen pixels for a click without a drag."""

    clicked = Signal(float, float)

    def __init__(self, xlabel, ylabel):
        super().__init__()
        self.canvas = scene.SceneCanvas(bgcolor=COLORS["plot_bg"], parent=self)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.canvas.native)
        grid = self.canvas.central_widget.add_grid(spacing=0)
        self.title = scene.Label("", color=COLORS["text"], font_size=11)
        grid.add_widget(self.title, row=0, col=1).height_max = 26
        style = dict(text_color=COLORS["text"], axis_color=COLORS["text_dim"], tick_color=COLORS["text_dim"],
                     axis_font_size=9, tick_font_size=8)
        self.yaxis = AxisWidget(orientation="left", axis_label=ylabel, **style)
        self.xaxis = AxisWidget(orientation="bottom", axis_label=xlabel, **style)
        self.yaxis.width_max, self.xaxis.height_max = 80, 46
        grid.add_widget(self.yaxis, row=1, col=0)
        self.view = grid.add_view(row=1, col=1, camera="panzoom")
        self.view.camera.aspect = None
        grid.add_widget(self.xaxis, row=2, col=1)
        grid.add_widget(row=3, col=1).height_max = 16  # room for the axis label
        grid.add_widget(row=1, col=2).width_max = 24  # and for the last tick label
        for a in (self.xaxis, self.yaxis):
            a.link_view(self.view)
        # VisPy relabels an axis only when the camera moves, not when the layout resizes or moves the axis
        # (a maximised window kept the old labels over the stretched axis): relabel before every draw
        self.canvas.events.draw.connect(lambda e: [a._view_changed() for a in (self.xaxis, self.yaxis)], position="first")
        self._press = None
        self.canvas.events.mouse_press.connect(lambda e: setattr(self, "_press", e.pos))
        self.canvas.events.mouse_release.connect(self._release)

    def _release(self, e):
        if e.button == 1 and self._press is not None and np.hypot(*(e.pos - self._press)) < 4:
            self.clicked.emit(float(e.pos[0]), float(e.pos[1]))
        self._press = None

    def set_xlabel(self, text):
        a = self.xaxis.axis  # ponytail: VisPy 0.16 has no setter for the axis label, its private fields
        a._axis_label = a._axis_label_vis.text = text

    def to_screen(self, pts):
        return self.view.scene.node_transform(self.canvas.scene).map(pts)[:, :2]

    def show_range(self, x, y):
        if len(x):
            px, py = np.ptp(x) * 0.04 or 1.0, np.ptp(y) * 0.12 or 1e-3
            self.view.camera.set_range(x=(x.min() - px, x.max() + px), y=(y.min() - py, y.max() + py), z=(0, 0))


class Run(QThread):
    """Processing of the sectors off the GUI thread."""

    progress = Signal(int, int)
    result = Signal(list)
    failed = Signal(str)

    def __init__(self, sectors):
        super().__init__()
        self.sectors = sectors

    def run(self):
        try:
            self.result.emit(X.process_star(self.sectors, lambda i, n: self.progress.emit(i, n)))
        except Exception as e:  # e.g. worker processes that cannot start
            self.failed.emit(f"Processing failed: {e}")


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OCStudio")
        self.setStyleSheet(STYLE)
        self.resize(1500, 920)
        self.folder, self.sectors, self.ext, self.overrides = None, [], [], {}
        self.pick = {}  # {window key: method}: per series the method with the least O-C scatter (oc.best_methods)
        self.manual = None  # (T0, P, refined) typed or refined by the user; None: automatic
        self.auto = self.oc = self.sel = self.run = None
        self.error = ""  # why there is no O-C, shown over the plot
        self.x = self.v = self.shown = np.array([])
        self.shape_params = {}

        # left: star and sectors
        self.star = QLabel("Open a star folder with .tess sectors")
        self.star.setWordWrap(True)
        self.sector_table = QTableWidget(0, 4)
        self.sector_table.setHorizontalHeaderLabels(["Sector", "P, d", "Type", "Points"])
        self.sector_table.verticalHeader().hide()
        self.sector_table.horizontalHeader().setStretchLastSection(True)
        self.sector_table.itemChanged.connect(lambda it: it.column() == 0 and self.recompute())
        self.bar = QProgressBar()
        left = QWidget()
        box(QVBoxLayout(left), button("Open folder…", self.choose_folder), self.star, self.sector_table,
            button("Run", self.start), self.bar)
        left.setFixedWidth(330)

        # middle: O-C over the point view and the table
        self.plot = Plot("JD − 2 457 000", "O − C, d")
        self.bars = visuals.Line(connect="segments", width=1, parent=self.plot.view.scene)  # ±σ under the points
        self.points = visuals.Markers(parent=self.plot.view.scene)
        self.curve = visuals.Line(color=COLORS["text"], width=2, parent=self.plot.view.scene)
        self.ring = visuals.Markers(parent=self.plot.view.scene)
        for v in (self.bars, self.points, self.curve, self.ring):
            v.set_gl_state(depth_test=False)
        self.plot.clicked.connect(self._pick)
        self.lc = Plot("JD − 2 457 000", "−magnitude")
        self.lc_points = visuals.Markers(parent=self.lc.view.scene)
        self.lc_fit = visuals.Line(color=COLORS["t0_marker"], width=2, parent=self.lc.view.scene)
        self.lc_t0 = visuals.Line(color=COLORS["accent"], width=1, parent=self.lc.view.scene)
        for v in (self.lc_points, self.lc_fit, self.lc_t0):
            v.set_gl_state(depth_test=False)
        self.info = QLabel("Click a point on the O−C")
        self.info.setMinimumWidth(260)
        self.info.setWordWrap(True)
        self.exclude_btn = button("Exclude (D)", self.toggle_exclude)
        self.type_box = QComboBox()
        self.type_box.activated.connect(self.set_type)
        self.method_box = QComboBox()
        self.method_box.addItems(list(X.METHODS))
        point = QWidget()
        controls = box(QVBoxLayout(), self.info, self.exclude_btn,
                       box(QHBoxLayout(), button("Correction −1", lambda: self.shift(-1)),
                           button("+1", lambda: self.shift(1))),
                       box(QHBoxLayout(), QLabel("Type:"), self.type_box),
                       box(QHBoxLayout(), self.method_box, button("Refit", self.refit_point)),
                       box(QHBoxLayout(), button("← (←)", lambda: self.step(-1)), button("(→) →", lambda: self.step(1))))
        controls.addStretch()
        box(QHBoxLayout(point), self.lc, controls)
        self.table = QTableWidget(0, len(COLS))
        self.table.setHorizontalHeaderLabels(COLS)
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.currentCellChanged.connect(lambda r, *_: r >= 0 and r != self.sel and self.select(r))
        self.tabs = QTabWidget()
        self.tabs.addTab(point, "Point")
        self.tabs.addTab(self.table, "Table")
        self.tabs.currentChanged.connect(lambda _: self.fill_table())
        middle = QSplitter(Qt.Orientation.Vertical)
        middle.addWidget(self.plot)
        middle.addWidget(self.tabs)
        middle.setSizes([560, 360])

        # right: ephemeris, series, shape, save
        self.t0_edit, self.p_edit = QLineEdit(), QLineEdit()
        for e in (self.t0_edit, self.p_edit):
            e.editingFinished.connect(self._typed)
        self.kb = QLabel()
        self.kb.setWordWrap(True)
        eph = QGroupBox("Ephemeris")
        grid = QGridLayout(eph)
        grid.addWidget(QLabel("T0"), 0, 0)
        grid.addWidget(self.t0_edit, 0, 1)
        grid.addWidget(QLabel("P"), 1, 0)
        grid.addWidget(self.p_edit, 1, 1)
        grid.addLayout(box(QHBoxLayout(), button("Refine", self.refine), button("Auto", self.reset_ephemeris)),
                       2, 0, 1, 2)
        grid.addWidget(self.kb, 3, 0, 1, 2)
        ser = QGroupBox("Series")
        grid = QGridLayout(ser)
        grid.addWidget(QLabel("show"), 0, 1)
        grid.addWidget(QLabel("in fit"), 0, 2)
        self.show_box, self.fit_box = {}, {}
        for r, (k, (label, color, _, glyph)) in enumerate(SERIES.items(), 1):
            name = QLabel(f"<span style='color:{color}'>{glyph}</span> {label}")
            self.show_box[k], self.fit_box[k] = QCheckBox(), QCheckBox()
            self.show_box[k].setChecked(True)
            self.fit_box[k].setChecked(k == "primary_min")
            self.show_box[k].toggled.connect(lambda _: self.redraw())
            self.fit_box[k].toggled.connect(lambda _: self.recompute())
            grid.addWidget(name, r, 0)
            grid.addWidget(self.show_box[k], r, 1)
            grid.addWidget(self.fit_box[k], r, 2)
        self.series_note = QLabel()
        self.series_note.setWordWrap(True)
        self.series_note.setStyleSheet(f"color: {COLORS['text_dim']}")
        grid.addWidget(self.series_note, len(SERIES) + 1, 0, 1, 3)
        self.x_cycles = QCheckBox("X axis: cycle number E")
        self.x_cycles.toggled.connect(lambda _: self.redraw(reset=True))
        grid.addWidget(self.x_cycles, len(SERIES) + 2, 0, 1, 3)
        self.bars_box = QCheckBox("Error bars ±σ (timing error of the fit)")
        self.bars_box.setChecked(True)
        self.bars_box.toggled.connect(lambda _: self.redraw())
        grid.addWidget(self.bars_box, len(SERIES) + 3, 0, 1, 3)
        form = QGroupBox("O−C shape")
        lay = QVBoxLayout(form)
        self.shape_group = QButtonGroup(self)
        for model, label in SHAPES.items():
            rb = QRadioButton(label)
            rb.setProperty("model", model)
            rb.setChecked(model == "")
            self.shape_group.addButton(rb)
            lay.addWidget(rb)
        self.shape_group.buttonToggled.connect(lambda b, on: on and self.redraw())
        self.shape_label = QLabel()
        self.shape_label.setWordWrap(True)
        lay.addWidget(self.shape_label)
        right = QWidget()
        box(QVBoxLayout(right), eph, ser, form, button("Save CSV…", self.save)).addStretch()
        right.setFixedWidth(320)

        central = QWidget()
        box(QHBoxLayout(central), left, middle, right)
        self.setCentralWidget(central)
        QShortcut(QKeySequence("D"), self, self.toggle_exclude)
        QShortcut(QKeySequence(Qt.Key.Key_Left), self, lambda: self.step(-1))
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, lambda: self.step(1))

    def status(self, text):
        self.statusBar().showMessage(text)

    def closeEvent(self, e):
        if self.run:  # the worker processes cannot be interrupted; let them finish
            self.status("Finishing the processing…")
            self.run.wait()
        super().closeEvent(e)

    # ---- folder and sectors
    def choose_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Star folder with .tess sectors")
        if path:
            self.open_folder(path)

    def open_folder(self, path):
        if self.run:
            return self.status("Wait: the sectors are still being processed")
        self.folder = Path(path).resolve()
        self.sectors, self.overrides, self.manual, self.pick = X.find_sectors(self.folder), {}, None, {}
        self.ext, self.auto, self.oc, self.sel = [], None, None, None
        self.error = "" if self.sectors else "No .tess files in the folder"
        self.series_note.setText("")
        self.star.setText(f"<b>{self.folder.name}</b>: {len(self.sectors)} .tess files")
        self._fill_sectors()
        self.start()
        self.redraw()

    def start(self):
        if self.run or not self.sectors:
            return
        self.bar.setRange(0, max(len(self.sectors), 1))
        self.bar.setValue(0)
        self.run = Run(X.find_sectors(self.folder))
        self.run.progress.connect(lambda i, n: self.bar.setValue(i))
        self.run.result.connect(self._processed)
        self.run.failed.connect(self.status)
        self.run.finished.connect(lambda: setattr(self, "run", None))
        self.status("Processing the sectors…")
        self.run.start()

    def _processed(self, sectors):
        self.sectors = sectors
        self._fill_sectors()
        self.pick = {}
        self.recompute()
        # every window was timed by several methods: per series, the one with the least O-C scatter
        methods, self.pick = ({}, {}) if self.auto is None else O.pick_methods(
            self.ext, self.auto, [s.period for s in self.sectors if s.number in self.enabled()], self.overrides,
            self.fit_types())
        if self.pick:
            self.recompute()
        # series far noisier than the minima (spot waves, wide humps) start hidden, so they do not bury the O-C
        noisy = O.noisy_series(self.ext, self.oc) if self.oc else {}
        for k, b in self.show_box.items():
            b.blockSignals(True)
            b.setChecked(k not in noisy)
            b.blockSignals(False)
        hidden = [f"{SERIES[k][0]} (O−C scatter {noisy[k]:.0f}× the minima's)" for k in SERIES if k in noisy]
        minutes = lambda d: f"{d * 1440:.2g}" if d * 1440 < 10 else f"{d * 1440:.0f}"
        better = [f"{SERIES[k][0]}: {METHOD_NAMES[methods[k][0]]} (timing scatter {minutes(methods[k][1])} → "
                  f"{minutes(methods[k][2])} min, off the points {methods[k][3]:.1f} → {methods[k][4]:.1f}× noise)"
                  for k in SERIES if k in methods and k not in noisy]
        self.series_note.setText("<br>".join(
            ([f"Methods used instead of Auto: {'; '.join(better)}."] if better else [])
            + ([f"Hidden: {', '.join(hidden)}. Tick show to see them."] if hidden else [])))
        self.redraw(reset=True)
        ok = sum(not s.error for s in sectors)
        self.status(f"Sectors processed: {ok} of {len(sectors)}")

    def _fill_sectors(self):
        t = self.sector_table
        t.blockSignals(True)
        t.setRowCount(len(self.sectors))
        for r, s in enumerate(self.sectors):
            first = QTableWidgetItem(str(s.number))
            first.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            first.setCheckState(Qt.CheckState.Unchecked if s.error else Qt.CheckState.Checked)
            done = s.x is not None or s.error
            cells = [first, QTableWidgetItem(f"{s.period:.6f}" if s.period else ""), QTableWidgetItem(s.type),
                     QTableWidgetItem(s.error or (f"{len(s.extrema)}/{s.windows}" if done else "…"))]
            for c, it in enumerate(cells):
                it.setToolTip(f"{s.path}\n{s.error}".strip())
                t.setItem(r, c, it)
        t.resizeColumnsToContents()
        t.blockSignals(False)

    def enabled(self):
        return {s.number for r, s in enumerate(self.sectors)
                if not s.error and self.sector_table.item(r, 0).checkState() == Qt.CheckState.Checked}

    def fit_types(self):
        return tuple(k for k, b in self.fit_box.items() if b.isChecked())

    # ---- O-C
    def recompute(self, reset=False):
        key = self.ext[self.sel].key if self.sel is not None else None
        on = self.enabled()
        self.ext = X.extrema_of(self.sectors, on)
        self.auto = self.oc = None
        self.error = ""
        try:
            self.ext, self.auto = O.compute_picked(self.ext, [s.period for s in self.sectors if s.number in on],
                                                   self.pick, self.overrides, self.fit_types())
            self.oc = self.auto
            if self.manual:
                T0, P, refined = self.manual
                self.oc = O.with_ephemeris(self.ext, self.auto, T0, P, self.overrides, self.fit_types())
                if refined:  # «Refine» stays on: every change is refitted like the button did
                    self.oc = O.refit(self.ext, self.oc, self.overrides, self.fit_types())
        except ValueError as e:
            if self.auto and self.manual:  # a wild T0 or P: back to the automatic ephemeris
                self.manual = None
                self.oc = self.auto
            self.error = "" if self.auto else str(e)
            self.status(str(e))
        keys = [e.key for e in self.ext]
        self.sel = keys.index(key) if key in keys else None
        self.redraw(reset)

    def redraw(self, reset=False):
        oc = self.oc
        self.plot.title.text = "" if oc else self.error
        for v in (self.bars, self.points, self.curve, self.ring):
            v.visible = oc is not None
        if oc is None:
            self.x = self.v = self.shown = np.array([])
            self.kb.setText("")
            self.plot.canvas.update()
            return self.show_point()
        jd = np.array([e.jd for e in self.ext])
        self.v = oc.values(jd)
        self.x = oc.E + O.corrections(oc.cls) if self.x_cycles.isChecked() else jd
        self.plot.set_xlabel("E, cycle number" if self.x_cycles.isChecked() else "JD − 2 457 000")
        self.shown = np.isin(oc.cls, [k for k, b in self.show_box.items() if b.isChecked()])
        m = self.shown
        color = np.array([Color(SERIES[k][1]).rgba for k in oc.cls]).reshape(-1, 4)
        face = color.copy()
        face[oc.excluded] = Color(COLORS["plot_bg"]).rgba  # hollow: out of the fit and the shape
        edge = color.copy()
        edge[oc.doubt] = Color(DOUBT_RIM).rgba
        self.points.visible = bool(m.any())
        if m.any():
            self.points.set_data(np.c_[self.x[m], self.v[m]], size=9, face_color=face[m], edge_color=edge[m],
                                 edge_width=np.where(oc.doubt, 2.5, 1.0)[m],
                                 symbol=np.array([SERIES[k][2] for k in oc.cls])[m])
        sig = np.array([e.sigma for e in self.ext])
        b = m & np.isfinite(sig) & self.bars_box.isChecked()
        self.bars.visible = bool(b.any())
        if b.any():
            color[:, 3] = 0.6
            ends = np.repeat(self.v[b], 2) + np.outer(sig[b], [-1, 1]).ravel()  # v - σ, v + σ of every point
            self.bars.set_data(np.c_[np.repeat(self.x[b], 2), ends], color=np.repeat(color[b], 2, axis=0))
        self._draw_shape(oc)
        self.t0_edit.setText(f"{oc.T0:.6f}")
        self.p_edit.setText(f"{oc.P:.8f}")
        self._typed_text = (self.t0_edit.text(), self.p_edit.text())
        k, b = O.control_line(jd - oc.T0, self.v, oc.used)
        self.kb.setText(f"{'Refined' if self.manual and self.manual[2] else 'Manual' if self.manual else 'Automatic'}. Control line O−C = k(JD − T0) + b:\n"
                        f"k = {k:.2e}, b = {b:.2e} d\npoints {len(jd)}: in fit {oc.used.sum()}, "
                        f"excluded {oc.excluded.sum()}, doubtful cycle {oc.doubt.sum()}, "
                        f"imprecise {oc.imprecise.sum()}")
        if reset:
            ok = m & ~oc.excluded if (m & ~oc.excluded).any() else m
            if ok.any():
                lo, hi = np.percentile(self.v[ok], [1, 99])
                self.plot.show_range(self.x[ok], np.array([lo, hi]))
        self.show_point()
        self.fill_table()

    def _draw_shape(self, oc):
        model = self.shape_group.checkedButton().property("model")
        self.curve.visible, self.shape_params = False, {}
        self.shape_label.setText("")
        if not model:
            return
        try:
            self.shape_params, f = O.shape(oc.E, self.v, oc.used, model)
        except Exception as e:  # too few points, or the sine fit gives up
            return self.shape_label.setText(str(e))
        E = np.linspace(oc.E[oc.used].min(), oc.E[oc.used].max(), 500)
        xs = E if self.x_cycles.isChecked() else oc.T0 + oc.P * E
        self.curve.set_data(np.c_[xs, f(E)])
        self.curve.visible = True
        self.shape_label.setText("\n".join(f"{k} = {v:.4g}" if isinstance(v, float) else f"{k}: {v}"
                                           for k, v in self.shape_params.items()))

    def _typed(self):
        if (self.t0_edit.text(), self.p_edit.text()) == getattr(self, "_typed_text", None):
            return
        try:
            T0, P = (float(e.text().replace(",", ".")) for e in (self.t0_edit, self.p_edit))
        except ValueError:
            return self.status("T0 and P must be numbers")
        self.manual = (T0, P, False)
        self.recompute()

    def refine(self):
        if self.oc is None:
            return
        T0, P = self.oc.T0, self.oc.P
        self.manual = (T0, P, True)
        self.recompute()

    def reset_ephemeris(self):
        self.manual = None
        self.recompute()

    # ---- the selected point
    def _pick(self, px, py):
        if not self.shown.any():
            return
        idx = np.flatnonzero(self.shown)
        d = np.hypot(*(self.plot.to_screen(np.c_[self.x[idx], self.v[idx]]) - [px, py]).T)
        if d.min() < 12:
            self.select(int(idx[d.argmin()]))

    def select(self, i):
        self.sel = i
        self.show_point()
        if self.tabs.currentIndex() == 1:
            self.table.blockSignals(True)
            self.table.setCurrentCell(i, 0)
            self.table.blockSignals(False)

    def step(self, d):
        idx = np.flatnonzero(self.shown)
        if idx.size == 0:
            return
        order = idx[np.argsort(self.x[idx], kind="stable")]
        if self.sel is None or self.sel not in order:
            return self.select(int(order[0]))
        self.select(int(order[np.clip(np.flatnonzero(order == self.sel)[0] + d, 0, order.size - 1)]))

    def show_point(self):
        i, oc = self.sel, self.oc
        has = oc is not None and i is not None
        self.ring.visible = has and bool(self.shown[i])
        for v in (self.lc_points, self.lc_fit, self.lc_t0):
            v.visible = has
        if not has:
            self.info.setText("Click a point on the O−C")
            self.lc.canvas.update()
            return
        e = self.ext[i]
        self.ring.set_data(np.array([[self.x[i], self.v[i]]]), size=22, face_color=(0, 0, 0, 0),
                           edge_color=COLORS["text"], edge_width=2)
        s = next(s for s in self.sectors if s.number == e.sector)
        w = e.end - e.start
        sl = slice(max(e.start - w, 0), min(e.end + w + 1, s.x.size))
        x, y = s.x[sl], -s.y[sl]
        self.lc_points.set_data(np.c_[x, y], size=4, face_color=COLORS["text_dim"], edge_color=None)
        seg = s.x[segment(s.x, e.fit.interval, e.fit.wings)]
        xs = np.linspace(*(e.fit.x_range or (seg.min(), seg.max())), 300)
        self.lc_fit.set_data(np.c_[xs, -evaluate(e.fit, xs)])
        self.lc_t0.set_data(np.array([[e.jd, y.min()], [e.jd, y.max()]]))
        self.lc.show_range(x, y)
        flag = O.flags(oc, i)
        self.info.setText(
            f"<b>{SERIES[oc.cls[i]][0]}</b> ({e.kind}), sector {e.sector}, window {e.start}–{e.end}<br>"
            f"JD {e.jd:.6f}, E {oc.E[i]}<br>O−C {self.v[i]:.6f} d = {self.v[i] * 1440:.2f} ± "
            f"{e.sigma * 1440:.2f} min<br>"
            f"method {e.method}; {'excluded' if oc.excluded[i] else 'included'}"
            + (f"<br><span style='color:{COLORS['warning']}'>⚠ {flag}</span>" if flag else ""))
        self.exclude_btn.setText("Include (D)" if oc.excluded[i] else "Exclude (D)")
        self.type_box.clear()
        for k in (O.MIN_TYPES if e.kind == "min" else ("max_I", "max_II")):
            self.type_box.addItem(SERIES[k][0], k)
        self.type_box.setCurrentIndex(self.type_box.findData(oc.cls[i]))
        self.plot.canvas.update()

    def _override(self):
        return self.overrides.setdefault(self.ext[self.sel].key, O.Override())

    def toggle_exclude(self):
        if self.oc is not None and self.sel is not None:
            self._override().excluded = not bool(self.oc.excluded[self.sel])
            self.recompute()

    def shift(self, d):
        if self.oc is not None and self.sel is not None:
            self._override().shift += d
            self.recompute()

    def set_type(self, idx):
        if self.oc is not None and self.sel is not None:
            self._override().cls = self.type_box.itemData(idx)
            self.recompute()

    def refit_point(self):
        if self.oc is None or self.sel is None:
            return
        e = self.ext[self.sel]
        s = next(s for s in self.sectors if s.number == e.sector)
        try:
            new = X.refit(s, e, self.method_box.currentText())
        except Exception as ex:
            return self.status(f"Refit failed: {ex}")
        s.extrema = [new if x.key == e.key else x for x in s.extrema]
        self.recompute()

    # ---- table and CSV
    def fill_table(self):
        if self.tabs.currentIndex() != 1:
            return
        rs = O.rows(self.ext, self.oc) if self.oc else []
        t = self.table
        t.blockSignals(True)
        t.setUpdatesEnabled(False)
        t.setRowCount(len(rs))
        for r, row in enumerate(rs):
            state = ("excluded" if row["excluded"] else "") + (f", {row['flag']}" if row["flag"] else "")
            vals = [f"{row['JD']:.6f}", f"{row['O-C']:.6f}", f"{row['sigma']:.6f}", row["min/max"], row["type"],
                    f"{row['N']:.4f}", str(row["[N]"]), f"{row['correction']:g}", str(row["sector"]), row["method"],
                    state.strip(", ")]
            for c, v in enumerate(vals):
                t.setItem(r, c, QTableWidgetItem(v))
        if self.sel is not None:
            t.setCurrentCell(self.sel, 0)
        t.setUpdatesEnabled(True)
        t.blockSignals(False)

    def save(self):
        if self.oc is None:
            return self.status("No O−C to save")
        name = self.folder.name
        path, _ = QFileDialog.getSaveFileName(self, "Save the O−C table", str(self.folder / f"{name}_oc.csv"),
                                              "CSV (*.csv)")
        if not path:
            return
        path = Path(path)
        eph_path = path.with_name(path.stem.removesuffix("_oc") + "_ephemeris.csv")
        jd = np.array([e.jd for e in self.ext])
        k, b = O.control_line(jd - self.oc.T0, self.v, self.oc.used)
        model = self.shape_group.checkedButton().property("model")
        eph = {"T0": self.oc.T0, "P": self.oc.P, "k": k, "b": b, "fit": " ".join(self.fit_types()),
               "shape": model or "none", **{f"shape {sk}": sv for sk, sv in self.shape_params.items()}}
        try:
            O.write_csv(path, eph_path, O.rows(self.ext, self.oc), eph)
        except OSError as e:
            return self.status(f"Could not save: {e}")
        self.status(f"Saved {path.name} and {eph_path.name}")
