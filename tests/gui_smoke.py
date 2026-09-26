"""The window under Xvfb: folder -> run -> pick a point -> exclude, shift, type -> refine -> typed and wild
ephemeris -> shapes -> E axis -> table -> sectors off -> CSV -> empty folder. Screenshots go to tests/out.
Run: xvfb-run -a uv run python tests/gui_smoke.py [STAR_FOLDER]"""

import csv
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from PySide6.QtWidgets import QApplication, QFileDialog  # noqa: E402
from vispy.app import use_app  # noqa: E402
from vispy.io import write_png  # noqa: E402

use_app("pyside6")
from ocstudio.window import Window  # noqa: E402
from ocstudio import oc as O  # noqa: E402


def wait(app, w, limit=300):
    t = time.time()
    while w.run is not None:
        app.processEvents()
        time.sleep(0.05)
        assert time.time() - t < limit, "processing hangs"
    app.processEvents()


def main():
    OUT.mkdir(exist_ok=True)
    app = QApplication([])
    tmp = tempfile.TemporaryDirectory()
    if len(sys.argv) > 1:
        folder = sys.argv[1]
    else:
        from test_extrema import write_star
        folder = Path(tmp.name, "TIC_1")
        write_star(folder, [1, 2, 28])
    w = Window()
    w.show()
    w.open_folder(folder)
    wait(app, w)
    assert w.oc is not None and len(w.ext) > 10, w.statusBar().currentMessage()
    n = len(w.ext)
    noisy = O.noisy_series(w.ext, w.oc)
    assert all(w.show_box[k].isChecked() == (k not in noisy) for k in w.show_box)
    assert ("Hidden" in w.series_note.text()) == bool(noisy) and w.bars.visible
    assert ("Methods used" not in w.series_note.text() or w.pick) and all(e.method for e in w.ext)
    w.bars_box.setChecked(False)
    assert not w.bars.visible and w.points.visible
    w.bars_box.setChecked(True)
    assert w.bars.visible
    w.resize(w.width() + 400, w.height() + 100)  # e.g. maximised after loading: the axis labels must follow
    for _ in range(10):
        app.processEvents()
    for plot in (w.plot, w.lc):
        for ax, k in ((plot.xaxis, 0), (plot.yaxis, 1)):
            shown = ax.node_transform(plot.view.scene).map(ax._axis_ends())[:, k]
            assert np.allclose(ax.axis.domain, shown), (ax.axis.domain, shown)

    i = int(np.flatnonzero(w.shown)[3])
    px, py = w.plot.to_screen(np.array([[w.x[i], w.v[i]]]))[0]
    w._pick(px, py)
    assert w.sel == i and "sector" in w.info.text() and " ± " in w.info.text()
    v0, E0 = w.v[i], w.oc.E[i]
    w.toggle_exclude()
    assert w.oc.excluded[i] and w.exclude_btn.text().startswith("Include")
    w.toggle_exclude()
    assert not w.oc.excluded[i]
    w.shift(1)  # (a point you included stays in the fit one cycle off, so the whole O-C may move: check E)
    assert w.oc.E[i] == E0 + 1
    w.shift(-1)
    assert w.oc.E[i] == E0 and abs(w.v[i] - v0) < 1e-6
    w.method_box.setCurrentText("poly")
    w.refit_point()
    # another method may move a noisy maximum by minutes, not out of its cycle
    assert w.ext[w.sel].method == "poly" and w.sel == i and abs(w.v[i] - v0) < 0.1 * w.oc.P
    w.method_box.setCurrentText("sym")
    w.refit_point()
    assert w.ext[w.sel].method == "sym" and w.sel == i and abs(w.v[i] - v0) < 0.1 * w.oc.P
    w.method_box.setCurrentText("auto")
    w.refit_point()
    w.step(1)
    assert w.sel != i
    w.step(-1)
    assert w.sel == i

    w.refine()
    assert w.manual is not None
    c = w.oc.clipped.sum(); u = w.oc.used.sum(); T0r, Pr = w.oc.T0, w.oc.P
    w.toggle_exclude()
    w.toggle_exclude()
    assert w.oc.clipped.sum() == c and w.oc.used.sum() == u and abs(w.oc.P - Pr) < 1e-12 and abs(w.oc.T0 - T0r) < 1e-9 and w.manual[2] is True
    T0, P = w.oc.T0, w.oc.P
    w.t0_edit.setText(f"{T0 + 0.01:.6f}")
    w._typed()
    assert abs(w.oc.T0 - (T0 + 0.01)) < 1e-6 and w.manual
    w.p_edit.setText("-1")
    w._typed()
    assert w.manual is None and w.oc is not None and "P" in w.statusBar().currentMessage()
    w.p_edit.setText("abc")
    w._typed()
    assert w.oc is not None
    w.reset_ephemeris()

    if len(sys.argv) <= 1:  # a period aliasing this star's own P=2.7; not meaningful for an arbitrary star
        w.overrides.clear()  # drop the no-op override left by the exclude toggle above, unrelated to this probe
        w.t0_edit.setText("0")
        w.p_edit.setText("0.9047619")
        w._typed()
        w.refine()
        assert w.manual is None and abs(w.oc.P - w.auto.P) < 1e-12
        w.reset_ephemeris()

    for b in w.shape_group.buttons():
        b.setChecked(True)
        app.processEvents()
        model = b.property("model")
        assert bool(w.shape_params) == bool(model) or model == "sine", (model, w.shape_label.text())
    w.x_cycles.setChecked(True)
    app.processEvents()
    write_png(str(OUT / "gui_oc_E.png"), w.plot.canvas.render())
    w.x_cycles.setChecked(False)
    write_png(str(OUT / "gui_oc.png"), w.plot.canvas.render())
    write_png(str(OUT / "gui_point.png"), w.lc.canvas.render())

    w.tabs.setCurrentIndex(1)
    app.processEvents()
    assert w.table.rowCount() == n and w.table.currentRow() == w.sel
    w.table.setCurrentCell(5, 0)
    assert w.sel == 5
    w.tabs.setCurrentIndex(0)

    w.sector_table.item(0, 0).setCheckState(w.sector_table.item(0, 0).checkState().__class__.Unchecked)
    assert len(w.ext) < n and w.oc is not None
    for r in range(w.sector_table.rowCount()):
        w.sector_table.item(r, 0).setCheckState(w.sector_table.item(r, 0).checkState().__class__.Unchecked)
    assert w.oc is None and "Fewer" in w.plot.title.text
    w.sector_table.item(0, 0).setCheckState(w.sector_table.item(0, 0).checkState().__class__.Checked)
    w.sector_table.item(1, 0).setCheckState(w.sector_table.item(1, 0).checkState().__class__.Checked)

    line_btn = next(b for b in w.shape_group.buttons() if b.property("model") == "line")
    line_btn.setChecked(True)
    app.processEvents()
    out = Path(tmp.name, "out_oc.csv")
    QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (str(out), ""))
    w.save()
    head = next(csv.reader(open(out)))
    assert head == ["JD", "O-C", "sigma", "min/max", "type", "N", "[N]", "correction", "sector", "method", "excluded", "flag"]
    eph = dict(csv.reader(open(out.with_name("out_ephemeris.csv"))))
    assert {"T0", "P", "k", "b", "fit", "shape"} <= set(eph)
    jd = np.array([e.jd for e in w.ext])
    assert abs(float(eph["b"]) - O.control_line(jd - w.oc.T0, w.v, w.oc.used)[1]) < 1e-12
    assert "shape a" in eph
    QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: ("/nonexistent_dir/x_oc.csv", ""))
    w.save()
    assert w.statusBar().currentMessage().startswith("Could not save")
    QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (str(out), ""))
    w.grab().save(str(OUT / "gui_window.png"))
    print("GUI smoke ok:", len(w.ext), "extrema,", w.statusBar().currentMessage())
    empty = Path(tmp.name, "empty")
    empty.mkdir()
    w.open_folder(empty)
    assert w.oc is None and w.plot.title.text == "No .tess files in the folder" and w.sector_table.rowCount() == 0

    junk = Path(tmp.name, "junk")
    junk.mkdir()
    (junk / "x-s0001-.tess").write_text("1 2\n3 4\n")
    w.open_folder(junk)
    wait(app, w)
    assert w.oc is None and w.plot.title.text.startswith("Fewer than 3")

    w.open_folder(folder)
    w.close()
    assert w.run is None or w.run.isFinished()


if __name__ == "__main__":
    main()
