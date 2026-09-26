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

    i = int(np.flatnonzero(w.shown)[3])
    px, py = w.plot.to_screen(np.array([[w.x[i], w.v[i]]]))[0]
    w._pick(px, py)
    assert w.sel == i and "сектор" in w.info.text()
    v0 = w.v[i]
    w.toggle_exclude()
    assert w.oc.excluded[i] and w.exclude_btn.text().startswith("Повернути")
    w.toggle_exclude()
    assert not w.oc.excluded[i]
    w.shift(1)
    assert abs(w.v[i] - (v0 - w.oc.P)) < 1e-3
    w.shift(-1)
    w.method_box.setCurrentText("poly")
    w.refit_point()
    assert w.ext[w.sel].method == "poly" and w.sel == i and abs(w.v[i] - v0) < 0.01
    w.method_box.setCurrentText("auto")
    w.refit_point()
    w.step(1)
    assert w.sel != i
    w.step(-1)
    assert w.sel == i

    w.refine()
    assert w.manual is not None
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
    assert w.oc is None and "Менше" in w.plot.title.text
    w.sector_table.item(0, 0).setCheckState(w.sector_table.item(0, 0).checkState().__class__.Checked)
    w.sector_table.item(1, 0).setCheckState(w.sector_table.item(1, 0).checkState().__class__.Checked)

    out = Path(tmp.name, "out_oc.csv")
    QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (str(out), ""))
    w.save()
    head = next(csv.reader(open(out)))
    assert head == ["JD", "O-C", "min/max", "type", "N", "[N]", "correction", "sector", "method", "excluded", "flag"]
    eph = dict(csv.reader(open(out.with_name("out_ephemeris.csv"))))
    assert {"T0", "P", "k", "b", "fit", "shape"} <= set(eph)
    w.grab().save(str(OUT / "gui_window.png"))
    print("GUI smoke ok:", len(w.ext), "extrema,", w.statusBar().currentMessage())
    empty = Path(tmp.name, "empty")
    empty.mkdir()
    w.open_folder(empty)
    assert w.oc is None and w.plot.title.text == "У теці немає файлів .tess" and w.sector_table.rowCount() == 0


if __name__ == "__main__":
    main()
