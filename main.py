"""OCStudio: O-C diagram from a folder of TESS sectors.
uv run main.py [FOLDER]      the window
uv run main.py --batch FOLDER  <star>_oc.csv and <star>_ephemeris.csv into the folder, no window"""

import sys
from pathlib import Path


def batch(folder):
    import numpy as np

    from ocstudio.extrema import extrema_of, find_sectors, process_star
    from ocstudio.oc import compute, compute_picked, control_line, pick_methods, rows, write_csv

    folder = Path(folder).resolve()
    sectors = process_star(find_sectors(folder), progress=lambda i, n: print(f"\r{i}/{n} sectors", end="", flush=True))
    print()
    for s in sectors:
        print(f"  sector {s.number}: " + (s.error or f"P {s.period:.6f} {s.type}, {len(s.extrema)}/{s.windows} fitted"))
    ext, periods = extrema_of(sectors), [s.period for s in sectors if not s.error]
    oc = compute(ext, periods)
    methods, pick = pick_methods(ext, oc, periods)  # per series the most precise method, as in the window
    ext, oc = compute_picked(ext, periods, pick)
    for k, (m, a, b) in methods.items():
        print(f"  {k}: {m}, timing scatter {a * 1440:.2g} -> {b * 1440:.2g} min")
    jd = np.array([e.jd for e in ext])
    k, b = control_line(jd - oc.T0, oc.values(jd), oc.used)
    eph = {"T0": oc.T0, "P": oc.P, "k": k, "b": b, "fit": "primary_min"}
    write_csv(folder / f"{folder.name}_oc.csv", folder / f"{folder.name}_ephemeris.csv", rows(ext, oc), eph)
    print(f"T0 {oc.T0:.6f}  P {oc.P:.8f}  {len(ext)} extrema, {int(oc.excluded.sum())} excluded -> {folder.name}_oc.csv")


def main():
    args = [a for a in sys.argv[1:] if a != "--batch"]
    if "--batch" in sys.argv:
        if len(args) != 1:
            sys.exit("usage: uv run main.py --batch FOLDER")
        try:
            return batch(args[0])
        except ValueError as e:
            sys.exit(str(e))

    from PySide6.QtCore import Qt
    from PySide6.QtGui import QFont, QFontDatabase
    from PySide6.QtWidgets import QApplication
    from vispy.app import use_app

    use_app("pyside6")
    from ocstudio.window import Window

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    # light like the style sheet and the plots, also under a dark system theme (Ubuntu): Fusion's palette
    # follows the system colour scheme unless the app asks for the light one
    app.styleHints().setColorScheme(Qt.ColorScheme.Light)
    app.setPalette(app.style().standardPalette())
    font_id = QFontDatabase.addApplicationFont(str(Path(__file__).parent / "fonts" / "inter.ttf"))
    families = QFontDatabase.applicationFontFamilies(font_id) if font_id != -1 else []
    app.setFont(QFont(families[0] if families else "Inter", 10))
    w = Window()
    w.show()
    if args:
        w.open_folder(args[0])
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
