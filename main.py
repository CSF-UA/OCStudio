"""OCStudio: O-C diagram from a folder of TESS sectors.
uv run main.py [FOLDER]      the window
uv run main.py --batch FOLDER  <star>_oc.csv and <star>_ephemeris.csv into the folder, no window"""

import sys
from pathlib import Path


def batch(folder):
    import numpy as np

    from ocstudio.extrema import extrema_of, find_sectors, process_star
    from ocstudio.oc import compute, control_line, rows, write_csv

    folder = Path(folder)
    sectors = process_star(find_sectors(folder), progress=lambda i, n: print(f"\r{i}/{n} sectors", end="", flush=True))
    print()
    for s in sectors:
        print(f"  sector {s.number}: " + (s.error or f"P {s.period:.6f} {s.type}, {len(s.extrema)}/{s.windows} fitted"))
    ext = extrema_of(sectors)
    oc = compute(ext, [s.period for s in sectors if not s.error])
    jd = np.array([e.jd for e in ext])
    k, b = control_line(jd, oc.values(jd), oc.used)
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

    from PySide6.QtGui import QFont, QFontDatabase
    from PySide6.QtWidgets import QApplication
    from vispy.app import use_app

    use_app("pyside6")
    from ocstudio.window import Window

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
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
