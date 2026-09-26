"""Self-checks of extrema.py on synthetic .tess sectors. Run: uv run python tests/test_extrema.py"""

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocstudio.extrema import extrema_of, find_sectors, process_star, refit  # noqa: E402
from ocstudio.oc import compute  # noqa: E402

P, T0 = 2.7, 1325.4


def ea(x, rng):
    """EA in magnitudes: primary 0.5 mag, secondary 0.2 mag, eclipse half-width 0.06 P."""
    ph = ((x - T0) / P) % 1
    d = lambda c, depth: depth * np.clip(1 - (np.abs((ph - c + 0.5) % 1 - 0.5) / 0.06) ** 2, 0, None) ** 1.5
    return 10 + d(0, 0.5) + d(0.5, 0.2) + 0.002 * rng.normal(size=x.size)


def write_star(folder, sectors, extra=()):
    rng = np.random.default_rng(0)
    for s in sectors:
        a = 1325 + 27.4 * (s - 1)
        x = np.arange(a, a + 26, 2 / 1440)
        x = x[(x < a + 13) | (x > a + 14)]
        d = Path(folder, f"tess2018206045859-s{s:04d}-0000000140659980-0120-s")
        d.mkdir(parents=True)
        np.savetxt(d / "TIC_1__mag.tess", np.c_[x, ea(x, rng)])
    for name, text in extra:
        Path(folder, name).write_text(text)


def test_find_sectors():
    with tempfile.TemporaryDirectory() as d:
        write_star(d, [3, 1])
        Path(d, "TIC_1_S94_x.tess").write_text("")
        Path(d, "other.tess").write_text("")
        dup = Path(d, "again-s0001-.tess")
        dup.write_text("")
        s = find_sectors(d)
        assert [x.number for x in s] == [1, 1, 3, 94, 1002]  # paths in order: TIC_1_S94, again, other, s0001, s0003
        assert [x.path.name for x in s if x.error] == ["TIC_1__mag.tess"] and "again-s0001-.tess" in s[1].error
        assert find_sectors(Path(d, "nothing")) == []


def test_star_to_oc():
    with tempfile.TemporaryDirectory() as d:
        write_star(d, [1, 2, 28], extra=[("broken-s0005-.tess", "1 2\n3 4\n")])
        done = []
        sectors = process_star(find_sectors(d), progress=lambda i, n: done.append((i, n)), workers=2)
        assert done[-1] == (4, 4)
        bad = [s for s in sectors if s.error]
        assert [s.number for s in bad] == [5] and "too few" in bad[0].error
        ext = extrema_of(sectors)
        assert all(abs(s.period - P) < 0.01 for s in sectors if not s.error)
        oc = compute(ext, [s.period for s in sectors if not s.error])
        assert abs(oc.P - P) < 2e-5, oc.P
        jd = np.array([e.jd for e in ext])
        prim = oc.cls == "primary_min"
        assert prim.sum() >= 20 and np.abs(oc.values(jd)[prim]).max() < 2e-3
        assert set(oc.cls) <= {"primary_min", "secondary_min"}
        sig = np.array([e.sigma for e in ext])
        assert (sig[prim] > 0).all() and np.median(sig[prim]) < 2e-3 and not oc.imprecise.any()  # the fit's error
        assert len(extrema_of(sectors, enabled={1, 2})) < len(ext)
        s = sectors[0]
        e = s.extrema[0]
        for m in ("poly", "brat", "auto"):
            r = refit(s, e, m)
            assert r.key == e.key and abs(r.jd - e.jd) < 5e-3 and 0 < r.sigma < 5e-3, (m, r.jd - e.jd, r.sigma)


def test_batch():
    import main

    with tempfile.TemporaryDirectory() as d:
        star = Path(d, "TIC_1")
        write_star(star, [1, 2])
        main.batch(star)
        rows = Path(star, "TIC_1_oc.csv").read_text().splitlines()
        assert rows[0] == "JD,O-C,sigma,min/max,type,N,[N],correction,sector,method,excluded,flag" and len(rows) > 20
        eph = dict(line.split(",") for line in Path(star, "TIC_1_ephemeris.csv").read_text().splitlines())
        assert abs(float(eph["P"]) - P) < 1e-4 and eph["fit"] == "primary_min"


if __name__ == "__main__":
    tests = [(k, v) for k, v in globals().items() if k.startswith("test_")]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
