"""Self-checks of oc.py on synthetic extrema. Run: uv run python tests/test_oc.py"""

import csv
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocstudio.oc import (C, Extremum, Override, compute, control_line, noisy_series, refit, rows,  # noqa: E402
                         shape, with_ephemeris, write_csv)

T0, P = 1325.3, 2.5
YEARS = (1, 2, 3, 27, 28, 61, 62)  # TESS sectors: 2 and 3 years apart


def star(P=P, T0=T0, sectors=YEARS, phase=None, depth=None, noise=2e-4, oc=lambda t: 0.0, seed=0):
    """Extrema of the sectors (27.4 d, 1 d gap in the middle), sorted; truth: types and cycle numbers."""
    phase = phase or {"primary_min": 0.0, "secondary_min": 0.5}
    depth = depth or {"primary_min": 0.5, "secondary_min": 0.3, "max_I": 0.1, "max_II": 0.1}
    rng = np.random.default_rng(seed)
    out = []
    for s in sectors:
        a = 1325 + 27.4 * (s - 1)
        for n in range(int((a - T0) / P) - 1, int((a + 27.4 - T0) / P) + 2):
            for k, ph in phase.items():
                t = T0 + P * (n + ph) + oc(T0 + P * n) + noise * rng.normal()
                if a <= t <= a + 26 and not a + 13 < t < a + 14:
                    kind = "min" if k.endswith("min") else "max"
                    out.append((Extremum(t, kind, depth[k] * (1 + 0.02 * rng.normal()), s, start=len(out),
                                         sigma=noise), k, n))
    out.sort(key=lambda o: o[0].jd)
    return [o[0] for o in out], np.array([o[1] for o in out], object), np.array([o[2] for o in out])


def same_cycles(oc, cls, n):
    """Types right and cycle numbers right up to one offset for all points."""
    return (oc.cls == cls).all() and np.ptp(oc.E - n) == 0


def test_recover_ephemeris():
    ext, cls, n = star()
    oc = compute(ext, [P * (1 + 3e-4)] * 7)
    assert same_cycles(oc, cls, n)
    assert abs(oc.P - P) < 1e-6 and abs((oc.T0 - T0) / P - round((oc.T0 - T0) / P)) * P < 1e-4
    assert not oc.doubt.any() and not oc.excluded.any()
    ocv = oc.values(np.array([e.jd for e in ext]))
    assert np.abs(ocv).max() < 1e-3


def test_half_and_double_period():
    ext, cls, n = star()
    for given in (P / 2, 2 * P, P / 2 * 1.0002):
        oc = compute(ext, [given])
        assert abs(oc.P - P) < 1e-6 and same_cycles(oc, cls, n), given


def test_sectors_disagree():
    ext, cls, n = star()
    for periods in ([P, P / 2], [P / 2, P / 2, P], [P, 2 * P], [0, -1, P]):
        oc = compute(ext, periods)
        assert abs(oc.P - P) < 1e-6 and same_cycles(oc, cls, n), periods


def test_eccentric():
    ext, cls, n = star(phase={"primary_min": 0.0, "secondary_min": 0.43})
    oc = compute(ext, [P])
    assert same_cycles(oc, cls, n) and abs(oc.phase["secondary_min"] - 0.43) < 1e-3
    ocv = oc.values(np.array([e.jd for e in ext]))
    assert np.allclose(ocv[cls == "secondary_min"], (0.43 - 0.5) * P, atol=1e-3)


def test_ew_with_maxima():
    Pw = 0.37
    ph = {"primary_min": 0.0, "max_I": 0.25, "secondary_min": 0.5, "max_II": 0.75}
    d = {"primary_min": 0.3, "secondary_min": 0.29, "max_I": 0.1, "max_II": 0.1}
    ext, cls, n = star(P=Pw, T0=1325.1, phase=ph, depth=d, noise=5e-4)
    oc = compute(ext, [Pw * (1 - 2e-4)])
    assert abs(oc.P - Pw) < 1e-6 and not oc.doubt.any()
    assert np.ptp(oc.E - n) == 0
    # equal minima: which one is primary is a coin toss, but the types stay consistent
    swap = {"primary_min": "secondary_min", "secondary_min": "primary_min", "max_I": "max_II", "max_II": "max_I"}
    assert (oc.cls == cls).all() or (oc.cls == [swap[k] for k in cls]).all()


def test_pulsator_maxima_off_quarter():
    ph = {"primary_min": 0.0, "max_I": 0.15}  # one minimum and one maximum per cycle, max at 0.15
    ext, cls, n = star(P=0.55, phase=ph, noise=1e-3)
    oc = compute(ext, [0.55])
    assert abs(oc.P - 0.55) < 1e-6 and same_cycles(oc, cls, n) and not oc.doubt.any()
    assert abs(oc.phase["max_I"] - 0.15) < 0.01


def test_spot_minima_are_not_secondary():
    ext, cls, n = star(depth={"primary_min": 0.5, "secondary_min": 0.05})
    rng = np.random.default_rng(3)
    spots = [Extremum(T0 + P * (k + 0.85 - 0.0002 * k) + 0.01 * rng.normal(), "min", 0.3, e.sector, start=10**6 + k)
             for k, e in ((round((e.jd - T0) / P), e) for e in ext if e.kind == "min" and e.depth > 0.3)]
    allx = sorted(ext + spots, key=lambda e: e.jd)
    real = np.array([e.start < 10**6 for e in allx])
    oc = compute(allx, [P])
    assert abs(oc.P - P) < 1e-6 and abs(oc.phase["secondary_min"] - 0.5) < 1e-3
    assert (oc.cls[real] == cls).all() and np.ptp(oc.E[real] - n) == 0


def test_ltte_over_gaps():
    ltte = lambda t: 0.05 * np.sin(2 * np.pi * (t - 1325) / 1500)
    ext, cls, n = star(P=1.7, sectors=(1, 2, 14, 15, 27, 40, 41, 55, 68, 69, 80), oc=ltte)
    oc = compute(ext, [1.7])
    assert same_cycles(oc, cls, n)
    ocv = oc.values(np.array([e.jd for e in ext]))
    assert np.ptp(ocv) > 0.08  # the light-time effect is in the O-C, not absorbed by wrong cycle numbers


def test_long_period_ea():
    Pl = 41.3
    ext, cls, n = star(P=Pl, T0=1330, sectors=range(1, 40))
    assert sum(e.jd < ext[0].jd + 30 for e in ext if e.kind == "min") < 3
    oc = compute(ext, [Pl * (1 + 1e-4)])
    assert abs(oc.P - Pl) < 1e-5 and same_cycles(oc, cls, n)


def test_outliers_clipped():
    ext, cls, n = star()
    bad = [i for i in range(len(ext)) if cls[i] == "primary_min"][5:8]
    for i in bad:
        ext[i].jd += 0.02
    oc = compute(ext, [P])
    assert oc.clipped[bad].all() and oc.excluded[bad].all() and oc.clipped.sum() == 3
    assert abs(oc.P - P) < 1e-6
    # returning an outlier: it stays in the fit
    oc2 = compute(ext, [P], {ext[bad[0]].key: Override(excluded=False)})
    assert oc2.used[bad[0]] and not oc2.clipped[bad[0]] and not oc2.excluded[bad[0]] and rows(ext, oc2)[bad[0]]["flag"] == ""
    assert oc2.clipped[bad[1:3]].all()  # the other two stay clipped


def test_doubtful_cycle():
    ext, cls, n = star()
    i = [i for i in range(len(ext)) if cls[i] == "primary_min"][20]
    ext[i].jd += 0.2 * P
    oc = compute(ext, [P])
    assert oc.doubt[i] and oc.excluded[i] and oc.doubt.sum() == 1
    assert rows(ext, oc)[i]["flag"] == "doubtful cycle"


def test_imprecise_timing():
    ext, cls, n = star()
    prim = [i for i in range(len(ext)) if cls[i] == "primary_min"]
    i, j, k = prim[10], prim[11], [i for i in range(len(ext)) if cls[i] == "secondary_min"][4]
    ext[i].sigma, ext[j].sigma, ext[k].sigma = 10 * ext[i].sigma, float("inf"), float("nan")
    ext[i].jd += 0.004  # 20 sigma of the others, but its own error is as large: it must not pull the fit
    oc = compute(ext, [P])
    assert oc.imprecise[i] and oc.imprecise[j] and not oc.imprecise[k] and oc.imprecise.sum() == 2
    assert oc.excluded[[i, j]].all() and not oc.used[[i, j]].any() and not oc.clipped[i]
    assert rows(ext, oc)[i]["flag"] == "imprecise timing" and rows(ext, oc)[i]["sigma"] == ext[i].sigma
    back = compute(ext, [P], {ext[i].key: Override(excluded=False)})  # the user's choice wins
    assert back.used[i] and not back.excluded[i] and back.imprecise[i]


def test_noisy_series_hidden():
    ph = {"primary_min": 0.0, "max_I": 0.25, "secondary_min": 0.5, "max_II": 0.75}
    ext, cls, n = star(phase=ph)
    oc = compute(ext, [P])
    assert noisy_series(ext, oc) == {}
    rng = np.random.default_rng(3)
    for e, c in zip(ext, cls):
        if c.startswith("max"):  # spot waves: humps that wander by 0.03 P
            e.jd += 0.03 * P * rng.normal()
    oc = compute(ext, [P])
    noisy = noisy_series(ext, oc)
    assert set(noisy) == {"max_I", "max_II"} and min(noisy.values()) > 20, noisy
    assert abs(oc.P - P) < 1e-6


def test_table_follows_the_guide():
    ext, cls, n = star(phase={"primary_min": 0.0, "secondary_min": 0.47})
    oc = compute(ext, [P])
    for r in rows(ext, oc):
        assert abs(r["O-C"] - oc.P * (r["N"] - r["[N]"] + r["correction"])) < 1e-9
        assert 0 <= r["N"] - r["[N]"] < 1
        frac = r["correction"] % 1
        assert min(abs(frac - (-C[r["type"]] % 1)), 1 - abs(frac - (-C[r["type"]] % 1))) < 1e-9


def test_overrides():
    ext, cls, n = star()
    base = compute(ext, [P])
    i = [i for i in range(len(ext)) if cls[i] == "primary_min"][10]
    j = [i for i in range(len(ext)) if cls[i] == "secondary_min"][4]
    jd = np.array([e.jd for e in ext])
    o = {ext[i].key: Override(excluded=True, shift=1), ext[j].key: Override(cls="primary_min")}
    oc = compute(ext, [P], o)
    assert oc.excluded[i] and not base.excluded[i]
    assert abs(oc.values(jd)[i] - (base.values(jd)[i] - P)) < 1e-5
    assert oc.cls[j] == "primary_min" and oc.cls0[j] == "secondary_min"
    assert abs(abs(oc.values(jd)[j]) - 0.5 * P) < 1e-3  # half a cycle off, as the user asked
    # a secondary turned primary would pull the fit; it is outside the default fit types only if excluded
    oc2 = compute(ext, [P], {ext[i].key: Override(excluded=False)})
    assert not oc2.excluded[i]


def test_manual_ephemeris_and_refit():
    ext, cls, n = star()
    jd = np.array([e.jd for e in ext])
    auto = compute(ext, [P])
    man = with_ephemeris(ext, auto, auto.T0 + 0.001, auto.P + 1e-6)
    assert same_cycles(man, cls, n) and man.T0 == auto.T0 + 0.001
    k, b = control_line(jd - man.T0, man.values(jd), man.used)
    assert abs((man.T0 + b) - auto.T0) < 2e-5 and abs(man.P * (1 + k) - auto.P) < 1e-9
    back = refit(ext, man)
    assert abs(back.P - auto.P) < 1e-9 and abs(back.T0 - auto.T0) < 1e-7
    k, b = control_line(jd - back.T0, back.values(jd), back.used)
    assert abs(k) < 1e-12 and abs(b) < 1e-8
    # refitting with far-off ephemeris: doubt is re-read
    off = with_ephemeris(ext, auto, auto.T0, auto.P + 6e-4)
    assert off.doubt.any() and same_cycles(off, cls, n)
    back2 = refit(ext, off)
    assert not back2.doubt.any() and not back2.excluded.any() and abs(back2.P - auto.P) < 1e-8


def test_wild_ephemeris():
    ext, cls, n = star()
    auto = compute(ext, [P])
    jd = np.array([e.jd for e in ext])
    for T0w, Pw in ((0.0, 1.234), (auto.T0, 3 * P + 0.1)):
        man = with_ephemeris(ext, auto, T0w, Pw)
        k, b = control_line(jd - man.T0, man.values(jd), man.used)  # no crash, nan when nothing is left
        try:
            refit(ext, man)
        except ValueError:
            pass
    for Pw in (0.0, -1.0, float("nan")):
        try:
            with_ephemeris(ext, auto, auto.T0, Pw)
            raise AssertionError(Pw)
        except ValueError:
            pass
    # wild T0/P inputs must raise oc.py's own ValueError, not a library error
    for T0w, Pw in (("abc", P), (None, P), (auto.T0, 1e9), (auto.T0, 1e5), (1e20, P), (auto.T0, 1e-300)):
        try:
            man = with_ephemeris(ext, auto, T0w, Pw)
            try:
                refit(ext, man)
            except ValueError as e:
                assert traceback.extract_tb(e.__traceback__)[-1].filename.endswith("oc.py"), e
        except ValueError as e:
            assert traceback.extract_tb(e.__traceback__)[-1].filename.endswith("oc.py"), e
    # far-off period with inf and None in compute
    assert abs(compute(ext, [float("inf"), P, None]).P - P) < 1e-6


def test_too_few_minima():
    ext, cls, n = star(sectors=(1,))
    mins = [e for e in ext if e.kind == "min"][:2]
    for bad in (mins, [], ext[:0]):
        try:
            compute(bad, [P])
            raise AssertionError
        except ValueError:
            pass
    try:
        compute(ext, [0.0])
        raise AssertionError
    except ValueError:
        pass


def test_fit_types_fallback():
    ext, cls, n = star()
    oc = compute(ext, [P], fit_types=("max_I",))  # no maxima in an EA: all minima are used
    assert abs(oc.P - P) < 1e-6
    oc = compute(ext, [P], fit_types=("primary_min", "secondary_min"))
    assert abs(oc.P - P) < 1e-6


def test_shapes():
    E = np.arange(0, 2000, 7.0)
    y = 3e-9 * E**2 - 1e-6 * E + 2e-4
    params, f = shape(E, y, np.ones(E.size, bool), "parabola")
    assert abs(params["dP/dE"] - 6e-9) < 1e-12 and np.allclose(f(E), y)
    params, f = shape(E, y, np.ones(E.size, bool), "line")
    assert abs(params["a"] - np.polyfit(E, y, 1)[0]) < 1e-12
    try:
        shape(E[:3], y[:3], np.ones(3, bool), "parabola")
        raise AssertionError
    except ValueError:
        pass
    try:
        import apps.oc_curve.logic  # noqa: F401
    except ImportError:
        print("  sine: astrolab is not installed, skipped")
        return
    rng = np.random.default_rng(1)
    y = 1e-6 * E + 0.004 * np.cos(2 * np.pi * (E - 300) / 900) + 2e-4 * rng.normal(size=E.size)
    params, f = shape(E, y, np.ones(E.size, bool), "sine")
    assert params["sine better (BIC)"] and abs(abs(params["A"]) - 0.004) < 5e-4 and abs(params["P2"] - 900) < 30


def test_csv():
    ext, cls, n = star()
    oc = compute(ext, [P])
    with tempfile.TemporaryDirectory() as d:
        a, b = Path(d, "s_oc.csv"), Path(d, "s_ephemeris.csv")
        write_csv(a, b, rows(ext, oc), {"T0": oc.T0, "P": oc.P})
        head = next(csv.reader(open(a)))
        assert head == ["JD", "O-C", "sigma", "min/max", "type", "N", "[N]", "correction", "sector", "method",
                        "excluded", "flag"]
        assert sum(1 for _ in open(a)) == len(ext) + 1
        assert dict(csv.reader(open(b)))["P"] == str(oc.P)


if __name__ == "__main__":
    tests = [(k, v) for k, v in globals().items() if k.startswith("test_")]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
