"""Seed sweep of oc.py: every scenario on clean data and on harsh data (30% of the extrema lost, 5% stray
minima at random phases, depth scatter), counting wrong periods, types and cycle numbers.
Run: uv run python tests/sweep.py [SEEDS]   (expected: failures 0)"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_oc import P, star  # noqa: E402

from ocstudio.oc import Extremum, compute  # noqa: E402

LTTE = lambda t: 0.05 * np.sin(2 * np.pi * (t - 1325) / 1500)
EW = {"primary_min": 0.0, "max_I": 0.25, "secondary_min": 0.5, "max_II": 0.75}
EWD = {"primary_min": 0.3, "secondary_min": 0.27, "max_I": 0.1, "max_II": 0.1}
SWAP = {"primary_min": "secondary_min", "secondary_min": "primary_min", "max_I": "max_II", "max_II": "max_I"}
SCENARIOS = {  # name: (star() arguments, true P, sector periods)
    "EA": (dict(), P, [P * 1.0003]),
    "EA P/2": (dict(), P, [P / 2]),
    "EA 2P": (dict(), P, [2 * P]),
    "EA mixed": (dict(), P, [P, P / 2, 2 * P, P]),
    "ecc 0.38": (dict(phase={"primary_min": 0.0, "secondary_min": 0.38}), P, [P]),
    "shallow sec": (dict(depth={"primary_min": 0.5, "secondary_min": 0.02}), P, [P / 2]),
    "EW": (dict(P=0.37, T0=1325.1, phase=EW, depth=EWD, noise=8e-4), 0.37, [0.37]),
    "EW 2P": (dict(P=0.37, T0=1325.1, phase=EW, depth=EWD, noise=8e-4), 0.37, [0.74]),
    "LTTE": (dict(P=1.7, sectors=(1, 2, 14, 15, 27, 40, 41, 55, 68, 69, 80), oc=LTTE), 1.7, [1.7]),
    "long P": (dict(P=41.3, T0=1330, sectors=range(1, 40)), 41.3, [41.3 * 1.0001]),
    "noisy EA": (dict(noise=3e-3), P, [P]),
    "prim only": (dict(phase={"primary_min": 0.0}), P, [P]),
    "pulsator": (dict(P=0.55, phase={"primary_min": 0.0, "max_I": 0.15}, noise=1e-3), 0.55, [0.55]),
    "short EA": (dict(P=0.9, noise=1e-3, sectors=(1, 28, 55)), 0.9, [0.9 * 1.0005]),
}


def harsh(ext, cls, n, Ptrue, rng, scatter):
    keep = rng.random(len(ext)) > 0.3
    ext, cls, n = [e for e, k in zip(ext, keep) if k], cls[keep], n[keep]
    for e in ext:
        e.depth *= 1 + scatter * rng.normal()
    junk = [Extremum(e.jd + rng.uniform(0.1, 0.4) * Ptrue, "min", 0.2, e.sector, start=10**6 + i)
            for i, e in enumerate(ext) if rng.random() < 0.05]
    return sorted(ext + junk, key=lambda e: e.jd), cls, n


def failed(name, oc, cls, n, real, Ptrue, max_doubt):
    c, E, d = oc.cls[real], oc.E[real], (oc.doubt | oc.clipped)[real]
    ok = ~d
    same = (c[ok] == cls[ok]).all() or (name.startswith("EW") and (c[ok] == [SWAP[k] for k in cls[ok]]).all())
    cycles = all(np.ptp((E - n)[ok & (c == k)]) == 0 for k in set(c[ok]))
    return abs(oc.P - Ptrue) > 1e-5 * Ptrue or not same or not cycles or d.mean() > max_doubt


def main(seeds=40):
    bad = 0
    for mode, scatter in (("clean", None), ("harsh 5%", 0.05), ("harsh 15%", 0.15)):
        for name, (kw, Ptrue, periods) in SCENARIOS.items():
            fails = []
            for seed in range(seeds):
                ext, cls, n = star(seed=seed, **kw)
                if scatter is not None:
                    ext, cls, n = harsh(ext, cls, n, Ptrue, np.random.default_rng(100 + seed), scatter)
                real = np.array([e.start < 10**6 for e in ext])
                try:
                    oc = compute(ext, periods)
                except ValueError as e:
                    fails.append((seed, str(e)))
                    continue
                if failed(name, oc, cls, n, real, Ptrue, 0.0 if scatter is None else 0.05):
                    fails.append((seed, round(oc.P, 6)))
            bad += len(fails)
            if fails:
                print(f"{mode:10s} {name:12s} {len(fails)}/{seeds} failed, e.g. {fails[:3]}")
    print("failures", bad)
    return bad


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 40) > 0)
