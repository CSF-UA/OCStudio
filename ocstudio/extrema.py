"""A star folder of TESS sectors -> extrema: Splitter Auto cuts every sector into single-extremum windows,
astrolab Auto times the extremum in each. Sectors run in parallel processes."""

import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from apps.approximation.logic import Interval, approximate_all, recompute_result
from splitter.auto import auto_split
from splitter.core import get_data

from ocstudio.oc import Extremum

SECTOR = re.compile(r"[-_][sS](\d{1,4})[-_]")
WINGS = 0.5  # of the window on each side, for the eclipse profile of astrolab Auto
# refit methods of the point view: astrolab order choice and wings
METHODS = {"auto": ({"method": "auto"}, WINGS), "brat": ({"method": "brat", "params": None, "slope": True,
                                                          "dip": True}, WINGS), "poly": ("auto", 0.0)}


@dataclass
class Sector:
    number: int
    path: Path
    x: np.ndarray = None
    y: np.ndarray = None
    period: float = 0.0
    type: str = ""
    windows: int = 0
    extrema: list = field(default_factory=list)
    error: str = ""


def find_sectors(folder):
    """Every .tess under the folder (subfolders too), numbered from its folder or file name (-s0012-, _S12_),
    else 1000 + its place. A second file of the same sector is left out with the reason."""
    out, seen = [], {}
    for i, p in enumerate(sorted(Path(folder).rglob("*.tess"))):
        m = SECTOR.search(f"{p.parent.name}/{p.name}")
        s = Sector(int(m.group(1)) if m else 1000 + i, p)
        if s.number in seen:
            s.error = f"той самий сектор, що й {seen[s.number].name}"
        seen.setdefault(s.number, p)
        out.append(s)
    return sorted(out, key=lambda s: s.number)


def process_sector(s):
    """Read, cut, fit; the reason goes to s.error instead of an exception."""
    if s.error:
        return s
    try:
        s.x, s.y = get_data(str(s.path))
        if s.x.size < 100:
            raise ValueError(f"замало точок ({s.x.size})")
        start, end, kind, info = auto_split(s.x, s.y)
        s.period, s.type, s.windows = float(info["period"]), info["type"], len(start)
        if not start:
            raise ValueError(f"немає вікон: {info['type']}")
        base = float(np.median(s.y))
        with np.errstate(all="ignore"):  # overflows of rejected trial profiles
            fits = approximate_all(s.x, s.y, [Interval(a, b, k) for a, b, k in zip(start, end, kind)],
                                   METHODS["auto"][0], WINGS)
        s.extrema = [Extremum(f.t0, f.kind, abs(f.y_at_t0 - base), s.number, f.interval.start, f.interval.end,
                              f.method, f) for f in fits]
    except Exception as e:  # a broken sector must not stop the star
        s.error = str(e) or type(e).__name__
    return s


def process_star(sectors, progress=None, workers=None):
    """All sectors in parallel; progress(done, total) after each. Returns the sectors in their order."""
    done = {}
    with ProcessPoolExecutor(workers) as pool:
        jobs = {pool.submit(process_sector, s): i for i, s in enumerate(sectors)}
        for f in as_completed(jobs):
            done[jobs[f]] = f.result()
            if progress:
                progress(len(done), len(sectors))
    return [done[i] for i in range(len(sectors))]


def extrema_of(sectors, enabled=None):
    """Extrema of the sectors that worked (and are enabled), sorted by time."""
    return sorted((e for s in sectors if not s.error and (enabled is None or s.number in enabled) for e in s.extrema),
                  key=lambda e: e.jd)


def refit(sector, e, method):
    """The extremum of the same window by another method; ValueError when the fit fails."""
    choice, wings = METHODS[method]
    with np.errstate(all="ignore"):
        f = recompute_result(sector.x, sector.y, e.fit, choice, wings)
    base = float(np.median(sector.y))
    return Extremum(f.t0, f.kind, abs(f.y_at_t0 - base), e.sector, e.start, e.end, f.method, f)
