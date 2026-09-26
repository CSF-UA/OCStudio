"""O-C by the AstroLab guide No. 5: types of extrema, corrections, cycle numbers, ephemeris, shape of O-C.

Every extremum gets a type (primary/secondary minimum, maximum I/II), the guide's correction for it
(0, 0.5, 0.25, 0.75 plus whole cycles) and O-C = JD - (T0 + P (E + c)). Cycle numbers are counted over
a growing time base, so year-long gaps between sectors do not lose count. No files, no GUI here.
"""

import csv
from dataclasses import dataclass, field
from statistics import median_high

import numpy as np
from scipy.stats import mannwhitneyu

C = {"primary_min": 0.0, "max_I": 0.25, "secondary_min": 0.5, "max_II": 0.75}
DOUBT = 0.1  # cycles off the expected phase of the type: the cycle number may be wrong
IMPRECISE = 5.0  # timing error over this many times the median of its type: left out like a doubtful point
NOISY = 5.0  # O-C scatter of a type over this many times that of the minima (and over 1% of P): hidden
BETTER = 0.9  # another method replaces the automatic fit of a type when it cuts the O-C scatter by 10 %+
MIN_TYPES = ("primary_min", "secondary_min")


@dataclass
class Extremum:
    jd: float
    kind: str  # "min" / "max" brightness
    depth: float  # from the sector median, magnitude units of the file
    sector: int
    start: int = 0  # window in the sector
    end: int = 0
    method: str = ""
    fit: object = None  # astrolab FitResult, drawn in the point view
    sigma: float = float("nan")  # 1-sigma error of jd from the fit, days
    alts: dict = field(default_factory=dict, repr=False)  # the same window timed by other methods: method -> Extremum

    @property
    def key(self):
        return self.sector, self.start


@dataclass
class Override:
    excluded: bool | None = None  # None: automatic (outliers and doubtful points are left out)
    shift: int = 0  # whole cycles added to E
    cls: str | None = None


@dataclass
class OC:
    T0: float
    P: float
    phase: dict  # expected phase of every type, measured on the first stretch of data
    cls0: np.ndarray  # automatic type of every extremum
    cls: np.ndarray  # type after the user's changes
    E: np.ndarray
    doubt: np.ndarray
    imprecise: np.ndarray  # the fit's timing error is far over that of its type (or infinite)
    clipped: np.ndarray  # automatic outliers of the ephemeris fit
    excluded: np.ndarray  # left out of the fit and the shape: user choice, else doubtful, imprecise or clipped
    used: np.ndarray  # in the ephemeris fit (and its control line)

    def values(self, jd):
        return jd - (self.T0 + self.P * (self.E + corrections(self.cls)))


def corrections(cls):
    return np.array([C[k] for k in cls], float)


def _phase(u):
    """Circular mean of phases."""
    return float(np.angle(np.exp(2j * np.pi * np.asarray(u)).mean()) / (2 * np.pi) % 1)


def phase_groups(t, P, gap=0.03):
    """Index arrays of the times whose phases lie together on the circle (neighbours closer than `gap`).
    Groups of under 3 points or 8% of the points are stray extrema and left out."""
    u = ((t - t[0]) / P) % 1
    o = np.argsort(u)
    cut = np.flatnonzero(np.diff(np.r_[u[o], u[o][0] + 1]) > gap)
    groups = [o] if cut.size == 0 else [
        o[a + 1 : b + 1] if a < b else np.r_[o[a + 1 :], o[: b + 1]] for a, b in zip(cut, np.roll(cut, -1))]
    return [g for g in groups if g.size >= max(3, 0.08 * t.size)]


def _alternates(t, d, P):
    """Depths differ between even and odd cycles: two different eclipses, so the period is 2P."""
    odd = np.round((t - t[0]) / P).astype(int) % 2 == 1
    a, b = d[~odd], d[odd]
    if min(a.size, b.size) < 3:
        return False
    ma, mb = np.median(a), np.median(b)
    return mannwhitneyu(a, b).pvalue < 1e-3 and abs(ma - mb) > 0.03 * max(ma, mb)


def choose_period(t, d, P):
    """P, P/2 or 2P from the minima of one stretch of data (guide, fig. 5-6)."""
    for _ in range(3):
        g = phase_groups(t, P)
        if len(g) == 4 and len(phase_groups(t, P / 2)) <= 2:
            P /= 2  # four groups of minima that fold into two: P was twice the period
        elif len(g) == 1 and _alternates(t[g[0]], d[g[0]], P):
            P *= 2
        else:
            break
    return P


def _classify(u, kind, phase):
    """Nearest type of the kind, its cycle number and whether the phase is too far off."""
    t = {k: v for k, v in phase.items() if (k in MIN_TYPES) == (kind == "min")}
    k = min(t, key=lambda k: abs((u - t[k] + 0.5) % 1 - 0.5))
    E = int(np.round(u - t[k]))
    return k, E, abs(u - t[k] - E) > DOUBT


def assign(jd, kind, depth, sector, P):
    """Types and cycle numbers of extrema sorted by jd. Returns T0, P, phase, cls, E, doubt."""
    is_min = kind == "min"
    tm = jd[is_min]
    # first stretch: the first max(30 d, 12 P) (from some minimum) with 3+ minima, stretched up to
    # 120 d to hold 40 minima; else all minima
    w = max(30.0, 12 * P)
    first = next((s for s in tm if np.sum((tm >= s) & (tm < s + w)) >= 3), None)
    if first is not None:
        later = tm[tm >= first]
        w = max(w, min(later[min(39, later.size - 1)] - first + 1e-6, 120.0))
    near = (jd >= first) & (jd < first + w) if first is not None else np.ones(jd.size, bool)
    sel = is_min & near
    P = choose_period(jd[sel], depth[sel], P)
    spread = lambda t: 1 - abs(np.exp(2j * np.pi * t / P).mean())
    # the two tightest groups are the eclipses (spot minima wander in phase); the deeper one is primary
    groups = sorted(sorted(phase_groups(jd[sel], P), key=lambda g: spread(jd[sel][g]))[:2],
                    key=lambda g: -np.median(depth[sel][g]))
    T0 = jd[sel][groups[0]].min() if groups else jd[sel][np.argmax(depth[sel])]
    # phases from the primaries' mean phase: the drift of a rough P over the stretch cancels
    rel = lambda t: (_phase((t - T0) / P) - (_phase((jd[sel][groups[0]] - T0) / P) if groups else 0.0)) % 1
    phase = {"primary_min": 0.0, "secondary_min": rel(jd[sel][groups[1]]) if len(groups) > 1 else 0.5}
    phase |= {"max_I": phase["secondary_min"] / 2, "max_II": (1 + phase["secondary_min"]) / 2}
    tx = jd[~is_min & near]
    for g in sorted(phase_groups(tx, P), key=lambda g: -spread(tx[g])) if tx.size else []:
        # maxima: measured phase instead of halfway between minima, the tightest group wins
        ph = rel(tx[g])
        phase["max_I" if ph < phase["secondary_min"] else "max_II"] = ph
    cls = np.empty(jd.size, object)
    E = np.zeros(jd.size, int)
    doubt = np.zeros(jd.size, bool)
    for idx in np.split(np.arange(jd.size), np.flatnonzero((np.diff(jd) > 5) | (np.diff(sector) != 0)) + 1):
        for i in idx:  # growing time base: the ephemeris so far numbers the next sector
            cls[i], E[i], doubt[i] = _classify((jd[i] - T0) / P, kind[i], phase)
        m = is_min & ~doubt & (np.arange(jd.size) <= idx[-1])
        if m.sum() >= 3 and np.ptp(E[m]) > 0:
            T0, P, _ = fit_ephemeris(jd, E + np.array([phase.get(k, 0.0) for k in cls]), m)
    return float(T0), float(P), phase, cls, E, doubt


def fit_ephemeris(jd, x, use, clip=6.0, keep_in=None):
    """Least squares JD = T0 + P x (x = E + correction) over `use`, dropping points > clip MAD off;
    keep_in are never dropped."""
    keep = use.copy()
    force = keep_in if keep_in is not None else np.zeros(len(jd), bool)
    for _ in range(10):
        P, T0 = np.polyfit(x[keep], jd[keep], 1)
        r = jd - (T0 + P * x)
        med = np.median(r[keep])
        mad = 1.4826 * np.median(np.abs(r[keep] - med))  # half the std: a tight core of few points keeps its wings
        new = use & (force | (np.abs(r - med) <= clip * max(mad, 0.5 * np.std(r[keep]), 1e-9)))
        if (new == keep).all() or (new.sum() < 3 and not force.any()):
            break
        if np.ptp(x[new]) == 0:
            break
        keep = new
    return float(T0), float(P), use & ~keep


def _user(ext, overrides):
    return np.array([overrides[e.key].excluded if e.key in overrides else None for e in ext], object)


def _fit_mask(cls, left_out, user, fit_types):
    auto = ~left_out & np.isin(cls, list(fit_types))
    return np.array([a if u is None else (not u and c in fit_types) for a, u, c in zip(auto, user, cls)], bool)


def imprecise(ext, cls):
    """Timing error over IMPRECISE times the median of its type, or infinite (the extremum is at the edge of
    its window); an unknown (NaN) error is not imprecise."""
    sig = np.array([e.sigma for e in ext], float)
    out = np.isposinf(sig)
    for k in set(cls):
        m = (cls == k) & ~np.isnan(sig)
        if m.any():
            out |= m & (sig > IMPRECISE * np.median(sig[m]))
    return out


def _finish(ext, T0, P, phase, cls0, cls, E, doubt, overrides, fit_types, refit):
    jd = np.array([e.jd for e in ext])
    user = _user(ext, overrides)
    imp = imprecise(ext, cls)
    use = _fit_mask(cls, doubt | imp, user, fit_types)
    if use.sum() < 3:  # too few of the chosen types: all minima
        use = _fit_mask(cls, doubt | imp, user, MIN_TYPES)
    clipped = np.zeros(len(ext), bool)
    if refit:
        if use.sum() < 3:
            raise ValueError("Fewer than 3 extrema for the ephemeris")
        x = E + corrections(cls)
        if np.ptp(x[use]) == 0:
            raise ValueError("All fitted points fall in one cycle: check T0 and P")
        keep_in = np.array([u is False for u in user])
        T0, P, clipped = fit_ephemeris(jd, x, use, keep_in=keep_in)
    auto = doubt | imp | clipped
    excluded = np.array([a if u is None else bool(u) for a, u in zip(auto, user)], bool)
    return OC(T0, P, phase, cls0, cls, E, doubt, imp, clipped, excluded, use & ~clipped)


def _numbered(ext, T0, P, phase, cls0, overrides):
    """Types (automatic, else the user's), cycle numbers by rounding plus the user's shifts, doubt."""
    jd = np.array([e.jd for e in ext])
    cls = np.array([overrides[e.key].cls if e.key in overrides and overrides[e.key].cls else c
                    for e, c in zip(ext, cls0)], object)
    u = (jd - T0) / P - np.array([phase[k] for k in cls])
    if not np.all(np.abs(u) < 1e9):
        raise ValueError("T0 or P is off: the cycle numbers are too large")
    E = np.round(u).astype(int)
    doubt = np.abs(u - E) > DOUBT
    E += np.array([overrides[e.key].shift if e.key in overrides else 0 for e in ext], int)
    return cls, E, doubt


def compute(ext, periods, overrides=None, fit_types=("primary_min",)):
    """Everything from the extrema (sorted by jd) and the periods of the sectors."""
    overrides = overrides or {}
    if sum(e.kind == "min" for e in ext) < 3:
        raise ValueError("Fewer than 3 minima: no O−C")
    good = [p for p in periods if p is not None and np.isfinite(p) and p > 0]
    if not good:
        raise ValueError("No sector has a period")
    jd = np.array([e.jd for e in ext])
    arr = lambda f: np.array([getattr(e, f) for e in ext])
    # the larger of two middle periods: a 2x period is repaired from the minima, a half one not always
    T0, P, phase, cls0, E, doubt = assign(jd, arr("kind"), arr("depth"), arr("sector"), median_high(good))
    cls = cls0.copy()
    if any(e.key in overrides for e in ext):
        cls, E2, doubt2 = _numbered(ext, T0, P, phase, cls0, overrides)
        changed = cls != cls0  # user types are re-numbered, the rest keep the growing-base numbers
        shift = np.array([overrides[e.key].shift if e.key in overrides else 0 for e in ext], int)
        E = np.where(changed, E2, E + shift)
        doubt = np.where(changed, doubt2, doubt)
    return _finish(ext, T0, P, phase, cls0, cls, E, doubt, overrides, fit_types, refit=True)


def with_ephemeris(ext, oc, T0, P, overrides=None, fit_types=("primary_min",)):
    """T0 and P typed by the user: cycle numbers by rounding, as in the guide's spreadsheet."""
    try:
        T0, P = float(T0), float(P)
    except (TypeError, ValueError):
        raise ValueError("T0 and P must be numbers") from None
    if not (np.isfinite(T0) and np.isfinite(P) and P > 0):
        raise ValueError("T0 and P must be finite, P > 0")
    overrides = overrides or {}
    cls, E, doubt = _numbered(ext, T0, P, oc.phase, oc.cls0, overrides)
    return _finish(ext, T0, P, oc.phase, oc.cls0, cls, E, doubt, overrides, fit_types, refit=False)


def refit(ext, oc, overrides=None, fit_types=("primary_min",)):
    """'Refine': least squares with the current cycle numbers and types; the doubtful cycle numbers are
    re-read against the refined T0 and P."""
    overrides = overrides or {}
    jd = np.array([e.jd for e in ext])
    base = oc.E - np.array([overrides[e.key].shift if e.key in overrides else 0 for e in ext], int)
    ph = np.array([oc.phase[k] for k in oc.cls])
    doubt = oc.doubt
    for _ in range(3):
        new = _finish(ext, oc.T0, oc.P, oc.phase, oc.cls0, oc.cls, oc.E, doubt, overrides, fit_types, refit=True)
        d = np.abs((jd - new.T0) / new.P - ph - base) > DOUBT
        if (d == doubt).all():
            break
        doubt = d
    return new


def _scatter(v):
    """Interquartile range / 1.349 (the std of a normal distribution): wide when the points jump between two
    phases, not moved by a few stray points."""
    return float(np.subtract(*np.percentile(v, [75, 25]))) / 1.349


def best_methods(ext, oc):
    """Per type, the fitting method whose O-C scatter is the least, among the alternative timings every
    extremum carries (e.alts; a point without the method keeps its own), if BETTER than the automatic fit
    (the scatter of a sample is noisy). The cycle numbers and the ephemeris stay those of oc.
    Returns {type: (method, scatter before, scatter after)} for the types that change."""
    jd = np.array([e.jd for e in ext])
    v = oc.values(jd)
    out = {}
    for k in C:
        m = np.flatnonzero(oc.cls == k)
        if m.size < 5:
            continue
        now = _scatter(v[m])
        best = ("", BETTER * now)
        for meth in sorted({a for i in m for a in ext[i].alts}):
            alt = v[m] + np.array([ext[i].alts[meth].jd - jd[i] if meth in ext[i].alts else 0.0 for i in m])
            if _scatter(alt) < best[1]:
                best = (meth, _scatter(alt))
        if best[0]:
            out[k] = (best[0], now, best[1])
    return out


def compute_picked(ext, periods, pick, overrides=None, fit_types=("primary_min",)):
    """compute() on the automatic timings, then the timings of the methods picked per window ({key: method}):
    a method moves the times, not the types or the cycle count across gaps, so the cycle numbers are read
    against the automatic ephemeris and the ephemeris is refitted. Returns (the extrema, in the order of ext;
    OC)."""
    oc = compute(ext, periods, overrides, fit_types)
    if not pick:
        return ext, oc
    ext = [e.alts.get(pick.get(e.key), e) for e in ext]
    return ext, refit(ext, with_ephemeris(ext, oc, oc.T0, oc.P, overrides, fit_types), overrides, fit_types)


def noisy_series(ext, oc):
    """Types whose O-C scatter is over NOISY times that of the more precise minima and over 1% of P: spot
    waves or flat maxima rather than timings, hidden by default. Primary minima are never noisy.
    The scatter is the interquartile range of all points of the type (/1.349: the std of a normal
    distribution), so a type whose points jump between two phases (the two ends of a flat maximum, one
    of them doubtful and excluded) is wide, while a few stray points are not. {type: scatter / minima's}"""
    v = oc.values(np.array([e.jd for e in ext]))
    iqr = {k: _scatter(v[oc.cls == k]) for k in C if (oc.cls == k).sum() >= 5}
    ref = min((iqr[k] for k in MIN_TYPES if k in iqr), default=None)
    if ref is None:
        return {}
    return {k: s / max(ref, 1e-12) for k, s in iqr.items()
            if k != "primary_min" and s > NOISY * ref and s > 0.01 * oc.P}


def flags(oc, i):
    return ", ".join(f for f, on in (("doubtful cycle", oc.doubt[i]), ("imprecise timing", oc.imprecise[i]),
                                     ("outlier", oc.clipped[i])) if on)


def control_line(x, oc_values, mask):
    """k, b of O-C = k (JD - T0) + b (guide: P_new = P (1 + k), T0_new = T0 + b); ~0 after the fit.
    x = JD - T0."""
    if mask.sum() < 2:
        return float("nan"), float("nan")
    k, b = np.polyfit(x[mask], oc_values[mask], 1)
    return float(k), float(b)


def shape(epoch, oc_values, mask, model):
    """Fit the shape of O-C against the cycle number: 'line', 'parabola' or 'sine' (+ line, BIC).
    Returns (parameters for the panel, curve function of the cycle number)."""
    if mask.sum() < {"line": 3, "parabola": 4, "sine": 7}[model]:
        raise ValueError("Too few points for the O−C shape")
    x, y = epoch[mask].astype(float), oc_values[mask]
    if model in ("line", "parabola"):
        p = np.polyfit(x, y, 1 if model == "line" else 2)
        params = {"a": p[0], "b": p[1]} if model == "line" else {"a": p[0], "b": p[1], "c": p[2], "dP/dE": 2 * p[0]}
        return params, lambda t: np.polyval(p, t)
    from apps.oc_curve.logic import fit_oc_curve, oc_model
    o = np.argsort(x)
    r = fit_oc_curve(x[o], y[o])
    params = {"c0": r.c0, "c1": r.c1, "A": r.amplitude, "P2": r.period, "phase": r.phase,
              "sine better (BIC)": r.sinusoid_preferred}
    return params, lambda t: oc_model(t, r.c0, r.c1, r.amplitude, r.period, r.phase)


def rows(ext, oc):
    """The guide's table: JD, O-C, its error sigma (days), min/max, type, N, [N], correction, sector, method,
    excluded, flag."""
    jd = np.array([e.jd for e in ext])
    N = (jd - oc.T0) / oc.P
    fl = np.floor(N)
    corr = fl - oc.E - corrections(oc.cls)
    ocv = oc.values(jd)
    return [{"JD": e.jd, "O-C": ocv[i], "sigma": e.sigma, "min/max": e.kind, "type": oc.cls[i], "N": N[i], "[N]": int(fl[i]),
             "correction": corr[i], "sector": e.sector, "method": e.method, "excluded": bool(oc.excluded[i]),
             "flag": flags(oc, i)}
            for i, e in enumerate(ext)]


def write_csv(path_oc, path_eph, table, ephemeris):
    """<star>_oc.csv (the table) and <star>_ephemeris.csv (key,value)."""
    with open(path_oc, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]) if table else ["JD"])
        w.writeheader()
        w.writerows(table)
    with open(path_eph, "w", newline="") as f:
        csv.writer(f).writerows(ephemeris.items())
