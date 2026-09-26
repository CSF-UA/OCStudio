"""O-C by the AstroLab guide No. 5: types of extrema, corrections, cycle numbers, ephemeris, shape of O-C.

Every extremum gets a type (primary/secondary minimum, maximum I/II), the guide's correction for it
(0, 0.5, 0.25, 0.75 plus whole cycles) and O-C = JD - (T0 + P (E + c)). Cycle numbers are counted over
a growing time base, so year-long gaps between sectors do not lose count. No files, no GUI here.
"""

import csv
from dataclasses import dataclass
from statistics import median_high

import numpy as np
from scipy.stats import mannwhitneyu

C = {"primary_min": 0.0, "max_I": 0.25, "secondary_min": 0.5, "max_II": 0.75}
DOUBT = 0.1  # cycles off the expected phase of the type: the cycle number may be wrong
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
    clipped: np.ndarray  # automatic outliers of the ephemeris fit
    excluded: np.ndarray  # left out of the fit and the shape: user choice, else doubtful or clipped
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


def fit_ephemeris(jd, x, use, clip=6.0):
    """Least squares JD = T0 + P x (x = E + correction) over `use`, dropping points > clip MAD off.
    T0, P, clipped."""
    keep = use.copy()
    for _ in range(10):
        P, T0 = np.polyfit(x[keep], jd[keep], 1)
        r = jd - (T0 + P * x)
        med = np.median(r[keep])
        mad = 1.4826 * np.median(np.abs(r[keep] - med))  # half the std: a tight core of few points keeps its wings
        new = use & (np.abs(r - med) <= clip * max(mad, 0.5 * np.std(r[keep]), 1e-9))
        if (new == keep).all() or new.sum() < 3:
            break
        keep = new
    return float(T0), float(P), use & ~keep


def _user(ext, overrides):
    return np.array([overrides[e.key].excluded if e.key in overrides else None for e in ext], object)


def _fit_mask(cls, doubt, user, fit_types):
    auto = ~doubt & np.isin(cls, list(fit_types))
    return np.array([a if u is None else (not u and c in fit_types) for a, u, c in zip(auto, user, cls)], bool)


def _finish(ext, T0, P, phase, cls0, cls, E, doubt, overrides, fit_types, refit):
    jd = np.array([e.jd for e in ext])
    user = _user(ext, overrides)
    use = _fit_mask(cls, doubt, user, fit_types)
    if use.sum() < 3:  # too few of the chosen types: all minima
        use = _fit_mask(cls, doubt, user, MIN_TYPES)
    clipped = np.zeros(len(ext), bool)
    if refit:
        if use.sum() < 3:
            raise ValueError("Менше 3 екстремумів для ефемериди")
        T0, P, clipped = fit_ephemeris(jd, E + corrections(cls), use)
    excluded = np.array([(d or k) if u is None else bool(u) for d, k, u in zip(doubt, clipped, user)], bool)
    return OC(T0, P, phase, cls0, cls, E, doubt, clipped, excluded, use & ~clipped)


def _numbered(ext, T0, P, phase, cls0, overrides):
    """Types (automatic, else the user's), cycle numbers by rounding plus the user's shifts, doubt."""
    jd = np.array([e.jd for e in ext])
    cls = np.array([overrides[e.key].cls if e.key in overrides and overrides[e.key].cls else c
                    for e, c in zip(ext, cls0)], object)
    u = (jd - T0) / P - np.array([phase[k] for k in cls])
    E = np.round(u).astype(int)
    doubt = np.abs(u - E) > DOUBT
    E += np.array([overrides[e.key].shift if e.key in overrides else 0 for e in ext], int)
    return cls, E, doubt


def compute(ext, periods, overrides=None, fit_types=("primary_min",)):
    """Everything from the extrema (sorted by jd) and the periods of the sectors."""
    overrides = overrides or {}
    if sum(e.kind == "min" for e in ext) < 3:
        raise ValueError("Менше 3 мінімумів: O−C не побудувати")
    good = [p for p in periods if p > 0]
    if not good:
        raise ValueError("Немає періоду в жодному секторі")
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
    if not (np.isfinite(T0) and np.isfinite(P) and P > 0):
        raise ValueError("P має бути додатним числом")
    overrides = overrides or {}
    cls, E, doubt = _numbered(ext, T0, P, oc.phase, oc.cls0, overrides)
    return _finish(ext, T0, P, oc.phase, oc.cls0, cls, E, doubt, overrides, fit_types, refit=False)


def refit(ext, oc, overrides=None, fit_types=("primary_min",)):
    """'Уточнити': least squares with the current cycle numbers and types."""
    return _finish(ext, oc.T0, oc.P, oc.phase, oc.cls0, oc.cls, oc.E, oc.doubt, overrides or {}, fit_types,
                   refit=True)


def control_line(jd, oc_values, mask):
    """k, b of O-C = k JD + b (guide: P_new = P (1 + k), T0_new = T0 + b); ~0 after the fit."""
    if mask.sum() < 2:
        return float("nan"), float("nan")
    k, b = np.polyfit(jd[mask], oc_values[mask], 1)
    return float(k), float(b)


def shape(epoch, oc_values, mask, model):
    """Fit the shape of O-C against the cycle number: 'line', 'parabola' or 'sine' (+ line, BIC).
    Returns (parameters for the panel, curve function of the cycle number)."""
    if mask.sum() < {"line": 3, "parabola": 4, "sine": 7}[model]:
        raise ValueError("Замало точок для форми O−C")
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
    """The guide's table: JD, O-C, min/max, type, N, [N], correction, sector, method, excluded, flag."""
    jd = np.array([e.jd for e in ext])
    N = (jd - oc.T0) / oc.P
    fl = np.floor(N)
    corr = fl - oc.E - corrections(oc.cls)
    ocv = oc.values(jd)
    return [{"JD": e.jd, "O-C": ocv[i], "min/max": e.kind, "type": oc.cls[i], "N": N[i], "[N]": int(fl[i]),
             "correction": corr[i], "sector": e.sector, "method": e.method, "excluded": bool(oc.excluded[i]),
             "flag": "doubtful cycle" if oc.doubt[i] else "outlier" if oc.clipped[i] else ""}
            for i, e in enumerate(ext)]


def write_csv(path_oc, path_eph, table, ephemeris):
    """<star>_oc.csv (the table) and <star>_ephemeris.csv (key,value)."""
    with open(path_oc, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]) if table else ["JD"])
        w.writeheader()
        w.writerows(table)
    with open(path_eph, "w", newline="") as f:
        csv.writer(f).writerows(ephemeris.items())
