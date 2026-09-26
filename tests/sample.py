"""The whole sample: every star folder -> sectors -> O-C; one PNG per star into tests/out and a summary line.
Run: uv run python tests/sample.py SAMPLE_DIR   (folders of stars at any depth)"""

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocstudio.extrema import SECTOR, extrema_of, find_sectors, process_star  # noqa: E402
from ocstudio import oc as O  # noqa: E402
from ocstudio.oc import C, compute  # noqa: E402

COLOR = {"primary_min": "#2a78d6", "secondary_min": "#eb6834", "max_I": "#1baf7a", "max_II": "#4a3aa7"}


def main(root):
    out = Path(__file__).resolve().parent / "out"
    out.mkdir(exist_ok=True)
    stars = sorted({p.parent.parent if SECTOR.search(p.parent.name + "/") else p.parent
                    for p in Path(root).rglob("*.tess")})
    for star in stars:
        sectors = process_star(find_sectors(star))
        ext = extrema_of(sectors)
        periods = [s.period for s in sectors if not s.error]
        name = f"{star.parent.name}_{star.name}"
        try:
            oc = compute(ext, periods)
        except ValueError as e:
            print(f"{name:22s} {e}")
            continue
        jd = np.array([e.jd for e in ext])
        v = oc.values(jd) * 1440
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for k in C:
            m = (oc.cls == k) & ~oc.excluded
            ax.plot(jd[m], v[m], ".", ms=3, color=COLOR[k], label=f"{k} {m.sum()}")
        ax.plot(jd[oc.excluded], v[oc.excluded], "x", ms=4, color="gray", label=f"excluded {oc.excluded.sum()}")
        ok = ~oc.excluded
        lo, hi = np.percentile(v[ok], [1, 99]) if ok.any() else (-1, 1)
        ax.set_ylim(lo - (hi - lo) * 0.3 - 1, hi + (hi - lo) * 0.3 + 1)
        u = v[oc.used]
        mad = 1.4826 * np.median(np.abs(u - np.median(u))) * 60
        ax.set_title(f"{name}  P = {oc.P:.7f}  secondary at {oc.phase['secondary_min']:.3f}  fit MAD {mad:.0f} s")
        ax.set_xlabel("JD - 2457000")
        ax.set_ylabel("O-C, min")
        ax.legend(fontsize=7, markerscale=2)
        fig.tight_layout()
        fig.savefig(out / f"sample_{name}.png", dpi=70)
        plt.close(fig)
        bad = sum(bool(s.error) for s in sectors)
        print(f"{name:22s} sectors {len(sectors) - bad}/{len(sectors)} extrema {len(ext):5d} "
              f"P {oc.P:.7f} (sectors {np.median(periods):.6f}) doubt {oc.doubt.sum():4d} "
              f"clipped {oc.clipped.sum():3d} imprecise {oc.imprecise.sum():3d} fit {oc.used.sum():4d} MAD {mad:.0f} s"
              f"{' hidden ' + ' '.join(O.noisy_series(ext, oc)) if O.noisy_series(ext, oc) else ''}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
