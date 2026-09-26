# OCStudio

O−C diagram of a star from its TESS sectors in one window, by the CSF AstroLab guide No. 5
(period calculation and O−C diagrams).

1. **Open folder…** — a star folder with `.tess` files (subfolders too, e.g. one per sector).
   The sector number comes from the folder or file name (`-s0012-`, `_S12_`).
2. Every sector runs on all cores: Splitter Auto cuts it into single-extremum windows,
   astrolab approximation Auto times the extremum in each window. Processing starts by itself when
   the folder is opened (**Run** runs it again).
   Every window is also timed by other methods of astrolab: minima by a polynomial, a symmetric polynomial
   and a wall-supported line, maxima by a symmetric polynomial and an asymptotic parabola (the near-extremum
   functions of MAVKA, Andrych & Andronov 2019). For each series only methods whose curves follow the
   points take part: the rms on the core of the window (beyond half the extremum's height) in units of the
   noise, near the best method's or at the noise level. Of those, the most precise method replaces Auto if
   it cuts the timing scatter by 10 % or more (e.g. flat maxima between eclipses: the symmetric polynomial
   times the centre of the plateau instead of its higher end), or at once if Auto's curves miss the points
   (e.g. Brat+ on flat-bottomed eclipses, where the wall-supported line fits). The timing scatter is taken around the
   local median of the primary minima (their own neighbours, for the primary minima), so real O−C changes
   that move every extremum alike (a third body, a period change) do not hide the difference between
   methods. The series in the fit (primary minima) change method only if the ephemeris fit does not get
   worse. The types and the cycle count come from the Auto timings; the note under **Series** names the
   methods taken, the table the method of every point, and **Refit** times a point by any method. Unlike
   astrolab Auto, which sees one light curve, the choice depends on the O−C of the whole star.
3. The O−C appears: types of extrema (primary/secondary minimum, maximum I/II), the guide's corrections
   (0, 0.5, 0.25, 0.75 plus whole cycles), cycle numbers counted across multi-year gaps, the ephemeris
   by least squares over the series ticked **in fit** (primary minima by default).
4. Check the points: click one to see its window and fit; **D** excludes or returns it, **Correction ±1**
   shifts its cycle, the type and the fitting method can be changed; ←/→ go to the neighbour.
   Hollow markers: points left out automatically (outliers, doubtful cycle, imprecise timing) or by you;
   series without **in fit** are drawn filled but are not fitted either. A yellow rim marks a doubtful cycle
   number. The bars are ±σ, the timing error of each fit (**Error bars** turns them off); a point whose σ
   is over 5× the median of its series (or infinite: the extremum at the edge of its window) is imprecise.
   A series whose O−C scatter is over 5× that of the minima and over 1% of P (spot waves, wide humps
   rather than eclipse timings) starts hidden, with a note under **Series**; tick **show** to see it.
5. T0 and P can be typed. **Refine** refits T0 and P and stays on — later changes are refitted too —
   until you type T0/P or press **Auto**. k and b of the control line O−C = k·(JD − T0) + b should be
   ≈ 0 (guide: P_new = P(1+k), T0_new = T0 + b).
6. Shape of O−C: line, parabola (dP/dE = 2a) or sine + line (astrolab oc_curve, BIC).
7. **Save CSV…** writes `<star>_oc.csv` (`JD,O-C,sigma,min/max,type,N,[N],correction,sector,method,excluded,flag`,
   O−C and its error sigma in days, O−C = P(N − [N] + correction)) and `<star>_ephemeris.csv`.

```bash
uv sync
uv run main.py [STAR_FOLDER]          # the window
uv run main.py --batch STAR_FOLDER    # the CSV files into the folder, no window
```

git must be installed: `uv sync` fetches Splitter and astrolab from GitHub by commit.

Checks: `uv run python tests/test_oc.py`, `tests/test_extrema.py`, `tests/sweep.py` (seed sweep of
the O−C logic, expected `failures 0`), `xvfb-run -a uv run python tests/gui_smoke.py [STAR_FOLDER]`,
`tests/sample.py SAMPLE_DIR` (O−C of every star into `tests/out`).
