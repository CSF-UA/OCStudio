# OCStudio

O−C diagram of a star from its TESS sectors in one window, by the CSF AstroLab guide No. 5
«Розрахунок періодів та побудова O−C діаграм».

1. **Відкрити теку…** — a star folder with `.tess` files (subfolders too, e.g. one per sector).
   The sector number comes from the folder or file name (`-s0012-`, `_S12_`).
2. Every sector runs on all cores: Splitter Auto cuts it into single-extremum windows,
   astrolab approximation Auto times the extremum in each window.
3. The O−C appears: types of extrema (primary/secondary minimum, maximum I/II), the guide's corrections
   (0, 0.5, 0.25, 0.75 plus whole cycles), cycle numbers counted across multi-year gaps, the ephemeris
   by least squares over the series ticked «в підгонці» (primary minima by default).
4. Check the points: click one to see its window and fit; **D** excludes or returns it, «Корекція ±1»
   shifts its cycle, the type and the fitting method can be changed; ←/→ go to the neighbour.
   Hollow markers are out of the fit (excluded, outliers, doubtful cycle); a yellow rim marks a doubtful
   cycle number.
5. T0 and P can be typed; «Уточнити» refits them, «Авто» goes back. k and b of the control line
   O−C = k·JD + b should be ≈ 0 (guide: P_new = P(1+k), T0_new = T0 + b).
6. Shape of O−C: line, parabola (dP/dE = 2a) or sine + line (astrolab oc_curve, BIC).
7. **Зберегти CSV…** writes `<star>_oc.csv` (`JD,O-C,min/max,type,N,[N],correction,sector,method,excluded,flag`,
   O−C in days, O−C = P(N − [N] + correction)) and `<star>_ephemeris.csv`.

```bash
uv sync
uv run main.py [STAR_FOLDER]          # the window
uv run main.py --batch STAR_FOLDER    # the CSV files into the folder, no window
```

Checks: `uv run python tests/test_oc.py`, `tests/test_extrema.py`, `tests/sweep.py` (seed sweep of
the O−C logic, expected `failures 0`), `xvfb-run -a uv run python tests/gui_smoke.py [STAR_FOLDER]`,
`tests/sample.py SAMPLE_DIR` (O−C of every star into `tests/out`).
