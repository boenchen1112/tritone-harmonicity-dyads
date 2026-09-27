# Reproducibility record

A clean rebuild of every result from a fresh clone, compared with the committed outputs.

## What was run

1. Fresh clone of the repository (commit `b05d8c3`), then every generated file deleted:
   `results/` (JSON, CSV and the `*.npz` model-curve caches), `tables/`, `figures/`,
   `numbers.tex`, `bib_*.tex`, the PDFs and LaTeX by-products, `__pycache__/`, `external/` and
   `core/behavioral_data/`.
2. `bash scripts/fetch_data.sh` (6 min 11 s).
3. `bash run_all.sh`, with no environment variables set (`TRITONE_ROOT` and `TRITONE_EXTERNAL`
   at their defaults).

The rebuild found two defects in the pipeline, both fixed in this repository:

- `analysis/run_incon_transposed.py` read the chord files relative to the working directory and
  failed from a fresh clone; it now changes to the project root first, as the other chord scripts
  do. Its outputs are byte-identical to the committed ones.
- `run_all.sh` did not run `analysis/robustness_screen.py --profile`, which writes
  `results/residual_profile.json` (read by `make_tables.py`); the step is now included.

The run was interrupted twice by the machine running low on memory; it was resumed at the start
of the interrupted script (`marjieh_dyads.py`), in the same clone, so every output still comes
from the clean run.

## Environment

Windows 11, Intel Core i5-1240P, 16 GB RAM; Python 3.14.6 (NumPy 2.4.6, SciPy 1.18.0; full list
in `requirements.txt`); R 4.6.1 with `incon` 0.5.0 and `hrep` 0.16.1; Tectonic 0.17.0.
Single-threaded BLAS (set by `run_all.sh`).

## Result

| Compared | Outcome |
|---|---|
| Every number in `results/*.json`, `results/*.csv` and `results/*.npz` (608,418 values) | All within 1e-9 of the committed values; largest absolute difference 3.1e-11 (a bootstrap share in `marjieh_dyads.json`); 448,214 exactly equal |
| Keys, array shapes and non-numeric fields | Identical |
| `numbers.tex`, the 30 files in `tables/`, `bib_paper.tex`, `bib_supplement.tex` | Byte-identical (line endings normalised) |
| `analysis/check_consistency.py` | OK |
| Main-text word count | 8,000 |

Every number and table in the paper and the Supplementary Information is therefore reproduced
exactly; the differences of order 1e-11 are floating-point summation order and do not reach any
reported digit.

## Run time

About 6 h 10 min for the analysis, single-threaded (plus about 6 min for the data download and
2 min for tables, figures and PDFs). Longest steps: `marjieh_dyads.py` 2 h 03 min,
`model_boot.py` 1 h 32 min, `sharpness.py` 50 min, `robustness_screen.py` 24 min,
`grid_bonus.py` 18 min, `stretched_scale.py` 17 min, `et_ji_positions.py` 10 min,
`run_incon_transposed.py` 9 min; every other step takes under 8 min.
