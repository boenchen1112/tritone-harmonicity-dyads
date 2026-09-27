# The tritone and the dissonance curve: code, results and manuscript

Code, derived results and manuscript source for

> Chen, D. (2026). *Harmonicity accounts for the tritone's shortfall on the dissonance curve
> for dyads*. Manuscript submitted to the Research Archive of Rising Scholars.

Listeners rate the tritone below the minor sixth, yet sensory dissonance (roughness) curves
rate the two intervals about equally. The paper reanalyses 95,865 ratings of dyads sampled
continuously from 0 to 15 semitones (Marjieh et al., 2024), with ten US and three Korean
experiments that vary the tone spectrum, and the chord ratings of Bowling et al. (2018) and
Johnson-Laird et al. (2012). It asks how much of the tritone's shortfall a roughness curve
leaves unexplained, whether adding harmonicity closes it, and whether anything specific to the
tritone remains. No new data were collected.

Every number in the manuscript and the Supplementary Information (SI) is a LaTeX macro written
by `analysis/make_tables.py` from the JSON files in `results/` (and `core/`); every figure is
drawn by `analysis/make_figures.py`. Nothing is typed by hand, and
`analysis/check_consistency.py` checks this.

## Layout

| Path | Contents |
|---|---|
| `analysis/` | Analysis scripts, each writing one `results/<name>.json`; `make_tables.py`, `make_figures.py`, `apa_refs.py` (reference lists), `make_docx.py` (Word version), `check_consistency.py`, word counts |
| `core/` | Modules and derived inputs from the author's first pipeline that the analysis reuses (roughness kernels, `incon` bridge, partials measured from recordings); see `core/README.md` |
| `results/` | Derived results (JSON; summary statistics only) and cached model curves (`*.npz`) |
| `tables/`, `figures/`, `numbers.tex`, `bib_*.tex` | Generated; do not edit |
| `paper.tex`, `supplement.tex`, `refs.bib` | Manuscript and SI |
| `rars/` | Created by `analysis/make_docx.py` (optional Word and PDF versions); not tracked |
| `scripts/` | `fetch_data.sh` (third-party data), `export_bowl18_ji.R` |

Each analysis script states in its docstring, and in its JSON `"status"`, whether it is
pre-specified or exploratory.

## Requirements

- Python 3.14 with the packages in `requirements.txt` (`pip install -r requirements.txt`)
- R 4.6.1 with `incon` 0.5.0 and `hrep` 0.16.1 (one step, `run_incon_transposed.py`)
- `tectonic`, or a LaTeX distribution with `xr-hyper`, `natbib` and `booktabs`
- `git` and `curl` (data download)
- Optional, for the Word/PDF version: Microsoft Word on Windows (`pywin32`) and a copy of the
  RARS Word template (set `RARS_TEMPLATE`); the template is not redistributed here

## Data

Third-party data are **not** included. `scripts/fetch_data.sh` downloads them at pinned
commits into `external/` (override with `TRITONE_EXTERNAL`) and `core/behavioral_data/`:

| Source | Used for | Terms |
|---|---|---|
| Marjieh et al. (2024), [gitlab.com/pmcharrison/timbre-and-consonance-paper](https://gitlab.com/pmcharrison/timbre-and-consonance-paper) | Dense dyad ratings and timbre parameters | No licence stated; fetched, not redistributed |
| DCMLab, [github.com/DCMLab/consonance](https://github.com/DCMLab/consonance) | Bowling et al. (2018) and Johnson-Laird et al. (2012) chord ratings | MIT |
| Harrison, [github.com/pmcharrison/inconData](https://github.com/pmcharrison/inconData) | Bowling et al. (2018) chords with presented pitches | CC0 |
| [Philharmonia sound samples](https://philharmonia.co.uk/resources/sound-samples/) | Measured spectra (SI S11) | Not redistributable; the partials extracted from them are included in `core/extracted_partials.json` |
| [University of Iowa MIS](https://theremin.music.uiowa.edu/MIS.html) | Measured spectra (SI S11) | As above |

    bash scripts/fetch_data.sh        # about 7 minutes; about 850 MB in external/

## Running

    bash run_all.sh                   # every analysis, then tables, figures, references, PDFs
    bash run_all.sh --no-analysis     # tables, figures, consistency check and PDFs only
    python analysis/check_consistency.py
    RARS_TEMPLATE="/path/to/RARS template.docx" python analysis/make_docx.py   # Word + PDF

(`make all`, `make docs`, `make check` and `make fetch` are thin wrappers.) The defaults need
no environment variables; `TRITONE_ROOT` (default `core/`) and `TRITONE_EXTERNAL` (default
`external/`) override the input locations.

**Run time.** The full pipeline takes about 6 h 10 min single-threaded on a laptop (Intel Core i5-1240P, 16 GB) with
the model-curve caches (`results/*.npz`) present, and about 20 min more when they are rebuilt;
with the data download, allow about 7 hours. The longest steps are `marjieh_dyads.py` (about
2 h), `model_boot.py` (1 h 40 min), `sharpness.py` (50 min), `robustness_screen.py` (25 min),
`grid_bonus.py` (20 min) and `stretched_scale.py` (15 min); `--no-analysis` takes about two
minutes. The bootstrap counts in `run_all.sh` are those used for the reported results, and
each script fixes its random seed. Re-running reproduces the committed JSON to floating-point
precision (differences of order 1e-11), and `numbers.tex` and `tables/` exactly; see
`REPRODUCIBILITY.md` for the clean rebuild from a fresh clone.

## Which results feed which section

Traced from `make_tables.py`: a file is listed where at least one number or table in the
section is computed from it. Each `results/<name>.json` is written by `analysis/<name>.py` (`residual_profile.json` by `robustness_screen.py --profile`);
the `core/` files by the scripts listed in `core/README.md`.

| Section | Results |
|---|---|
| Abstract | `marjieh_dyads`, `marjieh_summary`, `multiplicity`, `residual_profile_re`, `sharpness` |
| 1 Introduction | `sharpness` |
| 2 Data | `marjieh_dyads`, `bowling_ji` |
| 3 Models | `marjieh_dyads` |
| 4 Estimands and inference | `marjieh_dyads`, `multiplicity`, `re_intervals` |
| 5.1 The premise (Table 1, Figure 1) | `core/behavioral_data/bowling2018.tsv`, `core/clarinet_replication_results`, `marjieh_dyads` |
| 5.2 Roughness (Table 2, Figure 2) | `marjieh_dyads`, `marjieh_summary` |
| 5.3 Harmonicity (Tables 3–4) | `marjieh_dyads`, `marjieh_summary`, `model_boot`, `multiplicity`, `residual_profile_re`, `et_ji_positions`, `bowling_ji` |
| 5.4 Held-out spectra (Tables 5–6, Figure 3) | `model_boot`, `model_screen`, `robustness_screen`, `marjieh_dyads` |
| 5.5 Is the tritone special? (Tables 7–8) | `residual_profile_re`, `grid_bonus`, `decomposition_models`, `decomposition_re`, `marjieh_summary` |
| 5.6 What the residual depends on (Tables 9–10) | `stretched_scale`, `stretched_bonus`, `multiplicity`, `korean_heldout`, `model_boot`, `marjieh_dyads`, `marjieh_summary` |
| 5.7 Chords (Table 11) | `bowling_ji`, `decomposition_ji`, `chords_sigma`, `controlled_decomposition`, `tritone_gap` |
| 5.8 How much the curves miss (Figure 4) | `sharpness`, `sharpness_rolloff_implied`, `model_screen2`, `confirm_triads`, `ceiling` |
| 6 Discussion | `marjieh_summary`, `model_screen`, `chords_sigma`, `multiplicity`, `residual_profile_re`, `stretched_bonus`, `kernel_vs_instrument`, `unified_sensitivity` |
| 7 Limitations; 8 Conclusion | `re_intervals`, `robustness_screen`, `multiplicity` |
| SI S1–S3 | `marjieh_dyads`, `marjieh_summary`, `re_intervals`, `model_screen` |
| SI S4 | `decomposition_models`, `decomposition_re`, `grid_bonus`, `step_robustness` |
| SI S5–S7 | `model_boot`, `stretched_bonus`, `stretched_scale`, `korean_heldout`, `marjieh_dyads`, `sharpness` |
| SI S8 | `confirm_triads`, `sharpness`, `sharpness_rolloff_implied` |
| SI S9 | `et_ji_positions`, `et_ji_rule`, `residual_profile` |
| SI S10 | `bowling_ji`, `controlled_decomposition`, `decomposition_ji`, `core/robustness_checks` |
| SI S11 | `kernel_vs_instrument`, `unified_sensitivity`, `further_robustness`, `core/results_summary`, `core/clarinet_replication_results`, `core/extracted_partials`, `core/extraction_sensitivity_results` |

## Licence

- Code (`analysis/`, `core/*.py`, `core/*.R`, `scripts/`, `run_all.sh`, `Makefile`): MIT.
- Figures, the Supplementary Information, and the derived results (`results/`, `core/*.json`,
  `core/reference_data/`): CC BY 4.0. These files hold summary statistics and model outputs,
  not the third-party ratings or recordings.
- The manuscript text (`paper.tex` and PDFs built from it) is not licensed.
- Third-party data keep their own terms (table above).

See `LICENSE` for the full text.

## Citation

See `CITATION.cff`. Code: <https://github.com/boenchen1112/tritone-harmonicity-dyads>, release `v1.0-submission`, archived at
[ZENODO DOI].
