# core/: modules and derived inputs from the first pipeline

The manuscript's analysis (`analysis/`) reuses part of the author's first, pre-manuscript
pipeline. Only the parts it needs are kept here.

| File | Role |
|---|---|
| `dissonance_model.py` | Roughness kernels (Sethares, Vassilakis, Hutchinson–Knopoff) and dissonance curves |
| `chord_validation.py`, `behavioral_validation.py` | Chord loading and helpers used by the chord analyses |
| `incon_bridge.py`, `incon_bridge.R`, `incon_validation.py` | Calls R `incon` and caches its outputs |
| `reference_data/incon_*.csv` | Cached `incon` outputs for the chord sets (regenerable with R `incon` 0.5.0) |
| `extract_partials.py` → `extracted_partials.json` | Partials measured from the Philharmonia and Iowa recordings |
| `run_analysis.py` → `results_summary.json` | Instrument × kernel roughness differences (SI S11) |
| `clarinet_replication.py` → `clarinet_replication_results.json` | Clarinet at three registers, two libraries |
| `extraction_sensitivity.py` → `extraction_sensitivity_results.json` | Partial-extraction settings |
| `codec_ab.py` → `codec_ab_results.json` | MP3 vs lossless check |
| `robustness_checks.py` → `robustness_checks.json` | Further kernel robustness checks |
| `incon_validation.py` → `incon_validation_results.json` | Model orientations for the chord regressions |
| `test_*.py` | Unit tests of the roughness kernels and the incon bridge |

`behavioral_data/` is filled by `scripts/fetch_data.sh` (DCMLab consonance, MIT licence).

The recordings are not redistributable, so the JSON files derived from them are included and
are treated as inputs by `run_all.sh`. To regenerate them, download the notes listed in
Supplementary Section S11 into `core/audio_samples/` and run, from `core/`,
`python extract_partials.py`, then the other scripts above.
