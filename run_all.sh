#!/usr/bin/env bash
# Run the whole pipeline in order, one script at a time, then build the manuscript.
#
#   TRITONE_ROOT      reused modules and derived inputs (default: ./core; see core/README.md)
#   TRITONE_EXTERNAL  third-party data fetched by scripts/fetch_data.sh (default: ./external)
#
# Bootstrap counts are those used for the reported results. Single-threaded BLAS keeps memory
# use and run-to-run numerical differences small; expect about 6-7 hours in total (README.md).
# Usage: bash run_all.sh [--no-analysis]   (--no-analysis rebuilds tables, figures and PDFs only)
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONIOENCODING=utf-8
A=analysis
run() { echo "=== $(date +%T) $*"; env "$@"; }

if [ "${1:-}" != "--no-analysis" ]; then
  # measured spectra and chords (pre-specified in the earlier version)
  run python $A/kernel_vs_instrument.py
  run python $A/run_incon_transposed.py          # needs R with incon (see README)
  run python $A/controlled_decomposition.py
  run python $A/further_robustness.py
  run python $A/unified_sensitivity.py
  run python $A/tritone_gap.py
  run python $A/harmonicity.py
  run python $A/bowling_ji.py
  run python $A/decomposition_ji.py
  # dense dyads
  run NBOOT=1000 python $A/marjieh_dyads.py
  run python $A/marjieh_summary.py
  run NBOOT=1000 python $A/stretched_scale.py
  run python $A/model_screen.py
  run python $A/model_screen2.py
  run python $A/chords_sigma.py
  run NBOOT=500 python $A/model_boot.py
  run NBOOT=300 python $A/robustness_screen.py
  run python $A/robustness_screen.py --profile   # results/residual_profile.json (point estimates)
  run python $A/ceiling.py
  run NBOOT=300 python $A/sharpness.py
  run python $A/sharpness_rolloff_implied.py
  run NBOOT=200 python $A/grid_bonus.py
  run NBOOT=1000 python $A/confirm_triads.py
  # added in the revision (A1-A8 and the residual profile; status in each JSON)
  run NBOOT=1000 python $A/residual_profile.py
  run NBOOT=1000 python $A/decomposition_models.py
  run python $A/decomposition_re.py
  run NBOOT=1000 python $A/step_robustness.py
  run NBOOT=1000 python $A/korean_heldout.py
  run NBOOT=1000 python $A/stretched_bonus.py
  run python $A/multiplicity.py
  run python $A/re_intervals.py
  run NBOOT=1000 python $A/et_ji_positions.py
  run python $A/et_ji_rule.py
fi

run python $A/make_figures.py
run python $A/make_tables.py
run python $A/apa_refs.py
# notes rendered from numbers.tex (*.md.in), where present
for f in *.md.in; do if [ -e "$f" ]; then run python $A/render_md.py "$f"; fi; done
run python $A/check_consistency.py

# manuscript and SI: each document reads the other's .aux for cross-references, so build twice
TEX=${TECTONIC:-tectonic}
for i in 1 2; do
  "$TEX" -k paper.tex
  "$TEX" -k supplement.tex
done
python $A/word_count.py
