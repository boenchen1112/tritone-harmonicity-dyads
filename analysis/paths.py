"""Locations of inputs outside analysis/.

PROJECT_ROOT holds the modules and derived inputs of the author's first
(pre-manuscript) pipeline that this analysis still uses: the roughness kernels
(dissonance_model.py), chord loading (chord_validation.py), the incon bridge,
the partials measured from recordings (extracted_partials.json) and the JSON
outputs of the measured-spectra scripts. They live in core/ of this
repository. EXTERNAL holds the third-party data fetched by
scripts/fetch_data.sh. Set TRITONE_ROOT or TRITONE_EXTERNAL to override."""
import os
from pathlib import Path

MS = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(os.environ.get("TRITONE_ROOT", MS / "core")).resolve()
EXTERNAL = Path(os.environ.get("TRITONE_EXTERNAL", MS / "external")).resolve()
