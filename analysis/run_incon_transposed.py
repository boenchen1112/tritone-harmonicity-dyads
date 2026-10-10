"""Run the 15 incon models on every rated chord transposed so its lowest note is
MIDI 60 (interval structure preserved). Output cached to manuscript/results/.
Changes to the project root first: incon_bridge.R and the chord files are referenced relatively."""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import PROJECT_ROOT
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)
from chord_validation import DATASETS, load_chords
from incon_bridge import run_incon
from incon_validation import MODELS

RES = Path(__file__).resolve().parents[1] / "results"
for ds in DATASETS:
    chords, _ = load_chords(ds)
    tr = [[n - min(c) + 60 for c in [c] for n in c] for c in chords]
    run_incon(tr, MODELS, cache_path=str(RES / f"incon_{ds}_transposed.csv"))
    print("done", ds, flush=True)
