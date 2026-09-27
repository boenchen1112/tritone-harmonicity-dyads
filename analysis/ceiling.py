"""
How much reliable signal do the dense dyad profiles contain?

For each experiment: split participants at random into halves (NSPLIT times),
smooth each half with the same kernel on the evaluation grid (0.5-14.75
semitones), correlate the two half-profiles, and apply Spearman-Brown to get
the reliability of the full-sample profile. That reliability is the expected
R2 of a perfect model against the observed (noisy) profile, i.e. the ceiling
for the held-out R2 reported by model_screen.py.

Also reports, per experiment, the in-sample R2 of HK + H20 with weights fitted
on that experiment alone, so that the shortfall of the held-out R2 can be split
into noise (ceiling - in-sample) ... features (ceiling vs in-sample) and
transfer across timbres (in-sample vs held-out).

Writes results/ceiling.json.
"""
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import marjieh_dyads as MD
import model_screen as S

RES = HERE.parent / "results"
NSPLIT = 50
G = MD.GRID
EVAL = S.EVAL


def half_curves(d, rng):
    ids = np.unique(d["pid"])
    a = rng.choice(ids, size=len(ids) // 2, replace=False)
    m = np.isin(d["pid"], a)
    out = []
    for mm in (m, ~m):
        K = MD.kernel(G[EVAL], d["x"][mm])
        out.append(MD.smooth(K, np.ones(mm.sum()), d["z"][mm][:, None])[:, 0])
    return out


def main():
    rng = np.random.default_rng(11)
    base, _, _ = MD.load_curves()
    P = S.profiles(base, S.all_curves())
    out = {}
    for key, (label, fname, synth, coh, _, _) in MD.EXPS.items():
        d = MD.read_trials(fname, synth)
        rs = [np.corrcoef(*half_curves(d, rng))[0, 1] for _ in range(NSPLIT)]
        r = float(np.mean(rs))
        rel = 2 * r / (1 + r)
        h = P[key]["z"][EVAL]
        X = np.column_stack([np.ones(EVAL.sum()), P[key]["HK"][EVAL], P[key]["H20"][EVAL]])
        b, *_ = np.linalg.lstsq(X, h, rcond=None)
        r2_in = 1 - np.sum((h - X @ b) ** 2) / np.sum((h - h.mean()) ** 2)
        out[key] = {"label": label, "cohort": coh, "n_part": int(len(np.unique(d["pid"]))),
                    "n_trials": int(len(d["x"])), "r_half": r, "ceiling": rel, "r2_in_HKH20": float(r2_in)}
        print(f"{key:10s} N={out[key]['n_part']:5d} r_half={r:.3f} ceiling={rel:.3f} in-sample HK+H20={r2_in:.3f}",
              flush=True)
    json.dump(out, open(RES / "ceiling.json", "w"), indent=1)


if __name__ == "__main__":
    main()
