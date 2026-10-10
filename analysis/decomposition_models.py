"""
A1. Does the decomposition of the tritone--minor-sixth gap survive without the
sharpness term?  STATUS: exploratory (the decomposition was designed after
seeing the dense-dyad residuals; this is a robustness check of it).

The same held-out (leave-one-US-experiment-out) residuals, participant
bootstrap and statistics as grid_bonus.py (engine in decomp.py), under three
models:
    M1  HK + H20                      (no smooth term)
    M2  HK + H20 + X1                 (linear interval size)
    M3  HK + H20 + SHARP              (sharpness proxy; the previous headline)
Paired bootstrap differences M1 - M3 and M2 - M3 on the same draws.

Decision rule (fixed in the revision instructions before this was run): keep
the decomposition in the main text, based on M2, if the step and the reduced
tritone bonus are present under M1 and M2 (same sign, CIs overlapping M3's,
the CI of the bonus contrast excluding zero); otherwise stop and report.

Writes results/decomposition_models.json.
"""
import json
import os
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import marjieh_dyads as MD
import decomp as DC

RES = HERE.parent / "results"
NBOOT = int(os.environ.get("NBOOT", 1000))


def main():
    D = DC.load()
    settings = {"default": DC.DEFAULT}
    est = DC.run_settings(D, DC.draw_weights(D, None), settings, DC.MODELS)
    rng = np.random.default_rng(MD.SEED + 11)          # same draws as grid_bonus.py
    boots = []
    for b in range(NBOOT):
        boots.append(DC.flat(DC.run_settings(D, DC.draw_weights(D, rng), settings, DC.MODELS)))
        if b % 50 == 0:
            print("boot", b, flush=True)
    fe = DC.flat(est)
    summ = DC.summarise(fe, boots, pairs=[("default|M1", "default|M3"), ("default|M2", "default|M3")])
    out = {"status": "exploratory", "models": DC.MODELS, "settings": {k: {"kernel": list(v["kernel"]), "win": v["win"]}
                                                                     for k, v in settings.items()},
           "est": est, "boot": summ, "_nboot": NBOOT, "_seed": MD.SEED + 11}
    json.dump(out, open(RES / "decomposition_models.json", "w"), indent=1)
    ks = sorted(fe)
    np.savez_compressed(RES / "decomposition_models_draws.npz", keys=np.array(ks),
                        draws=np.array([[bf[k] for k in ks] for bf in boots]))
    for k in ("gap_resid", "gap_bonus_part", "gap_base_part", "q_step", "step_ind11", "step_ind14", "step_inc",
              "q6_minus_low", "other_minus_tt", "tt_minus_m2m7", "m2m7_minus_rest"):
        row = []
        for m in DC.MODELS:
            v = summ[f"default|{m}|{k}"]
            row.append(f"{m} {v['est']:+.3f} [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}]")
        print(f"{k:16s}", "  ".join(row))


if __name__ == "__main__":
    main()
