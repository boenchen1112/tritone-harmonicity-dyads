"""
Which end of the gap? Held-out residual r(c) = observed - predicted at every
equal-tempered interval c = 1..14 semitones, per harmonic US experiment,
pooled by random effects with Hartung-Knapp intervals. STATUS: the split of
the gap into r(8) - r(6) is a re-analysis of the pre-specified gap; the
per-interval profile is exploratory.

Models (leave one US experiment out, default settings: kernel SD 0.2,
+-0.6 exclusion around 6 and 8): HK alone; HK + H (6.83 cents); HK + H20
(the recommended curve, M1 of A1); HK + H20 + interval size (M2).
Residuals are relative to each experiment's intercept, fitted on the rest of
the interval axis, so they locate where the shape of the curve is wrong.
Also: the tritone residual minus the mean residual of the other intervals
1..11 (c != 6), and the minor-sixth residual minus the mean of the other
consonances (3, 4, 5, 7, 9).
Participant bootstrap with the draws of A1 (seed MD.SEED + 11), so the HK + H20 gap equals the primary estimand. Writes results/residual_profile_re.json (residual_profile.json is robustness_screen.py's).
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
import meta

RES = HERE.parent / "results"
NBOOT = int(os.environ.get("NBOOT", 1000))
HARM6 = ["harm3", "eq5", "no3", "flute", "guitar", "piano"]
EXPS = HARM6 + ["pure"]
MODELS = {"HK": ["HK"], "HK+H": ["HK", "H"], "HK+H20": ["HK", "H20"], "HK+H20+X": ["HK", "H20", "X1"]}
TRAIN = DC.train_mask(MD.WIN)
CONS = [3, 4, 5, 7, 9]


def one(D, W):
    P = DC.smooth_all(D, W, DC.DEFAULT["kernel"])
    out = {}
    for m, keys in MODELS.items():
        R, _ = DC.heldout_residuals(P, keys, TRAIN)
        om = {}
        for e in EXPS:
            r = {c: DC.at(R[e], c) for c in range(1, 15)}
            oe = {f"r{c}": r[c] for c in r}
            oe["tt_minus_other"] = r[6] - float(np.mean([r[c] for c in DC.OTHER]))
            oe["m6_minus_cons"] = r[8] - float(np.mean([r[c] for c in CONS]))
            oe["gap"] = r[8] - r[6]
            om[e] = oe
        out[m] = om
    return out


def main():
    D = DC.load()
    est = one(D, DC.draw_weights(D, None))
    rng = np.random.default_rng(MD.SEED + 11)
    boots = []
    for b in range(NBOOT):
        boots.append(DC.flat(one(D, DC.draw_weights(D, rng))))
        if b % 100 == 0:
            print("boot", b, flush=True)
    fe = DC.flat(est)
    summ = {k: {"est": v, **MD.ci([bf[k] for bf in boots])} for k, v in fe.items()}
    pooled = {}
    for m in MODELS:
        pooled[m] = {}
        for s in est[m][HARM6[0]]:
            y = [summ[f"{m}|{e}|{s}"]["est"] for e in HARM6]
            se = [summ[f"{m}|{e}|{s}"]["se"] for e in HARM6]
            pooled[m][s] = meta.dl(y, se)
    out = {"status": {"gap_split": "re-analysis of the pre-specified gap", "profile": "exploratory"},
           "models": MODELS, "experiments": EXPS, "est": est, "boot": summ, "pooled_harm6": pooled,
           "_nboot": NBOOT, "_seed": MD.SEED + 11}
    json.dump(out, open(RES / "residual_profile_re.json", "w"), indent=1)
    for m in MODELS:
        print(m, " ".join(f"{s}:{pooled[m][s]['est']:+.3f}[{pooled[m][s]['hk_ci'][0]:+.2f},{pooled[m][s]['hk_ci'][1]:+.2f}]"
                          for s in ["r6", "r8", "gap", "tt_minus_other", "m6_minus_cons"]))
        print("   profile", " ".join(f"{c}:{pooled[m][f'r{c}']['est']:+.2f}" for c in range(1, 15)))


if __name__ == "__main__":
    main()
