"""
Does adding harmonicity change the tritone's residual? Paired differences
between curves, on the same held-out fits and the same bootstrap draws as
residual_profile.py (seed MD.SEED + 11), so the two residuals being compared
come from one resampling of the participants.

For each harmonic US experiment: the residual under roughness + harmonicity
minus the residual under roughness alone, at the tritone (r6), at the minor
sixth (r8), for the gap r8 - r6 and for the tritone relative to the other
intervals 1..11. Pooled by random effects with Hartung-Knapp intervals.
STATUS: added after the results were seen (exploratory); it tests a
difference that the text otherwise only describes.

Also checks that the per-model estimates equal those stored by
residual_profile.py. Writes results/harmonicity_change.json.
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
import residual_profile as RP

RES = HERE.parent / "results"
NBOOT = int(os.environ.get("NBOOT", 1000))
PAIRS = {"HK+H20 - HK": ("HK+H20", "HK"), "HK+H - HK": ("HK+H", "HK"), "HK+H20 - HK+H": ("HK+H20", "HK+H")}
STATS = ["r6", "r8", "gap", "tt_minus_other"]


def diffs(o):
    return {p: {e: {s: o[a][e][s] - o[b][e][s] for s in STATS} for e in RP.EXPS} for p, (a, b) in PAIRS.items()}


def main():
    D = DC.load()
    est = RP.one(D, DC.draw_weights(D, None))
    rng = np.random.default_rng(MD.SEED + 11)
    boots = []
    for b in range(NBOOT):
        boots.append(DC.flat(diffs(RP.one(D, DC.draw_weights(D, rng)))))
        if b % 100 == 0:
            print("boot", b, flush=True)
    fe = DC.flat(diffs(est))
    summ = {k: {"est": v, **MD.ci([bf[k] for bf in boots])} for k, v in fe.items()}
    pooled = {p: {s: meta.dl([summ[f"{p}|{e}|{s}"]["est"] for e in RP.HARM6],
                             [summ[f"{p}|{e}|{s}"]["se"] for e in RP.HARM6]) for s in STATS} for p in PAIRS}
    ref_path = RES / "residual_profile_re.json"
    max_dev = None
    if ref_path.exists():
        ref = json.load(open(ref_path))["est"]
        max_dev = max(abs(est[m][e][s] - ref[m][e][s]) for m in RP.MODELS for e in RP.EXPS for s in STATS)
        assert max_dev < 1e-9, max_dev
    out = {"status": "exploratory: added after the results were seen", "pairs": {p: list(v) for p, v in PAIRS.items()},
           "experiments": RP.EXPS, "boot": summ, "pooled_harm6": pooled, "max_dev_from_residual_profile": max_dev,
           "_nboot": NBOOT, "_seed": MD.SEED + 11}
    json.dump(out, open(RES / "harmonicity_change.json", "w"), indent=1)
    for p in PAIRS:
        print(p, " ".join(f"{s}:{pooled[p][s]['est']:+.3f}[{pooled[p][s]['hk_ci'][0]:+.3f},{pooled[p][s]['hk_ci'][1]:+.3f}]"
                          for s in STATS))
        for e in RP.EXPS:
            v = summ[f"{p}|{e}|r6"]
            print("   ", e, f"r6 {v['est']:+.3f} [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}]")


if __name__ == "__main__":
    main()
