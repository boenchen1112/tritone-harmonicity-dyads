"""
Do listeners like the equal-tempered intervals themselves more than the curve
predicts? (Exploratory; found while checking where the residual lies.)

For a held-out residual profile r (leave-one-US-experiment-out, as in
model_screen.py), the "grid bonus" at interval c is
    b(c) = r(c) - [r(c - 0.5) + r(c + 0.5)] / 2,
how far the residual at an equal-tempered interval exceeds the residual at the
quarter-tones on either side. Averaged over c = 1..11 without the tritone, and
separately at the tritone. A learned category of the equal-tempered intervals
attached to the fundamental-frequency ratio predicts a bonus for every
spectrum, including pure, stretched and compressed tones; a bonus that appears
only for harmonic complex tones points to the sounds in which those intervals
are heard (or to a harmonicity model still wrong in detail). For stretched and
compressed tones the bonus is also computed on the stretched scale's own grid.

Models: HK + H20, and HK + H20 + sharpness (sharpness.py). Participant
bootstrap (NBOOT). Writes results/grid_bonus.json.
"""
import json
import os
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import marjieh_dyads as MD
import model_screen as S
import model_boot as MB
import sharpness as SH

RES = HERE.parent / "results"
NBOOT = int(os.environ.get("NBOOT", 200))
MODELS = {"HK+H20": ["HK", "H20"], "HK+H20+SHARP": ["HK", "H20", "SHARP"]}
OTHER = [c for c in range(1, 12) if c != 6]
UNIT = {"str3": np.log2(2.1), "comp3": np.log2(1.9)}


def bonus(r, c, u=1.0):
    return S.at(r, c * u) - (S.at(r, (c - 0.5) * u) + S.at(r, (c + 0.5) * u)) / 2


def metrics(P):
    out = {}
    for m, ks in MODELS.items():
        R = {}
        for e in S.US:
            beta, _ = S.fit(P, [o for o in S.US if o != e], ks)
            R[e] = P[e]["z"] - S.predict(P[e], ks, beta)
        o = {}
        h6 = [[bonus(R[e], c) for c in range(1, 15)] for e in S.HARM6]
        prof = np.mean(h6, axis=0)
        o["harm6_profile"] = prof.tolist()
        o["harm6_other"] = float(np.mean([prof[c - 1] for c in OTHER]))
        o["harm6_tt"] = float(prof[5])
        # decomposition r(c) = q(c) + b(c), q = mean of the two quarter-tone neighbours
        q6 = [[(S.at(R[e], c - 0.5) + S.at(R[e], c + 0.5)) / 2 for c in range(1, 15)] for e in S.HARM6]
        qprof = np.mean(q6, axis=0)
        o["harm6_qprofile"] = qprof.tolist()
        for c in range(1, 12):
            o[f"harm6_b{c}"] = float(prof[c - 1])
            o[f"harm6_q{c}"] = float(qprof[c - 1])
        o["gap_bonus_part"] = float(prof[7] - prof[5])
        o["gap_base_part"] = float(qprof[7] - qprof[5])
        o["q_low_mean"] = float(np.mean([qprof[c - 1] for c in (1, 2, 3, 4, 5, 7)]))
        o["q_high_mean"] = float(np.mean([qprof[c - 1] for c in (8, 9, 10, 11)]))
        o["q_step"] = o["q_high_mean"] - o["q_low_mean"]
        o["q6_minus_low"] = float(qprof[5]) - o["q_low_mean"]
        o["tt_minus_m2m7"] = float(prof[5] - (prof[1] + prof[9]) / 2)
        o["m2m7_minus_rest"] = float((prof[1] + prof[9]) / 2 - np.mean([prof[c - 1] for c in (1, 3, 4, 5, 7, 8, 9, 11)]))
        o["harm6_other_minus_tt"] = o["harm6_other"] - o["harm6_tt"]
        for e in ("pure", "bonang"):
            o[f"{e}_other"] = float(np.mean([bonus(R[e], c) for c in OTHER]))
        for e, u in UNIT.items():
            o[f"{e}_other_physical"] = float(np.mean([bonus(R[e], c) for c in OTHER]))
            o[f"{e}_other_ownscale"] = float(np.mean([bonus(R[e], c, u) for c in OTHER]))
        o["harm6_minus_pure"] = o["harm6_other"] - o["pure_other"]
        o["harm6_minus_str3"] = o["harm6_other"] - o["str3_other_physical"]
        out[m] = o
    return out


def flat(d):
    return {f"{m}|{k}": v for m, o in d.items() for k, v in o.items() if not isinstance(v, list)}


def main():
    base, _, _ = MD.load_curves()
    f1, f3 = S.all_curves(), SH.all_curves()
    feats = {k: {**f1[k], **f3[k]} for k in MD.EXPS}
    pre = MB.precompute(base, feats)
    est = metrics(MB.profiles(pre))
    rng = np.random.default_rng(MD.SEED + 11)
    boots = []
    for b in range(NBOOT):
        boots.append(flat(metrics(MB.profiles(pre, rng))))
        if b % 25 == 0:
            print("boot", b, flush=True)
    fe = flat(est)
    out = {"est": est, "boot": {k: {"est": v, **MD.ci([bb[k] for bb in boots])} for k, v in fe.items()},
           "_nboot": NBOOT}
    json.dump(out, open(RES / "grid_bonus.json", "w"), indent=1)
    for k, v in out["boot"].items():
        print(f"{k:40s} {v['est']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]")


if __name__ == "__main__":
    main()
