"""
A4. The in-tune bonus at the stretched (and compressed) scale's own intervals.
STATUS: exploratory.

If listeners re-centred learned interval categories on the octave they hear
(2.1 for stretched, 1.9 for compressed partials), the in-tune bonus found for
harmonic tones should reappear at the stretched scale's own intervals,
c * log2(2.1) semitones, as the tritone dip does. For each stretched or
compressed experiment (US and Korean) and each c = 1..11, the bonus
    b(c) = r(c u) - [r((c - 0.5) u) + r((c + 0.5) u)] / 2,   u = log2(octave ratio)
is computed on the held-out residual r of model M2 (HK + H20 + interval size):
leave-one-out over the US experiments for US data, weights fitted to all ten
US experiments for Korean data. Harmonic experiments (u = 1) are included for
reference. Participant bootstrap; minimum detectable effect of the mean bonus
over c != 6 at 80% power, two-sided alpha = .05: MDE = 2.80 * bootstrap SE.
Writes results/stretched_bonus.json.
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
KEYS = DC.MODELS["M2"]
UNIT = {"str3": np.log2(2.1), "comp3": np.log2(1.9), "str3_kr": np.log2(2.1), "comp3_kr": np.log2(1.9),
        "harm3": 1.0, "harm3_kr": 1.0}
KR = ["harm3_kr", "str3_kr", "comp3_kr"]
TRAIN = DC.train_mask(MD.WIN)


def at(v, x):
    return float(np.interp(x, DC.G, v))


def bonus(r, c, u):
    return at(r, c * u) - (at(r, (c - 0.5) * u) + at(r, (c + 0.5) * u)) / 2


def one(D, W):
    P = DC.smooth_all(D, W, DC.DEFAULT["kernel"])
    R, _ = DC.heldout_residuals(P, KEYS, TRAIN)
    beta_all = DC.fit(P, DC.US, KEYS, TRAIN)
    for e in KR:
        R[e] = P[e]["z"] - DC.predict(P[e], KEYS, beta_all, TRAIN)
    out = {}
    for e, u in UNIT.items():
        o = {f"b{c}": bonus(R[e], c, u) for c in range(1, 12)}
        o["other"] = float(np.mean([o[f"b{c}"] for c in DC.OTHER]))
        if u != 1.0:
            o["other_physical"] = float(np.mean([bonus(R[e], c, 1.0) for c in DC.OTHER]))
        out[e] = o
    return out


def main():
    D = DC.load(exps=DC.US + KR)
    est = one(D, DC.draw_weights(D, None))
    rng = np.random.default_rng(MD.SEED + 15)
    boots = []
    for b in range(NBOOT):
        boots.append(DC.flat(one(D, DC.draw_weights(D, rng))))
        if b % 100 == 0:
            print("boot", b, flush=True)
    fe = DC.flat(est)
    summ = {k: {"est": v, **MD.ci([bf[k] for bf in boots])} for k, v in fe.items()}
    mde = {e: 2.80 * summ[f"{e}|other"]["se"] for e in UNIT}
    out = {"status": "exploratory", "model": KEYS, "unit": UNIT, "est": est, "boot": summ, "mde_other": mde,
           "_nboot": NBOOT, "_seed": MD.SEED + 15}
    json.dump(out, open(RES / "stretched_bonus.json", "w"), indent=1)
    for e in UNIT:
        v = summ[f"{e}|other"]
        print(f"{e:9s} own-scale mean bonus {v['est']:+.3f} [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}]  MDE {mde[e]:.3f}  ",
              " ".join(f"{c}:{summ[f'{e}|b{c}']['est']:+.2f}" for c in range(1, 12)))


if __name__ == "__main__":
    main()
