"""
Sensitivity of the held-out gap to the half-width of the windows excluded from
the fit around the tritone (6) and the minor sixth (8). STATUS: exploratory
(added after the main analyses; the reported results use 0.6 semitones, three
kernel SDs).

Half-widths 0.4, 0.6 and 0.8 semitones; everything else as residual_profile.py
(leave one US experiment out, kernel SD 0.2, the same participant-bootstrap
draws, seed MD.SEED + 11), so the 0.6 row of HK + H20 equals the primary
estimand. Models: HK alone and HK + H20 (the recommended curve). Per harmonic
US experiment: residuals r(6), r(8) and the gap r(8) - r(6); pooled by random
effects with Hartung-Knapp intervals.
Writes results/window_sensitivity.json.
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
MODELS = {"HK": ["HK"], "HK+H20": ["HK", "H20"]}
WINS = {"0.4": 0.4, "0.6": 0.6, "0.8": 0.8}


def one(D, W):
    P = DC.smooth_all(D, W, DC.DEFAULT["kernel"])
    out = {}
    for w, win in WINS.items():
        train = DC.train_mask(win)
        out[w] = {}
        for m, keys in MODELS.items():
            R, _ = DC.heldout_residuals(P, keys, train)
            out[w][m] = {}
            for e in HARM6:
                r6, r8 = DC.at(R[e], 6), DC.at(R[e], 8)
                out[w][m][e] = {"r6": r6, "r8": r8, "gap": r8 - r6}
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
    for w in WINS:
        pooled[w] = {}
        for m in MODELS:
            pooled[w][m] = {}
            for s in ("r6", "r8", "gap"):
                y = [summ[f"{w}|{m}|{e}|{s}"]["est"] for e in HARM6]
                se = [summ[f"{w}|{m}|{e}|{s}"]["se"] for e in HARM6]
                pooled[w][m][s] = meta.dl(y, se)
    # paired differences from the default window, pooled over the six spectra (plain mean, same draws)
    paired = {}
    for w in WINS:
        if w == "0.6":
            continue
        for m in MODELS:
            def mean_gap(f, ww):
                return float(np.mean([f[f"{ww}|{m}|{e}|gap"] for e in HARM6]))
            d = [mean_gap(bf, w) - mean_gap(bf, "0.6") for bf in boots]
            paired[f"{w}|{m}"] = {"est": mean_gap(fe, w) - mean_gap(fe, "0.6"), **MD.ci(d)}
    out = {"status": "exploratory", "windows": WINS, "models": MODELS, "experiments": HARM6, "est": est,
           "boot": summ, "pooled_harm6": pooled, "paired_mean_gap_vs_default": paired,
           "_nboot": NBOOT, "_seed": MD.SEED + 11}
    json.dump(out, open(RES / "window_sensitivity.json", "w"), indent=1)
    for w in WINS:
        for m in MODELS:
            p = pooled[w][m]
            print(w, m, " ".join(f"{s}:{p[s]['est']:+.3f}[{p[s]['hk_ci'][0]:+.2f},{p[s]['hk_ci'][1]:+.2f}]"
                                 for s in ("r6", "r8", "gap")))


if __name__ == "__main__":
    main()
