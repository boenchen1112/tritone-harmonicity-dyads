"""
A3. Korean listeners: held-out fits with weights from US experiments only,
next to the Korean split-half noise ceilings.  STATUS: exploratory (the
Korean comparison is a pre-specified targeted test, but the smooth-trend terms
were added after seeing the residuals).

For each Korean experiment, the model is fitted to all ten US experiments
(default settings: kernel SD 0.2, +-0.6 exclusion around 6 and 8) and the
Korean profile is predicted with an intercept only (as model_screen.predict).
R^2 over the evaluation range 0.5-14.75 semitones. Models M1 (HK + H20),
M2 (+ interval size), M3 (+ sharpness proxy), and Marjieh et al.'s composite.
Participant bootstrap over all experiments (US weights re-estimated in each
draw). Ceilings: results/ceiling.json (Spearman-Brown split-half reliability,
the upper bound on R^2 for a profile measured with this noise).
Writes results/korean_heldout.json.
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
KR = ["harm3_kr", "str3_kr", "comp3_kr"]
MODELS = {**DC.MODELS, "Composite": ["C"]}
TRAIN = DC.train_mask(MD.WIN)


def r2(h, p):
    m = DC.EVAL
    return float(1 - np.sum((h[m] - p[m]) ** 2) / np.sum((h[m] - h[m].mean()) ** 2))


def one(D, W):
    P = DC.smooth_all(D, W, DC.DEFAULT["kernel"])
    out = {}
    for m, keys in MODELS.items():
        beta = DC.fit(P, DC.US, keys, TRAIN)
        o = {e: r2(P[e]["z"], DC.predict(P[e], keys, beta, TRAIN)) for e in KR}
        o["mean"] = float(np.mean([o[e] for e in KR]))
        out[m] = o
    return out


def main():
    D = DC.load(exps=DC.US + KR)
    est = one(D, DC.draw_weights(D, None))
    rng = np.random.default_rng(MD.SEED + 16)
    boots = []
    for b in range(NBOOT):
        boots.append(DC.flat(one(D, DC.draw_weights(D, rng))))
        if b % 100 == 0:
            print("boot", b, flush=True)
    fe = DC.flat(est)
    summ = {k: {"est": v, **MD.ci([bf[k] for bf in boots])} for k, v in fe.items()}
    for m in MODELS:
        if m != "M1":
            for e in KR + ["mean"]:
                d = [bf[f"{m}|{e}"] - bf[f"M1|{e}"] for bf in boots]
                summ[f"{m}-M1|{e}"] = {"est": fe[f"{m}|{e}"] - fe[f"M1|{e}"], **MD.ci(d)}
    ceil = json.load(open(RES / "ceiling.json"))
    out = {"status": "exploratory", "models": MODELS, "est": est, "boot": summ,
           "ceiling": {e: ceil[e]["ceiling"] for e in KR}, "_nboot": NBOOT, "_seed": MD.SEED + 16}
    json.dump(out, open(RES / "korean_heldout.json", "w"), indent=1)
    for m in MODELS:
        print(f"{m:10s}", "  ".join(f"{e}: {summ[f'{m}|{e}']['est']:+.3f} [{summ[f'{m}|{e}']['ci'][0]:+.3f},"
                                    f"{summ[f'{m}|{e}']['ci'][1]:+.3f}]" for e in KR + ["mean"]))
    print("ceilings", out["ceiling"])


if __name__ == "__main__":
    main()
