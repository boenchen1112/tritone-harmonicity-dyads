"""
A2. Is the 7->8 semitone step in the held-out residual an artefact of analysis
choices?  STATUS: exploratory (robustness of a quantity found after seeing the
residuals).

Model M2 (HK + H20 + linear interval size; chosen under A1). One factor varied
at a time from the default (Gaussian kernel SD 0.2 semitones, exclusion
half-width 0.6 around 6 and 8 semitones in the leave-one-out fit, pooled over
the six harmonic US experiments):

  sd0.1, sd0.3   Gaussian kernel SD 0.1 / 0.3 semitones
  win0.4, win0.8 exclusion half-width 0.4 / 0.8. q(6), q(7) and q(8) are built
                 from the quarter-tones 5.5, 6.5, 7.5 and 8.5, all inside the
                 default +-0.6 exclusion; at 0.4, 7.5 and 8.5 enter the fit, so
                 this is the test of whether the step is the fit's blind zone.
  bin0.25        no kernel smoothing: within-listener z (and model features)
                 averaged in 0.25-semitone bins (running bins, so the values at
                 the quarter-tone points are bins centred on them)
Per-experiment values come from the same draws (key 'exp').

Statistics as decomp.py: q_step (range means), step_ind11/step_ind14
(indicator 1[c >= 8] over a linear trend in c), step_inc (q(8) - q(7) minus the
mean other increment), and the bonus/baseline parts of the gap.
Writes results/step_robustness.json.
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
MODELS = {"M2": DC.MODELS["M2"]}
SETTINGS = {
    "default": DC.DEFAULT,
    "sd0.1": {"kernel": ("gauss", 0.1), "win": MD.WIN},
    "sd0.3": {"kernel": ("gauss", 0.3), "win": MD.WIN},
    "win0.4": {"kernel": ("gauss", MD.SD), "win": 0.4},
    "win0.8": {"kernel": ("gauss", MD.SD), "win": 0.8},
    "bin0.25": {"kernel": ("box", 0.125), "win": MD.WIN},
}


def main():
    D = DC.load()
    est = DC.run_settings(D, DC.draw_weights(D, None), SETTINGS, MODELS)
    rng = np.random.default_rng(MD.SEED + 12)
    boots = []
    for b in range(NBOOT):
        boots.append(DC.flat(DC.run_settings(D, DC.draw_weights(D, rng), SETTINGS, MODELS)))
        if b % 50 == 0:
            print("boot", b, flush=True)
    fe = DC.flat(est)
    summ = DC.summarise(fe, boots, pairs=[(f"{s}|M2", "default|M2") for s in SETTINGS if s != "default"])
    out = {"status": "exploratory", "model": MODELS,
           "settings": {k: {"kernel": list(v["kernel"]), "win": v["win"]} for k, v in SETTINGS.items()},
           "est": est, "boot": summ, "_nboot": NBOOT, "_seed": MD.SEED + 12}
    json.dump(out, open(RES / "step_robustness.json", "w"), indent=1)
    for s in SETTINGS:
        row = []
        for k in ("q_step", "step_ind11", "step_ind14", "step_inc", "gap_bonus_part", "gap_base_part"):
            v = summ[f"{s}|M2|{k}"]
            row.append(f"{k} {v['est']:+.3f} [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}]")
        print(f"{s:8s}", " ".join(row))
    for e in DC.HARM6:
        row = []
        for k in ("q_step", "step_ind11", "step_inc", "gap_bonus_part", "gap_base_part"):
            v = summ[f"default|M2|exp|{e}|{k}"]
            row.append(f"{k} {v['est']:+.3f} [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}]")
        print(f"{e:8s}", " ".join(row))


if __name__ == "__main__":
    main()
