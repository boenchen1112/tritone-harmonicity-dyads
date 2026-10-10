"""
How much of the flattening of the interval-size trend across roll-off does the
sharpness proxy account for? For each roll-off slice (as in sharpness.py),
the slope over interval size implied by the shared sharpness weight is
b_S * dS/dx (per octave), set against the observed residual slope. Point
estimates. Writes results/sharpness_rolloff_implied.json.
"""
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sharpness as SH

RES = HERE.parent / "results"


def main():
    K, V, W, pid = SH.rolloff_design()
    fit = SH.rolloff_fit(K, V, W, pid)
    b_s = fit["HK+H+SHARP"]["beta"][2]
    implied = []
    for w in W:
        Sm = SH.MD.smooth(K, w, V)[SH.EV]
        implied.append(float(12 * b_s * np.polyfit(Sm[:, 7], Sm[:, 8], 1)[0]))
    obs = fit["slice_x_slope_per_octave"]
    out = {"levels": SH.ROLL_LEV, "observed": obs, "implied": implied, "b_s": b_s,
           "obs_ratio_13_1": obs[-1] / obs[0], "implied_ratio_13_1": implied[-1] / implied[0]}
    json.dump(out, open(RES / "sharpness_rolloff_implied.json", "w"), indent=1)
    print(out)


if __name__ == "__main__":
    main()
