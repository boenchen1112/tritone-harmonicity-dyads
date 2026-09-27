"""
Is the tritone residual attached to 600 cents, or to the tritone of the
spectrum's own scale? With partials stretched to an octave ratio r, interval k
of the matching stretched scale lies at k*log2(r) equal-tempered semitones
(the stretched tritone at 6.42 for r = 2.1, 5.56 for r = 1.9). This script
repeats the gap/dip analysis of marjieh_dyads.py at those positions, with the
roughness + harmonicity calibration excluding +/-0.6 semitones around the
stretched tritone and minor sixth, and participant-bootstrap CIs, and the paired difference from the same analysis at the
equal-tempered positions (same bootstrap draws).

Writes results/stretched_scale.json.
"""
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import marjieh_dyads as MD

RES = HERE.parent / "results"
CASES = [("str3", 2.1), ("comp3", 1.9), ("harm3", 2.0), ("str3_kr", 2.1), ("comp3_kr", 1.9), ("harm3_kr", 2.0)]
KEYS = ("HK", "H")


def at(v, x):
    return np.interp(x, MD.GRID, v)


def stats(S, ratio, fit_keys):
    u = np.log2(ratio)
    tt, m6, p4, p5 = 6 * u, 8 * u, 5 * u, 7 * u
    g = MD.GRID
    mask = (g >= MD.LO) & (g <= 14.75) & (np.abs(g - tt) >= MD.WIN) & (np.abs(g - m6) >= MD.WIN)
    h = S[:, 0]
    cols = [S[:, 1 + MD.MODEL_KEYS.index(k)] for k in fit_keys]
    pred, _ = MD.ols_pred(h, cols, mask)
    gap = at(h, m6) - at(h, tt)
    dip = (at(h, p4) + at(h, p5)) / 2 - at(h, tt)
    pgap = at(pred, m6) - at(pred, tt)
    pdip = (at(pred, p4) + at(pred, p5)) / 2 - at(pred, tt)
    return {"obs_gap": gap, "obs_dip": dip, "unexpl_gap": gap - pgap, "unexpl_dip": dip - pdip}


def main():
    rng = np.random.default_rng(MD.SEED)
    curves_by_exp, _, _ = MD.load_curves()
    out = {}
    for key, ratio in CASES:
        _, fname, synth, _, _, _ = MD.EXPS[key]
        d = MD.read_trials(fname, synth)
        V = np.column_stack([d["z"]] + [np.interp(d["x"], MD.FINE, curves_by_exp[key][k]) for k in MD.MODEL_KEYS])
        K = MD.kernel(MD.GRID, d["x"])
        w0 = np.ones(len(d["x"]))
        res = {"ratio": ratio, "tt_position": 6 * np.log2(ratio)}
        fits = ((("HK", "H"), "HK+H"), (("HK",), "HK"))
        S0 = MD.smooth(K, w0, V)
        boots = [MD.smooth(K, w0 * MD.boot_weights(d["pid"], rng), V) for _ in range(MD.NBOOT)]
        for fk, name in fits:
            est = stats(S0, ratio, fk)
            bs = [stats(S, ratio, fk) for S in boots]
            # paired contrast: stretched-scale position minus the 600-cent analysis, same draws
            est6 = stats(S0, 2.0, fk)
            bs6 = [stats(S, 2.0, fk) for S in boots]
            for k, v in est.items():
                if k.startswith("obs") and name != "HK+H":
                    continue
                kk = k if k.startswith("obs") else f"{name}_{k}"
                res[kk] = {"est": float(v), **MD.ci([b[k] for b in bs])}
                if ratio != 2.0:
                    res[f"{kk}_minus600"] = {"est": float(v - est6[k]),
                                             **MD.ci([b[k] - b6[k] for b, b6 in zip(bs, bs6)])}
        out[key] = res
        print(key, {k: (round(v["est"], 3), [round(c, 3) for c in v["ci"]]) if isinstance(v, dict) else round(v, 3)
                    for k, v in res.items()}, flush=True)
    json.dump(out, open(RES / "stretched_scale.json", "w"), indent=1)


if __name__ == "__main__":
    main()
