"""
A1/A2 follow-up: the decomposition pooled by random effects over the six
harmonic spectra, so that between-spectrum variation counts (as for the
primary estimand). STATUS: exploratory (the decomposition was found after
seeing the residuals).

Per-experiment estimates and participant-bootstrap SEs come from
results/decomposition_models.json (models M1 = HK + H20, M2 = + interval size,
M3 = + sharpness proxy; default settings). Pooled with meta.dl:
DerSimonian-Laird tau^2, Hartung-Knapp interval (t, k - 1 df).
The A1 decision rule is re-applied to the Hartung-Knapp version: the
decomposition is kept in the main text if, under M1 and M2, (a) the gap left
(gap_resid) and (b) the step (step_ind11) both have intervals excluding zero.
Writes results/decomposition_re.json.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import meta

RES = HERE.parent / "results"
HARM6 = ["harm3", "eq5", "no3", "flute", "guitar", "piano"]
STATS = ["gap_resid", "gap_bonus_part", "gap_base_part", "q_step", "step_ind11", "step_ind14", "step_inc",
         "bonus_other", "bonus_tt", "other_minus_tt", "tt_minus_m2m7", "m2m7_minus_rest", "q6_minus_low"]


def excl(ci):
    return ci[0] > 0 or ci[1] < 0


def main():
    B = json.load(open(RES / "decomposition_models.json"))["boot"]
    out = {}
    for m in ("M1", "M2", "M3"):
        out[m] = {}
        for s in STATS:
            y = [B[f"default|{m}|exp|{e}|{s}"]["est"] for e in HARM6]
            se = [B[f"default|{m}|exp|{e}|{s}"]["se"] for e in HARM6]
            r = meta.dl(y, se)
            r["plain_mean"] = B[f"default|{m}|{s}"]
            r["per_exp"] = dict(zip(HARM6, y))
            out[m][s] = r
    rule = {m: {"gap_resid_excl0": excl(out[m]["gap_resid"]["hk_ci"]),
                "step_ind11_excl0": excl(out[m]["step_ind11"]["hk_ci"]),
                "gap_bonus_part_excl0": excl(out[m]["gap_bonus_part"]["hk_ci"]),
                "gap_base_part_excl0": excl(out[m]["gap_base_part"]["hk_ci"])} for m in out}
    for m in rule:
        rule[m]["keep"] = rule[m]["gap_resid_excl0"] and rule[m]["step_ind11_excl0"]
    res = {"status": "exploratory", "pooling": "random effects, DL tau2, Hartung-Knapp CI", "experiments": HARM6,
           "pooled": out, "rule": rule}
    json.dump(res, open(RES / "decomposition_re.json", "w"), indent=1)
    for m in out:
        for s in ("gap_resid", "gap_bonus_part", "gap_base_part", "step_ind11", "step_inc", "other_minus_tt",
                  "tt_minus_m2m7", "m2m7_minus_rest"):
            r = out[m][s]
            pm = r["plain_mean"]
            print(f"{m} {s:16s} RE {r['est']:+.3f} HK[{r['hk_ci'][0]:+.3f},{r['hk_ci'][1]:+.3f}] "
                  f"tau {r['tau']:.3f} I2 {100 * r['I2']:.0f}%   plain {pm['est']:+.3f} [{pm['ci'][0]:+.3f},{pm['ci'][1]:+.3f}]")
        print(m, rule[m])


if __name__ == "__main__":
    main()
