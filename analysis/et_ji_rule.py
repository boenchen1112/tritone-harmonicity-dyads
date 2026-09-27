"""
A8: which interpretation rule applies, computed from results/et_ji_positions.json
with the thresholds stated here (fixed before reading the pooled output only in
the sense that the rules were given in the revision instructions; the numeric
translation below was written after the results were seen, so it is recorded
explicitly). STATUS: exploratory.

Main width: kernel SD 0.1. Main model for the gap: HK + H20 (the recommended
curve); HK + H (6.83 cents) and the composite are reported alongside.
  peak ~ 800     : pooled listener m6 peak CI contains 800 and excludes 813.7
  peak ~ 813.7   : CI contains 813.7 and excludes 800
  model peak     : the HK + H (6.83-cent) curve (the 20-cent curve has no
                   interior m6 peak at this width; see edge flags)
  loc difference : listener - model peak, pooled; 'excludes 0' from its HK CI
  JI gap clearly smaller : upper HK CI of (JI - ET) unexplained gap < 0 for
                   JIa or JIb
  JI gap ~ ET gap : both (JI - ET) HK CIs within +-0.05 SD (an equivalence
                   margin of about a quarter of the ET gap)
Rules (instruction A8):
  1  peak ~ 800, model ~ 813.7, difference CI excludes 0
  2  peak ~ 813.7 and JI gap ~ ET gap
  3  peak ~ 813.7 and JI gap clearly smaller  -> stop before Section 3
  4  otherwise (intervals too wide to decide)
Writes results/et_ji_rule.json.
"""
import json
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / "results"
W = "sd0.1"
ET, JI = 8.0, 12 * 0.6780719051126377   # 813.686 cents = 1200 log2(8/5)
MARGIN = 0.05


def contains(ci, x):
    return ci[0] <= x <= ci[1]


def main():
    A = json.load(open(RES / "et_ji_positions.json"))
    P = A["pooled"][W]
    pk = P["listener"]["loc_m6"]
    near800 = contains(pk["hk_ci"], ET) and not contains(pk["hk_ci"], JI)
    near814 = contains(pk["hk_ci"], JI) and not contains(pk["hk_ci"], ET)
    mdl = P["HK+H"]["loc_m6"]
    model814 = abs(mdl["est"] - JI) <= 0.02
    ld = P["HK+H"]["locdiff_m6"]
    ld_excl = not contains(ld["hk_ci"], 0.0)
    diffs = {p: P["HK+H20"][f"ugap_{p}_minus_ET"] for p in ("JIa", "JIb")}
    smaller = any(d["hk_ci"][1] < 0 for d in diffs.values())
    equal = all(d["hk_ci"][0] > -MARGIN and d["hk_ci"][1] < MARGIN for d in diffs.values())
    if near800 and model814 and ld_excl:
        rule = 1
    elif near814 and smaller:
        rule = 3
    elif near814 and equal:
        rule = 2
    else:
        rule = 4
    out = {"status": "exploratory", "width": W, "rule": rule,
           "criteria": {"listener_peak": pk, "listener_peak_near_800": near800, "listener_peak_near_813.7": near814,
                        "model_peak_HK+H": mdl, "model_peak_near_813.7": model814,
                        "locdiff_HK+H": ld, "locdiff_excludes_0": ld_excl,
                        "ugap_JI_minus_ET_HK+H20": diffs, "JI_gap_clearly_smaller": smaller,
                        "JI_gap_equivalent_within_margin": equal, "margin": MARGIN}}
    json.dump(out, open(RES / "et_ji_rule.json", "w"), indent=1)
    print("A8 rule", rule, {k: v for k, v in out["criteria"].items() if isinstance(v, bool)})


if __name__ == "__main__":
    main()
