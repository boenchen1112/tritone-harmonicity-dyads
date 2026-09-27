"""
A6. Primary estimand and Holm correction of the targeted tests.

Primary estimand (declared in the revised Section 4): the held-out
(leave-one-US-experiment-out) unexplained tritone--minor-sixth gap under the
recommended curve, HK + H20, pooled over the six harmonic US experiments by
random effects with a Hartung-Knapp interval. Per-experiment estimates and
participant-bootstrap SEs come from results/decomposition_models.json
(model M1 = HK + H20, statistic gap_resid = r(8) - r(6)). STATUS: the gap is the
paper's pre-specified question; the held-out HK + H20 curve was chosen by the
pre-specified screen (model_screen.py).

Targeted tests (pre-specified family of four; one contrast each), all on the
roughness + harmonicity (6.83-cent) residual as reported in the paper:
  culture        Korean minus US, residual gap, harmonic tones
  musicianship   musicians minus non-musicians, residual gap, random effects
                 over the eight unstretched US experiments
  third harmonic (no-3rd minus 5-equal-harmonics) change in the observed dip
                 minus the change predicted by H&K roughness
  roll-off       change in the observed gap from 2 to 12 dB/octave minus the
                 change predicted by H&K roughness
Two-sided p from a normal approximation to the bootstrap distribution,
p = 2 * (1 - Phi(|est| / se)). Where the paired draws were not stored (third
harmonic, roll-off), the SE of a difference is sqrt(se_a^2 + se_b^2); for the
roll-off levels, which share participants, this is conservative. Holm
step-down over the four p-values. Nothing is re-run.
Writes results/multiplicity.json.
"""
import json
import sys
from pathlib import Path
import numpy as np
from scipy.stats import norm, t as t_dist

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import meta

RES = HERE.parent / "results"
HARM6 = ["harm3", "eq5", "no3", "flute", "guitar", "piano"]


def pval(est, se):
    return float(2 * (1 - norm.cdf(abs(est) / se)))


def holm(ps):
    names = sorted(ps, key=ps.get)
    m, out, run = len(ps), {}, 0.0
    for i, n in enumerate(names):
        run = max(run, min(1.0, (m - i) * ps[n]))
        out[n] = run
    return out


def main():
    DM = json.load(open(RES / "decomposition_models.json"))["boot"]
    y = [DM[f"default|M1|exp|{e}|gap_resid"]["est"] for e in HARM6]
    se = [DM[f"default|M1|exp|{e}|gap_resid"]["se"] for e in HARM6]
    primary = {"per_exp": {e: {"est": a, "se": b} for e, a, b in zip(HARM6, y, se)}, **meta.dl(y, se),
               "mean_boot": DM["default|M1|gap_resid"]}

    MJ = json.load(open(RES / "marjieh_dyads.json"))
    MS = json.load(open(RES / "marjieh_summary.json"))
    tests = {}
    c = MJ["cohort_diff"]["harm3"]["HK+H_unexpl_gap"]
    tests["culture"] = {"est": c["est"], "se": c["se"]}
    c = MS["musicianship_re"]["diff_HK+H_unexpl_gap"]
    # random effects over k experiments: Hartung-Knapp SE recovered from its interval, p from t on k - 1 df (A5)
    tq = t_dist.ppf(0.975, c["k"] - 1)
    se_hk = (c["hksj_ci"][1] - c["hksj_ci"][0]) / (2 * tq)
    tests["musicianship"] = {"est": c["est"], "se": se_hk, "df": c["k"] - 1, "se_dl": c["se"],
                             "note": "random effects; Hartung-Knapp SE, t on k - 1 df"}
    n3 = MJ["no3_minus_eq5"]
    tests["third_harmonic"] = {"est": n3["obs_dip"]["est"] - n3["HK_pred_dip"]["est"],
                               "se": float(np.hypot(n3["obs_dip"]["se"], n3["HK_pred_dip"]["se"]))}
    ro = MJ["rolloff"]
    d_obs = ro["2"]["obs_gap"]["est"] - ro["12"]["obs_gap"]["est"]
    d_hk = ro["2"]["HK_pred_gap"]["est"] - ro["12"]["HK_pred_gap"]["est"]
    se_ro = float(np.sqrt(sum(ro[l][k]["se"] ** 2 for l in ("2", "12") for k in ("obs_gap", "HK_pred_gap"))))
    tests["rolloff"] = {"est": d_obs - d_hk, "se": se_ro, "obs_change": d_obs, "hk_pred_change": d_hk}
    for v in tests.values():
        v["p"] = (float(2 * t_dist.sf(abs(v["est"]) / v["se"], v["df"])) if "df" in v else pval(v["est"], v["se"]))
    adj = holm({k: v["p"] for k, v in tests.items()})
    for k, v in tests.items():
        v["p_holm"] = adj[k]
        v["survives"] = adj[k] < 0.05
    out = {"status": {"primary": "pre-specified question; curve chosen by the pre-specified screen",
                      "tests": "pre-specified family; Holm correction added in revision"},
           "primary": primary, "targeted_tests": tests}
    json.dump(out, open(RES / "multiplicity.json", "w"), indent=1)
    print("primary: RE %.3f HK[%.3f, %.3f] DL[%.3f, %.3f] tau %.3f I2 %.0f%% PI[%.3f, %.3f]" % (
        primary["est"], *primary["hk_ci"], *primary["dl_ci"], primary["tau"], 100 * primary["I2"], *primary["pi"]))
    for k, v in tests.items():
        print(f"{k:15s} est {v['est']:+.3f} se {v['se']:.3f} p {v['p']:.4f} holm {v['p_holm']:.4f} {v['survives']}")


if __name__ == "__main__":
    main()
