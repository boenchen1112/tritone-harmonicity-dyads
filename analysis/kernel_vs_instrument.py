"""
Re-analysis R1: does the choice of roughness kernel move the tritone/m6 gap
further than the choice of instrument?

The original paper (paper.html, section 3 title and abstract) asserts it does.
This script tests that claim on the paper's own Table 2 numbers
(results_summary.json) with three metrics, each computed with and without
Vassilakis (whose magnitudes the paper itself says are not reproducible):

  1. Two-way additive variance partition of the gap (instrument x model, one
     observation per cell): share of total sum of squares attributable to the
     instrument main effect, the model main effect, and the residual
     (interaction). Done on the gap in percent and on the symmetric log-ratio
     ln(tritone/m6), which does not privilege either interval as denominator.
  2. Sign-disagreement rates: fraction of model pairs (instrument fixed) whose
     gaps differ in sign vs fraction of instrument pairs (model fixed).
  3. Ranges: spread across models for each instrument vs spread across
     instruments for each model.

Only measured spectra are used (the idealised row is not an instrument).
"""
import json
from itertools import combinations
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import PROJECT_ROOT
import numpy as np

ROOT = PROJECT_ROOT
OUT = Path(__file__).resolve().parents[1] / "results" / "kernel_vs_instrument.json"
MODELS = ["Sethares", "Sethares-min", "Vassilakis", "Hutch-Knopoff"]


def load(section):
    d = json.load(open(ROOT / "results_summary.json"))
    rows = [r for r in d[section] if not r["timbre"].startswith("idealized")]
    timbres = list(dict.fromkeys(r["timbre"] for r in rows))
    gap = np.array([[next(r["gap_pct"] for r in rows if r["timbre"] == t and r["model"] == m)
                     for m in MODELS] for t in timbres])
    lr = np.array([[np.log(next(r["tritone"] / r["m6"] for r in rows
                                if r["timbre"] == t and r["model"] == m))
                    for m in MODELS] for t in timbres])
    return timbres, gap, lr


def partition(Y):
    grand = Y.mean()
    ss_tot = float(((Y - grand) ** 2).sum())
    ss_row = float(Y.shape[1] * ((Y.mean(1) - grand) ** 2).sum())
    ss_col = float(Y.shape[0] * ((Y.mean(0) - grand) ** 2).sum())
    ss_res = ss_tot - ss_row - ss_col
    return {"instrument_share": round(ss_row / ss_tot, 3),
            "model_share": round(ss_col / ss_tot, 3),
            "interaction_share": round(ss_res / ss_tot, 3)}


def sign_disagreement(Y):
    s = np.sign(Y)
    n_i, n_m = s.shape
    mp = [s[i, a] != s[i, b] for i in range(n_i) for a, b in combinations(range(n_m), 2)]
    ip = [s[a, j] != s[b, j] for j in range(n_m) for a, b in combinations(range(n_i), 2)]
    return {"model_pairs_disagree": int(sum(mp)), "model_pairs_total": len(mp),
            "model_pair_rate": round(float(np.mean(mp)), 3),
            "instrument_pairs_disagree": int(sum(ip)), "instrument_pairs_total": len(ip),
            "instrument_pair_rate": round(float(np.mean(ip)), 3)}


def ranges(Y, timbres, models):
    across_models = {t: round(float(np.ptp(Y[i])), 1) for i, t in enumerate(timbres)}
    across_instr = {m: round(float(np.ptp(Y[:, j])), 1) for j, m in enumerate(models)}
    return {"across_models_by_instrument": across_models,
            "across_instruments_by_model": across_instr,
            "median_across_models": round(float(np.median(list(across_models.values()))), 1),
            "median_across_instruments": round(float(np.median(list(across_instr.values()))), 1)}


def flips_per_instrument(Y, timbres, models):
    return {t: [m for j, m in enumerate(models) if Y[i, j] < 0] for i, t in enumerate(timbres)}


def main():
    out = {}
    for section in ("instrument_comparison", "register_test"):
        timbres, gap, lr = load(section)
        for label, cols in (("all4", [0, 1, 2, 3]), ("noVass", [0, 1, 3])):
            ms = [MODELS[c] for c in cols]
            g, l = gap[:, cols], lr[:, cols]
            out[f"{section}/{label}"] = {
                "timbres": timbres, "models": ms,
                "partition_gap_pct": partition(g),
                "partition_log_ratio": partition(l),
                "sign": sign_disagreement(g),
                "ranges_pp": ranges(g, timbres, ms),
                "flipping_models_by_instrument": flips_per_instrument(g, timbres, ms),
            }
    OUT.write_text(json.dumps(out, indent=2))
    for k, v in out.items():
        print(k)
        print("  gap%  partition:", v["partition_gap_pct"])
        print("  lnR   partition:", v["partition_log_ratio"])
        print("  sign:", v["sign"])
        print("  ranges medians: models", v["ranges_pp"]["median_across_models"],
              " instruments", v["ranges_pp"]["median_across_instruments"])
        print("  flips:", {t: f for t, f in v["flipping_models_by_instrument"].items() if f})


if __name__ == "__main__":
    main()
