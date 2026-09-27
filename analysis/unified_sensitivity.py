"""
Re-analysis R4: one metric for every free choice.

The original paper's Figure 2 compared a cross-kernel RANGE (86.4 pp, clarinet
C4 only) with worst-case DELTAS from a baseline for recording, extraction and
codec, and its codec bar (3.6 pp) came from Vassilakis although Vassilakis was
excluded from the extraction bar. Here every factor is measured the same way:

    for each (spectrum, model) cell, the range (max - min) of the tritone/m6 gap
    across the levels of that factor, in percentage points,

summarised by median and maximum over cells, for the three models whose
magnitudes are reproducible (Sethares, Sethares-min, Hutchinson & Knopoff).
Vassilakis is reported separately. Instrument choice is included as the
reference quantity -- it is the thing the analysis is meant to measure, not a
nuisance -- so the nuisance factors can be read against it.
"""
import json
from itertools import combinations
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import PROJECT_ROOT
import numpy as np

ROOT = PROJECT_ROOT
RES = Path(__file__).resolve().parents[1] / "results"
STABLE = ["Sethares", "Sethares-min", "Hutch-Knopoff"]
ALL = STABLE + ["Vassilakis"]


def summ(vals):
    v = np.asarray(vals, float)
    return {"n_cells": int(v.size), "median": round(float(np.median(v)), 2),
            "max": round(float(v.max()), 2), "values": [round(float(x), 2) for x in v]}


def main():
    rs = json.load(open(ROOT / "results_summary.json"))
    ex = json.load(open(ROOT / "extraction_sensitivity_results.json"))
    cod = json.load(open(ROOT / "codec_ab_results.json"))
    rep = json.load(open(ROOT / "clarinet_replication_results.json"))["registers"]
    fr = json.load(open(RES / "further_robustness.json"))

    # 14 distinct measured spectra: 8 C4 timbres + the 6 non-C4 register-test
    # spectra ("(mid)" rows duplicate the C4 timbres and are skipped)
    measured = {}
    for r in rs["instrument_comparison"]:
        if not r["timbre"].startswith("idealized"):
            measured.setdefault(r["timbre"], {})[r["model"]] = r["gap_pct"]
    c4 = list(measured)
    for r in rs["register_test"]:
        if "(mid)" not in r["timbre"]:
            measured.setdefault(r["timbre"], {})[r["model"]] = r["gap_pct"]
    assert len(measured) == 14
    # Iowa clarinet spectra count as extra spectra for the kernel factor
    for reg, d in rep.items():
        measured[f"Iowa clarinet {reg}"] = {m: d["models"][m]["iowa_gap_pct"] for m in ALL}

    out = {}
    for label, models in (("stable", STABLE), ("vassilakis", ["Vassilakis"])):
        f = {}
        # instrument (reference): range across the 8 C4 instruments, per model
        f["instrument (reference)"] = summ([np.ptp([measured[t][m] for t in c4]) for m in models])
        # kernel: range across kernels, per spectrum (only meaningful for 'stable' set;
        # for the Vassilakis row report the range across all four kernels)
        ks = STABLE if label == "stable" else ALL
        f["kernel"] = summ([np.ptp([g[m] for m in ks]) for g in measured.values()])
        # recording: |Iowa - Philharmonia|, 3 clarinet registers
        f["recording"] = summ([abs(rep[reg]["models"][m]["difference_pp"]) for reg in rep for m in models])
        # extraction constants: range across the 10-point grid (deltas include the 0 baseline)
        f["extraction constants"] = summ([np.ptp(list(ex["timbres"][t]["models"][m]["deltas_pp"].values()))
                                          for t in ex["timbres"] for m in models])
        # codec: range across lossless + 4 bitrates
        vals = []
        for t in cod["targets"].values():
            for m in models:
                vals.append(np.ptp([v["gaps"][m]["gap_pct"] for v in t["variants"].values()]))
        f["audio codec"] = summ(vals)
        # idealised-spectrum constants (n_partials x rolloff)
        f["idealised constants"] = summ([np.ptp(list(ex["idealized_baseline"]["models"][m]["deltas_pp"].values()))
                                         for m in models])
        if label == "stable":
            # H&K kernel constants a x b (H&K only)
            f["H&K constants a,b, wide (H&K only)"] = summ(list(fr["hk_ab_sweep"]["range_pp"].values()))
            f["H&K constants a,b, narrow (H&K only)"] = summ(list(fr["hk_ab_sweep"]["range_pp_narrow"].values()))
        else:
            f["Vassilakis exponent"] = summ([np.ptp(list(v.values()))
                                             for v in fr["vassilakis_exponent"].values()])
        out[label] = f

    # the original Figure 2 numbers, for the erratum table
    codec_by_model = {m: max(abs(v["gaps"][m]["gap_pct"] - t["variants"]["lossless"]["gaps"][m]["gap_pct"])
                             for t in cod["targets"].values() for v in t["variants"].values())
                      for m in ALL}
    out["original_figure2_check"] = {"codec_worst_delta_by_model": {k: round(v, 2) for k, v in codec_by_model.items()},
                                     "codec_worst_delta_stable_models": round(max(codec_by_model[m] for m in STABLE), 2)}
    (RES / "unified_sensitivity.json").write_text(json.dumps(out, indent=2))
    for lab, f in out.items():
        print(lab)
        for k, v in f.items():
            print(f"   {k:32s} {({kk: vv for kk, vv in v.items() if kk != 'values'} if isinstance(v, dict) and 'values' in v else v)}")


if __name__ == "__main__":
    main()
