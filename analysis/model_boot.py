"""
Participant-bootstrap uncertainty for the leave-one-experiment-out comparison
of model_screen.py / model_screen2.py: the pre-specified winner (H&K +
harmonicity with a 20-cent tolerance, chosen in every fold), the baseline
H&K + harmonicity, Marjieh et al.'s components and composite, and the
timbre-relative ("self-template") harmonicity. Every experiment's participants
are resampled, the whole leave-one-out procedure is rerun, and paired
differences are taken within each replicate.

Writes results/model_boot.json.
"""
import json
import os
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import marjieh_dyads as MD
import model_screen as S
import model_screen2 as S2

RES = HERE.parent / "results"
NBOOT = int(os.environ.get("NBOOT", 500))
MODELS = {"HK": [["HK"]], "HK+H": [["HK", "H"]], "HK+H20": [["HK", "H20"]], "revHK+H": [["revHK", "H"]],
          "Composite": [["C"]], "HK+Hself": [["HK", "Hself6.83"]], "HK+H20+Hself20": [["HK", "H20", "Hself20"]]}
METRICS = ("r2_us_mean", "r2_harm6_mean", "gap_harm6", "dip_harm6", "gap_us8", "dip_us8", "tt_minus_other",
           "str3_udip600", "str3_udip_ssc", "str3_pdip_ssc")


def precompute(base, feats):
    """Per experiment: kernel matrix and trial-level design, so each bootstrap
    replicate only re-weights."""
    pre = {}
    for key, (_, fname, synth, _, _, _) in MD.EXPS.items():
        d = MD.read_trials(fname, synth)
        cols = {**{k: base[key][k] for k in MD.MODEL_KEYS}, **feats[key]}
        names = list(cols)
        V = np.column_stack([d["z"]] + [np.interp(d["x"], MD.FINE, cols[k]) for k in names])
        pre[key] = (MD.kernel(MD.GRID, d["x"]), V, names, d["pid"])
    return pre


def profiles(pre, rng=None):
    out = {}
    for key, (K, V, names, pid) in pre.items():
        w = np.ones(len(pid)) if rng is None else MD.boot_weights(pid, rng)
        Sm = MD.smooth(K, w, V)
        out[key] = {"z": Sm[:, 0], **{k: Sm[:, 1 + i] for i, k in enumerate(names)}}
    return out


def run(P):
    return {m: {k: S.summarise(S.lodo(P, o)[0])[k] for k in METRICS} for m, o in MODELS.items()}


def main():
    base, _, _ = MD.load_curves()
    f1, f2 = S.all_curves(), S2.all_curves()
    feats = {k: {**f1[k], **f2[k]} for k in MD.EXPS}
    pre = precompute(base, feats)
    est = run(profiles(pre))
    rng = np.random.default_rng(MD.SEED)
    boots = []
    for b in range(NBOOT):
        boots.append(run(profiles(pre, rng)))
        if b % 50 == 0:
            print("boot", b, flush=True)
    out = {}
    for m in MODELS:
        out[m] = {}
        for k in METRICS:
            bs = [bb[m][k] for bb in boots]
            out[m][k] = {"est": est[m][k], **MD.ci(bs)}
    diffs = {}
    for a, b in (("HK+H20", "HK+H"), ("HK+H20", "Composite"), ("HK+H20", "revHK+H"), ("HK+H", "Composite"),
                 ("HK+Hself", "HK+H"), ("HK+H20+Hself20", "HK+H20")):
        diffs[f"{a} - {b}"] = {k: {"est": est[a][k] - est[b][k], **MD.ci([bb[a][k] - bb[b][k] for bb in boots])}
                               for k in METRICS}
    out["_diffs"] = diffs
    out["_nboot"] = NBOOT
    json.dump(out, open(RES / "model_boot.json", "w"), indent=1)
    for m in MODELS:
        print(m, {k: (round(v["est"], 3), [round(c, 3) for c in v["ci"]]) for k, v in out[m].items()})
    for d, v in diffs.items():
        print(d, {k: (round(x["est"], 3), [round(c, 3) for c in x["ci"]]) for k, x in v.items()})


if __name__ == "__main__":
    main()
