"""
Follow-up R5: does the tritone carry dissonance that the roughness curve misses,
and if so, how much -- and does harmonicity or familiarity account for it?

A. Dyads (Bowling, n = 12). Fit ratings on model output(s); report the
   tritone's and minor sixth's residuals (rating units) and the share of the
   observed TT-m6 rating difference each model reproduces. Leave-one-out
   residuals guard against the 12-point overfit.
B. Chords (both datasets). Tritone count (pairs at interval class 6) added to
   chord-size dummies + component models. Its coefficient is the 'tritone
   penalty' left unexplained by the models: rating units per tritone, with a
   2000-resample bootstrap CI. Compared with the penalty for semitone pairs
   (interval class 1) as a yardstick.
All ratings oriented dissonance-up. Uses incon outputs on idealised spectra.
"""
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import PROJECT_ROOT
import numpy as np

ROOT = PROJECT_ROOT
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from chord_validation import load_chords
from incon_bridge import _read_csv

RES = Path(__file__).resolve().parents[1] / "results"
SIGNS = json.load(open(ROOT / "incon_validation_results.json"))["orientations"]
RNG = np.random.default_rng(20260926)
MODELS = {"I": "hutch_78_roughness", "S": "seth_93_roughness", "H": "har_18_harmonicity",
          "F": "har_19_corpus", "P": "stolz_15_periodicity"}
SETS = {"roughness (H&K)": ["I"], "roughness (Sethares)": ["S"], "H&K + harmonicity": ["I", "H"],
        "H&K + familiarity": ["I", "F"], "H&K + harmonicity + familiarity": ["I", "H", "F"],
        "H&K + periodicity (Stolzenburg)": ["I", "P"]}


def load(ds):
    chords, y = load_chords(ds)
    rows = _read_csv(f"reference_data/incon_{ds}.csv")
    assert [r["chord"] for r in rows] == [" ".join(map(str, c)) for c in chords], "row order mismatch"
    X = {k: np.array([float(r[m]) for r in rows]) * SIGNS[m] for k, m in MODELS.items()}
    return chords, y, X


def ols(y, cols):
    A = np.column_stack([np.ones(len(y))] + cols)
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    return b, A


def ic_count(chord, ic):
    return sum(min((a - b) % 12, (b - a) % 12) == ic for i, a in enumerate(chord) for b in chord[i + 1:])


def dyads():
    chords, y, X = load("bowling2018")
    idx = [i for i, c in enumerate(chords) if len(c) == 2]
    iv = {chords[i][1] - chords[i][0]: n for n, i in enumerate(idx)}
    yd = y[idx]
    tt, m6 = iv[6], iv[8]
    obs = yd[tt] - yd[m6]
    out = {"observed_tt_minus_m6": obs, "rating_sd_dyads": yd.std(ddof=1), "models": {}}
    for name, keys in SETS.items():
        cols = [X[k][idx] for k in keys]
        b, A = ols(yd, cols)
        pred = A @ b
        loo = {}
        for j in (tt, m6):
            keep = np.arange(len(yd)) != j
            bj, _ = ols(yd[keep], [c[keep] for c in cols])
            loo[j] = yd[j] - np.r_[1, [c[j] for c in cols]] @ bj
        r2 = 1 - ((yd - pred) ** 2).sum() / ((yd - yd.mean()) ** 2).sum()
        out["models"][name] = {
            "r2": r2, "resid_tt": yd[tt] - pred[tt], "resid_m6": yd[m6] - pred[m6],
            "loo_resid_tt": loo[tt], "loo_resid_m6": loo[m6],
            "pred_tt_minus_m6": pred[tt] - pred[m6],
            "share_of_gap_reproduced": (pred[tt] - pred[m6]) / obs,
            "tt_resid_rank": int((np.argsort(np.argsort(-(yd - pred))))[tt]) + 1}
    return out


def penalty(ds):
    chords, y, X = load(ds)
    size = np.array([len(c) for c in chords])
    cov = [(size == k).astype(float) for k in sorted(set(size))[1:]]
    ic6 = np.array([ic_count(c, 6) for c in chords], float)
    ic1 = np.array([ic_count(c, 1) for c in chords], float)
    res = {"n": len(y), "n_with_tritone": int((ic6 > 0).sum()), "rating_sd": y.std(ddof=1), "models": {}}
    for name, keys in {"size only": [], **SETS}.items():
        cols = cov + [X[k] for k in keys] + [ic6, ic1]
        b, A = ols(y, cols)
        bt = []
        for _ in range(2000):
            s = RNG.integers(0, len(y), len(y))
            bb, *_ = np.linalg.lstsq(A[s], y[s], rcond=None)
            bt.append(bb[-2:])
        bt = np.array(bt)
        res["models"][name] = {"tritone_penalty": b[-2], "tritone_ci": list(np.percentile(bt[:, 0], [2.5, 97.5])),
                               "semitone_penalty": b[-1], "semitone_ci": list(np.percentile(bt[:, 1], [2.5, 97.5]))}
    return res


def rnd(o):
    if isinstance(o, dict):
        return {k: rnd(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [rnd(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return round(float(o), 4)
    return o


if __name__ == "__main__":
    out = {"dyads_bowling": dyads(), "penalty": {ds: penalty(ds) for ds in ("bowling2018", "johnson-laird2012")}}
    json.dump(rnd(out), open(RES / "tritone_gap.json", "w"), indent=1)
    d = out["dyads_bowling"]
    print(f"observed TT-m6 (dissonance-up rating units): {d['observed_tt_minus_m6']:.3f}; dyad SD {d['rating_sd_dyads']:.3f}")
    for k, v in d["models"].items():
        print(f"  {k:34s} R2={v['r2']:.3f} share={v['share_of_gap_reproduced']:+.2f} "
              f"residTT={v['resid_tt']:+.3f} (LOO {v['loo_resid_tt']:+.3f}, rank {v['tt_resid_rank']}) residm6={v['resid_m6']:+.3f}")
    for ds, r in out["penalty"].items():
        print(ds, "n", r["n"], "with TT", r["n_with_tritone"], "SD", round(r["rating_sd"], 3))
        for k, v in r["models"].items():
            print(f"  {k:34s} TT {v['tritone_penalty']:+.3f} [{v['tritone_ci'][0]:+.3f},{v['tritone_ci'][1]:+.3f}]"
                  f"   m2 {v['semitone_penalty']:+.3f} [{v['semitone_ci'][0]:+.3f},{v['semitone_ci'][1]:+.3f}]")
