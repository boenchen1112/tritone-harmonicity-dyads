"""
Confirmatory test of the sharpness term on data not yet analysed.
Predictions written 2026-09-27, before any rating in the file below was read.

Data: Marjieh et al.'s triad rating experiment timbre_3rdhar_har3rate.csv:
40 triads (two cumulative intervals of 1-10 semitones, total span 2-12) on C4,
each with five equal harmonics ("with_3rd") or the same without the third
harmonic ("without_3rd"); 200 participants; ratings z-scored within
participant and averaged per chord (80 chord means).

Features per chord, from the stated partials (lower tone C4): H&K roughness of
the pooled spectrum; harmonicity with a 20-cent tolerance (H20, as in the
dyads); the sharpness proxy S of sharpness.py.

Pre-specified tests (a test passes only if its 95% participant-bootstrap CI,
NBOOT resamples, lies on the predicted side of zero):
  T1  Fitted on the triads: intercept per timbre + HK + H20 + S. The weight of
      S is negative.
  T2  Leave-one-chord-out R2 of HK + H20 + S exceeds that of HK + H20
      (both with an intercept per timbre).
  T3  Frozen dyad weights (pooled over the ten US dyad experiments,
      sharpness.json): the within-timbre correlation between chord means and
      b_R*HK + b_H*H20 + b_S*S exceeds that for the two-term curve with its own
      frozen dyad weights.
Known risk, stated in advance: in the dyads the trend was absent for five
equal harmonics (the timbre used here), so a failure here is a live
possibility and will be reported as such.

Writes results/confirm_triads.json.
"""
import csv
import json
import os
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import marjieh_dyads as MD
import model_screen as S
import sharpness as SH
from dissonance_model import hutch_knopoff_dissonance, pool_partials

RES = HERE.parent / "results"
NBOOT = int(os.environ.get("NBOOT", 1000))
FILE = MD.DATA / "timbre_3rdhar_har3rate.csv"


def chord_features(ivs, amps):
    a = np.asarray(amps, float)
    h = np.arange(1, len(a) + 1, dtype=float)
    keep = a > 0
    roots = MD.F0 * 2 ** (np.cumsum([0] + list(ivs)) / 12)
    tones = [(r * h[keep], a[keep]) for r in roots]
    F, A = pool_partials(tones)
    Fall = np.concatenate([t[0] for t in tones])
    Aall = np.concatenate([t[1] for t in tones])
    return {"HK": float(hutch_knopoff_dissonance(F, A)), "H20": S.harm_sigma(Fall, Aall, 20.0),
            "S": SH.spectral_features(Fall, Aall, 0.0)["SHARP"]}


def load():
    rows = list(csv.DictReader(open(FILE, encoding="utf-8")))
    pid = np.array([r["participant_id"] for r in rows])
    y = np.array([float(r["rating"]) for r in rows])
    key = np.array([r["timbre"] + "|" + r["intervals"] for r in rows])
    z = np.full(len(y), np.nan)
    for p in np.unique(pid):
        m = pid == p
        s = y[m].std(ddof=1) if m.sum() > 1 else 0
        if s > 0:
            z[m] = (y[m] - y[m].mean()) / s
    ok = ~np.isnan(z)
    spec = {r["timbre"] + "|" + r["intervals"]: (json.loads(r["intervals"]),
                                                 json.loads(json.loads(r["chord_spec"])["custom_timbre"]))
            for r in rows}
    return pid[ok], key[ok], z[ok], spec


def design(keys, spec):
    feats = [chord_features(*spec[k]) for k in keys]
    X = {f: np.array([d[f] for d in feats]) for f in ("HK", "H20", "S")}
    X["with3"] = np.array([k.startswith("with_3rd") for k in keys], float)
    return X


def cols(X, names):
    return np.column_stack([np.ones(len(X["with3"])), X["with3"]] + [X[n] for n in names])


def loo_r2(y, M):
    pred = np.empty(len(y))
    for i in range(len(y)):
        m = np.arange(len(y)) != i
        b, *_ = np.linalg.lstsq(M[m], y[m], rcond=None)
        pred[i] = M[i] @ b
    return 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)


def within_r(y, p, g):
    yc, pc = y.copy(), p.copy()
    for v in (0.0, 1.0):
        m = g == v
        yc[m] -= yc[m].mean()
        pc[m] -= pc[m].mean()
    return float(np.corrcoef(yc, pc)[0, 1])


def stats(y, X, bd):
    M3, M2 = cols(X, ["HK", "H20", "S"]), cols(X, ["HK", "H20"])
    b3, *_ = np.linalg.lstsq(M3, y, rcond=None)
    p3 = bd["three"][0] * X["HK"] + bd["three"][1] * X["H20"] + bd["three"][2] * X["S"]
    p2 = bd["two"][0] * X["HK"] + bd["two"][1] * X["H20"]
    r3, r2 = within_r(y, p3, X["with3"]), within_r(y, p2, X["with3"])
    l3, l2 = loo_r2(y, M3), loo_r2(y, M2)
    return {"T1_bS": float(b3[4]), "T2_loo3": float(l3), "T2_loo2": float(l2), "T2_diff": float(l3 - l2),
            "T3_r3": r3, "T3_r2": r2, "T3_diff": r3 - r2}


def main():
    pid, key, z, spec = load()
    keys = sorted(spec)
    X = design(keys, spec)
    sj = json.load(open(RES / "sharpness.json"))["screen"]
    bd = {"three": sj["HK+H20+SHARP"]["all_us_beta"], "two": sj["HK+H20"]["all_us_beta"]}
    idx = np.array([keys.index(k) for k in key])

    def means(w):
        return np.bincount(idx, weights=w * z, minlength=len(keys)) / np.bincount(idx, weights=w, minlength=len(keys))

    est = stats(means(np.ones(len(z))), X, bd)
    rng = np.random.default_rng(MD.SEED + 3)
    boots = [stats(means(MD.boot_weights(pid, rng)), X, bd) for _ in range(NBOOT)]
    out = {"n_chords": len(keys), "n_participants": int(len(np.unique(pid))), "dyad_weights": bd,
           "tests": {k: {"est": v, **MD.ci([b[k] for b in boots])} for k, v in est.items()}}
    t = out["tests"]
    out["pass"] = {"T1": t["T1_bS"]["ci"][1] < 0, "T2": t["T2_diff"]["ci"][0] > 0, "T3": t["T3_diff"]["ci"][0] > 0}
    out["feature_corr"] = {f"{a}~{b}": float(np.corrcoef(X[a], X[b])[0, 1])
                           for a, b in (("S", "HK"), ("S", "H20"), ("HK", "H20"))}
    json.dump(out, open(RES / "confirm_triads.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
