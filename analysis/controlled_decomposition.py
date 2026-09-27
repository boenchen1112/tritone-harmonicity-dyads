"""
Re-analysis R2: the three-component variance decomposition (paper Table 4) and
the 15-model ranking, redone with the controls the paper itself argues for.

The original (incon_validation.py) regresses raw pooled ratings on raw model
outputs with no covariates. Two problems, both acknowledged elsewhere in the
original paper but not applied here:
  * chord size: Bowling et al. (2018) blocked presentation by number of notes,
    and Harrison & Pearce (2020) evaluate every model on Bowling as a partial
    correlation controlling for number of notes (categorical) for that reason;
    model normalisations also scale differently with chord size.
  * register: Johnson-Laird et al. (2012) root height correlates with rated
    unpleasantness (r = +.43 in tetrachords); interference models depend on
    absolute pitch, harmonicity/corpus models do not.

Specifications (all OLS, y = human dissonance, oriented dissonance-up):
  A  raw          no covariates                      (reproduces paper Table 4)
  B  size         chord-size dummies
  C  size+reg     chord-size dummies + lowest MIDI note
  D  size+transp  chord-size dummies; every model evaluated on the chord
                  transposed so its lowest note is MIDI 60 (register-fair
                  for pitch-dependent models; needs run_incon_transposed.py)
For each: R^2 of each component over covariates, unique Delta-R^2 of each over
the other two + covariates, nested-F p, and a percentile bootstrap 95% CI
(2000 resamples of chords).
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import PROJECT_ROOT
import numpy as np
from scipy.stats import f as fdist

ROOT = PROJECT_ROOT
sys.path.insert(0, str(ROOT))
import os
os.chdir(ROOT)
from chord_validation import DATASETS, load_chords
from incon_bridge import MODEL_CLASS, _read_csv
from incon_validation import MODELS

RES = Path(__file__).resolve().parents[1] / "results"
COMP = {"interference": "hutch_78_roughness", "harmonicity": "har_18_harmonicity",
        "familiarity": "har_19_corpus"}
# orientation measured by incon_validation.measure_orientations (cached result)
SIGNS = json.load(open(ROOT / "incon_validation_results.json"))["orientations"]
RNG = np.random.default_rng(20260925)
NBOOT = 2000


def load_cache(path, models):
    rows = _read_csv(path)
    return {m: np.array([float(r[m]) for r in rows]) * SIGNS[m] for m in models}


def design(cov, *preds):
    cols = [np.ones(len(cov[0]) if cov else len(preds[0]))] + list(cov) + list(preds)
    return np.column_stack(cols)


def rsq(y, X):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    return 1 - res @ res / ((y - y.mean()) @ (y - y.mean()))


def covariates(chords, spec):
    size = np.array([len(c) for c in chords])
    cov = [(size == k).astype(float) for k in sorted(set(size))[1:]] if spec != "raw" else []
    if spec == "size+reg":
        cov.append(np.array([min(c) for c in chords], float))
    return cov


def decompose(y, cov, P):
    names = list(P)
    k_cov = len(cov)
    full = rsq(y, design(cov, *P.values()))
    base = rsq(y, design(cov)) if cov else 0.0
    out = {"r2_covariates": base, "r2_full": full, "alone_over_cov": {}, "unique": {}}
    for nm in names:
        out["alone_over_cov"][nm] = rsq(y, design(cov, P[nm])) - base
        red = rsq(y, design(cov, *[P[o] for o in names if o != nm]))
        n, kf = len(y), k_cov + len(names)
        F = (full - red) / ((1 - full) / (n - kf - 1))
        out["unique"][nm] = {"delta_r2": full - red, "F": F,
                             "p": float(1 - fdist.cdf(F, 1, n - kf - 1))}
    return out


def boot(y, cov, P):
    n = len(y)
    draws = {nm: [] for nm in P}
    for _ in range(NBOOT):
        i = RNG.integers(0, n, n)
        c = [v[i] for v in cov]
        # a resample can drop a whole size class; skip degenerate dummies
        c = [v for v in c if v.std() > 0]
        d = decompose(y[i], c, {k: v[i] for k, v in P.items()})
        for nm in P:
            draws[nm].append(d["unique"][nm]["delta_r2"])
    return {nm: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            for nm, v in draws.items()}


def partial_r(y, x, cov):
    if not cov:
        return float(np.corrcoef(x, y)[0, 1])
    X = design(cov)
    ry = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    rx = x - X @ np.linalg.lstsq(X, x, rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1])


def rnd(o):
    if isinstance(o, dict):
        return {k: rnd(v) for k, v in o.items()}
    if isinstance(o, list):
        return [rnd(v) for v in o]
    return round(float(o), 4) if isinstance(o, (float, np.floating)) else o


def main():
    out = {}
    for ds in DATASETS:
        chords, y = load_chords(ds)
        raw = load_cache(ROOT / f"reference_data/incon_{ds}.csv", MODELS)
        tpath = RES / f"incon_{ds}_transposed.csv"
        tr = load_cache(tpath, MODELS) if tpath.exists() else None
        specs = [("raw", raw), ("size", raw), ("size+reg", raw)]
        if tr is not None:
            specs.append(("size+transp", tr))
        out[ds] = {"n": len(y)}
        for spec, src in specs:
            cov = covariates(chords, "size" if spec == "size+transp" else spec)
            P = {k: src[m] for k, m in COMP.items()}
            d = decompose(y, cov, P)
            d["boot_ci_unique"] = boot(y, cov, P)
            ranking = {m: partial_r(y, src[m], cov) for m in MODELS}
            d["model_partial_r"] = dict(sorted(ranking.items(), key=lambda kv: -kv[1]))
            d["model_rank"] = {m: i + 1 for i, m in enumerate(d["model_partial_r"])}
            out[ds][spec] = d
            u = d["unique"]
            print(f"{ds:18s} {spec:12s} R2cov={d['r2_covariates']:.3f} full={d['r2_full']:.3f} | "
                  + "  ".join(f"{k[:4]} dR2={u[k]['delta_r2']:+.3f} p={u[k]['p']:.4f} "
                              f"CI[{d['boot_ci_unique'][k][0]:+.3f},{d['boot_ci_unique'][k][1]:+.3f}]"
                              for k in u))
            top = list(d["model_partial_r"].items())
            print("      ranks:", ", ".join(f"{i+1}.{m}={r:+.3f}" for i, (m, r) in enumerate(top)))
    (RES / "controlled_decomposition.json").write_text(json.dumps(rnd(out), indent=2))


if __name__ == "__main__":
    main()
