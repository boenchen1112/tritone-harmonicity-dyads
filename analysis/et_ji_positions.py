"""
A8. Equal temperament vs just intonation: where are listeners' peaks and dips,
and is the tritone--minor-sixth gap partly produced by evaluating it at
equal-tempered (ET) positions?  STATUS: exploratory (added after the
dense-dyad residuals had been seen).

Experiments: the six harmonic US experiments; pure, stretched (str3) and
compressed (comp3) tones as controls. Kernel SD 0.1 semitones (main), 0.05 and
0.2 (sensitivity), on MD.GRID (0.01 semitone = 1 cent).

A8.1 Locations, per experiment, with a participant-bootstrap CI:
    m6 peak   argmax over 7.5-8.5 semitones    (ET 800, 8:5 813.7 cents)
    TT dip    argmin over 5.5-6.5              (ET 600; 7:5 582.5, 45:32 590.2,
                                                64:45 609.8, 10:7 617.5)
    controls  P5 argmax 6.5-7.5 (ET 700, 3:2 702.0), M3 argmax 3.5-4.5
              (400, 5:4 386.3), M6 argmax 8.5-9.5 (900, 5:3 884.4)
  on the listener profile and on the smoothed held-out predictions of HK,
  HK + H (6.83 cents), HK + H20 and Marjieh et al.'s composite. Model weights
  are the leave-one-US-experiment-out weights of the main analysis (fitted on
  0.2-SD profiles, excluding +-0.6 around 6 and 8) applied to the model
  features smoothed at each width; intercept as model_screen.predict. An
  extremum on the edge of its window is flagged ("edge share" = fraction of
  draws on the edge). Reliability: whole-profile split-half ceiling
  (Spearman-Brown, as ceiling.py) at each width, and the split-half agreement
  of the locations themselves (SD of the half-A minus half-B location).

A8.2 Gaps: observed and held-out unexplained gap P(m6) - P(TT) at
    ET     800 / 600
    JIa    813.7 / 582.5   (8:5 vs 7:5)
    JIb    813.7 / 590.2   (8:5 vs 45:32)
    emp    listener peak / listener dip, positions re-estimated in every draw
           (by construction >= the fixed-position gap; not used to choose
           between interpretations)
  paired JI - ET differences on the same draws. Pooled over the six harmonic
  experiments by random effects with Hartung-Knapp intervals (meta.dl, using
  per-experiment bootstrap SEs), and as a plain mean with bootstrap CI.

Writes results/et_ji_positions.json.
"""
import json
import os
import time
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import marjieh_dyads as MD
import decomp as DC
import meta

RES = HERE.parent / "results"
NBOOT = int(os.environ.get("NBOOT", 1000))
NSPLIT = 50
G = DC.G
HARM6 = DC.HARM6
EXPS = HARM6 + ["pure", "str3", "comp3"]
WIDTHS = {"sd0.1": ("gauss", 0.1), "sd0.05": ("gauss", 0.05), "sd0.2": ("gauss", 0.2)}
MODELS = {"HK": ["HK"], "HK+H": ["HK", "H"], "HK+H20": ["HK", "H20"], "Composite": ["C"]}
WINDOWS = {"m6": (7.5, 8.5, "max"), "tt": (5.5, 6.5, "min"), "p5": (6.5, 7.5, "max"),
           "M3": (3.5, 4.5, "max"), "M6": (8.5, 9.5, "max")}
C = lambda cents: cents / 100
POS = {"ET": (8.0, 6.0), "JIa": (C(1200 * np.log2(8 / 5)), C(1200 * np.log2(7 / 5))),
       "JIb": (C(1200 * np.log2(8 / 5)), C(1200 * np.log2(45 / 32)))}
REF = {"m6": {"ET": 8.0, "JI": C(1200 * np.log2(8 / 5))}, "p5": {"ET": 7.0, "JI": C(1200 * np.log2(3 / 2))},
       "M3": {"ET": 4.0, "JI": C(1200 * np.log2(5 / 4))}, "M6": {"ET": 9.0, "JI": C(1200 * np.log2(5 / 3))},
       "tt": {"ET": 6.0, "7:5": C(1200 * np.log2(7 / 5)), "45:32": C(1200 * np.log2(45 / 32)),
              "64:45": C(1200 * np.log2(64 / 45)), "10:7": C(1200 * np.log2(10 / 7))}}
TRAIN = DC.train_mask(MD.WIN)


def at(v, x):
    return float(np.interp(x, G, v))


def locate(v, lo, hi, kind):
    m = (G >= lo - 1e-9) & (G <= hi + 1e-9)
    i = np.argmax(v[m]) if kind == "max" else np.argmin(v[m])
    x = G[m][i]
    return float(x), bool(i == 0 or i == m.sum() - 1)


def one_draw(D, W):
    """Everything for one participant weighting (W) of all experiments."""
    P = {w: DC.smooth_all(D, W, k) for w, k in WIDTHS.items()}
    beta = {}
    for m, keys in MODELS.items():                      # held-out weights, main-analysis settings
        for e in EXPS:
            beta[(m, e)] = DC.fit(P["sd0.2"], [o for o in DC.US if o != e], keys, TRAIN)
    out = {}
    for w in WIDTHS:
        ow = {}
        for e in EXPS:
            Pe = P[w][e]
            curves = {"listener": Pe["z"]}
            for m, keys in MODELS.items():
                curves[m] = DC.predict(Pe, keys, beta[(m, e)], TRAIN)
            oe = {}
            for c, v in curves.items():
                oc = {}
                for loc, (lo, hi, kind) in WINDOWS.items():
                    x, edge = locate(v, lo, hi, kind)
                    oc[f"loc_{loc}"] = x
                    oc[f"edge_{loc}"] = float(edge)
                for p, (a, b) in POS.items():
                    oc[f"gap_{p}"] = at(v, a) - at(v, b)
                oe[c] = oc
            pk, dp = oe["listener"]["loc_m6"], oe["listener"]["loc_tt"]
            for c, v in curves.items():
                oe[c]["gap_emp"] = at(v, pk) - at(v, dp)
            for m in MODELS:
                for p in list(POS) + ["emp"]:
                    oe[m][f"ugap_{p}"] = oe["listener"][f"gap_{p}"] - oe[m][f"gap_{p}"]
                for loc in WINDOWS:
                    oe[m][f"locdiff_{loc}"] = oe["listener"][f"loc_{loc}"] - oe[m][f"loc_{loc}"]
            for c in oe:
                for p in ("JIa", "JIb", "emp"):
                    g = "ugap" if c != "listener" else "gap"
                    oe[c][f"{g}_{p}_minus_ET"] = oe[c][f"{g}_{p}"] - oe[c][f"{g}_ET"]
            ow[e] = oe
        # plain mean over the six harmonic experiments
        ow["harm6mean"] = {c: {k: float(np.mean([ow[e][c][k] for e in HARM6])) for k in ow[HARM6[0]][c]}
                           for c in ow[HARM6[0]]}
        out[w] = ow
    return out, P


def split_half(D, rng, kernel):
    """Whole-profile ceiling and location agreement between random halves."""
    res = {}
    for e in EXPS:
        v = D[e]
        ups, inv = np.unique(v["pid"], return_inverse=True)
        K = DC.kmat(v["x"], kernel)
        rs, dloc = [], {k: [] for k in WINDOWS}
        for _ in range(NSPLIT):
            a = rng.choice(len(ups), size=len(ups) // 2, replace=False)
            ma = np.isin(inv, a)
            halves = []
            for mm in (ma, ~ma):
                halves.append((K[:, mm] @ v["V"][mm, 0]) / K[:, mm].sum(1))
            rs.append(np.corrcoef(halves[0][DC.EVAL], halves[1][DC.EVAL])[0, 1])
            for loc, (lo, hi, kind) in WINDOWS.items():
                dloc[loc].append(locate(halves[0], lo, hi, kind)[0] - locate(halves[1], lo, hi, kind)[0])
        r = float(np.mean(rs))
        res[e] = {"r_half": r, "ceiling": 2 * r / (1 + r), **{f"locsd_{k}": float(np.std(x, ddof=1))
                                                              for k, x in dloc.items()}}
    return res


def main():
    D = DC.load(exps=DC.US)
    est, P0 = one_draw(D, DC.draw_weights(D, None))
    rng = np.random.default_rng(MD.SEED + 13)
    boots = []
    for b in range(NBOOT):
        boots.append(DC.flat(one_draw(D, DC.draw_weights(D, rng))[0]))
        if b % 50 == 0:
            print("boot", b, time.strftime("%X"), flush=True)
    fe = DC.flat(est)
    summ = {}
    for k, v in fe.items():
        bs = np.array([bf[k] for bf in boots])
        summ[k] = {"est": v, **MD.ci(bs)}
    # random-effects pooling (Hartung-Knapp) over the six harmonic experiments
    pooled = {}
    for w in WIDTHS:
        pooled[w] = {}
        for c in est[w][HARM6[0]]:
            pooled[w][c] = {}
            for k in est[w][HARM6[0]][c]:
                if k.startswith("edge_"):
                    continue
                y = [summ[f"{w}|{e}|{c}|{k}"]["est"] for e in HARM6]
                se = [summ[f"{w}|{e}|{c}|{k}"]["se"] for e in HARM6]
                if min(se) <= 1e-9:  # (near-)deterministic model quantity: report the plain mean only
                    pooled[w][c][k] = {"est": float(np.mean(y)), "hk_ci": [float(min(y)), float(max(y))],
                                       "note": "no sampling variance; range over experiments"}
                    continue
                pooled[w][c][k] = meta.dl(y, se)
    rng2 = np.random.default_rng(MD.SEED + 14)
    rel = {w: split_half(D, rng2, k) for w, k in WIDTHS.items()}
    # pooled harmonic profile for the figure (plain mean over the six experiments), 4.5-9.5 semitones
    m = (G >= 3.0) & (G <= 9.75)
    fig = {"grid": G[m].tolist()}
    for w in ("sd0.1", "sd0.2"):
        beta = {(mm, e): DC.fit(P0["sd0.2"], [o for o in DC.US if o != e], keys, TRAIN)
                for mm, keys in MODELS.items() for e in HARM6}
        fig[w] = {"listener": np.mean([P0[w][e]["z"][m] for e in HARM6], axis=0).tolist()}
        for mm, keys in MODELS.items():
            fig[w][mm] = np.mean([DC.predict(P0[w][e], keys, beta[(mm, e)], TRAIN)[m] for e in HARM6],
                                 axis=0).tolist()
    out = {"status": "exploratory", "experiments": EXPS, "harm6": HARM6, "widths": {k: list(v) for k, v in WIDTHS.items()},
           "windows": WINDOWS, "positions": POS, "reference": REF, "est": est, "boot": summ, "pooled": pooled,
           "reliability": rel, "figure": fig, "_nboot": NBOOT, "_seed": MD.SEED + 13}
    json.dump(out, open(RES / "et_ji_positions.json", "w"), indent=1)
    w = "sd0.1"
    print("reliability", {w_: {e: round(r["ceiling"], 3) for e, r in rel[w_].items()} for w_ in rel})
    print("loc SD (split-half diff, semitones)", {e: round(r["locsd_m6"], 3) for e, r in rel[w].items()})
    for c in ("listener", "HK+H", "HK+H20", "Composite"):
        for loc in WINDOWS:
            v = pooled[w][c][f"loc_{loc}"]
            print(f"{w} {c:9s} {loc:3s} pooled {100 * v['est']:7.1f} c HK[{100 * v['hk_ci'][0]:6.1f},{100 * v['hk_ci'][1]:6.1f}]",
                  " ".join(f"{e}:{100 * summ[f'{w}|{e}|{c}|loc_{loc}']['est']:.0f}"
                           f"[{100 * summ[f'{w}|{e}|{c}|loc_{loc}']['ci'][0]:.0f},{100 * summ[f'{w}|{e}|{c}|loc_{loc}']['ci'][1]:.0f}]"
                           for e in EXPS))
    for c in ("listener", "HK", "HK+H", "HK+H20", "Composite"):
        g = "gap" if c == "listener" else "ugap"
        for p in ("ET", "JIa", "JIb", "emp", "JIa_minus_ET", "JIb_minus_ET"):
            v = pooled[w][c][f"{g}_{p}"]
            vm = summ[f"{w}|harm6mean|{c}|{g}_{p}"]
            print(f"{w} {c:9s} {g}_{p:13s} RE {v['est']:+.3f} HK[{v['hk_ci'][0]:+.3f},{v['hk_ci'][1]:+.3f}]"
                  f"  mean {vm['est']:+.3f} [{vm['ci'][0]:+.3f},{vm['ci'][1]:+.3f}]")


if __name__ == "__main__":
    main()
