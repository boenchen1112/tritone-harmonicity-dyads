"""
The largest thing the consonance models miss: a sharpness trend.

Found after the fact (exploratory). In-sample fits of H&K + harmonicity (20-cent
tolerance) to single experiments left a broad residual over the whole interval
axis: intervals of 3-9 semitones rated above the curve, intervals beyond about
9.5 semitones, and above all beyond the octave, below it (-0.3 to -0.6 SD).
A linear term in interval size removed most of it. Interval size is confounded
with the pitch of the upper tone and with the sharpness (high-frequency
weighting) of the dyad, so three stimulus features were compared:

  X1     interval size in semitones (equivalently, upper-tone log frequency)
  CEN    log2 amplitude-weighted spectral centroid
  SHARP  a sharpness proxy: loudness-weighted (amplitude^0.6) mean Bark rate of
         the partials with Zwicker's high-frequency weighting
         g(z) = 1 for z < 15.8 Bark, 0.066 exp(0.171 z) above

SHARP was selected on the leave-one-US-experiment-out screen (model_screen.py
rules). It is then tested on data not used for selection: Marjieh et al.'s
roll-off experiment, in which the spectral slope of harmonic dyads varies from
0 to 15 dB/octave across trials of the same participants. If the residual
trend is sharpness, its slope over interval size must shrink as the roll-off
steepens, and a single shared sharpness weight must account for it better than
a single shared slope in interval size; with both in the model, the
interval-size term should vanish.

Participant bootstrap (NBOOT) for: the leave-one-out comparison of
HK + H20 with and without SHARP (paired), and the shared-weight roll-off fits.
Writes results/sharpness.json and caches results/sharp_curves.npz.
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
import model_boot as MB

RES = HERE.parent / "results"
NBOOT = int(os.environ.get("NBOOT", 300))
NAMES = ["X1", "CEN", "SHARP"]
ROLL_LEV = (1.0, 4.0, 7.0, 10.0, 13.0)


# ---------------------------------------------------------------- features
def bark(f):
    return 13 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500) ** 2)


def spectral_features(F, A, x):
    F = np.asarray(F, float)
    A = np.asarray(A, float)
    z = bark(F)
    L = A ** 0.6
    g = np.where(z < 15.8, 1.0, 0.066 * np.exp(0.171 * z))
    return {"X1": x, "CEN": np.log2(np.sum(F * A ** 2) / np.sum(A ** 2)),
            "SHARP": np.sum(z * g * L) / np.sum(L)}


def feature_curves(upper, bass=None, xs=MD.FINE):
    ru, au = upper
    rb, ab = bass if bass is not None else upper
    out = {k: np.empty(len(xs)) for k in NAMES}
    for n, iv in enumerate(xs):
        f = spectral_features(np.r_[MD.F0 * rb, MD.F0 * 2 ** (iv / 12) * ru], np.r_[ab, au], iv)
        for k in NAMES:
            out[k][n] = f[k]
    return out


def all_curves():
    cache = RES / "sharp_curves.npz"
    if cache.exists():
        z = np.load(cache)
        return {k: {m: z[f"{k}__{m}"] for m in NAMES} for k in MD.EXPS}
    cur = {key: feature_curves(MD.timbre(v[4]), v[5]) for key, v in MD.EXPS.items()}
    np.savez_compressed(cache, **{f"{k}__{m}": v[m] for k, v in cur.items() for m in NAMES})
    return cur


# ---------------------------------------------------------------- dense screen
MODELS = {"HK+H20": [["HK", "H20"]], "HK+H20+X1": [["HK", "H20", "X1"]], "HK+H20+CEN": [["HK", "H20", "CEN"]],
          "HK+H20+SHARP": [["HK", "H20", "SHARP"]], "HK+H20+SHARP+X1": [["HK", "H20", "SHARP", "X1"]],
          "HK+H+SHARP": [["HK", "H", "SHARP"]], "revHK+H": [["revHK", "H"]],
          "revHK+H+SHARP": [["revHK", "H", "SHARP"]], "Composite (fixed)": [["C"]],
          "Composite+SHARP": [["C", "SHARP"]]}
BOOT_MODELS = ("HK+H20", "HK+H20+SHARP", "Composite (fixed)", "Composite+SHARP")
METRICS = ("r2_us_mean", "r2_harm6_mean", "r2_kr_mean", "gap_harm6", "dip_harm6", "tt_minus_other",
           "str3_udip600", "str3_udip_ssc")


def summ(res):
    s = S.summarise(res)
    s["per_exp_r2"] = {e: float(res[e]["r2"]) for e in res}
    return s


def insample(P, e, ks):
    E = S.EVAL
    h = P[e]["z"][E]
    X = np.column_stack([np.ones(E.sum())] + [P[e][k][E] for k in ks])
    b, *_ = np.linalg.lstsq(X, h, rcond=None)
    return float(1 - np.sum((h - X @ b) ** 2) / np.sum((h - h.mean()) ** 2)), b[1:].tolist()


def residual_profile(P, ks):
    rows = {}
    for e in S.US:
        beta, _ = S.fit(P, [o for o in S.US if o != e], ks)
        r = P[e]["z"] - S.predict(P[e], ks, beta)
        rows[e] = [S.at(r, c) for c in range(1, 15)]
    return {"harm6": np.mean([rows[e] for e in S.HARM6], axis=0).tolist(), "pure": rows["pure"]}


# ---------------------------------------------------------------- roll-off test
def rolloff_design():
    _, RC, RL = MD.load_curves()
    d = MD.read_trials(MD.ROLLOFF_FILE, xcol="intervals")
    V = MD.rolloff_values(d, RC, RL)  # z, HK, Seth, revHK, H, H_incon, C
    sh = np.empty(len(d["x"]))
    for n, (x, db) in enumerate(zip(d["x"], d["rolloff"])):
        r, a = MD.harmonic_rolloff(db)
        sh[n] = spectral_features(np.r_[MD.F0 * r, MD.F0 * 2 ** (x / 12) * r], np.r_[a, a], x)["SHARP"]
    V = np.column_stack([V, d["x"], sh])
    K = MD.kernel(MD.GRID, d["x"])
    W = [np.exp(-0.5 * ((d["rolloff"] - lev) / MD.ROLLOFF_SD) ** 2) for lev in ROLL_LEV]
    return K, V, W, d["pid"]


ROLL_MODELS = {"HK+H": [1, 4], "HK+H+X1": [1, 4, 7], "HK+H+SHARP": [1, 4, 8], "HK+H+X1+SHARP": [1, 4, 7, 8]}
EV = (MD.GRID >= MD.LO) & (MD.GRID <= 14.75)


def rolloff_fit(K, V, W, pid, rng=None):
    bw = np.ones(len(pid)) if rng is None else MD.boot_weights(pid, rng)
    SL = [MD.smooth(K, w * bw, V)[EV] for w in W]
    out = {}
    for name, cols in ROLL_MODELS.items():
        Y, X = [], []
        for i, Sm in enumerate(SL):
            D = np.zeros((len(Sm), len(SL)))
            D[:, i] = 1
            Y.append(Sm[:, 0])
            X.append(np.column_stack([D] + [Sm[:, c] for c in cols]))
        b, *_ = np.linalg.lstsq(np.vstack(X), np.concatenate(Y), rcond=None)
        r2 = [1 - np.sum((Sm[:, 0] - X[i] @ b) ** 2) / np.sum((Sm[:, 0] - Sm[:, 0].mean()) ** 2)
              for i, Sm in enumerate(SL)]
        out[name] = {"r2_mean": float(np.mean(r2)), "r2": [float(v) for v in r2], "beta": b[len(SL):].tolist()}
    # separate fits per slice: slope over interval size (per octave) with HK+H+X1
    slopes = []
    for Sm in SL:
        X = np.column_stack([np.ones(len(Sm)), Sm[:, 1], Sm[:, 4], Sm[:, 7]])
        b, *_ = np.linalg.lstsq(X, Sm[:, 0], rcond=None)
        slopes.append(float(12 * b[3]))
    out["slice_x_slope_per_octave"] = slopes
    return out


# ---------------------------------------------------------------- main
def main():
    base, _, _ = MD.load_curves()
    f1 = S.all_curves()
    f3 = all_curves()
    feats = {k: {**f1[k], **f3[k]} for k in MD.EXPS}
    pre = MB.precompute(base, feats)
    P = MB.profiles(pre)
    out = {"screen": {}, "insample": {}}
    for name, opts in MODELS.items():
        res, _, _, beta = S.lodo(P, opts)
        s = summ(res)
        s["all_us_beta"] = beta.tolist()
        out["screen"][name] = s
        print(f"{name:18s} US {s['r2_us_mean']:.3f} harm6 {s['r2_harm6_mean']:.3f} KR {s['r2_kr_mean']:.3f} "
              f"gap6 {s['gap_harm6']:+.3f} dip6 {s['dip_harm6']:+.3f} TT-oth {s['tt_minus_other']:+.3f} "
              f"str {s['str3_udip600']:+.3f}/{s['str3_udip_ssc']:+.3f} beta {np.round(beta, 3)}", flush=True)
    for ks in (["HK", "H20"], ["HK", "H20", "X1"], ["HK", "H20", "SHARP"]):
        out["insample"]["+".join(ks)] = {e: insample(P, e, ks) for e in MD.EXPS}
    out["resid_profile"] = {"HK+H20": residual_profile(P, ["HK", "H20"]),
                            "HK+H20+SHARP": residual_profile(P, ["HK", "H20", "SHARP"])}

    K, V, W, pid = rolloff_design()
    roll_est = rolloff_fit(K, V, W, pid)
    print("roll-off:", {k: (v["r2_mean"], np.round(v["beta"], 3).tolist()) if isinstance(v, dict) else np.round(v, 3)
                        for k, v in roll_est.items()}, flush=True)

    rng = np.random.default_rng(MD.SEED + 7)
    boots, rboots = [], []
    for b in range(NBOOT):
        Pb = MB.profiles(pre, rng)
        boots.append({m: {k: S.summarise(S.lodo(Pb, MODELS[m])[0])[k] for k in METRICS} for m in BOOT_MODELS})
        rboots.append(rolloff_fit(K, V, W, pid, rng))
        if b % 25 == 0:
            print("boot", b, flush=True)
    est = {m: {k: out["screen"][m][k] for k in METRICS} for m in BOOT_MODELS}
    out["boot"] = {m: {k: {"est": est[m][k], **MD.ci([bb[m][k] for bb in boots])} for k in METRICS}
                   for m in BOOT_MODELS}
    out["boot_diffs"] = {}
    for a, c in (("HK+H20+SHARP", "HK+H20"), ("Composite+SHARP", "Composite (fixed)"),
                 ("HK+H20+SHARP", "Composite (fixed)")):
        out["boot_diffs"][f"{a} - {c}"] = {k: {"est": est[a][k] - est[c][k],
                                               **MD.ci([bb[a][k] - bb[c][k] for bb in boots])} for k in METRICS}
    rb = {}
    for name in ROLL_MODELS:
        rb[name] = {"r2_mean": {"est": roll_est[name]["r2_mean"], **MD.ci([x[name]["r2_mean"] for x in rboots])},
                    "beta": [{"est": roll_est[name]["beta"][j], **MD.ci([x[name]["beta"][j] for x in rboots])}
                             for j in range(len(roll_est[name]["beta"]))]}
    rb["SHARP_minus_X1_r2"] = {"est": roll_est["HK+H+SHARP"]["r2_mean"] - roll_est["HK+H+X1"]["r2_mean"],
                               **MD.ci([x["HK+H+SHARP"]["r2_mean"] - x["HK+H+X1"]["r2_mean"] for x in rboots])}
    rb["slice_x_slope_per_octave"] = [{"est": roll_est["slice_x_slope_per_octave"][i],
                                       **MD.ci([x["slice_x_slope_per_octave"][i] for x in rboots])}
                                      for i in range(len(ROLL_LEV))]
    rb["slope_1_minus_13"] = {"est": roll_est["slice_x_slope_per_octave"][0] - roll_est["slice_x_slope_per_octave"][-1],
                              **MD.ci([x["slice_x_slope_per_octave"][0] - x["slice_x_slope_per_octave"][-1]
                                       for x in rboots])}
    rb["levels"] = ROLL_LEV
    out["rolloff"] = rb
    out["_nboot"] = NBOOT
    out["ceiling"] = json.load(open(RES / "ceiling.json"))
    json.dump(out, open(RES / "sharpness.json", "w"), indent=1)
    for k, v in out["boot_diffs"].items():
        print(k, {m: (round(x["est"], 3), [round(c, 3) for c in x["ci"]]) for m, x in v.items()})
    print("roll-off boot:", json.dumps({k: v for k, v in rb.items() if k != "levels"}, default=float)[:2000])


if __name__ == "__main__":
    main()
