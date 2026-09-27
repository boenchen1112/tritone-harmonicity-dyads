"""
Robustness of the third-factor screen (model_screen.py / model_screen2.py) to
three objections, all judged out of sample exactly as in the screen.

  A. Smoothing width. The 20-cent harmonicity optimum might be an interaction
     with the 0.2-semitone (20-cent) kernel that smooths ratings and model
     outputs alike. The fixed-sigma sweep is rerun with kernel SD 0.1 and 0.3
     semitones (held-out windows kept at 3 kernel SDs). Rule fixed before
     running: if the best sigma stays within 15-25 cents at both widths, the
     recommendation stands; if it moves with the kernel, it is a property of
     the analysis. R2 is not comparable across widths; only the argmax is.
  B. Which end of the gap? The held-out unexplained gap r(8) - r(6) is split
     into the residual at the tritone and at the minor sixth (r = observed -
     predicted, so a negative r(6) means the model over-rates the tritone),
     with a participant bootstrap (paired within replicates).
  C. Register. Marjieh et al. sampled the lower tone uniformly over G3-F4
     (MIDI 55-65); our model curves put it at C4. Harmonicity is invariant to
     transposition in this implementation, roughness is not, so H&K roughness
     is recomputed as the average over lower tones MIDI 55..65 (1-semitone
     steps) and the models are refitted with it.

Writes results/robustness_screen.json; with --profile, results/residual_profile.json
(held-out residual at every integer interval).
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
from dissonance_model import hutch_knopoff_dissonance, pool_partials

RES = HERE.parent / "results"
NBOOT = int(os.environ.get("NBOOT", 200))
SIGMAS = S2.ALL_SIGMAS
BASS_MIDI = np.arange(55, 66)


def hk_register(upper, bass, xs=MD.FINE):
    ru, au = upper
    rb, ab = bass if bass is not None else upper
    ku, kb = au > 0, ab > 0
    out = np.zeros(len(xs))
    for m in BASS_MIDI:
        f0 = 440.0 * 2 ** ((m - 69) / 12)
        for n, iv in enumerate(xs):
            F, A = pool_partials([(f0 * rb[kb], ab[kb]), (f0 * 2 ** (iv / 12) * ru[ku], au[ku])])
            out[n] += hutch_knopoff_dissonance(F, A) if len(F) > 1 else 0.0
    return out / len(BASS_MIDI)


def register_curves():
    cache = RES / "register_hk_curves.npz"
    if cache.exists():
        z = np.load(cache)
        return {k: z[k] for k in MD.EXPS}
    cur = {}
    for key, (label, _, _, _, folder, bass) in MD.EXPS.items():
        print("register H&K:", label, flush=True)
        cur[key] = hk_register(MD.timbre(folder), bass)
    np.savez_compressed(cache, **cur)
    return cur


def precompute(cols_by_exp, sd):
    pre = {}
    for key, (_, fname, synth, _, _, _) in MD.EXPS.items():
        d = MD.read_trials(fname, synth)
        cols = cols_by_exp[key]
        names = list(cols)
        V = np.column_stack([d["z"]] + [np.interp(d["x"], MD.FINE, cols[k]) for k in names])
        pre[key] = (MD.kernel(MD.GRID, d["x"], sd=sd), V, names, d["pid"])
    return pre


def profiles(pre, rng=None):
    out = {}
    for key, (K, V, names, pid) in pre.items():
        w = np.ones(len(pid)) if rng is None else MD.boot_weights(pid, rng)
        Sm = MD.smooth(K, w, V)
        out[key] = {"z": Sm[:, 0], **{k: Sm[:, 1 + i] for i, k in enumerate(names)}}
    return out


def set_window(win):
    S.TRAIN = S.EVAL & (np.abs(S.G - 6) >= win) & (np.abs(S.G - 8) >= win)


def ends(P, keys):
    """Held-out residuals at 6 and 8 semitones, mean over the six harmonic experiments."""
    r6, r8 = {}, {}
    for e in S.US:
        others = [o for o in S.US if o != e]
        beta, _ = S.fit(P, others, keys)
        r = P[e]["z"] - S.predict(P[e], keys, beta)
        r6[e], r8[e] = S.at(r, 6), S.at(r, 8)
    return {"r6_harm6": float(np.mean([r6[e] for e in S.HARM6])), "r8_harm6": float(np.mean([r8[e] for e in S.HARM6])),
            "r6_us8": float(np.mean([r6[e] for e in S.US8])), "r8_us8": float(np.mean([r8[e] for e in S.US8])),
            "per_exp": {e: [r6[e], r8[e]] for e in S.US}}


def main():
    base, _, _ = MD.load_curves()
    f1, f2 = S.all_curves(), S2.all_curves()
    reg = register_curves()
    cols = {k: {"HK": base[k]["HK"], "C": base[k]["C"], "HKreg": reg[k],
                **{f"H{s:g}": {**f1[k], **f2[k]}[f"H{s:g}"] for s in SIGMAS}} for k in MD.EXPS}
    out = {"_sigmas": list(SIGMAS), "_bass_midi": BASS_MIDI.tolist(), "_nboot": NBOOT}

    # A. smoothing width
    out["kernel"] = {}
    for sd in (0.1, 0.2, 0.3):
        set_window(3 * sd)
        P = profiles(precompute(cols, sd))
        row = {}
        for s in SIGMAS:
            sm = S.summarise(S.lodo(P, [["HK", f"H{s:g}"]])[0])
            row[f"{s:g}"] = {"r2_us_mean": sm["r2_us_mean"], "gap_harm6": sm["gap_harm6"], "dip_harm6": sm["dip_harm6"]}
        best = max(row, key=lambda k: row[k]["r2_us_mean"])
        infold = S.lodo(P, [["HK", f"H{s:g}"] for s in S.SIGMAS])[1]
        out["kernel"][f"{sd:g}"] = {"sweep": row, "best_sigma": float(best),
                                    "chosen_in_fold": sorted({ks[1] for ks in infold.values()})}
        print(f"kernel SD {sd}: best sigma {best}; in-fold choices {out['kernel'][f'{sd:g}']['chosen_in_fold']}; "
              + ", ".join(f"{k}:{v['r2_us_mean']:.3f}" for k, v in row.items()), flush=True)
    set_window(MD.WIN)

    # B and C at the standard width, with a participant bootstrap
    pre = precompute(cols, MD.SD)
    MODELS = {"HK": ["HK"], "HK+H": ["HK", "H6.83"], "HK+H20": ["HK", "H20"], "Composite": ["C"],
              "HKreg+H": ["HKreg", "H6.83"], "HKreg+H20": ["HKreg", "H20"]}

    def run(P):
        o = {}
        for m, ks in MODELS.items():
            sm = S.summarise(S.lodo(P, [ks])[0])
            o[m] = {"r2_us_mean": sm["r2_us_mean"], "gap_harm6": sm["gap_harm6"], "dip_harm6": sm["dip_harm6"],
                    "tt_minus_other": sm["tt_minus_other"], **{k: v for k, v in ends(P, ks).items() if k != "per_exp"}}
        return o

    est = run(profiles(pre))
    rng = np.random.default_rng(MD.SEED + 7)
    boots = []
    for b in range(NBOOT):
        boots.append(run(profiles(pre, rng)))
        if b % 50 == 0:
            print("boot", b, flush=True)
    out["models"] = {}
    for m in MODELS:
        out["models"][m] = {k: {"est": v, **MD.ci([bb[m][k] for bb in boots])} for k, v in est[m].items()}
    out["diffs"] = {}
    for a, b in (("HKreg+H", "HK+H"), ("HKreg+H20", "HK+H20")):
        out["diffs"][f"{a} - {b}"] = {k: {"est": est[a][k] - est[b][k],
                                          **MD.ci([bb[a][k] - bb[b][k] for bb in boots])} for k in est[a]}
    json.dump(out, open(RES / "robustness_screen.json", "w"), indent=1)
    for m, v in out["models"].items():
        print(m, {k: (round(x["est"], 3), [round(c, 3) for c in x["ci"]]) for k, x in v.items()})
    for d, v in out["diffs"].items():
        print(d, {k: (round(x["est"], 3), [round(c, 3) for c in x["ci"]]) for k, x in v.items()})


def residual_profile():
    """Held-out residual at every integer interval 1..14 (point estimates, standard width):
    mean of the six harmonic experiments, and pure tones. Writes results/residual_profile.json."""
    base, _, _ = MD.load_curves()
    f1 = S.all_curves()
    cols = {k: {"HK": base[k]["HK"], "C": base[k]["C"], "H6.83": f1[k]["H6.83"], "H20": f1[k]["H20"]}
            for k in MD.EXPS}
    P = profiles(precompute(cols, MD.SD))
    out = {}
    for m, ks in (("HK", ["HK"]), ("HK+H", ["HK", "H6.83"]), ("HK+H20", ["HK", "H20"]), ("Composite", ["C"])):
        res = {}
        for e in S.US:
            beta, _ = S.fit(P, [o for o in S.US if o != e], ks)
            r = P[e]["z"] - S.predict(P[e], ks, beta)
            res[e] = [S.at(r, c) for c in range(1, 15)]
        out[m] = {"harm6": np.mean([res[e] for e in S.HARM6], 0).tolist(), "pure": res["pure"]}
        print(m, np.round(out[m]["harm6"], 2))
    json.dump(out, open(RES / "residual_profile.json", "w"), indent=1)


if __name__ == "__main__":
    if "--profile" in sys.argv:
        residual_profile()
    else:
        main()
