"""
Follow-up to model_screen.py (added after seeing its results, so exploratory):

  1. The pre-specified tolerance grid chose its edge (sigma = 20 cents) in every
     fold. Extend the grid to 25, 30, 40 and 60 cents to locate the optimum.
  2. Self-template harmonicity: the virtual-pitch template is the timbre's own
     partial pattern (the upper tone's partial ratios and amplitudes) instead of
     an ideal harmonic series. For harmonic timbres it is close to ordinary
     harmonicity; for stretched and compressed timbres the template is stretched
     or compressed with them, so the "harmonic" intervals move to the timbre's own
     scale. The pitch-class circle wraps at the tone's own octave: the ratio of
     its first two partials when that lies between 1.8 and 2.2, else 2 (so a
     stretched harmonic tone is scored exactly as a harmonic tone is on a
     stretched pitch axis). This formalises the idea that harmonic templates are
     learned from the sounds heard (Terhardt 1974; Shamma & Klein 2000).

Same leave-one-experiment-out rules as model_screen.py. Writes
results/model_screen2.json and results/screen2_curves.npz.
"""
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import marjieh_dyads as MD
import harmonicity as HM
import model_screen as S

RES = HERE.parent / "results"
EXT_SIGMAS = (25.0, 30.0, 40.0, 60.0)
SELF_SIGMAS = (6.83, 20.0)


def own_octave(ru, au):
    r = ru[au > 0]
    q = r[1] / r[0] if len(r) > 1 else 2.0
    return float(np.log2(q)) if 1.8 < q < 2.2 else 1.0


def self_template(ru, au, sigma):
    keep = au > 0
    r, a = ru[keep], au[keep]
    beta = own_octave(ru, au)
    t = HM.pc_spectrum(np.round(1200 * np.log2(r / r[0]) / beta), a, sigma=sigma, quantise=True)
    return np.conj(np.fft.fft(t)), np.linalg.norm(t), beta


def harm_self(F, A, tmpl, sigma):
    tf, tn, beta = tmpl
    spec = HM.pc_spectrum(np.round(1200 * np.log2(F / MD.F0) / beta), A, sigma=sigma, quantise=True)
    y = np.real(np.fft.ifft(np.fft.fft(spec) * tf)) / (tn * np.linalg.norm(spec))
    y = np.clip(y, 1e-12, None)
    return HM.kl_from_uniform(y)


def curves(upper, bass=None, xs=MD.FINE):
    ru, au = upper
    rb, ab = bass if bass is not None else upper
    ku, kb = au > 0, ab > 0
    tm = {s: self_template(ru, au, s) for s in SELF_SIGMAS}
    out = {f"H{s:g}": np.empty(len(xs)) for s in EXT_SIGMAS}
    out.update({f"Hself{s:g}": np.empty(len(xs)) for s in SELF_SIGMAS})
    for n, iv in enumerate(xs):
        F = np.r_[MD.F0 * rb[kb], MD.F0 * 2 ** (iv / 12) * ru[ku]]
        A = np.r_[ab[kb], au[ku]]
        for s in EXT_SIGMAS:
            out[f"H{s:g}"][n] = S.harm_sigma(F, A, s)
        for s in SELF_SIGMAS:
            out[f"Hself{s:g}"][n] = harm_self(F, A, tm[s], s)
    return out


def all_curves():
    cache = RES / "screen2_curves.npz"
    if cache.exists():
        z = np.load(cache)
        return {k: {m: z[f"{k}__{m}"] for m in z["names"]} for k in MD.EXPS}
    cur = {}
    for key, (label, _, _, _, folder, bass) in MD.EXPS.items():
        print("features:", label, flush=True)
        cur[key] = curves(MD.timbre(folder), bass)
    names = list(next(iter(cur.values())))
    np.savez_compressed(cache, names=np.array(names), **{f"{k}__{m}": v[m] for k, v in cur.items() for m in names})
    return cur


ALL_SIGMAS = S.SIGMAS + EXT_SIGMAS
MODELS = {
    "HK+H": [["HK", "H"]],
    "HK+H(sigma, pre-specified grid)": [["HK", f"H{s:g}"] for s in S.SIGMAS],
    "HK+H(sigma, extended grid)": [["HK", f"H{s:g}"] for s in ALL_SIGMAS],
    "HK+Hself": [["HK", "Hself6.83"]],
    "HK+Hself(20)": [["HK", "Hself20"]],
    "HK+H+Hself": [["HK", "H", "Hself6.83"]],
    "HK+H(20)+Hself(20)": [["HK", "H20", "Hself20"]],
    "Composite (fixed)": [["C"]],
}
for s in ALL_SIGMAS:
    MODELS[f"HK+H{s:g} (fixed sigma)"] = [["HK", f"H{s:g}"]]


def main():
    base, _, _ = MD.load_curves()
    f1 = S.all_curves()
    f2 = all_curves()
    feats = {k: {**f1[k], **f2[k]} for k in MD.EXPS}
    P = S.profiles(base, feats)
    out = {}
    for name, opts in MODELS.items():
        res, chosen, ks_all, beta = S.lodo(P, opts)
        s = S.summarise(res)
        s["chosen"] = chosen
        s["all_us_keys"], s["all_us_beta"] = ks_all, beta.tolist()
        s["per_exp"] = {e: {k: v for k, v in r.items() if k != "udip_null"} for e, r in res.items()}
        out[name] = s
        print(f"{name:34s} R2us {s['r2_us_mean']:.3f} R2harm {s['r2_harm6_mean']:.3f} R2kr {s['r2_kr_mean']:.3f} "
              f"gap6 {s['gap_harm6']:+.3f} dip6 {s['dip_harm6']:+.3f} TT-oth {s['tt_minus_other']:+.3f} "
              f"rank {s['tt_rank']:2d} | str dip600 {s['str3_udip600']:+.3f} ssc {s['str3_udip_ssc']:+.3f} "
              f"(pred {s['str3_pdip600']:+.3f}/{s['str3_pdip_ssc']:+.3f}) chosen {sorted(set(v[-1] for v in chosen.values()))}",
              flush=True)
    json.dump(out, open(RES / "model_screen2.json", "w"), indent=1)


if __name__ == "__main__":
    main()
