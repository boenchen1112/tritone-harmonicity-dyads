"""
Is there a third factor? A pre-specified screen of candidate additions to the
roughness + harmonicity curve, judged only out of sample.

Rules fixed before running (2026-09-26):
  Candidates (all computed from the stimulus, no interval-specific parameter):
    Hs     harmonicity with a wider tolerance: sigma in {6.83, 10, 15, 20} cents
           (spectrum and template); sigma chosen INSIDE the leave-one-out loop
           by training fit.
    RA     root ambiguity (Terhardt/Parncutt): in the harmonicity model's
           virtual-pitch profile, (second peak - mean) / (top peak - mean),
           second peak at least 100 cents from the top one.
    PER    temporal periodicity: max over lags 2..30 ms of the normalised
           autocorrelation of the partials, r(t) = sum a^2 cos(2 pi f t) / sum a^2
           (phase-free; 30 ms ~ the lower limit of pitch, ~33 Hz).
    HE     foil computed on the f0 ratio only: harmonic entropy (ratios n/d with
           n*d <= 10000, Gaussian tolerance 17 cents, weights (n*d)^-1/2).
           Any f0-based factor predicts a dip at 600 cents for stretched tones.
  Baselines: HK, H, HK+H, revHK+H (Marjieh et al.'s components, re-weighted in
  the loop) and Marjieh et al.'s fixed composite C (one weight in the loop).
  Evaluation, leave one US experiment out (weights shared across the other nine,
  intercept per experiment; the held-out experiment contributes only its
  intercept, fitted outside the tritone/minor-sixth windows):
    * R2 of the whole held-out profile (0.5-14.75 semitones)
    * held-out unexplained gap and dip at 600 cents (mean of the six harmonic)
    * tritone residual dip minus the mean of intervals 1..14 (eight unstretched)
    * stretched tones: residual dip at 6.00 and at the stretched tritone 6.42
  Korean experiments: predicted with weights fitted on all ten US experiments.
Point estimates only; the chosen model is bootstrapped separately.

Writes results/model_screen.json and results/screen_curves.npz.
"""
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import marjieh_dyads as MD
import harmonicity as HM

RES = HERE.parent / "results"
SIGMAS = (6.83, 10.0, 15.0, 20.0)
US = [k for k, v in MD.EXPS.items() if v[3] == "US"]
KR = [k for k, v in MD.EXPS.items() if v[3] == "KR"]
HARM6 = ["harm3", "eq5", "no3", "flute", "guitar", "piano"]
US8 = [k for k in US if k not in ("str3", "comp3")]
LAGS = np.arange(0.002, 0.030, 0.00001)


# ---------------------------------------------------------------- features
def harm_sigma(freqs, amps, sigma):
    pcs = np.round(1200 * np.log2(np.asarray(freqs) / MD.F0))
    spec = HM.pc_spectrum(pcs, amps, sigma=sigma, quantise=True)
    return HM.kl_from_uniform(HM.virtual_pitch_profile(spec, 11, 1.0, sigma))


def root_ambiguity(freqs, amps):
    pcs = np.round(1200 * np.log2(np.asarray(freqs) / MD.F0))
    y = HM.virtual_pitch_profile(HM.pc_spectrum(pcs, amps, quantise=True), 11, 1.0)
    i1 = int(np.argmax(y))
    d = np.abs((np.arange(len(y)) - i1 + 600) % 1200 - 600)
    p2 = y[d >= 100].max()
    return float((p2 - y.mean()) / (y[i1] - y.mean()))


def periodicity(freqs, amps):
    f = np.asarray(freqs, float)
    w = np.asarray(amps, float) ** 2
    r = (w[None, :] * np.cos(2 * np.pi * LAGS[:, None] * f[None, :])).sum(1) / w.sum()
    return float(r.max())


def harmonic_entropy_curve(cents, N=10000, s=17.0):
    fr = []
    for d in range(1, 101):
        for n in range(d, int(2.6 * d) + 2):
            if n * d <= N and np.gcd(n, d) == 1:
                fr.append((n, d))
    fr = np.array(fr, float)
    c = 1200 * np.log2(fr[:, 0] / fr[:, 1])
    w = (fr[:, 0] * fr[:, 1]) ** -0.5
    out = np.empty(len(cents))
    for i, x in enumerate(cents):
        p = w * np.exp(-0.5 * ((x - c) / s) ** 2)
        p = p / p.sum()
        p = p[p > 1e-300]
        out[i] = -np.sum(p * np.log2(p))
    return out


def feature_curves(upper, bass=None, xs=MD.FINE, he=None):
    ru, au = upper
    rb, ab = bass if bass is not None else upper
    ku, kb = au > 0, ab > 0
    out = {f"H{s:g}": np.empty(len(xs)) for s in SIGMAS}
    out.update(RA=np.empty(len(xs)), PER=np.empty(len(xs)))
    for n, iv in enumerate(xs):
        F = np.r_[MD.F0 * rb[kb], MD.F0 * 2 ** (iv / 12) * ru[ku]]
        A = np.r_[ab[kb], au[ku]]
        for s in SIGMAS:
            out[f"H{s:g}"][n] = harm_sigma(F, A, s)
        out["RA"][n] = root_ambiguity(F, A)
        out["PER"][n] = periodicity(F, A)
    out["HE"] = he
    return out


def all_curves():
    cache = RES / "screen_curves.npz"
    if cache.exists():
        z = np.load(cache)
        return {k: {m: z[f"{k}__{m}"] for m in z["names"]} for k in MD.EXPS}
    he = harmonic_entropy_curve(100 * MD.FINE)
    cur = {}
    for key, (label, _, _, _, folder, bass) in MD.EXPS.items():
        print("features:", label, flush=True)
        cur[key] = feature_curves(MD.timbre(folder), bass, he=he)
    names = list(next(iter(cur.values())))
    np.savez_compressed(cache, names=np.array(names), **{f"{k}__{m}": v[m] for k, v in cur.items() for m in names})
    return cur


# ---------------------------------------------------------------- profiles
def profiles(base, feats, boot_rng=None):
    """Smoothed [z, features...] per experiment on MD.GRID."""
    out = {}
    for key, (_, fname, synth, _, _, _) in MD.EXPS.items():
        d = MD.read_trials(fname, synth)
        cols = {**{k: base[key][k] for k in MD.MODEL_KEYS}, **feats[key]}
        names = list(cols)
        V = np.column_stack([d["z"]] + [np.interp(d["x"], MD.FINE, cols[k]) for k in names])
        K = MD.kernel(MD.GRID, d["x"])
        w = np.ones(len(d["x"])) if boot_rng is None else MD.boot_weights(d["pid"], boot_rng)
        S = MD.smooth(K, w, V)
        out[key] = {"z": S[:, 0], **{k: S[:, 1 + i] for i, k in enumerate(names)}}
    return out


G = MD.GRID
EVAL = (G >= MD.LO) & (G <= 14.75)
TRAIN = EVAL & (np.abs(G - 6) >= MD.WIN) & (np.abs(G - 8) >= MD.WIN)


def at(v, x):
    return float(np.interp(x, G, v))


def dip_at(v, c, u=1.0):
    return (at(v, (c - 1) * u) + at(v, (c + 1) * u)) / 2 - at(v, c * u)


def fit(P, exps, keys):
    Y, X = [], []
    for n, e in enumerate(exps):
        D = np.zeros((TRAIN.sum(), len(exps)))
        D[:, n] = 1
        Y.append(P[e]["z"][TRAIN])
        X.append(np.column_stack([D] + [P[e][k][TRAIN] for k in keys]))
    b, *_ = np.linalg.lstsq(np.vstack(X), np.concatenate(Y), rcond=None)
    Xall, Yall = np.vstack(X), np.concatenate(Y)
    sse = float(np.sum((Yall - Xall @ b) ** 2))
    return b[len(exps):], sse


def predict(Pe, keys, beta):
    m = np.column_stack([Pe[k] for k in keys]) @ beta
    return m + np.mean(Pe["z"][TRAIN] - m[TRAIN])


def evaluate(P, e, pred):
    h = P[e]["z"]
    r = h - pred
    o = {"r2": 1 - np.sum(r[EVAL] ** 2) / np.sum((h[EVAL] - h[EVAL].mean()) ** 2),
         "unexpl_gap": (at(h, 8) - at(h, 6)) - (at(pred, 8) - at(pred, 6)),
         "unexpl_dip": dip_at(h, 6) - dip_at(pred, 6),
         "udip_null": [dip_at(h, c) - dip_at(pred, c) for c in range(1, 15)]}
    if e.startswith("str3") or e.startswith("comp3"):
        u = np.log2(2.1 if e.startswith("str3") else 1.9)
        o["udip_ssc"] = dip_at(h, 6, u) - dip_at(pred, 6, u)
        o["pdip_ssc"] = dip_at(pred, 6, u)
        o["pdip_600"] = dip_at(pred, 6)
    return o


def lodo(P, keys_options):
    """keys_options: list of feature lists; the option with the smallest
    training SSE is chosen inside each fold."""
    res, chosen = {}, {}
    for e in US:
        others = [o for o in US if o != e]
        fits = [(fit(P, others, ks), ks) for ks in keys_options]
        (beta, _), ks = min(fits, key=lambda t: t[0][1])
        chosen[e] = ks
        res[e] = evaluate(P, e, predict(P[e], ks, beta))
    (beta, _), ks = min(((fit(P, US, ks), ks) for ks in keys_options), key=lambda t: t[0][1])
    for e in KR:
        res[e] = evaluate(P, e, predict(P[e], ks, beta))
    return res, chosen, ks, beta


MODELS = {
    "HK": [["HK"]],
    "H": [["H"]],
    "HK+H": [["HK", "H"]],
    "revHK+H": [["revHK", "H"]],
    "Composite (fixed)": [["C"]],
    "HK+H(sigma)": [["HK", f"H{s:g}"] for s in SIGMAS],
    "HK+H+RA": [["HK", "H", "RA"]],
    "HK+H+PER": [["HK", "H", "PER"]],
    "HK+PER": [["HK", "PER"]],
    "HK+H(sigma)+RA+PER": [["HK", f"H{s:g}", "RA", "PER"] for s in SIGMAS],
    "revHK+H+RA": [["revHK", "H", "RA"]],
    "revHK+H+PER": [["revHK", "H", "PER"]],
    "HK+H+HE (f0 foil)": [["HK", "H", "HE"]],
    "HK+HE (f0 foil)": [["HK", "HE"]],
}


def summarise(res):
    s = {"r2_us_mean": float(np.mean([res[e]["r2"] for e in US])),
         "r2_harm6_mean": float(np.mean([res[e]["r2"] for e in HARM6])),
         "r2_kr_mean": float(np.mean([res[e]["r2"] for e in KR])),
         "gap_harm6": float(np.mean([res[e]["unexpl_gap"] for e in HARM6])),
         "dip_harm6": float(np.mean([res[e]["unexpl_dip"] for e in HARM6])),
         "gap_us8": float(np.mean([res[e]["unexpl_gap"] for e in US8])),
         "dip_us8": float(np.mean([res[e]["unexpl_dip"] for e in US8]))}
    nul = np.mean([res[e]["udip_null"] for e in US8], axis=0)
    s["tt_minus_other"] = float(nul[5] - np.delete(nul, 5).mean())
    s["tt_rank"] = int(np.sum(nul >= nul[5]))
    s["null_mean"] = nul.tolist()
    for e in ("str3", "comp3"):
        s[f"{e}_udip600"] = res[e]["unexpl_dip"]
        s[f"{e}_udip_ssc"] = res[e]["udip_ssc"]
        s[f"{e}_pdip600"] = res[e]["pdip_600"]
        s[f"{e}_pdip_ssc"] = res[e]["pdip_ssc"]
    return s


def main():
    base, _, _ = MD.load_curves()
    feats = all_curves()
    P = profiles(base, feats)
    out = {}
    for name, opts in MODELS.items():
        res, chosen, ks_all, beta = lodo(P, opts)
        s = summarise(res)
        s["chosen"] = {e: ks for e, ks in chosen.items()}
        s["all_us_keys"], s["all_us_beta"] = ks_all, beta.tolist()
        s["per_exp"] = {e: {k: v for k, v in r.items() if k != "udip_null"} for e, r in res.items()}
        out[name] = s
        print(f"{name:22s} R2us {s['r2_us_mean']:.3f} R2harm {s['r2_harm6_mean']:.3f} R2kr {s['r2_kr_mean']:.3f} "
              f"gap6 {s['gap_harm6']:+.3f} dip6 {s['dip_harm6']:+.3f} TT-oth {s['tt_minus_other']:+.3f} "
              f"rank {s['tt_rank']:2d} | str dip600 {s['str3_udip600']:+.3f} ssc {s['str3_udip_ssc']:+.3f} "
              f"(pred {s['str3_pdip600']:+.3f}/{s['str3_pdip_ssc']:+.3f})", flush=True)
    obs = {e: {"dip600": dip_at(P[e]["z"], 6), "gap600": at(P[e]["z"], 8) - at(P[e]["z"], 6)} for e in P}
    for e, u in (("str3", 2.1), ("comp3", 1.9)):
        obs[e]["dip_ssc"] = dip_at(P[e]["z"], 6, np.log2(u))
    out["_observed"] = obs
    json.dump(out, open(RES / "model_screen.json", "w"), indent=1)


if __name__ == "__main__":
    main()
