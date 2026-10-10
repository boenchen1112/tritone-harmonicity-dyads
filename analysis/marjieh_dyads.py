"""
The tritone against dense dyad-rating data (Marjieh, van Rijn, Sucholutsky,
Sumers, Lee, Griffiths & Jacoby 2024, Nature Communications 15:1482; data and
code: OSF 83w2b, gitlab.com/pmcharrison/timbre-and-consonance-paper).

Question. The "dissonance graph" (a roughness curve) places the tritone (6.00
semitones) no higher than the minor sixth (8.00). Listeners disagree. How much
of the listeners' tritone--minor-sixth difference does each model reproduce,
what is left over at the tritone, is the leftover special to the tritone, and
does it behave like a spectral effect or like a learned, interval-bound one?

Pipeline (follows Marjieh et al. where they specify it):
  * Ratings z-scored within participant; Nadaraya-Watson smoothing with a
    Gaussian kernel, SD 0.2 semitones; participant bootstrap (1000 reps).
  * Models evaluated with the bass at C4 (MIDI 60) on a 0.005-semitone grid,
    then interpolated to every trial's interval and smoothed with the SAME
    kernel and the SAME (bootstrap-weighted) design as the ratings.
    Our Python H&K and harmonicity reproduce Marjieh et al.'s published model
    profiles to r >= .9999 (validate()).
  * Calibration: pleasantness profile ~ a + b.model on the grid EXCLUDING
    windows of +/-0.6 semitones (3 kernel SDs) around 6 and 8, and excluding
    intervals below 0.5 (the unison edge). The fit therefore never sees the
    two test intervals; residuals there are out-of-window predictions.
  * Specialness null: the same held-out residual computed with the window
    centred on every integer 1..14 and every quarter tone in between.
Every quantity is recomputed inside every bootstrap replicate.

Outputs results/marjieh_dyads.json.
"""
import csv
import glob
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import PROJECT_ROOT, EXTERNAL
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = PROJECT_ROOT
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import harmonicity as HM
from dissonance_model import hutch_knopoff_dissonance, sethares_dissonance, pool_partials

EXT = EXTERNAL / "timbre-and-consonance-paper"
DATA = EXT / "input" / "data-csv" / "Rating"
BATCH = EXT / "output" / "batches"
RES = HERE.parent / "results"
RES.mkdir(exist_ok=True)

F0 = 261.6255653005986  # C4
SD = 0.2                # smoothing bandwidth (semitones), as Marjieh et al.
WIN = 0.6               # half-width of held-out windows (3 kernel SDs)
LO = 0.5                # fit region starts here (unison edge excluded)
TT, M6 = 6.0, 8.0
NBOOT = int(os.environ.get("NBOOT", 1000))
SEED = 20260926
# Marjieh et al. (2024) final composite: revised H&K (slow-beat boundary p,
# strength q, amplitude power r) minus-weighted, plus 0.837 x harmonicity.
P_REV, Q_REV, R_REV, W_HARM = 0.096, 1.632, 1.359, 0.837

FINE = np.round(np.arange(0, 15.0001, 0.005), 3)
GRID = np.round(np.arange(0, 15.0001, 0.01), 2)

# key: (label, csv, synth filter, cohort, batch folder, bass timbre override)
EXPS = {
    "harm3": ("Harmonic, 3 dB/oct", "rating_dyh3dd.csv", None, "US", "Harmonic dyads (3 dB roll-off)", None),
    "str3": ("Stretched (2.1), 3 dB/oct", "rating_dys3dd.csv", None, "US", "Stretched dyads (3 dB roll-off)", None),
    "comp3": ("Compressed (1.9), 3 dB/oct", "rating_dyc3dd.csv", None, "US", "Compressed dyads (3 dB roll-off)", None),
    "eq5": ("5 equal harmonics", "rating_w3rdd.csv", None, "US", "Harmonic dyads (5 equal harmonics)", None),
    "no3": ("5 harmonics, no 3rd", "rating_wo3rdd.csv", None, "US", "Harmonic dyads (no 3rd harmonic)", None),
    "pure": ("Pure tones", "pure_dyad_purdyrt.csv", None, "US", "Pure dyads", None),
    "bonang": ("Bonang (over harmonic bass)", "gamelan_dyad_gamdyrt.csv", None, "US", "Bonang dyads",
               (np.array([1.0, 2.0, 3.0, 4.0]), np.ones(4))),
    "flute": ("Flute (sampled)", "rating_flute_harmonic_harflt.csv", "flute", "US", "Flute dyads", None),
    "guitar": ("Guitar (sampled)", "rating_guitar_harmonic_hargtr.csv", "guitar", "US", "Guitar dyads", None),
    "piano": ("Piano (sampled)", "rating_piano_harmonic_harpno.csv", "piano", "US", "Piano dyads", None),
    "harm3_kr": ("Harmonic, 3 dB/oct (Korean)", "korean_dyad_harm.csv", None, "KR",
                 "Harmonic dyads (3 dB roll-off) (Korean)", None),
    "str3_kr": ("Stretched (2.1) (Korean)", "korean_dyad_str.csv", None, "KR",
                "Stretched dyads (3 dB roll-off) (Korean)", None),
    "comp3_kr": ("Compressed (1.9) (Korean)", "korean_dyad_comp.csv", None, "KR",
                 "Compressed dyads (3 dB roll-off) (Korean)", None),
}
ROLLOFF_FILE = "rolloff_dyad_rodyrt.csv"
ROLLOFF_SLICES = (2.0, 7.0, 12.0)
ROLLOFF_SD = 1.5


# ---------------------------------------------------------------- spectra & models
def timbre(folder):
    rows = list(csv.DictReader(open(glob.glob(str(BATCH / folder / "timbre" / "*.csv"))[0])))
    return (np.array([float(r["frequency"]) for r in rows]), np.array([float(r["amplitude"]) for r in rows]))


def harmonic_rolloff(db):
    i = np.arange(1, 11)
    return i.astype(float), 10 ** (-db * np.log2(i) / 20)


def revised_hk(freqs, amps, p=P_REV, q=Q_REV, r=R_REV):
    """Marjieh et al.'s revised H&K (their eq. 6 and 8): slow-beat pleasantness
    below p critical bandwidths and amplitude power r. Transcribed from
    src/custom_hutchinson_knopoff_model.R (cutoff 1.2 retained there)."""
    f = np.asarray(freqs, float)
    a = np.asarray(amps, float)
    i, j = np.triu_indices(len(f), 1)
    y = np.abs(f[i] - f[j]) / (1.72 * ((f[i] + f[j]) / 2) ** 0.65)
    g = ((y / 0.25) * np.exp(1 - y / 0.25)) ** 2
    g = np.where(y > 1.2, 0.0, g)
    slow = (y / p) * g + (1 - y / p) * -q * (1 + np.sin(-np.pi / 2 + y * 2 * np.pi / p))
    d = np.where(y > p, g, slow)
    return float(np.sum((a[i] * a[j]) ** (r / 2) * d) / np.sum(a ** r))


def harm_marjieh(freqs, amps):
    """Harrison-Pearce harmonicity as run by Marjieh et al.: partial amplitudes
    as pitch-class weights (coherent sum per cent), template = 11 harmonics
    with 1/i amplitudes (hrep default), sigma 6.83 cents, KL from uniform."""
    pcs = np.round(1200 * np.log2(np.asarray(freqs) / F0))
    spec = HM.pc_spectrum(pcs, amps, quantise=True)
    return HM.kl_from_uniform(HM.virtual_pitch_profile(spec, 11, 1.0))


def harm_incon(freqs, amps):
    """Robustness variant: incon's har_18 template (i^-0.75) and weights
    re-expressed on the same scale (amplitude^1)."""
    pcs = np.round(1200 * np.log2(np.asarray(freqs) / F0))
    spec = HM.pc_spectrum(pcs, amps, quantise=True)
    return HM.kl_from_uniform(HM.virtual_pitch_profile(spec, 11, 0.75))


MODEL_FNS = {"HK": hutch_knopoff_dissonance, "Seth": sethares_dissonance, "revHK": revised_hk,
             "H": harm_marjieh, "H_incon": harm_incon}


def model_curves(upper, bass=None, xs=FINE):
    ru, au = upper
    rb, ab = bass if bass is not None else upper
    keep_u, keep_b = au > 0, ab > 0
    out = {k: np.empty(len(xs)) for k in MODEL_FNS}
    for n, iv in enumerate(xs):
        F, A = pool_partials([(F0 * rb[keep_b], ab[keep_b]), (F0 * 2 ** (iv / 12) * ru[keep_u], au[keep_u])])
        for k, fn in MODEL_FNS.items():
            if k.startswith("H"):
                out[k][n] = fn(np.r_[F0 * rb[keep_b], F0 * 2 ** (iv / 12) * ru[keep_u]],
                               np.r_[ab[keep_b], au[keep_u]])
            else:
                out[k][n] = fn(F, A) if len(F) > 1 else 0.0
    # pleasantness orientation: roughness enters negated
    out["C"] = -out["revHK"] + W_HARM * out["H"]
    return out


# ---------------------------------------------------------------- data
def read_trials(fname, synth=None, xcol="v1"):
    rows = list(csv.DictReader(open(DATA / fname, encoding="utf-8")))
    if synth is not None:
        rows = [r for r in rows if r["synth"] == synth]
    pid = np.array([r["participant_id"] for r in rows])
    x = np.array([float(r[xcol]) for r in rows])
    y = np.array([float(r["rating"]) for r in rows])
    mexp = np.array([float(r["musical_exp"]) if r.get("musical_exp") not in (None, "", "NA") else np.nan
                     for r in rows])
    extra = {k: np.array([float(r[k]) for r in rows]) for k in ("rolloff",) if k in rows[0]}
    # z-score within participant (R's scale(): sample SD); constant raters dropped
    z = np.full(len(y), np.nan)
    raw_sd = []
    for p in np.unique(pid):
        m = pid == p
        s = y[m].std(ddof=1) if m.sum() > 1 else 0
        if s > 0:
            z[m] = (y[m] - y[m].mean()) / s
            raw_sd.append(s)
    ok = ~np.isnan(z)
    d = {"pid": pid[ok], "x": x[ok], "z": z[ok], "mexp": mexp[ok], "raw_sd_mean": float(np.mean(raw_sd)),
         "rating_range": [float(y.min()), float(y.max())]}
    d.update({k: v[ok] for k, v in extra.items()})
    return d


# ---------------------------------------------------------------- smoothing & fitting
def kernel(grid, x, sd=SD):
    return np.exp(-0.5 * ((grid[:, None] - x[None, :]) / sd) ** 2)


def smooth(K, w, V):
    """Weighted Nadaraya-Watson: columns of V smoothed with trial weights w."""
    num = K @ (w[:, None] * V)
    den = K @ w
    return num / den[:, None]


FIT_MASK_MAIN = (GRID >= LO) & (GRID <= 15 - 0.25) & (np.abs(GRID - TT) >= WIN) & (np.abs(GRID - M6) >= WIN)
IDX = {c: int(np.argmin(np.abs(GRID - c))) for c in np.round(np.arange(0, 15.001, 0.25), 2)}
I5, I6, I7, I8 = IDX[5.0], IDX[TT], IDX[7.0], IDX[M6]
NULL_CENTRES = np.arange(1, 15)  # integer intervals 1..14 (both neighbours inside 0..15)
NULL_MASKS = [(GRID >= LO) & (GRID <= 14.75) & (np.abs(GRID - c) >= WIN) for c in NULL_CENTRES]

FITS = {"HK": ["HK"], "Seth": ["Seth"], "H": ["H"], "HK+H": ["HK", "H"], "revHK+H": ["revHK", "H"],
        "HK+H_incon": ["HK", "H_incon"], "Composite": ["C"]}
NULL_FITS = ("HK", "HK+H", "Composite")
MODEL_KEYS = ["HK", "Seth", "revHK", "H", "H_incon", "C"]
SUB = slice(0, None, 5)  # 0.05-semitone grid stored for the pooled fits


def ols_pred(h, M, mask):
    A = np.column_stack([np.ones(len(h))] + M)
    b, *_ = np.linalg.lstsq(A[mask], h[mask], rcond=None)
    return A @ b, b


def dip(v, c):
    """How far interval c sits below the mean of its two semitone neighbours."""
    return (v[IDX[c - 1.0]] + v[IDX[c + 1.0]]) / 2 - v[IDX[float(c)]]


def summarise(S):
    """S: smoothed profiles, columns [z, *MODEL_KEYS]. Pleasantness orientation:
    gap = P(m6) - P(TT) > 0 and dip = mean(P4, P5) - P(TT) > 0 both mean the
    tritone is heard as less pleasant."""
    h = S[:, 0]
    mod = {k: S[:, 1 + n] for n, k in enumerate(MODEL_KEYS)}
    o = {"obs_P6": h[I6], "obs_P8": h[I8], "obs_gap": h[I8] - h[I6], "obs_dip": dip(h, TT)}
    for k in ("HK", "Seth", "revHK", "H", "C"):
        o[f"raw_{k}_6"] = mod[k][I6]
        o[f"raw_{k}_8"] = mod[k][I8]
    for name, keys in FITS.items():
        pred, b = ols_pred(h, [mod[k] for k in keys], FIT_MASK_MAIN)
        o[f"{name}_pred_gap"] = pred[I8] - pred[I6]
        o[f"{name}_unexpl_gap"] = o["obs_gap"] - o[f"{name}_pred_gap"]
        o[f"{name}_share"] = o[f"{name}_pred_gap"] / o["obs_gap"]
        o[f"{name}_pred_dip"] = dip(pred, TT)
        o[f"{name}_unexpl_dip"] = o["obs_dip"] - o[f"{name}_pred_dip"]
        o[f"{name}_res6"] = h[I6] - pred[I6]
        o[f"{name}_res8"] = h[I8] - pred[I8]
        o[f"{name}_r2fit"] = 1 - np.var(h[FIT_MASK_MAIN] - pred[FIT_MASK_MAIN]) / np.var(h[FIT_MASK_MAIN])
        for n, k in enumerate(keys):
            o[f"{name}_b_{k}"] = b[1 + n]
    # specialness null: (unexplained) dip at every integer interval, each with its own held-out window
    o["obs_dip_null"] = np.array([dip(h, float(c)) for c in NULL_CENTRES])
    for name in NULL_FITS:
        keys = FITS[name]
        u = []
        for c, m in zip(NULL_CENTRES, NULL_MASKS):
            pred, _ = ols_pred(h, [mod[k] for k in keys], m)
            u.append(dip(h, float(c)) - dip(pred, float(c)))
        o[f"{name}_udip_null"] = np.array(u)
    return o


def boot_weights(pid, rng):
    ups, inv = np.unique(pid, return_inverse=True)
    counts = rng.multinomial(len(ups), np.full(len(ups), 1 / len(ups)))
    return counts[inv].astype(float)


BOOT_KEYS = ("obs_gap", "obs_dip") + tuple(f"{n}_{m}" for n in FITS for m in
                                           ("share", "unexpl_gap", "unexpl_dip", "pred_gap", "pred_dip"))


def ci(bs):
    bs = np.asarray(bs)
    return {"se": float(bs.std(ddof=1)), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}


def analyse(d, curves, rng, nboot=NBOOT, extra_w=None, keep_profiles=False, V=None):
    x = d["x"]
    if V is None:
        V = np.column_stack([d["z"]] + [np.interp(x, FINE, curves[k]) for k in MODEL_KEYS])
    K = kernel(GRID, x)
    w0 = np.ones(len(x)) if extra_w is None else extra_w
    S0 = smooth(K, w0, V)
    est = summarise(S0)
    boots, profiles = [], []
    for _ in range(nboot):
        S = smooth(K, w0 * boot_weights(d["pid"], rng), V)
        boots.append(summarise(S))
        profiles.append(S[SUB].astype(np.float32))
    out = {}
    for k, v in est.items():
        bs = np.array([b[k] for b in boots])
        if np.ndim(v) == 0:
            out[k] = {"est": float(v), **ci(bs)}
        else:
            out[k] = {"est": v.tolist(), "se": bs.std(ddof=1, axis=0).tolist()}
    # specialness: is the tritone's (unexplained) dip larger than at other integer intervals?
    i6 = list(NULL_CENTRES).index(int(TT))
    for key in ["obs_dip_null"] + [f"{n}_udip_null" for n in NULL_FITS]:
        e = np.array(est[key])
        out[key.replace("_null", "_rank")] = int(np.sum(e >= e[i6]))  # 1 = largest dip of the 14
        diffs = np.array([b[key][i6] - np.mean(np.delete(b[key], i6)) for b in boots])
        out[key.replace("_null", "_tt_minus_mean_other")] = {"est": float(e[i6] - np.mean(np.delete(e, i6))),
                                                             **ci(diffs)}
    out["_boot"] = {k: [float(b[k]) for b in boots] for k in BOOT_KEYS}
    P = np.array(profiles)
    out["profile"] = {"grid": GRID[SUB].tolist(), "human": S0[SUB, 0].tolist(),
                      "human_lo": np.percentile(P[:, :, 0], 2.5, axis=0).tolist(),
                      "human_hi": np.percentile(P[:, :, 0], 97.5, axis=0).tolist(),
                      **{k: S0[SUB, 1 + n].tolist() for n, k in enumerate(MODEL_KEYS)}}
    if keep_profiles:
        return out, S0[SUB], P
    return out


def lodo(stack, keys=("HK", "H")):
    """Common interference/harmonicity weights across experiments, with an
    intercept per experiment; each experiment predicted from weights fitted on
    the others. stack: {exp: (S_est, S_boot)} on the 0.05 grid."""
    g = GRID[SUB]
    mask = (g >= LO) & (g <= 14.75) & (np.abs(g - TT) >= WIN) & (np.abs(g - M6) >= WIN)
    cols = [1 + MODEL_KEYS.index(k) for k in keys]
    i5, i6, i7, i8 = (int(np.argmin(np.abs(g - c))) for c in (5.0, TT, 7.0, M6))

    def fit(profs):
        names = list(profs)
        Y, X = [], []
        for n, e in enumerate(names):
            S = profs[e]
            D = np.zeros((mask.sum(), len(names)))
            D[:, n] = 1
            Y.append(S[mask, 0])
            X.append(np.column_stack([D] + [S[mask, c] for c in cols]))
        b, *_ = np.linalg.lstsq(np.vstack(X), np.concatenate(Y), rcond=None)
        return b[len(names):]

    def predict(S, beta):
        m = S[:, cols] @ beta
        a = np.mean(S[mask, 0] - m[mask])  # the held-out experiment contributes only its intercept
        p = a + m
        h = S[:, 0]
        return {"unexpl_gap": (h[i8] - h[i6]) - (p[i8] - p[i6]),
                "unexpl_dip": ((h[i5] + h[i7]) / 2 - h[i6]) - ((p[i5] + p[i7]) / 2 - p[i6]),
                "pred_gap": p[i8] - p[i6], "pred_dip": (p[i5] + p[i7]) / 2 - p[i6]}

    exps = list(stack)
    nb = min(len(v[1]) for v in stack.values())
    out = {"pooled_beta": fit({e: stack[e][0] for e in exps}).tolist(), "keys": list(keys), "held_out": {}}
    for e in exps:
        others = [o for o in exps if o != e]
        beta = fit({o: stack[o][0] for o in others})
        est = predict(stack[e][0], beta)
        bs = [predict(stack[e][1][r], fit({o: stack[o][1][r] for o in others})) for r in range(nb)]
        out["held_out"][e] = {"beta": beta.tolist(),
                              **{k: {"est": float(v), **ci([b[k] for b in bs])} for k, v in est.items()}}
    return out


def pooled(results, exps, key):
    """Inverse-variance weighted mean of an estimate across independent experiments."""
    est = np.array([results[e][key]["est"] for e in exps])
    se = np.array([results[e][key]["se"] for e in exps])
    w = 1 / se ** 2
    m = float(np.sum(w * est) / np.sum(w))
    q = float(np.sum(w * (est - m) ** 2))
    return {"est": m, "se": float(np.sqrt(1 / np.sum(w))), "Q": q, "df": len(exps) - 1,
            "I2": max(0.0, (q - (len(exps) - 1)) / q) if q > 0 else 0.0, "experiments": list(exps)}


def diff_boot(a, b, key):
    """Difference between two independent samples' estimates (a - b)."""
    bs = np.array(a["_boot"][key]) - np.array(b["_boot"][key])
    return {"est": a[key]["est"] - b[key]["est"], **ci(bs)}


# ---------------------------------------------------------------- validation
def validate(curves_by_exp):
    """Our model curves, smoothed on a uniform design, vs Marjieh et al.'s
    published profiles."""
    out = {}
    names = {"HK": "Hutchinson & Knopoff (1978)", "H": "Harrison & Pearce (2018)",
             "revHK": "Hutchinson & Knopoff (1978) (revised)", "Seth": "Sethares (1993)"}
    for key, (_, _, _, _, folder, _) in EXPS.items():
        curves = curves_by_exp[key]
        out[key] = {}
        for k, fname in names.items():
            rows = list(csv.DictReader(open(BATCH / folder / "models" / f"{fname}.csv")))
            gx = np.array([float(r["interval"]) for r in rows])
            gy = np.array([float(r["output"]) for r in rows])
            K = kernel(gx, FINE)
            ours = (K @ curves[k]) / K.sum(1)
            if k in ("HK", "revHK", "Seth"):
                ours = -ours  # theirs is consonance-oriented
            out[key][k] = {"r": float(np.corrcoef(ours, gy)[0, 1]), "max_abs_diff": float(np.abs(ours - gy).max()),
                           "range": float(gy.max() - gy.min())}
    return out


def load_curves():
    cache = RES / "marjieh_model_curves.npz"
    if cache.exists():
        z = np.load(cache)
        curves_by_exp = {k: {m: z[f"{k}__{m}"] for m in MODEL_KEYS} for k in EXPS}
        return curves_by_exp, {m: z[f"roll__{m}"] for m in MODEL_KEYS}, z["roll_levels"]
    curves_by_exp = {}
    for key, (label, _, _, _, folder, bass) in EXPS.items():
        print("model curves:", label, flush=True)
        curves_by_exp[key] = model_curves(timbre(folder), bass)
    roll_levels = np.arange(0, 15.01, 0.5)
    per = [model_curves(harmonic_rolloff(db)) for db in roll_levels]
    roll_curves = {m: np.array([p[m] for p in per]) for m in MODEL_KEYS}
    np.savez_compressed(cache, roll_levels=roll_levels,
                        **{f"{k}__{m}": v for k, c in curves_by_exp.items() for m, v in c.items()},
                        **{f"roll__{m}": v for m, v in roll_curves.items()})
    return curves_by_exp, roll_curves, roll_levels


def rolloff_values(d, roll_curves, roll_levels):
    """Each trial's model value at its own interval and roll-off (bilinear)."""
    cols = []
    ri = np.interp(d["rolloff"], roll_levels, np.arange(len(roll_levels)))
    lo = np.floor(ri).astype(int)
    hi = np.minimum(lo + 1, len(roll_levels) - 1)
    t = ri - lo
    xi = np.interp(d["x"], FINE, np.arange(len(FINE)))
    xl = np.floor(xi).astype(int)
    xh = np.minimum(xl + 1, len(FINE) - 1)
    u = xi - xl
    for m in MODEL_KEYS:
        C = roll_curves[m]
        cols.append((1 - t) * ((1 - u) * C[lo, xl] + u * C[lo, xh]) + t * ((1 - u) * C[hi, xl] + u * C[hi, xh]))
    return np.column_stack([d["z"]] + cols)


def line(label, r):
    g, dp = r["obs_gap"], r["obs_dip"]
    return (f"{label:34s} gap={g['est']:+.3f} [{g['ci'][0]:+.3f},{g['ci'][1]:+.3f}] dip={dp['est']:+.3f} "
            f"[{dp['ci'][0]:+.3f},{dp['ci'][1]:+.3f}] | unexpl gap HK={r['HK_unexpl_gap']['est']:+.3f} "
            f"HK+H={r['HK+H_unexpl_gap']['est']:+.3f} C={r['Composite_unexpl_gap']['est']:+.3f} | unexpl dip "
            f"HK={r['HK_unexpl_dip']['est']:+.3f} HK+H={r['HK+H_unexpl_dip']['est']:+.3f} "
            f"C={r['Composite_unexpl_dip']['est']:+.3f} | dip rank obs={r['obs_dip_rank']} "
            f"HK+H={r['HK+H_udip_rank']}")


def main():
    rng = np.random.default_rng(SEED)
    curves_by_exp, roll_curves, roll_levels = load_curves()
    results = {"settings": {"sd": SD, "window": WIN, "fit_from": LO, "nboot": NBOOT, "seed": SEED,
                            "composite": {"p": P_REV, "q": Q_REV, "r": R_REV, "w_harm": W_HARM}},
               "validation": validate(curves_by_exp), "experiments": {}}
    stack = {}
    for key, (label, fname, synth, cohort, folder, bass) in EXPS.items():
        d = read_trials(fname, synth)
        r, S0, SB = analyse(d, curves_by_exp[key], rng, keep_profiles=True)
        stack[key] = (S0, SB)
        r.update({"label": label, "cohort": cohort, "n_trials": int(len(d["x"])),
                  "n_participants": int(len(np.unique(d["pid"]))), "raw_sd_mean": d["raw_sd_mean"],
                  "rating_range": d["rating_range"]})
        if np.isfinite(d["mexp"]).any():  # Marjieh et al.'s median split at 2 years
            r["musicianship"] = {}
            for grp, m in (("nonmus", d["mexp"] <= 2), ("mus", d["mexp"] > 2)):
                sub = {k: (v[m] if isinstance(v, np.ndarray) else v) for k, v in d.items()}
                rr = analyse(sub, curves_by_exp[key], rng)
                rr.pop("profile")
                rr["n_participants"] = int(len(np.unique(sub["pid"])))
                r["musicianship"][grp] = rr
            for k in ("obs_gap", "obs_dip", "HK+H_unexpl_gap", "HK+H_unexpl_dip", "Composite_unexpl_gap",
                      "Composite_unexpl_dip"):
                r["musicianship"][f"diff_{k}"] = diff_boot(r["musicianship"]["mus"], r["musicianship"]["nonmus"], k)
            for grp in ("mus", "nonmus"):
                r["musicianship"][grp].pop("_boot")
        results["experiments"][key] = r
        print(line(label, r), flush=True)
        json.dump(results, open(RES / "marjieh_dyads.json", "w"), indent=1)

    E = results["experiments"]
    # Korean vs US, same stimuli
    results["cohort_diff"] = {k: {m: diff_boot(E[f"{k}_kr"], E[k], m) for m in
                                  ("obs_gap", "obs_dip", "HK+H_unexpl_gap", "HK+H_unexpl_dip",
                                   "Composite_unexpl_gap", "Composite_unexpl_dip")}
                              for k in ("harm3", "str3", "comp3")}
    # removing the 3rd harmonic: observed vs predicted change
    results["no3_minus_eq5"] = {m: diff_boot(E["no3"], E["eq5"], m) for m in
                                ("obs_gap", "obs_dip", "HK_pred_gap", "HK_pred_dip", "HK+H_pred_gap",
                                 "HK+H_pred_dip", "Composite_pred_gap", "Composite_pred_dip")}
    # the 'constant': pooled unexplained gap/dip across independent US experiments
    us_harmonic = ["harm3", "eq5", "no3", "flute", "guitar", "piano"]
    us_all = [k for k, v in EXPS.items() if v[3] == "US"]
    results["pooled"] = {grp: {m: pooled(E, exps, m) for m in
                               ("obs_gap", "obs_dip", "HK_unexpl_gap", "HK_unexpl_dip", "Seth_unexpl_gap",
                                "Seth_unexpl_dip", "H_unexpl_gap", "H_unexpl_dip", "HK+H_unexpl_gap",
                                "HK+H_unexpl_dip", "revHK+H_unexpl_gap", "revHK+H_unexpl_dip",
                                "Composite_unexpl_gap", "Composite_unexpl_dip")}
                         for grp, exps in (("us_harmonic", us_harmonic), ("us_all", us_all))}
    # leave-one-experiment-out two-mechanism curve (US experiments)
    results["lodo"] = {"HK+H": lodo({e: stack[e] for e in us_all}, ("HK", "H")),
                       "HK": lodo({e: stack[e] for e in us_all}, ("HK",))}
    for e in us_all:
        v = results["lodo"]["HK+H"]["held_out"][e]
        print(f"LODO {e:8s} unexpl gap {v['unexpl_gap']['est']:+.3f} [{v['unexpl_gap']['ci'][0]:+.3f},"
              f"{v['unexpl_gap']['ci'][1]:+.3f}] dip {v['unexpl_dip']['est']:+.3f}", flush=True)
    for grp, p in results["pooled"].items():
        print(grp, {k: (round(v["est"], 3), round(v["se"], 3), round(v["I2"], 2)) for k, v in p.items()})
    json.dump(results, open(RES / "marjieh_dyads.json", "w"), indent=1)

    # roll-off slices (Study 3): 2D kernel, roll-off SD 1.5 dB/oct
    d = read_trials(ROLLOFF_FILE, xcol="intervals")
    V = rolloff_values(d, roll_curves, roll_levels)
    results["rolloff"] = {}
    for lev in ROLLOFF_SLICES:
        wr = np.exp(-0.5 * ((d["rolloff"] - lev) / ROLLOFF_SD) ** 2)
        rr = analyse(d, None, rng, extra_w=wr, V=V)
        rr.pop("_boot")
        results["rolloff"][f"{lev:g}"] = rr
        print(line(f"roll-off {lev:g} dB/oct", rr), flush=True)
    results["rolloff_meta"] = {"n_trials": int(len(d["x"])), "n_participants": int(len(np.unique(d["pid"]))),
                               "raw_sd_mean": d["raw_sd_mean"]}
    json.dump(results, open(RES / "marjieh_dyads.json", "w"), indent=1)


if __name__ == "__main__":
    main()
