"""
Shared engine for the gap decomposition (A1, A2). Exploratory: the decomposition
was designed after seeing the dense-dyad residuals.

For each US experiment the listener profile and the model features are
smoothed on MD.GRID; each experiment is predicted from weights fitted to the
other nine (leave one experiment out, as model_screen.py). From the held-out
residual profile r:

    b(c) = r(c) - [r(c-0.5) + r(c+0.5)] / 2      in-tune ("grid") bonus
    q(c) = [r(c-0.5) + r(c+0.5)] / 2             local baseline, r = b + q

so the residual tritone--minor-sixth gap r(8) - r(6) splits into a bonus part
b(8) - b(6) and a baseline part q(8) - q(6). Pooled values are means over the
six harmonic US experiments (as grid_bonus.py).

Step statistics. q_step (mean q over 8..11 minus mean over 1..5, 7) cannot tell
a step from a leftover monotone trend, so two step-specific statistics are
added: the coefficient of an indicator 1[c >= 8] in an OLS of q(c) on (1, c)
over c = 1..11 and 1..14 ("step_ind11", "step_ind14"), and the increment
q(8) - q(7) minus the mean of the other adjacent increments over 1..11
("step_inc").

Participant bootstrap: weights are drawn for all 13 experiments in MD.EXPS
order with one generator, exactly as model_boot.profiles does, so at default
settings the draws (and results) match grid_bonus.py; every model and every
analysis setting is evaluated on the same draws, so differences are paired.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import marjieh_dyads as MD
import model_screen as S
import sharpness as SH

G = MD.GRID
EVAL = (G >= MD.LO) & (G <= 14.75)
HARM6 = S.HARM6
US = S.US
OTHER = [c for c in range(1, 12) if c != 6]
COLS = ["HK", "H", "C", "H20", "X1", "SHARP"]
MODELS = {"M1": ["HK", "H20"], "M2": ["HK", "H20", "X1"], "M3": ["HK", "H20", "SHARP"]}
DEFAULT = {"kernel": ("gauss", MD.SD), "win": MD.WIN}


def train_mask(win):
    return EVAL & (np.abs(G - 6) >= win) & (np.abs(G - 8) >= win)


def load(exps=US):
    """Trial-level design per experiment: x, pid, V = [z, features...]."""
    base, _, _ = MD.load_curves()
    f1, f3 = S.all_curves(), SH.all_curves()
    out = {}
    for key, (_, fname, synth, _, _, _) in MD.EXPS.items():
        d = MD.read_trials(fname, synth)
        if key in exps:
            cols = {**base[key], **f1[key], **f3[key]}
            V = np.column_stack([d["z"]] + [np.interp(d["x"], MD.FINE, cols[k]) for k in COLS])
        else:
            V = None
        out[key] = {"x": d["x"], "pid": d["pid"], "V": V}
    return out


def kmat(x, kernel):
    kind, h = kernel
    if kind == "gauss":
        return MD.kernel(G, x, sd=h)
    if kind == "box":  # running mean over +-h (0.25-semitone bins centred on the grid point when h = 0.125)
        return (np.abs(G[:, None] - x[None, :]) <= h + 1e-9).astype(float)
    raise ValueError(kind)


def draw_weights(D, rng):
    """One participant-bootstrap draw for every experiment, in MD.EXPS order."""
    return {k: (np.ones(len(v["pid"])) if rng is None else MD.boot_weights(v["pid"], rng)) for k, v in D.items()}


def participant_sums(v, kernel):
    """Per-participant kernel sums, so a bootstrap draw (constant weight within a
    participant) smooths by one weighted sum: num = sum_p w_p A_p, den = sum_p w_p B_p.
    Same result as MD.smooth up to floating-point summation order."""
    cache = v.setdefault("_sums", {})
    if kernel not in cache:
        ups, inv = np.unique(v["pid"], return_inverse=True)
        K = kmat(v["x"], kernel)
        A = np.empty((len(ups), len(G), v["V"].shape[1]))
        B = np.empty((len(ups), len(G)))
        for p in range(len(ups)):
            m = inv == p
            A[p] = K[:, m] @ v["V"][m]
            B[p] = K[:, m].sum(1)
        first = np.array([np.flatnonzero(inv == p)[0] for p in range(len(ups))])
        cache[kernel] = (A, B, first)
    return cache[kernel]


def smooth_all(D, W, kernel):
    P = {}
    for k, v in D.items():
        if v["V"] is None:
            continue
        A, B, first = participant_sums(v, kernel)
        wp = W[k][first]
        Sm = np.tensordot(wp, A, axes=1) / (wp @ B)[:, None]
        P[k] = {"z": Sm[:, 0], **{c: Sm[:, 1 + i] for i, c in enumerate(COLS)}}
    return P


def fit(P, exps, keys, train):
    Y, X = [], []
    for n, e in enumerate(exps):
        Dm = np.zeros((train.sum(), len(exps)))
        Dm[:, n] = 1
        Y.append(P[e]["z"][train])
        X.append(np.column_stack([Dm] + [P[e][k][train] for k in keys]))
    b, *_ = np.linalg.lstsq(np.vstack(X), np.concatenate(Y), rcond=None)
    return b[len(exps):]


def predict(Pe, keys, beta, train):
    m = np.column_stack([Pe[k] for k in keys]) @ beta
    return m + np.mean(Pe["z"][train] - m[train])


def heldout_residuals(P, keys, train, exps=US):
    R, Pred = {}, {}
    for e in exps:
        beta = fit(P, [o for o in exps if o != e], keys, train)
        Pred[e] = predict(P[e], keys, beta, train)
        R[e] = P[e]["z"] - Pred[e]
    return R, Pred


def at(v, x):
    return float(np.interp(x, G, v))


def bq(r):
    q = np.array([(at(r, c - 0.5) + at(r, c + 0.5)) / 2 for c in range(1, 15)])
    b = np.array([at(r, c) for c in range(1, 15)]) - q
    return b, q


def step_ind(q, cmax):
    c = np.arange(1, cmax + 1)
    X = np.column_stack([np.ones(cmax), c, (c >= 8).astype(float)])
    coef, *_ = np.linalg.lstsq(X, q[:cmax], rcond=None)
    return float(coef[2])


def stats(b, q):
    o = {}
    o["bonus_other"] = float(np.mean([b[c - 1] for c in OTHER]))
    o["bonus_tt"] = float(b[5])
    o["other_minus_tt"] = o["bonus_other"] - o["bonus_tt"]
    o["gap_resid"] = float((b[7] + q[7]) - (b[5] + q[5]))
    o["gap_bonus_part"] = float(b[7] - b[5])
    o["gap_base_part"] = float(q[7] - q[5])
    o["q_low_mean"] = float(np.mean([q[c - 1] for c in (1, 2, 3, 4, 5, 7)]))
    o["q_high_mean"] = float(np.mean([q[c - 1] for c in (8, 9, 10, 11)]))
    o["q_step"] = o["q_high_mean"] - o["q_low_mean"]
    o["q6_minus_low"] = float(q[5]) - o["q_low_mean"]
    o["step_ind11"] = step_ind(q, 11)
    o["step_ind14"] = step_ind(q, 14)
    inc = np.diff(q[:11])                       # q(c+1) - q(c), c = 1..10
    o["step_inc"] = float(inc[6] - np.mean(np.delete(inc, 6)))
    o["tt_minus_m2m7"] = float(b[5] - (b[1] + b[9]) / 2)
    o["m2m7_minus_rest"] = float((b[1] + b[9]) / 2 - np.mean([b[c - 1] for c in (1, 3, 4, 5, 7, 8, 9, 11)]))
    return o


def decompose(P, models, train, per_exp=True):
    out = {}
    for m, keys in models.items():
        R, _ = heldout_residuals(P, keys, train)
        BQ = {e: bq(R[e]) for e in HARM6}
        b = np.mean([BQ[e][0] for e in HARM6], axis=0)
        q = np.mean([BQ[e][1] for e in HARM6], axis=0)
        o = {"b": b.tolist(), "q": q.tolist(), **stats(b, q)}
        if per_exp:
            o["exp"] = {e: stats(*BQ[e]) for e in HARM6}
        out[m] = o
    return out


def flat(d, prefix=""):
    """Scalars only, keys 'setting|model|stat' or 'setting|model|exp:stat'."""
    f = {}
    for k, v in d.items():
        if isinstance(v, dict):
            f.update(flat(v, f"{prefix}{k}|"))
        elif isinstance(v, (int, float)):
            f[f"{prefix}{k}"] = float(v)
        elif isinstance(v, list):
            for i, x in enumerate(v):
                f[f"{prefix}{k}{i + 1}"] = float(x)
    return f


def run_settings(D, W, settings, models):
    """settings: {name: {"kernel": (kind, h), "win": w}}; kernels shared across windows."""
    cache, out = {}, {}
    for name, s in settings.items():
        if s["kernel"] not in cache:
            cache[s["kernel"]] = smooth_all(D, W, s["kernel"])
        out[name] = decompose(cache[s["kernel"]], models, train_mask(s["win"]))
    return out


def summarise(est_flat, boot_flats, pairs=()):
    """est/CI for every key; paired differences for (a, b) key-prefix pairs."""
    res = {}
    for k, v in est_flat.items():
        bs = np.array([bf[k] for bf in boot_flats])
        res[k] = {"est": v, **MD.ci(bs)}
    for a, bpre in pairs:
        for k in est_flat:
            if k.startswith(a + "|"):
                kb = bpre + k[len(a):]
                if kb in est_flat:
                    bs = np.array([bf[k] - bf[kb] for bf in boot_flats])
                    res[f"{a}-{bpre}{k[len(a):]}"] = {"est": est_flat[k] - est_flat[kb], **MD.ci(bs)}
    return res
