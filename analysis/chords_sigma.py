"""
Independent test of the wider harmonicity tolerance chosen on the dyad data
(model_screen.py): does scoring chord harmonicity with sigma = 20 cents instead
of 6.83 change the residual tritone penalty in Bowling et al.'s chords (just
intonation, as presented) and in Johnson-Laird et al.'s chords (equal
temperament)? Same regression as tritone_gap.penalty (chord-size dummies, H&K,
harmonicity, tritone and semitone counts; bootstrap over chords).

Writes results/chords_sigma.json.
"""
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tritone_gap as TG
import bowling_ji as BJ
import harmonicity as HM

SIGMAS = (6.83, 20.0)


def harm_chord(pitches, sigma, n=11, rho=0.75):
    i = np.arange(1, n + 1)
    pcs, ws, seen = [], [], set()
    for p in pitches:
        pc = round((p % 12) * 100) % 1200
        if pc in seen:
            continue
        seen.add(pc)
        pcs.extend(pc + 1200 * np.log2(i))
        ws.extend(i ** -rho)
    spec = HM.pc_spectrum(pcs, ws, sigma=sigma, quantise=True)
    return HM.kl_from_uniform(HM.virtual_pitch_profile(spec, n, rho, sigma))


def penalty(chords, y, I, H, rng):
    size = np.array([len(c) for c in chords])
    cov = [(size == k).astype(float) for k in sorted(set(size))[1:]]
    ic6 = np.array([TG.ic_count(c, 6) for c in chords], float)
    ic1 = np.array([TG.ic_count(c, 1) for c in chords], float)
    b, A = TG.ols(y, cov + [I, -H, ic6, ic1])
    bt = []
    for _ in range(2000):
        s = rng.integers(0, len(y), len(y))
        bb, *_ = np.linalg.lstsq(A[s], y[s], rcond=None)
        bt.append(bb[-2])
    r2 = 1 - np.var(y - A @ b) / np.var(y)
    return {"tritone_penalty": float(b[-2]), "ci": np.percentile(bt, [2.5, 97.5]).tolist(), "r2": float(r2),
            "boot": bt}


def main():
    out = {}
    chords, y, ji, X_ji, _ = BJ.build()
    I_ji = X_ji["I"] * TG.SIGNS["hutch_78_roughness"]
    _, _, X_et = TG.load("bowling2018")
    cases = {"bowling_JI": (chords, y, I_ji, ji), "bowling_ET": (chords, y, X_et["I"], chords)}
    jc, jy, jX = TG.load("johnson-laird2012")
    cases["johnson-laird_ET"] = (jc, jy, jX["I"], jc)
    for name, (ch, yy, I, pitches) in cases.items():
        out[name] = {}
        for s in SIGMAS:
            H = np.array([harm_chord(p, s) for p in pitches])
            out[name][f"sigma{s:g}"] = penalty(ch, yy, I, H, np.random.default_rng(20260926))
            r = out[name][f"sigma{s:g}"]
            print(f"{name:18s} sigma {s:5.2f}: penalty {r['tritone_penalty']:+.3f} "
                  f"[{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}] R2 {r['r2']:.3f}", flush=True)
        # same seed and sample size, so the chord resamples are identical: a paired difference
        a, b = (np.array(out[name][f"sigma{s:g}"].pop("boot")) for s in (20.0, 6.83))
        d = out[name]["sigma20"]["tritone_penalty"] - out[name]["sigma6.83"]["tritone_penalty"]
        out[name]["diff20_minus_6.83"] = {"est": d, "ci": np.percentile(a - b, [2.5, 97.5]).tolist()}
        print(f"{name:18s} diff {d:+.3f} {out[name]['diff20_minus_6.83']['ci']}", flush=True)
    json.dump(out, open(TG.RES / "chords_sigma.json", "w"), indent=1)


if __name__ == "__main__":
    main()
