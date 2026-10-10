"""
The three-component decomposition of controlled_decomposition.py, redone for
Bowling et al. (2018) with interference and harmonicity scored at the
just-intonation pitches actually presented (bowling_ji.py). Familiarity
(har_19_corpus) is defined on pitch-class categories and is unchanged.
Johnson-Laird et al. (2012) used equal-tempered pitches (integer pi_chord in
inconData jl12a/jl12b), so its decomposition needs no rescoring.

Writes results/decomposition_ji.json.
"""
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bowling_ji as BJ
import controlled_decomposition as CDm

RES = HERE.parent / "results"


def main():
    chords, y_ours, ji, X_ji, pub = BJ.build()
    chords_cd, y = CDm.load_chords("bowling2018")
    assert [list(c) for c in chords_cd] == [list(c) for c in chords]
    raw = CDm.load_cache(CDm.ROOT / "reference_data/incon_bowling2018.csv", CDm.MODELS)
    fam = raw["har_19_corpus"]
    # JI chords transposed so the lowest note is MIDI 60 (interference only depends on register)
    I_tr = np.array([BJ.hk_chord(np.asarray(j) - min(j) + 60) for j in ji])
    signI, signH = CDm.SIGNS["hutch_78_roughness"], CDm.SIGNS["har_18_harmonicity"]
    CDm.RNG = np.random.default_rng(20260925)
    out = {"n": len(y), "r_et_ji": {}}
    for spec in ("raw", "size", "size+reg", "size+transp"):
        cov = CDm.covariates(chords, "size" if spec == "size+transp" else spec)
        I = (I_tr if spec == "size+transp" else X_ji["I"]) * signI
        P = {"interference": I, "harmonicity": X_ji["H"] * signH, "familiarity": fam}
        d = CDm.decompose(y, cov, P)
        d["boot_ci_unique"] = CDm.boot(y, cov, P)
        d["partial_r"] = {k: CDm.partial_r(y, v, cov) for k, v in P.items()}
        out[spec] = d
        u = d["unique"]
        print(f"JI {spec:12s} full={d['r2_full']:.3f} | " + "  ".join(
            f"{k[:4]} dR2={u[k]['delta_r2']:+.3f} p={u[k]['p']:.4f} "
            f"CI[{d['boot_ci_unique'][k][0]:+.3f},{d['boot_ci_unique'][k][1]:+.3f}]" for k in u))
    (RES / "decomposition_ji.json").write_text(json.dumps(CDm.rnd(out), indent=2))


if __name__ == "__main__":
    main()
