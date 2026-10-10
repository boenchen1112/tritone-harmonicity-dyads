"""
Bowling et al. (2018) presented their chords in JUST INTONATION (inconData
`bowl18`, fractional `pi_chord`); every earlier analysis here, like incon's
published model outputs, scored the equal-tempered integer pitches. The
tritone is where this matters most: the presented tritone is 7/5 (582.5
cents) and the minor sixth 8/5 (813.7 cents).

This script rescores the 298 Bowling chords at their presented JI pitches
(interference: H&K on idealised 11-harmonic tones, amplitudes 1/i, as incon;
harmonicity: har_18 on the continuous pitch-class spectrum, cent resolution;
familiarity: har_19_corpus, which is defined on pitch-class categories and is
unchanged) and reruns tritone_gap.py's dyad and chord analyses on them.

Outputs results/bowling_ji.json.
"""
import csv
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import EXTERNAL
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tritone_gap as TG  # sets cwd to the project root and imports load_chords
import harmonicity as HM
from dissonance_model import hutch_knopoff_dissonance, pool_partials

JI_CSV = EXTERNAL / "bowl18_ji.csv"


def midi_hz(m):
    return 440.0 * 2 ** ((np.asarray(m, float) - 69) / 12)


def hk_chord(pitches, n=11):
    i = np.arange(1, n + 1)
    F, A = pool_partials([(midi_hz(p) * i, 1.0 / i) for p in pitches])
    return hutch_knopoff_dissonance(F, A)


def harm_chord(pitches, n=11, rho=0.75):
    i = np.arange(1, n + 1)
    pcs, ws, seen = [], [], set()
    for p in pitches:
        pc = round((p % 12) * 100) % 1200  # cent resolution; octave duplicates collapse as in incon
        if pc in seen:
            continue
        seen.add(pc)
        pcs.extend(pc + 1200 * np.log2(i))
        ws.extend(i ** -rho)
    spec = HM.pc_spectrum(pcs, ws, quantise=True)
    return HM.kl_from_uniform(HM.virtual_pitch_profile(spec, n, rho))


def build():
    rows = list(csv.DictReader(open(JI_CSV)))
    ji = [[float(v) for v in r["pi_chord"].split()] for r in rows]
    chords, y = TG.load_chords("bowling2018")
    et_rows = TG._read_csv("reference_data/incon_bowling2018.csv")
    # row alignment: JI chord rounded to ET must be the same pitch-class set as our chord, same size
    for c, j in zip(chords, ji):
        assert len(c) == len(j)
        js = sorted(j)
        assert [round(p - js[0]) for p in js] == [p - min(c) for p in sorted(c)], (c, j)
    # published rating (1-4, consonance-up) against our dissonance-up rating
    pub = np.array([float(r["rating"]) for r in rows])
    assert np.corrcoef(pub, -y)[0, 1] > 0.999, "rating order mismatch"
    hk_et = np.array([hk_chord(c) for c in chords])
    hk_ref = np.array([float(r["hutch_78_roughness"]) for r in et_rows])
    assert np.corrcoef(hk_et, hk_ref)[0, 1] > 0.99999, "H&K reimplementation disagrees with incon"
    h_et = np.array([harm_chord(c) for c in chords])
    h_ref = np.array([float(r["har_18_harmonicity"]) for r in et_rows])
    assert np.abs(h_et - h_ref).max() < 1e-9, "harmonicity reimplementation disagrees with incon"
    X_ji = {"I": np.array([hk_chord(j) for j in ji]), "H": np.array([harm_chord(j) for j in ji])}
    return chords, y, ji, X_ji, pub


def main():
    chords, y, ji, X_ji, pub = build()
    base_load = TG.load
    _, _, X_et = base_load("bowling2018")
    out = {"ET": {}, "JI": {}}

    def load_ji(ds):
        c, yy, X = base_load(ds)
        if ds == "bowling2018":
            X = dict(X)
            X["I"] = X_ji["I"] * TG.SIGNS["hutch_78_roughness"]
            X["H"] = X_ji["H"] * TG.SIGNS["har_18_harmonicity"]
        return c, yy, X

    for tag, loader in (("ET", base_load), ("JI", load_ji)):
        TG.load = loader
        TG.RNG = np.random.default_rng(20260926)
        out[tag]["dyads"] = TG.dyads()
        out[tag]["penalty"] = TG.penalty("bowling2018")
    TG.load = base_load
    # the two dyads in question, as presented
    idx = {round(ji[n][1] - ji[n][0], 3): n for n, c in enumerate(chords) if len(c) == 2}
    dy = {}
    for name, semis in (("tritone", 6), ("minor_sixth", 8)):
        n = [k for k, c in enumerate(chords) if len(c) == 2 and c[1] - c[0] == semis][0]
        dy[name] = {"ji_cents": round(100 * (ji[n][1] - ji[n][0]), 1), "rating": float(pub[n]),
                    "hk_et": float(X_et["I"][n] * TG.SIGNS["hutch_78_roughness"]), "hk_ji": float(X_ji["I"][n]),
                    "harm_et": float(X_et["H"][n] * TG.SIGNS["har_18_harmonicity"]), "harm_ji": float(X_ji["H"][n])}
    out["dyad_values"] = dy
    out["r_et_ji"] = {"I": float(np.corrcoef(X_et["I"], X_ji["I"] * TG.SIGNS["hutch_78_roughness"])[0, 1]),
                      "H": float(np.corrcoef(X_et["H"], X_ji["H"] * TG.SIGNS["har_18_harmonicity"])[0, 1])}
    json.dump(TG.rnd(out), open(TG.RES / "bowling_ji.json", "w"), indent=1)
    for tag in ("ET", "JI"):
        d = out[tag]["dyads"]
        print(tag, "obs TT-m6", round(d["observed_tt_minus_m6"], 3))
        for k, v in d["models"].items():
            print(f"   {k:34s} share={v['share_of_gap_reproduced']:+.2f} residTT={v['resid_tt']:+.3f} "
                  f"LOO={v['loo_resid_tt']:+.3f} rank={v['tt_resid_rank']} R2={v['r2']:.3f}")
        for k, v in out[tag]["penalty"]["models"].items():
            print(f"   penalty {k:34s} {v['tritone_penalty']:+.3f} [{v['tritone_ci'][0]:+.3f},{v['tritone_ci'][1]:+.3f}]")
    print(json.dumps(TG.rnd(dy), indent=1), out["r_et_ji"])


if __name__ == "__main__":
    main()
