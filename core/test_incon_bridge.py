"""
Cross-validate this project's own Python model implementations against
pmcharrison/incon (Harrison & Pearce 2020), the field-standard R package, on
every chord in both rating datasets -- not on unit-test fixtures.

Two things are being checked, and only the second is a real test of my code:

  1. Orientation. Each of incon's 17 models is run on reference chords whose
     direction nobody disputes, and the measured sign is compared against what
     I would have written down by hand. (I got two of them wrong by hand.)

  2. Agreement. My Stolzenburg, Sethares, Sethares-min, Vassilakis and
     Hutchinson & Knopoff are run on byte-identical input to incon's -- MIDI
     pitches for Stolzenburg, and hrep's own 11-harmonic 1/n spectrum for the
     roughness models -- and correlated against incon's output over all 401
     chords. Anything short of r = 1.0000 is a discrepancy to explain, not to
     average away.

The interesting expected discrepancy: this project normalizes its roughness by
sqrt(sum(amps)) and incon/dycon does not. On a fixed-size interval sweep that is
a per-curve constant and harmless, which is what FINDINGS.md claims. Across
chords of DIFFERENT sizes it is not constant, so the two should disagree exactly
where Result 5 pools 2-, 3- and 4-note chords. This test measures how much.
"""
import numpy as np
from scipy.stats import pearsonr, spearmanr

from chord_validation import DATASETS, load_chords
from dissonance_model import (
    sethares_dissonance, sethares_min_dissonance, vassilakis_dissonance,
    hutch_knopoff_dissonance,
)
from harmonicity_model import smooth_log_periodicity
from incon_bridge import ALL_MODELS, hrep_partials, measure_orientations, run_incon

# What I would have written by hand before measuring. Kept only so the test can
# report where hand-reasoning fails; the measured values are what get used.
HAND_GUESS_HIGHER_IS_CONSONANT = {
    "gill_09_harmonicity", "har_18_harmonicity", "milne_13_harmonicity",
    "parn_94_complex", "bowl_18_min_freq_dist", "huron_94_dyadic",
    "jl_12_tonal", "har_19_corpus", "har_19_composite",
}

MY_ROUGHNESS = {
    "seth_93_roughness": sethares_dissonance,
    "vass_01_roughness": vassilakis_dissonance,
    "hutch_78_roughness": hutch_knopoff_dissonance,
}


def midi_to_hz(m):
    return 440.0 * 2 ** ((m - 69) / 12.0)


def my_roughness(chord, model_fn):
    """Pool hrep's default spectrum (11 harmonics, amplitude 1/n) for every note
    -- byte-identical input to what incon builds internally."""
    fs, amps = [], []
    for m in chord:
        f, a = hrep_partials(midi_to_hz(m))
        fs.append(f)
        amps.append(a)
    return model_fn(np.concatenate(fs), np.concatenate(amps))


def main():
    print("=" * 88)
    print("1. Orientation of all 17 incon models, measured rather than assumed")
    print("=" * 88)
    signs = measure_orientations()
    wrong = []
    for m in ALL_MODELS:
        s = signs[m]
        measured = "higher = MORE dissonant" if s == 1 else (
            "higher = MORE consonant" if s == -1 else "DID NOT SEPARATE")
        guessed_consonant = m in HAND_GUESS_HIGHER_IS_CONSONANT
        agree = (s == -1) == guessed_consonant
        if not agree:
            wrong.append(m)
        print(f"  {m:24s} {measured:24s} {'' if agree else '<-- my hand guess was WRONG'}")
    print(f"\n  Hand-written orientations wrong for {len(wrong)}/{len(ALL_MODELS)}: {wrong}")
    print("  This is why orientation is measured at runtime. A 17-model table typed from")
    print("  intuition would have inverted these two everywhere they were used.")

    print("\n" + "=" * 88)
    print("2. My Python implementations vs incon's R, on identical input, all 401 chords")
    print("=" * 88)
    all_chords, labels = [], []
    for ds in DATASETS:
        ch, _ = load_chords(ds)
        all_chords += ch
        labels += [ds] * len(ch)
    sizes = np.array([len(c) for c in all_chords])
    print(f"  {len(all_chords)} chords "
          f"({(sizes == 2).sum()} dyads, {(sizes == 3).sum()} triads, {(sizes == 4).sum()} tetrachords)")

    ref = run_incon(all_chords, ["stolz_15_periodicity"] + list(MY_ROUGHNESS),
                    cache_path="reference_data/incon_all_chords.csv")

    print(f"\n  {'model':22s} {'pearson r':>10s} {'spearman':>9s} {'max |rel err|':>14s}  verdict")

    mine = np.array([smooth_log_periodicity(c) for c in all_chords])
    theirs = ref["stolz_15_periodicity"]
    rel = np.max(np.abs(mine - theirs) / np.maximum(np.abs(theirs), 1e-12))
    r, _ = pearsonr(mine, theirs)
    print(f"  {'stolz_15_periodicity':22s} {r:10.6f} {spearmanr(mine, theirs)[0]:9.6f} "
          f"{rel:14.2e}  {'EXACT MATCH' if rel < 1e-9 else 'DISCREPANCY'}")
    stolz_ok = rel < 1e-9

    rough_report = {}
    for name, fn in MY_ROUGHNESS.items():
        mine = np.array([my_roughness(c, fn) for c in all_chords])
        theirs = ref[name]
        r, _ = pearsonr(mine, theirs)
        rho, _ = spearmanr(mine, theirs)
        ratio = mine / theirs
        rel = float(np.max(np.abs(ratio / np.median(ratio) - 1)))
        verdict = ("EXACT up to a global constant" if rel < 1e-9 else
                   "proportional WITHIN each chord size" if r > 0.999 else
                   "DIFFERS")
        print(f"  {name:22s} {r:10.6f} {rho:9.6f} {rel:14.2e}  {verdict}")
        rough_report[name] = (mine, theirs, r)

    print("\n  Per-chord-size agreement (isolating the sqrt(sum(amps)) normalization):")
    print(f"  {'model':22s} " + " ".join(f"{f'{k}-note r':>12s}" for k in (2, 3, 4)))
    for name, (mine, theirs, _) in rough_report.items():
        cells = []
        for k in (2, 3, 4):
            idx = sizes == k
            cells.append(f"{pearsonr(mine[idx], theirs[idx])[0]:12.6f}")
        print(f"  {name:22s} " + " ".join(cells))

    print("\n  Sethares-min (my sethares_min_dissonance) vs incon's seth_93_roughness --")
    print("  dycon defaults to min_amplitude=TRUE, so this may be the closer match:")
    mine_min = np.array([my_roughness(c, sethares_min_dissonance) for c in all_chords])
    for k in (2, 3, 4):
        idx = sizes == k
        r_prod = pearsonr(np.array([my_roughness(c, sethares_dissonance)
                                    for c in np.array(all_chords, dtype=object)[idx]]),
                          ref["seth_93_roughness"][idx])[0]
        r_min = pearsonr(mine_min[idx], ref["seth_93_roughness"][idx])[0]
        better = "min" if r_min > r_prod else "product"
        print(f"    {k}-note: product r={r_prod:.6f}   min r={r_min:.6f}   -> incon matches {better}")

    print("\n" + "=" * 88)
    if stolz_ok:
        print("Stolzenburg: EXACT agreement with the reference implementation on all 401")
        print("chords, independent of the 2048-chord table check in test_harmonicity.py.")
    else:
        print("Stolzenburg: DISCREPANCY -- investigate before trusting Result 7.")
    print("=" * 88)
    if not stolz_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
