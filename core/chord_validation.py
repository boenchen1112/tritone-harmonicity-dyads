"""
Chord-level validation of the roughness models against human ratings.

Two datasets, both via github.com/DCMLab/consonance:
  bowling2018       298 chords -- 12 dyads, 66 triads, 220 tetrachords. Piano
                    stimuli, roots MIDI 51-59, nothing above MIDI 67.
  johnson-laird2012 103 chords -- 55 triads, 48 tetrachords. Roots MIDI 43-64,
                    notes to 79.

dissonance_model.py's model functions already operate on a pooled partial set
with no assumption of exactly two tones, so extending to N-tone chords needs no
change to the models -- only a chord-level wrapper that expands each note into
partials and pools them all.

POLARITY. The two datasets' rating scales run in OPPOSITE directions, and an
earlier version of this script applied Bowling's polarity to Johnson-Laird
unchecked, silently inverting every correlation it reported. The fix is not to
sprinkle negations at the call sites -- that is how the bug happened -- but to
declare each dataset's polarity ONCE, in DATASETS below, and convert at load
time into a single internal convention:

    everything downstream of load_chords() is DISSONANCE: higher = less pleasant.

That is the same direction as model output, so every correlation reported here
should be POSITIVE if the model is working. There are no other sign flips in
this file, deliberately.
"""
import csv
import json
import numpy as np
from scipy.stats import pearsonr, spearmanr

from dissonance_model import (
    harmonic_partials, pool_partials, sethares_dissonance, sethares_min_dissonance,
    vassilakis_dissonance, hutch_knopoff_dissonance,
)
from behavioral_validation import fisher_ci

MODELS = [
    ("Sethares", sethares_dissonance),
    ("Sethares-min", sethares_min_dissonance),
    ("Vassilakis", vassilakis_dissonance),
    ("Hutch-Knopoff", hutch_knopoff_dissonance),
]

# "higher_means" records what a HIGHER published rating value means, and is the
# only place in this file where either dataset's polarity is asserted. Both
# claims are checkable: Bowling's is verified against the DCMLab/consonance
# README's own rescaling call for that dataset (scale_ratings(r_con=4, r_dis=1),
# i.e. consonant maps to the high end) and against the local file, where the
# octave scores highest (3.893) and the minor 2nd lowest (1.317). Johnson-Laird's
# is verified against the paper's Methods: "a scale from '1' = 'highly pleasant'
# to '7' = 'highly unpleasant'" (Johnson-Laird, Kang & Leong 2012, Music
# Perception 30(1):19-35).
DATASETS = {
    "bowling2018": {
        "path": "behavioral_data/bowling2018.tsv",
        "higher_means": "consonant",
        "cite": "Bowling, Purves & Gill (2018), PNAS 115(1):216-221",
    },
    "johnson-laird2012": {
        "path": "behavioral_data/johnson-laird2012.tsv",
        "higher_means": "dissonant",
        "cite": "Johnson-Laird, Kang & Leong (2012), Music Perception 30(1):19-35",
    },
}


def midi_to_hz(m, a4=440.0):
    return a4 * 2 ** ((m - 69) / 12.0)


def make_real_partials_fn(ratios, rel_amps):
    ratios = np.asarray(ratios, dtype=float)
    rel_amps = np.asarray(rel_amps, dtype=float)

    def fn(f0, **kwargs):
        return f0 * ratios, rel_amps.copy()

    return fn


def chord_dissonance(midi_notes, partials_fn, model_fn, merge=True, **kwargs):
    """Pool every note's partials into one spectrum and score it. Coincident
    partials are merged in quadrature by default, matching hrep/incon -- see
    dissonance_model.pool_partials for why that is not just concatenation."""
    per_tone = [partials_fn(midi_to_hz(m), **kwargs) for m in midi_notes]
    freqs, amps = pool_partials(per_tone, merge=merge)
    return model_fn(freqs, amps)


def load_chords(name):
    """Returns (chords, human_dissonance). human_dissonance is ALWAYS oriented
    higher = less pleasant, whatever the published scale did -- see the module
    docstring. The sanity check below is not decoration: it is what would have
    caught the original Result 4 bug on the first run."""
    spec = DATASETS[name]
    rows = list(csv.DictReader(open(spec["path"]), delimiter="\t"))
    chords = [[int(x) for x in r["pitches"].split(",")] for r in rows]
    published = np.array([float(r["rating"]) for r in rows])
    human_diss = published if spec["higher_means"] == "dissonant" else -published

    _assert_polarity(name, chords, human_diss)
    return chords, human_diss


_MAJOR_MINOR_PCS = [frozenset((r + i) % 12 for i in t)
                    for t in ((0, 4, 7), (0, 3, 7)) for r in range(12)]


def _reference_class(chord):
    """Label a chord as an uncontroversially consonant or dissonant reference,
    or neither. Works on PITCH CLASSES, not absolute intervals, because
    Johnson-Laird's chords are widely spaced open voicings -- its most pleasant
    triad is [48,64,67], a C major spread over 19 semitones, which no
    adjacent-interval test recognises as major."""
    pcs = frozenset(n % 12 for n in chord)
    if len(chord) == 2:
        ic = min((chord[1] - chord[0]) % 12, (chord[0] - chord[1]) % 12)
        if ic in (0, 5):        # octave/unison, P4/P5
            return "consonant"
        if ic in (1, 2):        # m2/M7, M2/m7
            return "dissonant"
        return None
    if pcs in _MAJOR_MINOR_PCS:
        return "consonant"
    # any two pitch classes a semitone apart (mod 12) is a clash nobody defends
    if any(((a - b) % 12) in (1, 11) for a in pcs for b in pcs if a != b):
        return "dissonant"
    return None


def _assert_polarity(name, chords, human_diss):
    """Independent check that human_diss really is oriented higher = less
    pleasant, using chords whose direction nobody disputes. If the sign is
    wrong the whole dataset is inverted and every downstream correlation would
    be too -- which is exactly what happened the first time Result 4 was run."""
    groups = {"consonant": [], "dissonant": []}
    for c, d in zip(chords, human_diss):
        cls = _reference_class(c)
        if cls:
            groups[cls].append(d)
    if not groups["consonant"] or not groups["dissonant"]:
        raise SystemExit(
            f"POLARITY CHECK could not run for {name}: no reference chords matched. "
            f"A skipped check is not a passed check -- extend _reference_class() before "
            f"trusting any correlation from this dataset."
        )
    rough, smooth = np.mean(groups["dissonant"]), np.mean(groups["consonant"])
    ok = rough > smooth
    print(f"  [polarity] {name}: {len(groups['dissonant'])} reference-dissonant chords "
          f"mean {rough:+.3f} vs {len(groups['consonant'])} reference-consonant chords "
          f"mean {smooth:+.3f} -> {'OK' if ok else 'INVERTED'}")
    if not ok:
        raise SystemExit(
            f"POLARITY CHECK FAILED for {name}: after conversion, chords that are "
            f"uncontroversially rough score LOWER than chords that are uncontroversially "
            f"smooth. DATASETS['{name}']['higher_means'] is wrong. Fix it there -- do not "
            f"add a negation at a call site."
        )


def transpose_to_common_root(chord, target_root=60):
    """Shift every note by the same amount so the chord's lowest note lands
    on target_root, preserving interval structure. Isolates chord quality
    from register -- Sethares/Vassilakis/H&K are NOT scale-invariant (their
    bandwidth terms depend on absolute frequency), so uncontrolled register is
    a real confound here."""
    shift = target_root - min(chord)
    return [n + shift for n in chord]


def partial_r(r_xy, r_xz, r_yz):
    """Partial correlation of x,y controlling for z, from pairwise Pearson r's.
    All three inputs must be computed on the SAME sign convention for y; since
    load_chords() returns a single canonical orientation, they are."""
    denom = np.sqrt((1 - r_xz ** 2) * (1 - r_yz ** 2))
    return (r_xy - r_xz * r_yz) / denom if denom > 0 else float("nan")


def analyse(dataset_name, extracted, timbre_labels):
    spec = DATASETS[dataset_name]
    print(f"\n{'=' * 100}")
    print(f"{dataset_name} -- {spec['cite']}")
    chords, human_diss = load_chords(dataset_name)

    sizes = sorted({len(c) for c in chords})
    counts = {k: sum(1 for c in chords if len(c) == k) for k in sizes}
    root_midi_all = np.array([min(c) for c in chords])
    print(f"  {len(chords)} chords by size: {counts}; "
          f"roots MIDI {root_midi_all.min()}-{root_midi_all.max()}, "
          f"highest note {max(max(c) for c in chords)}")

    subsets = {"all": list(range(len(chords)))}
    for k in sizes:
        if counts[k] >= 12:
            subsets[f"{k}-note"] = [i for i, c in enumerate(chords) if len(c) == k]

    results = []
    for subset_name, idx in subsets.items():
        sub_chords = [chords[i] for i in idx]
        y = human_diss[idx]
        root_midi = np.array([min(c) for c in sub_chords])
        n = len(idx)

        r_root, p_root = pearsonr(root_midi, y)
        print(f"\n  --- subset '{subset_name}' (n={n}) --- "
              f"register confound: root MIDI vs human dissonance r={r_root:+.3f} p={p_root:.4f}"
              f"{'  [SIGNIFICANT]' if p_root < 0.05 else ''}")
        print(f"  {'variant':11s} {'timbre':16s} {'model':14s} {'r':>7s} {'95% CI':>15s} "
              f"{'p':>9s} {'partial_r':>10s}")

        variants = {"raw": sub_chords,
                    "transposed": [transpose_to_common_root(c) for c in sub_chords]}

        for tag, chord_set in variants.items():
            for label, pfn, kwargs in timbre_labels:
                for model_name, model_fn in MODELS:
                    x = np.array([chord_dissonance(c, pfn, model_fn, **kwargs) for c in chord_set])
                    if np.allclose(x, x[0]):
                        continue
                    r, p = pearsonr(x, y)
                    rho, ps = spearmanr(x, y)
                    lo, hi = fisher_ci(r, n)
                    r_xz, _ = pearsonr(x, root_midi)
                    rp = partial_r(r, r_xz, r_root) if tag == "raw" else float("nan")
                    print(f"  {tag:11s} {label:16s} {model_name:14s} {r:+7.3f} "
                          f"[{lo:+.2f},{hi:+.2f}] {p:9.5f} {rp:+10.3f}")
                    results.append({
                        "dataset": dataset_name, "subset": subset_name, "n": n,
                        "variant": tag, "timbre": label, "model": model_name,
                        "pearson_r": round(r, 3), "pearson_ci95": [round(lo, 3), round(hi, 3)],
                        "pearson_p": round(p, 5), "spearman_rho": round(rho, 3),
                        "spearman_p": round(ps, 5),
                        "partial_r_controlling_root": (None if np.isnan(rp) else round(float(rp), 3)),
                    })

    return {"n_chords": len(chords), "sizes": counts,
            "root_midi_range": [int(root_midi_all.min()), int(root_midi_all.max())],
            "results": results}


def main():
    with open("extracted_partials.json") as f:
        extracted = json.load(f)
    c4_labels = ["cello_C4", "clarinet_C4", "oboe_C4", "flute_C4", "bassoon_C4",
                 "saxophone_C4", "french-horn_C4", "piano_C4"]

    timbres = [("idealized", harmonic_partials, {"n_partials": 7, "rolloff": 0.9})]
    for label in c4_labels:
        data = extracted[label]
        timbres.append((data["instrument"],
                        make_real_partials_fn(data["ratios"], data["rel_amps"]), {}))

    out = {}
    for name in DATASETS:
        out[name] = analyse(name, extracted, timbres)

    with open("chord_validation_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved chord_validation_results.json")
    print("\nAll r values are model dissonance vs human dissonance: POSITIVE = model correct.")
    print("partial_r controls for the chord's lowest MIDI note (raw variant only; the")
    print("transposed variant has a constant root, so there is nothing left to partial out).")
    print("No multiple-comparison correction is applied -- read the CIs, not the p values.")


if __name__ == "__main__":
    main()
