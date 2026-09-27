"""
Result 9b: how much of Results 1-3 is decided by two constants in the extractor
rather than by the recordings?

Result 9a (codec_ab.py) ruled out the audio codec: encoding a lossless recording
down to 48 kbps moves the tritone/m6 gap by at most 3.6 percentage points and
never flips an ordering. That kills the explanation FINDINGS.md offered for its
two low-confidence spectra -- french horn (8 partials) and oboe C5 (7) -- and
leaves the question of what did cause them.

The answer is in extract_partials.py, not in the audio. Peaks are accepted at
`prominence = 0.005 * (loudest partial)`. That threshold is relative to the
loudest partial in the same spectrum, so a timbre with one dominant harmonic
raises the acceptance bar for all its other harmonics. Philharmonia's oboe C5 has
its third harmonic 20x louder than its fundamental; its upper harmonics are
therefore measured against a very tall reference and rejected. Lowering the
threshold recovers them, and they are at integer ratios -- 9.00, 10.00, ... 17.00
for the horn -- so they are real harmonics that were being discarded, not noise
being admitted:

    oboe C5      7 partials at 0.005  ->  15 at 0.0002
    french horn  8                    ->  16
    clarinet C5  9                    ->  15

The second constant is `N_PARTIALS = 15`, a hard cap. Most timbres hit it, so
"15 partials found" was a ceiling rather than a measurement for those.

Neither constant is retuned here. The peaks the loose setting admits sit at
amplitudes of 0.001-0.004 relative to the loudest, against a median noise floor
around 0.0003 -- real, but barely above the floor, and no evidence in this project
says including them is more correct than excluding them. What this script does is
measure how much each published number depends on the choice.

The reason to expect that dependence to differ by model, and the specific thing to
watch for: Vassilakis weights each pair by (a1*a2)^0.1. That exponent is close to
zero, so it compresses six orders of magnitude of amplitude into a factor of four
-- a pair of partials at amplitude 0.001 contributes (1e-6)^0.1 = 0.25, against
1.0 for a pair at full scale. Sethares' a1*a2 gives the same pair 1e-6. So
Vassilakis should be the model most sensitive to whether marginal partials are
admitted, and Sethares the least, for a reason that is visible in the formulas
before any measurement.
"""
import json
import os

import numpy as np

from dissonance_model import (
    interval_sweep, sethares_dissonance, sethares_min_dissonance,
    vassilakis_dissonance, hutch_knopoff_dissonance,
)
from dissonance_model import harmonic_partials
from extract_partials import extract_partials, note_to_hz, N_PARTIALS, PROMINENCE

MODELS = [
    ("Sethares", sethares_dissonance),
    ("Sethares-min", sethares_min_dissonance),
    ("Vassilakis", vassilakis_dissonance),
    ("Hutch-Knopoff", hutch_knopoff_dissonance),
]

# The published settings are the first entry in each list, so the "as published"
# row is always grid[0] and every delta below is measured against it.
PROMINENCES = [PROMINENCE, 0.002, 0.001, 0.0005, 0.0002]
CAPS = [N_PARTIALS, 25]

SOURCES = [
    ("bassoon_C4", "audio_samples/bassoon_C4.mp3", "C4", None),
    ("cello_C3", "audio_samples/cello_C3.mp3", "C3", None),
    ("cello_C4", "audio_samples/cello_C4.mp3", "C4", None),
    ("cello_C5", "audio_samples/cello_C5.mp3", "C5", None),
    ("clarinet_C4", "audio_samples/clarinet_C4.mp3", "C4", None),
    ("clarinet_C5", "audio_samples/clarinet_C5.mp3", "C5", None),
    ("clarinet_D3", "audio_samples/clarinet_D3.mp3", "D3", None),
    ("flute_C4", "audio_samples/flute_C4.mp3", "C4", None),
    ("french-horn_C4", "audio_samples/french-horn_C4.mp3", "C4", None),
    ("oboe_As3", "audio_samples/oboe_As3.mp3", "As3", None),
    ("oboe_C4", "audio_samples/oboe_C4.mp3", "C4", None),
    ("oboe_C5", "audio_samples/oboe_C5.mp3", "C5", None),
    ("saxophone_C4", "audio_samples/saxophone_C4.mp3", "C4", None),
    ("piano_C4", "audio_samples/piano_C4.aiff", "C4", (0.3, 2.0)),
]


def gaps_for(ratios, rel_amps, f0):
    """Tritone/m6 gap per model, on run_analysis.py's normalize-to-curve-max
    convention so these numbers are directly comparable to Result 1's table."""
    ratios, rel_amps = np.asarray(ratios), np.asarray(rel_amps)

    def pfn(f, **_):
        return f * ratios, rel_amps.copy()

    out = {}
    for name, fn in MODELS:
        cents, diss = interval_sweep(f0=f0, partials_fn=pfn, model=fn)
        diss = diss / diss.max()
        t = float(diss[np.argmin(np.abs(cents - 600))])
        m = float(diss[np.argmin(np.abs(cents - 800))])
        out[name] = {"gap_pct": round(100 * (t - m) / m, 2), "tritone_gt_m6": bool(t > m)}
    return out


def idealized_sweep():
    """The idealized baseline row of Result 1 is not extracted from audio, so the
    two constants swept above do not apply to it -- but it has two of its own,
    `n_partials=7` and `rolloff=0.9`, equally unjustified, and it is the row every
    real-instrument gap in Result 1 is read against. Swept here so the claim that
    no cell in that table changes sign covers all nine rows rather than eight."""
    published = (7, 0.9)
    grid = [(n, r) for n in (5, 7, 11) for r in (0.8, 0.9, 1.0)]
    per = {}
    for n, r in grid:
        f, a = harmonic_partials(261.63, n_partials=n, rolloff=r)
        per[(n, r)] = gaps_for(f / 261.63, a, 261.63)

    base = per[published]
    print(f"{'=' * 104}")
    print("idealized baseline (Result 1's reference row) -- its own two constants, "
          "n_partials x rolloff")
    print(f"  {'n':>3s} {'roll':>5s} | " + " ".join(f"{m:>21s}" for m, _ in MODELS))
    out = {}
    for n, r in grid:
        cells = []
        for m, _ in MODELS:
            g, g0 = per[(n, r)][m]["gap_pct"], base[m]["gap_pct"]
            flip = per[(n, r)][m]["tritone_gt_m6"] != base[m]["tritone_gt_m6"]
            cells.append(f"{g:+8.1f}%({g - g0:+6.1f}){'!' if flip else ' '}")
        mark = "  <- published" if (n, r) == published else ""
        print(f"  {n:3d} {r:5.1f} | " + " ".join(cells) + mark)
    for m, _ in MODELS:
        g0 = base[m]["gap_pct"]
        deltas = {f"n{n}_r{r}": round(per[(n, r)][m]["gap_pct"] - g0, 2) for n, r in grid}
        flips = [f"n{n}_r{r}" for n, r in grid
                 if per[(n, r)][m]["tritone_gt_m6"] != base[m]["tritone_gt_m6"]]
        worst = max(deltas, key=lambda k: abs(deltas[k]))
        out[m] = {"published_gap_pct": g0, "deltas_pp": deltas,
                  "max_abs_shift_pp": abs(deltas[worst]), "at": worst,
                  "settings_that_flip_ordering": flips}
    flipped = {m: v["settings_that_flip_ordering"] for m, v in out.items()
               if v["settings_that_flip_ordering"]}
    print(f"  max |shift|: " + "  ".join(f"{m}={out[m]['max_abs_shift_pp']:.1f}pp"
                                         for m, _ in MODELS))
    print(f"  ordering flips: {flipped if flipped else 'none'}")
    print()
    return {"published_setting": {"n_partials": 7, "rolloff": 0.9},
            "grid": [{"n_partials": n, "rolloff": r} for n, r in grid], "models": out}


def main():
    grid = [(p, c) for p in PROMINENCES for c in CAPS]
    published = grid[0]
    results = {"grid": [{"prominence": p, "cap": c} for p, c in grid],
               "published_setting": {"prominence": published[0], "cap": published[1]},
               "timbres": {}}

    print(f"Sweeping prominence {PROMINENCES} x cap {CAPS} over {len(SOURCES)} spectra.")
    print(f"Published setting is prominence={published[0]}, cap={published[1]}.\n")

    for key, path, note, window in SOURCES:
        if not os.path.exists(path):
            raise SystemExit(f"missing {path}")
        per_setting = {}
        for p, c in grid:
            d = extract_partials(path, n_partials=c, nominal_hz=note_to_hz(note),
                                 window=window, prominence=p)
            per_setting[(p, c)] = {"n": d["n_partials_found"],
                                   "pre_cap": d["n_peaks_before_cap"],
                                   "highest_ratio": round(max(d["ratios"]), 2),
                                   "gaps": gaps_for(d["ratios"], d["rel_amps"], d["f0_hz"])}

        base = per_setting[published]
        print(f"{'=' * 104}")
        print(f"{key}   published: {base['n']} partials, highest ratio {base['highest_ratio']}")
        print(f"  {'prom':>7s} {'cap':>4s} {'n':>4s} {'pre':>4s} | "
              + " ".join(f"{m:>21s}" for m, _ in MODELS))
        for p, c in grid:
            r = per_setting[(p, c)]
            cells = []
            for m, _ in MODELS:
                g, g0 = r["gaps"][m]["gap_pct"], base["gaps"][m]["gap_pct"]
                flip = r["gaps"][m]["tritone_gt_m6"] != base["gaps"][m]["tritone_gt_m6"]
                cells.append(f"{g:+8.1f}%({g - g0:+6.1f}){'!' if flip else ' '}")
            mark = "  <- published" if (p, c) == published else ""
            print(f"  {p:7g} {c:4d} {r['n']:4d} {r['pre_cap']:4d} | " + " ".join(cells) + mark)

        summary = {}
        for m, _ in MODELS:
            g0 = base["gaps"][m]["gap_pct"]
            deltas = {f"p{p:g}_c{c}": round(per_setting[(p, c)]["gaps"][m]["gap_pct"] - g0, 2)
                      for p, c in grid}
            flips = [k for (p, c), k in zip(grid, deltas)
                     if per_setting[(p, c)]["gaps"][m]["tritone_gt_m6"]
                     != base["gaps"][m]["tritone_gt_m6"]]
            worst = max(deltas, key=lambda k: abs(deltas[k]))
            summary[m] = {"published_gap_pct": g0, "deltas_pp": deltas,
                          "max_abs_shift_pp": abs(deltas[worst]), "at": worst,
                          "settings_that_flip_ordering": flips}
        results["timbres"][key] = {
            "published": {"n_partials": base["n"], "highest_ratio": base["highest_ratio"]},
            "n_partials_by_setting": {f"p{p:g}_c{c}": per_setting[(p, c)]["n"] for p, c in grid},
            "models": summary}
        print()

    results["idealized_baseline"] = idealized_sweep()

    # Aggregate: which model's published numbers are actually settled by the data,
    # and which are settled by the two constants.
    print(f"{'=' * 104}")
    print("AGGREGATE -- movement of the tritone/m6 gap across the extraction grid, "
          "over all 14 spectra")
    print(f"{'=' * 104}")
    print(f"  {'model':14s} {'median |shift|':>15s} {'max |shift|':>13s} {'worst timbre':>16s} "
          f"{'orderings flipped':>19s}")
    agg = {}
    for m, _ in MODELS:
        shifts = {k: v["models"][m]["max_abs_shift_pp"] for k, v in results["timbres"].items()}
        flipped = [k for k, v in results["timbres"].items()
                   if v["models"][m]["settings_that_flip_ordering"]]
        worst_t = max(shifts, key=lambda k: shifts[k])
        agg[m] = {"median_max_shift_pp": round(float(np.median(list(shifts.values()))), 2),
                  "max_shift_pp": round(shifts[worst_t], 2), "worst_timbre": worst_t,
                  "timbres_with_an_ordering_flip": flipped,
                  "n_timbres_flipped": len(flipped)}
        print(f"  {m:14s} {agg[m]['median_max_shift_pp']:15.2f} {agg[m]['max_shift_pp']:13.2f} "
              f"{worst_t:>16s} {len(flipped):11d} of 14  {flipped if flipped else ''}")
    results["aggregate"] = agg

    print("\n  Read this as: how many percentage points the published tritone/m6 gap moves")
    print("  when two extraction constants with no physical justification are varied over a")
    print("  defensible range. A model whose ordering flips has no result here to report -- the")
    print("  sign of its answer is set by the extractor, not by the instrument.")

    with open("extraction_sensitivity_results.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print("\nSaved extraction_sensitivity_results.json")


if __name__ == "__main__":
    main()
