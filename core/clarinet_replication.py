"""
Result 10: does the clarinet finding replicate on a second recording library?

Result 2 is the one register-stable finding in this project: under Hutchinson &
Knopoff, a real clarinet spectrum makes the minor 6th MORE dissonant than the
tritone at every register tested -- D3 -30.8%, C4 -46.6%, C5 -43.2%. Result 9
showed that finding is robust to both free constants in the extractor (it moves
by at most 0.54 pp across the whole grid) and to the audio codec. What Result 9
explicitly could not rule out is the recording itself: every number rests on one
performance by one player in one room on one microphone. "The clarinet flips" and
"this clarinet recording flips" are not distinguished by anything done so far.

This runs the same three registers on a second library -- University of Iowa MIS,
lossless, a different player, instrument, room, microphone and recording decade.

WHAT THIS CAN AND CANNOT SHOW. This is a five-factor change, not a controlled
experiment. If the two libraries disagree, this cannot say which factor did it. If
they agree, that is meaningful: agreement across five uncontrolled factors is
evidence the finding tracks something about clarinets rather than something about
one recording. Asymmetric, and worth running for that reason, but the asymmetry
should be stated rather than discovered later.

VERDICT CRITERION, FIXED BEFORE THE NUMBERS WERE SEEN. Result 2's claim is a claim
about SIGN: H&K puts the tritone below the minor 6th at all three registers.
Magnitudes are not comparable across libraries -- different player, room and mic
change absolute gap sizes for reasons that have nothing to do with the hypothesis.
So:

    REPLICATION SUCCEEDS iff the Iowa clarinet's H&K gap is negative at D3, C4
    and C5. Nothing else counts, and a -12% is as much a success as a -46%.

Written here before running, because a -12% sitting next to a -46.6% invites
negotiating with yourself about what "replicates" means.

SECONDARY, AND EXPLICITLY WEAKER: Philharmonia's Sethares C4 gap is -3.4%. That is
a near-tie balanced on a sign boundary, and this project's own history includes a
DSP precision bug that flipped exactly this quantity from +1.6% to -3.6%. If Iowa's
Sethares C4 lands positive, that is a near-zero quantity falling on the other side
of zero, not a second failure of Result 2. Flagged in advance so it cannot be read
as one afterwards.

The extraction grid from Result 9b is re-run on the Iowa spectra here, in the same
script. Without it, any Iowa/Philharmonia difference has two candidate causes --
the library, or the extraction settings -- and they cannot be separated. With it,
the comparison becomes "the libraries differ by X; extraction moves either one by
at most Y."
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from codec_ab import WORK, cut_note
from dissonance_model import (
    interval_sweep, sethares_dissonance, sethares_min_dissonance,
    vassilakis_dissonance, hutch_knopoff_dissonance,
)
from extract_partials import extract_partials, note_to_hz
from extraction_sensitivity import PROMINENCES, CAPS

MODELS = [
    ("Sethares", sethares_dissonance),
    ("Sethares-min", sethares_min_dissonance),
    ("Vassilakis", vassilakis_dissonance),
    ("Hutch-Knopoff", hutch_knopoff_dissonance),
]

# note -> (Iowa chromatic run containing it, Philharmonia file, key in extracted_partials.json)
REGISTERS = [
    ("D3", "BbClar.ff.D3B3.aiff", "audio_samples/clarinet_D3.mp3", "clarinet_D3"),
    ("C4", "BbClar.ff.C4B4.aiff", "audio_samples/clarinet_C4.mp3", "clarinet_C4"),
    ("C5", "BbClar.ff.C5B5.aiff", "audio_samples/clarinet_C5.mp3", "clarinet_C5"),
]


def sweep(ratios, rel_amps, f0):
    ratios, rel_amps = np.asarray(ratios), np.asarray(rel_amps)

    def pfn(f, **_):
        return f * ratios, rel_amps.copy()

    curves, gaps = {}, {}
    for name, fn in MODELS:
        cents, diss = interval_sweep(f0=f0, partials_fn=pfn, model=fn)
        diss = diss / diss.max()
        curves[name] = (cents, diss)
        t = float(diss[np.argmin(np.abs(cents - 600))])
        m = float(diss[np.argmin(np.abs(cents - 800))])
        gaps[name] = {"gap_pct": round(100 * (t - m) / m, 2), "tritone_gt_m6": bool(t > m)}
    return curves, gaps


def even_odd_ratio(ratios, amps):
    """Mean amplitude of even-numbered harmonics over mean amplitude of odd ones.

    The clarinet's textbook signature is suppressed even harmonics (a cylindrical
    bore closed at one end). If both libraries show the suppression and the H&K
    signs still disagree, then the flip does not follow from the feature anyone
    would name as its cause -- which is a sharper finding than a gap table alone.
    Partials are assigned to a harmonic number by rounding their ratio; anything
    more than a quarter-tone from an integer is inharmonic and excluded."""
    ratios, amps = np.asarray(ratios), np.asarray(amps)
    n = np.round(ratios)
    keep = (np.abs(ratios - n) < 0.03 * np.maximum(1.0, n)) & (n >= 1)
    n, a = n[keep].astype(int), amps[keep]
    ev, od = a[n % 2 == 0], a[n % 2 == 1]
    if len(ev) == 0 or len(od) == 0:
        return float("nan"), len(ev), len(od)
    return float(np.mean(ev) / np.mean(od)), len(ev), len(od)


def extraction_range(path, note, window=None):
    """Max |shift| of each model's gap over the Result 9b grid, for one spectrum.
    This is the yardstick the library difference has to beat to mean anything."""
    base = None
    per = {}
    for p in PROMINENCES:
        for c in CAPS:
            d = extract_partials(path, n_partials=c, nominal_hz=note_to_hz(note),
                                 window=window, prominence=p)
            _, g = sweep(d["ratios"], d["rel_amps"], d["f0_hz"])
            if base is None:
                base = g
            for m, _ in MODELS:
                per.setdefault(m, []).append(g[m]["gap_pct"] - base[m]["gap_pct"])
    return {m: round(float(np.max(np.abs(v))), 2) for m, v in per.items()}


def main():
    phil_all = json.load(open("extracted_partials.json"))
    out = {"criterion": "H&K gap negative at D3, C4 and C5 (sign only; magnitudes "
                        "are not comparable across libraries)",
           "registers": {}}
    curves_by_reg = {}

    for note, iowa_run, phil_path, phil_key in REGISTERS:
        wav = os.path.join(WORK, f"clarinet_{note}_lossless.wav")
        if not os.path.exists(wav):
            cut_note(iowa_run, note, wav)
        iowa = extract_partials(wav, nominal_hz=note_to_hz(note))
        phil = phil_all[phil_key]

        print(f"\n{'=' * 100}")
        print(f"clarinet {note}   Iowa: {iowa_run}   Philharmonia: {os.path.basename(phil_path)}")
        print(f"{'=' * 100}")
        eo_i = even_odd_ratio(iowa["ratios"], iowa["rel_amps"])
        eo_p = even_odd_ratio(phil["ratios"], phil["rel_amps"])
        print(f"  {'':14s} {'f0 (Hz)':>9s} {'partials':>9s} {'highest ratio':>14s} "
              f"{'even/odd amp':>13s}")
        print(f"  {'Iowa':14s} {iowa['f0_hz']:9.2f} {iowa['n_partials_found']:9d} "
              f"{max(iowa['ratios']):14.2f} {eo_i[0]:13.3f}")
        print(f"  {'Philharmonia':14s} {phil['f0_hz']:9.2f} {phil['n_partials_found']:9d} "
              f"{max(phil['ratios']):14.2f} {eo_p[0]:13.3f}")

        c_i, g_i = sweep(iowa["ratios"], iowa["rel_amps"], iowa["f0_hz"])
        c_p, g_p = sweep(phil["ratios"], phil["rel_amps"], phil["f0_hz"])
        curves_by_reg[note] = {"Iowa": c_i, "Philharmonia": c_p}

        rng_i = extraction_range(wav, note)
        rng_p = extraction_range(phil_path, note)

        print(f"\n  {'model':14s} {'Iowa gap':>10s} {'Phil gap':>10s} {'difference':>11s} "
              f"{'extraction +/-':>15s} {'sign agrees':>12s}")
        rows = {}
        for m, _ in MODELS:
            gi, gp = g_i[m]["gap_pct"], g_p[m]["gap_pct"]
            same = (gi < 0) == (gp < 0)
            tol = max(rng_i[m], rng_p[m])
            rows[m] = {"iowa_gap_pct": gi, "phil_gap_pct": gp,
                       "difference_pp": round(gi - gp, 2),
                       "iowa_extraction_range_pp": rng_i[m],
                       "phil_extraction_range_pp": rng_p[m],
                       "difference_exceeds_extraction_noise": bool(abs(gi - gp) > tol),
                       "sign_agrees": bool(same)}
            print(f"  {m:14s} {gi:+9.1f}% {gp:+9.1f}% {gi - gp:+10.1f}pp "
                  f"{tol:14.2f}  {'YES' if same else 'NO':>12s}")

        # r between the two libraries' full dissonance curves. Reported alongside the
        # gap table, never instead of it: every dissonance curve shares the same gross
        # shape (peak near unison, dips at simple ratios, low at the octave), so this
        # runs high for reasons that have nothing to do with clarinets.
        print(f"\n  full-curve correlation Iowa vs Philharmonia (0-1200c, 5c steps):")
        curve_r = {}
        for m, _ in MODELS:
            r = float(np.corrcoef(c_i[m][1], c_p[m][1])[0, 1])
            curve_r[m] = round(r, 4)
            print(f"    {m:14s} r={r:+.4f}")

        out["registers"][note] = {
            "iowa": {"source": iowa_run, "f0_hz": round(iowa["f0_hz"], 2),
                     "n_partials": iowa["n_partials_found"],
                     "even_odd_amp_ratio": round(eo_i[0], 4),
                     "ratios": [round(x, 4) for x in iowa["ratios"]],
                     "rel_amps": [round(x, 4) for x in iowa["rel_amps"]]},
            "philharmonia": {"source": os.path.basename(phil_path),
                             "f0_hz": round(phil["f0_hz"], 2),
                             "n_partials": phil["n_partials_found"],
                             "even_odd_amp_ratio": round(eo_p[0], 4)},
            "models": rows, "curve_r": curve_r}

    # --- verdict, against the criterion fixed in the docstring ---
    print(f"\n{'=' * 100}")
    print("VERDICT -- criterion fixed before the run: H&K gap negative at all three registers")
    print(f"{'=' * 100}")
    hk = {n: out["registers"][n]["models"]["Hutch-Knopoff"] for n, *_ in REGISTERS}
    for n in hk:
        print(f"  {n}:  Iowa {hk[n]['iowa_gap_pct']:+7.1f}%   Philharmonia "
              f"{hk[n]['phil_gap_pct']:+7.1f}%   -> {'both negative' if hk[n]['sign_agrees'] and hk[n]['iowa_gap_pct'] < 0 else 'DISAGREE'}")
    replicated = all(v["iowa_gap_pct"] < 0 for v in hk.values())
    out["replication"] = {
        "hutch_knopoff_negative_at_all_three": bool(replicated),
        "verdict": "REPLICATED" if replicated else "NOT REPLICATED"}
    print(f"\n  Result 2 {'REPLICATES' if replicated else 'DOES NOT REPLICATE'} "
          f"on the Iowa clarinet.")

    seth_c4 = out["registers"]["C4"]["models"]["Sethares"]
    print(f"\n  Secondary (weaker, flagged in advance): Sethares at C4 -- "
          f"Philharmonia {seth_c4['phil_gap_pct']:+.1f}%, Iowa {seth_c4['iowa_gap_pct']:+.1f}%. "
          f"{'Sign agrees.' if seth_c4['sign_agrees'] else 'Sign differs -- a near-zero quantity landing on the other side of zero, not a failure of the H&K claim.'}")
    out["secondary_sethares_c4"] = seth_c4

    for m, _ in MODELS:
        fig, ax = plt.subplots(figsize=(10, 6))
        for note in curves_by_reg:
            for lib, style in (("Philharmonia", "--"), ("Iowa", "-")):
                cents, diss = curves_by_reg[note][lib][m]
                ax.plot(cents, diss, style, linewidth=1.6, label=f"{lib} {note}")
        ax.axvline(600, color="red", linestyle=":", alpha=0.7)
        ax.axvline(800, color="orange", linestyle=":", alpha=0.7)
        ax.set_xlabel("Interval above root (cents)")
        ax.set_ylabel("Dissonance (normalized to curve max)")
        ax.set_title(f"{m}: clarinet, two libraries x three registers "
                     f"(dotted = tritone 600c / m6 800c)")
        ax.set_xlim(0, 1200)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        path = f"clarinet_replication_{m.lower()}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  saved {path}")

    with open("clarinet_replication_results.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nSaved clarinet_replication_results.json")


if __name__ == "__main__":
    main()
