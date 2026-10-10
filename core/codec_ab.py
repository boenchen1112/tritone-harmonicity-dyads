"""
Result 9a: does mp3 encoding damage partial extraction enough to move the
tritone/m6 gap?

FINDINGS.md's Limitations flag the source audio as lossy and name two spectra as
low-confidence for that reason (french horn, 8 partials; oboe C5, 7). The obvious
repair -- re-extract from lossless -- is not actually available for the
Philharmonia recordings (see PROVENANCE below), and swapping in a different
library would change the player, the instrument, the room and the microphone at
the same time as the codec. Any change in the results would then be
uninterpretable: exactly the confound that turned out to be driving Result 4
before it was corrected.

So this asks the codec question directly instead, with nothing else varying.
One lossless University of Iowa recording is encoded to mp3 by this script at the
bitrates the Philharmonia mirror actually uses, and the identical extraction
pipeline runs on the lossless original and on each encode. Same performance, same
note, same room, same microphone, same code. The only difference is the codec.

PROVENANCE, corrected. LITERATURE_BEYOND_ROUGHNESS.md sec.5 and sec.7.6 state that the
Philharmonia library "is available in lossless form from its own site". It is not.
The official distribution at philharmonia.co.uk/resources/sound-samples/ links five
zips on philharmonia-assets.s3-eu-west-1.amazonaws.com; reading the first local
file header of Woodwind.zip (273,360,818 bytes) gives a nested per-instrument zip
whose first entry is `bass-clarinet_A2_025_forte_normal.mp3`, 7,312 bytes
uncompressed. The official source is mp3. Corrected in that file.

BITRATES, corrected. FINDINGS.md describes the samples as "~64kbps mp3 for 7 of 8
instruments". ffprobe on the local files says otherwise -- 64 kbps for bassoon and
the three clarinets, 80 kbps for saxophone, 96 kbps for everything else, all mono
44.1 kHz. The two spectra flagged as codec casualties (oboe C5, french horn) are
both 96 kbps, i.e. the *least* compressed material in the set, which already makes
the codec explanation for them less likely than the write-up assumed.

IOWA LABELS ARE SOUNDING PITCH, verified rather than assumed. A Bb clarinet part
is normally written a major second high, so `BbClar.ff.C4B4.aiff` could plausibly
mean written C4 = sounding Bb3 (233.1 Hz), which would shift every note label in
the run. The first segment's strongest peak below 700 Hz is at 263.8 Hz, so the
labels are sounding pitch. (Its second harmonic sits at 0.03 of that amplitude --
the clarinet's suppressed even harmonics, visible in the raw spectrum.)
"""
import json
import os
import re
import subprocess

import numpy as np
import soundfile as sf

from dissonance_model import (
    interval_sweep, sethares_dissonance, sethares_min_dissonance,
    vassilakis_dissonance, hutch_knopoff_dissonance,
)
from extract_partials import extract_partials, note_to_hz

MODELS = [
    ("Sethares", sethares_dissonance),
    ("Sethares-min", sethares_min_dissonance),
    ("Vassilakis", vassilakis_dissonance),
    ("Hutch-Knopoff", hutch_knopoff_dissonance),
]

# The bitrates measured in audio_samples/ by ffprobe, plus 48k as a deliberately
# worse-than-anything-used case to show which direction the effect runs.
BITRATES = [96, 80, 64, 48]

WORK = "audio_samples/iowa"
CHROMATIC = ["C", "Cs", "D", "Ds", "E", "F", "Fs", "G", "Gs", "A", "As", "B"]
_FLAT_TO_SHARP = {"Cb": "B", "Db": "Cs", "Eb": "Ds", "Fb": "E",
                  "Gb": "Fs", "Ab": "Gs", "Bb": "As"}

# Which lossless run to pull each test note from. Both are the instruments the
# question is actually about: clarinet carries Result 2's one register-stable
# finding, and oboe C5 is one of the two spectra FINDINGS.md calls low-confidence.
TARGETS = [
    ("BbClar.ff.C4B4.aiff", "C4", "clarinet"),
    ("Oboe.ff.C4B4.aiff", "C4", "oboe"),
    ("Oboe.ff.C5B5.aiff", "C5", "oboe"),
]


def _norm_note(name, octave):
    name = _FLAT_TO_SHARP.get(name, name.replace("#", "s"))
    return f"{name}{octave}"


def parse_range(filename):
    """`Oboe.ff.C5B5.aiff` -> ['C5', 'Cs5', ... 'B5']. Iowa encodes each run's
    span in the filename; generating the chromatic sequence from it and checking
    the length against the segmenter is what makes the note labels checkable
    rather than assumed."""
    m = re.search(r"\.([A-G][b#]?)(-?\d)([A-G][b#]?)(-?\d)\.", filename)
    if not m:
        raise ValueError(f"no note range in {filename}")
    lo = _norm_note(m.group(1), m.group(2))
    hi = _norm_note(m.group(3), m.group(4))
    out, cur = [], lo
    for _ in range(64):
        out.append(cur)
        if cur == hi:
            return out
        i = CHROMATIC.index(cur[:-1]) if cur[:-1] in CHROMATIC else None
        octave = int(cur[-1])
        i += 1
        if i == 12:
            i, octave = 0, octave + 1
        cur = f"{CHROMATIC[i]}{octave}"
    raise ValueError(f"runaway range in {filename}")


def segment_notes(y, sr, thr_db=-35.0, min_dur=0.25):
    """Split a chromatic run into individual notes on a 30 ms RMS envelope.
    Verified threshold-stable: -40, -35 and -30 dB all return exactly 12 notes
    for all three files used here, with segment durations moving under 40 ms."""
    hop, win = int(0.010 * sr), int(0.030 * sr)
    n = (len(y) - win) // hop
    rms = np.array([np.sqrt(np.mean(y[i * hop:i * hop + win] ** 2)) for i in range(n)])
    db = 20 * np.log10(rms / (rms.max() + 1e-12) + 1e-12)
    on = db > thr_db
    d = np.diff(on.astype(int))
    starts, ends = np.nonzero(d == 1)[0] + 1, np.nonzero(d == -1)[0] + 1
    if on[0]:
        starts = np.r_[0, starts]
    if on[-1]:
        ends = np.r_[ends, len(on)]
    return [(a * hop / sr, b * hop / sr) for a, b in zip(starts, ends)
            if (b - a) * hop / sr > min_dur]


def cut_note(aiff_name, note, out_wav):
    """Write one note of a chromatic run to a lossless WAV, so that every encode
    below starts from the same PCM and the only variable is the codec."""
    path = os.path.join(WORK, aiff_name)
    y, sr = sf.read(path, always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    notes = parse_range(aiff_name)
    spans = segment_notes(y, sr)
    if len(spans) != len(notes):
        raise SystemExit(
            f"{aiff_name}: segmenter found {len(spans)} notes but the filename "
            f"declares {len(notes)} ({notes[0]}..{notes[-1]}). Do not guess which "
            f"is right -- inspect the envelope before trusting any label."
        )
    a, b = spans[notes.index(note)]
    sf.write(out_wav, y[int(a * sr):int(b * sr)], sr, subtype="PCM_16")
    return out_wav, sr, b - a


def encode(src_wav, kbps, out_mp3):
    """CBR mono 44.1 kHz LAME, matching the Philharmonia files' own encoding
    parameters as reported by ffprobe."""
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", src_wav,
         "-codec:a", "libmp3lame", "-b:a", f"{kbps}k", "-ar", "44100", "-ac", "1",
         out_mp3],
        check=True, capture_output=True,
    )
    return out_mp3


def gaps_for(ratios, rel_amps, f0):
    """Tritone/m6 gap per model for one measured spectrum, on the same
    normalize-to-curve-max convention run_analysis.py uses, so the numbers here
    are directly comparable to Result 1."""
    out = {}
    ratios, rel_amps = np.asarray(ratios), np.asarray(rel_amps)

    def pfn(f, **_):
        return f * ratios, rel_amps.copy()

    for name, fn in MODELS:
        cents, diss = interval_sweep(f0=f0, partials_fn=pfn, model=fn)
        diss = diss / diss.max()
        t = float(diss[np.argmin(np.abs(cents - 600))])
        m = float(diss[np.argmin(np.abs(cents - 800))])
        out[name] = {"tritone": round(t, 4), "m6": round(m, 4),
                     "tritone_gt_m6": bool(t > m),
                     "gap_pct": round(100 * (t - m) / m, 1)}
    return out


def spectrum_distance(a, b):
    """How far apart two measured spectra are, on the terms the models see.
    Resampling both amplitude sets onto the union of their partial ratios (nearest
    within a quarter-tone, 0 if the other spectrum has nothing there) is the honest
    comparison when the two do not even agree on how many partials exist."""
    ra, aa = np.asarray(a["ratios"]), np.asarray(a["rel_amps"])
    rb, ab = np.asarray(b["ratios"]), np.asarray(b["rel_amps"])
    grid = np.union1d(np.round(ra, 2), np.round(rb, 2))

    def resample(r, amp):
        out = np.zeros(len(grid))
        for i, g in enumerate(grid):
            j = np.argmin(np.abs(r - g))
            if abs(r[j] - g) < 0.03 * max(1.0, g):
                out[i] = amp[j]
        return out

    va, vb = resample(ra, aa), resample(rb, ab)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return {"n_grid": int(len(grid)),
            "cosine": round(float(va @ vb / denom), 5) if denom > 0 else float("nan"),
            "max_abs_amp_diff": round(float(np.max(np.abs(va - vb))), 4)}


def main():
    os.makedirs(WORK, exist_ok=True)
    results = {"bitrates_kbps": BITRATES, "targets": {}}

    for aiff, note, instrument in TARGETS:
        key = f"{instrument}_{note}"
        print(f"\n{'=' * 96}\n{key}   source: {aiff}\n{'=' * 96}")
        wav = os.path.join(WORK, f"{key}_lossless.wav")
        _, sr, dur = cut_note(aiff, note, wav)
        print(f"  cut {dur:.2f}s of lossless PCM_16 @ {sr} Hz")

        variants = {"lossless": wav}
        for kb in BITRATES:
            variants[f"mp3_{kb}k"] = encode(wav, kb, os.path.join(WORK, f"{key}_{kb}k.mp3"))

        rows = {}
        for label, path in variants.items():
            data = extract_partials(path, nominal_hz=note_to_hz(note))
            rows[label] = data
            print(f"  {label:10s} f0={data['f0_hz']:8.2f}  partials={data['n_partials_found']:3d}  "
                  f"highest ratio={max(data['ratios']):5.2f}  "
                  f"weakest amp={min(data['rel_amps']):.4f}")

        ref = rows["lossless"]
        print(f"\n  {'variant':10s} {'cos vs lossless':>16s} {'maxdAmp':>9s} | "
              + " ".join(f"{m:>15s}" for m, _ in MODELS))
        summary = {}
        for label, data in rows.items():
            g = gaps_for(data["ratios"], data["rel_amps"], data["f0_hz"])
            dist = spectrum_distance(ref, data)
            summary[label] = {"f0_hz": round(data["f0_hz"], 2),
                              "n_partials": data["n_partials_found"],
                              "highest_ratio": round(max(data["ratios"]), 3),
                              "spectrum_vs_lossless": dist, "gaps": g}
            cells = " ".join(f"{g[m]['gap_pct']:+9.1f}% {'T>m6' if g[m]['tritone_gt_m6'] else ' m6>T'}"
                             for m, _ in MODELS)
            print(f"  {label:10s} {dist['cosine']:16.5f} {dist['max_abs_amp_diff']:9.4f} | {cells}")

        # The number that decides whether the caveat mattered: how far any encode
        # moves the gap away from the lossless value, per model.
        print(f"\n  worst-case gap shift vs lossless, over all tested bitrates:")
        shifts = {}
        for m, _ in MODELS:
            base = summary["lossless"]["gaps"][m]["gap_pct"]
            deltas = {k: round(summary[k]["gaps"][m]["gap_pct"] - base, 2)
                      for k in variants if k != "lossless"}
            worst = max(deltas, key=lambda k: abs(deltas[k]))
            flipped = [k for k in variants
                       if summary[k]["gaps"][m]["tritone_gt_m6"]
                       != summary["lossless"]["gaps"][m]["tritone_gt_m6"]]
            shifts[m] = {"lossless_gap_pct": base, "deltas": deltas,
                         "worst": worst, "worst_delta_pp": deltas[worst],
                         "bitrates_that_flip_the_ordering": flipped}
            print(f"    {m:14s} lossless {base:+7.1f}%   worst {deltas[worst]:+6.2f} pp "
                  f"at {worst}" + (f"   ORDERING FLIPS at {flipped}" if flipped else "   ordering stable"))

        results["targets"][key] = {"source_file": aiff, "note": note,
                                   "instrument": instrument,
                                   "variants": summary, "gap_shifts": shifts}

    with open("codec_ab_results.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print("\nSaved codec_ab_results.json")


if __name__ == "__main__":
    main()
