"""
Extract real overtone spectra (partial frequencies + amplitudes) from recorded
instrument audio via STFT + peak-picking, replacing idealized harmonic partials.

Deliberately avoids librosa's audio-loading/pitch path: this machine's
Application Control policy blocks llvmlite.dll, which numba (a hard
import-time dependency of librosa.core.audio) requires. soundfile (libsndfile,
no numba) handles decoding instead, and scipy.signal.stft handles the STFT.
"""
import numpy as np
import soundfile as sf
from scipy.signal import stft, find_peaks
import json

N_FFT = 8192
N_PARTIALS = 15
MAX_HARMONIC = 18  # search window in units of f0
HPS_HARMONICS = 5  # harmonic product spectrum downsampling factors

# Peak-acceptance threshold, as a fraction of the LOUDEST partial's magnitude.
# This is a free constant with no physical justification, and extraction_sensitivity.py
# exists because it turned out to be load-bearing: it is relative to the loudest
# partial, so a timbre with one dominant harmonic raises the bar for every other
# peak in the same spectrum. That is what produced the low partial counts
# FINDINGS.md flags for french horn (8) and oboe C5 (7) -- both of which recover
# a full set of integer-ratio harmonics when it is lowered. Exposed as a parameter
# rather than retuned, because the peaks it admits are genuinely near the noise
# floor and no evidence here says including them is more correct.
PROMINENCE = 0.005


def stable_region(y):
    """Skip attack transient and release; use the middle 60% of the signal."""
    n = len(y)
    return y[int(0.20 * n): int(0.80 * n)]


def note_to_hz(note, a4=440.0):
    """Equal-temperament note name (e.g. 'C4', 'Cs4' for C#4) to Hz, A4=440."""
    names = {"C": -9, "Cs": -8, "D": -7, "Ds": -6, "E": -5, "F": -4,
             "Fs": -3, "G": -2, "Gs": -1, "A": 0, "As": 1, "B": 2}
    for name_len in (2, 1):
        letters, digits = note[:name_len], note[name_len:]
        if letters in names and digits.lstrip("-").isdigit():
            octave = int(digits)
            semitones_from_a4 = names[letters] + (octave - 4) * 12
            return a4 * 2 ** (semitones_from_a4 / 12.0)
    raise ValueError(f"Unrecognized note name: {note}")


def parabolic_interp(mag, k, bin_hz):
    """Sub-bin refinement of a peak at integer bin k via parabolic interpolation
    on log-magnitude of the three bins around it. Returns (freq_offset_hz, amp).
    Without this, every extracted frequency is quantized to the nearest FFT
    bin (5.38 Hz at N_FFT=8192, sr=44100) -- enough error to visibly bias
    upper-partial ratios and materially move close roughness comparisons."""
    if k <= 0 or k >= len(mag) - 1:
        return 0.0, mag[k]
    y0, y1, y2 = np.log(mag[k - 1] + 1e-12), np.log(mag[k] + 1e-12), np.log(mag[k + 1] + 1e-12)
    denom = (y0 - 2 * y1 + y2)
    delta = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
    delta = float(np.clip(delta, -0.5, 0.5))
    amp = mag[k] * np.exp(0.25 * (y0 - y2) * delta)
    return delta * bin_hz, float(amp)


def detect_f0_hps(y, sr, fmin=60.0, fmax=2000.0):
    """Harmonic Product Spectrum f0 estimate from the averaged STFT magnitude.
    Unreliable for missing/weak-fundamental timbres (e.g. clarinet) -- prefer
    detect_f0_near() with a known nominal pitch when available."""
    f, t, Z = stft(y, fs=sr, nperseg=N_FFT, noverlap=N_FFT - N_FFT // 4)
    mag = np.abs(Z).mean(axis=1)

    hps = mag.copy()
    for h in range(2, HPS_HARMONICS + 1):
        decim = mag[::h]
        hps[: len(decim)] *= decim

    band = (f >= fmin) & (f <= fmax)
    f_band, hps_band = f[band], hps[band]
    f0 = float(f_band[np.argmax(hps_band)])
    return f0


def detect_f0_near(y, sr, nominal_hz, tol=0.05):
    """Find the strongest spectral peak within +/- tol of a known nominal pitch,
    refined to sub-bin precision via parabolic interpolation. Avoids the octave
    errors blind HPS makes on missing-fundamental timbres (e.g. clarinet's weak
    even harmonics), by using the sample library's own note label as a prior
    instead of pure blind estimation."""
    f, t, Z = stft(y, fs=sr, nperseg=N_FFT, noverlap=N_FFT - N_FFT // 4)
    mag = np.abs(Z).mean(axis=1)
    bin_hz = sr / N_FFT
    band = (f >= nominal_hz * (1 - tol)) & (f <= nominal_hz * (1 + tol))
    idx_in_band = np.nonzero(band)[0]
    k = idx_in_band[np.argmax(mag[band])]
    offset, _ = parabolic_interp(mag, k, bin_hz)
    return float(f[k] + offset)


def extract_partials(path, n_partials=N_PARTIALS, nominal_hz=None, window=None,
                     prominence=PROMINENCE):
    """window=None: use the middle 60% of the file (assumes a short,
    attack-sustain-release sample-library note, as used by Philharmonia).
    window=(start_s, dur_s): use an explicit slice instead -- needed for
    long full-decay recordings (e.g. University of Iowa's ~36s piano notes),
    where "middle 60%" would land in the near-silent tail rather than the
    sustained portion."""
    y, sr = sf.read(path, always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)

    if window is not None:
        start_s, dur_s = window
        y_stable = y[int(start_s * sr): int((start_s + dur_s) * sr)]
    else:
        y_stable = stable_region(y)

    if nominal_hz is not None:
        f0 = detect_f0_near(y_stable, sr, nominal_hz)
    else:
        f0 = detect_f0_hps(y_stable, sr)

    f, t, Z = stft(y_stable, fs=sr, nperseg=N_FFT, noverlap=N_FFT - N_FFT // 4)
    mag = np.abs(Z).mean(axis=1)

    fmax = f0 * MAX_HARMONIC
    mask = (f > f0 * 0.5) & (f < fmax)
    freqs_m, mag_m = f[mask], mag[mask]

    bin_hz = sr / N_FFT
    peak_idx, _ = find_peaks(
        mag_m, distance=max(1, int((f0 * 0.5) / bin_hz)), prominence=mag_m.max() * prominence
    )
    refined = [parabolic_interp(mag_m, k, bin_hz) for k in peak_idx]
    peak_freqs = freqs_m[peak_idx] + np.array([r[0] for r in refined])
    peak_amps = np.array([r[1] for r in refined])

    top = np.argsort(peak_amps)[::-1][:n_partials]
    pf, pa = peak_freqs[top], peak_amps[top]
    order = np.argsort(pf)
    pf, pa = pf[order], pa[order]

    pa = pa / pa.max()
    ratios = pf / f0

    return {
        "f0_hz": f0,
        "n_partials_found": len(pf),
        # n_partials_found is min(peaks accepted, n_partials). Recording the
        # pre-cap count separates "this timbre has few resolvable harmonics"
        # from "the cap truncated it", which the flagged spectra made worth
        # distinguishing -- 15 was a ceiling for most timbres, not a measurement.
        "n_peaks_before_cap": int(len(peak_freqs)),
        "cap_was_binding": bool(len(peak_freqs) > n_partials),
        "prominence": prominence,
        "n_partials_cap": n_partials,
        "ratios": ratios.tolist(),
        "freqs_hz": pf.tolist(),
        "rel_amps": pa.tolist(),
    }


if __name__ == "__main__":
    import glob, os

    results = {}
    paths = sorted(glob.glob("audio_samples/*.mp3")) + sorted(glob.glob("audio_samples/*.aiff"))
    for path in paths:
        stem = os.path.basename(path).rsplit(".", 1)[0]
        instrument, note = stem.rsplit("_", 1)
        # Iowa piano recordings are ~36s full-decay captures, not short
        # sample-library notes -- use an explicit window (skip the hammer
        # attack, use the next 2s of sustain) instead of "middle 60%".
        window = (0.3, 2.0) if instrument == "piano" else None
        data = extract_partials(path, nominal_hz=note_to_hz(note), window=window)
        results[stem] = {"instrument": instrument, "note": note, **data}
        print(f"{stem}: f0={data['f0_hz']:.2f} Hz (nominal {note}), {data['n_partials_found']} partials found")
        for r, a in zip(data["ratios"], data["rel_amps"]):
            print(f"    ratio={r:6.3f}  rel_amp={a:5.3f}")

    with open("extracted_partials.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print("\nSaved to extracted_partials.json")
