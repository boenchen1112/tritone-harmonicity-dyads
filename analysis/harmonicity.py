"""
Spectral harmonicity (Harrison & Pearce 2018/2020, after Milne 2013) for
ARBITRARY partial spectra at CONTINUOUS pitches.

incon's har_18_harmonicity takes pitch-class sets and idealised harmonic tones.
This reimplementation takes the pooled partials themselves (frequency, linear
amplitude), so it can be evaluated on stretched, compressed, gamelan and
measured instrument spectra and on a continuous interval grid.

Algorithm (transcribed from incon 0.5.0 / hrep: pc_harmonicity,
sweep_harmonic_template, pc_spectrum_template_1/2, kl_div_from_uniform):
  1. Pitch-class spectrum: 1200 one-cent bins on a circle. Each partial adds a
     Gaussian (sigma = 6.83 cents, truncated at 12 sigma) of mass = its weight
     at its pitch class. Idealised tones: harmonics i = 1..11, weight i^-rho,
     rho = 0.75 (incon passes rho = roll_off * 0.75 with roll_off = 1).
  2. Harmonic template: the same spectrum for one idealised tone at pc 0.
  3. Virtual-pitch profile: cosine similarity between the spectrum and the
     template rotated to every one of the 1200 offsets.
  4. Harmonicity: KL divergence of the normalised profile from uniform (bits).
     Peaked profile (one clear virtual pitch) = high harmonicity.
Validated against incon's har_18_harmonicity on all 401 rated chords
(see __main__).
"""
import numpy as np

N_BINS = 1200
SIGMA = 6.83
NUM_HARMONICS = 11
RHO = 0.75
REF_HZ = 261.6255653005986  # C4: pitch class 0


def _kernel(sigma=SIGMA):
    limit = int(np.floor(sigma * 12))
    d = np.arange(-limit, limit + 1)
    return d, np.exp(-0.5 * (d / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def pc_spectrum(pcs_cents, weights, sigma=SIGMA, quantise=False):
    """Smoothed pitch-class spectrum from partial positions (cents mod 1200)."""
    out = np.zeros(N_BINS)
    pcs = np.mod(np.asarray(pcs_cents, float), N_BINS)
    w = np.asarray(weights, float)
    if quantise:  # incon rounds each partial to the nearest bin
        d, k = _kernel(sigma)
        for p, m in zip(np.round(pcs).astype(int) % N_BINS, w):
            np.add.at(out, (p + d) % N_BINS, m * k)
        return out
    # continuous: evaluate the Gaussian at bin centres with the exact offset
    limit = int(np.floor(sigma * 12))
    for p, m in zip(pcs, w):
        base = int(np.floor(p))
        idx = np.arange(base - limit, base + limit + 2)
        dist = idx - p
        out_idx = idx % N_BINS
        np.add.at(out, out_idx, m * np.exp(-0.5 * (dist / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi)))
    return out


def harmonic_template(num_harmonics=NUM_HARMONICS, rho=RHO, sigma=SIGMA):
    i = np.arange(1, num_harmonics + 1)
    return pc_spectrum(1200 * np.log2(i), i ** -rho, sigma, quantise=True)


_TEMPLATE_FFT = {}


def virtual_pitch_profile(spec, num_harmonics=NUM_HARMONICS, rho=RHO, sigma=SIGMA):
    key = (num_harmonics, rho, sigma)
    if key not in _TEMPLATE_FFT:
        t = harmonic_template(num_harmonics, rho, sigma)
        _TEMPLATE_FFT[key] = (np.conj(np.fft.fft(t)), np.linalg.norm(t))
    tf, tn = _TEMPLATE_FFT[key]
    # y[k] = sum_j t[j - k] x[j]  (template shifted to offset k)
    xc = np.real(np.fft.ifft(np.fft.fft(spec) * tf))
    return xc / (tn * np.linalg.norm(spec))


def kl_from_uniform(y):
    p = y / y.sum()
    p = p[p > 0]
    return float(np.sum(p * np.log2(p * len(y))))


def harmonicity_partials(freqs, amps, ref_hz=REF_HZ, weight_power=1.0, **kw):
    """Harmonicity of a pooled partial spectrum (Hz, linear amplitude)."""
    freqs = np.asarray(freqs, float)
    pcs = 1200 * np.log2(freqs / ref_hz)
    spec = pc_spectrum(pcs, np.asarray(amps, float) ** weight_power, kw.get("sigma", SIGMA))
    return kl_from_uniform(virtual_pitch_profile(spec, **kw))


def harmonicity_pc_set(pcs_semitones, quantise=True, num_harmonics=NUM_HARMONICS, rho=RHO):
    """incon-compatible: idealised harmonic tones on a pitch-class set."""
    i = np.arange(1, num_harmonics + 1)
    pcs, ws = [], []
    for pc in sorted(set(np.mod(np.asarray(pcs_semitones, float), 12))):
        pcs.extend(pc * 100 + 1200 * np.log2(i))
        ws.extend(i ** -rho)
    spec = pc_spectrum(pcs, ws, quantise=quantise)
    return kl_from_uniform(virtual_pitch_profile(spec, num_harmonics, rho))


if __name__ == "__main__":
    import csv
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from paths import PROJECT_ROOT
    root = PROJECT_ROOT
    for ds in ("bowling2018", "johnson-laird2012"):
        rows = list(csv.DictReader(open(root / f"reference_data/incon_{ds}.csv")))
        ref = np.array([float(r["har_18_harmonicity"]) for r in rows])
        ours = np.array([harmonicity_pc_set([int(x) % 12 for x in r["chord"].split()]) for r in rows])
        cont = np.array([harmonicity_pc_set([int(x) % 12 for x in r["chord"].split()], quantise=False)
                         for r in rows])
        print(ds, "n", len(rows), "max|diff|", np.abs(ours - ref).max(), "r", np.corrcoef(ours, ref)[0, 1],
              "continuous-vs-ref max|diff|", np.abs(cont - ref).max())
