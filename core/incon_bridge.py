"""
Python side of the bridge to pmcharrison/incon (Harrison & Pearce 2020), the
field-standard consonance package: 17 models spanning interference,
periodicity/harmonicity, culture, numerosity, and their composite.

FINDINGS.md previously listed this as "do not do this yet -- environment risk",
on the grounds that incon is a compiled R package with GitHub-only dependencies
on a machine whose Application Control policy already blocked llvmlite.dll. That
assessment is now obsolete: R 4.6.1 and incon 0.5.0 are installed and all 17
models run, including har_19_corpus, which is the cultural-familiarity component
this project could not previously touch at all.

Transport is Rscript + CSV rather than rpy2, deliberately. rpy2 links R's shared
library into the Python process, which is exactly the class of thing that has
already failed once on this machine. A subprocess writes a file and exits.

Note on spectra: incon takes MIDI pitches and expands them itself via
hrep::sparse_fr_spectrum, whose defaults are 11 harmonics with amplitude 1/n.
It will NOT accept an explicit (freqs, amps) pair, so incon's roughness values
cannot be run on this project's measured instrument spectra -- only on idealized
harmonics. hrep_partials() below reproduces that spectrum in Python so the two
implementations can be compared on identical input.
"""
import csv
import os
import subprocess
import tempfile

import numpy as np

R_CANDIDATES = [
    r"C:\Program Files\R\R-4.6.1\bin\x64\Rscript.exe",
    r"C:\Program Files\R\R-4.6.1\bin\Rscript.exe",
    "Rscript",
]

ALL_MODELS = [
    "gill_09_harmonicity", "har_18_harmonicity", "milne_13_harmonicity",
    "parn_88_root_ambig", "parn_94_complex", "stolz_15_periodicity",
    "bowl_18_min_freq_dist", "huron_94_dyadic", "hutch_78_roughness",
    "parn_94_pure", "seth_93_roughness", "vass_01_roughness", "wang_13_roughness",
    "jl_12_tonal", "har_19_corpus", "parn_94_mult", "har_19_composite",
]

MODEL_CLASS = {
    "gill_09_harmonicity": "harmonicity", "har_18_harmonicity": "harmonicity",
    "milne_13_harmonicity": "harmonicity", "parn_88_root_ambig": "harmonicity",
    "parn_94_complex": "harmonicity", "stolz_15_periodicity": "harmonicity",
    "bowl_18_min_freq_dist": "interference", "huron_94_dyadic": "interference",
    "hutch_78_roughness": "interference", "parn_94_pure": "interference",
    "seth_93_roughness": "interference", "vass_01_roughness": "interference",
    "wang_13_roughness": "interference",
    "jl_12_tonal": "culture", "har_19_corpus": "culture",
    "parn_94_mult": "numerosity", "har_19_composite": "composite",
}

# Reference chords for determining each model's sign convention EMPIRICALLY.
# 17 models with no shared orientation convention is the polarity bug of Result 4
# waiting to happen at seventeen times the scale, so orientation is measured, not
# declared. My first hand-written guess had jl_12_tonal and har_19_corpus
# backwards -- both are cost-like (higher = less consonant) despite reading like
# "tonalness" and "corpus fit" -- which is why this is computed rather than typed.
ORIENTATION_PROBE = {
    "consonant": [[60, 64, 67], [60, 65, 69], [60, 63, 67], [60, 67]],   # maj/min triads, P5
    "dissonant": [[60, 61, 62], [60, 61, 66], [60, 61], [60, 61, 62, 63]],  # clusters, m2
}


def measure_orientations(models=None, cache_path="reference_data/incon_orientation.csv"):
    """Return {model: +1 or -1}, where +1 means the model's raw output already
    increases with DISSONANCE and -1 means it must be negated to do so.

    Determined by running each model on chords whose direction nobody disputes.
    Models that fail to separate the two reference sets return 0 and must not be
    used as a consonance axis without a hand decision."""
    models = list(models or ALL_MODELS)
    probe = ORIENTATION_PROBE["consonant"] + ORIENTATION_PROBE["dissonant"]
    n_con = len(ORIENTATION_PROBE["consonant"])
    out = run_incon(probe, models, cache_path=cache_path)
    signs = {}
    for m in models:
        v = out[m]
        con, dis = float(np.nanmean(v[:n_con])), float(np.nanmean(v[n_con:]))
        signs[m] = 1 if dis > con else (-1 if dis < con else 0)
    return signs


def rscript_path():
    for cand in R_CANDIDATES:
        if cand == "Rscript" or os.path.exists(cand):
            return cand
    raise RuntimeError("Rscript not found; edit R_CANDIDATES in incon_bridge.py")


def hrep_partials(f0, num_harmonics=11, roll_off=1.0):
    """One note's partials under hrep::sparse_fr_spectrum's defaults: partial n
    at n*f0 with amplitude 1/n^roll_off."""
    n = np.arange(1, num_harmonics + 1)
    return f0 * n, 1.0 / n ** roll_off


def hrep_chord_spectrum(midi_notes, num_harmonics=11, roll_off=1.0, digits=6,
                        coherent=False):
    """hrep's spectrum for a whole CHORD, which is not simply the concatenation
    of each note's partials.

    Where two notes contribute partials at the SAME frequency (rounded to
    `digits`), hrep merges them into one, and by default combines their
    amplitudes in QUADRATURE -- sqrt(a1^2 + a2^2), not a1 + a2. That is
    incoherent (random-phase) power summation, which is the physically right
    default for independently-phased sources: C4 plus C5 gives 17 partials, not
    22, and the 0.5 and 1.0 amplitudes at 523.25 Hz become 1.118.

    In equal temperament exact coincidence happens only at unisons and octaves
    -- a 3:2 fifth puts partials at 783.99 and 784.88 Hz, near but not equal --
    so this only bites on chords containing an octave. It bites hard when it
    does, because merging changes every pairwise term that partial takes part
    in, not just the pair that collided.

    `coherent=True` sums amplitudes linearly instead, matching hrep's
    non-default branch."""
    freqs, amps = [], []
    for m in midi_notes:
        f, a = hrep_partials(440.0 * 2 ** ((m - 69) / 12.0), num_harmonics, roll_off)
        freqs.append(f)
        amps.append(a)
    freqs = np.concatenate(freqs)
    amps = np.concatenate(amps)

    keys = np.round(freqs, digits)
    uniq, inverse = np.unique(keys, return_inverse=True)
    if len(uniq) == len(keys):
        order = np.argsort(freqs)
        return freqs[order], amps[order]
    merged = np.zeros(len(uniq))
    if coherent:
        np.add.at(merged, inverse, amps)
    else:
        np.add.at(merged, inverse, amps ** 2)
        merged = np.sqrt(merged)
    return uniq, merged


def run_incon(chords, models=None, cache_path=None):
    """chords: iterable of MIDI-number sequences. Returns {model: np.ndarray}.

    cache_path, if given, is read instead of invoking R when it already exists
    and covers the same chords -- the corpus model in particular is slow, and
    re-running it on every analysis edit is wasteful."""
    models = list(models or ALL_MODELS)
    chords = [list(c) for c in chords]
    keys = [" ".join(str(x) for x in c) for c in chords]

    if cache_path and os.path.exists(cache_path):
        cached = _read_csv(cache_path)
        if cached and [r["chord"] for r in cached] == keys and all(m in cached[0] for m in models):
            return {m: np.array([float(r[m]) for r in cached]) for m in models}

    tmpdir = tempfile.mkdtemp(prefix="incon_")
    in_path = os.path.join(tmpdir, "chords.csv")
    out_path = cache_path or os.path.join(tmpdir, "out.csv")
    with open(in_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["chord"])
        for k in keys:
            w.writerow([k])

    proc = subprocess.run([rscript_path(), "incon_bridge.R", in_path, out_path,
                           ",".join(models)],
                          capture_output=True, text=True)
    if proc.returncode != 0 or not os.path.exists(out_path):
        raise RuntimeError(f"incon_bridge.R failed:\n{proc.stderr[-3000:]}")

    rows = _read_csv(out_path)
    return {m: np.array([float(r[m]) if r[m] not in ("", "NA") else np.nan for r in rows])
            for m in models}


def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


if __name__ == "__main__":
    out = run_incon([[60, 64, 67], [60, 63, 66], [60, 61, 62]])
    print(f"{'model':24s} {'class':13s} {'major':>10s} {'dim':>10s} {'cluster':>10s}")
    for m in ALL_MODELS:
        v = out[m]
        print(f"{m:24s} {MODEL_CLASS[m]:13s} {v[0]:10.4f} {v[1]:10.4f} {v[2]:10.4f}")
