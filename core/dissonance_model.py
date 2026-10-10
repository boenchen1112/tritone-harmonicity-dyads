"""
Sensory-dissonance / roughness models, all taking the same input: one pooled
list of partial frequencies and amplitudes, with no assumption about how many
tones produced them.

  sethares_dissonance      Sethares (1993), product amplitude weighting
  sethares_min_dissonance  Sethares (2005) / Weisser & Lartillot (2013) variant,
                           min amplitude weighting
  vassilakis_dissonance    Vassilakis (2001), amplitude-fluctuation weighting
  hutch_knopoff_dissonance Hutchinson & Knopoff (1978), critical-bandwidth model

Sethares formula validated against the reference implementation at
github.com/SebastianJiroSchlecht/dissonanceSethares (dissonanceMeasureFromPartials.m):
constants, product amplitude weighting (a1*a2), all-pairs combination, and
sum(D)/sqrt(sum(amp)) normalization are matched exactly to that source.

Vassilakis (2001) keeps Sethares' frequency-difference term but replaces the
amplitude weighting with the amplitude-fluctuation term from Vassilakis' own
paper: (a1*a2)^0.1 * (2*min(a1,a2)/(a1+a2))^3.11.

Cross-checked against pmcharrison/dycon (the R package underlying Harrison &
Pearce 2020), R/sethares.R and R/vassilakis.R. Three deliberate divergences,
all constant multipliers that cannot change an ordering or a correlation:
dycon's Vassilakis carries a leading 0.5 that this one omits, uses -3.5 where
this one uses Sethares' published -3.51, and normalizes by nothing where this
one applies Sethares' sqrt(sum(amp)) for comparability across the two models.
"""
import numpy as np
from itertools import combinations

# Sethares (1993) constants
_DSTAR, _S1, _S2, _A1, _A2 = 0.24, 0.0207, 18.96, -3.51, -5.75


def _pairs(freqs, amps):
    freqs = np.asarray(freqs, dtype=float)
    amps = np.asarray(amps, dtype=float)
    order = np.argsort(freqs)
    freqs, amps = freqs[order], amps[order]
    n = len(freqs)
    idx = np.array(list(combinations(range(n), 2)))
    f1, f2 = freqs[idx[:, 0]], freqs[idx[:, 1]]
    a1, a2 = amps[idx[:, 0]], amps[idx[:, 1]]
    return f1, f2, a1, a2, amps


def sethares_dissonance(freqs, amps):
    """Sethares (1993) dissonance for a combined set of partials (both tones pooled)."""
    f1, f2, a1, a2, amps = _pairs(freqs, amps)
    fmin = np.minimum(f1, f2)
    fdif = np.abs(f2 - f1)
    s = _DSTAR / (_S1 * fmin + _S2)
    fdiss = np.exp(_A1 * s * fdif) - np.exp(_A2 * s * fdif)
    d = a1 * a2 * fdiss
    return float(np.sum(d) / np.sqrt(np.sum(amps)))


def vassilakis_dissonance(freqs, amps):
    """Vassilakis (2001) roughness for a combined set of partials (both tones pooled)."""
    f1, f2, a1, a2, amps = _pairs(freqs, amps)
    fmin = np.minimum(f1, f2)
    fdif = np.abs(f2 - f1)
    s = _DSTAR / (_S1 * fmin + _S2)
    fdiss = np.exp(_A1 * s * fdif) - np.exp(_A2 * s * fdif)
    amp_term = (a1 * a2) ** 0.1 * (2 * np.minimum(a1, a2) / (a1 + a2)) ** 3.11
    d = amp_term * fdiss
    return float(np.sum(d) / np.sqrt(np.sum(amps)))


def sethares_min_dissonance(freqs, amps):
    """Sethares (1993) frequency term with min(a1,a2) amplitude weighting instead
    of the product. This is the variant Sethares uses in *Tuning, Timbre,
    Spectrum, Scale* (2nd ed., 2005) and that Weisser & Lartillot (2013) adopt;
    it is also pmcharrison/dycon's default (roughness_seth(min_amplitude=TRUE)).
    Included because the product-vs-min choice is a free modelling decision that
    nothing in the data settles, and this project's headline claim is about how
    much model choice moves the tritone/m6 ordering."""
    f1, f2, a1, a2, amps = _pairs(freqs, amps)
    fmin = np.minimum(f1, f2)
    fdif = np.abs(f2 - f1)
    s = _DSTAR / (_S1 * fmin + _S2)
    fdiss = np.exp(_A1 * s * fdif) - np.exp(_A2 * s * fdif)
    d = np.minimum(a1, a2) * fdiss
    return float(np.sum(d) / np.sqrt(np.sum(amps)))


# Hutchinson & Knopoff (1978) constants, as parameterized by Mashinter (2006)
# and implemented in pmcharrison/dycon R/hutch.R (verified against that source).
_HK_A, _HK_B, _HK_CBW_CUTOFF = 0.25, 2.0, 1.2


def hutch_knopoff_dissonance(freqs, amps, cbw_cut_off=_HK_CBW_CUTOFF):
    """Hutchinson & Knopoff (1978), "The acoustic component of Western consonance."

    Structurally different from Sethares/Vassilakis in three ways that matter:
    the frequency difference is expressed in critical bandwidths rather than Hz,
    the roughness kernel is a polynomial-times-exponential rather than a
    difference of two exponentials, and contributions are hard-cut to zero
    beyond 1.2 critical bandwidths instead of decaying smoothly. It is the
    interference component Harrison & Pearce (2020) chose for their composite
    model, which is why it is the natural third model here.

        cbw(f1,f2) = 1.72 * ((f1+f2)/2)^0.65
        y          = |f1-f2| / cbw
        g(y)       = ((y/a) * exp(1 - y/a))^b,  a=0.25, b=2;  g=0 for y > 1.2
        roughness  = sum_{i<j} a_i a_j g_ij  /  sum_i a_i^2

    The sum(a^2) denominator is H&K's own normalization, so unlike the other
    models here this one is not rescaled to Sethares' convention -- it is
    reported as its published form. Curves are normalized to their own max
    before any cross-model comparison anyway.

    cbw_cut_off is exposed because dycon's own documentation says 1.2 exists
    "for replicating Mashinter's results" rather than on physical grounds, and
    this project's one register-stable finding (Result 2, clarinet) depends on
    it. Pass None to disable the cutoff and let g decay smoothly."""
    f1, f2, a1, a2, amps = _pairs(freqs, amps)
    cbw = 1.72 * ((f1 + f2) / 2.0) ** 0.65
    y = np.abs(f1 - f2) / cbw
    g = ((y / _HK_A) * np.exp(1.0 - y / _HK_A)) ** _HK_B
    if cbw_cut_off is not None:
        g = np.where(y > cbw_cut_off, 0.0, g)
    denom = np.sum(amps ** 2)
    return float(np.sum(a1 * a2 * g) / denom) if denom > 0 else 0.0


def harmonic_partials(f0, n_partials=7, rolloff=0.9):
    """Idealized harmonic timbre: partials at integer multiples of f0, geometric amplitude rolloff."""
    n = np.arange(1, n_partials + 1)
    return f0 * n, rolloff ** (n - 1)


def pool_partials(per_tone, merge=True, digits=6):
    """Combine several tones' (freqs, amps) into one pooled spectrum.

    When two tones put partials at the SAME frequency, they are one partial in
    the air, not two. hrep::sparse_fr_spectrum -- the spectrum builder behind
    Harrison & Pearce's `incon`, and therefore behind the published values these
    models are usually compared against -- merges them and combines their
    amplitudes in QUADRATURE, sqrt(a1^2 + a2^2), which is incoherent
    (random-phase) power summation. Concatenating instead leaves a zero-Hz pair
    contributing no roughness while every other pair sees two half-amplitude
    partials rather than one larger one, which changes the result.

    In equal temperament exact coincidence only happens at unisons and octaves
    (a 3:2 fifth lands partials at 783.99 and 784.88 Hz -- near, not equal), so
    this is inert for most chords and inert for MEASURED instrument spectra,
    where inharmonicity means nothing coincides exactly. It matters only for
    idealized harmonic partials in chords containing an octave. Verified: with
    merging on, hutch_knopoff_dissonance reproduces incon's hutch_78_roughness
    to r = 1.000000 over all 401 rated chords; with it off, r = 0.992.

    merge=False recovers the plain concatenation, for the robustness check."""
    freqs = np.concatenate([f for f, _ in per_tone])
    amps = np.concatenate([a for _, a in per_tone])
    if not merge:
        return freqs, amps
    keys = np.round(freqs, digits)
    uniq, inverse = np.unique(keys, return_inverse=True)
    if len(uniq) == len(keys):
        order = np.argsort(freqs)
        return freqs[order], amps[order]
    merged = np.zeros(len(uniq))
    np.add.at(merged, inverse, amps ** 2)
    return uniq, np.sqrt(merged)


def dyad_dissonance(f0_a, f0_b, partials_fn=harmonic_partials, model=sethares_dissonance,
                    merge=True, **kwargs):
    """Dissonance of a two-tone dyad, each tone expanded into partials via partials_fn."""
    freqs, amps = pool_partials([partials_fn(f0_a, **kwargs), partials_fn(f0_b, **kwargs)],
                                merge=merge)
    return model(freqs, amps)


def interval_sweep(f0=261.63, cents=None, partials_fn=harmonic_partials,
                    model=sethares_dissonance, **kwargs):
    """Dissonance curve for a fixed root f0 against a moving tone from unison (0c) to octave (1200c)."""
    if cents is None:
        cents = np.linspace(0, 1200, 241)  # 5-cent resolution
    diss = np.array([
        dyad_dissonance(f0, f0 * 2 ** (c / 1200.0), partials_fn=partials_fn, model=model, **kwargs)
        for c in cents
    ])
    return cents, diss


if __name__ == "__main__":
    f0 = 261.63  # C4
    for name, model in [("Sethares", sethares_dissonance), ("Vassilakis", vassilakis_dissonance)]:
        cents, diss = interval_sweep(f0=f0, model=model, n_partials=7, rolloff=0.9)
        tritone = diss[np.argmin(np.abs(cents - 600))]
        m6 = diss[np.argmin(np.abs(cents - 800))]
        print(f"{name}: tritone={tritone:.4f}  m6={m6:.4f}  tritone>m6? {tritone > m6}  "
              f"gap={100*(tritone-m6)/m6:.1f}%")
