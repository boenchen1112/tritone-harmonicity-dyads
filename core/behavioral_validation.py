"""
Validate model dissonance curves against real human consonance ratings:
Bowling et al. (2018), 12 dyads spanning 1-12 semitones, via DCMLab/consonance
(github.com/DCMLab/consonance/blob/main/data/bowling2018_consonance.tsv).

Caveat stated up front, not buried: Bowling's stimuli were piano tones. None
of the instruments analyzed here are piano (Philharmonia's mirror doesn't
include one). This checks whether each timbre's *modeled* dissonance curve
correlates with *human* consonance judgments made on a different, unmatched
timbre -- a real limitation, not a controlled test of any one instrument.
"""
import csv
import json
import numpy as np
from scipy.stats import pearsonr, spearmanr, norm


def fisher_ci(r, n, alpha=0.05):
    """95% CI on a Pearson r via Fisher z-transform. n=12 here, so this is a
    small-sample approximation, not exact -- but it's what shows whether two
    r values are actually distinguishable rather than just rank-ordered."""
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    zcrit = norm.ppf(1 - alpha / 2)
    lo, hi = np.tanh(z - zcrit * se), np.tanh(z + zcrit * se)
    return lo, hi

from dissonance_model import (
    interval_sweep, sethares_dissonance, sethares_min_dissonance,
    vassilakis_dissonance, hutch_knopoff_dissonance,
)

MODELS = [
    ("Sethares", sethares_dissonance),
    ("Sethares-min", sethares_min_dissonance),
    ("Vassilakis", vassilakis_dissonance),
    ("Hutch-Knopoff", hutch_knopoff_dissonance),
]


def make_real_partials_fn(ratios, rel_amps):
    ratios = np.asarray(ratios, dtype=float)
    rel_amps = np.asarray(rel_amps, dtype=float)

    def fn(f0, **kwargs):
        return f0 * ratios, rel_amps.copy()

    return fn


def load_bowling_dyads(path="behavioral_data/bowling2018.tsv"):
    rows = list(csv.DictReader(open(path), delimiter="\t"))
    dyads = [r for r in rows if len(r["pitches"].split(",")) == 2]
    out = {}
    for r in dyads:
        p1, p2 = map(int, r["pitches"].split(","))
        out[abs(p2 - p1)] = float(r["rating"])
    return out  # semitone -> consonance rating (higher = more consonant)


def model_curve_at_semitones(root_f0, semitones, model_fn, ratios=None, rel_amps=None):
    cents = np.array([s * 100.0 for s in semitones])
    if ratios is None:
        _, diss = interval_sweep(f0=root_f0, cents=cents, model=model_fn, n_partials=7, rolloff=0.9)
    else:
        pfn = make_real_partials_fn(ratios, rel_amps)
        _, diss = interval_sweep(f0=root_f0, cents=cents, partials_fn=pfn, model=model_fn)
    return diss


def main():
    bowling = load_bowling_dyads()
    semitones = sorted(bowling)
    ratings = np.array([bowling[s] for s in semitones])
    print(f"Bowling (2018) dyad ratings loaded: {len(semitones)} intervals (1-12 semitones)\n")

    with open("extracted_partials.json") as f:
        extracted = json.load(f)
    c4_labels = ["cello_C4", "clarinet_C4", "oboe_C4", "flute_C4",
                 "bassoon_C4", "saxophone_C4", "french-horn_C4", "piano_C4"]

    n = len(semitones)
    results = []

    def run(label, f0, ratios, amps, model_name, model_fn):
        diss = model_curve_at_semitones(f0, semitones, model_fn, ratios, amps)
        r, p = pearsonr(diss, -ratings)
        rho, ps = spearmanr(diss, -ratings)
        lo, hi = fisher_ci(r, n)
        print(f"{label:20s} {model_name:10s} {r:7.3f} [{lo:5.2f},{hi:5.2f}] {p:7.3f} {rho:9.3f} {ps:8.3f}")
        results.append({"timbre": label, "model": model_name,
                         "pearson_r": round(r, 3), "pearson_ci95": [round(lo, 3), round(hi, 3)],
                         "pearson_p": round(p, 3),
                         "spearman_rho": round(rho, 3), "spearman_p": round(ps, 3)})

    print(f"{'timbre':20s} {'model':10s} {'r':>7s} {'95% CI':>13s} {'p':>7s} {'rho':>9s} {'p':>8s}")
    for model_name, model_fn in MODELS:
        run("idealized", 261.63, None, None, model_name, model_fn)  # nominal C4, nothing to anchor to
    for label in c4_labels:
        data = extracted[label]
        for model_name, model_fn in MODELS:
            run(data["instrument"], data["f0_hz"], data["ratios"], data["rel_amps"], model_name, model_fn)

    with open("behavioral_validation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved behavioral_validation_results.json")
    print("\n(r/rho: model dissonance vs NEGATIVE human consonance rating; positive = correct direction.")
    print(" 95% CI via Fisher z, n=12 -- a small-sample approximation. Widely overlapping CIs mean the")
    print(" corresponding r values are NOT distinguishable from each other at this sample size, whatever")
    print(" their point-estimate rank order looks like. No correction applied for the 16 comparisons run.)")


if __name__ == "__main__":
    main()
