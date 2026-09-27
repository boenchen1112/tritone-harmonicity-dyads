"""
Result 8: the three-component decomposition, using pmcharrison/incon.

Harrison & Pearce (2020, Psychological Review 127(2):216-244) argue simultaneous
consonance derives from three things: interference, periodicity/harmonicity, and
cultural familiarity. Everything in this project up to Result 7 covered the first
two. `har_19_corpus` -- a corpus-derived familiarity model -- is the third, and
until R was installed it was unreachable here. FINDINGS.md called it "the single
largest gap".

The question Result 7 could not answer: Lahdelma, Eerola & Armitage (2022) argue
harmonicity may be inseparable from cultural familiarity on Western listeners'
ratings. With a familiarity predictor in hand that becomes testable rather than
a caveat. Specifically:

  * Does harmonicity survive controlling for familiarity, and vice versa?
  * Is this project's Hutchinson & Knopoff interference term redundant once both
    are present?
  * Does the published composite (har_19_composite) beat anything assembled here?

Orientations are MEASURED, not declared (see incon_bridge.measure_orientations):
17 models with no shared sign convention is the Result 4 polarity bug waiting to
happen at scale, and my hand-written table got 4 of 17 backwards.

Caveat that applies to every number below: incon builds its own idealized
spectra from MIDI and cannot accept measured partials, so nothing here touches
this project's real instrument spectra. This is a chord-structure analysis.
"""
import json
import numpy as np
from scipy.stats import pearsonr, f as fdist

from chord_validation import DATASETS, load_chords
from incon_bridge import ALL_MODELS, MODEL_CLASS, measure_orientations, run_incon

# Two models are excluded for runtime, measured rather than guessed (timings on this
# machine, per 10 chords, extrapolated to the 401 rated chords):
#   gill_09_harmonicity  147 s / 10  -> ~98 min per dataset   <- the real bottleneck
#   wang_13_roughness    Hilbert-Huang decomposition per chord, minutes each
# Everything else totals well under a minute per 10 chords. Both exclusions are
# reported here and in FINDINGS.md rather than left silent: gill_09 is a harmonicity
# model and its absence narrows Result 8's harmonicity evidence to har_18 and milne_13,
# while wang_13 is a fifth interference model where four are already present.
MODELS = [m for m in ALL_MODELS if m not in ("wang_13_roughness", "gill_09_harmonicity")]
EXCLUDED = ["gill_09_harmonicity (~98 min/dataset)", "wang_13_roughness (EMD, minutes/chord)"]

CACHE = {"bowling2018": "reference_data/incon_bowling2018.csv",
         "johnson-laird2012": "reference_data/incon_johnson-laird2012.csv"}

# The three components of Harrison & Pearce's account, one representative each.
INTERFERENCE = "hutch_78_roughness"     # the model this project independently found best
HARMONICITY = "har_18_harmonicity"      # Harrison & Pearce's own harmonicity term
FAMILIARITY = "har_19_corpus"           # the component previously unreachable here


def r2(y, *preds):
    X = np.column_stack([np.ones(len(y))] + [np.asarray(p, float) for p in preds])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    ss_res = float(np.sum((y - X @ beta) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def nested_p(r2_full, r2_red, n, k_full, k_red):
    num = (r2_full - r2_red) / (k_full - k_red)
    den = (1 - r2_full) / (n - k_full - 1)
    if den <= 0:
        return float("nan"), float("nan")
    f = num / den
    return float(f), float(1 - fdist.cdf(f, k_full - k_red, n - k_full - 1))


def main():
    print("Measuring model orientations empirically (not from a hand-written table)...")
    signs = measure_orientations(MODELS)
    flipped = [m for m in MODELS if signs[m] == -1]
    print(f"  {len(flipped)} of {len(MODELS)} models are consonance-up and get negated: "
          f"{', '.join(flipped)}")
    unusable = [m for m in MODELS if signs[m] == 0]
    if unusable:
        print(f"  {len(unusable)} did not separate the reference chords and are EXCLUDED: {unusable}")

    out = {"orientations": {m: int(signs[m]) for m in MODELS}, "excluded_models": EXCLUDED, "datasets": {}}

    for ds in DATASETS:
        chords, y = load_chords(ds)
        n = len(chords)
        raw = run_incon(chords, MODELS, cache_path=CACHE[ds])
        # orient every model dissonance-up so all correlations are comparable
        X = {m: raw[m] * signs[m] for m in MODELS if signs[m] != 0}

        print(f"\n{'=' * 94}")
        print(f"{ds}  (n={n})   {len(X)} incon models, oriented dissonance-up "
          f"(excluded: {', '.join(EXCLUDED)})")
        print(f"{'=' * 94}")
        print(f"  {'model':24s} {'class':13s} {'r':>8s} {'r^2':>7s}")
        ranked = sorted(X, key=lambda m: -abs(pearsonr(X[m], y)[0]))
        model_rs = {}
        for m in ranked:
            r, p = pearsonr(X[m], y)
            model_rs[m] = round(float(r), 3)
            print(f"  {m:24s} {MODEL_CLASS[m]:13s} {r:+8.3f} {r ** 2:7.3f}")

        print(f"\n  --- three-component decomposition "
              f"(interference={INTERFERENCE}, harmonicity={HARMONICITY}, familiarity={FAMILIARITY}) ---")
        I, H, F = X[INTERFERENCE], X[HARMONICITY], X[FAMILIARITY]
        print(f"  pairwise: r(I,H)={pearsonr(I, H)[0]:+.3f}  r(I,F)={pearsonr(I, F)[0]:+.3f}  "
              f"r(H,F)={pearsonr(H, F)[0]:+.3f}")
        full = r2(y, I, H, F)
        combos = {"I": r2(y, I), "H": r2(y, H), "F": r2(y, F),
                  "I+H": r2(y, I, H), "I+F": r2(y, I, F), "H+F": r2(y, H, F),
                  "I+H+F": full}
        for k, v in combos.items():
            print(f"    R2({k:5s}) = {v:.3f}")
        print(f"\n  unique contribution of each, over the other two:")
        uniq = {}
        for name, red in [("interference", r2(y, H, F)), ("harmonicity", r2(y, I, F)),
                          ("familiarity", r2(y, I, H))]:
            f_, p_ = nested_p(full, red, n, 3, 2)
            uniq[name] = {"delta_r2": round(full - red, 4), "F": round(f_, 2),
                          "p": round(p_, 5)}
            verdict = "SIGNIFICANT" if p_ < 0.05 else "not significant -- redundant here"
            print(f"    {name:13s} dR2={full - red:+.4f}  F={f_:7.2f}  p={p_:.5f}   {verdict}")

        pub = pearsonr(X["har_19_composite"], y)[0]
        print(f"\n  published composite (har_19_composite): r={pub:+.3f}  R2={pub ** 2:.3f}")
        print(f"  my three-predictor regression:           R2={full:.3f}")
        print(f"  best single model here:                  {ranked[0]} r={model_rs[ranked[0]]:+.3f}")

        out["datasets"][ds] = {"n": n, "model_r": model_rs, "r2_combinations":
                               {k: round(v, 4) for k, v in combos.items()},
                               "unique_contributions": uniq,
                               "r_har19_composite": round(float(pub), 3),
                               "pairwise": {"I_H": round(float(pearsonr(I, H)[0]), 3),
                                            "I_F": round(float(pearsonr(I, F)[0]), 3),
                                            "H_F": round(float(pearsonr(H, F)[0]), 3)}}

    with open("incon_validation_results.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nSaved incon_validation_results.json")


if __name__ == "__main__":
    main()
