"""
Three checks that FINDINGS.md's prose depends on but its numeric audit does not
cover, because each asks whether a number MEANS what the text claims rather than
whether it was transcribed correctly.

  A. Is Hutchinson & Knopoff's advantage on the pooled 298-chord Bowling set
     real, or an artifact of the two models normalizing differently as chord
     size grows? Sethares divides by sqrt(sum(a)) and H&K by sum(a^2), so their
     outputs scale differently with the number of notes. Pooling N=2,3,4 is
     exactly where that bites. Fix: z-score model output and human ratings
     WITHIN each size bin, then pool.

  B. Does Result 2's one register-stable finding (clarinet flips under H&K at
     all three registers) survive relaxing H&K's hard 1.2-critical-bandwidth
     cutoff? dycon's own docs say 1.2 exists "for replicating Mashinter's
     results", not on physical grounds, and clarinet's suppressed even harmonics
     make its pair-spacing distribution unusual -- so the cutoff may be doing
     the work rather than the model structure.

  C. Do the aggregate claims in the write-up ("best Spearman rho for 7 of 9
     timbres", "best Pearson r on every subset", "Vassilakis uniformly weakest")
     actually hold? These are counts over the JSON that no per-cell audit checks.
"""
import json
import numpy as np
from scipy.stats import pearsonr

from dissonance_model import interval_sweep, hutch_knopoff_dissonance
from chord_validation import (
    DATASETS, MODELS, load_chords, chord_dissonance, make_real_partials_fn,
)
from dissonance_model import harmonic_partials


def check_a_pooling_artifact():
    print("=" * 78)
    print("A. Is H&K's pooled-set advantage a normalization artifact?")
    print("=" * 78)
    chords, human = load_chords("bowling2018")
    sizes = sorted({len(c) for c in chords})
    out = {}

    for model_name, model_fn in MODELS:
        raw = np.array([chord_dissonance(c, harmonic_partials, model_fn,
                                          n_partials=7, rolloff=0.9) for c in chords])
        r_pooled, _ = pearsonr(raw, human)

        # z-score both model output and ratings within each chord-size bin,
        # which removes any between-size offset/scale difference -- including
        # whatever the differing denominators introduce.
        xz, yz = np.empty(len(chords)), np.empty(len(chords))
        for k in sizes:
            idx = [i for i, c in enumerate(chords) if len(c) == k]
            xs, ys = raw[idx], human[idx]
            xz[idx] = (xs - xs.mean()) / xs.std()
            yz[idx] = (ys - ys.mean()) / ys.std()
        r_within, _ = pearsonr(xz, yz)

        per_size = {k: pearsonr(raw[[i for i, c in enumerate(chords) if len(c) == k]],
                                human[[i for i, c in enumerate(chords) if len(c) == k]])[0]
                    for k in sizes}
        out[model_name] = {"pooled_raw": r_pooled, "pooled_within_size_z": r_within,
                           "per_size": per_size}
        print(f"  {model_name:14s} pooled(raw) r={r_pooled:+.3f}   "
              f"pooled(within-size z) r={r_within:+.3f}   "
              f"per size: " + "  ".join(f"{k}n={v:+.3f}" for k, v in per_size.items()))

    gap_raw = out["Hutch-Knopoff"]["pooled_raw"] - out["Sethares"]["pooled_raw"]
    gap_z = out["Hutch-Knopoff"]["pooled_within_size_z"] - out["Sethares"]["pooled_within_size_z"]
    gap_sub = np.mean([out["Hutch-Knopoff"]["per_size"][k] - out["Sethares"]["per_size"][k]
                       for k in sizes])
    print(f"\n  H&K minus Sethares:  pooled raw {gap_raw:+.3f} | "
          f"pooled within-size-z {gap_z:+.3f} | mean of per-size gaps {gap_sub:+.3f}")
    verdict = ("ARTIFACT: the pooled margin is inflated by mixing chord sizes; report the "
               "within-subset margin instead"
               if gap_raw - gap_z > 0.05 else
               "REAL: the margin survives removing all between-size variance")
    print(f"  -> {verdict}")
    return out, {"gap_pooled_raw": gap_raw, "gap_within_size_z": gap_z,
                 "gap_mean_per_size": gap_sub, "verdict": verdict}


def check_b_cutoff_sensitivity(extracted):
    print()
    print("=" * 78)
    print("B. Does the clarinet flip survive relaxing H&K's 1.2 CBW cutoff?")
    print("=" * 78)
    rows = {}
    keys = [("clarinet", ["clarinet_D3", "clarinet_C4", "clarinet_C5"]),
            ("cello", ["cello_C3", "cello_C4", "cello_C5"]),
            ("oboe", ["oboe_As3", "oboe_C4", "oboe_C5"])]
    cutoffs = [1.2, 1.5, 2.0, None]
    print(f"  {'sample':16s} " + " ".join(f"{'cut=' + str(c):>12s}" for c in cutoffs))
    for instrument, sample_keys in keys:
        for key in sample_keys:
            data = extracted[key]
            pfn = make_real_partials_fn(data["ratios"], data["rel_amps"])
            cells, gaps = [], {}
            for c in cutoffs:
                def model(f, a, _c=c):
                    return hutch_knopoff_dissonance(f, a, cbw_cut_off=_c)
                cents, diss = interval_sweep(f0=data["f0_hz"], partials_fn=pfn, model=model)
                tri = diss[np.argmin(np.abs(cents - 600))]
                m6 = diss[np.argmin(np.abs(cents - 800))]
                gap = 100 * (tri - m6) / m6
                gaps[str(c)] = round(float(gap), 1)
                cells.append(f"{gap:+11.1f}%")
            rows[key] = gaps
            print(f"  {key:16s} " + " ".join(cells))

    clarinet_keys = ["clarinet_D3", "clarinet_C4", "clarinet_C5"]
    survives = {str(c): all(rows[k][str(c)] < 0 for k in clarinet_keys) for c in cutoffs}
    print(f"\n  Clarinet flips at ALL THREE registers?  " +
          "  ".join(f"cut={c}: {'yes' if survives[str(c)] else 'NO'}" for c in cutoffs))
    n_ok = sum(survives.values())
    verdict = ("ROBUST: the flip does not depend on the cutoff constant"
               if n_ok == len(cutoffs) else
               f"CUTOFF-DEPENDENT: holds for {n_ok}/{len(cutoffs)} cutoff settings -- the claim "
               f"is about a hardcoded constant, not about model structure")
    print(f"  -> {verdict}")
    return rows, {"clarinet_all_three_flip": survives, "verdict": verdict}


def check_c_aggregate_claims():
    print()
    print("=" * 78)
    print("C. Do the write-up's aggregate claims hold?")
    print("=" * 78)
    out = {}

    bv = json.load(open("behavioral_validation_results.json"))
    timbres = []
    for r in bv:
        if r["timbre"] not in timbres:
            timbres.append(r["timbre"])
    wins = [t for t in timbres
            if max((x for x in bv if x["timbre"] == t),
                   key=lambda x: x["spearman_rho"])["model"] == "Hutch-Knopoff"]
    print(f"  'H&K best Spearman rho for 7 of 9 timbres': H&K wins {len(wins)}/{len(timbres)}"
          f"  {'OK' if len(wins) == 7 else 'MISMATCH'}")
    print(f"     losers: {[t for t in timbres if t not in wins]}")
    out["hk_spearman_wins"] = [len(wins), len(timbres)]

    cv = json.load(open("chord_validation_results.json"))
    subset_wins, subset_total, losses = 0, 0, []
    for ds, blk in cv.items():
        keys = {(r["subset"], r["variant"], r["timbre"]) for r in blk["results"]}
        for k in keys:
            cands = [r for r in blk["results"]
                     if (r["subset"], r["variant"], r["timbre"]) == k]
            subset_total += 1
            best = max(cands, key=lambda r: r["pearson_r"])
            if best["model"] == "Hutch-Knopoff":
                subset_wins += 1
            else:
                losses.append((ds, *k, best["model"], best["pearson_r"]))
    print(f"  'H&K best Pearson r on every subset/timbre/variant': "
          f"{subset_wins}/{subset_total}  {'OK' if subset_wins == subset_total else 'MISMATCH'}")
    for l in losses[:10]:
        print(f"     lost: {l}")
    out["hk_pearson_wins"] = [subset_wins, subset_total]

    vass_worst, vass_total = 0, 0
    for ds, blk in cv.items():
        keys = {(r["subset"], r["variant"], r["timbre"]) for r in blk["results"]}
        for k in keys:
            cands = [r for r in blk["results"]
                     if (r["subset"], r["variant"], r["timbre"]) == k]
            vass_total += 1
            if min(cands, key=lambda r: r["pearson_r"])["model"] == "Vassilakis":
                vass_worst += 1
    print(f"  'Vassilakis uniformly weakest': weakest in {vass_worst}/{vass_total} "
          f"chord comparisons  {'OK' if vass_worst == vass_total else 'MISMATCH -- soften'}")
    out["vassilakis_weakest"] = [vass_worst, vass_total]
    return out


def main():
    with open("extracted_partials.json") as f:
        extracted = json.load(f)
    a_detail, a_summary = check_a_pooling_artifact()
    b_detail, b_summary = check_b_cutoff_sensitivity(extracted)
    c = check_c_aggregate_claims()
    with open("robustness_checks.json", "w") as f:
        json.dump({"a_pooling": {"per_model": a_detail, "summary": a_summary},
                   "b_cutoff_sensitivity": {"gaps_pct": b_detail, "summary": b_summary},
                   "c_aggregate_claims": c}, f, indent=2, default=str)
    print("\nSaved robustness_checks.json")


if __name__ == "__main__":
    main()
