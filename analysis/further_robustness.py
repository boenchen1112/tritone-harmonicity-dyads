"""
Re-analysis R3: new robustness and mechanism checks on the paper's one
surviving timbre-specific claim (under Hutchinson & Knopoff, clarinet spectra
put the minor sixth above the tritone at D3, C4, C5 on two libraries).

  (a) H&K kernel constants a (peak location, in CBW) and b (exponent), listed as
      unswept in the original paper, swept over a in {0.15..0.35}, b in {1,2,3}.
  (b) Bandwidth scale: H&K's CBW = 1.72 f^0.65 replaced by the ERB of Glasberg &
      Moore (1990), ERB = 24.7 (4.37 f/1000 + 1), keeping H&K's kernel. Tests the
      original claim that the effect comes from H&K's critical-bandwidth
      frequency scaling.
  (c) Mechanism: per-pair roughness contributions for the clarinet C4 spectrum
      at 600 and 800 cents, under H&K and Sethares, to show which partial pairs
      decide the ordering in each kernel.
  (d) Vassilakis' amplitude exponent 0.1 replaced by {0.25, 0.5, 1.0} on all
      spectra (sign and magnitude of the gap).
  (e) Stiffness-inharmonicity fit to the Iowa piano ratios:
      f_n / f_1 = n sqrt(1 + B n^2) / sqrt(1 + B).
All spectra are the stored extractions (extracted_partials.json and
clarinet_replication_results.json); nothing is re-extracted.
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import PROJECT_ROOT
from itertools import product
import numpy as np
from scipy.optimize import curve_fit

ROOT = PROJECT_ROOT
sys.path.insert(0, str(ROOT))
from dissonance_model import _pairs, _DSTAR, _S1, _S2, _A1, _A2

OUT = Path(__file__).resolve().parents[1] / "results" / "further_robustness.json"
PH = json.load(open(ROOT / "extracted_partials.json"))
REP = json.load(open(ROOT / "clarinet_replication_results.json"))["registers"]


def hk_general(freqs, amps, a=0.25, b=2.0, cut=1.2, bw="cbw"):
    f1, f2, a1, a2, allamps = _pairs(freqs, amps)
    fm = (f1 + f2) / 2
    width = 1.72 * fm ** 0.65 if bw == "cbw" else 24.7 * (4.37 * fm / 1000 + 1)
    y = np.abs(f1 - f2) / width
    g = ((y / a) * np.exp(1 - y / a)) ** b
    if cut is not None:
        g = np.where(y > cut, 0.0, g)
    return float(np.sum(a1 * a2 * g) / np.sum(allamps ** 2))


def vass_exp(freqs, amps, e=0.1):
    f1, f2, a1, a2, allamps = _pairs(freqs, amps)
    s = _DSTAR / (_S1 * np.minimum(f1, f2) + _S2)
    fd = np.abs(f2 - f1)
    x = np.exp(_A1 * s * fd) - np.exp(_A2 * s * fd)
    return float(np.sum((a1 * a2) ** e * (2 * np.minimum(a1, a2) / (a1 + a2)) ** 3.11 * x)
                 / np.sqrt(np.sum(allamps)))


def dyad(model, ratios, amps, f0, cents, **kw):
    r, a = np.asarray(ratios), np.asarray(amps)
    f = np.concatenate([f0 * r, f0 * 2 ** (cents / 1200) * r])
    return model(f, np.concatenate([a, a]), **kw)


def gap(model, ratios, amps, f0, **kw):
    t = dyad(model, ratios, amps, f0, 600, **kw)
    m = dyad(model, ratios, amps, f0, 800, **kw)
    return 100 * (t - m) / m


def spectra():
    s = {}
    for k, v in PH.items():
        lib = "Iowa" if k == "piano_C4" else "Philharmonia"
        s[f"{lib} {k}"] = (v["ratios"], v["rel_amps"], v["f0_hz"])
    for reg, d in REP.items():
        i = d["iowa"]
        s[f"Iowa clarinet_{reg}"] = (i["ratios"], i["rel_amps"], i["f0_hz"])
    return s


def _hk_pair(f1, f2, a1, a2, norm):
    y = abs(f2 - f1) / (1.72 * ((f1 + f2) / 2) ** 0.65)
    g = 0.0 if y > 1.2 else ((y / .25) * np.exp(1 - y / .25)) ** 2
    return a1 * a2 * g / norm


def _seth_pair(f1, f2, a1, a2, norm):
    lo, d = min(f1, f2), abs(f2 - f1)
    s = _DSTAR / (_S1 * lo + _S2)
    return a1 * a2 * (np.exp(_A1 * s * d) - np.exp(_A2 * s * d)) / norm


def decompose_dyad(ratios, amps, f0, cents, kernel):
    """Split a dyad's total roughness into cross-tone pairs (one partial from each
    note) and within-tone pairs. Normalisers match the pooled model exactly."""
    r, a = np.asarray(ratios), np.asarray(amps)
    fa, fb = f0 * r, f0 * 2 ** (cents / 1200) * r
    if kernel == "hk":
        pf, norm = _hk_pair, 2 * np.sum(a ** 2)
    else:
        pf, norm = _seth_pair, np.sqrt(2 * np.sum(a))
    cross = []
    for i, j in product(range(len(r)), range(len(r))):
        v = pf(fa[i], fb[j], a[i], a[j], norm)
        cross.append({"lower_h": int(round(r[i])), "upper_h": int(round(r[j])),
                      "f_lower": round(float(fa[i]), 1), "f_upper": round(float(fb[j]), 1),
                      "beat_hz": round(float(abs(fb[j] - fa[i])), 1), "contrib": float(v)})
    within = 0.0
    for f in (fa, fb):
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                within += pf(f[i], f[j], a[i], a[j], norm)
    return cross, within


def main():
    S = spectra()
    out = {}
    clar = [k for k in S if "clarinet" in k]

    # (a) a x b sweep
    # wide grid: a in 0.15..0.35 CBW, b in 1..3; narrow grid: +-0.05 CBW, b +-0.5
    A, B = [0.15, 0.20, 0.25, 0.30, 0.35], [1.0, 1.5, 2.0, 2.5, 3.0]
    ab = {k: {f"a={x},b={y:g}": round(gap(hk_general, r, a, f0, a=x, b=y), 2) for x in A for y in B}
          for k, (r, a, f0) in S.items()}
    narrow_keys = [f"a={x},b={y:g}" for x in (0.20, 0.25, 0.30) for y in (1.5, 2.0, 2.5)]
    nar = {k: {kk: v[kk] for kk in narrow_keys} for k, v in ab.items()}
    out["hk_ab_sweep"] = {
        "grid": {"a": A, "b": B}, "narrow_grid": narrow_keys, "gaps": ab,
        "clarinet_negative_everywhere": {k: all(v < 0 for v in ab[k].values()) for k in clar},
        "clarinet_negative_narrow": {k: all(v < 0 for v in nar[k].values()) for k in clar},
        "clarinet_range": {k: [min(ab[k].values()), max(ab[k].values())] for k in clar},
        "clarinet_range_narrow": {k: [min(nar[k].values()), max(nar[k].values())] for k in clar},
        "positive_cells_clarinet": {k: {kk: v for kk, v in ab[k].items() if v >= 0} for k in clar},
        "sign_stable": {k: len({np.sign(v) for v in ab[k].values()}) == 1 for k in S},
        "sign_stable_narrow": {k: len({np.sign(v) for v in nar[k].values()}) == 1 for k in S},
        "range_pp": {k: round(float(np.ptp(list(v.values()))), 2) for k, v in ab.items()},
        "range_pp_narrow": {k: round(float(np.ptp(list(v.values()))), 2) for k, v in nar.items()}}

    # (b) ERB instead of CBW
    out["hk_erb"] = {k: {"cbw_cut1.2": round(gap(hk_general, r, a, f0), 2),
                         "erb_cut1.2": round(gap(hk_general, r, a, f0, bw="erb"), 2),
                         "erb_nocut": round(gap(hk_general, r, a, f0, bw="erb", cut=None), 2)}
                     for k, (r, a, f0) in S.items()}

    # (c) mechanism
    mech = {}
    for key in ["Philharmonia clarinet_C4", "Iowa clarinet_C4", "Philharmonia flute_C4",
                "Philharmonia cello_C4"]:
        r, a, f0 = S[key]
        mech[key] = {}
        for kern in ("hk", "sethares"):
            res = {}
            for c in (600, 800):
                rows, wt = decompose_dyad(r, a, f0, c, kern)
                cross = sum(x["contrib"] for x in rows)
                top = sorted(rows, key=lambda x: -x["contrib"])[:6]
                res[str(c)] = {"cross_tone_total": cross, "within_tone_total": wt,
                               "top_pairs": [{**t, "share_of_cross": t["contrib"] / cross} for t in top]}
            mech[key][kern] = res
    out["mechanism"] = mech

    # (d) Vassilakis exponent
    out["vassilakis_exponent"] = {k: {str(e): round(gap(vass_exp, r, a, f0, e=e), 2)
                                      for e in (0.1, 0.25, 0.5, 1.0)} for k, (r, a, f0) in S.items()}

    # (e) piano inharmonicity
    r = np.asarray(PH["piano_C4"]["ratios"])
    # The 15 retained piano partials are consecutive (spacings all ~1), so the
    # harmonic numbers are 1..15; rounding the ratios would mislabel the top
    # partials, whose stretch exceeds half a harmonic (15.52 -> "16").
    assert np.all(np.abs(np.diff(r) - 1) < 0.2)
    n = np.arange(1, len(r) + 1)
    fn = lambda n, B: n * np.sqrt(1 + B * n ** 2) / np.sqrt(1 + B)
    (Bhat,), cov = curve_fit(fn, n, r, p0=[1e-4])
    out["piano_inharmonicity"] = {"harmonic_numbers": n.tolist(), "ratios": r.round(4).tolist(),
                                  "B": float(Bhat), "B_se": float(np.sqrt(cov[0, 0])),
                                  "max_abs_resid": float(np.max(np.abs(fn(n, Bhat) - r)))}

    OUT.write_text(json.dumps(out, indent=2))
    hs = out["hk_ab_sweep"]
    print("(a) clarinet negative on whole a x b grid:", hs["clarinet_negative_everywhere"])
    print("    ranges:", hs["clarinet_range"])
    print("    narrow:", hs["clarinet_negative_narrow"], hs["clarinet_range_narrow"])
    print("    positive clarinet cells:", hs["positive_cells_clarinet"])
    print("    sign-stable spectra:", sum(hs["sign_stable"].values()), "/", len(S),
          " unstable:", [k for k, v in hs["sign_stable"].items() if not v])
    print("    sign-stable (narrow):", sum(hs["sign_stable_narrow"].values()), "/", len(S),
          " unstable:", [k for k, v in hs["sign_stable_narrow"].items() if not v])
    print("    range median wide/narrow:", np.median(list(hs["range_pp"].values())),
          np.median(list(hs["range_pp_narrow"].values())))
    print("(b) ERB:")
    for k, v in out["hk_erb"].items():
        print(f"    {k:28s} {v}")
    print("(c) mechanism (lower_h, upper_h, beat Hz, share of cross-tone):")
    for key, d in mech.items():
        for kern, res in d.items():
            print(f"   {key} {kern}: cross600={res['600']['cross_tone_total']:.4f} "
                  f"cross800={res['800']['cross_tone_total']:.4f} within600={res['600']['within_tone_total']:.4f} "
                  f"within800={res['800']['within_tone_total']:.4f}")
            for c in ("600", "800"):
                print("      ", c, [(t['lower_h'], t['upper_h'], t['beat_hz'], round(t['share_of_cross'], 2))
                                    for t in res[c]['top_pairs'][:4]])
    print("(d) Vassilakis exponent:")
    for k, v in out["vassilakis_exponent"].items():
        print(f"    {k:28s} {v}")
    print("(e) piano:", {k: v for k, v in out["piano_inharmonicity"].items() if k in ("B", "B_se", "max_abs_resid")})


if __name__ == "__main__":
    main()
