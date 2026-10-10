"""
Derived summaries of results/marjieh_dyads.json that the manuscript quotes:
the specialness null pooled across experiments (inverse-variance weights per
interval), and the fixed two-mechanism curve's weights in standardised form.
Writes results/marjieh_summary.json.
"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import t as tdist

RES = Path(__file__).resolve().parents[1] / "results"
D = json.load(open(RES / "marjieh_dyads.json"))
E = D["experiments"]
US_HARM = ["harm3", "eq5", "no3", "flute", "guitar", "piano"]
US_ALL = [k for k, v in E.items() if v["cohort"] == "US"]
INTERVALS = list(range(1, 15))


def pool_null(exps, key):
    est = np.array([E[e][key]["est"] for e in exps])
    se = np.array([E[e][key]["se"] for e in exps])
    w = 1 / se ** 2
    m = (w * est).sum(0) / w.sum(0)
    s = np.sqrt(1 / w.sum(0))
    order = np.argsort(-m)
    return {"interval": INTERVALS, "est": m.tolist(), "se": s.tolist(),
            "tt_rank": int(np.where(order == INTERVALS.index(6))[0][0]) + 1,
            "ranking": [INTERVALS[i] for i in order]}


def dl(exps, key):
    """DerSimonian-Laird random-effects pooling, with a 95% prediction interval
    for the value in a new experiment of the same kind."""
    y = np.array([E[e][key]["est"] for e in exps])
    v = np.array([E[e][key]["se"] for e in exps]) ** 2
    w = 1 / v
    fe = (w * y).sum() / w.sum()
    Q = float((w * (y - fe) ** 2).sum())
    k = len(y)
    tau2 = max(0.0, (Q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    wr = 1 / (v + tau2)
    m = float((wr * y).sum() / wr.sum())
    se = float(np.sqrt(1 / wr.sum()))
    t = float(tdist.ppf(0.975, k - 2)) if k > 2 else float("nan")
    half = t * np.sqrt(tau2 + se ** 2)
    # Hartung-Knapp-Sidik-Jonkman interval (sensitivity check: DL intervals are too narrow for small k)
    q = float((wr * (y - m) ** 2).sum() / (k - 1)) if k > 1 else float("nan")
    se_hk = float(np.sqrt(q / wr.sum())) if k > 1 else float("nan")
    t1 = float(tdist.ppf(0.975, k - 1)) if k > 1 else float("nan")
    return {"est": m, "se": se, "ci": [m - 1.96 * se, m + 1.96 * se], "tau": float(np.sqrt(tau2)),
            "pi": [m - half, m + half], "hksj_ci": [m - t1 * se_hk, m + t1 * se_hk], "Q": Q, "k": k,
            "I2": max(0.0, (Q - (k - 1)) / Q) if Q > 0 else 0.0, "fe_est": float(fe),
            "fe_se": float(np.sqrt(1 / w.sum())), "experiments": list(exps)}


GROUPS = {"us_harmonic": US_HARM, "us8": US_HARM + ["pure", "bonang"], "us_all": US_ALL}
KEYS = ["obs_gap", "obs_dip"] + [f"{m}_unexpl_{g}" for m in ("HK", "Seth", "H", "HK+H", "HK+H_incon", "revHK+H",
                                                                 "Composite")
                                 for g in ("gap", "dip")]
out = {"null": {}, "random_effects": {g: {k: dl(x, k) for k in KEYS} for g, x in GROUPS.items()}}
# is the tritone's dip larger than the mean dip at the other 13 intervals? (per-experiment bootstrap SEs)
out["tt_minus_mean_other"] = {g: {k: dl(x, k) for k in ("obs_dip_tt_minus_mean_other", "HK_udip_tt_minus_mean_other",
                                                       "HK+H_udip_tt_minus_mean_other",
                                                       "Composite_udip_tt_minus_mean_other")}
                              for g, x in GROUPS.items()}
# does the stretched/compressed profile still have structure? SD of the listener profile over the fit region
prof_sd = {}
for e, v in E.items():
    g = np.asarray(v["profile"]["grid"])
    h = np.asarray(v["profile"]["human"])
    mask = (g >= 0.5) & (g <= 14.75)
    prof_sd[e] = {"sd": float(h[mask].std()), "range": float(h[mask].max() - h[mask].min()),
                  "HK+H_r2fit": v["HK+H_r2fit"]["est"], "HK_r2fit": v["HK_r2fit"]["est"]}
out["profile_structure"] = prof_sd


def dl_items(items):
    """dl() on a list of {est, se} dicts."""
    global E
    saved = E
    E = {str(i): {"x": r} for i, r in enumerate(items)}
    try:
        return dl(list(E), "x")
    finally:
        E = saved


# musicians minus non-musicians, random effects over the eight unstretched US experiments
MUS8 = [e for e in US_HARM + ["pure", "bonang"] if "musicianship" in E[e]]
out["musicianship_re"] = {}
for k in ("diff_obs_gap", "diff_obs_dip", "diff_HK+H_unexpl_gap", "diff_HK+H_unexpl_dip"):
    items = [{"est": E[e]["musicianship"][k]["est"], "se": E[e]["musicianship"][k]["se"]} for e in MUS8]
    out["musicianship_re"][k] = dl_items(items)
for grp in ("mus", "nonmus"):
    for k in ("obs_gap", "obs_dip", "HK+H_unexpl_gap", "HK+H_unexpl_dip"):
        items = [{"est": E[e]["musicianship"][grp][k]["est"], "se": E[e]["musicianship"][grp][k]["se"]} for e in MUS8]
        out["musicianship_re"][f"{grp}_{k}"] = dl_items(items)

# stretched/compressed: gap and dip at the stretched scale's own tritone, fifth, fourth and minor sixth
# (interval k of a scale with octave ratio r lies at k*log2(r) equal-tempered semitones). Point estimates
# from the stored 0.05-semitone profiles; calibration excludes +/-0.6 around the stretched TT and m6.
def at(g, v, x):
    return float(np.interp(x, g, v))


stretch = {}
for e, ratio in (("str3", 2.1), ("comp3", 1.9), ("harm3", 2.0), ("str3_kr", 2.1), ("comp3_kr", 1.9), ("harm3_kr", 2.0)):
    p = E[e]["profile"]
    g = np.asarray(p["grid"])
    h = np.asarray(p["human"])
    u = np.log2(ratio)
    tt, m6, p4, p5 = 6 * u, 8 * u, 5 * u, 7 * u
    mask = (g >= 0.5) & (g <= 14.75) & (np.abs(g - tt) >= 0.6) & (np.abs(g - m6) >= 0.6)
    A = np.column_stack([np.ones(len(g)), np.asarray(p["HK"]), np.asarray(p["H"])])
    b, *_ = np.linalg.lstsq(A[mask], h[mask], rcond=None)
    pred = A @ b
    gap = at(g, h, m6) - at(g, h, tt)
    dip = (at(g, h, p4) + at(g, h, p5)) / 2 - at(g, h, tt)
    pgap = at(g, pred, m6) - at(g, pred, tt)
    pdip = (at(g, pred, p4) + at(g, pred, p5)) / 2 - at(g, pred, tt)
    stretch[e] = {"tt_pos": tt, "obs_gap": gap, "obs_dip": dip, "HK+H_unexpl_gap": gap - pgap,
                  "HK+H_unexpl_dip": dip - pdip}
out["stretched_scale"] = stretch
for grp, exps in GROUPS.items():
    out["null"][grp] = {k: pool_null(exps, k) for k in
                        ("obs_dip_null", "HK_udip_null", "HK+H_udip_null", "Composite_udip_null")}
# pooled two-mechanism weights, re-expressed per SD of each model's profile in the harmonic experiment
lo = D["lodo"]["HK+H"]
p = E["harm3"]["profile"]
out["curve"] = {"beta_HK": lo["pooled_beta"][0], "beta_H": lo["pooled_beta"][1],
                "sd_HK_harm3": float(np.std(p["HK"])), "sd_H_harm3": float(np.std(p["H"])),
                "lambda_std": float(lo["pooled_beta"][1] * np.std(p["H"]) /
                                    (-lo["pooled_beta"][0] * np.std(p["HK"])))}
json.dump(out, open(RES / "marjieh_summary.json", "w"), indent=1)
for grp, v in out["null"].items():
    for k, n in v.items():
        print(grp, k, "TT rank", n["tt_rank"], "top:", n["ranking"][:4],
              " ".join(f"{i}:{e:+.2f}" for i, e in zip(n["interval"], n["est"])))
print(out["curve"])
for g, v in out["random_effects"].items():
    for k in ("obs_gap", "HK_unexpl_gap", "HK+H_unexpl_gap", "Composite_unexpl_gap", "obs_dip", "HK+H_unexpl_dip"):
        r = v[k]
        print(f"RE {g:12s} {k:22s} {r['est']:+.3f} [{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}] PI[{r['pi'][0]:+.3f},"
              f"{r['pi'][1]:+.3f}] tau={r['tau']:.3f} I2={r['I2']:.2f} FE={r['fe_est']:+.3f}")
for g, v in out["tt_minus_mean_other"].items():
    print(g, {k: (round(r["est"], 3), [round(c, 3) for c in r["ci"]]) for k, r in v.items()})
print({e: round(v["sd"], 3) for e, v in prof_sd.items()})
for k, r in out["musicianship_re"].items():
    print("MUS RE", k, round(r["est"], 3), [round(c, 3) for c in r["ci"]], "I2", round(r["I2"], 2))
for e, r in stretch.items():
    print("STRETCH", e, {k: round(v, 3) for k, v in r.items()})
for g in ("us_harmonic",):
    for k in ("H_unexpl_dip", "HK+H_incon_unexpl_gap", "HK+H_incon_unexpl_dip", "Composite_unexpl_dip"):
        r = out["random_effects"][g][k]
        print("RE", g, k, round(r["est"], 3), [round(c, 3) for c in r["ci"]])
