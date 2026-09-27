"""Build every manuscript figure (vector PDF) from the study's JSON outputs and
stored spectra. No number in a figure is typed by hand."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import PROJECT_ROOT
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = PROJECT_ROOT
MS = Path(__file__).resolve().parents[1]
RES, FIG = MS / "results", MS / "figures"
sys.path.insert(0, str(ROOT))
from dissonance_model import (hutch_knopoff_dissonance, sethares_dissonance, interval_sweep,
                              harmonic_partials)

plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6})
MODELS = ["Sethares", "Sethares-min", "Vassilakis", "Hutch-Knopoff"]
MLAB = {"Sethares": "Sethares (product)", "Sethares-min": "Sethares (min)",
        "Vassilakis": "Vassilakis", "Hutch-Knopoff": "Hutchinson–Knopoff"}
MCOL = {"Sethares": "#4C72B0", "Sethares-min": "#8FB0D8", "Vassilakis": "#999999",
        "Hutch-Knopoff": "#C44E52"}
W1, W2 = 3.4, 7.0   # single / double column width, inches


def fig_curves():
    ph = json.load(open(ROOT / "extracted_partials.json"))["clarinet_C4"]
    io = json.load(open(ROOT / "clarinet_replication_results.json"))["registers"]["C4"]["iowa"]
    specs = [("Clarinet C4, Philharmonia", ph["ratios"], ph["rel_amps"], ph["f0_hz"], "#C44E52", "-"),
             ("Clarinet C4, Iowa MIS", io["ratios"], io["rel_amps"], io["f0_hz"], "#DD8452", "--"),
             ("Idealised harmonic (7 partials)", None, None, 261.63, "#555555", ":")]
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.7), sharey=True)
    for ax, (mname, fn) in zip(axes, [("Hutchinson–Knopoff", hutch_knopoff_dissonance),
                                      ("Sethares (product)", sethares_dissonance)]):
        for lab, r, a, f0, col, ls in specs:
            if r is None:
                c, d = interval_sweep(f0=f0, model=fn, n_partials=7, rolloff=0.9)
            else:
                rr, aa = np.asarray(r), np.asarray(a)
                c, d = interval_sweep(f0=f0, model=fn, partials_fn=lambda f, **k: (f * rr, aa.copy()))
            ax.plot(c, d / d.max(), color=col, ls=ls, lw=1.1, label=lab)
        for x, t in ((600, "TT"), (800, "m6")):
            ax.axvline(x, color="#bbbbbb", lw=0.6, zorder=0)
            ax.text(x, 1.02, t, ha="center", va="bottom", fontsize=7.5)
        ax.set_title(mname, fontsize=9, pad=12)
        ax.set_xlim(0, 1200)
        ax.set_xticks(range(0, 1201, 200))
        ax.set_xlabel("Interval above root (cents)")
    axes[0].set_ylabel("Dissonance (normalised to curve max)")
    axes[0].set_ylim(0, 1.08)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=7.5, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIG / "fig3_curves.pdf")
    plt.close(fig)


def fig_gaps():
    rs = json.load(open(ROOT / "results_summary.json"))["instrument_comparison"]
    timbres = list(dict.fromkeys(r["timbre"] for r in rs))
    names = {t: ("Idealised" if t.startswith("idealized") else t.replace("-", " ").capitalize())
             for t in timbres}
    fig, ax = plt.subplots(figsize=(W1, 3.3))
    y = np.arange(len(timbres))[::-1]
    off = {m: o for m, o in zip(MODELS, (-0.27, -0.09, 0.09, 0.27))}
    for m in MODELS:
        g = [next(r["gap_pct"] for r in rs if r["timbre"] == t and r["model"] == m) for t in timbres]
        ax.scatter(np.clip(g, -60, 80), y + off[m], s=14, color=MCOL[m], label=MLAB[m], zorder=3,
                   edgecolor="white", linewidth=0.4)
        for gi, yi in zip(g, y + off[m]):
            if gi > 80:
                ax.annotate(f"{gi:+.0f}%", (80, yi), xytext=(-2, 0), textcoords="offset points",
                            ha="right", va="center", fontsize=6)
    ax.axvline(0, color="black", lw=0.6)
    ax.axvspan(-60, 0, color="#f3e3e3", zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels([names[t] for t in timbres])
    ax.set_xlim(-60, 80)
    ax.set_xlabel("Tritone vs minor-sixth gap (%)\n(shaded: minor sixth rougher)")
    ax.legend(frameon=False, fontsize=6.5, loc="lower center", bbox_to_anchor=(0.45, 1.0),
              ncol=2, handletextpad=0.2, columnspacing=0.8)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_gaps.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_sensitivity():
    u = json.load(open(RES / "unified_sensitivity.json"))["stable"]
    order = ["instrument (reference)", "H&K constants a,b, wide (H&K only)", "kernel",
             "H&K constants a,b, narrow (H&K only)", "idealised constants", "recording",
             "extraction constants", "audio codec"]
    lab = {"instrument (reference)": "Instrument (reference)",
           "H&K constants a,b, wide (H&K only)": "H&K constants, wide grid",
           "H&K constants a,b, narrow (H&K only)": "H&K constants, narrow grid",
           "kernel": "Kernel (3 stable models)", "idealised constants": "Idealised-spectrum constants",
           "recording": "Recording (2 libraries)", "extraction constants": "Extraction constants",
           "audio codec": "Audio codec (lossless–48 kbps)"}
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    rng = np.random.default_rng(1)
    for i, k in enumerate(order):
        v = np.maximum(np.asarray(u[k]["values"]), 0.05)
        yy = len(order) - 1 - i
        ax.scatter(v, yy + rng.uniform(-0.18, 0.18, len(v)), s=7, color="#4C72B0" if i else "#C44E52",
                   alpha=0.75, edgecolor="none", zorder=3)
        ax.plot([u[k]["median"]] * 2, [yy - 0.3, yy + 0.3], color="black", lw=1.2, zorder=4)
    ax.set_xscale("log")
    ax.set_xlim(0.04, 300)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([lab[k] for k in order][::-1])
    ax.set_xlabel("Range of tritone/m6 gap across factor levels (pp, log scale)")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_decomposition():
    d = json.load(open(RES / "controlled_decomposition.json"))
    comps = ["interference", "harmonicity", "familiarity"]
    ccol = {"interference": "#C44E52", "harmonicity": "#55A868", "familiarity": "#8172B2"}
    specs = [s for s in ("raw", "size", "size+reg", "size+transp") if s in d["bowling2018"]]
    slab = {"raw": "No controls\n(original)", "size": "Chord size", "size+reg": "Size +\nlowest note",
            "size+transp": "Size +\ntransposed"}
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.3), sharey=True)
    for ax, (ds, title) in zip(axes, [("bowling2018", "Bowling et al. (2018), n = 298"),
                                     ("johnson-laird2012", "Johnson-Laird et al. (2012), n = 103")]):
        for j, s in enumerate(specs):
            for k, c in enumerate(comps):
                x = j + (k - 1) * 0.25
                v = d[ds][s]["unique"][c]["delta_r2"]
                lo, hi = d[ds][s]["boot_ci_unique"][c]
                ax.bar(x, v, width=0.23, color=ccol[c], label=c.capitalize() if j == 0 else None)
                ax.errorbar(x, v, yerr=[[v - lo], [hi - v]], color="black", lw=0.7, capsize=1.5)
        ax.set_xticks(range(len(specs)))
        ax.set_xticklabels([slab[s] for s in specs], fontsize=7.5)
        ax.set_title(title, fontsize=9)
    axes[0].set_ylabel(r"Unique $\Delta R^2$ (95% bootstrap CI)")
    axes[0].legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig5_decomposition.pdf")
    plt.close(fig)


def fig_mechanism():
    m = json.load(open(RES / "further_robustness.json"))["mechanism"]["Philharmonia clarinet_C4"]
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.3), sharey=False)
    for ax, (kern, title) in zip(axes, [("hk", "Hutchinson–Knopoff"), ("sethares", "Sethares (product)")]):
        pairs = {}
        for c in ("600", "800"):
            for t in m[kern][c]["top_pairs"][:5]:
                key = f"{t['lower_h']}×{t['upper_h']}"
                pairs.setdefault(key, {"600": 0.0, "800": 0.0})[c] = t["contrib"]
        keys = sorted(pairs, key=lambda k: -(pairs[k]["600"] + pairs[k]["800"]))
        x = np.arange(len(keys))
        tot = {c: m[kern][c]["cross_tone_total"] + m[kern][c]["within_tone_total"] for c in ("600", "800")}
        norm = max(tot.values())
        ax.bar(x - 0.2, [pairs[k]["600"] / norm for k in keys], 0.38, color="#4C72B0",
               label=f"Tritone (total {tot['600'] / norm:.2f})")
        ax.bar(x + 0.2, [pairs[k]["800"] / norm for k in keys], 0.38, color="#DD8452",
               label=f"Minor sixth (total {tot['800'] / norm:.2f})")
        ax.set_xticks(x)
        ax.set_xticklabels(keys, fontsize=7.5)
        ax.set_xlabel("Partial pair (harmonic of lower note × of upper note)")
        ax.set_title(title, fontsize=9)
        ax.legend(frameon=False, fontsize=7)
    axes[0].set_ylabel("Contribution (share of larger total)")
    fig.tight_layout()
    fig.savefig(FIG / "fig4_mechanism.pdf")
    plt.close(fig)


def _calibrated(p, keys):
    """Human profile regressed on model profile(s) over the fit region used in
    marjieh_dyads.py (0.5-14.75 st, +/-0.6 st windows around 6 and 8 held out)."""
    g = np.asarray(p["grid"])
    h = np.asarray(p["human"])
    mask = (g >= 0.5) & (g <= 14.75) & (np.abs(g - 6) >= 0.6) & (np.abs(g - 8) >= 0.6)
    A = np.column_stack([np.ones(len(g))] + [np.asarray(p[k]) for k in keys])
    b, *_ = np.linalg.lstsq(A[mask], h[mask], rcond=None)
    return A @ b


def fig_profiles():
    E = json.load(open(RES / "marjieh_dyads.json"))["experiments"]
    panels = [("harm3", "Harmonic tones, 3 dB/octave (US)"), ("pure", "Pure tones (US)"),
              ("str3", "Stretched harmonics, octave 2.1 (US)"), ("harm3_kr", "Harmonic tones, 3 dB/octave (Korea)")]
    fig, axes = plt.subplots(2, 2, figsize=(W2, 4.4), sharex=True)
    for ax, (k, title) in zip(axes.flat, panels):
        p = E[k]["profile"]
        g = np.asarray(p["grid"])
        ax.fill_between(g, p["human_lo"], p["human_hi"], color="#cccccc", lw=0, label="Listeners (95% CI)")
        ax.plot(g, p["human"], color="black", lw=1.1, label="Listeners")
        ax.plot(g, _calibrated(p, ["HK"]), color="#C44E52", lw=1.0, ls="--",
                label="Roughness curve (H&K), calibrated")
        ax.plot(g, _calibrated(p, ["HK", "H"]), color="#4C72B0", lw=1.0,
                label="Roughness + harmonicity, calibrated")
        for x, t in ((6, "TT"), (8, "m6")):
            ax.axvspan(x - 0.6, x + 0.6, color="#f3e3e3", lw=0, zorder=0)
            ax.text(x, 1.0, t, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=7.5)
        ax.set_title(title, fontsize=8.5, pad=11)
        ax.set_xlim(0, 15)
        ax.set_xticks(range(0, 16, 1))
    for ax in axes[1]:
        ax.set_xlabel("Interval (semitones)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Pleasantness (z)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h[1:], l[1:], frameon=False, fontsize=7.5, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIG / "fig_profiles.pdf")
    plt.close(fig)


EXP_ORDER = [("harm3", "Harmonic, 3 dB/oct"), ("eq5", "5 equal harmonics"), ("no3", "5 harmonics, no 3rd"),
             ("flute", "Flute (sampled)"), ("guitar", "Guitar (sampled)"), ("piano", "Piano (sampled)"),
             ("pure", "Pure tones"), ("bonang", "Bonang over harmonic bass"),
             ("str3", "Stretched (octave 2.1)"), ("comp3", "Compressed (octave 1.9)"),
             ("harm3_kr", "Harmonic, 3 dB/oct (Korea)"), ("str3_kr", "Stretched (Korea)"),
             ("comp3_kr", "Compressed (Korea)")]


def fig_forest():
    E = json.load(open(RES / "marjieh_dyads.json"))["experiments"]
    series = [("obs", "Observed", "black", "o"), ("HK", "Left by roughness (H&K)", "#C44E52", "s"),
              ("HK+H", "Left by roughness + harmonicity", "#4C72B0", "D"),
              ("Composite", "Left by Marjieh et al. composite", "#55A868", "^")]
    fig, axes = plt.subplots(1, 2, figsize=(W2, 3.6), sharey=True)
    y = np.arange(len(EXP_ORDER))[::-1].astype(float)
    y[-3:] -= 0.0
    for ax, (m, title) in zip(axes, [("gap", "Minor sixth minus tritone"), ("dip", "Fourth/fifth mean minus tritone")]):
        for j, (s, lab, col, mk) in enumerate(series):
            key = f"obs_{m}" if s == "obs" else f"{s}_unexpl_{m}"
            for yi, (k, _) in zip(y, EXP_ORDER):
                r = E[k][key]
                yy = yi + (1.5 - j) * 0.17
                ax.plot(r["ci"], [yy, yy], color=col, lw=0.8)
                ax.plot(r["est"], yy, marker=mk, ms=3.2, color=col, lw=0, label=lab if yi == y[0] else None)
        ax.axvline(0, color="black", lw=0.6)
        ax.axhline(y[9] - 0.5, color="#999999", lw=0.5, ls=":")
        ax.set_title(title, fontsize=8.5)
        ax.set_xlabel("Pleasantness difference (z)")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([lab for _, lab in EXP_ORDER], fontsize=7.5)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=7.5, loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(FIG / "fig_forest.pdf")
    plt.close(fig)


def fig_null():
    S = json.load(open(RES / "marjieh_summary.json"))["null"]["us8"]
    names = ["m2", "M2", "m3", "M3", "P4", "TT", "P5", "m6", "M6", "m7", "M7", "P8", "m9", "M9"]
    fig, ax = plt.subplots(figsize=(W2 * 0.62, 2.4))
    x = np.arange(1, 15)
    for j, (k, lab, col) in enumerate([("obs_dip_null", "Observed dip", "#999999"),
                                       ("HK_udip_null", "Left by roughness", "#C44E52"),
                                       ("HK+H_udip_null", "Left by roughness + harmonicity", "#4C72B0")]):
        v = S[k]
        ax.bar(x + (j - 1) * 0.27, v["est"], 0.26, yerr=1.96 * np.asarray(v["se"]), color=col, label=lab,
               error_kw={"lw": 0.6, "capsize": 1})
    ax.axhline(0, color="black", lw=0.6)
    ax.axvspan(5.55, 6.45, color="#f3e3e3", zorder=0, lw=0)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=7.5)
    ax.set_ylabel("Dip below neighbours (z)")
    ax.set_xlabel("Interval")
    ax.legend(frameon=False, fontsize=7, ncol=1)
    fig.tight_layout()
    fig.savefig(FIG / "fig_null.pdf")
    plt.close(fig)


def fig_sigma():
    sc = json.load(open(RES / "model_screen2.json"))
    sig = [6.83, 10, 15, 20, 25, 30, 40, 60]
    rows = [sc[f"HK+H{x:g} (fixed sigma)"] for x in sig]
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.5))
    ax = axes[0]
    ax.plot(sig, [r["r2_us_mean"] for r in rows], "o-", color="#4C72B0", ms=3.5, lw=1,
            label=r"H&K + harmonicity($\sigma$)")
    for key, lab, col, ls in (("Composite (fixed)", "Marjieh et al. composite", "#55A868", "--"),
                              ("revHK+H", "Revised H&K + harmonicity", "#8172B2", ":")):
        ax.axhline(sc[key]["r2_us_mean"] if key in sc else json.load(open(RES / "model_screen.json"))[key]["r2_us_mean"],
                   color=col, ls=ls, lw=1, label=lab)
    ax.set_xscale("log")
    ax.set_xticks(sig)
    ax.set_xticklabels([f"{x:g}" for x in sig], fontsize=7)
    ax.minorticks_off()
    ax.set_xlabel(r"Harmonicity tolerance $\sigma$ (cents)")
    ax.set_ylabel("Mean held-out $R^2$")
    ax.legend(frameon=False, fontsize=6.5, loc="lower left")
    ax = axes[1]
    for key, lab, col, mk in (("gap_harm6", "Gap left (m6 $-$ TT)", "#C44E52", "o"),
                              ("dip_harm6", "Dip left (P4/P5 $-$ TT)", "#DD8452", "s"),
                              ("tt_minus_other", "TT residual dip $-$ mean of other 13", "#555555", "^")):
        ax.plot(sig, [r[key] for r in rows], marker=mk, ms=3.5, lw=1, color=col, label=lab)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xscale("log")
    ax.set_xticks(sig)
    ax.set_xticklabels([f"{x:g}" for x in sig], fontsize=7)
    ax.minorticks_off()
    ax.set_xlabel(r"Harmonicity tolerance $\sigma$ (cents)")
    ax.set_ylabel("Held-out residual (z)")
    ax.legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig_sigma.pdf")
    plt.close(fig)


def fig_sharp():
    sh = json.load(open(RES / "sharpness.json"))
    ce = sh["ceiling"]
    lab = {"harm3": "Harm. 3 dB", "str3": "Stretched", "comp3": "Compressed", "eq5": "5 equal",
           "no3": "No 3rd", "pure": "Pure", "bonang": "Bonang", "flute": "Flute", "guitar": "Guitar",
           "piano": "Piano", "harm3_kr": "KR harm.", "str3_kr": "KR str.", "comp3_kr": "KR comp."}
    exps = list(lab)
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.6), gridspec_kw={"width_ratios": [1.6, 1, 1.6]})
    ax = axes[0]
    y = np.arange(len(exps))
    sl = [12 * sh["insample"]["HK+H20+X1"][e][1][2] for e in exps]
    ax.barh(y, sl, color=["#C44E52" if e.endswith("_kr") else "#4C72B0" for e in exps], height=0.6)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels([lab[e] for e in exps], fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel("Residual slope (SD per octave)")
    ax.set_title("A  Trend over interval size", fontsize=8, loc="left")
    ax = axes[1]
    ro = sh["rolloff"]
    lev = ro["levels"]
    est = [r["est"] for r in ro["slice_x_slope_per_octave"]]
    lo = [r["est"] - r["ci"][0] for r in ro["slice_x_slope_per_octave"]]
    hi = [r["ci"][1] - r["est"] for r in ro["slice_x_slope_per_octave"]]
    ax.errorbar(lev, est, yerr=[lo, hi], fmt="o-", color="#4C72B0", ms=3.5, lw=1, capsize=2)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Spectral roll-off (dB/octave)")
    ax.set_ylabel("Residual slope (SD per octave)")
    ax.set_title("B  Roll-off experiment", fontsize=8, loc="left")
    ax = axes[2]
    base = sh["screen"]["HK+H20"]["per_exp_r2"]
    shp = sh["screen"]["HK+H20+SHARP"]["per_exp_r2"]
    ax.scatter([max(base[e], -0.6) for e in exps], y, s=12, color="#999999", label="H&K + H$_{20}$", zorder=3)
    ax.scatter([max(shp[e], -0.6) for e in exps], y, s=12, color="#C44E52", label="+ sharpness", zorder=3)
    ax.scatter([ce[e]["ceiling"] for e in exps], y, s=18, marker="|", color="k", label="Ceiling", zorder=3)
    for i, e in enumerate(exps):
        ax.plot([max(base[e], -0.6), max(shp[e], -0.6)], [i, i], color="#cccccc", lw=0.8, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.set_xlim(-0.65, 1.02)
    ax.set_xlabel("Held-out $R^2$ (clipped at $-0.6$)")
    ax.set_title("C  Prediction vs ceiling", fontsize=8, loc="left")
    ax.legend(frameon=False, fontsize=6, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "fig_sharp.pdf")
    plt.close(fig)


def fig_etji():
    """A8 (exploratory): pooled harmonic profile at kernel SD 0.1 over 500-900 cents with the held-out
    model curves; equal-tempered positions solid, just-intonation ratios dashed; pooled listener
    peak/dip locations with Hartung-Knapp CIs."""
    A = json.load(open(RES / "et_ji_positions.json"))
    F = A["figure"]
    g = 100 * np.asarray(F["grid"])
    m = (g >= 500) & (g <= 900)
    fig, ax = plt.subplots(figsize=(W2, 3.6))
    for x in (500, 600, 700, 800, 900):
        ax.axvline(x, color="#bbbbbb", lw=0.7, ls="-", zorder=0)
    ji = {"4:3": 1200 * np.log2(4 / 3), "7:5": 1200 * np.log2(7 / 5), "45:32": 1200 * np.log2(45 / 32),
          "3:2": 1200 * np.log2(3 / 2), "8:5": 1200 * np.log2(8 / 5), "5:3": 1200 * np.log2(5 / 3)}
    for t, x in ji.items():
        ax.axvline(x, color="#bbbbbb", lw=0.7, ls="--", zorder=0)
        ha = {"7:5": "right", "45:32": "left"}.get(t, "center")
        ax.text(x, 1.0, t, transform=ax.get_xaxis_transform(), ha=ha, va="bottom", fontsize=6.5, color="#777777")
    cols = {"HK+H": ("#4C72B0", "-", "H&K + harmonicity (6.83 cents)"),
            "HK+H20": ("#55A868", "-", "H&K + harmonicity (20 cents)"),
            "Composite": ("#8172B2", ":", "Marjieh et al. composite")}
    ax.plot(g[m], np.asarray(F["sd0.1"]["listener"])[m], color="k", lw=1.3, label="Listeners (6 harmonic spectra)")
    for k, (c, ls, lab) in cols.items():
        ax.plot(g[m], np.asarray(F["sd0.1"][k])[m], color=c, lw=1.0, ls=ls, label=lab + ", held out")
    P = A["pooled"]["sd0.1"]["listener"]
    y0, y1 = ax.get_ylim()
    yb = y0 + 0.06 * (y1 - y0)
    for loc, mk in (("m6", "v"), ("tt", "^"), ("p5", "v"), ("M6", "v")):
        v = P[f"loc_{loc}"]
        if not (5.0 <= v["est"] <= 9.0):
            continue
        ax.errorbar(100 * v["est"], yb, xerr=[[100 * (v["est"] - v["hk_ci"][0])], [100 * (v["hk_ci"][1] - v["est"])]],
                    fmt=mk, color="#C44E52", ms=4, lw=1.0, capsize=2, zorder=4)
    ax.errorbar([], [], xerr=[], fmt="v", color="#C44E52", ms=4, lw=1.0, label="Listener peak / dip, pooled [95% CI]")
    ax.set_xlim(500, 900)
    ax.set_xlabel("Interval (cents)")
    ax.set_ylabel("Pleasantness (z)")
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, fontsize=6.5, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(FIG / "fig_etji.pdf")
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    for f in (fig_etji, fig_sharp, fig_sigma, fig_profiles, fig_forest, fig_null, fig_curves, fig_gaps, fig_sensitivity, fig_decomposition,
              fig_mechanism):
        f()
        print("built", f.__name__)
