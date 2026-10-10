"""
Interval sweep (unison -> octave) using real instrument overtone spectra,
compared against the idealized-harmonic-partial baseline, under both the
Sethares (1993) and Vassilakis (2001) dissonance models.

Two analyses:
  1. Instrument comparison: 7 real instruments at ~C4 vs. idealized baseline.
  2. Register test: does the clarinet's Sethares tritone<m6 flip (found at C4)
     hold at other registers, and do cello/oboe behave the same way as controls?
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dissonance_model import (
    harmonic_partials, interval_sweep, sethares_dissonance, sethares_min_dissonance,
    vassilakis_dissonance, hutch_knopoff_dissonance,
)

INTERVAL_NAMES = {
    0: "unison", 100: "m2", 200: "M2", 300: "m3", 400: "M3", 500: "P4",
    600: "TRITONE", 700: "P5", 800: "m6", 900: "M6", 1000: "m7", 1100: "M7", 1200: "octave",
}
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


def value_at(cents, diss, target_cents):
    return diss[np.argmin(np.abs(cents - target_cents))]


def sweep_for(label, root_f0, ratios=None, rel_amps=None):
    """Run both models for one timbre; ratios=None means idealized harmonic baseline."""
    rows = []
    curves = {}
    for model_name, model_fn in MODELS:
        if ratios is None:
            cents, diss = interval_sweep(f0=root_f0, model=model_fn, n_partials=7, rolloff=0.9)
        else:
            pfn = make_real_partials_fn(ratios, rel_amps)
            cents, diss = interval_sweep(f0=root_f0, partials_fn=pfn, model=model_fn)
        diss = diss / diss.max()
        curves[model_name] = (cents, diss)
        tritone, m6 = value_at(cents, diss, 600), value_at(cents, diss, 800)
        rows.append({
            "timbre": label, "model": model_name,
            "tritone": round(float(tritone), 4), "m6": round(float(m6), 4),
            "tritone_gt_m6": bool(tritone > m6),
            "gap_pct": round(100 * float(tritone - m6) / float(m6), 1),
        })
    return rows, curves


def print_table(rows):
    print(f"{'timbre':30s} {'model':10s} {'tritone':>8s} {'m6':>8s} {'trit>m6':>8s} {'gap%':>7s}")
    for row in rows:
        print(f"{row['timbre']:30s} {row['model']:10s} {row['tritone']:8.4f} {row['m6']:8.4f} "
              f"{str(row['tritone_gt_m6']):>8s} {row['gap_pct']:7.1f}")


def plot_curves(curves_by_label, title, out_path, model_name):
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, curves in curves_by_label.items():
        cents, diss = curves[model_name]
        ax.plot(cents, diss, label=label, linewidth=1.8)
    ax.axvline(600, color="red", linestyle="--", alpha=0.6, label="tritone (600c)")
    ax.axvline(800, color="orange", linestyle="--", alpha=0.6, label="minor 6th (800c)")
    for c in INTERVAL_NAMES:
        ax.axvline(c, color="gray", alpha=0.12, linewidth=1)
    ax.set_xlabel("Interval above root (cents)")
    ax.set_ylabel("Dissonance (normalized to curve max)")
    ax.set_title(f"{model_name}: {title}")
    ax.set_xlim(0, 1200)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def main():
    with open("extracted_partials.json") as f:
        extracted = json.load(f)

    all_rows = []

    # --- Analysis 1: instrument comparison at ~C4 ---
    print("\n=== Analysis 1: 7 real instruments at C4 vs. idealized baseline ===")
    c4_labels = ["cello_C4", "clarinet_C4", "oboe_C4", "flute_C4",
                 "bassoon_C4", "saxophone_C4", "french-horn_C4", "piano_C4"]
    curves1 = {}
    rows, curves = sweep_for("idealized (7 harm, rolloff 0.9)", 261.63)  # nominal C4, no measurement to anchor to
    all_rows += rows
    curves1["idealized (7 harm, rolloff 0.9)"] = curves
    for label in c4_labels:
        data = extracted[label]
        # root = each timbre's own measured f0 (not nominal 261.63), matching Analysis 2's convention
        rows, curves = sweep_for(data["instrument"], data["f0_hz"], data["ratios"], data["rel_amps"])
        all_rows += rows
        curves1[data["instrument"]] = curves
    print_table([r for r in all_rows])
    for model_name, _ in MODELS:
        plot_curves(curves1, "idealized vs. 7 real instruments (root=C4)",
                    f"dissonance_curves_{model_name.lower()}.png", model_name)

    # --- Analysis 2: register test ---
    print("\n=== Analysis 2: register test (does the C4 clarinet flip hold elsewhere?) ===")
    register_sets = {
        "clarinet": [("D3 (low)", "clarinet_D3"), ("C4 (mid)", "clarinet_C4"),
                     ("C5 (high)", "clarinet_C5")],
        "cello": [("C3 (low)", "cello_C3"), ("C4 (mid)", "cello_C4"),
                  ("C5 (high)", "cello_C5")],
        "oboe": [("A#3 (low)", "oboe_As3"), ("C4 (mid)", "oboe_C4"),
                 ("C5 (high)", "oboe_C5")],
    }
    register_rows = []
    for instrument, notes in register_sets.items():
        curves2 = {}
        for reg_label, key in notes:
            data = extracted[key]
            rows, curves = sweep_for(f"{instrument} {reg_label}", data["f0_hz"], data["ratios"], data["rel_amps"])
            register_rows += rows
            curves2[reg_label] = curves
        for model_name, _ in MODELS:
            plot_curves(curves2, f"{instrument} across registers (root moves with each note)",
                        f"register_test_{instrument}_{model_name.lower()}.png", model_name)
    print_table(register_rows)

    with open("results_summary.json", "w") as f:
        json.dump({"instrument_comparison": all_rows, "register_test": register_rows}, f, indent=2)
    print("\nSaved results_summary.json")


if __name__ == "__main__":
    main()
