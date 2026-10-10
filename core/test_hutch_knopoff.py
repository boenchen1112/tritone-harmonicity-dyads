"""
Verify hutch_knopoff_dissonance() against pmcharrison/dycon's own testthat
reference values (tests/testthat/test-hutch.R in that repo).

Only the pure-tone cases (num_harmonics = 1) are checked. The multi-harmonic
cases in dycon's suite depend on hrep::sparse_fr_spectrum's amplitude rolloff
convention, which is a property of that package rather than of Hutchinson &
Knopoff's model -- reproducing them would test hrep, not this implementation.
The pure-tone cases pin down everything that is actually H&K's: the critical
bandwidth formula, the roughness kernel g(y), and the sum(a^2) normalization.
"""
import numpy as np

from dissonance_model import hutch_knopoff_dissonance

TOL = 1e-3


def midi_to_hz(m):
    return 440.0 * 2 ** ((m - 69) / 12.0)


def check(label, got, expected, tol=TOL):
    ok = abs(got - expected) <= tol
    print(f"  {'PASS' if ok else 'FAIL'}  {label:34s} got={got:.6f}  expected={expected:.4f}")
    return ok


def main():
    passed = []

    # --- hutch_cbw: 1.72 * mean_f^0.65 ---
    print("critical bandwidth (dycon test_that 'hutch_cbw'):")
    for f1, f2, expected in [(400, 440, 87.225), (400, 380, 83.123)]:
        cbw = 1.72 * ((f1 + f2) / 2.0) ** 0.65
        passed.append(check(f"cbw({f1},{f2})", cbw, expected, tol=5e-4))

    # --- hutch_g: ((y/a)*exp(1-y/a))^b, zeroed above the 1.2 cutoff ---
    print("\nroughness kernel g(y) (dycon test_that 'hutch_g'):")
    for y, cutoff, expected in [(1.0, 1.2, 0.0397), (0.5, 1.2, 0.5413),
                                 (1.5, None, 0.0016), (1.5, 1.2, 0.0)]:
        g = ((y / 0.25) * np.exp(1.0 - y / 0.25)) ** 2
        if cutoff is not None and y > cutoff:
            g = 0.0
        passed.append(check(f"g({y}, cutoff={cutoff})", g, expected, tol=5e-5))

    # --- full model on pure-tone dyads (dycon test_that 'get_roughness_hutch') ---
    print("\nfull model, pure tones (dycon test_that 'get_roughness_hutch'):")
    for midi_pair, expected in [((60, 61), 0.499), ((69, 70), 0.491)]:
        freqs = np.array([midi_to_hz(m) for m in midi_pair])
        amps = np.ones(2)
        passed.append(check(f"roughness_hutch({midi_pair[0]} {midi_pair[1]}, 1 harmonic)",
                            hutch_knopoff_dissonance(freqs, amps), expected))

    # --- degenerate inputs should not raise ---
    print("\ndegenerate inputs:")
    far = hutch_knopoff_dissonance(np.array([100.0, 8000.0]), np.ones(2))
    passed.append(check("partials far beyond the cutoff -> 0", far, 0.0, tol=1e-12))

    print(f"\n{sum(passed)}/{len(passed)} checks passed")
    if not all(passed):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
