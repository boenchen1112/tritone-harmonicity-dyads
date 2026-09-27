"""Generate manuscript/numbers.tex (one \\newcommand per quoted number) and
manuscript/tables/*.tex from the study's JSON outputs. The .tex manuscript
contains no hand-typed results: every figure in the text is a macro defined
here, so re-running the analyses and this script updates the paper."""
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import PROJECT_ROOT
import numpy as np

ROOT = PROJECT_ROOT
MS = Path(__file__).resolve().parents[1]
RES = MS / "results"
TAB = MS / "tables"
TAB.mkdir(exist_ok=True)

J = lambda p: json.load(open(p))
RS = J(ROOT / "results_summary.json")
EX = J(ROOT / "extraction_sensitivity_results.json")
COD = J(ROOT / "codec_ab_results.json")
REP = J(ROOT / "clarinet_replication_results.json")["registers"]
ROB = J(ROOT / "robustness_checks.json")
KVI = J(RES / "kernel_vs_instrument.json")
CD = J(RES / "controlled_decomposition.json")
FR = J(RES / "further_robustness.json")
US = J(RES / "unified_sensitivity.json")
PH = J(ROOT / "extracted_partials.json")

MODELS = ["Sethares", "Sethares-min", "Vassilakis", "Hutch-Knopoff"]
M = {}   # macro name -> string


def mac(name, value):
    assert name.isalpha(), name
    assert name not in M, name
    M[name] = value


def pct(x, d=1):
    if round(x, d) == 0:
        return f"{0:.{d}f}"
    s = f"{x:+.{d}f}"
    return s.replace("-", "\\textminus{}")


def num(x, d=3):
    return f"{x:.{d}f}".replace("-", "\\textminus{}")


def gap(section, timbre, model):
    return next(r["gap_pct"] for r in RS[section] if r["timbre"] == timbre and r["model"] == model)


# ---------------------------------------------------------------- premise (Bowling dyads)
rows = list(csv.DictReader(open(ROOT / "behavioral_data/bowling2018.tsv"), delimiter="\t"))
NAMES = {1: "Minor second", 2: "Major second", 3: "Minor third", 4: "Major third", 5: "Perfect fourth",
         6: "Tritone", 7: "Perfect fifth", 8: "Minor sixth", 9: "Major sixth", 10: "Minor seventh",
         11: "Major seventh", 12: "Octave"}
dy = []
for r in rows:
    p = [int(x) for x in r["pitches"].split(",")]
    if len(p) == 2:
        dy.append((p[1] - p[0], float(r["rating"])))
assert len(dy) == 12
dy.sort(key=lambda t: t[1])
with open(TAB / "tab_dyads.tex", "w") as f:
    for rank, (st, rt) in enumerate(dy, 1):
        bold = st in (6, 8)
        cells = [NAMES[st], str(st), f"{rt:.3f}", str(rank)]
        if bold:
            cells = [f"\\textbf{{{c}}}" for c in cells]
        f.write(" & ".join(cells) + " \\\\\n")
tt = dict((s, r) for s, r in dy)
mac("ratingTT", f"{tt[6]:.3f}")
mac("ratingMsix", f"{tt[8]:.3f}")
mac("nBelowTT", str(sum(1 for s, r in dy if r < tt[6])))

# ---------------------------------------------------------------- gap tables
def gap_table(section, fname, label_fn):
    timbres = list(dict.fromkeys(r["timbre"] for r in RS[section]))
    with open(TAB / fname, "w") as f:
        for t in timbres:
            cells = [label_fn(t)]
            for m in MODELS:
                g = gap(section, t, m)
                s = pct(g)
                cells.append(f"\\flip{{{s}}}" if g < 0 else s)
            f.write(" & ".join(cells) + " \\\\\n")
        if section == "instrument_comparison":
            meas = [t for t in timbres if not t.startswith("idealized")]
            f.write("\\midrule\nNegative gaps (of 8 measured) & " +
                    " & ".join(str(sum(gap(section, t, m) < 0 for t in meas)) for m in MODELS) + " \\\\\n")


def c4label(t):
    if t.startswith("idealized"):
        return "Idealised (7 harmonics, $0.9^{n-1}$)"
    lab = {"french-horn": "French horn"}.get(t, t.capitalize())
    return lab + (" (Iowa)" if t == "piano" else "")


gap_table("instrument_comparison", "tab_gaps_c4.tex", c4label)
gap_table("register_test", "tab_gaps_register.tex",
          lambda t: t.replace(" (low)", "").replace(" (mid)", "").replace(" (high)", "")
          .replace("A#3", "A\\sh3").capitalize().replace("c3", "C3").replace("c4", "C4")
          .replace("c5", "C5").replace("d3", "D3").replace("a\\sh3", "A\\sh3"))

for m, short in zip(MODELS, ["S", "Smin", "V", "HK"]):
    mac(f"gapClarCFour{short}", pct(gap("instrument_comparison", "clarinet", m)))
mac("gapBassoonHK", pct(gap("instrument_comparison", "bassoon", "Hutch-Knopoff")))
mac("gapHornV", pct(gap("instrument_comparison", "french-horn", "Vassilakis")))

# ---------------------------------------------------------------- kernel vs instrument
for key, tag in (("instrument_comparison/all4", "All"), ("instrument_comparison/noVass", "NoV"),
                 ("register_test/all4", "RegAll"), ("register_test/noVass", "RegNoV")):
    k = KVI[key]
    p = k["partition_gap_pct"]
    q = k["partition_log_ratio"]
    mac(f"shareInstr{tag}", f"{100 * p['instrument_share']:.0f}")
    mac(f"shareModel{tag}", f"{100 * p['model_share']:.0f}")
    mac(f"shareInter{tag}", f"{100 * p['interaction_share']:.0f}")
    mac(f"shareInstrLR{tag}", f"{100 * q['instrument_share']:.0f}")
    mac(f"shareModelLR{tag}", f"{100 * q['model_share']:.0f}")
    mac(f"shareInterLR{tag}", f"{100 * q['interaction_share']:.0f}")
    s = k["sign"]
    mac(f"signModel{tag}", f"{s['model_pairs_disagree']}/{s['model_pairs_total']}")
    mac(f"signModelPct{tag}", f"{100 * s['model_pair_rate']:.0f}")
    mac(f"signInstr{tag}", f"{s['instrument_pairs_disagree']}/{s['instrument_pairs_total']}")
    mac(f"signInstrPct{tag}", f"{100 * s['instrument_pair_rate']:.0f}")
    r = k["ranges_pp"]
    mac(f"medRangeModels{tag}", f"{r['median_across_models']:.1f}")
    mac(f"medRangeInstr{tag}", f"{r['median_across_instruments']:.1f}")
ci = KVI["instrument_comparison/all4"]["ranges_pp"]
mac("rangeClarKernels", f"{ci['across_models_by_instrument']['clarinet']:.1f}")
mac("rangeHKInstr", f"{ci['across_instruments_by_model']['Hutch-Knopoff']:.1f}")

# ---------------------------------------------------------------- unified sensitivity table
ORDER = [("instrument (reference)", "Instrument (8 at C4; reference quantity)"),
         ("kernel", "Kernel (Sethares, Sethares-min, H\\&K)"),
         ("H&K constants a,b, wide (H&K only)", "H\\&K constants $a,b$, wide grid$^\\dagger$"),
         ("H&K constants a,b, narrow (H&K only)", "H\\&K constants $a,b$, narrow grid$^\\dagger$"),
         ("idealised constants", "Idealised-spectrum constants (partials, roll-off)"),
         ("recording", "Recording (Philharmonia vs Iowa clarinet)"),
         ("extraction constants", "Extraction constants (prominence, cap)"),
         ("audio codec", "Audio codec (lossless to 48 kbit/s)")]
with open(TAB / "tab_sensitivity.tex", "w") as f:
    for k, lab in ORDER:
        s = US["stable"][k]
        vk = {"H&K constants a,b, wide (H&K only)": "Vassilakis exponent",
              "H&K constants a,b, narrow (H&K only)": None}.get(k, k)
        v = US["vassilakis"].get(vk) if vk else None
        vcell = f"{v['median']:.1f} & {v['max']:.1f}" if v else "-- & --"
        if k == "H&K constants a,b, wide (H&K only)":
            vcell += "$^\\ddagger$"
        f.write(f"{lab} & {s['n_cells']} & {s['median']:.1f} & {s['max']:.1f} & {vcell} \\\\\n")
for k, tag in (("kernel", "Kernel"), ("recording", "Recording"), ("extraction constants", "Extraction"),
               ("audio codec", "Codec"), ("idealised constants", "Ideal"),
               ("H&K constants a,b, wide (H&K only)", "ABWide"),
               ("H&K constants a,b, narrow (H&K only)", "ABNarrow"), ("instrument (reference)", "Instr")):
    mac(f"sens{tag}Med", f"{US['stable'][k]['median']:.1f}")
    mac(f"sens{tag}Max", f"{US['stable'][k]['max']:.1f}")
mac("sensVassExtractMax", f"{US['vassilakis']['extraction constants']['max']:.1f}")
mac("sensVassExtractMed", f"{US['vassilakis']['extraction constants']['median']:.1f}")
oc = US["original_figure2_check"]["codec_worst_delta_by_model"]
mac("codecWorstVass", f"{oc['Vassilakis']:.1f}")
mac("codecWorstStable", f"{US['original_figure2_check']['codec_worst_delta_stable_models']:.1f}")
agg = EX["aggregate"]
mac("extrVassWorst", f"{agg['Vassilakis']['max_shift_pp']:.1f}")
mac("extrHKWorst", f"{agg['Hutch-Knopoff']['max_shift_pp']:.1f}")
mac("hornVassLoose", pct(EX["timbres"]["french-horn_C4"]["models"]["Vassilakis"]["published_gap_pct"]
                        + min(EX["timbres"]["french-horn_C4"]["models"]["Vassilakis"]["deltas_pp"].values())))
ib = EX["idealized_baseline"]["models"]["Hutch-Knopoff"]
mac("idealHKFlat", pct(ib["published_gap_pct"] + ib["deltas_pp"]["n5_r1.0"]))
mac("idealHKBase", pct(ib["published_gap_pct"]))

# ---------------------------------------------------------------- clarinet robustness table
cut = ROB["b_cutoff_sensitivity"]["gaps_pct"]
ab = FR["hk_ab_sweep"]
erb = FR["hk_erb"]
with open(TAB / "tab_clarinet.tex", "w") as f:
    for reg in ("D3", "C4", "C5"):
        for lib in ("Philharmonia", "Iowa"):
            key = f"{lib} clarinet_{reg}"
            if lib == "Philharmonia":
                base = gap("register_test", {"D3": "clarinet D3 (low)", "C4": "clarinet C4 (mid)",
                                             "C5": "clarinet C5 (high)"}[reg], "Hutch-Knopoff")
                extr = REP[reg]["models"]["Hutch-Knopoff"]["phil_extraction_range_pp"]
                cvals = list(cut[f"clarinet_{reg}"].values())
                ccell = f"[{pct(min(cvals))}, {pct(max(cvals))}]"
            else:
                base = REP[reg]["models"]["Hutch-Knopoff"]["iowa_gap_pct"]
                extr = REP[reg]["models"]["Hutch-Knopoff"]["iowa_extraction_range_pp"]
                ccell = "--"
            nlo, nhi = ab["clarinet_range_narrow"][key]
            wlo, whi = ab["clarinet_range"][key]
            wcell = f"[{pct(wlo)}, {pct(whi)}]"
            if whi >= 0:
                wcell = f"[{pct(wlo)}, \\flip{{{pct(whi)}}}]"
            f.write(f"{reg} & {lib} & {pct(base)} & {extr:.2f} & {ccell} & [{pct(nlo)}, {pct(nhi)}] & "
                    f"{wcell} & {pct(erb[key]['erb_cut1.2'])} \\\\\n")
for reg in ("D3", "C4", "C5"):
    t = {"D3": "DThree", "C4": "CFour", "C5": "CFive"}[reg]
    mac(f"iowaHK{t}", pct(REP[reg]["models"]["Hutch-Knopoff"]["iowa_gap_pct"]))
    mac(f"philHK{t}", pct(REP[reg]["models"]["Hutch-Knopoff"]["phil_gap_pct"]))
    mac(f"diffHK{t}", f"{abs(REP[reg]['models']['Hutch-Knopoff']['difference_pp']):.1f}")
    mac(f"eoIowa{t}", f"{REP[reg]['iowa']['even_odd_amp_ratio']:.3f}")
    mac(f"eoPhil{t}", f"{REP[reg]['philharmonia']['even_odd_amp_ratio']:.3f}")
mac("sethCFiveIowa", pct(REP["C5"]["models"]["Sethares"]["iowa_gap_pct"]))
mac("sethCFivePhil", pct(REP["C5"]["models"]["Sethares"]["phil_gap_pct"]))
with open(TAB / "tab_replication.tex", "w") as f:
    for m in MODELS:
        cells = [{"Hutch-Knopoff": "Hutchinson--Knopoff", "Sethares-min": "Sethares-min"}.get(m, m)]
        for reg in ("D3", "C4", "C5"):
            d = REP[reg]["models"][m]
            s = f"{pct(d['phil_gap_pct'])} / {pct(d['iowa_gap_pct'])}"
            if np.sign(d["phil_gap_pct"]) != np.sign(d["iowa_gap_pct"]):
                s += "$^{*}$"
            cells.append(s)
        noise = [REP[r]["models"][m][f"{lib}_extraction_range_pp"] for r in REP for lib in ("iowa", "phil")]
        cells.append(f"{max(noise):.1f}")
        f.write(" & ".join(cells) + " \\\\\n")

# ---------------------------------------------------------------- mechanism
mc = FR["mechanism"]["Philharmonia clarinet_C4"]
hk8 = mc["hk"]["800"]["top_pairs"][0]
mac("mechHKpairShare", f"{100 * hk8['share_of_cross']:.0f}")
mac("mechHKpairBeat", f"{hk8['beat_hz']:.0f}")
mac("mechHKpairFlo", f"{hk8['f_lower']:.0f}")
mac("mechHKpairFhi", f"{hk8['f_upper']:.0f}")
s6 = next(t for t in mc["sethares"]["600"]["top_pairs"] if t["lower_h"] == 1 and t["upper_h"] == 1)
mac("mechSethFundShare", f"{100 * s6['share_of_cross']:.0f}")
mac("mechSethFundBeat", f"{s6['beat_hz']:.0f}")
fm = (s6["f_lower"] + s6["f_upper"]) / 2
mac("mechFundCBW", f"{s6['beat_hz'] / (1.72 * fm ** 0.65):.2f}")
mac("mechFundERB", f"{s6['beat_hz'] / (24.7 * (4.37 * fm / 1000 + 1)):.2f}")
hk_ratio = (mc["hk"]["600"]["cross_tone_total"] + mc["hk"]["600"]["within_tone_total"]) / \
           (mc["hk"]["800"]["cross_tone_total"] + mc["hk"]["800"]["within_tone_total"])
mac("mechHKTTovMsix", f"{hk_ratio:.2f}")
ce = FR["mechanism"]["Philharmonia cello_C4"]["sethares"]["600"]["top_pairs"]
mac("mechCelloFundShare", f"{100 * next(t['share_of_cross'] for t in ce if t['lower_h'] == 1 and t['upper_h'] == 1):.0f}")

# ---------------------------------------------------------------- H&K a,b and ERB summary macros
mac("abStableWide", f"{sum(ab['sign_stable'].values())}")
mac("abStableNarrow", f"{sum(ab['sign_stable_narrow'].values())}")
mac("abNSpectra", f"{len(ab['sign_stable'])}")
mac("abUnstableNarrow", ", ".join(k.replace("Philharmonia ", "").replace("Iowa ", "").replace("_", " ")
                                  .replace("french-horn", "French horn").replace("As3", "A\\sh3")
                                  for k, v in ab["sign_stable_narrow"].items() if not v))
mac("cutCelloCFour", pct(cut["cello_C4"]["1.5"]))
mac("gapCelloCFourHK", pct(gap("instrument_comparison", "cello", "Hutch-Knopoff")))
mac("gapHornHK", pct(gap("instrument_comparison", "french-horn", "Hutch-Knopoff")))
mac("clarHKExtrMax", f"{max(REP[r]['models']['Hutch-Knopoff'][f'{l}_extraction_range_pp'] for r in REP for l in ('iowa', 'phil')):.1f}")
mac("clarCutMaxRange", f"{max(max(v.values()) - min(v.values()) for k, v in cut.items() if k.startswith('clarinet')):.1f}")
unstable = [k for k, v in ab["sign_stable_narrow"].items() if not v]
mac("abUnstableMaxAbs", f"{max(abs(erb[k]['cbw_cut1.2']) for k in unstable):.1f}")
pos = ab["positive_cells_clarinet"]
mac("abClarPositiveCells", "; ".join(
    f"{k.split(' ')[0]} {k.split('_')[1]}: " + ", ".join(f"${c.replace(',', ', ')}$" for c in v)
    for k, v in pos.items() if v))
mac("mechFundMean", f"{fm:.0f}")
yv = s6["beat_hz"] / (1.72 * fm ** 0.65) / 0.25
mac("mechFundGpct", f"{100 * (yv * np.exp(1 - yv)) ** 2:.2f}")
erb_flips = [k for k, v in erb.items() if np.sign(v["cbw_cut1.2"]) != np.sign(v["erb_cut1.2"])]
mac("erbSignChanges", ", ".join(k.split(" ", 1)[1].replace("_", " ") for k in erb_flips))
mac("erbNSignChanges", str(len(erb_flips)))
pi = FR["piano_inharmonicity"]
mac("pianoB", f"{pi['B'] * 1e4:.2f}")
mac("pianoBse", f"{pi['B_se'] * 1e7:.1f}")
mac("pianoResid", f"{pi['max_abs_resid']:.3f}")
mac("pianoTopRatio", f"{PH['piano_C4']['ratios'][-1]:.2f}")


# ---------------------------------------------------------------- supplementary: per-spectrum H&K sweeps, Vassilakis exponent
def slabel(k):
    lib, sp = k.split(" ", 1)
    inst, reg = sp.rsplit("_", 1)
    inst = {"french-horn": "French horn"}.get(inst, inst.capitalize())
    reg = reg.replace("As3", "A\\sh3")
    return f"{inst} {reg} ({lib[:4] if lib == 'Iowa' else 'Phil.'})"


def main_hk(k):
    """Baseline H&K gap as printed in the main tables (results_summary / replication JSON)."""
    lib, sp = k.split(" ", 1)
    inst, reg = sp.rsplit("_", 1)
    if lib == "Iowa" and inst == "clarinet":
        return REP[reg]["models"]["Hutch-Knopoff"]["iowa_gap_pct"]
    if inst == "piano":
        return gap("instrument_comparison", "piano", "Hutch-Knopoff")
    reg_names = {r["timbre"] for r in RS["register_test"]}
    name = next((t for t in reg_names if t.startswith(f"{inst} {reg.replace('As3', 'A#3')} ")), None)
    if name:
        return gap("register_test", name, "Hutch-Knopoff")
    return gap("instrument_comparison", inst, "Hutch-Knopoff")


def signed_cell(x, ref):
    s = pct(x)
    return f"\\flip{{{s}}}" if np.sign(x) != np.sign(ref) else s


with open(TAB / "tab_supp_hk.tex", "w") as f:
    for k, g in ab["gaps"].items():
        base = main_hk(k)
        nv = [g[q] for q in ab["narrow_grid"]]
        wv = list(g.values())
        f.write(" & ".join([slabel(k), pct(base), f"[{pct(min(nv))}, {pct(max(nv))}]",
                            f"[{pct(min(wv))}, {pct(max(wv))}]",
                            signed_cell(erb[k]["erb_cut1.2"], base), signed_cell(erb[k]["erb_nocut"], base)])
                + " \\\\\n")
ve = FR["vassilakis_exponent"]
with open(TAB / "tab_supp_vass.tex", "w") as f:
    for k, g in ve.items():
        f.write(" & ".join([slabel(k)] + [signed_cell(g[e], g["0.1"]) for e in ("0.1", "0.25", "0.5", "1.0")])
                + " \\\\\n")
mac("vassExpSignChanges", str(sum(len({np.sign(v) for v in g.values()}) > 1 for g in ve.values())))
mac("vassExpNSpectra", str(len(ve)))
wide = ab["grid"]
mac("abGridA", ", ".join(str(x) for x in wide["a"]))
mac("abGridB", ", ".join(str(x) for x in wide["b"]))

# ---------------------------------------------------------------- model fit (Bowling per size)
ap = ROB["a_pooling"]
with open(TAB / "tab_fit.tex", "w") as f:
    for sz, lab, n in (("2", "Dyads", 12), ("3", "Triads", 66), ("4", "Tetrachords", 220)):
        f.write(f"{lab} & {n} & " + " & ".join(f"{ap['per_model'][m]['per_size'][sz]:.3f}" for m in MODELS)
                + " \\\\\n")
    f.write("\\midrule\nAll, within-size $z$ & 298 & " +
            " & ".join(f"{ap['per_model'][m]['pooled_within_size_z']:.3f}" for m in MODELS) + " \\\\\n")
    f.write("All, raw pooled (not interpretable) & 298 & " +
            " & ".join(f"{ap['per_model'][m]['pooled_raw']:.3f}" for m in MODELS) + " \\\\\n")
mac("hkGapRaw", f"{ap['summary']['gap_pooled_raw']:.3f}")
mac("hkGapZ", f"{ap['summary']['gap_within_size_z']:.3f}")
mac("hkGapMean", f"{ap['summary']['gap_mean_per_size']:.3f}")

# ---------------------------------------------------------------- decomposition
SPECS = [("raw", "None (as originally reported)"), ("size", "Chord size"),
         ("size+reg", "Chord size + lowest note"), ("size+transp", "Chord size; models on transposed chords")]
COMPS = ["interference", "harmonicity", "familiarity"]


def pcell(p):
    return "$<$.001" if p < 0.001 else f"{p:.3f}".lstrip("0")


with open(TAB / "tab_decomp.tex", "w") as f:
    for ds, dlab in (("bowling2018", "Bowling et al.\\ (2018), $n=298$"),
                     ("johnson-laird2012", "Johnson-Laird et al.\\ (2012), $n=103$")):
        f.write(f"\\multicolumn{{5}}{{l}}{{\\textit{{{dlab}}}}} \\\\\n")
        for s, slab in SPECS:
            d = CD[ds][s]
            cells = [slab, f"{d['r2_full']:.3f}"]
            for c in COMPS:
                u = d["unique"][c]
                lo, hi = d["boot_ci_unique"][c]
                cells.append(f"{u['delta_r2']:.3f} [{lo:.3f}, {hi:.3f}] ({pcell(u['p'])})")
            f.write(" & ".join(cells) + " \\\\\n")
        if ds == "bowling2018":
            f.write("\\addlinespace\n")
            f.write("\\multicolumn{5}{l}{\\textit{Bowling et al.\\ (2018), interference and harmonicity rescored "
                    "at the presented just-intonation pitches}} \\\\\n")
            dj = J(RES / "decomposition_ji.json")
            for s_, slab in SPECS:
                d = dj[s_]
                cells = [slab, f"{d['r2_full']:.3f}"]
                for c in COMPS:
                    u = d["unique"][c]
                    lo, hi = d["boot_ci_unique"][c]
                    cells.append(f"{u['delta_r2']:.3f} [{lo:.3f}, {hi:.3f}] ({pcell(u['p'])})")
                f.write(" & ".join(cells) + " \\\\\n")
            f.write("\\addlinespace\n")
tagd = {"bowling2018": "B", "johnson-laird2012": "J"}
tags = {"raw": "Raw", "size": "Size", "size+reg": "Reg", "size+transp": "Tr"}
tagc = {"interference": "I", "harmonicity": "H", "familiarity": "F"}
for ds in tagd:
    for s in tags:
        for c in COMPS:
            u = CD[ds][s]["unique"][c]
            mac(f"dR{tagd[ds]}{tags[s]}{tagc[c]}", f"{u['delta_r2']:.3f}")
            mac(f"pR{tagd[ds]}{tags[s]}{tagc[c]}", "$p<.001$" if u["p"] < 0.001 else f"$p={pcell(u['p'])}$")
        mac(f"Rfull{tagd[ds]}{tags[s]}", f"{CD[ds][s]['r2_full']:.3f}")
    for s in tags:
        pr = CD[ds][s]["model_partial_r"]
        for m, t in (("hutch_78_roughness", "HK"), ("huron_94_dyadic", "Huron"), ("seth_93_roughness", "Seth"),
                     ("vass_01_roughness", "Vass"), ("har_19_corpus", "Corpus"), ("stolz_15_periodicity", "Stolz"),
                     ("har_19_composite", "Composite")):
            mac(f"r{t}{tagd[ds]}{tags[s]}", num(pr[m]))
            mac(f"rank{t}{tagd[ds]}{tags[s]}", str(CD[ds][s]["model_rank"][m]))
raw = CD["bowling2018"]["raw"]["unique"]
mac("ratioFamIntBRaw", f"{raw['familiarity']['delta_r2'] / raw['interference']['delta_r2']:.1f}")
rj = CD["johnson-laird2012"]["raw"]["unique"]
mac("ratioFamIntJRaw", f"{rj['familiarity']['delta_r2'] / rj['interference']['delta_r2']:.0f}")


# ================================================================ the tritone residual (new analyses)
MJ = J(RES / "marjieh_dyads.json")
MS_ = J(RES / "marjieh_summary.json")
BJI = J(RES / "bowling_ji.json")
DJI = J(RES / "decomposition_ji.json")
TGJ = J(RES / "tritone_gap.json")
EXP = MJ["experiments"]


def z(x, d=2):
    """Signed effect in within-listener SD units, e.g. +0.19."""
    return pct(x, d)


def zci(r, d=2):
    return f"{z(r['est'], d)} [{z(r['ci'][0], d)}, {z(r['ci'][1], d)}]"


def hk(r):
    """Random-effects record with the Hartung-Knapp interval as its CI (main text, A5).
    The DerSimonian-Laird z interval stays available as r['ci'] for the supplement."""
    return {**r, "ci": r["hksj_ci"], "dl_ci": r["ci"]}


def excl0(r):
    return r["ci"][0] > 0 or r["ci"][1] < 0


def zcib(r, d=2):
    """est [CI], bold when the CI excludes zero."""
    s = zci(r, d)
    return f"\\textbf{{{s}}}" if excl0(r) else s


EXP_ROWS = [("harm3", "Harmonic, 3\\,dB/oct"), ("eq5", "5 equal harmonics"), ("no3", "5 harmonics, no 3rd"),
            ("flute", "Flute"), ("guitar", "Guitar"), ("piano", "Piano"), ("pure", "Pure tones"),
            ("bonang", "Bonang (harmonic bass)"), ("str3", "Stretched (octave 2.1)"),
            ("comp3", "Compressed (octave 1.9)")]
KR_ROWS = [("harm3_kr", "Harmonic, 3\\,dB/oct"), ("str3_kr", "Stretched (octave 2.1)"),
           ("comp3_kr", "Compressed (octave 1.9)")]


def exp_table(fname, m):
    with open(TAB / fname, "w") as f:
        for block, rows in (("United States", EXP_ROWS), ("South Korea", KR_ROWS)):
            f.write(f"\\multicolumn{{6}}{{l}}{{\\textit{{{block}}}}} \\\\\n")
            for k, lab in rows:
                e = EXP[k]
                f.write(" & ".join([lab, str(e["n_participants"]), zcib(e[f"obs_{m}"]),
                                    zcib(e[f"HK_unexpl_{m}"]), zcib(e[f"HK+H_unexpl_{m}"]),
                                    zcib(e[f"Composite_unexpl_{m}"])]) + " \\\\\n")
            if block == "United States":
                f.write("\\addlinespace\n")


exp_table("tab_mj_gap.tex", "gap")
exp_table("tab_mj_dip.tex", "dip")

# per-experiment macros for the text
TAGS = {"harm3": "Harm", "eq5": "Eqfive", "no3": "Nothree", "flute": "Flute", "guitar": "Guitar",
        "piano": "Piano", "pure": "Pure", "bonang": "Bonang", "str3": "Str", "comp3": "Comp",
        "harm3_kr": "HarmKr", "str3_kr": "StrKr", "comp3_kr": "CompKr"}
FT = {"obs": "Obs", "HK": "HK", "HK+H": "HKH", "Composite": "Comp", "Seth": "Seth", "H": "H"}
for k, t in TAGS.items():
    e = EXP[k]
    mac(f"n{t}", str(e["n_participants"]))
    mac(f"ntrials{t}", f"{e['n_trials']:,}".replace(",", "{,}"))
    for m in ("gap", "dip"):
        M_ = m.capitalize()
        for fk, ft in FT.items():
            r = e[f"obs_{m}"] if fk == "obs" else e[f"{fk}_unexpl_{m}"]
            mac(f"{m}{ft}{t}", z(r["est"]))
            mac(f"{m}{ft}{t}CI", f"[{z(r['ci'][0])}, {z(r['ci'][1])}]")
    for fk, ft in (("HK", "HK"), ("HK+H", "HKH"), ("Composite", "Comp"), ("H", "H")):
        mac(f"share{ft}{t}", f"{100 * e[f'{fk}_share']['est']:.0f}")
        mac(f"rfit{ft}{t}", f"{e[f'{fk}_r2fit']['est']:.2f}")
    mac(f"rawsd{t}", f"{e['raw_sd_mean']:.2f}")
mac("nUSExp", str(len([k for k in EXP if EXP[k]["cohort"] == "US"])))
mac("nUSListeners", f"{sum(EXP[k]['n_participants'] for k in EXP if EXP[k]['cohort'] == 'US'):,}".replace(",", "{,}"))
mac("nKRListeners", f"{sum(EXP[k]['n_participants'] for k in EXP if EXP[k]['cohort'] != 'US')}")
mac("nAllTrials", f"{sum(EXP[k]['n_trials'] for k in EXP):,}".replace(",", "{,}"))
mac("rawSDmin", f"{min(EXP[k]['raw_sd_mean'] for k in EXP):.2f}")
mac("rawSDmax", f"{max(EXP[k]['raw_sd_mean'] for k in EXP):.2f}")
mac("nboot", f"{MJ['settings']['nboot']:,}".replace(",", "{,}"))
# model validation against Marjieh et al.'s published model profiles
V = MJ["validation"]
for mk, t in (("HK", "HK"), ("H", "H"), ("revHK", "RevHK"), ("Seth", "Seth")):
    rs = [v[mk]["r"] for v in V.values() if mk in v]
    mac(f"valMin{t}", f"{np.floor(min(rs) * 1e4) / 1e4:.4f}" if mk != "Seth" else f"{np.floor(min(rs) * 100) / 100:.2f}")
    if mk == "Seth":
        mac("valMaxSeth", f"{np.floor(max(rs) * 100) / 100:.2f}")
mac("nValExp", str(len(V)))

# random-effects pooling: the 'constant'
RE = MS_["random_effects"]
POOL_ROWS = [("obs", "Observed"), ("HK", "After roughness (H\\&K)"), ("Seth", "After roughness (Sethares)"),
             ("H", "After harmonicity alone"), ("HK+H", "After roughness + harmonicity"),
             ("HK+H_incon", "\\quad harmonicity template $i^{-0.75}$ (\\texttt{incon})"),
             ("revHK+H", "After revised H\\&K + harmonicity"),
             ("Composite", "After Marjieh et al.\\ composite$^{a}$")]


def rekey(fk, m):
    return f"obs_{m}" if fk == "obs" else f"{fk}_unexpl_{m}"


with open(TAB / "tab_pooled.tex", "w") as f:
    for fk, lab in POOL_ROWS:
        cells = [lab]
        for g, m in (("us_harmonic", "gap"), ("us8", "gap"), ("us_harmonic", "dip")):
            r = hk(RE[g][rekey(fk, m)])
            cells.append(f"{z(r['est'])} [{z(r['ci'][0])}, {z(r['ci'][1])}]")
            cells.append(f"[{z(r['pi'][0])}, {z(r['pi'][1])}]")
        f.write(" & ".join(cells) + " \\\\\n")
# supplement: the same rows with DerSimonian-Laird (z) and Hartung-Knapp intervals side by side (A5)
with open(TAB / "tab_supp_pooled_dl.tex", "w") as f:
    for fk, lab in POOL_ROWS:
        cells = [lab]
        for g, m in (("us_harmonic", "gap"), ("us8", "gap"), ("us_harmonic", "dip")):
            r = RE[g][rekey(fk, m)]
            cells += [z(r["est"]), f"[{z(r['ci'][0])}, {z(r['ci'][1])}]",
                      f"[{z(r['hksj_ci'][0])}, {z(r['hksj_ci'][1])}]"]
        f.write(" & ".join(cells) + " \\\\\n")
for m in ("gap", "dip"):
    r = hk(RE["us_harmonic"][f"HK+H_incon_unexpl_{m}"])
    mac(f"pool{m.capitalize()}InconHarm", f"{z(r['est'])} [{z(r['ci'][0])}, {z(r['ci'][1])}]")
GT = {"us_harmonic": "Harm", "us8": "Eight", "us_all": "All"}
for g, gt in GT.items():
    mac(f"k{gt}", str(RE[g]["obs_gap"]["k"]))
    for fk, ft in FT.items():
        for m in ("gap", "dip"):
            r = hk(RE[g][rekey(fk, m)])
            b = f"pool{m.capitalize()}{ft}{gt}"
            mac(b, z(r["est"]))
            mac(b + "CI", f"[{z(r['ci'][0])}, {z(r['ci'][1])}]")
            mac(b + "DLCI", f"[{z(r['dl_ci'][0])}, {z(r['dl_ci'][1])}]")
            mac(b + "PI", f"[{z(r['pi'][0])}, {z(r['pi'][1])}]")
            mac(b + "Tau", f"{r['tau']:.2f}")
            mac(b + "Itwo", f"{100 * r['I2']:.0f}")
# the constant in raw rating units (7-point scale): z times the mean within-listener raw SD
hs = [EXP[k]["raw_sd_mean"] for k in ("harm3", "eq5", "no3", "flute", "guitar", "piano")]
mac("rawSDharm", f"{np.mean(hs):.2f}")
mac("poolGapHKHHarmRaw", f"{RE['us_harmonic']['HK+H_unexpl_gap']['est'] * np.mean(hs):.2f}")
mac("poolGapObsHarmRaw", f"{RE['us_harmonic']['obs_gap']['est'] * np.mean(hs):.2f}")
# share of the pooled observed gap left after each model
for fk, ft in (("HK", "HK"), ("HK+H", "HKH"), ("Composite", "Comp")):
    mac(f"poolLeft{ft}", f"{100 * RE['us_harmonic'][rekey(fk, 'gap')]['est'] / RE['us_harmonic']['obs_gap']['est']:.0f}")

# specialness
TTO = MS_["tt_minus_mean_other"]
for g, gt in GT.items():
    for key, kt in (("obs_dip", "Obs"), ("HK_udip", "HK"), ("HK+H_udip", "HKH"), ("Composite_udip", "Comp")):
        r = hk(TTO[g][f"{key}_tt_minus_mean_other"])
        mac(f"tto{kt}{gt}", z(r["est"]))
        mac(f"tto{kt}{gt}CI", f"[{z(r['ci'][0])}, {z(r['ci'][1])}]")
        mac(f"tto{kt}{gt}DLCI", f"[{z(r['dl_ci'][0])}, {z(r['dl_ci'][1])}]")
NULL = MS_["null"]["us8"]
INAME = {1: "minor second", 2: "major second", 3: "minor third", 4: "major third", 5: "fourth", 6: "tritone",
         7: "fifth", 8: "minor sixth", 9: "major sixth", 10: "minor seventh", 11: "major seventh", 12: "octave",
         13: "minor ninth", 14: "major ninth"}
for key, kt in (("obs_dip_null", "Obs"), ("HK_udip_null", "HK"), ("HK+H_udip_null", "HKH"),
                ("Composite_udip_null", "Comp")):
    n = NULL[key]
    mac(f"ttRank{kt}", str(n["tt_rank"]))
    mac(f"nullFirst{kt}", INAME[n["ranking"][0]])
    mac(f"nullFirstVal{kt}", z(n["est"][n["interval"].index(n["ranking"][0])]))
    mac(f"nullSecond{kt}", INAME[n["ranking"][1]])
    mac(f"nullSecondVal{kt}", z(n["est"][n["interval"].index(n["ranking"][1])]))
    mac(f"nullTT{kt}", z(n["est"][n["interval"].index(6)]))
    mac(f"nullTTse{kt}", f"{n['se'][n['interval'].index(6)]:.2f}")

# profile structure under stretching
PS = MS_["profile_structure"]
for k in ("harm3", "str3", "comp3", "harm3_kr", "str3_kr"):
    mac(f"profSD{TAGS[k]}", f"{PS[k]['sd']:.2f}")
    mac(f"profRfit{TAGS[k]}", f"{PS[k]['HK+H_r2fit']:.2f}")

# the two-mechanism curve
CV = MS_["curve"]
LO_ = MJ["lodo"]["HK+H"]
mac("betaHKpooled", num(CV["beta_HK"], 2))
mac("betaHpooled", num(CV["beta_H"], 2))
mac("lambdaStd", f"{CV['lambda_std']:.2f}")
LODO_ROWS = [r for r in EXP_ROWS]
with open(TAB / "tab_lodo.tex", "w") as f:
    for k, lab in LODO_ROWS:
        v = LO_["held_out"][k]
        h = MJ["lodo"]["HK"]["held_out"][k]
        f.write(" & ".join([lab, f"{num(v['beta'][0], 2)}", f"{num(v['beta'][1], 2)}", zcib(h["unexpl_gap"]),
                            zcib(v["unexpl_gap"]), zcib(v["unexpl_dip"])]) + " \\\\\n")
held = {k: LO_["held_out"][k]["unexpl_gap"] for k, _ in LODO_ROWS}
mac("lodoPos", str(sum(v["est"] > 0 for v in held.values())))
mac("lodoExcl", str(sum(v["ci"][0] > 0 for v in held.values())))
mac("lodoN", str(len(held)))
harmk = ("harm3", "eq5", "no3", "flute", "guitar", "piano")
mac("lodoPosHarm", str(sum(held[k]["est"] > 0 for k in harmk)))
mac("lodoExclHarm", str(sum(held[k]["ci"][0] > 0 for k in harmk)))
for k in ("harm3", "pure", "guitar", "str3"):
    mac(f"lodoGap{TAGS[k]}", zci(held[k]))

# targeted tests
CO = MJ["cohort_diff"]
for k in ("harm3", "str3", "comp3"):
    for m, mt in (("obs_gap", "ObsGap"), ("obs_dip", "ObsDip"), ("HK+H_unexpl_gap", "HKHGap"),
                  ("HK+H_unexpl_dip", "HKHDip"), ("Composite_unexpl_gap", "CompGap")):
        mac(f"coh{mt}{TAGS[k]}", zci(CO[k][m]))
N3 = MJ["no3_minus_eq5"]
for m, mt in (("obs_gap", "ObsGap"), ("obs_dip", "ObsDip"), ("HK_pred_gap", "HKGap"), ("HK_pred_dip", "HKDip"),
              ("HK+H_pred_gap", "HKHGap"), ("HK+H_pred_dip", "HKHDip"), ("Composite_pred_dip", "CompDip")):
    mac(f"nthree{mt}", zci(N3[m]))
with open(TAB / "tab_tests.tex", "w") as f:
    f.write("\\multicolumn{5}{l}{\\textit{Korean minus US listeners, same stimuli}} \\\\\n")
    for k, lab in (("harm3", "Harmonic"), ("str3", "Stretched"), ("comp3", "Compressed")):
        c = CO[k]
        f.write(" & ".join([lab, zcib(c["obs_gap"]), zcib(c["HK+H_unexpl_gap"]), zcib(c["obs_dip"]),
                            zcib(c["HK+H_unexpl_dip"])]) + " \\\\\n")
    f.write("\\addlinespace\n\\multicolumn{5}{l}{\\textit{Musicians minus non-musicians (US)}} \\\\\n")
    for k, lab in EXP_ROWS:
        if "musicianship" not in EXP[k]:
            continue
        c = EXP[k]["musicianship"]
        f.write(" & ".join([lab + f" ({c['mus']['n_participants']}/{c['nonmus']['n_participants']})",
                            zcib(c["diff_obs_gap"]), zcib(c["diff_HK+H_unexpl_gap"]), zcib(c["diff_obs_dip"]),
                            zcib(c["diff_HK+H_unexpl_dip"])]) + " \\\\\n")
MU = {k: EXP[k]["musicianship"] for k, _ in EXP_ROWS if "musicianship" in EXP[k]}
mac("nMusExp", str(len(MU)))
mac("nMusExcl", str(sum(excl0(v["diff_obs_gap"]) for v in MU.values())))
mac("nMusExclHKH", str(sum(excl0(v["diff_HK+H_unexpl_gap"]) for v in MU.values())))
mac("nMusPos", str(sum(v["diff_obs_gap"]["est"] > 0 for v in MU.values())))
# musicianship pooled (fixed-effect over experiments; independent samples)
def fe(rs):
    est = np.array([r["est"] for r in rs])
    se = np.array([r["se"] for r in rs])
    w = 1 / se ** 2
    m = (w * est).sum() / w.sum()
    s = np.sqrt(1 / w.sum())
    return {"est": m, "ci": [m - 1.96 * s, m + 1.96 * s]}


MU8 = {k: v for k, v in MU.items() if k not in ("str3", "comp3")}
mac("nMusEight", str(len(MU8)))
for m, mt in (("diff_obs_gap", "Gap"), ("diff_HK+H_unexpl_gap", "GapHKH"), ("diff_obs_dip", "Dip"),
              ("diff_HK+H_unexpl_dip", "DipHKH")):
    # random effects over the eight unstretched US experiments (marjieh_summary.py)
    mac(f"musPool{mt}", zci(hk(MS_["musicianship_re"][m])))
    mac(f"musPool{mt}DL", zci(MS_["musicianship_re"][m]))
    mac(f"musPoolStr{mt}", zci(fe([MU[k][m] for k in ("str3", "comp3")])))
for grp, gt in (("mus", "Mus"), ("nonmus", "Non")):
    for m, mt in (("obs_gap", "Gap"), ("HK+H_unexpl_gap", "GapHKH"), ("obs_dip", "Dip"), ("HK+H_unexpl_dip", "DipHKH")):
        mac(f"grp{gt}{mt}", zci(hk(MS_["musicianship_re"][f"{grp}_{m}"])))

# stretched/compressed spectra analysed at the stretched scale's own tritone (stretched_scale.py)
SSC = J(RES / "stretched_scale.json")
for k in ("str3", "comp3", "harm3"):
    t = TAGS[k]
    mac(f"sscPos{t}", f"{SSC[k]['tt_position']:.2f}")
    for m, mt in (("obs_gap", "ObsGap"), ("obs_dip", "ObsDip"), ("HK+H_unexpl_gap", "HKHGap"),
                  ("HK+H_unexpl_dip", "HKHDip"), ("HK_unexpl_dip", "HKDip")):
        mac(f"ssc{mt}{t}", zci(SSC[k][m]))
        if f"{m}_minus600" in SSC[k]:
            mac(f"ssc{mt}{t}Diff", zci(SSC[k][f"{m}_minus600"]))
mac("poolDipHKHHarmAbs", f"{abs(RE['us_harmonic']['HK+H_unexpl_dip']['est']):.2f}")

RO = MJ["rolloff"]
with open(TAB / "tab_rolloff.tex", "w") as f:
    for lev, r in RO.items():
        f.write(" & ".join([f"{lev}", zcib(r["obs_gap"]), zci(r["HK_pred_gap"]), zcib(r["HK_unexpl_gap"]),
                            zcib(r["HK+H_unexpl_gap"]), zcib(r["obs_dip"]), zcib(r["HK+H_unexpl_dip"])]) + " \\\\\n")
for lev, r in RO.items():
    t = {"2": "Two", "7": "Seven", "12": "Twelve"}[lev]
    mac(f"roll{t}Gap", zci(r["obs_gap"]))
    mac(f"roll{t}HKpred", zci(r["HK_pred_gap"]))
    mac(f"roll{t}HKHGap", zci(r["HK+H_unexpl_gap"]))
mac("nRoll", str(MJ["rolloff_meta"]["n_participants"]))
mac("nRollTrials", f"{MJ['rolloff_meta']['n_trials']:,}".replace(",", "{,}"))
mac("nTrialsTotal", f"{sum(EXP[k]['n_trials'] for k in EXP) + MJ['rolloff_meta']['n_trials']:,}".replace(",", "{,}"))

# Bowling: equal-tempered vs just-intonation scoring
BM = [("roughness (H&K)", "Roughness (H\\&K)"), ("roughness (Sethares)", "Roughness (Sethares)"),
      ("H&K + harmonicity", "H\\&K + harmonicity"), ("H&K + familiarity", "H\\&K + familiarity"),
      ("H&K + harmonicity + familiarity", "H\\&K + harmonicity + familiarity"),
      ("H&K + periodicity (Stolzenburg)", "H\\&K + periodicity (Stolzenburg)")]
with open(TAB / "tab_bowling_ji.tex", "w") as f:
    f.write("Chord size only & -- & -- & " + " & ".join(
        f"{pct(BJI[s]['penalty']['models']['size only']['tritone_penalty'], 3)}" for s in ("ET", "JI")) + " \\\\\n")
    for k, lab in BM:
        cells = [lab]
        for s in ("ET", "JI"):
            cells.append(num(100 * BJI[s]['dyads']['models'][k]['share_of_gap_reproduced'], 0))
        for s in ("ET", "JI"):
            p = BJI[s]["penalty"]["models"][k]
            cells.append(f"{pct(p['tritone_penalty'], 3)} [{pct(p['tritone_ci'][0], 3)}, {pct(p['tritone_ci'][1], 3)}]")
        f.write(" & ".join(cells) + " \\\\\n")
for s in ("ET", "JI"):
    for k, t in (("roughness (H&K)", "HK"), ("H&K + harmonicity", "HKH"),
                 ("H&K + harmonicity + familiarity", "HKHF"), ("H&K + familiarity", "HKF"), ("size only", "Size")):
        p = BJI[s]["penalty"]["models"][k]
        mac(f"bpen{t}{s}", pct(p["tritone_penalty"], 3))
        mac(f"bpen{t}{s}CI", f"[{pct(p['tritone_ci'][0], 3)}, {pct(p['tritone_ci'][1], 3)}]")
        if k != "size only":
            mac(f"bshare{t}{s}", num(100 * BJI[s]['dyads']['models'][k]['share_of_gap_reproduced'], 0))
            mac(f"bresTT{t}{s}", num(BJI[s]["dyads"]["models"][k]["resid_tt"], 2))
            mac(f"bresRank{t}{s}", str(BJI[s]["dyads"]["models"][k]["tt_resid_rank"]))
mac("bObsGap", num(BJI["ET"]["dyads"]["observed_tt_minus_m6"], 2))
mac("bNTT", str(BJI["ET"]["penalty"]["n_with_tritone"]))
DV = BJI["dyad_values"]
for n, t in (("tritone", "TT"), ("minor_sixth", "Msix")):
    mac(f"jiCents{t}", f"{DV[n]['ji_cents']:.1f}")
    mac(f"harmET{t}", f"{DV[n]['harm_et']:.3f}")
    mac(f"harmJI{t}", f"{DV[n]['harm_ji']:.3f}")
mac("rEtJiH", f"{BJI['r_et_ji']['H']:.3f}")
mac("rEtJiI", f"{BJI['r_et_ji']['I']:.3f}")
# Johnson-Laird chord-level penalties (ET is correct there)
JLP = TGJ["penalty"]["johnson-laird2012"]["models"]
for k, t in (("size only", "Size"), ("roughness (H&K)", "HK"), ("H&K + harmonicity", "HKH"),
             ("H&K + harmonicity + familiarity", "HKHF")):
    mac(f"jpen{t}", pct(JLP[k]["tritone_penalty"], 2))
    mac(f"jpen{t}CI", f"[{pct(JLP[k]['tritone_ci'][0], 2)}, {pct(JLP[k]['tritone_ci'][1], 2)}]")
mac("jNTT", str(TGJ["penalty"]["johnson-laird2012"]["n_with_tritone"]))
mac("jRatingSD", f"{TGJ['penalty']['johnson-laird2012']['rating_sd']:.2f}")
mac("bRatingSD", f"{BJI['ET']['penalty']['rating_sd']:.2f}")

# decomposition with JI scoring
for s in ("raw", "size", "size+reg", "size+transp"):
    for c in COMPS:
        u = DJI[s]["unique"][c]
        mac(f"dRBji{tags[s]}{tagc[c]}", f"{u['delta_r2']:.3f}")
    mac(f"RfullBji{tags[s]}", f"{DJI[s]['r2_full']:.3f}")

# ---------------------------------------------------------------- third-factor screen (model_screen*.py)
SC1, SC2 = J(RES / "model_screen.json"), J(RES / "model_screen2.json")
SCREEN_ROWS = [
    ("\\multicolumn{8}{l}{\\textit{Existing models (weights refitted in each fold)}}", None, None),
    ("Roughness (\\HK)", SC1, "HK"),
    ("Harmonicity alone", SC1, "H"),
    ("\\HK{} + harmonicity ($\\sigma=6.83$ cents)", SC1, "HK+H"),
    ("Revised \\HK{} + harmonicity", SC1, "revHK+H"),
    ("Composite of Marjieh et al.", SC1, "Composite (fixed)"),
    ("\\multicolumn{8}{l}{\\textit{Pre-specified candidates}}", None, None),
    ("\\HK{} + harmonicity, tolerance $\\sigma$ chosen in fold", SC1, "HK+H(sigma)"),
    ("\\HK{} + harmonicity + root ambiguity", SC1, "HK+H+RA"),
    ("\\HK{} + harmonicity + periodicity", SC1, "HK+H+PER"),
    ("\\HK{} + periodicity", SC1, "HK+PER"),
    ("\\HK{} + harmonicity($\\sigma$) + ambiguity + periodicity", SC1, "HK+H(sigma)+RA+PER"),
    ("\\HK{} + harmonicity + harmonic entropy of $f_0$ ratio", SC1, "HK+H+HE (f0 foil)"),
    ("\\multicolumn{8}{l}{\\textit{Exploratory, after the screen}}", None, None),
    ("\\HK{} + harmonicity, $\\sigma$ grid extended to 60 cents", SC2, "HK+H(sigma, extended grid)"),
    ("\\HK{} + timbre-relative harmonicity", SC2, "HK+Hself"),
    ("\\HK{} + harmonicity(20) + timbre-relative(20)", SC2, "HK+H(20)+Hself(20)"),
]
# main text: existing models and pre-specified candidates; SI: the rows examined after the screen
_split = [i for i, r in enumerate(SCREEN_ROWS) if "Exploratory" in r[0]][0]
for fname, rows in (("tab_screen.tex", SCREEN_ROWS[:_split]), ("tab_supp_screen_expl.tex", SCREEN_ROWS[_split + 1:])):
  with open(TAB / fname, "w") as f:
    for lab, src, key in rows:
        if src is None:
            f.write(lab + " \\\\\n")
            continue
        s = src[key]
        f.write(" & ".join([lab, f"{s['r2_us_mean']:.3f}", f"{s['r2_harm6_mean']:.3f}", z(s["gap_harm6"]),
                            z(s["dip_harm6"]), z(s["tt_minus_other"]), z(s["str3_udip600"]),
                            z(s["str3_udip_ssc"])]) + " \\\\\n")
for s in (6.83, 10, 15, 20, 25, 30, 40, 60):
    t = {6.83: "Seven", 10: "Ten", 15: "Fifteen", 20: "Twenty", 25: "TwentyFive", 30: "Thirty", 40: "Forty",
         60: "Sixty"}[s]
    r = SC2[f"HK+H{s:g} (fixed sigma)"]
    mac(f"sigRtwo{t}", f"{r['r2_us_mean']:.3f}")
    mac(f"sigGap{t}", z(r["gap_harm6"]))
mac("scrChosenSigma", "20")
mac("scrHEpredDipSix", z(SC1["HK+H+HE (f0 foil)"]["str3_pdip600"]))
mac("scrHEresDipSix", z(SC1["HK+H+HE (f0 foil)"]["str3_udip600"]))
mac("scrHEonlyPredDipSix", z(SC1["HK+HE (f0 foil)"]["str3_pdip600"]))
mac("scrRtwoRA", f"{SC1['HK+H+RA']['r2_us_mean']:.3f}")
mac("scrRtwoPER", f"{SC1['HK+H+PER']['r2_us_mean']:.3f}")
mac("scrRtwoHE", f"{SC1['HK+H+HE (f0 foil)']['r2_us_mean']:.3f}")
mac("scrRtwoAll", f"{SC1['HK+H(sigma)+RA+PER']['r2_us_mean']:.3f}")
mac("scrGapAll", z(SC1["HK+H(sigma)+RA+PER"]["gap_harm6"]))
mac("scrBetaR", f"{SC1['HK+H(sigma)']['all_us_beta'][0]:.2f}".replace("-", "\\textminus{}"))
mac("scrBetaH", f"{SC1['HK+H(sigma)']['all_us_beta'][1]:.2f}")
for e, t in (("pure", "Pure"), ("flute", "Flute"), ("guitar", "Guitar"), ("piano", "Piano"), ("harm3", "Harm")):
    mac(f"scrGapTw{t}", z(SC1["HK+H(sigma)"]["per_exp"][e]["unexpl_gap"]))

MB = J(RES / "model_boot.json")
BT = {"HK": "HK", "HK+H": "HKH", "HK+H20": "HKHtw", "revHK+H": "Rev", "Composite": "Comp", "HK+Hself": "Self",
      "HK+H20+Hself20": "TwSelf"}
MT = {"r2_us_mean": "Rtwo", "r2_harm6_mean": "RtwoHarm", "gap_harm6": "Gap", "dip_harm6": "Dip",
      "tt_minus_other": "TTo", "str3_udip600": "StrSix", "str3_udip_ssc": "StrSsc", "str3_pdip_ssc": "StrPred",
      "gap_us8": "GapEight", "dip_us8": "DipEight"}
for m, mt in BT.items():
    for k, kt in MT.items():
        r = MB[m][k]
        d = 3 if k.startswith("r2") else 2
        mac(f"mb{kt}{mt}", f"{r['est']:.3f}" if k.startswith("r2") else z(r["est"]))
        mac(f"mb{kt}{mt}CI", f"[{num(r['ci'][0], d)}, {num(r['ci'][1], d)}]" if k.startswith("r2")
            else f"[{z(r['ci'][0])}, {z(r['ci'][1])}]")
DT = {"HK+H20 - HK+H": "TwVsHKH", "HK+H20 - Composite": "TwVsComp", "HK+H20 - revHK+H": "TwVsRev",
      "HK+H - Composite": "HKHVsComp", "HK+Hself - HK+H": "SelfVsHKH", "HK+H20+Hself20 - HK+H20": "TwSelfVsTw"}
for dk, dt in DT.items():
    for k, kt in MT.items():
        r = MB["_diffs"][dk][k]
        d = 3 if k.startswith("r2") else 2
        f_ = (lambda x: pct(x, 3)) if k.startswith("r2") else z
        mac(f"md{kt}{dt}", f"{f_(r['est'])} [{f_(r['ci'][0])}, {f_(r['ci'][1])}]")
mac("mbNboot", str(MB["_nboot"]))

CS = J(RES / "chords_sigma.json")
for c, ct in (("bowling_JI", "BJI"), ("bowling_ET", "BET"), ("johnson-laird_ET", "JL")):
    for s, st in (("sigma6.83", "Seven"), ("sigma20", "Twenty")):
        r = CS[c][s]
        mac(f"cs{ct}{st}", pct(r["tritone_penalty"], 2))
        mac(f"cs{ct}{st}CI", f"[{pct(r['ci'][0], 2)}, {pct(r['ci'][1], 2)}]")
        mac(f"cs{ct}{st}Rtwo", f"{r['r2']:.3f}")
    r = CS[c]["diff20_minus_6.83"]
    mac(f"cs{ct}Diff", f"{pct(r['est'], 2)} [{pct(r['ci'][0], 2)}, {pct(r['ci'][1], 2)}]")

# Hartung-Knapp-Sidik-Jonkman intervals for the headline pooled estimates (sensitivity to DL)
for k, t in (("obs_gap", "ObsGap"), ("obs_dip", "ObsDip"), ("HK+H_unexpl_gap", "HKHGap"), ("HK+H_unexpl_dip", "HKHDip")):
    r = RE["us_harmonic"][k]
    mac(f"hksj{t}Harm", f"[{z(r['hksj_ci'][0])}, {z(r['hksj_ci'][1])}]")

# robustness of the screen: smoothing width, ends of the gap, register (robustness_screen.py)
RB = J(RES / "robustness_screen.json")
for sd, t in (("0.1", "One"), ("0.2", "Two"), ("0.3", "Three")):
    k = RB["kernel"][sd]
    mac(f"rbBestSigma{t}", f"{k['best_sigma']:g}")
    sw = k["sweep"]
    mac(f"rbGain{t}", f"{max(v['r2_us_mean'] for v in sw.values()) - sw['6.83']['r2_us_mean']:.3f}")
    for s, st in (("6.83", "Seven"), ("10", "Ten"), ("15", "Fifteen"), ("20", "Twenty")):
        mac(f"rbRtwo{t}{st}", f"{sw[s]['r2_us_mean']:.3f}")
RBT = {"HK": "HK", "HK+H": "HKH", "HK+H20": "HKHtw", "Composite": "Comp", "HKreg+H": "RegHKH", "HKreg+H20": "RegHKHtw"}
for m, mt in RBT.items():
    for k, kt in (("r6_harm6", "RSix"), ("r8_harm6", "REight"), ("r6_us8", "RSixEight"), ("r8_us8", "REightEight"),
                  ("gap_harm6", "Gap"), ("dip_harm6", "Dip"), ("r2_us_mean", "Rtwo")):
        r = RB["models"][m][k]
        if k.startswith("r2"):
            mac(f"rb{kt}{mt}", f"{r['est']:.3f}")
        else:
            mac(f"rb{kt}{mt}", f"{z(r['est'])} [{z(r['ci'][0])}, {z(r['ci'][1])}]")
for d, dt in (("HKreg+H - HK+H", "RegVsHKH"), ("HKreg+H20 - HK+H20", "RegVsHKHtw")):
    for k, kt in (("gap_harm6", "Gap"), ("dip_harm6", "Dip"), ("r2_us_mean", "Rtwo")):
        r = RB["diffs"][d][k]
        mac(f"rbd{kt}{dt}", f"{pct(r['est'], 3)} [{pct(r['ci'][0], 3)}, {pct(r['ci'][1], 3)}]")

# held-out residual at every integer interval (robustness_screen.residual_profile)
RP = J(RES / "residual_profile.json")
INT_T = {1: "MinSec", 2: "MajSec", 3: "MinThird", 4: "MajThird", 5: "Fourth", 6: "Tritone", 7: "Fifth",
         8: "MinSixth", 9: "MajSixth", 10: "MinSev", 11: "MajSev", 12: "Octave"}
for m, mt in (("HK+H20", "HKHtw"), ("HK", "HK")):
    for grp, gt in (("harm6", "Harm"), ("pure", "Pure")):
        for i, it in INT_T.items():
            mac(f"rp{mt}{gt}{it}", z(RP[m][grp][i - 1]))

# ---------------------------------------------------------------- noise ceiling and sharpness (ceiling.py, sharpness.py)
SH = J(RES / "sharpness.json")
CE = SH["ceiling"]
US_E = [k for k, v in CE.items() if v["cohort"] == "US"]
KR_E = [k for k, v in CE.items() if v["cohort"] == "KR"]
ceil_us = float(np.mean([CE[e]["ceiling"] for e in US_E]))
mac("ceilUS", f"{ceil_us:.2f}")
mac("ceilKR", f"{np.mean([CE[e]['ceiling'] for e in KR_E]):.2f}")
mac("ceilMin", f"{min(CE[e]['ceiling'] for e in CE):.2f}")
mac("ceilMax", f"{max(CE[e]['ceiling'] for e in CE):.2f}")
mac("ceilHarmThree", f"{CE['harm3']['ceiling']:.2f}")
INS = SH["insample"]
for ks, kt in (("HK+H20", "Base"), ("HK+H20+X1", "X"), ("HK+H20+SHARP", "Sharp")):
    mac(f"ins{kt}HarmThree", f"{INS[ks]['harm3'][0]:.2f}")
    mac(f"ins{kt}Bonang", f"{INS[ks]['bonang'][0]:.2f}")
    mac(f"ins{kt}US", f"{np.mean([INS[ks][e][0] for e in US_E]):.2f}")
    mac(f"ins{kt}KR", f"{np.mean([INS[ks][e][0] for e in KR_E]):.2f}")
    mac(f"ins{kt}KRmin", f"{min(INS[ks][e][0] for e in KR_E):.2f}")
    mac(f"ins{kt}KRmax", f"{max(INS[ks][e][0] for e in KR_E):.2f}")
# in-sample interval-size slope per octave (HK+H20+X1), US synthetic vs Korean
slope = {e: 12 * INS["HK+H20+X1"][e][1][2] for e in CE}
for e, et in (("harm3", "HarmThree"), ("pure", "Pure"), ("bonang", "Bonang"), ("flute", "Flute"),
              ("eq5", "EqFive"), ("piano", "Piano"), ("guitar", "Guitar"), ("harm3_kr", "HarmThreeKr")):
    mac(f"slope{et}", z(slope[e]))
mac("slopeKrMin", z(max(slope[e] for e in KR_E)))
mac("slopeKrMax", z(min(slope[e] for e in KR_E)))
SC = SH["screen"]
SCT = {"HK+H20": "Base", "HK+H20+X1": "X", "HK+H20+CEN": "Cen", "HK+H20+SHARP": "Sharp", "HK+H20+SHARP+X1": "SharpX",
       "HK+H+SHARP": "HKHSharp", "revHK+H": "Rev", "revHK+H+SHARP": "RevSharp", "Composite (fixed)": "Comp",
       "Composite+SHARP": "CompSharp"}
for m, mt in SCT.items():
    s = SC[m]
    mac(f"sh{mt}RtwoUS", f"{s['r2_us_mean']:.3f}")
    mac(f"sh{mt}RtwoHarm", f"{s['r2_harm6_mean']:.3f}")
    mac(f"sh{mt}RtwoKR", num(s["r2_kr_mean"]))
    mac(f"sh{mt}Gap", z(s["gap_harm6"]))
    mac(f"sh{mt}Dip", z(s["dip_harm6"]))
    mac(f"sh{mt}Frac", f"{100 * s['r2_us_mean'] / ceil_us:.0f}")
    for e, et in (("harm3", "HarmThree"), ("flute", "Flute"), ("str3", "Str"), ("bonang", "Bonang")):
        mac(f"sh{mt}Rtwo{et}", f"{s['per_exp_r2'][e]:.2f}")
b = SC["HK+H20+SHARP"]["all_us_beta"]
mac("shBetaR", num(b[0], 2))
mac("shBetaH", num(b[1], 2))
mac("shBetaS", num(b[2], 3))
BT = SH["boot"]["HK+H20+SHARP"]
mac("shbGap", zci(BT["gap_harm6"]))
mac("shbDip", zci(BT["dip_harm6"]))
for dname, dt in (("HK+H20+SHARP - HK+H20", "VsBase"), ("Composite+SHARP - Composite (fixed)", "CompVsComp"),
                  ("HK+H20+SHARP - Composite (fixed)", "VsComp")):
    D = SH["boot_diffs"][dname]
    for k, kt in (("r2_us_mean", "RtwoUS"), ("r2_harm6_mean", "RtwoHarm"), ("r2_kr_mean", "RtwoKR"),
                  ("gap_harm6", "Gap"), ("dip_harm6", "Dip")):
        r = D[k]
        mac(f"shd{kt}{dt}", f"{pct(r['est'], 3)} [{pct(r['ci'][0], 3)}, {pct(r['ci'][1], 3)}]")
RO = SH["rolloff"]
for m, mt in (("HK+H", "Base"), ("HK+H+X1", "X"), ("HK+H+SHARP", "Sharp"), ("HK+H+X1+SHARP", "XSharp")):
    mac(f"ro{mt}Rtwo", f"{RO[m]['r2_mean']['est']:.2f}")
mac("roXSharpBetaX", f"{num(12 * RO['HK+H+X1+SHARP']['beta'][2]['est'], 2)} [{num(12 * RO['HK+H+X1+SHARP']['beta'][2]['ci'][0], 2)}, {num(12 * RO['HK+H+X1+SHARP']['beta'][2]['ci'][1], 2)}]")
mac("roXBetaX", f"{num(12 * RO['HK+H+X1']['beta'][2]['est'], 2)} [{num(12 * RO['HK+H+X1']['beta'][2]['ci'][0], 2)}, {num(12 * RO['HK+H+X1']['beta'][2]['ci'][1], 2)}]")
mac("roXSharpBetaS", f"{num(RO['HK+H+X1+SHARP']['beta'][3]['est'], 2)} [{num(RO['HK+H+X1+SHARP']['beta'][3]['ci'][0], 2)}, {num(RO['HK+H+X1+SHARP']['beta'][3]['ci'][1], 2)}]")
r = RO["SHARP_minus_X1_r2"]
mac("roSharpMinusX", f"{pct(r['est'], 3)} [{pct(r['ci'][0], 3)}, {pct(r['ci'][1], 3)}]")
for i, t in enumerate(("One", "Four", "Seven", "Ten", "Thirteen")):
    mac(f"roSlope{t}", zci(RO["slice_x_slope_per_octave"][i]))
mac("roSlopeDiff", zci(RO["slope_1_minus_13"]))
for grp, gt in (("harm6", "Harm"), ("pure", "Pure")):
    for i, it in INT_T.items():
        mac(f"rpS{gt}{it}", z(SH["resid_profile"]["HK+H20+SHARP"][grp][i - 1]))
S2J = J(RES / "model_screen2.json")
tol_gain = S2J["HK+H20 (fixed sigma)"]["r2_us_mean"] - S2J["HK+H"]["r2_us_mean"]
mac("shTolGain", f"{tol_gain:+.3f}")
mac("shRatioTol", f"{(SC['HK+H20+SHARP']['r2_us_mean'] - SC['HK+H20']['r2_us_mean']) / tol_gain:.0f}")


# ---------------------------------------------------------------- equal-tempered grid bonus (grid_bonus.py)
GB = J(RES / "grid_bonus.json")["boot"]
for m, mt in (("HK+H20", "Base"), ("HK+H20+SHARP", "Sharp")):
    for k, kt in (("harm6_other", "HarmOther"), ("harm6_tt", "HarmTT"), ("harm6_other_minus_tt", "HarmDiff"),
                  ("pure_other", "Pure"), ("bonang_other", "Bonang"), ("str3_other_physical", "StrPhys"),
                  ("str3_other_ownscale", "StrOwn"), ("comp3_other_physical", "CompPhys"),
                  ("comp3_other_ownscale", "CompOwn"), ("harm6_minus_pure", "HarmMinusPure"),
                  ("harm6_minus_str3", "HarmMinusStr")):
        mac(f"gb{mt}{kt}", zci(GB[f"{m}|{k}"]))
for k, kt in (("gap_bonus_part", "GapBonus"), ("gap_base_part", "GapBase"), ("q_low_mean", "QLow"),
              ("q_high_mean", "QHigh"), ("q_step", "QStep"), ("q6_minus_low", "QSixLow"),
              ("tt_minus_m2m7", "TTvsMM"), ("m2m7_minus_rest", "MMvsRest"), ("harm6_b2", "BMajSec"),
              ("harm6_b10", "BMinSev"), ("harm6_q6", "QTritone"), ("harm6_q8", "QMinSixth")):
    mac(f"gbd{kt}", zci(GB[f"HK+H20+SHARP|{k}"]))
_Q = J(RES / "grid_bonus.json")["est"]["HK+H20+SHARP"]["harm6_qprofile"]
for c, ct in ((12, "Twelve"), (13, "Thirteen"), (14, "Fourteen")):
    mac(f"gbdQ{ct}", z(_Q[c - 1]))
_B = J(RES / "grid_bonus.json")["est"]["HK+H20+SHARP"]["harm6_profile"]
_R = [q + b for q, b in zip(_Q, _B)]
for c, ct in ((2, "MajSec"), (6, "Tritone"), (10, "MinSev")):
    mac(f"gbdDip{ct}", z((_R[c - 2] + _R[c]) / 2 - _R[c - 1]))
GBP = J(RES / "grid_bonus.json")["est"]["HK+H20+SHARP"]["harm6_profile"]
for i, it in INT_T.items():
    if i <= 11:
        mac(f"gbp{it}", z(GBP[i - 1]))

# ---------------------------------------------------------------- pre-registered triad test (confirm_triads.py)
CT = J(RES / "confirm_triads.json")
t = CT["tests"]
mac("ctBS", f"{pct(t['T1_bS']['est'], 3)} [{pct(t['T1_bS']['ci'][0], 3)}, {pct(t['T1_bS']['ci'][1], 3)}]")
mac("ctLooDiff", f"{pct(t['T2_diff']['est'], 3)} [{pct(t['T2_diff']['ci'][0], 3)}, {pct(t['T2_diff']['ci'][1], 3)}]")
mac("ctLooTwo", f"{t['T2_loo2']['est']:.3f}")
mac("ctLooThree", f"{t['T2_loo3']['est']:.3f}")
mac("ctRDiff", f"{pct(t['T3_diff']['est'], 3)} [{pct(t['T3_diff']['ci'][0], 3)}, {pct(t['T3_diff']['ci'][1], 3)}]")
mac("ctRTwo", f"{t['T3_r2']['est']:.3f}")
mac("ctRThree", f"{t['T3_r3']['est']:.3f}")
mac("ctNChords", str(CT["n_chords"]))
mac("ctNPart", str(CT["n_participants"]))

# ---------------------------------------------------------------- roll-off slope implied by sharpness
RI = J(RES / "sharpness_rolloff_implied.json")
for i, tt in enumerate(("One", "Four", "Seven", "Ten", "Thirteen")):
    mac(f"roImp{tt}", z(RI["implied"][i]))
mac("roImpShare", f"{100 * (RI['implied'][0] - RI['implied'][-1]) / (RI['observed'][0] - RI['observed'][-1]):.0f}")

# ---------------------------------------------------------------- A8: ET vs JI (et_ji_positions.py; exploratory)
EJ = J(RES / "et_ji_positions.json")
EJR = J(RES / "et_ji_rule.json")
EJW = "sd0.1"


def cents(x):
    return f"{100 * x:.0f}"


def cci(r, key="ci"):
    return f"{cents(r['est'])} [{cents(r[key][0])}, {cents(r[key][1])}]"


def dcents(r, key="hk_ci"):
    f = lambda x: pct(100 * x, 0)
    return f"{f(r['est'])} [{f(r[key][0])}, {f(r[key][1])}]"


def ejloc(e, c, loc, w=EJW):
    """Location [bootstrap CI] in cents; dagger when the point estimate is on the window edge."""
    r = EJ["boot"][f"{w}|{e}|{c}|loc_{loc}"]
    edge = EJ["est"][w][e][c][f"edge_{loc}"] > 0
    return cci(r) + ("$^\\dagger$" if edge else "")


EJ_ROWS = [("harm3", "Harmonic, 3\\,dB/oct"), ("eq5", "5 equal harmonics"), ("no3", "5 harmonics, no 3rd"),
           ("flute", "Flute"), ("guitar", "Guitar"), ("piano", "Piano"), ("pure", "Pure tones"),
           ("str3", "Stretched (2.1)"), ("comp3", "Compressed (1.9)")]
EJ_LOCS = ("m6", "tt", "p5", "M3", "M6")
with open(TAB / "tab_supp_etji_loc.tex", "w") as f:
    for c, clab in (("listener", "Listeners"), ("HK+H", "H\\&K + harmonicity (6.83 cents)"),
                    ("HK+H20", "H\\&K + harmonicity (20 cents)"), ("Composite", "Composite")):
        f.write(f"\\multicolumn{{6}}{{l}}{{\\emph{{{clab}}}}} \\\\\n")
        for e, lab in EJ_ROWS:
            f.write(" & ".join([lab] + [ejloc(e, c, l) for l in EJ_LOCS]) + " \\\\\n")
        cells = ["Pooled, 6 harmonic (RE)"]
        for l in EJ_LOCS:
            edge = any(EJ["est"][EJW][e][c][f"edge_{l}"] > 0 for e in EJ["harm6"])
            cells.append("edge" if (edge and c != "listener") else cci(EJ["pooled"][EJW][c][f"loc_{l}"], "hk_ci"))
        f.write(" & ".join(cells) + " \\\\\n")
        if c != "Composite":
            f.write("\\addlinespace\n")
with open(TAB / "tab_supp_etji_gap.tex", "w") as f:
    for w, wl in (("sd0.1", "0.1"), ("sd0.05", "0.05"), ("sd0.2", "0.2")):
        for c, clab in (("listener", "Observed"), ("HK", "Left by H\\&K"), ("HK+H", "Left by H\\&K + H (6.83)"),
                        ("HK+H20", "Left by H\\&K + H (20)"), ("Composite", "Left by composite")):
            g = "gap" if c == "listener" else "ugap"
            P = EJ["pooled"][w][c]
            cells = [wl if c == "listener" else "", clab]
            for p in ("ET", "JIa", "JIb", "JIa_minus_ET", "JIb_minus_ET"):
                r = P[f"{g}_{p}"]
                cells.append(f"{z(r['est'])} [{z(r['hk_ci'][0])}, {z(r['hk_ci'][1])}]")
            f.write(" & ".join(cells) + " \\\\\n")
        if w != "sd0.2":
            f.write("\\addlinespace\n")
P = EJ["pooled"][EJW]
for l, lt in (("m6", "Msix"), ("tt", "TT"), ("p5", "Pfive"), ("M3", "MThree"), ("M6", "MajSix")):
    mac(f"ejLoc{lt}", cci(P["listener"][f"loc_{l}"], "hk_ci"))
    mac(f"ejModel{lt}", cci(P["HK+H"][f"loc_{l}"], "hk_ci"))
    mac(f"ejDiff{lt}", dcents(P["HK+H"][f"locdiff_{l}"]))
mac("ejDiffMsixNarrow", dcents(EJ["pooled"]["sd0.05"]["HK+H"]["locdiff_m6"]))
mac("ejLocMsixNarrow", cci(EJ["pooled"]["sd0.05"]["listener"]["loc_m6"], "hk_ci"))
for p, pt in (("ET", "ET"), ("JIa", "JIa"), ("JIb", "JIb"), ("JIa_minus_ET", "JIaDiff"), ("JIb_minus_ET", "JIbDiff")):
    r = P["HK+H20"][f"ugap_{p}"]
    mac(f"ejGap{pt}", f"{z(r['est'])} [{z(r['hk_ci'][0])}, {z(r['hk_ci'][1])}]")
    r = P["listener"][f"gap_{p}"]
    mac(f"ejObsGap{pt}", f"{z(r['est'])} [{z(r['hk_ci'][0])}, {z(r['hk_ci'][1])}]")
REL = EJ["reliability"]
mac("ejCeilOneMin", f"{min(REL['sd0.1'][e]['ceiling'] for e in EJ['harm6']):.2f}")
mac("ejCeilHalfMin", f"{min(REL['sd0.05'][e]['ceiling'] for e in EJ['harm6']):.2f}")
mac("ejCeilHalfStr", f"{REL['sd0.05']['str3']['ceiling']:.2f}")
mac("ejCeilHalfComp", f"{REL['sd0.05']['comp3']['ceiling']:.2f}")
mac("ejRule", str(EJR["rule"]))
mac("ejNboot", str(EJ["_nboot"]))

# ---------------------------------------------------------------- revision analyses A1-A6 and residual profile
def zhk(r, key="hk_ci"):
    return f"{z(r['est'])} [{z(r[key][0])}, {z(r[key][1])}]"


def zhkb(r, key="hk_ci"):
    s = zhk(r, key)
    return f"\\textbf{{{s}}}" if (r[key][0] > 0 or r[key][1] < 0) else s


def pfmt(p):
    return "<.001" if p < 0.001 else f"{p:.3f}".lstrip("0")


HARM6 = ["harm3", "eq5", "no3", "flute", "guitar", "piano"]
H6LAB = dict(EXP_ROWS)

# A6: primary estimand and Holm family (multiplicity.py)
MP = J(RES / "multiplicity.json")
pr = MP["primary"]
mac("primGap", z(pr["est"]))
mac("primGapCI", f"[{z(pr['hk_ci'][0])}, {z(pr['hk_ci'][1])}]")
mac("primGapDLCI", f"[{z(pr['dl_ci'][0])}, {z(pr['dl_ci'][1])}]")
mac("primGapPI", f"[{z(pr['pi'][0])}, {z(pr['pi'][1])}]")
mac("primTau", f"{pr['tau']:.2f}")
mac("primItwo", f"{100 * pr['I2']:.0f}")
mac("primK", str(pr["k"]))
for e in HARM6:
    mac(f"prim{TAGS[e]}", z(pr["per_exp"][e]["est"]))
TT_ = MP["targeted_tests"]
HOLM_ROWS = [("culture", "Culture: Korean $-$ US, gap after rough.\\ + harm.\\ (harmonic tones)", "Culture"),
             ("musicianship", "Training: musicians $-$ non-musicians, gap after rough.\\ + harm.\\ (RE, $k=8$)", "Mus"),
             ("third_harmonic", "Third harmonic: observed $-$ roughness-predicted change in the dip", "Third"),
             ("rolloff", "Roll-off: observed $-$ roughness-predicted change in the gap, 2 vs 12\\,dB/oct", "Roll")]
with open(TAB / "tab_holm.tex", "w") as f:
    for k, lab, t in HOLM_ROWS:
        v = TT_[k]
        f.write(" & ".join([lab, z(v["est"]), f"{v['se']:.2f}", pfmt(v["p"]), pfmt(v["p_holm"])]) + " \\\\\n")
        mac(f"holmP{t}", pfmt(v["p"]))
        mac(f"holmPadj{t}", pfmt(v["p_holm"]))
        mac(f"holmEst{t}", z(v["est"]))
mac("holmNsurvive", str(sum(v["survives"] for v in TT_.values())))

# which end of the gap: residual profile (residual_profile.py)
RPR = J(RES / "residual_profile_re.json")
RPP = RPR["pooled_harm6"]
RPM = {"HK": "HK", "HK+H": "HKH", "HK+H20": "Tw", "HK+H20+X": "TwX"}
for m, mt in RPM.items():
    for s, st in (("r6", "Six"), ("r8", "Eight"), ("gap", "Gap"), ("tt_minus_other", "TTOther"),
                  ("m6_minus_cons", "MsixCons")):
        mac(f"rpf{st}{mt}", zhk(RPP[m][s]))
    for c, ct in ((3, "MinThird"), (4, "MajThird"), (5, "Fourth"), (7, "Fifth"), (9, "MajSixth"), (2, "MajSec"),
                  (10, "MinSev"), (1, "MinSec"), (11, "MajSev"), (12, "Octave")):
        mac(f"rpf{ct}{mt}", z(RPP[m][f"r{c}"]["est"]))
    mac(f"rpfPure{mt}Six", z(RPR["boot"][f"{m}|pure|r6"]["est"]))
    mac(f"rpfPure{mt}Eight", z(RPR["boot"][f"{m}|pure|r8"]["est"]))
with open(TAB / "tab_resprofile.tex", "w") as f:
    for m, lab in (("HK", "H\\&K"), ("HK+H", "H\\&K + H (6.83)"), ("HK+H20", "H\\&K + H (20)"),
                   ("HK+H20+X", "H\\&K + H (20) + size")):
        f.write(" & ".join([lab] + [z(RPP[m][f"r{c}"]["est"]) + ("$^*$" if (RPP[m][f"r{c}"]["hk_ci"][0] > 0 or
                                                                            RPP[m][f"r{c}"]["hk_ci"][1] < 0) else "")
                                    for c in range(1, 15)]) + " \\\\\n")
with open(TAB / "tab_ends.tex", "w") as f:
    for e in HARM6 + ["pure"]:
        B_ = RPR["boot"]
        cells = [H6LAB.get(e, "Pure tones")]
        for m in ("HK", "HK+H20"):
            cells += [zcib(B_[f"{m}|{e}|r6"]), zcib(B_[f"{m}|{e}|r8"])]
        cells.append(zcib(B_[f"HK+H20|{e}|gap"]))
        f.write(" & ".join(cells) + " \\\\\n")
    f.write("\\midrule\n")
    cells = [f"Pooled, harmonic (RE, $k={len(HARM6)}$)"]
    for m in ("HK", "HK+H20"):
        cells += [zhkb(RPP[m]["r6"]), zhkb(RPP[m]["r8"])]
    cells.append(zhkb(RPP["HK+H20"]["gap"]))
    f.write(" & ".join(cells) + " \\\\\n")

# A1/A2: the decomposition, per spectrum and pooled (decomposition_models.py, decomposition_re.py)
DM_ = J(RES / "decomposition_models.json")["boot"]
DR = J(RES / "decomposition_re.json")["pooled"]
DSTATS = [("gap_resid", "Gap"), ("gap_bonus_part", "BonusPart"), ("gap_base_part", "BasePart"), ("step_ind11", "Step"),
          ("step_inc", "StepInc"), ("bonus_other", "BonusOther"), ("bonus_tt", "BonusTT"),
          ("other_minus_tt", "OtherMinusTT"), ("tt_minus_m2m7", "TTvsMM"), ("m2m7_minus_rest", "MMvsRest")]
for m in ("M1", "M2", "M3"):
    for s, st in DSTATS:
        mac(f"dec{m[1]}{st}".replace("1", "One").replace("2", "Two").replace("3", "Three"), zhk(DR[m][s]))
        mac(f"dec{m[1]}{st}Fixed".replace("1", "One").replace("2", "Two").replace("3", "Three"),
            zci(DM_[f"default|{m}|{s}"]))
    mac(f"dec{m[1]}StepItwo".replace("1", "One").replace("2", "Two").replace("3", "Three"),
        f"{100 * DR[m]['step_ind11']['I2']:.0f}")
for c, ct in ((1, "MinSec"), (2, "MajSec"), (3, "MinThird"), (4, "MajThird"), (5, "Fourth"), (6, "Tritone"),
              (7, "Fifth"), (8, "MinSixth"), (9, "MajSixth"), (10, "MinSev"), (11, "MajSev")):
    mac(f"decTwoB{ct}", z(DM_[f"default|M2|b{c}"]["est"]))  # per-interval bonus, plain mean (point estimates)
for e in HARM6:
    for s, st in (("step_ind11", "Step"), ("gap_bonus_part", "BonusPart"), ("gap_base_part", "BasePart"),
                  ("other_minus_tt", "OtherMinusTT")):
        mac(f"decTwo{st}{TAGS[e]}", zci(DM_[f"default|M2|exp|{e}|{s}"]))
with open(TAB / "tab_decomp_spectra.tex", "w") as f:
    for e in HARM6:
        f.write(" & ".join([H6LAB[e]] + [zcib(DM_[f"default|M2|exp|{e}|{s}"]) for s in
                                          ("gap_resid", "gap_bonus_part", "gap_base_part", "step_ind11",
                                           "other_minus_tt")]) + " \\\\\n")
    f.write("\\midrule\n")
    f.write(" & ".join(["Pooled (RE, Hartung--Knapp)"] + [zhkb(DR["M2"][s]) for s in
                                                          ("gap_resid", "gap_bonus_part", "gap_base_part", "step_ind11",
                                                           "other_minus_tt")]) + " \\\\\n")
    f.write(" & ".join(["$I^2$ (\\%)"] + [f"{100 * DR['M2'][s]['I2']:.0f}" for s in
                                        ("gap_resid", "gap_bonus_part", "gap_base_part", "step_ind11",
                                         "other_minus_tt")]) + " \\\\\n")
with open(TAB / "tab_supp_decomp_models.tex", "w") as f:
    for s, lab in (("gap_resid", "Gap left, $r(8)-r(6)$"), ("gap_bonus_part", "Bonus part, $b(8)-b(6)$"),
                   ("gap_base_part", "Baseline part, $q(8)-q(6)$"), ("step_ind11", "Step (indicator, 1--11)"),
                   ("step_ind14", "Step (indicator, 1--14)"), ("step_inc", "Step (7$\\to$8 increment)"),
                   ("bonus_other", "Mean bonus, other intervals"), ("bonus_tt", "Tritone bonus"),
                   ("other_minus_tt", "Other $-$ tritone bonus"), ("tt_minus_m2m7", "Tritone $-$ M2/m7 bonus"),
                   ("m2m7_minus_rest", "M2/m7 $-$ other bonus")):
        cells = [lab]
        for m in ("M1", "M2", "M3"):
            cells += [zcib(DM_[f"default|{m}|{s}"]), zhkb(DR[m][s])]
        f.write(" & ".join(cells) + " \\\\\n")
SR = J(RES / "step_robustness.json")["boot"]
with open(TAB / "tab_supp_step.tex", "w") as f:
    for st, lab in (("default", "Default (SD 0.2, $\\pm0.6$)"), ("sd0.1", "Kernel SD 0.1"), ("sd0.3", "Kernel SD 0.3"),
                    ("win0.4", "Exclusion $\\pm0.4$"), ("win0.8", "Exclusion $\\pm0.8$"),
                    ("bin0.25", "Raw 0.25-semitone bins")):
        cells = [lab] + [zcib(SR[f"{st}|M2|{s}"]) for s in ("step_ind11", "step_inc", "gap_bonus_part", "gap_base_part")]
        cells += [z(SR[f"{st}|M2|exp|{e}|step_ind11"]["est"]) for e in HARM6]
        f.write(" & ".join(cells) + " \\\\\n")
for st, t in (("sd0.1", "SdOne"), ("sd0.3", "SdThree"), ("win0.4", "WinFour"), ("win0.8", "WinEight"), ("bin0.25", "Bin")):
    mac(f"srStep{t}", zci(SR[f"{st}|M2|step_ind11"]))

# A3: Korean held-out (korean_heldout.py)
KH = J(RES / "korean_heldout.json")
KRX = ["harm3_kr", "str3_kr", "comp3_kr"]
with open(TAB / "tab_supp_korean.tex", "w") as f:
    for m, lab in (("M1", "H\\&K + H (20)"), ("M2", "+ interval size"), ("M3", "+ sharpness proxy"),
                   ("Composite", "Composite (Marjieh et al.)")):
        f.write(" & ".join([lab] + [zci(KH["boot"][f"{m}|{e}"]).replace("+", "") if False else
                                    f"{pct(KH['boot'][f'{m}|{e}']['est'], 2)} [{pct(KH['boot'][f'{m}|{e}']['ci'][0], 2)}, {pct(KH['boot'][f'{m}|{e}']['ci'][1], 2)}]"
                                    for e in KRX + ["mean"]]) + " \\\\\n")
    f.write(" & ".join(["Split-half ceiling"] + [f"{KH['ceiling'][e]:.2f}" for e in KRX] + [""]) + " \\\\\n")
for m, mt in (("M1", "Tw"), ("M2", "Size"), ("M3", "Sharp"), ("Composite", "Comp")):
    r = KH["boot"][f"{m}|mean"]
    mac(f"krHo{mt}", f"{pct(r['est'], 2)} [{pct(r['ci'][0], 2)}, {pct(r['ci'][1], 2)}]")
mac("krCeilMin", f"{min(KH['ceiling'].values()):.3f}")
mac("krCeilMax", f"{max(KH['ceiling'].values()):.3f}")

# A4: bonus at the stretched scale's own intervals (stretched_bonus.py)
SBN = J(RES / "stretched_bonus.json")
with open(TAB / "tab_supp_strbonus.tex", "w") as f:
    for e, lab in (("harm3", "Harmonic (US)"), ("str3", "Stretched 2.1 (US)"), ("comp3", "Compressed 1.9 (US)"),
                   ("harm3_kr", "Harmonic (Korea)"), ("str3_kr", "Stretched 2.1 (Korea)"),
                   ("comp3_kr", "Compressed 1.9 (Korea)")):
        b = SBN["boot"]
        phys = zci(b[f"{e}|other_physical"]) if f"{e}|other_physical" in b else "--"
        f.write(" & ".join([lab, zcib(b[f"{e}|other"]), phys, f"{SBN['mde_other'][e]:.2f}"]) + " \\\\\n")
for e in ("harm3", "str3", "comp3", "harm3_kr", "str3_kr", "comp3_kr"):
    t = TAGS.get(e, e)
    mac(f"sbOwn{t}", zci(SBN["boot"][f"{e}|other"]))
    mac(f"sbMde{t}", f"{SBN['mde_other'][e]:.2f}")

# A5: interval flips (re_intervals.py)
RI_ = J(RES / "re_intervals.json")
mac("reNest", str(RI_["n_estimates"]))
mac("reNflip", str(len(RI_["flips"])))

# ---------------------------------------------------------------- misc Methods constants
mac("nMeasuredSpectra", str(len(PH)))

with open(MS / "numbers.tex", "w") as f:
    f.write("% AUTO-GENERATED by analysis/make_tables.py -- do not edit by hand.\n")
    for k, v in M.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(f"wrote {len(M)} macros and {len(list(TAB.glob('*.tex')))} tables")
