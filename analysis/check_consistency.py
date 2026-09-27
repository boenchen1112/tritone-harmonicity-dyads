"""
A7. Consistency checks for the manuscript. Exits non-zero (and prints every
failure) if:

  (a) a summary section (abstract, introduction, conclusion) contains a number
      typed by hand instead of a numbers.tex macro, or a summary macro listed in
      CLAIMS does not equal the table/JSON value it summarises;
  (b) curve weights are quoted without their harmonicity tolerance: every
      sentence using a weight macro (WEIGHT_MACROS) must also name the
      tolerance (the tolerance macro or "6.83"/"20-cent"), and the two
      tolerances' weights must come from distinct macros;
  (c) a \\ref/\\eqref/\\cref points to a label that does not exist (main text
      and supplement pooled, since they cross-reference through xr), or a
      label is defined twice;
  (d) a reference in refs.bib is never cited, or a citation has no entry;
  (e) CHANGELOG_REVISION.md, where present, does not name the A8 interpretation rule that
      results/et_ji_rule.json computes.

Usage: python analysis/check_consistency.py [--tex paper.tex --tex supplement.tex]
"""
import json
import re
import sys
from pathlib import Path

MS = Path(__file__).resolve().parents[1]
RES = MS / "results"
TEX = [MS / "paper.tex"] + ([MS / "supplement.tex"] if (MS / "supplement.tex").exists() else [])
FAIL = []


def fail(kind, msg):
    FAIL.append(f"[{kind}] {msg}")


def macros():
    M = {}
    for line in (MS / "numbers.tex").read_text(encoding="utf-8").splitlines():
        m = re.match(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}$", line)
        if m:
            M[m.group(1)] = m.group(2)
    return M


def strip_comments(s):
    return re.sub(r"(?<!\\)%.*", "", s)


def block(tex, start_pat, end_pat):
    m = re.search(start_pat + r"(.*?)" + end_pat, tex, re.S)
    return m.group(1) if m else ""


# ---------------------------------------------------------------- (a) hand-typed numbers in summaries
ALLOW = [
    r"\\(?:ref|eqref|cref|label|cite[pt]?|citealp|citeauthor|input|tabinput|includegraphics|url|href)\*?(\[[^\]]*\])?\{[^}]*\}",
    r"\$[^$]*\$",                       # inline math (symbols, indices)
    r"\\[A-Za-z]+",                     # macro names
    r"(?i)\b(?:section|table|figure|appendix|claim|study|experiment)s?~?\s*S?\d+(\.\d+)*",  # cross-refs typed
    r"\b(?:19|20)\d\d\b",               # years
    r"\b\d+(?:\.\d+)?\s*(?:-|--)?\s*(?:semitones?|cents?|dB|harmonics?|pt|kHz|Hz)\b",  # design constants
    r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten)\b",
    r"\b\d:\d\b|\b\d+:\d+\b",           # frequency ratios
    r"\b(?:i|ii|iii|iv|v)\)",           # claim numbering
]
# design constants that may appear as plain digits in summaries (not results)
ALLOW_DIGITS = {"0.2", "6.83", "20", "7", "5", "3", "2", "1", "0", "15", "12", "14", "8", "6", "10", "25", "100"}


def handtyped(text, where):
    s = strip_comments(text)
    for pat in ALLOW:
        s = re.sub(pat, " ", s)
    for m in re.finditer(r"(?<![A-Za-z\\])[-+]?\d+(?:\.\d+)?", s):
        tok = m.group(0).lstrip("+-")
        if tok not in ALLOW_DIGITS:
            ctx = s[max(0, m.start() - 40): m.end() + 40].replace("\n", " ")
            fail("a", f"{where}: hand-typed number '{m.group(0)}' ... {ctx.strip()} ...")


# summary macro -> (json file, key path, formatter). Filled as the summaries are rewritten.
def fmt_z(x):
    s = f"{x:+.2f}"
    return "0.00" if s in ("+0.00", "-0.00") else s.replace("-", "\\textminus{}")


CLAIMS = {}
CLAIMS_FILE = Path(__file__).with_name("consistency_claims.json")
if CLAIMS_FILE.exists():
    CLAIMS = json.loads(CLAIMS_FILE.read_text(encoding="utf-8"))


def get_path(d, path):
    for p in path:
        d = d[p] if not isinstance(d, list) else d[int(p)]
    return d


def check_claims(M):
    for mac, spec in CLAIMS.items():
        if mac not in M:
            fail("a", f"claim macro \\{mac} not defined in numbers.tex")
            continue
        val = get_path(json.load(open(RES / spec["json"])), spec["path"])
        exp = {"z": fmt_z, "r3": lambda x: f"{x:.3f}", "r2": lambda x: f"{x:.2f}",
               "pct0": lambda x: f"{100 * x:.0f}"}[spec.get("fmt", "z")](val)
        got = M[mac]
        if spec.get("part") == "est":
            got = got.split(" [")[0]
        if got != exp:
            fail("a", f"\\{mac} = '{got}' but {spec['json']}:{'/'.join(map(str, spec['path']))} gives '{exp}'")


# ---------------------------------------------------------------- (b) curve weights carry their tolerance
WEIGHT_MACROS = {"betaHKpooled": "6.83", "betaHpooled": "6.83", "shBetaR": "20", "shBetaH": "20",
                 "scrBetaR": "20", "scrBetaH": "20", "cvTwentyBetaR": "20", "cvTwentyBetaH": "20"}
TOL_WORDS = {"6.83": [r"6\.83", r"\\sigmaPub", r"published tolerance"],
             "20": [r"\b20[- ]?cent", r"\\sigmaTwenty", r"20\\,cents", r"wider tolerance"]}


def check_weights(tex, where, M):
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z\\])", strip_comments(tex))
    for s in sentences:
        for mac, tol in WEIGHT_MACROS.items():
            if re.search(r"\\" + mac + r"(?![A-Za-z])", s):
                if not any(re.search(w, s) for w in TOL_WORDS[tol]):
                    fail("b", f"{where}: weight \\{mac} ({tol}-cent curve) quoted without its tolerance: "
                              f"{s[:120].strip()}...")
    groups = {}
    for mac, tol in WEIGHT_MACROS.items():
        if mac in M:
            groups.setdefault(tol, set()).add(M[mac])
    if "6.83" in groups and "20" in groups and groups["6.83"] & groups["20"]:
        fail("b", "a weight macro of the 6.83-cent curve has the same value as one of the 20-cent curve")


# ---------------------------------------------------------------- (e) A8 interpretation rule recorded
def check_a8():
    r = RES / "et_ji_rule.json"
    if not r.exists():
        fail("e", "results/et_ji_rule.json missing (run analysis/et_ji_rule.py)")
        return
    rule = json.load(open(r))["rule"]
    log = MS / "CHANGELOG_REVISION.md"
    if not log.exists():  # the revision log is not part of the public release
        return
    lines = log.read_text(encoding="utf-8").splitlines()
    if not any("A8" in l and re.search(rf"rule {rule}(?!\d)", l) for l in lines):
        fail("e", f"CHANGELOG_REVISION.md does not record that A8 interpretation rule {rule} applied")


# ---------------------------------------------------------------- (c) cross-references
def check_refs(texts):
    """Unprefixed refs resolve in the same file; S-x (paper -> supplement) and M-x
    (supplement -> paper) resolve in the companion file, as xr-hyper does."""
    labels, refs = {}, []
    for where, t in texts.items():
        t = strip_comments(t)
        own = labels.setdefault(where, set())
        for m in re.finditer(r"\\label\{([^}]*)\}", t):
            if m.group(1) in own:
                fail("c", f"label '{m.group(1)}' defined twice in {where}")
            own.add(m.group(1))
        for m in re.finditer(r"\\(?:ref|eqref|cref|autoref)\{([^}]*)\}", t):
            for k in m.group(1).split(","):
                refs.append((where, k.strip()))
    target = {"S-": "supplement.tex", "M-": "paper.tex"}
    for where, k in refs:
        pre = k[:2] if k[:2] in target else ""
        file = target[pre] if pre else where
        if k[len(pre):] not in labels.get(file, set()):
            fail("c", f"{where}: \\ref{{{k}}} has no \\label in {file}")


# ---------------------------------------------------------------- (d) bibliography
def check_bib(texts):
    bib = (MS / "refs.bib").read_text(encoding="utf-8")
    keys = set(re.findall(r"@\w+\{([^,\s]+),", bib))
    cited = set()
    for t in texts.values():
        for m in re.finditer(r"\\cite[a-z]*\*?(?:\[[^\]]*\])*\{([^}]*)\}", strip_comments(t)):
            cited |= {k.strip() for k in m.group(1).split(",")}
    for k in sorted(cited - keys):
        fail("d", f"citation '{k}' has no entry in refs.bib")
    for k in sorted(keys - cited):
        fail("d", f"refs.bib entry '{k}' is never cited")


def main():
    M = macros()
    texts = {p.name: p.read_text(encoding="utf-8") for p in TEX}
    paper = texts["paper.tex"]
    summaries = {"abstract": block(paper, r"\\begin\{abstract\}", r"\\end\{abstract\}"),
                 "introduction": block(paper, r"\\section\{Introduction\}", r"\\section\{"),
                 "conclusion": block(paper, r"\\section\{Conclusion\}", r"\\section\*?\{")}
    for where, t in summaries.items():
        if not t.strip():
            fail("a", f"summary section '{where}' not found")
        handtyped(t, where)
    check_claims(M)
    for where, t in texts.items():
        check_weights(t, where, M)
        used = set(re.findall(r"\\([A-Za-z]+)", t))
        # undefined macros that look like result macros are caught by LaTeX; nothing to do here
    check_refs(texts)
    check_bib(texts)
    check_a8()
    if FAIL:
        print(f"check_consistency: {len(FAIL)} problem(s)")
        for f in FAIL:
            print("  " + f)
        sys.exit(1)
    print("check_consistency: OK")


if __name__ == "__main__":
    main()
