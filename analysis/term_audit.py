"""Term audit for paper.tex: every technical term must be defined (\\emph{...}) at or before
its first use in the body, and every abbreviation expanded at or before its first use.

Prints one line per term; exits with status 1 if any term is used before its definition.
"""
import re
import sys
from pathlib import Path

MS = Path(__file__).resolve().parent.parent
T = (MS / "paper.tex").read_text(encoding="utf-8")
BODY = T[T.index("\\section{Introduction}"):T.index("\\section*{Acknowledgements}")]
# labels, references and headings are not uses of a term
BODY = re.sub(r"\\(label|ref|includegraphics|tabinput)\{[^}]*\}", "", BODY)
BODY = re.sub(r"(?m)^\\(section|subsection|paragraph)\*?\{.*$", "", BODY)

# term -> (regex of a use, regex of the defining passage)
TERMS = {
    "dyad": (r"\bdyads?\b", r"\\emph\{dyad\}"),
    "dense dyad ratings": (r"dense dyad", r"\\emph\{dense dyad ratings\}"),
    "harmonic complex tone": (r"harmonic complex tones?", r"\\emph\{harmonic complex tone\}"),
    "pure tone": (r"pure tones?", r"\\emph\{pure tone\}"),
    "partial": (r"\bpartials?\b", r"\\emph\{partials\}"),
    "spectrum": (r"\bspectr(um|a)\b", r"\\emph\{spectrum\}"),
    "dissonance curve": (r"dissonance curves?", r"\\emph\{dissonance curve\}"),
    "roughness": (r"\broughness\b", r"\\emph\{roughness\}|heard as rough"),
    "beating": (r"\bbeating\b", r"\\emph\{beating\}"),
    "critical bandwidth": (r"critical\s+bandwidth", r"\\emph\{critical\s+bandwidth\}"),
    "harmonicity": (r"\bharmonicity\b", r"\\emph\{harmonicity\}"),
    "virtual pitch": (r"virtual[- ]pitch", r"\\emph\{virtual pitch\}"),
    "tolerance": (r"\btolerance\b", r"\\emph\{tolerance\}"),
    "semitone": (r"\bsemitones?\b", r"\\emph\{semitones\}"),
    "cent": (r"\bcents?\b", r"\\emph\{cents\}"),
    "tritone": (r"\btritone\b", r"\\emph\{tritone\}|the tritone,\s+two notes six piano keys apart"),
    "minor sixth": (r"minor sixth", r"\\emph\{minor sixth\}|the minor sixth, the pair eight keys apart"),
    "equal temperament": (r"equal temperament|equal-tempered", r"Equal temperament: the|\\emph\{equal temperament\}"),
    "just intonation": (r"just intonation|just-intoned", r"just intonation: intervals tuned|\\emph\{just intonation\}"),
    "familiarity": (r"\bfamiliarity\b", r"\\emph\{familiarity\}"),
    "periodicity": (r"\bperiodicity\b", r"periodicity model of|periodicity term"),
    "z-score": (r"\$z\$-scores?", r"\\emph\{\$z\$-scores\}"),
    "kernel smoothing": (r"\bkernel\b", r"\\emph\{kernel smoothing\}"),
    "least squares": (r"least squares", r"\\emph\{least squares\}"),
    "intercept": (r"\bintercept\b", r"\\emph\{intercept\}"),
    "residual": (r"\bresiduals?\b", r"\\emph\{residual\}"),
    "estimand": (r"\bestimands?\b", r"\\emph\{estimand\}"),
    "held-out": (r"held[- ]out", r"\\emph\{held-out\}"),
    "in-sample": (r"in-sample", r"\\emph\{in-sample\}"),
    "bootstrap": (r"\bbootstraps?\b", r"\\emph\{bootstrap\}"),
    "random effects": (r"random[- ]effects", r"\\emph\{random-effects"),
    "Hartung--Knapp": (r"Hartung--Knapp", r"Hartung--Knapp interval, which"),
    "prediction interval": (r"prediction interval", r"\\emph\{prediction interval\}"),
    "I2": (r"\$I\^2", r"\$I\^2\$, the share"),
    "tau": (r"\\tau", r"\$\\tau\$, the SD"),
    "Holm": (r"\bHolm", r"\\emph\{Holm's procedure\}"),
    "noise ceiling": (r"noise ceiling|\bceilings?\b", r"\\emph\{noise ceiling\}|noise ceiling, the best fit"),
    "split-half": (r"split-half", r"\\emph\{split-half reliability\}"),
    "Spearman--Brown": (r"Spearman--Brown", r"Spearman--Brown formula"),
    "R2": (r"\$R\^2", r"\$R\^2\$ is the share|\$R\^2\$, the share"),
    "sharpness": (r"\bsharpness\b", r"sharpness, the weighting of a spectrum"),
    "roll-off": (r"roll-off", r"\\emph\{roll-off\}|roll-off, the rate"),
    "stretched": (r"\bstretched\b", r"\\emph\{stretched\}|stretched spectra \(tones whose"),
    "SD": (r"\bSD\b|\\SDu", r"standard deviation \(SD\)"),
    "CI": (r"\bCIs?\b", r"\\emph\{confidence interval\} \(CI\)"),
    "H&K": (r"\\HK\b", r"written \\HK\{\} from here on"),
    "composite": (r"\bcomposite\b", r"\\emph\{composite\}"),
    "gap": (r"\bgap\b", r"\\emph\{gap\}"),
    "dip": (r"\bdip\b", r"\\emph\{dip\}"),
    "profile": (r"\bprofiles?\b", r"\\emph\{pleasantness profile\}"),
    "fitted curve": (r"fitted curve", r"\\emph\{fitted curve\}"),
    "template": (r"\btemplate\b", r"template"),
    "KL": (r"Kullback|\bKL\b", r"Kullback"),
}
bad = 0
for name, (use, dfn) in TERMS.items():
    u = re.search(use, BODY)
    d = re.search(dfn, BODY)
    if u is None:
        print(f"  --    {name}: not used")
        continue
    line = BODY[:u.start()].count("\n") + 1
    if d is None:
        print(f"MISSING {name}: used (body line {line}) but no definition found")
        bad += 1
    elif d.start() > u.start() + 400:
        dl = BODY[:d.start()].count("\n") + 1
        print(f"LATE    {name}: first use at body line {line}, defined at {dl}: "
              f"{BODY[max(0, u.start() - 50):u.end() + 30]!r}")
        bad += 1
    else:
        print(f"  ok    {name}")
print("term audit:", "OK" if not bad else f"{bad} problem(s)")
sys.exit(1 if bad else 0)
