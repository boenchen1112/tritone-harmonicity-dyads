"""
Render Markdown templates (*.md.in) by replacing \\macro tokens with their values from numbers.tex,
so that numbers quoted in the release documents come from the same JSON pipeline as the manuscript.
LaTeX markup inside macro values (\\textminus{}, \\,) is converted to plain text.
Usage: python analysis/render_md.py CORRECTIONS_FROM_EARLIER_VERSION.md.in [...]
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAC = {}
for line in (ROOT / "numbers.tex").read_text(encoding="utf-8").splitlines():
    m = re.match(r"\\newcommand\{\\(\w+)\}\{(.*)\}\s*$", line)
    if m:
        MAC[m.group(1)] = m.group(2)


def plain(v):
    v = v.replace("\\textminus{}", "\u2212").replace("\\,", " ").replace("\\%", "%")
    return re.sub(r"\\textbf\{([^}]*)\}", r"\1", v)


def sub(m):
    name = m.group(1)
    if name not in MAC:
        raise SystemExit(f"unknown macro \\{name}")
    return plain(MAC[name])


for fn in sys.argv[1:]:
    src = ROOT / fn
    out = src.with_suffix("")  # strip .in
    out.write_text(re.sub(r"\\([A-Za-z]+)(?:\{\})?", sub, src.read_text(encoding="utf-8")), encoding="utf-8")
    print("wrote", out.name)
