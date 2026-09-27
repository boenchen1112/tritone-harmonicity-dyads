"""
Word count of the main text, by a fixed rule: the body from \\section{Introduction} up to
\\section*{Data availability}, excluding the abstract, table and figure environments
(captions included), comments and references. A result macro counts as one word, a citation
command as two words per cited key (author, year), inline math as one word; other LaTeX
commands are dropped but their braced arguments are kept.
Usage: python analysis/word_count.py [paper.tex]
"""
import re
import sys
from pathlib import Path

MS = Path(__file__).resolve().parents[1]
tex = (MS / (sys.argv[1] if len(sys.argv) > 1 else "paper.tex")).read_text(encoding="utf-8")
tex = re.sub(r"(?<!\\)%.*", "", tex)
body = tex.split(r"\section{Introduction}", 1)[1].split(r"\section*{Data availability}", 1)[0]
body = re.sub(r"\\begin\{(table|figure)\*?\}.*?\\end\{\1\*?\}", " ", body, flags=re.S)
body = re.sub(r"\\cite[a-z]*\*?(?:\[[^\]]*\])*\{([^}]*)\}",
              lambda m: " ".join(["CITE CITE"] * len(m.group(1).split(","))), body)
body = re.sub(r"\\(?:label|ref|eqref)\{[^}]*\}", " REF ", body)
body = re.sub(r"\$[^$]*\$", " MATH ", body)
M = set(re.findall(r"\\newcommand\{\\(\w+)\}", (MS / "numbers.tex").read_text(encoding="utf-8")))
body = re.sub(r"\\([A-Za-z]+)(\{\})?", lambda m: " NUM " if m.group(1) in M else " ", body)
body = re.sub(r"[{}~]", " ", body)
words = [w for w in body.split() if re.search(r"[A-Za-z0-9]", w)]
secs = re.split(r"\\section\{", tex.split(r"\section*{Data availability}", 1)[0])
print(f"main text words: {len(words)}")
