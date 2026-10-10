"""
Word counts from the built PDF (pymupdf), with numbers rendered as typeset: the abstract
(from "Abstract" to "Keywords"), the Conclusion (to "Data availability"), and the
main text from the Introduction to the Conclusion's end, with and without captions and tables
(a caption or table line is not separable in plain text, so "with" counts everything on the
pages of that range). A bracketed CI such as "[0.04, 0.32]" counts as two words.
Usage: python analysis/pdf_word_count.py [paper.pdf]
"""
import re
import sys
from pathlib import Path
import pymupdf as fitz

MS = Path(__file__).resolve().parents[1]
pdf = fitz.open(MS / (sys.argv[1] if len(sys.argv) > 1 else "paper.pdf"))
text = "\n".join(p.get_text() for p in pdf)


def words(s):
    return len([w for w in s.split() if re.search(r"[A-Za-z0-9]", w)])


def between(a, b):
    i = text.index(a)
    return text[i + len(a): text.index(b, i)]


abstract = between("Abstract", "Keywords")
conclusion = between("Conclusion\n", "Data availability")
body_all = between("Introduction\n", "Data availability")
print(f"abstract words (PDF): {words(abstract)}")
print(f"conclusion words (PDF): {words(conclusion)}")
print(f"main text incl. captions and tables (PDF): {words(body_all)}")
