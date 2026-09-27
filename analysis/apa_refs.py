"""
APA 7 citations and reference lists from refs.bib, shared by the LaTeX and
Word versions so that both carry the same references.

Run as a script, it writes bib_paper.tex and bib_supplement.tex: natbib
thebibliography environments holding only the works each document cites,
sorted and formatted in APA 7. paper.tex and supplement.tex \\input these
files, so BibTeX is not used.

Run from the manuscript directory:  python analysis/apa_refs.py
"""
import re
from pathlib import Path

MS = Path(__file__).resolve().parents[1]

ACCENTS = {'"': "\u0308", "'": "\u0301", "`": "\u0300", "^": "\u0302", "~": "\u0303", "c": "\u0327"}


def group(s, i):
    """s[i] == '{': return (content, index after the closing brace)."""
    assert s[i] == "{", s[i:i + 30]
    depth, j = 0, i
    while True:
        c = s[j]
        if c == "\\":
            j += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
        j += 1


def load_bib():
    src = (MS / "refs.bib").read_text(encoding="utf-8")
    src = re.sub(r"(?m)^\s*%.*\n", "", src)
    out = {}
    for m in re.finditer(r"@(\w+)\{([^,]+),", src):
        typ, key = m.group(1).lower(), m.group(2).strip()
        body, _ = group(src, src.index("{", m.start()))
        fields, j = {}, len(key) + 1
        # walk the fields at the top level of the entry only
        while True:
            f = re.compile(r"\s*,?\s*(\w+)\s*=\s*").match(body, j)
            if not f:
                break
            j = f.end()
            if body[j] == "{":
                val, j = group(body, j)
            else:
                v = re.match(r"[^,\n]*", body[j:]).group(0)
                val, j = v.strip(), j + len(v)
            fields[f.group(1).lower()] = val
        out[key] = {"type": typ, **fields}
    return out


BIB = load_bib()


def clean_bib(s):
    """BibTeX field -> plain Unicode text."""
    s = re.sub(r"\{\\([\"'`^~c])\{?(\w)\}?\}|\\([\"'`^~c])\{?(\w)\}?",
               lambda m: (m.group(2) or m.group(4)) + ACCENTS[m.group(1) or m.group(3)], s)
    s = s.replace("\\&", "&").replace("\\ ", " ").replace("--", "\u2013")
    s = re.sub(r"\\url\{([^}]*)\}", r"\1", s)
    s = s.replace("{", "").replace("}", "")
    import unicodedata
    return unicodedata.normalize("NFC", re.sub(r"\s+", " ", s).strip())


def split_authors(s):
    s = s.strip()
    if s.startswith("{") and group(s, 0)[1] == len(s):
        return [s]  # corporate author in braces
    return [a.strip() for a in re.split(r"\s+and\s+", s)]


def is_corporate(a):
    return a.startswith("{") and a.endswith("}")


def initials(given):
    parts = []
    for tok in given.split():
        sub = [p for p in tok.split("-") if p]
        parts.append("-".join(p[0] + "." for p in sub))
    return " ".join(parts)


def author_ref(a):
    if is_corporate(a):
        return clean_bib(a)
    last, given = [x.strip() for x in a.split(",", 1)]
    return f"{clean_bib(last)}, {initials(clean_bib(given))}"


def author_last(a):
    return clean_bib(a) if is_corporate(a) else clean_bib(a.split(",")[0])


def year_of(e):
    return e.get("year", "n.d.")


def _suffixes():
    """APA: same author list and year -> a, b, c ordered by title."""
    groups = {}
    for k, e in BIB.items():
        sig = (tuple(author_last(a) for a in split_authors(e.get("author", ""))), year_of(e))
        groups.setdefault(sig, []).append(k)
    suf = {}
    for ks in groups.values():
        if len(ks) > 1:
            for n, k in enumerate(sorted(ks, key=lambda k: clean_bib(BIB[k]["title"]).lower())):
                suf[k] = "abcdefgh"[n]
    return suf


SUFFIX = _suffixes()


def year_label(k):
    y = year_of(BIB[k])
    s = SUFFIX.get(k, "")
    return f"{y}-{s}" if (s and y == "n.d.") else f"{y}{s}"


def cite_names(k, narrative):
    au = [author_last(a) for a in split_authors(BIB[k].get("author", ""))]
    if len(au) == 1:
        return au[0]
    if len(au) == 2:
        return f"{au[0]} {'and' if narrative else '&'} {au[1]}"
    return f"{au[0]} et al."


def ref_entry(k):
    """APA 7 reference as a list of (text, italic) runs."""
    e = BIB[k]
    au = [author_ref(a) for a in split_authors(e.get("author", ""))]
    if len(au) == 1:
        a = au[0]
    elif len(au) == 2:
        a = f"{au[0]}, & {au[1]}"
    else:
        a = ", ".join(au[:-1]) + ", & " + au[-1]
    if not a.endswith("."):
        a += "."
    runs = [(f"{a} ({year_label(k)}). ", False)]
    title = clean_bib(e["title"])
    t = e["type"]
    link = (f" https://doi.org/{e['doi']}" if "doi" in e
            else f" {clean_bib(e['url'])}" if "url" in e else "")
    endp = lambda s: s if s.endswith((".", "?", "!")) else s + "."
    if t == "article":
        runs.append((endp(title) + " ", False))
        runs.append((clean_bib(e["journal"]), True))
        if "volume" in e:
            runs.append((", ", False))
            runs.append((clean_bib(e["volume"]), True))
        if "number" in e:
            runs.append((f"({clean_bib(e['number'])})", False))
        if "pages" in e:
            runs.append((f", {clean_bib(e['pages'])}", False))
        runs.append(("." + link, False))
    elif t == "book":
        ed = f" ({clean_bib(e['edition'])} ed.)" if "edition" in e else ""
        runs.append((title, True))
        runs.append((f"{ed}. {clean_bib(e['publisher'])}.{link}", False))
    elif t == "phdthesis":
        runs.append((title, True))
        runs.append((f" [Doctoral dissertation, {clean_bib(e['school'])}].{link}", False))
    elif t == "inproceedings":
        runs.append((endp(title) + " In ", False))
        runs.append((clean_bib(e["booktitle"]), True))
        pg = f" (pp. {clean_bib(e['pages'])})" if "pages" in e else ""
        runs.append((f"{pg}.{link}", False))
    else:  # misc: software, data sets, web pages, preprints
        version = ""
        m = re.search(r"\s*\((Version [^)]*)\)$", title)
        if m:
            title, version = title[:m.start()], f" ({m.group(1)})"
        runs.append((title, True))
        desc = clean_bib(e.get("howpublished", ""))
        url = ""
        if desc.startswith("http"):
            url, desc = desc, ""
        source = ""
        pm = re.match(r"(\S*arXiv) preprint", desc, re.I)
        if pm:
            desc, source = "Preprint", f" {pm.group(1)}."
        desc = clean_bib(e.get("apatype", desc))
        runs.append((version + (f" [{desc}]" if desc else "") + "." + source, False))
        if "retrieved" in e:
            runs.append((f" Retrieved {clean_bib(e['retrieved'])}, from {url}", False))
        else:
            runs.append((link or (" " + url if url else ""), False))
    return runs


def ref_sort_key(k):
    e = BIB[k]
    return ([author_last(a).lower() for a in split_authors(e.get("author", ""))], year_of(e), SUFFIX.get(k, ""))


# ------------------------------------------------------------------ LaTeX output
TEX_ESC = {"&": r"\&", "%": r"\%", "#": r"\#", "_": r"\_", "$": r"\$"}


def tex_text(s):
    """Plain text -> LaTeX, with URLs set in \\url."""
    out, pos = [], 0
    for m in re.finditer(r"https?://\S+", s):
        out.append(re.sub(r"[&%#_$]", lambda c: TEX_ESC[c.group(0)], s[pos:m.start()]))
        url = m.group(0)
        trail = "." if url.endswith(".") and not url.endswith("..") and "doi.org" not in url else ""
        url = url[:-1] if trail else url
        out.append(r"\url{" + url.replace("%", r"\%").replace("#", r"\#") + "}" + trail)
        pos = m.end()
    out.append(re.sub(r"[&%#_$]", lambda c: TEX_ESC[c.group(0)], s[pos:]))
    return "".join(out).replace("\u2013", "--")


def natbib_label(k):
    """natbib author-year label: short names(year)long names. Two authors are
    joined by \\apaand, which the preamble sets to "and" in narrative and "&"
    in parenthetical citations, as APA asks."""
    au = [author_last(a) for a in split_authors(BIB[k].get("author", ""))]
    short = cite_names(k, True)
    full = short if len(au) < 3 else ", ".join(au[:-1]) + ", and " + au[-1]
    short = tex_text(short).replace(" and ", r" \apaand{} ").replace("et al.", "et~al.")
    return f"{short}({year_label(k)}){tex_text(full)}"


def cited_keys(tex):
    body = tex[tex.index("\\begin" + "{document}"):]
    body = re.sub(r"(?<!\\)%[^\n]*", "", body)
    keys = []
    for m in re.finditer(r"\\cite[a-z]*\*?(?:\[[^\]]*\]){0,2}\{([^}]*)\}", body):
        for k in m.group(1).split(","):
            k = k.strip()
            if k not in BIB:
                raise KeyError(f"unknown citation key {k}")
            if k not in keys:
                keys.append(k)
    return keys


def write_bibliography(tex_name, out_name):
    keys = sorted(cited_keys((MS / tex_name).read_text(encoding="utf-8")), key=ref_sort_key)
    lines = ["% Generated by analysis/apa_refs.py from refs.bib (APA 7). Do not edit by hand.",
             "\\begin{thebibliography}{%d}" % len(keys)]
    for k in keys:
        text = "".join(("\\emph{" + tex_text(t) + "}") if it else tex_text(t) for t, it in ref_entry(k))
        lines += [f"\\bibitem[{{{natbib_label(k)}}}]{{{k}}}", text, ""]
    lines.append("\\end{thebibliography}")
    (MS / out_name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(keys)


if __name__ == "__main__":
    for src, dst in (("paper.tex", "bib_paper.tex"), ("supplement.tex", "bib_supplement.tex")):
        print(f"wrote {dst}: {write_bibliography(src, dst)} references")
