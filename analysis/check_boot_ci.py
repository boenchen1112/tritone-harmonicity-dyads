import json
import re
from pathlib import Path

MS = Path(__file__).resolve().parents[1]

MB = json.load(open(MS / 'results' / 'model_boot.json'))
print("nboot", MB["_nboot"])
for m, v in MB.items():
    if m.startswith("_"):
        continue
    for k, r in v.items():
        lo, hi = r["ci"]
        flag = "" if lo <= r["est"] <= hi else "  <-- est outside"
        print(f"{m:16s} {k:14s} {r['est']:+.3f} [{lo:+.3f}, {hi:+.3f}]{flag}")
for d, v in MB["_diffs"].items():
    for k, r in v.items():
        lo, hi = r["ci"]
        flag = "" if lo <= r["est"] <= hi else "  <-- est outside"
        print(f"{d:26s} {k:14s} {r['est']:+.3f} [{lo:+.3f}, {hi:+.3f}]{flag}")
tex = open(MS / 'paper.tex', encoding='utf-8').read()
print("macros used:", sorted(set(re.findall(r'\\(m[bd][A-Z][A-Za-z]*)', tex))))
