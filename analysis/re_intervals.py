"""
A5. Hartung-Knapp vs DerSimonian-Laird intervals for every random-effects
estimate. STATUS: pre-specified estimates; the choice of interval is a revision.

Scans results/marjieh_summary.json for every pooled record carrying both a
DerSimonian-Laird z interval ('ci') and a Hartung-Knapp(-Sidik-Jonkman)
interval ('hksj_ci', t on k-1 df), plus the new pooled estimates in
multiplicity.json and et_ji_positions.json, and lists the estimates whose
interval changes from excluding zero to including it (or back).
The main text reports the Hartung-Knapp interval; the DL interval goes to the
supplement (tab_supp_pooled_dl). Writes results/re_intervals.json.
"""
import json
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / "results"


def excl(ci):
    return ci[0] > 0 or ci[1] < 0


def walk(d, path=()):
    if isinstance(d, dict):
        if "hksj_ci" in d and "ci" in d:
            yield path, d["est"], d["ci"], d["hksj_ci"], d.get("k")
        elif "hk_ci" in d and "dl_ci" in d:
            yield path, d["est"], d["dl_ci"], d["hk_ci"], d.get("k")
        else:
            for k, v in d.items():
                yield from walk(v, path + (k,))


def main():
    rows = []
    for f in ("marjieh_summary.json", "multiplicity.json", "decomposition_re.json", "et_ji_positions.json"):
        p = RES / f
        if not p.exists():
            continue
        D = json.load(open(p))
        for path, est, dl, hk, k in walk(D):
            if f == "et_ji_positions.json" and path[-1].startswith(("loc_", "locdiff_")):
                w, c, key = path[1], path[2], path[-1].split("_", 1)[1]
                if c != "listener" and any(D["est"][w][e][c][f"edge_{key}"] > 0 for e in D["harm6"]):
                    continue   # model extremum on the window edge: no interior peak/dip to locate
            rows.append({"source": f, "key": "/".join(path), "est": est, "dl_ci": dl, "hk_ci": hk, "k": k,
                         "dl_excl0": excl(dl), "hk_excl0": excl(hk)})
    flips = [r for r in rows if r["dl_excl0"] != r["hk_excl0"]]
    out = {"status": "pre-specified estimates; Hartung-Knapp interval adopted in revision",
           "n_estimates": len(rows), "flips": flips, "all": rows}
    json.dump(out, open(RES / "re_intervals.json", "w"), indent=1)
    print(f"{len(rows)} random-effects estimates, {len(flips)} change excludes-zero status")
    for r in flips:
        print(f"  {r['source']}:{r['key']}  est {r['est']:+.3f}  DL [{r['dl_ci'][0]:+.3f}, {r['dl_ci'][1]:+.3f}]"
              f"  HK [{r['hk_ci'][0]:+.3f}, {r['hk_ci'][1]:+.3f}]")


if __name__ == "__main__":
    main()
