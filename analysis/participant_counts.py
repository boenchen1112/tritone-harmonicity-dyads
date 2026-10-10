"""
Participants per experiment: the number Marjieh et al. (2024) report in their
article, the number of participant identifiers in the released rating file,
and the number analysed here.

The only exclusion made in this study is in marjieh_dyads.read_trials: a
participant whose ratings are all the same (or who has a single rating) has
no within-participant SD, so the z-score is undefined and the participant is
dropped. This script counts those participants. Nothing is re-analysed.

REPORTED holds the N printed in the article (text of Marjieh et al. 2024,
Nat. Commun. 15:1482; Korean experiments from its Supplementary Table 1, which
sums to the 68 given in the text); these are values quoted from the
literature. Supplementary Table 1 gives 202 for the piano experiment where the
text gives 198; the text value is used. REPORTED_RATINGS_HARM is the number of
ratings Supplementary Table 1 lists for the ten-harmonic tones.
Writes results/participant_counts.json.
"""
import csv
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import marjieh_dyads as MD

RES = HERE.parent / "results"
REPORTED = {"harm3": 198, "str3": 194, "comp3": 202, "eq5": 149, "no3": 160, "pure": 176, "bonang": 170,
            "flute": 190, "guitar": 210, "piano": 198, "rolloff": 322,
            "harm3_kr": 24, "str3_kr": 20, "comp3_kr": 24}
REPORTED_KOREAN_TOTAL = 68
REPORTED_RATINGS_HARM = 7500


def count(fname, synth=None, xcol="v1"):
    rows = list(csv.DictReader(open(MD.DATA / fname, encoding="utf-8")))
    if synth is not None:
        rows = [r for r in rows if r["synth"] == synth]
    released = len({r["participant_id"] for r in rows})
    d = MD.read_trials(fname, synth, xcol=xcol)
    analysed = int(len(np.unique(d["pid"])))
    return {"released": released, "analysed": analysed, "dropped_constant": released - analysed,
            "trials_released": len(rows), "trials_analysed": int(len(d["x"]))}


def main():
    out = {"experiments": {}, "reported_korean_total": REPORTED_KOREAN_TOTAL,
           "reported_ratings_harm": REPORTED_RATINGS_HARM}
    for key, (label, fname, synth, cohort, *_rest) in MD.EXPS.items():
        c = count(fname, synth)
        c.update({"label": label, "cohort": cohort, "reported": REPORTED.get(key)})
        out["experiments"][key] = c
    c = count(MD.ROLLOFF_FILE, xcol="intervals")
    c.update({"label": "Roll-off (Study 3)", "cohort": "US", "reported": REPORTED["rolloff"]})
    out["rolloff"] = c
    assert sum(REPORTED[k] for k in ("harm3_kr", "str3_kr", "comp3_kr")) == REPORTED_KOREAN_TOTAL
    assert out["experiments"]["harm3"]["trials_released"] == REPORTED_RATINGS_HARM
    json.dump(out, open(RES / "participant_counts.json", "w"), indent=1)
    for k, v in {**out["experiments"], "rolloff": out["rolloff"]}.items():
        print(f"{k:10s} reported={v['reported']} released={v['released']} dropped={v['dropped_constant']} "
              f"analysed={v['analysed']}")


if __name__ == "__main__":
    main()
