#!/usr/bin/env bash
# Fetch the third-party data this analysis uses into ./external (or $TRITONE_EXTERNAL).
# Nothing here is redistributed with the repository; see README.md, "Data".
set -euo pipefail
cd "$(dirname "$0")/.."
EXT=${TRITONE_EXTERNAL:-external}
mkdir -p "$EXT"

# Marjieh et al. (2024) code, stimuli parameters and ratings (no licence stated: fetched, not copied)
if [ ! -d "$EXT/timbre-and-consonance-paper" ]; then
  git -c core.longpaths=true clone https://gitlab.com/pmcharrison/timbre-and-consonance-paper \
    "$EXT/timbre-and-consonance-paper"
fi
git -C "$EXT/timbre-and-consonance-paper" checkout -q d6628d090978a58a1eba5ce4e4cb0e2760f75f31

# inconData (Harrison; CC0): Bowling et al. (2018) chords with their presented pitches
if [ ! -d "$EXT/inconData" ]; then
  git clone https://github.com/pmcharrison/inconData "$EXT/inconData"
fi
git -C "$EXT/inconData" checkout -q b95b4db27356a04241ed1a2e38ffc8f5e1f68bbe

# Bowling chords at the presented just-intoned pitches (pi_chord), exported with R
Rscript scripts/export_bowl18_ji.R "$EXT/inconData/data/bowl18.rda" "$EXT/bowl18_ji.csv"

# DCMLab consonance (MIT): chord ratings of Bowling et al. (2018) and Johnson-Laird et al.
# (2012), read by core/chord_validation.py from $TRITONE_ROOT/behavioral_data/
ROOTDIR=${TRITONE_ROOT:-core}
DCM=https://raw.githubusercontent.com/DCMLab/consonance/a13bcb0a52ac41cff9647e8b6406aed32a8250c6/data
mkdir -p "$ROOTDIR/behavioral_data"
curl -fsSL "$DCM/bowling2018_consonance.tsv" -o "$ROOTDIR/behavioral_data/bowling2018.tsv"
curl -fsSL "$DCM/johnson-laird2012_consonance.tsv" -o "$ROOTDIR/behavioral_data/johnson-laird2012.tsv"

cat <<'MSG'
Fetched the dense-dyad and chord data.

The measured-spectra analyses (Supplementary Section S11) also need instrument recordings,
which their licences do not allow us to redistribute:
  Philharmonia sound samples  https://philharmonia.co.uk/resources/sound-samples/
  Univ. of Iowa MIS           https://theremin.music.uiowa.edu/MIS.html
The partials measured from them are included (core/extracted_partials.json), so every number
in the paper can be reproduced without the recordings. To re-measure the partials, download
the notes listed in Supplementary Section S11 into core/audio_samples/.
MSG
