# Export the Bowling et al. (2018) chords from inconData with their presented (just-intoned)
# MIDI pitches, for analysis/bowling_ji.py.
# Usage: Rscript scripts/export_bowl18_ji.R <path to inconData/data/bowl18.rda> <output csv>
args <- commandArgs(trailingOnly = TRUE)
load(args[1])
d <- bowl18
pitches <- vapply(d$pi_chord, function(x) paste(vapply(as.numeric(x), format, "", digits = 10), collapse = " "), "")
out <- data.frame(pi_chord = pitches, rating = d$rating, rating_sd = d$rating_sd, rating_se = d$rating_se)
write.csv(out, args[2], row.names = FALSE, quote = TRUE)
