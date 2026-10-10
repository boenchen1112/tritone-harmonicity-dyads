# Batch bridge to pmcharrison/incon (Harrison & Pearce 2020), called from Python
# via Rscript rather than rpy2 -- a subprocess + CSV needs no Python/R binary
# binding, which matters on a machine whose Application Control policy already
# blocks llvmlite.dll (see extract_partials.py).
#
# Usage: Rscript incon_bridge.R <chords.csv> <out.csv> [models_comma_separated]
#   chords.csv must have a column `chord` holding space-separated MIDI numbers.
#   Output has one column per model, in the same row order.

suppressMessages(library(incon))

args <- commandArgs(trailingOnly = TRUE)
in_path <- args[1]
out_path <- args[2]
models <- if (length(args) >= 3 && nzchar(args[3])) {
  strsplit(args[3], ",", fixed = TRUE)[[1]]
} else {
  incon::incon_models$label
}

chords_df <- read.csv(in_path, stringsAsFactors = FALSE)
chords <- lapply(strsplit(trimws(chords_df$chord), "\\s+"), as.numeric)

out <- matrix(NA_real_, nrow = length(chords), ncol = length(models),
              dimnames = list(NULL, models))
for (j in seq_along(models)) {
  m <- models[j]
  for (i in seq_along(chords)) {
    v <- tryCatch(incon(chords[[i]], m), error = function(e) NA_real_)
    # a couple of models can return a named vector; keep the first element
    out[i, j] <- if (length(v) >= 1) as.numeric(v)[1] else NA_real_
  }
  cat(sprintf("  [R] %-24s done (%d chords)\n", m, length(chords)), file = stderr())
}

write.csv(cbind(chords_df, as.data.frame(out)), out_path, row.names = FALSE)
cat(sprintf("  [R] wrote %s\n", out_path), file = stderr())
