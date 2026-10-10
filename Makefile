# Thin wrapper around run_all.sh; see README.md.
.PHONY: all analysis docs check fetch clean

all: analysis

analysis:
	bash run_all.sh

# tables, figures, consistency check and PDFs from the committed results/*.json
docs:
	bash run_all.sh --no-analysis

check:
	python analysis/check_consistency.py

fetch:
	bash scripts/fetch_data.sh

clean:
	rm -f *.aux *.log *.out *.bbl *.blg *.synctex.gz
