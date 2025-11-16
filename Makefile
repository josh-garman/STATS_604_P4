.DEFAULT_GOAL := all
.PHONY: all dirs setup rawdata predictions clean

PY := python -u

dirs:
	mkdir -p Data/raw Data/processed Data/intermediates results models intermediates_results

setup: dirs
	@echo "Setup complete."

rawdata: setup
	bash Scripts/PJM_zip_download.sh

all: setup
	# $(PY) Scripts/run_analysis.py

predictions: all
	$(PY) Scripts/predict.py

clean:
	@echo "Cleaning intermediates, results, and models (keeping Data/raw)..."
	rm -rf Data/processed results models
	mkdir -p results models
	@echo "Done."
