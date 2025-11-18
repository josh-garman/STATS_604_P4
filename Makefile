.DEFAULT_GOAL := all
.PHONY: all dirs setup rawdata predictions clean train1 train2 data_proc1 data_proc2

#PY := python -u
PY := python3 -u

RAW_DIR  := Data/raw
INT_DIR  := Data/intermediate
PROC_DIR := Data/processed

FULL_DATA := $(PROC_DIR)/full_data.csv   # output of PJM_weather_merge.py

dirs:
	@mkdir -p \
		Data/raw \
		Data/processed \
		Data/intermediate \
		Models/production \
		Models/validation_and_training \
		results


setup: dirs
	

rawdata: setup
	@rm -rf $(RAW_DIR)
	@mkdir -p $(RAW_DIR)
	@bash Scripts/raw_data_download/PJM_zip_download.sh $(RAW_DIR)
	@$(PY) Scripts/raw_data_download/weather_hist.py


all: data_proc1 train1 data_proc2 train2 

data_proc1:
	@$(PY) Scripts/data_processing/PJM_data_cleaning.py
	@$(PY) Scripts/data_processing/PJM_weather_merge.py

train1:
	@$(PY) Scripts/hourly_mw_train.py

data_proc2:
	@$(PY) Scripts/build_train_and_val_max_day.py

train2:
	@$(PY) Scripts/max_day_model_train.py

validation:
	@$(PY) Scripts/max_day_validation_2024.py
	@$(PY) Scripts/hourly_max_validation_2024.py

predictions:
	@$(PY) Scripts/weather_tomorrow.py 
	@$(PY) Scripts/predict_data_input_load.py
	@$(PY) Scripts/predict.py

# wipe everything except Data/raw
clean:
	@rm -rf Data/processed Data/intermediate
	@rm -rf Models/production Models/validation_and_training
	@rm -rf results
	@mkdir -p \
		Data/processed \
		Data/intermediate \
		Models/production \
		Models/validation_and_training \
		results
