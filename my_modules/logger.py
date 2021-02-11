#!/usr/bin/env python
"""
module with functions to enable logging
"""

import time, os, re, csv, sys, uuid, joblib
from datetime import date

LOGS_DIR = os.path.join(os.path.dirname(__file__), '..', 'logs')

if not os.path.exists(LOGS_DIR):
	os.mkdir(LOGS_DIR)

# update train log file
def update_train_log(tag, data_shape, eval_test, runtime, MODEL_VERSION, MODEL_VERSION_NOTE, test=False):
	# name the logfile using something that cycles with date (year, month)    
	today = date.today()
	if test:
		logfile = os.path.join(LOGS_DIR, "train-test.log")
	else:
		logfile = os.path.join(LOGS_DIR, "train-{}-{}.log".format(today.year, today.month))
		
	# write the data to a csv file    
	header = ['unique_id','timestamp','tag', 'data_shape', 'eval_test','model_version', 'model_version_note','runtime']
	write_header = False
	if not os.path.exists(logfile):
		write_header = True
	with open(logfile, 'a') as csvfile:
		writer = csv.writer(csvfile, delimiter=',')
		if write_header:
			writer.writerow(header)

		to_write = map(str, [uuid.uuid4(), time.time(), tag, data_shape, eval_test, MODEL_VERSION, MODEL_VERSION_NOTE, runtime])
		writer.writerow(to_write)

# update predict log file
def update_predict_log(tag, y_pred, y_proba, query, runtime, MODEL_VERSION, test=False):
	# name the logfile using something that cycles with date (year, month)    
	today = date.today()
	if test:
		logfile = os.path.join(LOGS_DIR, "predict-test.log")
	else:
		logfile = os.path.join(LOGS_DIR, "predict-{}-{}.log".format(today.year, today.month))
		
	## write the data to a csv file    
	header = ['unique_id','timestamp', 'tag', 'y_pred', 'y_proba', 'query', 'model_version', 'runtime']
	write_header = False
	if not os.path.exists(logfile):
		write_header = True
	with open(logfile,'a') as csvfile:
		writer = csv.writer(csvfile, delimiter=',')
		if write_header:
			writer.writerow(header)

		to_write = map(str,[uuid.uuid4(), time.time(), tag, y_pred, y_proba,query, MODEL_VERSION, runtime])
		writer.writerow(to_write)

# basic test procedure for logger.py
if __name__ == "__main__":
	from model import MODEL_VERSION, MODEL_VERSION_NOTE
	
	## train logger
	update_train_log('test', str((100, 10)), "{'rmse': 0.5}", "00:00:01", MODEL_VERSION, MODEL_VERSION_NOTE, test=True)
	## predict logger
	update_predict_log('test', "[0]", "[0.6, 0.4]", "['united_states', 24, 'aavail_basic', 8]", "00:00:01", MODEL_VERSION, test=True)
