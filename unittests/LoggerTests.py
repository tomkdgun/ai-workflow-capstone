#!/usr/bin/env python
"""
logger tests
"""

import os, sys
import csv
import unittest
from ast import literal_eval
import pandas as pd
sys.path.insert(1, os.path.join(os.path.dirname(__file__), '..'))

# import loger functions
from my_modules.logger import update_train_log, update_predict_log, LOGS_DIR

class LoggerTest(unittest.TestCase):

	# ensure log file is created
	def test_01_train(self):
		log_file = os.path.join(LOGS_DIR, "train-test.log")
		if os.path.exists(log_file):
			os.remove(log_file)
		
		# update the log
		tag = 'test'
		data_shape = (100, 10)
		eval_test = {'rmse': 0.5}
		runtime = "00:00:01"
		model_version = 0.1
		model_version_note = "test model"
		
		update_train_log(tag, data_shape, eval_test, runtime, model_version, model_version_note, test=True)

		self.assertTrue(os.path.exists(log_file))

	# ensure that content can be retrieved from log file
	def test_02_train(self):
		log_file = os.path.join(LOGS_DIR, "train-test.log")
		
		# update the log
		tag = 'test'
		data_shape = (100, 10)
		eval_test = {'rmse': 0.5}
		runtime = "00:00:01"
		model_version = 0.1
		model_version_note = "test model"
		
		update_train_log(tag, data_shape,eval_test, runtime, model_version, model_version_note, test=True)

		df = pd.read_csv(log_file)
		logged_eval_test = [literal_eval(i) for i in df['eval_test'].copy()][-1]
		self.assertEqual(eval_test, logged_eval_test)

	# ensure log file is created
	def test_03_predict(self):
		log_file = os.path.join(LOGS_DIR, "predict-test.log")
		if os.path.exists(log_file):
			os.remove(log_file)
		
		# update the log
		tag = 'test' 
		y_pred = [0]
		y_proba = [0.6, 0.4]
		runtime = "00:00:02"
		model_version = 0.1
		query = ['united_states', 24, 'aavail_basic', 8]

		update_predict_log(tag, y_pred, y_proba, query, runtime, model_version, test=True)
		
		self.assertTrue(os.path.exists(log_file))

	# ensure that content can be retrieved from log file
	def test_04_predict(self):
		log_file = os.path.join(LOGS_DIR, "predict-test.log")

		# update the log
		tag = 'test'
		y_pred = [0]
		y_proba = [0.6, 0.4]
		runtime = "00:00:02"
		model_version = 0.1
		query = ['united_states', 24, 'aavail_basic', 8]

		update_predict_log(tag, y_pred, y_proba, query, runtime, model_version, test=True)

		df = pd.read_csv(log_file)
		logged_y_pred = [literal_eval(i) for i in df['y_pred'].copy()][-1]
		self.assertEqual(y_pred, logged_y_pred)


# Run the tests
if __name__ == '__main__':
	unittest.main()
