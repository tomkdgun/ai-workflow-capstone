#!/usr/bin/env python
"""
model tests
"""

import sys, os
import unittest
sys.path.insert(1, os.path.join(os.path.dirname(__file__), '..'))

# import model specific functions and variables
from my_modules.model import model_train, model_load, model_predict, DATA_DIR, MODEL_DIR

class ModelTest(unittest.TestCase):
	# test the train functionality
	def test_01_train(self):
		# train the model
		data_dir = os.path.join(DATA_DIR, "cs-train")
		model_train(data_dir, test=True)
		self.assertTrue(os.path.exists(os.path.join(MODEL_DIR, "test-all-0_1.joblib")))

	# test the load functionality
	def test_02_load(self):
		# load the model
		all_data, all_models = model_load(test=True)
		model = all_models['all']

		self.assertTrue('predict' in dir(model))
		self.assertTrue('fit' in dir(model))

	# test the predict function input
	def test_03_predict(self):
		# load model first
		all_data, all_models = model_load(test=True)
	
		# test predict
		country = 'all'
		year = '2018'
		month = '01'
		day= '05'

		result = model_predict(country, year, month, day, all_data, all_models, test=True)
		y_pred = result['y_pred']

		self.assertTrue(y_pred[0] > 10000)
		self.assertIsNone(result['y_proba'])

		  
# Run the tests
if __name__ == '__main__':
	unittest.main()
