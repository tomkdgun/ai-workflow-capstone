#!/usr/bin/env python
"""
module with functions for ploting
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Image


def plotarima(n_periods, timeseries, automodel, title):
	# Forecast
	fc, confint = automodel.predict(n_periods=n_periods, return_conf_int=True)
	# Daily index
	fc_ind = pd.date_range(timeseries.index[timeseries.shape[0]-1], periods=n_periods, freq='D')
	# Forecast series
	fc_series = pd.Series(fc, index=fc_ind)
	# Upper and lower confidence bounds
	lower_series = pd.Series(confint[:, 0], index=fc_ind)
	upper_series = pd.Series(confint[:, 1], index=fc_ind)
	# Create plot
	plt.figure(figsize=(12, 8))
	plt.plot(timeseries)
	plt.plot(fc_series, color="red")
	plt.title(label=title)
	plt.xlabel("Date")
	plt.ylabel(timeseries.name)
	plt.fill_between(lower_series.index, 
					 lower_series, 
					 upper_series, 
					 color="k", alpha=.25)
	plt.legend(("past", "forecast", "95% confidence interval"), loc="upper left")
	plt.show()

def create_correlation_grid_plot(df, columns, title):
	"""
	Create correlation grid using data from DataFrame df, specify which 
	columns to display with columns param. 

	Inspired by: https://seaborn.pydata.org/examples/many_pairwise_correlations.html
	"""
	plt.style.use('seaborn')
	# Compute the correlation matrix
	corr = df[columns].corr()

	# Generate a mask for the upper triangle
	mask = np.zeros_like(corr, dtype=np.bool)
	mask[np.triu_indices_from(mask)] = True

	# Set up the matplotlib figure
	# f, ax = plt.subplots(figsize=(11, 9))
	f, ax = plt.subplots(figsize=(11, 7),dpi=150,facecolor='white')
	f.suptitle(title)

	# Generate a custom diverging colormap
	cmap = sns.diverging_palette(220, 10, as_cmap=True)

	# Draw the heatmap with the mask and correct aspect ratio
	sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})