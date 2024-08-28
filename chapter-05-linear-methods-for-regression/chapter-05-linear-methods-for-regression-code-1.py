#!/usr/bin/env python

########################################################################################################################
# The current script reproduces Figure 3.5 in The Elements of Statistical Learning
# (Friedman, Tibshirani, and Hastie, 2009)
########################################################################################################################

import pandas as pd
from itertools import combinations
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

if __name__ == '__main__':
	
	df = pd.read_table('../data/prostate.data', index_col=0, header=0)
	
	n_vars = df.shape[1] - 2
	X = df[['lcavol', 'lweight', 'age', 'lbph', 'svi', 'lcp', 'gleason', 'pgg45']]
	Y = df['lpsa']
	train_X = df[df.train == 'T'][['lcavol', 'lweight', 'age', 'lbph', 'svi', 'lcp', 'gleason', 'pgg45']]
	train_Y = df[df.train == 'T']['lpsa']
	test_X = df[df.train == 'F'][['lcavol', 'lweight', 'age', 'lbph', 'svi', 'lcp', 'gleason', 'pgg45']]
	test_Y = df[df.train == 'F']['lpsa']
	
	rss_list = [[((Y - Y.mean()) ** 2).sum()]]
	rss_min = [((Y - Y.mean()) ** 2).sum()]
	for i in range(1, n_vars + 1):

		comb = combinations(list(range(n_vars)), i)
		temp = []
		
		for j in comb:
			curr_cols = X.iloc[:, list(j)]
			reg = LinearRegression()
			reg.fit(curr_cols, Y)
			pred = reg.predict(curr_cols)
			temp.append(((pred - Y) ** 2).sum())
		
		temp = sorted(temp)
		rss_list.append(temp)
		rss_min.append(temp[0])
	
	for i in range(n_vars + 1):
		plt.scatter([i] * len(rss_list[i]), rss_list[i], marker='.', color='grey')
	plt.plot(list(range(n_vars + 1)), rss_min, ls='-', color='r')
	plt.plot(list(range(n_vars + 1)), rss_min, marker='p', color='r')
	plt.xlabel('Subset Size')
	plt.ylabel('Residual Sum of Squares')
	plt.show()
