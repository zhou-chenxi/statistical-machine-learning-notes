#!/usr/bin/env python

########################################################################################################################
# The current script reproduces Figure 3.8 in The Elements of Statistical Learning
# (Friedman, Tibshirani, and Hastie, 2009)
########################################################################################################################

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def df_ridge(X, pen_param):
	sing_vals = np.linalg.svd(X).S
	return sum(sing_vals ** 2 / (sing_vals ** 2 + pen_param))


if __name__ == '__main__':
	df = pd.read_table('../data/prostate.data', index_col=0, header=0)
	
	n_vars = df.shape[1] - 2
	X = df[['lcavol', 'lweight', 'age', 'lbph', 'svi', 'lcp', 'gleason', 'pgg45']]
	Y = df[['lpsa']]
	X = StandardScaler().fit_transform(X)
	Y = StandardScaler().fit_transform(Y)
	
	best = RidgeCV(alphas=np.exp(np.linspace(-10., 10., num=50)), fit_intercept=False).fit(X, Y)
	
	info = []
	for alpha in np.exp(np.linspace(-10., 10., num=50)):
		temp = [alpha]
		reg = Ridge(fit_intercept=False, alpha=alpha)
		reg.fit(X, Y)
		temp.extend(reg.coef_.tolist()[0])
		temp.append(df_ridge(X, alpha))
		info.insert(0, temp)
	
	info = np.array(info)
	for i in range(n_vars):
		plt.plot(info[:, -1], info[:, i + 1], marker='.', color='b')
	plt.hlines(y=0., xmin=-1, xmax=10, color='k', ls='--')
	plt.vlines(x=df_ridge(X, best.alpha_), ymin=-0.2, ymax=0.8, color='orchid', ls='--', alpha=0.8)
	plt.xlim(-0.5, 8.5)
	plt.ylim(-0.15, 0.62)
	plt.xlabel('df')
	plt.ylabel('coefficients')
	plt.show()
	
