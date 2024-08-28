#!/usr/bin/env python

########################################################################################################################
# The current script reproduces Figure 3.10 in The Elements of Statistical Learning
# (Friedman, Tibshirani, and Hastie, 2009)
########################################################################################################################

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


if __name__ == '__main__':
	df = pd.read_table('../data/prostate.data', index_col=0, header=0)
	
	n_vars = df.shape[1] - 2
	X = df[['lcavol', 'lweight', 'age', 'lbph', 'svi', 'lcp', 'gleason', 'pgg45']]
	Y = df[['lpsa']]
	X = StandardScaler().fit_transform(X)
	Y = StandardScaler().fit_transform(Y)
	
	ols = LinearRegression()
	ols.fit(X, Y)
	ols_coef = ols.coef_
	best = LassoCV(alphas=np.exp(np.linspace(-10., 10., num=50)), fit_intercept=False).fit(X, Y)
	print([best.coef_, np.sum(best.coef_)])
	
	info = []
	for alpha in np.exp(np.linspace(-10., 10., num=50)):
		temp = [alpha]
		reg = Lasso(fit_intercept=False, alpha=alpha)
		reg.fit(X, Y)
		temp.extend(reg.coef_.tolist())
		temp.append(np.sum(np.abs(reg.coef_)) / np.sum(np.abs(ols_coef)))
		info.insert(0, temp)
	
	info = np.array(info)
	for i in range(n_vars):
		plt.plot(info[:, -1], info[:, i + 1], marker='.', color='b')
	plt.hlines(y=0., xmin=-1, xmax=10, color='k', ls='--')
	plt.vlines(x=np.sum(np.abs(best.coef_)) / np.sum(np.abs(ols_coef)), ymin=-0.2, ymax=0.8, color='orchid', ls='--', alpha=0.8)
	plt.xlim(-0.05, 1.12)
	plt.ylim(-0.15, 0.62)
	plt.xlabel('s')
	plt.ylabel('coefficients')
	plt.show()

