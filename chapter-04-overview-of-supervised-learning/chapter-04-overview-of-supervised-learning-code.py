#!/usr/bin/env python
import scipy
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from configparser import ConfigParser

if __name__ == '__main__':
	
	constants = ConfigParser()
	constants.read("../CONSTANTS.ini")
	seed = int(constants.get("RANDOMSEED", "random_seed"))
	
	# generate mean values for two classes
	mean_cnt = 10
	b_mean = scipy.stats.multivariate_normal(np.array([1, 0]), np.eye(2), seed=seed)
	b_mean_samples = b_mean.rvs(mean_cnt)
	
	r_mean = scipy.stats.multivariate_normal(np.array([0, 1]), np.eye(2), seed=seed)
	r_mean_samples = r_mean.rvs(mean_cnt)
	
	# generate samples for two classes
	# blue classes
	sample_cnt_train = 100
	sample_cnt_test = 10000
	
	b_train_samples_X = (
		scipy.stats.multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]], seed=seed).rvs(size=sample_cnt_train) +
		b_mean_samples[np.random.randint(0, 10, sample_cnt_train), ]
	)
	r_train_samples_X = (
		scipy.stats.multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]], seed=seed).rvs(size=sample_cnt_train) +
		r_mean_samples[np.random.randint(0, 10, sample_cnt_train), ]
	)
	train_data_X = np.concatenate((b_train_samples_X, r_train_samples_X))
	train_data_Y = np.concatenate((
		np.zeros((sample_cnt_train, 1)),  # blue class has label 0
		np.ones((sample_cnt_train, 1))    # red class has label 1
	)).flatten()
	
	b_test_samples_X = (
		scipy.stats.multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]], seed=seed).rvs(size=sample_cnt_test) +
		b_mean_samples[np.random.randint(0, 10, sample_cnt_test), ]
	)
	r_test_samples_X = (
		scipy.stats.multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]], seed=seed).rvs(size=sample_cnt_test) +
		r_mean_samples[np.random.randint(0, 10, sample_cnt_test), ]
	)
	test_data_X = np.concatenate((b_test_samples_X, r_test_samples_X))
	test_data_Y = np.concatenate((
		np.zeros((sample_cnt_test, 1)),  # blue class has label 0
		np.ones((sample_cnt_test, 1))    # red class has label 1
	)).flatten()
	
	# linear classifier
	reg = LinearRegression()
	reg.fit(train_data_X, train_data_Y)
	train_data_Y_lregpred = reg.predict(train_data_X)
	lreg_train_error = 1 - accuracy_score(train_data_Y, train_data_Y_lregpred > 0.5)
	
	test_data_Y_lregpred = reg.predict(test_data_X)
	lreg_test_error = 1 - accuracy_score(test_data_Y, test_data_Y_lregpred > 0.5)
	
	# knn classifier
	knn_test_error = []
	knn_train_error = []
	for K in range(1, sample_cnt_train * 2 + 1):
		if K % 50 == 0:
			print('-' * 80)
			print(f'current K value is {K}')
		knn = KNeighborsClassifier(n_neighbors=K)
		knn.fit(train_data_X, train_data_Y)

		train_data_Y_knnpred = knn.predict(train_data_X)
		knn_train_error.append(1 - accuracy_score(train_data_Y, train_data_Y_knnpred))

		test_data_Y_knnpred = knn.predict(test_data_X)
		knn_test_error.append(1 - accuracy_score(test_data_Y, test_data_Y_knnpred))
	
	plt.plot(list(range(1, sample_cnt_train * 2 + 1)), knn_train_error, color='tab:blue', linestyle='-')
	plt.plot(list(range(1, sample_cnt_train * 2 + 1)), knn_test_error, color='red', linestyle='-')
	plt.plot(list(range(1, sample_cnt_train * 2 + 1)), knn_train_error, color='tab:blue', marker='.', label='Train')
	plt.plot(list(range(1, sample_cnt_train * 2 + 1)), knn_test_error, color='red', marker='.', label='Test')
	plt.xlabel('K')
	plt.ylabel('Test Error')
	
	plt.plot([3], lreg_train_error, color='tab:blue', marker='v')
	plt.plot([3], lreg_test_error, color='red', marker='v')
	
	plt.legend()
	plt.show()
	