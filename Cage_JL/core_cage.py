import torch
from torch import optim
import numpy as np
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from utils import *

class Cage:
	'''
	Cage class:
		Class for Data Programming using CAGE
		[Note: from here on, graphical model imply CAGE algorithm]

	Args:
		n_classes: Number of classes/labels, type is integer
		path: Path to pickle file of input data
		metric_avg: List of average metric to be used in calculating f1_score, default is 'binary'
		n_epochs:Number of epochs, default is 100
		lr: Learning rate for torch.optim, default is 0.01
		n_lfs: Number of LFs
		qt: Quality guide of shape (n_lfs,) and type numpy.ndarray OR a float. Values must be between 0 and 1
		qc: Quality index of shape (n_lfs,) and type numpy.ndarray OR a float. Values must be between 0 and 1

	'''
	def __init__(self, n_classes, path, metric_avg = ['binary'], n_epochs = 100, lr = 0.01, n_lfs = None, qt = None, qc = None):
		assert type(n_classes) == np.int or type(n_classes) == np.float
		assert type(path) == str
		assert os.path.exists(path)
		for temp in metric_avg:
			assert temp in ['micro', 'macro', 'samples','weighted', 'binary'] or metric_avg is None
		assert type(n_epochs) == np.int or type(n_epochs) == np.float
		assert type(lr) == np.int or type(lr) == np.float
		if n_lfs != None:
			assert type(n_lfs) == np.int
		if type(qt) == float:
			assert qt >= 0 and qt <= 1
		elif type(qt) == np.ndarray:
			assert np.all(np.logical_and(qt>=0, qt<=1))
		elif type(qt) == np.int:
			assert qt == 0 or qt == 1
		else:
			print("Invalid type for qt in Cage class")
			exit(1)

		if n_lfs != None:
			assert type(n_lfs) == np.int
		if type(qc) == float:
			assert qc >= 0 and qc <= 1
		elif type(qc) == np.ndarray:
			assert np.all(np.logical_and(qc>=0, qc<=1))
		elif type(qc) == np.int:
			assert qc == 0 or qc == 1
		else:
			print("Invalid type for qc in Cage class")
			exit(1)

		data = get_data(path)

		assert type(data[1]) == np.ndarray
		assert type(data[6]) == np.ndarray
		assert type(data[7]) == np.ndarray
		assert type(data[8]) == np.ndarray

		self.raw_l = data[1]
		self.l = np.copy(self.raw_l)
		(self.l)[self.l == n_classes] = 0
		(self.l)[self.l != n_classes] = 1

		self.l = torch.abs(torch.tensor(self.l).long())
		self.s = torch.tensor(data[6]).double() # continuous score
		self.n = torch.tensor(data[7]).double() # Mask for s/continuous_mask
		self.k = torch.tensor(data[8]).long() # LF's classes
		self.s[self.s > 0.999] = 0.999 # clip s
		self.s[self.s < 0.001] = 0.001 # clip s

		self.n_classes = int(n_classes)
		self.metric_avg = list(set(metric_avg))
		self.n_epochs = int(n_epochs)
		self.lr = lr
		self.n_lfs = n_lfs if n_lfs != None else self.l.shape[1]

		if n_lfs != none:
			assert self.l.shape[1] == n_lfs
		assert self.l.shape == self.s.shape
		assert self.n.shape == (n_lfs,)
		assert self.k.shape == (n_lfs,)

		self.n_instances, self.n_features = data[0].shape

		self.qt = torch.tensor(qt).double() if qt != None and type(qt) == np.ndarray else \
		((torch.ones(self.n_lfs).double() * qt) if qt != None else (torch.ones(self.n_lfs).double() * 0.9)) 
		self.qc = torch.tensor(qt).double() if qc != None and type(qc) == np.ndarray else (qc if qc != None else 0.85)

		self.pi = torch.ones((n_classes, n_lfs)).double()
		(self.pi).requires_grad = True

		self.theta = torch.ones((n_classes, n_lfs)).double()
		(self.theta).requires_grad = True

	def fit(self, m_test, s_test, path = None):
		'''
		Args:
			s_test: numpy arrays of shape (num_instances, num_rules), s_test[i][j] is the continuous score of jth LF on ith instance
			m_test: numpy arrays of shape (num_instances, num_rules), m_test[i][j] is 1 if jth LF is triggered on ith instance, else it is 0
			path: Path to log file, default value is None. No log is producede if path is None
		Return:
			numpy.ndarray of shape (num_instances,) which are aggregated/predicted labels
		'''
		use_cuda = torch.cuda.is_available()
		device = torch.device("cuda:0" if use_cuda else "cpu")
		torch.backends.cudnn.benchmark = True

		optimizer = optim.Adam([self.theta, self.pi], lr=self.lr, weight_decay=0)

		file = None
		if path != None:
			file = open(path, "a+")

		for epoch in range(self.n_epochs):
			optimizer.zero_grad()
			loss = log_likelihood_loss(self.theta, self.pi, self.l, self.s, self.k, self.n_classes, self.n, self.qc)
			prec_loss = precision_loss(self.theta, self.k, self.n_classes, self.qt)
			loss += prec_loss

			if path != None:
				y_pred = predict(s_test, m_test)
				file.write("Epoch: {}\taccuracy_score: {}".format(epoch, accuracy_score(y_true_test, y_pred)))
				for temp in self.metric_avg:
					file.write("Epoch: {}\tmetric_avg: {}\tf1_score: {}".format(epoch, temp, f1_score(y_true_test, y_pred, average = temp)))

			loss.backward()
			optimizer.step()

		if path != None:
			file.close()

		return predict_gm(self.theta, self.pi, self.l, self.s, self.k, self.n_classes, self.n, self.qc)

	def predict(self, s_test, m_test):
		'''
		Args:
			s_test: numpy arrays of shape (num_instances, num_rules), s_test[i][j] is the continuous score of jth LF on ith instance
			m_test: numpy arrays of shape (num_instances, num_rules), m_test[i][j] is 1 if jth LF is triggered on ith instance, else it is 0
		Return:
			numpy.ndarray of shape (num_instances,) which are predicted labels
			[Note: no aggregration/algorithm-running will be done using the current input]
		'''
		s_temp = torch.tensor(s_test).double()
		s_temp[s_temp > 0.999] = 0.999
		s_temp[s_temp < 0.001] = 0.001
		assert m_test.shape == s_test.shape
		assert m_test.shape[1] == self.n_lfs
		assert np.all(np.logical_or(m_test == 1 or m_test == 0))
		m_temp = torch.abs(torch.tensor(m_test).long())
		return predict_gm(self.theta, self.pi, m_temp, s_temp, self.k, self.n_classes, self.n, self.qc)