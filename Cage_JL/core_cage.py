import torch
from torch import optim
import numpy as np
import os

from utils import *

class Cage:
	'''
	Cage class:
	constructor args: n_classes, path, n_epochs = 100, lr = 0.01, n_lfs = None, qt = None, qc = None
	where
	n_classes: number of classes/labels, type is integer
	path: path to pickle file
	n_epochs:number of epochs
	lr: learning rate
	n_lfs: number of LFs
	qt is quality guide of shape (n_lfs,) and type numpy OR a float. Values must be between 0 and 1.
	qc is average score of s when there is an aggrement of shape (n_lfs,) and type numpy OR a float. Values must be between 0 and 1.
	'''
	def __init__(self, n_classes, path, n_epochs = 100, lr = 0.01, n_lfs = None, qt = None, qc = None):

		assert type(n_classes) == np.int or type(n_classes) == np.float
		assert type(path) == str
		assert os.path.exists(path)
		assert type(n_epochs) == np.int or type(n_epochs) == np.float
		assert type(lr) == np.int or type(lr) == np.float #todo: range for lr?
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
		self.n_epochs = int(n_epochs)
		self.lr = lr
		self.n_lfs = n_lfs if n_lfs != None else self.l.shape[1]

		if n_lfs != none:
			assert self.l.shape[1] = n_lfs
		assert self.l.shape == self.s.shape
		assert self.n.shape == (n_lfs,)
		assert self.k.shape == (n_lfs,)

		self.n_instances, self.n_features = data[0].shape

		self.qt = torch.tensor(qt).double() if qt != None and type(qt) == np.ndarray else \
		((torch.ones(self.n_lfs).double() * qt) if qt != None else (torch.ones(self.n_lfs).double() * 0.9)) 
		self.qc = torch.tensor(qt).double() if qc != None and type(qc) == np.ndarray else (qc if qc != None else 0.85)

		self.pi = torch.ones((n_classes, n_lfs)).double()
		(self.pi).requires_grad = True

		self.theta = torch.ones((n_classes, n_lfs)).double() * 1 #todo: 1 needed?
		(self.theta).requires_grad = True

		self.pi_y = torch.ones(n_classes).double()
		(self.pi_y).requires_grad = True

	def fit():
		'''
		input args: none
		return: numpy array of shape (num_instances,) which are aggregated/predicted labels
		'''
		optimizer = optim.Adam([self.theta, self.pi, self.pi_y], lr=self.lr, weight_decay=0) #todo: need to look about weight decay, other choices than adam
		for epoch in range(self.n_epochs):
			optimizer.zero_grad()
			loss = log_likelihood_loss(self.theta, self.pi_y, self.pi, self.l, self.s, self.k, self.n_classes, self.n, self.qc)
			prec_loss = precision_loss(self.theta, self.k, self.n_classes, self.qt)
			loss += prec_loss
			loss.backward()
			optimizer.step()
		return pred_gm(self.theta, self.pi_y, self.pi, self.l, self.s, self.k, self.n_classes, self.n, self.qc)

	def predict(l_test, s_test):
		'''
		input args:
		l_test, s_test: numpy arrays of shape (num_instances, num_rules)
		return: numpy array of shape (num_instances,) which are predicted labels
		(Note: no aggregration/algorithm-running will be done using the current input)
		'''
		assert l_test.shape == s_test.shape
		assert l_test.shape[1] == n_lfs

		s_temp = torch.tensor(s_test).double()
		s_temp[s_temp > 0.999] = 0.999
		s_temp[s_temp < 0.001] = 0.001
		l_temp = l_test
		(l_temp)[l_temp == n_classes] = 0
		(l_temp)[l_temp != n_classes] = 1
		l_temp = torch.abs(torch.tensor(l_temp).long())
		return pred_gm(self.theta, self.pi_y, self.pi, l_temp, s_temp, self.k, self.n_classes, self.n, self.qc)