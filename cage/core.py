import torch
from torch import optim
import numpy as np

from utils import *

#todo: using np arguments, thoughts?
#todo: need to keep assert for type checks?

class Cage:
	'''
	constructor args: n_classes, path, n_epochs = 100, lr = 0.01, n_lfs = None, qt = None, qc = None
	where
	path is pickle file path obtained after applying labeling functions
	qt is quality guide of shape (n_lfs,) and type numpy OR a float 
	qc is average score of s when there is an aggrement of shape (n_lfs,) and type numpy OR a float
	'''
	def __init__(self, n_classes, path, n_epochs = 100, lr = 0.01, n_lfs = None, qt = None, qc = None):
		data = get_data(path)
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

		self.n_classes = n_classes
		self.n_epochs = n_epochs
		self.lr = lr
		self.n_lfs = n_lfs if n_lfs != None else self.l.shape[1]

		self.qt = torch.tensor(qt).double() if qt != None and type(qt) == np.ndarray else \
		((torch.ones(self.n_lfs).double() * qt) if qt != None else (torch.ones(self.n_lfs).double() * 0.9)) 
		self.qc = torch.tensor(qt).double() if qc != None and type(qc) == np.ndarray else (qc if qc != None else 0.85)

		self.l = torch.abs((self.l).long())
		self.s = torch.abs((self.s).double())
		self.k = torch.tensor(self.k)

		self.pi = torch.ones((n_classes, n_lfs)).double()
		(self.pi).requires_grad = True

		self.theta = torch.ones((n_classes, n_lfs)).double() * 1
		(self.theta).requires_grad = True

		self.pi_y = torch.ones(n_classes).double()
		(self.pi_y).requires_grad = True

	def fit():
		'''
		return: numpy array of shape (num_instances,) which are aggregated/predicted labels
		'''
		optimizer = optim.Adam([self.theta, self.pi, self.pi_y], lr=self.lr, weight_decay=0) #todo: need to look about weight decay, choice of adam
		for epoch in range(self.n_epochs):
			optimizer.zero_grad()
			loss = log_likelihood_loss(self.theta, self.pi_y, self.pi, self.l, self.s, self.k, self.n_classes, self.n, self.qc)
			prec_loss = precision_loss(self.theta, self.k, self.n_classes, self.qt)
			loss += prec_loss
			loss.backward()
			optimizer.step()
		return np.argmax(probability(self.theta, self.pi_y, self.pi, self.l, self.s, self.k,\
		 self.n_classes, self.n, self.qc).detach().numpy(), 1)

	def predict(l_test, s_test):
		'''
		l_test, s_test: numpy arrays of shape (num_instances, num_rules)
		return: numpy array of shape (num_instances,) which are predicted labels
		Note: no aggregration/algorithm-running will be done using the current input
		'''
		s_temp = torch.tensor(s_test).double()
		s_temp[s_temp > 0.999] = 0.999
		s_temp[s_temp < 0.001] = 0.001
		l_temp = l_test
		(l_temp)[l_temp == n_classes] = 0
		(l_temp)[l_temp != n_classes] = 1
		l_temp = torch.abs(torch.tensor(l_temp).long())
		return np.argmax(probability(self.theta, self.pi_y, self.pi, l_temp,\
		 s_temp, self.k, self.n_classes, self.n, self.qc).detach().numpy(), 1)