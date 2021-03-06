import pickle
import numpy as np 
import torch
from torch.distributions.beta import Beta

def get_data(path):
	'''
	expected order in pickle file is NUMPY arrays x, l, m, L, d, r, s, n, k
	x: [num_instances, num_features]
	l: [num_instances, num_rules]
	m: [num_instances, num_rules]
	L: [num_instances, 1]
	d: [num_instances, 1]
	r: [num_instances, num_rules]
	s: [num_instances, num_rules]
	n: [num_rules] Mask for s
	k: [num_rules] LF classes, range 0 to num_classes-1
	'''
	data = []
	with open(path, 'rb') as file:
		contents = pickle.load(file)
		for i in range(9):
			if i == 0:
				data.append(pickle.load(f))
			if i == 6:
				data.append(pickle.load(f).astype(np.float32))
			data.append(pickle.load(f).astype(np.int32))
	return data


def probability_y(pi_y):
	pi = torch.exp(pi_y)
	return pi / pi.sum()


def phi(theta, l):
	return theta * torch.abs(l).double()


def calculate_normalizer(theta, k, n_classes):
	z = 0
	for y in range(n_classes):
		m_y = torch.exp(phi(theta[y], torch.ones(k.shape)))
		z += (1 + m_y).prod()
	return z


def probability_l_y(theta, l, k, n_classes):
	probability = torch.zeros((l.shape[0], n_classes))
	z = calculate_normalizer(theta, k, n_classes)
	for y in range(n_classes):
		probability[:, y] = torch.exp(phi(theta[y], l).sum(1)) / z

	return probability.double()


def probability_s_given_y_l(pi, s, y, l, k, continuous_mask, qc):
	eq = torch.eq(k.view(-1, 1), y).double().t()
	r = qc * eq.squeeze() + (1 - qc) * (1 - eq.squeeze())
	params = torch.exp(pi)
	probability = 1
	for i in range(k.shape[0]):
		m = Beta(r[i] * params[i], params[i] * (1 - r[i]))
		probability *= torch.exp(m.log_prob(s[:, i].double())) * l[:, i].double() * continuous_mask[i] \
		+ (1 - l[:, i]).double() + (1 - continuous_mask[i])
	return probability


def probability(theta, pi_y, pi, l, s, k, n_classes, continuous_mask, qc):
	p_l_y = probability_l_y(theta, l, k, n_classes)
	p_s = torch.ones(s.shape[0], n_classes).double()
	for y in range(n_classes):
		p_s[:, y] = probability_s_given_y_l(pi[y], s, y, l, k, continuous_mask, qc)
	return p_l_y * p_s


def log_likelihood_loss(theta, pi_y, pi, l, s, k, n_classes, continuous_mask, qc):
	eps = 1e-8
	return - torch.log(probability(theta, pi_y, pi, l, s, k, n_classes, continuous_mask, qc).sum(1) + eps).sum() / s.shape[0]


def precision_loss(theta, k, n_classes, a): # a is qt, the quality guide
	n_lfs = k.shape[0]
	prob = torch.ones(n_lfs, n_classes).double()
	z_per_lf = 0
	for y in range(n_classes):
		m_y = torch.exp(phi(theta[y], torch.ones(n_lfs)))
		per_lf_matrix = torch.tensordot((1 + m_y).view(-1, 1), torch.ones(m_y.shape).double().view(1, -1), 1) \
		- torch.eye(n_lfs).double()
		prob[:, y] = per_lf_matrix.prod(0).double()
		z_per_lf += prob[:, y].double()
	prob /= z_per_lf.view(-1, 1)
	correct_prob = torch.zeros(n_lfs)
	for i in range(n_lfs):
		correct_prob[i] = prob[i, k[i]]
	loss = a * torch.log(correct_prob).double() + (1 - a) * torch.log(1 - correct_prob).double()
	return -loss.sum()