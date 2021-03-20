import pickle
import numpy as np 
import torch
from torch.distributions.beta import Beta
import torch.nn as nn
from torch.nn import MultiLabelMarginLoss as mlml
from random import sample


def get_data(path):
	'''
	expected order in pickle file is NUMPY arrays x, l, m, L, d, r, s, n, k
	x: (num_instances, num_features)
	l: (num_instances, num_rules)
	m: (num_instances, num_rules)
	L: (num_instances, 1)
	d: (num_instances, 1)
	r: (num_instances, num_rules)
	s: (num_instances, num_rules)
	n: (num_rules,) Mask for s
	k: (num_rules,) LF classes, range 0 to num_classes-1
	'''
	data = []
	with open(path, 'rb') as file:
		for i in range(9):
			if i == 0:
				data.append(pickle.load(f))
			elif i == 6:
				data.append(pickle.load(f).astype(np.float32))
			else:
				data.append(pickle.load(f).astype(np.int32))
	return data


def probability_y(pi_y):
	pi = torch.exp(pi_y)
	return pi / pi.sum()


def phi(theta, l):
	return theta * torch.abs(l).double()


def calculate_normalizer(theta, k, n_classes):
	'''
	Used to find Z, the normaliser
	input args:
	theta: [n_classes, n_lfs], the parameters
	k: [n_lfs], labels corresponding to LFs
	n_classes: num of classes/labels
	return: a real value, representing the normaliser
	'''
	z = 0
	for y in range(n_classes):
		m_y = torch.exp(phi(theta[y], torch.ones(k.shape)))
		z += (1 + m_y).prod()
	return z


def probability_l_y(theta, l, k, n_classes):
	'''
	Used to find probability involving the term psi_theta, the potential function for all LFs
	input args:
	theta: [n_classes, n_lfs], the parameters
	l: [n_instances, n_lfs], elemets are 0 or 1, 1 indicates that LF is triggered
	k: [n_lfs] labels corresponding to LFs
	n_classes: num of classes/labels
	return: [n_instances, n_classes], the psi_theta value for each instance, for each label(true label y)
	'''
	probability = torch.zeros((l.shape[0], n_classes))
	z = calculate_normalizer(theta, k, n_classes)
	for y in range(n_classes):
		probability[:, y] = torch.exp(phi(theta[y], l).sum(1)) / z

	return probability.double()


def probability_s_given_y_l(pi, s, y, l, k, continuous_mask, qc):
	'''
	Used to find probability involving the term psi_pi, the potential function for all continuous LFs
	input args:
	s: [n_instances, n_lfs], continuous scores
	y: a value in [0, n_classes-1], representing true label, for which psi_pi is calculated
	l: [n_instances, n_lfs], elemets are 0 or 1, 1 indicates that LF is triggered
	k: [n_lfs] labels corresponding to LFs
	continuous_mask: [n_lfs], elements are 0 or 1, 1 implies LF has a continuous counterpart
	qc: a real value or [n_lfs], qc value for each LF
	return:  [n_instances], the psi_pi value for each instance, for the given label(true label y)
	'''
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
	'''
	Used to find probability of given instances for all possible y's
	input args:
	theta: [n_classes, n_lfs], the parameters
	pi_y: [n_classses], the parameters
	pi: [n_classes, n_lfs], the parameters
	l: [n_instances, n_lfs], elemets are 0 or 1, 1 indicates that LF is triggered
	s: [n_instances, n_lfs], continuous scores
	k: [n_lfs] labels corresponding to LFs
	n_classes: num of classes/labels
	continuous_mask: [n_lfs], elements are 0 or 1, 1 implies LF has a continuous counterpart
	qc: a real value or [n_lfs, qc value for each LF
	return: [n_instances, n_classes], the probability for an instance being a particular class
	'''
	p_l_y = probability_l_y(theta, l, k, n_classes)
	p_s = torch.ones(s.shape[0], n_classes).double()
	for y in range(n_classes):
		p_s[:, y] = probability_s_given_y_l(pi[y], s, y, l, k, continuous_mask, qc)
	return p_l_y * p_s


def log_likelihood_loss(theta, pi_y, pi, l, s, k, n_classes, continuous_mask, qc):
	'''
	log likelihood loss
	input args:
	theta: [n_classes, n_lfs], the parameters
	pi_y: [n_classses], the parameters
	pi: [n_classes, n_lfs], the parameters
	l: [n_instances, n_lfs], elemets are 0 or 1, 1 indicates that LF is triggered
	s: [n_instances, n_lfs], continuous scores
	k: [n_lfs] labels corresponding to LFs
	n_classes: num of classes/labels
	continuous_mask: [n_lfs], elements are 0 or 1, 1 implies LF has a continuous counterpart
	qc: a real value or [n_lfs, qc value for each LF
	return: a real value, sigma over (the log of probability for an instance, marginalised over y(true labels))
	'''
	eps = 1e-8
	return - torch.log(probability(theta, pi_y, pi, l, s, k, n_classes, continuous_mask, qc).sum(1) + eps).sum() / s.shape[0]


def precision_loss(theta, k, n_classes, a): 
	'''
	Precison loss, the R(t) term
	input args:
	theta: [n_classes, n_lfs], the parameters
	k: [n_lfs] labels corresponding to LFs
	n_classes: num of classes/labels
	a: [n_lfs], the qt, quality guide of each LF
	return: a real value, R(t) term
	'''
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

def pred_gm(theta, pi_y, pi, l, s, k, n_classes, continuous_mask, qc):
	'''
	pred_gm means predictions from graphical model(cage)
	Used to predict the labels after the training is done
	input args:
	theta: [n_classes, n_lfs], the parameters
	pi_y: [n_classses], the parameters
	pi: [n_classes, n_lfs], the parameters
	l: [n_instances, n_lfs], elemets are 0 or 1, 1 indicates that LF is triggered
	s: [n_instances, n_lfs], continuous scores
	k: [n_lfs] labels corresponding to LFs
	n_classes: num of classes/labels
	continuous_mask: [n_lfs], elements are 0 or 1, 1 implies LF has a continuous counterpart
	qc: a real value or [n_lfs, qc value for each LF
	return: [n_instances], the predicted class for an instance
	'''
	return np.argmax(probability(theta, pi_y, pi, l, s, k, n_classes, continuous_mask, qc).detach().numpy(), 1)

########## below ones are exclusively for core_jl #############

def log_likelihood_loss_supervised(theta, pi_y, pi, y, l, s, k, n_classes, continuous_mask, qc):
	eps = 1e-8
	prob = probability(theta, pi_y, pi, l, s, k, n_classes, continuous_mask, qc)
	prob = (prob.t() / prob.sum(1)).t()
	return torch.nn.NLLLoss()(torch.log(prob), y)

def svm_supervised(w, b, y, x):
	y = torch.tensor(y)
	wx = x.matmul(w) + b
	ywx = (wx*y[:,None]).sum(1)
	sy = torch.tensor([0] * y.shape[0]).double()
	sty = 1-(ywx - wx)
	mx = torch.max(sty, sy)
	return torch.sum(mx, 0)/y.shape[0] + torch.norm(w)*torch.norm(w)


def entropy(probabilities):
	entropy = - (probabilities * torch.log(probabilities)).sum(1)
	return entropy.sum() / probabilities.shape[0]

def entropy_pre(probabilities):
	entropy = - (probabilities * torch.log(probabilities)).sum(1)
	return entropy/ probabilities.shape[0]


def cross_entropy(probabilities, y):
	return - torch.log(probabilities[:, y].sum() / y.shape[0])


def kl_divergence(probs_p, probs_q):
	return (probs_p * torch.log(probs_p / probs_q)).sum() / probs_p.shape[0]

def vat_loss(model, x, y, xi=1e-6, eps=2.5, n_iters=1):
	d = torch.Tensor(x.shape).double().normal_()
	for i in range(n_iters):
		d = xi * _l2_normalize(d)
		d = Variable(d, requires_grad=True)
		y_hat = model(x + d)
		delta_kl = kl_div_with_logit(y.detach(), y_hat)
		delta_kl.backward()

		d = d.grad.data.clone().cpu()
		model.zero_grad()

	d = _l2_normalize(d)
	d = Variable(d)
	r_adv = eps * d
	# compute lds
	y_hat = model(x + r_adv.detach())
	delta_kl = kl_div_with_logit(y.detach(), y_hat)
	return delta_kl


def kl_div_with_logit(q_logit, p_logit):

	q = F.softmax(q_logit, dim=1)
	logq = F.log_softmax(q_logit, dim=1)
	logp = F.log_softmax(p_logit, dim=1)

	qlogq = ( q *logq).sum(dim=1).mean(dim=0)
	qlogp = ( q *logp).sum(dim=1).mean(dim=0)

	return qlogq - qlogp

def getDiverseInstances(ent, budget,n_lfs, count): #ent is dict of {lfs, [indices]}
	if count < budget :
		print("budget cannot exceed total instances")
		return 
	each = int(budget/n_lfs)
	print('each is ', each)
	bud = each * n_lfs
	indic = []
	for j in ent.keys():
		if each > len(ent[j]):
			indic.extend(ent[j])
		else:
			x = sample(list(ent[j]), each)
			indic.extend(x)
	return indic