import torch
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from utils import *
from model_feature import *


class Joint_Learning:
	'''
	Joint Learning class
	constructor args:
	n_classes: number of classes/labels, type is integer
	path_L: path to pickle file of labelled instances
	path_U: path to pickle file of unlabelled instances
	path_V: path to pickle file of validation instances
	path_T: path to pickle file of test instances
	(Note: each pickle file should follow the standard convention(can be found in __doc__ of get_data in utils.py) for data storage)
	loss_func_mask: list/numpy array of size 7 or (7,) where loss_func_mask[i] should be 1 if Loss function (i+1) should be included, note Loss function 7 is quality guide(qt)
	feature_model: the model intended to be used for features, allowed values are 'lr' or 'nn'
	is_qt: True if quality guide is available. False if quality guide is intended to be found from validation instances
	is_qc: True if qc is available. False if qc is intended to be found from validation instances
	batch_size: batch size, type is integer
	lr_feature: learning rate for feature model, type is integer or float
	lr_gm: learning rate for graphical model(cage), type is integer or float
	use_accuracy_score: True for accuracy_score, False for f1_score
	metric_avg: average metric to be used in calculating f1_score/precision/recall
	qt: quality guide of shape (n_lfs,) and type numpy OR a float. Values must be between 0 and 1.
	qc: average score of s when there is an aggrement of shape (n_lfs,) and type numpy OR a float. Values must be between 0 and 1.
	n_hidden: the number of hidden layer nodes if feature model is 'nn', type is integer
	n_epochs: number of epochs in each run, type is integer
	n_runs: number of runs ,type is integer
	'''
	def __init__(self, n_classes, path_L, path_U, path_V, path_T , loss_func_mask, feature_model, is_qt, is_qc, batch_size, lr_feature, lr_gm, use_accuracy_score, metric_avg = 'macro', qt = None, qc = None, n_hidden = 512, n_epochs = 100, n_runs = 10):
		
		assert type(n_classes) == np.int or type(n_classes) == np.float
		assert type(path_L) == str and type(path_V) == str and type(path_V) == str and type(path_T) == str and type(metric_avg) == str
		assert os.path.exists(path_L) and os.path.exists(path_U) and os.path.exists(path_V) and os.path.exists(path_T)
		assert len(loss_func_mask) == 7
		assert feature_model == 'lr' or feature_model == 'nn'
		assert type(is_qt) == np.bool and type(is_qc) == np.bool and type(use_accuracy_score) == np.bool
		assert type(batch_size) == np.int or type(batch_size) == np.float
		assert type(lr_feature) == np.int or type(lr_feature) == np.float
		assert type(lr_gm) == np.int or type(lr_gm) == np.float

		if n_lfs != None:
			assert type(n_lfs) == np.int
		if type(qt) == float:
			assert qt >= 0 and qt <= 1
		elif type(qt) == np.ndarray:
			assert np.all(np.logical_and(qt>=0, qt<=1))
		elif type(qt) == np.int:
			assert qt == 0 or qt == 1
		else:
			print("Invalid type for qt in Joint_Learning class")
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
			print("Invalid type for qc in Joint_Learning class")
			exit(1)

		assert type(n_hidden) == np.int or type(n_hidden) == np.float
		assert type(n_epochs) == np.int or type(n_epochs) == np.float
		assert type(n_runs) == np.int or type(n_runs) == np.float

		torch.set_default_dtype(torch.float64)
		self.n_classes = int(n_classes)
		self.loss_func_mask = loss_func_mask
		self.feature_model = feature_model
		self.batch_size = int(batch_size)
		self.n_hidden = int(n_hidden)
		self.lr_feature = lr_feature
		self.lr_gm = lr_gm
		self.use_accuracy_score = use_accuracy_score
		self.metric_avg = metric_avg
		self.n_epochs = int(n_epochs)
		self.n_runs = int(n_runs)

		if self.use_accuracy_score:
			from sklearn.metrics import accuracy_score as score
		else:
			from sklearn.metrics import f1_score as score
			from sklearn.metrics import precision_score as prec_score
			from sklearn.metrics import recall_score as recall_score


		data_L = get_data(path_L)
		data_U = get_data(path_U)
		data_V = get_data(path_V)
		data_T = get_data(path_T)

		self.x_sup = torch.tensor(data_L[0]).double()
		self.y_sup = torch.tensor(data_L[3]).long()
		self.l_sup = torch.tensor(data_L[2]).long()
		self.s_sup = torch.tensor(data_L[6]).double()

		excluding = []
		temp_index = 0
		for temp in data_U[1]:
			if(np.all(x == int(self.n_classes)) ):
				excluding.append(index_temp)
			index_temp+=1

		self.x_unsup = torch.tensor(np.delete(data_U[0], excluding, axis=0)).double()
		self.y_unsup = torch.tensor(np.delete(data_U[3], excluding, axis=0)).long()
		self.l_unsup = torch.tensor(np.delete(data_U[2], excluding, axis=0)).long()
		self.s_unsup = torch.tensor(np.delete(data_U[6], excluding, axis=0)).double()

		self.x_valid = torch.tensor(data_V[0]).double()
		self.y_valid = data_V[3]
		self.l_valid = torch.tensor(data_V[2]).long()
		self.s_valid = torch.tensor(data_V[6]).double()

		self.x_test = torch.tensor(data_T[0]).double()
		self.y_test = data_T[3]
		self.l_test = torch.tensor(data_T[2]).long()
		self.s_test = torch.tensor(data_T[6]).double()

		self.n_features = self.x_sup.shape[1]
		self.k = torch.tensor(data_L[8]).long() # LF's classes
		self.n_lfs = self.l_sup.shape[1]
		self.continuous_mask = torch.tensor(data_L[7]).double() # Mask for s/continuous_mask

		if is_qt:
			self.qt = torch.tensor(qt).double() if qt != None and type(qt) == np.ndarray else \
			((torch.ones(self.n_lfs).double() * qt) if qt != None else (torch.ones(self.n_lfs).double() * 0.9)) 
		else:
			prec_lfs=[]
			for i in range(self.n_lfs):
				correct = 0
				for j in range(len(y_valid)):
					if y_valid[j] == l_valid[j][i]:
						correct+=1
				prec_lfs.append(correct/len(y_valid))
			self.qt = torch.tensor(prec_lfs).double()

		if is_qg:
			self.qc = torch.tensor(qt).double() if qc != None and type(qc) == np.ndarray else (qc if qc != None else 0.85)
		else:
			self.qc = torch.tensor(np.mean(s_valid, axis = 0))

		# clip s
		self.s_sup[self.s_sup > 0.999] = 0.999
		self.s_sup[self.s_sup < 0.001] = 0.001
		self.s_unsup[self.s_unsup > 0.999] = 0.999
		self.s_unsup[self.s_unsup < 0.001] = 0.001
		self.s_valid[self.s_valid > 0.999] = 0.999
		self.s_valid[self.s_valid < 0.001] = 0.001
		self.s_test[self.s_test > 0.999] = 0.999
		self.s_test[self.s_test < 0.001] = 0.001

		self.l = torch.cat([l_sup, l_unsup])
		self.s = torch.cat([s_sup, s_unsup])
		self.x_train = torch.cat([x_sup, x_unsup])
		self.y_train = torch.cat([y_sup, y_unsup])
		self.supervised_mask = torch.cat([torch.ones(l_sup.shape[0]), torch.zeros(l_unsup.shape[0])])

		self.pi = torch.ones((self.n_classes, self.n_lfs)).double()
		self.theta = torch.ones((self.n_classes, self.n_lfs)).double() * 1
		self.pi_y = torch.ones(self.n_classes).double()

		if self.feature_model == 'lr':
			self.lr_model = LogisticRegression(self.n_features, self.n_classes)
		elif self.feature_model =='nn':
			self.lr_model = DeepNet(n_features, self.n_hidden, self.n_classes)

	def fit():
		'''
		no input args
		return: two predicted labels of numpy array of shape (num_instances,). first one is through gm(cage), other one through feature model
		'''
		final_score_gm, final_score_lr, final_score_gm_val, final_score_lr_val = [],[],[],[]
		final_score_lr_prec, final_score_lr_recall, final_score_gm_prec, final_score_gm_recall = [],[],[],[]
		for lo in range(0,self.n_runs):
			self.pi = torch.ones((self.n_classes, self.n_lfs)).double()
			(self.pi).requires_grad = True

			self.theta = torch.ones((self.n_classes, self.n_lfs)).double() * 1
			(self.theta).requires_grad = True

			self.pi_y = torch.ones(self.n_classes).double()
			(self.pi_y).requires_grad = True
			
			optimizer = torch.optim.Adam([{"params": self.lr_model.parameters()}, {"params": [self.pi, self.pi_y, self.theta]}], lr=0.001)
			optimizer_lr = torch.optim.Adam(self.lr_model.parameters(), lr = self.lr_feature)
			optimizer_gm = torch.optim.Adam([self.theta, self.pi, self.pi_y], lr = self.lr_gm, weight_decay=0)
			supervised_criterion = torch.nn.CrossEntropyLoss()

			dataset = TensorDataset(self.x_train, self.y_train, self.l, self.s, self.supervised_mask)

			loader = DataLoader(dataset, batch_size = self.batch_size, shuffle = True, pin_memory = True)

			best_score_lr,best_score_gm,best_epoch_lr,best_epoch_gm,best_score_lr_val, best_score_gm_val = 0,0,0,0,0,0
			best_score_lr_prec,best_score_lr_recall ,best_score_gm_prec,best_score_gm_recall= 0,0,0,0

			stop_early, stop_early_gm = [], []

			for epoch in range(self.n_epochs):
				self.lr_model.train()

				for batch_ndx, sample in enumerate(loader):
					optimizer_lr.zero_grad()
					optimizer_gm.zero_grad()

					unsup = []
					sup = []
					supervised_indices = sample[4].nonzero().view(-1)
					# unsupervised_indices = indices  ## Uncomment for entropy
					unsupervised_indices = (1-sample[4]).nonzero().squeeze()


					if(self.loss_func_mask[0]):
						if len(supervised_indices) > 0:
							loss_1 = supervised_criterion(self.lr_model(sample[0][supervised_indices]), sample[1][supervised_indices])
						else:
							loss_1 = 0
					else:
						loss_1=0

					if(self.loss_func_mask[1]):
						unsupervised_lr_probability = torch.nn.Softmax()(self.lr_model(sample[0][unsupervised_indices]))
						loss_2 = entropy(unsupervised_lr_probability)
					else:
						loss_2=0

					if(self.loss_func_mask[2]):
						y_pred_unsupervised = pred_gm(self.theta, self.pi_y, self.pi, sample[2][unsupervised_indices], sample[3][unsupervised_indices], self.k, self.n_classes, self.continuous_mask, self.qc)
						loss_3 = supervised_criterion(self.lr_model(sample[0][unsupervised_indices]), torch.tensor(y_pred_unsupervised))
					else:
						loss_3 = 0

					if (self.loss_func_mask[3] and len(supervised_indices) > 0):
						loss_4 = log_likelihood_loss_supervised(self.theta, self.pi_y, self.pi, sample[1][supervised_indices], sample[2][supervised_indices], sample[3][supervised_indices], self.k, self.n_classes, self.continuous_mask, self.qc)
					else:
						loss_4 = 0

					if(self.loss_func_mask[4]):
						loss_5 = log_likelihood_loss(self.theta, self.pi_y, self.pi, sample[2][unsupervised_indices], sample[3][unsupervised_indices], self.k, self.n_classes, self.continuous_mask, self.qc)
					else:
						loss_5 =0

					if(self.loss_func_mask[5]):
						if(len(supervised_indices) >0):
							supervised_indices = supervised_indices.tolist()
							probs_graphical = probability(self.theta, self.pi_y, self.pi, torch.cat([sample[2][unsupervised_indices], sample[2][supervised_indices]]),\
							torch.cat([sample[3][unsupervised_indices],sample[3][supervised_indices]]), self.k, self.n_classes, self.continuous_mask, self.qc)
						else:
							probs_graphical = probability(self.theta, self.pi_y, self.pi,sample[2][unsupervised_indices],sample[3][unsupervised_indices],\
								self.k, self.n_classes, self.continuous_mask, self.qc)
						probs_graphical = (probs_graphical.t() / probs_graphical.sum(1)).t()
						probs_lr = torch.nn.Softmax()(self.lr_model(sample[0]))
						#loss_6 = kl_divergence(probs_lr, probs_graphical) # todo: include experiment?
						loss_6 = kl_divergence(probs_graphical, probs_lr) #original version
					else:
						loss_6= 0

					if(self.loss_func_mask[6]):
						prec_loss = precision_loss(self.theta, self.k, self.n_classes, self.qt)
					else:
						prec_loss =0

					loss = loss_1 + loss_2 + loss_3 + loss_4 + loss_6+loss_5 + prec_loss
					if loss != 0:
						loss.backward()
						optimizer_gm.step()
						optimizer_lr.step()

				#gm Test
				y_pred = pred_gm(self.theta, self.pi_y, self.pi, self.l_test, self.s_test, self.k, self.n_classes, self.continuous_mask, self.qc)
				if self.use_accuracy_score:
					gm_acc = score(self.y_test, y_pred)
				else:
					gm_acc = score(self.y_test, y_pred, average = self.metric_avg)
					gm_prec = prec_score(self.y_test, y_pred, average = self.metric_avg)
					gm_recall = recall_score(self.y_test, y_pred, average = self.metric_avg)

				#gm Validation
				y_pred = pred_gm(self.theta, self.pi_y, self.pi, self.l_valid, self.s_valid, self.k, self.n_classes, self.continuous_mask, self.qc)
				if self.use_accuracy_score:
					gm_valid_acc = score(self.y_valid, y_pred)
				else:
					gm_valid_acc = score(self.y_valid, y_pred, average = self.metric_avg)

				#LR Test
				probs = torch.nn.Softmax()(self.lr_model(self.x_test))
				y_pred = np.argmax(probs.detach().numpy(), 1)
				if self.use_accuracy_score:
					lr_acc =score(self.y_test, y_pred)
				else:
					lr_acc =score(self.y_test, y_pred, average = self.metric_avg)
					lr_prec = prec_score(self.y_test, y_pred, average = self.metric_avg)
					lr_recall = recall_score(self.y_test, y_pred, average = self.metric_avg)

				#LR Validation
				probs = torch.nn.Softmax()(self.lr_model(self.x_valid))
				y_pred = np.argmax(probs.detach().numpy(), 1)
				if self.use_accuracy_score:
					lr_valid_acc = score(self.y_valid, y_pred)
				else:
					lr_valid_acc = score(self.y_valid, y_pred, average = self.metric_avg)

				if epoch > 5 and gm_valid_acc >= best_score_gm_val and gm_valid_acc >= best_score_lr_val:
					if gm_valid_acc == best_score_gm_val or gm_valid_acc == best_score_lr_val:
						if best_score_gm < gm_acc or best_score_lr < lr_acc:
							best_epoch_lr = epoch
							best_score_lr_val = lr_valid_acc
							best_score_lr = lr_acc

							best_epoch_gm = epoch
							best_score_gm_val = gm_valid_acc
							best_score_gm = gm_acc

							best_score_lr_prec = lr_prec
							best_score_lr_recall  = lr_recall
							best_score_gm_prec = gm_prec
							best_score_gm_recall  = gm_recall
					else:
						best_epoch_lr = epoch
						best_score_lr_val = lr_valid_acc
						best_score_lr = lr_acc

						best_epoch_gm = epoch
						best_score_gm_val = gm_valid_acc
						best_score_gm = gm_acc

						best_score_lr_prec = lr_prec
						best_score_lr_recall  = lr_recall
						best_score_gm_prec = gm_prec
						best_score_gm_recall  = gm_recall
						stop_early = []
						stop_early_gm = []
					#checkpoint = {'theta': theta,'pi': pi}
					# torch.save(checkpoint, save_folder+"/gm_"+str(epoch)    +".pt")
					#checkpoint = {'params': self.lr_model.state_dict()}
					# torch.save(checkpoint, save_folder+"/lr_"+ str(epoch)+".pt")
					

				if epoch > 5 and lr_valid_acc >= best_score_lr_val and lr_valid_acc >= best_score_gm_val:
					if lr_valid_acc == best_score_lr_val or lr_valid_acc == best_score_gm_val:
						if best_score_lr < lr_acc or best_score_gm < gm_acc:
							
							best_epoch_lr = epoch
							best_score_lr_val = lr_valid_acc
							best_score_lr = lr_acc

							best_epoch_gm = epoch
							best_score_gm_val = gm_valid_acc
							best_score_gm = gm_acc

							best_score_lr_prec = lr_prec
							best_score_lr_recall  = lr_recall
							best_score_gm_prec = gm_prec
							best_score_gm_recall  = gm_recall
					else:
						best_epoch_lr = epoch
						best_score_lr_val = lr_valid_acc
						best_score_lr = lr_acc
						best_epoch_gm = epoch
						best_score_gm_val = gm_valid_acc
						best_score_gm = gm_acc
						best_score_lr_prec = lr_prec
						best_score_lr_recall  = lr_recall
						best_score_gm_prec = gm_prec
						best_score_gm_recall  = gm_recall
						stop_early = []
						stop_early_gm = []
					#checkpoint = {'theta': theta,'pi': pi}
					# torch.save(checkpoint, save_folder+"/gm_"+str(epoch)    +".pt")
					#checkpoint = {'params': self.lr_model.state_dict()}
					# torch.save(checkpoint, save_folder+"/lr_"+ str(epoch)+".pt")


				if len(stop_early) > 10 and len(stop_early_gm) > 10 and (all(best_score_lr_val >= k for k in stop_early) or \
				all(best_score_gm_val >= k for k in stop_early_gm)):
					print('Early Stopping at', best_epoch_gm, best_score_gm, best_score_lr)
					print('Validation score Early Stopping at', best_epoch_gm, best_score_lr_val, best_score_gm_val)
					break
				else:
					stop_early.append(lr_valid_acc)
					stop_early_gm.append(gm_valid_acc)

			print('Best Epoch LR', best_epoch_lr)
			print('Best Epoch GM', best_epoch_gm)
			print("Run \t",lo, "Epoch, GM, LR \t", best_score_gm, best_score_lr)
			print("Run \t",lo, "GM Val, LR Val \t", best_score_gm_val, best_score_lr_val)
			final_score_gm.append(best_score_gm)
			final_score_lr.append(best_score_lr)
			final_score_lr_prec.append(best_score_lr_prec)
			final_score_lr_recall.append(best_score_lr_recall)

			final_score_gm_prec.append(best_score_gm_prec)
			final_score_gm_recall.append(best_score_gm_recall)

			final_score_gm_val.append(best_score_gm_val)
			final_score_lr_val.append(best_score_lr_val)


		print("===================================================")
		print("TEST Averaged scores are for LR", np.mean(final_score_lr))
		print("TEST Precision average scores are for LR", np.mean(final_score_lr_prec))
		print("TEST Recall average scores are for LR", np.mean(final_score_lr_recall))
		print("===================================================")
		print("TEST Averaged scores are for GM",  np.mean(final_score_gm))
		print("TEST Precision average scores are for GM", np.mean(final_score_gm_prec))
		print("TEST Recall average scores are for GM", np.mean(final_score_gm_recall))
		print("===================================================")
		print("VALIDATION Averaged scores are for GM,LR", np.mean(final_score_gm_val), np.mean(final_score_lr_val))
		print("TEST STD  are for GM,LR", np.std(final_score_gm), np.std(final_score_lr))
		print("VALIDATION STD  are for GM,LR", np.std(final_score_gm_val), np.std(final_score_lr_val))

	return pred_gm(self.theta, self.pi_y, self.pi, self.l_unsup, self.s_unsup, self.k, self.n_classes, self.continuous_mask, self.qc),\
	 np.argmax((torch.nn.Softmax()(self.lr_model(self.x_unsup))).detach().numpy(), 1)