from my_utils import get_data

class Implyloss:
	def __init__(self,data):
		self.x=data[0]
		self.l=data[1]
		self.m=data[2]
		self.L=data[3]
		self.d=data[4]
		self.r=data[5]
		self.s=data[6]
		self.n=data[7]
		self.k=data[8]
	
	# need to define get_weights_and_logits, f_network, 
	# joint_scores_from_f_and_w, softmax_cross_entropy_with_logits,
	# compute_LL_phi
	def optimize(self):
		# w_network
		# weights: [batch_size, num_rules]
    	# w_logits: [batch_size, num_rules]
		weights, w_logits = self.get_weights_and_logits(self.x)
		self.weights = weights

		# f_network
		f_dict = {'x': self.x}
        f_logits = self.f_network(f_dict, self.num_classes, reuse=True, dropout_keep_prob=self.dropout_keep_prob)
        self.probs = tf.math.softmax(f_logits, axis=-1)
        self.preds = tf.argmax(self.f_probs, axis=-1)
        self.joint_f_w_score = self.joint_scores_from_f_and_w(self.weights,self.m,self.probs)

        # Do this so that the cross-entropy does not blow for data from U
        # The actual value of cross-entropy for U does not matter since it
        # will be multiplied by 0 anyway.        
        L = L % self.num_classes

        # LL_theta term which is on d data
        L_one_hot = tf.one_hot(L, self.num_classes)
        LL_theta = tf.nn.softmax_cross_entropy_with_logits(logits=f_logits,
                labels=L_one_hot)
        LL_theta = d * LL_theta
        LL_theta = tf.reduce_mean(LL_theta) # loss of f network on labeled data d
        # loss of f network on labeled data d
        # first term in eqn 5 (LL(\theta))


        # LL(\phi) term
        LL_phi = self.compute_LL_phi(w_logits=w_logits, weights=self.weights, l=self.l, m=self.m, L=self.L, d=self.d, r=self.r)
        
        self.adam_lr = tf.placeholder(tf.float32,name='adam_lr')
        f_cross_training_optimizer = tf.train.AdamOptimizer(learning_rate=self.adam_lr, name='adam')

        training_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        # need to write loss functions

if __name__ == __main__:
	data = get_data(path) # path will be the path of pickle file
	Il = Implyloss(data)