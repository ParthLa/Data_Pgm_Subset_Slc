from my_utils import get_data

class Implyloss:
	def __init__(self,data, num_classes):
		'''
        * x : feature representation of instance
            - shape : [batch_size, num_features]

        * l : Labels assigned by rules
            - shape [batch_size, num_rules]
            - l[i][j] provides the class label provided by jth rule on ith instance
            - if jth rule does not fire on ith instance, then l[i][j] = num_classes (convention)
            - in snorkel, convention is to keep l[i][j] = -1, if jth rule doesn't cover ith instance
            - class labels belong to {0, 1, 2, .. num_classes-1}

        * m : Rule coverage mask
            - A binary matrix of shape [batch_size, num_rules]
            - m[i][j] = 1 if jth rule cover ith instance
            - m[i][j] = 0 otherwise

        * L : Instance labels
            - shape : [batch_size, 1]
            - L[i] = label of ith instance, if label is available i.e. if instance is from labeled set d
            - Else, L[i] = num_clases if instances comes from the unlabeled set U
            - class labels belong to {0, 1, 2, .. num_classes-1}

        * d : binary matrix of shape [batch_size, 1]
            - d[i] = 1 if instance belongs to labeled data (d), d[i]=0 otherwise
            - d[i]=1 for all instances is from d_processed.p
            - d[i]=0 for all instances in other 3 pickles {U,validation,test}_processed.p
        
        * r : A binary matrix of shape [batch_size, num_rules]
            - r[i][j]=1 if jth rule was associated with ith instance
            - Highly sparse matrix
            - r is a 0 matrix for all the pickles except d_processed.p
            - Note that this is different from rule coverage matrix "m"
            - This matrix defines the rule,example pairs provided as supervision 

        * s : A similarity measure matrix shape [num_instances, num_rules]
        	- s[i][j] is in [0,1]

        * n : A vector of size [num_rules,]
        	- Mask for s (denotes whether particular rule is continuous or discrete)

        * k : a vector of size [num_rules,]
        	- #LF classes ie., what class each LF correspond to, range: 0 to num_classes-1
        '''
		self.x=data[0]
		self.l=data[1]
		self.m=data[2]
		self.L=data[3]
		self.d=data[4]
		self.r=data[5]
		self.s=data[6]
		self.n=data[7]
		self.k=data[8]
		self.num_features=self.x[1] # 1st dimension of x
		self.num_rules_to_train=self.l[1]
		self.num_classes = num_classes
		self.w_network = functools.partial(w_network, self.w_var_scope, self.num_rules)
		self.f_network = functools.partial(f_network, self.f_var_scope)
	
	# need to define f_network, 
	# softmax_cross_entropy_with_logits,
	# compute_LL_phi

	def compute_LL_phi(self, w_logits, weights, l, m, L, d, r):
        psi = 1e-25
        L = tf.expand_dims(L,1)
        # [batch_size, num_rules]
        L = tf.tile(L,[1,self.num_rules])
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.to_float(tf.equal(l,L)),
                                                       logits=w_logits)
        loss = m*loss
        loss = (tf.to_float(tf.not_equal(l,L)) * loss) + (tf.to_float(tf.equal(l,L)) * r * loss)
        gcross_loss = gcross_utils.generalized_cross_entropy_bernoulli(weights,0.2)
        gcross_loss = gcross_loss * m * tf.to_float(tf.equal(l,L)) * (1-r)
        loss = loss + gcross_loss
        loss = tf.reduce_sum(loss,axis=-1)
        loss = loss * d
        loss = tf.reduce_mean(loss)
        return loss
    # joint_scores_from_f_and_w: Input - weights [batch_size, num_rules], m [batch_size, num_rules], f_probs [batch_size, num_classes], result - scalar 
	def joint_scores_from_f_and_w(self,weights,m,f_probs):
        num_classes = self.num_classes
        rule_classes = self.rule_classes
        #[batch_size, num_rules, 1]
        weights = tf.expand_dims(weights,-1)
        weights_mask = tf.to_float(tf.greater(weights,0.5))
        #[batch_size, num_rules, 1]
        m = tf.expand_dims(m,-1)
        m = m*weights_mask
        #[num_rules, num_classes]
        one_hot_rule_classes = tf.one_hot(rule_classes,num_classes,dtype=tf.float32)
        #[1, num_rules, num_classes]
        one_hot_rule_classes = tf.expand_dims(one_hot_rule_classes,0)
        #[batch_size, num_rules, num_classes]
        rule_weight_product = weights * one_hot_rule_classes + (1-weights)*(1-one_hot_rule_classes)
        sum_rule_firings = tf.reduce_sum(m,1)
        result = m*rule_weight_product #+ (1-m)
        #[batch_size, num_classes]
        result = tf.reduce_sum(result,1)/(sum_rule_firings+1e-20)
        result = result + f_probs
        return result    

	# get_weights_and_logits: Input - x [batch_size, num_rules], Output - weights [batch_size, num_rules], w_logits [batch_size, num_rules]
	def get_weights_and_logits(self, x):
        # Need to run the w network for each rule for the same x
        #
        # [batch_size, num_rules, num_features]
        x_shape = tf.shape(x)
        mul = tf.convert_to_tensor([1, self.num_rules_to_train])
        expanded_x = tf.tile(x, mul)
        # Need a python integer as the last dimension so that defining neural
        # networks work later. Hence use num_features instead of x_shape[1]
        x = tf.reshape(expanded_x , [x_shape[0], self.num_rules_to_train,
            self.num_features])

        batch_size = x_shape[0]
        rules_int = tf.convert_to_tensor([list(range(0,
            self.num_rules_to_train))])
        # Need to tile rules_int batch_size times
        #
        # tilevar should be a 1-D tensor with number of values equal to number
        # of columns in rules_int. Each column specifies the number of times
        # that axis in rules_int will be replicated.
        #
        # Following will replicate the rows of rules_int batch_size times and
        # leave the columns unchanged
        tilevar = tf.convert_to_tensor([batch_size, 1])
        rules_int = tf.tile(rules_int, tilevar) 
        rules_one_hot = tf.one_hot(rules_int, self.num_rules_to_train)
        rules_int = tf.expand_dims(rules_int, axis=-1)
        w_dict = {'x': x, 'rules' : rules_one_hot,
                'rules_int': rules_int}
        w_logits = self.w_network(w_dict, dropout_keep_prob=self.dropout_keep_prob)
        w_logits = tf.squeeze(w_logits)
        weights = tf.nn.sigmoid(w_logits)
        return weights, w_logits

	def optimize(self):
		# w_network
		# weights: [batch_size, num_rules]
    	# w_logits: [batch_size, num_rules]
		weights, w_logits = self.get_weights_and_logits(self.x)
		self.weights = weights

		# f_network
		# f_dict: [batch_size, num_classes]
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

num_classes = 6
if __name__ == '__main__':
	data = get_data(path, num_classes) # path will be the path of pickle file
	Il = Implyloss(data)
	Il.optimize()