import tensorflow as tf


class Train():
    def __init__(self, output=None,ground_truth=None,loss='softmax_crossentropy', optimizer='Adam', opt_params=[]):
        self.output = output
        self.ground_truth = ground_truth
        self.loss = loss
        self.optimizer = optimizer
        self.opt_params = opt_params

        self.learning_rate = self.opt_params[0]

        self.compute_cost()
        self.optimize()
        
    def compute_cost(self):
        if self.loss=='softmax_crossentropy':
            #Cross Entropy loss:
            with tf.name_scope('cross_entropy'):
                diff = tf.nn.softmax_cross_entropy_with_logits(self.output, self.ground_truth)
                with tf.name_scope('total'):
                    self.cost = tf.reduce_mean(diff)
            tf.scalar_summary('cross entropy', self.cost)

        if self.loss=='sigmoid_crossentropy':
            #Cross Entropy loss:
            with tf.name_scope('cross_entropy'):
                diff = tf.nn.sigmoid_cross_entropy_with_logits(self.output, self.ground_truth)
                with tf.name_scope('total'):
                    self.cost = tf.reduce_mean(diff)
            tf.scalar_summary('cross entropy', self.cost)
        
        if self.loss=='L2':
            with tf.name_scope('l2_loss'):
                self.cost = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(self.output, self.ground_truth))))
            tf.scalar_summary('l2_loss', self.cost)
    

    def optimize(self):
        
        with tf.name_scope('train'):
            if self.optimizer == 'adam':
                self.train = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

            if self.optimizer == 'rmsprop':
                self.train = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

            if self.optimizer == 'grad_descent':
                self.train = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)
            if self.optimizer == 'adagrad':
                self.train = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.cost)

            if self.optimizer == 'adadelta':
                self.train = tf.train.AdadeltaOptimizer(self.learning_rate).minimize(self.cost)


