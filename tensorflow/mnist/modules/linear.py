import tensorflow as tf
from module import Module
import variables


class Linear(Module):
    '''
    Linear Layer
    '''

    def __init__(self, input_dim, output_dim, activation_bool=False, activation_fn=tf.nn.relu,name="linear"):

        Module.__init__(self)

        self.input_dim = input_dim
        #self.input_shape = self.input_tensor.get_shape().as_list()
        self.output_dim = output_dim
        #self.check_input_shape()
        self.name = name
        
        self.weights_shape = [self.input_dim, self.output_dim]
        with tf.variable_scope(name):
            self.weights = variables.weights(self.weights_shape)
            self.biases = variables.biases(self.output_dim)

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        with tf.name_scope('activations'):
            linear = tf.matmul(self.input_tensor, self.weights)
            self.activations = tf.nn.bias_add(linear, self.biases)
            #activations = activation_fn(conv, name='activation')
            tf.histogram_summary(self.name + '/activations', self.activations)
        return self.activations

    def check_input_shape(self):
        if len(self.input_shape)!=2:
            raise ValueError('Expected dimension of input tensor: 2')


    def lrp(self, R):
        return self._simple_lrp(R)
        
    def _simple_lrp(self, R):
        self.R = R
        Z = tf.expand_dims(self.weights, 0) * tf.expand_dims(self.input_tensor, -1)
        Zs = tf.expand_dims(tf.reduce_sum(Z, 1), 1) + tf.expand_dims(tf.expand_dims(self.biases, 0), 0)
        return tf.reduce_sum((Z / Zs) * tf.expand_dims(self.R, 1),2)