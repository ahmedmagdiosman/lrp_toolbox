'''
@author: Vignesh Srinivasan
@author: Sebastian Lapuschkin
@author: Gregoire Montavon
@maintainer: Vignesh Srinivasan
@maintainer: Sebastian Lapuschkin
@contact: vignesh.srinivasan@hhi.fraunhofer.de
@date: 20.12.2016
@version: 1.0+
@copyright: Copyright (c)  2015, Sebastian Lapuschkin, Alexander Binder, Gregoire Montavon, Klaus-Robert Mueller, Wojciech Samek
@license : BSD-2-Clause
'''

import tensorflow as tf
from module import Module
import variables


class Linear(Module):
    '''
    Linear Layer
    '''

    def __init__(self, input_dim, output_dim, input_shape=(10,784), activation_bool=False, activation_fn=tf.nn.relu,name="linear"):
        self.name = name
        Module.__init__(self)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_shape = input_shape
        
        
        self.weights_shape = [self.input_dim, self.output_dim]
        with tf.variable_scope(self.name):
            self.weights = variables.weights(self.weights_shape)
            self.biases = variables.biases(self.output_dim)

    def forward(self, input_tensor, batch_size=10, img_dim=28):
        self.input_tensor = input_tensor
        inp_shape = self.input_tensor.get_shape().as_list()
                
        #import pdb;pdb.set_trace()
        if len(inp_shape)!=2:
            import numpy as np
            self.input_tensor = tf.reshape(self.input_tensor,[batch_size, np.prod(inp_shape[1:])])

        with tf.name_scope('activations'):
            linear = tf.matmul(self.input_tensor, self.weights)
            self.activations = tf.nn.bias_add(linear, self.biases)
            #activations = activation_fn(conv, name='activation')
            tf.histogram_summary(self.name + '/activations', self.activations)
        return self.activations

    def check_input_shape(self):
        if len(self.input_shape)!=2:
            raise ValueError('Expected dimension of input tensor: 2')


    # def lrp(self, R):
    #     return self._simple_lrp(R)
        
    def _simple_lrp(self, R):
        self.R = R
        R_shape = self.R.get_shape().as_list()
        if len(R_shape)!=2:
            activations_shape = self.activations.get_shape().as_list()
            self.R = tf.reshape(self.R, activations_shape)

        Z = tf.expand_dims(self.weights, 0) * tf.expand_dims(self.input_tensor, -1)
        Zs = tf.expand_dims(tf.reduce_sum(Z, 1), 1) + tf.expand_dims(tf.expand_dims(self.biases, 0), 0)
        return tf.reduce_sum((Z / Zs) * tf.expand_dims(self.R, 1),2)

    
    def _flat_lrp(self,R):
        '''
        distribute relevance for each output evenly to the output neurons' receptive fields.
        note that for fully connected layers, this results in a uniform lower layer relevance map.
        '''
        self.R= R
        Z = tf.ones_like(tf.expand_dims(self.weights, 0))
        Zs = tf.expand_dims( tf.reduce_sum(Z, 1), 1)
        return tf.reduce_sum((Z / Zs) * tf.expand_dims(self.R, 1),2)
                         
    def _ww_lrp(self,R):
        '''
        LRR according to Eq(12) in https://arxiv.org/pdf/1512.02479v1.pdf
        '''
        self.R= R
        Z = tf.square( tf.expand_dims(self.weights,0) )
        Zs = tf.expand_dims( tf.reduce_sum(Z, 1), 1)
        return tf.reduce_sum((Z / Zs) * tf.expand_dims(self.R, 1),2)
        
    def _epsilon_lrp(self,R,epsilon):
        '''
        LRP according to Eq(58) in DOI: 10.1371/journal.pone.0130140
        '''
        self.R= R
        Z = tf.expand_dims(self.weights, 0) * tf.expand_dims(self.input_tensor, -1)
        Zs = tf.expand_dims(tf.reduce_sum(Z, 1), 1) + tf.expand_dims(tf.expand_dims(self.biases, 0), 0)
        Zs = Zs + tf.select(tf.greater_equal(Zs,0), tf.ones_like(Zs)*-1, tf.ones_like(Zs))
        return tf.reduce_sum((Z / Zs) * tf.expand_dims(self.R, 1),2)
        

        # add slack to denominator. we require sign(0) = 1. since np.sign(0) = 0 would defeat the purpose of the numeric stabilizer we do not use it.
        Zs += epsilon * ((Zs >= 0)*2-1)
        

    def _alphabeta_lrp(self,R,alpha):
        '''
        LRP according to Eq(60) in DOI: 10.1371/journal.pone.0130140
        '''
        self.R= R
        beta = 1 - alpha
        Z = tf.expand_dims(self.weights, 0) * tf.expand_dims(self.input_tensor, -1)

        if not alpha == 0:
            Zp = tf.select(tf.greater(Z,0),Z, tf.zeros_like(Z))
            term2 = tf.expand_dims(tf.expand_dims(tf.select(tf.greater(self.biases,0),self.biases, tf.zeros_like(self.biases)), 0 ), 0)
            term1 = tf.expand_dims( tf.reduce_sum(Zp, 1), 1)
            Zsp = term1 + term2
            Ralpha = alpha + tf.reduce_sum((Z / Zsp) * tf.expand_dims(self.R, 1),2)
        else:
            Ralpha = 0

        if not beta == 0:
            Zn = tf.select(tf.lesser(Z,0),Z, tf.zeros_like(Z))
            term2 = tf.expand_dims(tf.expand_dims(tf.select(tf.lesser(self.biases,0),self.biases, tf.zeros_like(self.biases)), 0 ), 0)
            term1 = tf.expand_dims( tf.reduce_sum(Zn, 1), 1)
            Zsp = term1 + term2
            Rbeta = beta + tf.reduce_sum((Z / Zsp) * tf.expand_dims(self.R, 1),2)
        else:
            Rbeta = 0

        return Ralpha + Rbeta
