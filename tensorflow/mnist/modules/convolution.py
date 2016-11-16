import tensorflow as tf
from module import Module
import variables


class Convolution(Module):
    '''
    Convolutional Layer
    '''

    def __init__(self, input_shape=(256,256,3), output_dim=64, kernel_size=(5,5), stride_size=(2,2), activation_bool=False, activation_fn=tf.nn.relu, pad = 'SAME',name="conv2d"):
        
        Module.__init__(self)

        self.input_shape = input_shape
        #self.check_input_shape()

        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.stride_size = stride_size

        self.weights_shape = [self.kernel_size[0], self.kernel_size[1], self.input_shape[-1], self.output_dim]
        self.strides = [1,self.stride_size[0], self.stride_size[1],1]
        self.pad = pad
        self.name = name
        
        with tf.variable_scope(self.name):
            self.weights = variables.weights(self.weights_shape)
            self.biases = variables.biases(self.output_dim)
        

    def check_input_shape(self):
        if len(self.input_shape)!=4:
            raise ValueError('Expected dimension of input tensor: 4')

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        
        inp_shape = self.input_tensor.get_shape().as_list()
        print inp_shape
        if len(inp_shape)!=4:
            self.input_tensor = tf.reshape(self.input_tensor,[-1, 28,28,1])
        
        
        with tf.name_scope('activations'):
            conv = tf.nn.conv2d(self.input_tensor, self.weights, strides = self.strides, padding=self.pad)
            self.activations = tf.reshape(tf.nn.bias_add(conv, self.biases), [-1]+conv.get_shape().as_list()[1:])
            #activations = activation_fn(conv, name='activation')
            tf.histogram_summary(self.name + '/activations', self.activations)
        return self.activations

    def lrp(self):
        return 0
        
