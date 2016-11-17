import tensorflow as tf
from module import Module



class Softmax(Module):
    '''
    Softmax Layer
    '''

    def __init__(self, name):
        Module.__init__(self)
        self.name = name
    def forward(self,input_tensor):
        self.input_tensor = input_tensor
        with tf.variable_scope(self.name):
            with tf.name_scope('activations'):
                self.activations = tf.nn.softmax(self.input_tensor, name=self.name)
            tf.histogram_summary(self.name + '/activations', self.activations)
        return self.activations

    def clean(self):
        self.activations = None

    def lrp(self,R,*args,**kwargs):
        # component-wise operations within this layer
        # ->
        # just propagate R further down.
        # makes sure subroutines never get called.
        return tf.nn.softmax(self.activations) * self.activations
