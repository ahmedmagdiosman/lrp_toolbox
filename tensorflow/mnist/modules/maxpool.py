import tensorflow as tf
from module import Module


class MaxPool(Module):

    def __init__(self, pool_size=(2,2), pool_stride=None, pad = 'SAME',name='maxpool'):
        Module.__init__(self)
        self.pool_size = pool_size
        self.pool_stride = pool_stride
        if self.pool_stride is None:
            self.pool_stride=self.pool_size
        self.pad = pad
        self.name = name

    def forward(self,input_tensor):
        self.input_tensor = input_tensor
        with tf.variable_scope(self.name):
            with tf.name_scope('activations'):
                self.activations = tf.nn.max_pool(self.input_tensor, ksize=self.pool_size,strides=self.pool_stride,padding=self.pad, name=self.name )
            tf.histogram_summary(self.name + '/activations', self.activations)
        return self.activations

    def clean(self):
        self.activations = None

    def lrp(self,R,*args,**kwargs):
        return R



class AvgPool(Module):

    def __init__(self, pool_size=(2,2), pool_stride=None, pad = 'SAME',name='maxpool'):
        Module.__init__(self)
        self.pool_size = pool_size
        self.pool_stride = pool_stride
        if self.pool_stride is None:
            self.pool_stride=self.pool_size
        self.pad = pad
        self.name = name

    def forward(self,input_tensor):
        self.input_tensor = input_tensor
        with tf.variable_scope(self.name):
            with tf.name_scope('activations'):
                self.activations = tf.nn.avg_pool(self.input_tensor, ksize=self.pool_size,strides=self.pool_stride,padding=self.pad, name=self.name )
            tf.histogram_summary(self.name + '/activations', self.activations)
        return self.activations

    def clean(self):
        self.activations = None

    def lrp(self,R,*args,**kwargs):
        return R
