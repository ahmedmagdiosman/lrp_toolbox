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

        
    def _simple_lrp(self,R):
        '''
        LRP according to Eq(56) in DOI: 10.1371/journal.pone.0130140
        '''

        self.R = R
        R_shape = self.R.get_shape().as_list()
        if len(R_shape)!=4:
            activations_shape = self.activations.get_shape().as_list()
            self.R = tf.reshape(self.R, [-1]+activations_shape[1:])
            

        
        N,Hout,Wout,NF = self.R.get_shape().as_list()
        hf,wf,df,NF = self.weights_shape
        _, hstride, wstride, _ = self.strides

        out_N, out_rows, out_cols, out_depth = self.activations.get_shape().as_list()
        in_N, in_rows, in_cols, in_depth = self.input_tensor.get_shape().as_list()

        if self.pad == 'SAME':
            pr = (Hout -1) * hstride + hf - in_rows
            pc =  (Wout -1) * wstride + hf - in_cols
            pr = pr/2
            pc = pc - (pc/2)
            self.pad_input_tensor = tf.pad(self.input_tensor, [[0,0],[pr,pr], [pc,pc],[0,0]], "CONSTANT")

        pad_in_N, pad_in_rows, pad_in_cols, pad_in_depth = self.pad_input_tensor.get_shape().as_list()
        
        out = []
        term1 = tf.expand_dims(self.weights, 0)
        t2 = tf.expand_dims(tf.expand_dims(tf.expand_dims(self.biases, 0), 0), 0)
        for i in xrange(Hout):
            for j in xrange(Wout):
                input_slice = self.pad_input_tensor[:, i*hstride:i*hstride+hf , j*wstride:j*wstride+wf , : ]
                if input_slice.get_shape().as_list()[1] == input_slice.get_shape().as_list()[2]:
                    term2 =  tf.expand_dims(input_slice, -1)
                    Z = term1 * term2
                    t1 = tf.reduce_sum(Z, [1,2,3], keep_dims=True)
                    Zs = t1 + t2
                    stabilizer = 1e-8*(tf.select(tf.greater_equal(Zs,0), tf.ones_like(Zs)*-1, tf.ones_like(Zs)))
                    Zs += stabilizer
                    result = tf.reduce_sum((Z/Zs) * tf.expand_dims(self.R[:,i:i+1,j:j+1,:], 1), 4)

                    #pad each result to the dimension of the out
                    pad_right = pad_in_rows - (i*hstride+hf) if( pad_in_rows - (i*hstride+hf))>0 else 0
                    pad_left = i*hstride
                    pad_bottom = pad_in_cols - (j*wstride+wf) if ( pad_in_cols - (j*wstride+wf) > 0) else 0
                    pad_up = j*wstride

                    result = tf.pad(result, [[0,0],[pad_left, pad_right],[pad_up, pad_bottom],[0,0]], "CONSTANT")
                    out.append(result)
                   
        return tf.reduce_sum(tf.pack(out),0)[:, pr:in_rows+pr, pc:in_cols+pc,:]
