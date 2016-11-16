import tensorflow as tf




def weights(weights_shape, initializer=tf.truncated_normal_initializer(stddev=0.01), name='weights'):
    weights_shape = weights_shape
    initial = initializer
    name = name
        
    return tf.get_variable(name, shape=weights_shape, initializer=initializer)


def biases( bias_shape, initializer = 0, name = 'biases'):
    bias_shape = bias_shape
    initializer = tf.constant_initializer(initializer)
    name = name
        
    return tf.get_variable(name, bias_shape, initializer=initializer)

