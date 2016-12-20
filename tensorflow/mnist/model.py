
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append("..")
from modules.sequential import Sequential
from modules.linear import Linear
from modules.softmax import Softmax
from modules.relu import Relu
from modules.tanh import Tanh
from modules.convolution import Convolution
import modules.render as render
import input_data

import argparse
import tensorflow as tf
import numpy as np
import pdb
import os

flags = tf.flags
logging = tf.logging



flags.DEFINE_integer("batch_size", 100,'Number of steps to run trainer.')
flags.DEFINE_string("data_dir", 'data','Directory for storing data')
flags.DEFINE_string("summaries_dir", 'my_model_logs','Summaries directory')
flags.DEFINE_boolean("relevance_bool", True,'Compute relevances')
flags.DEFINE_string("checkpoint_dir", 'mnist_linear_model','Checkpoint dir')


FLAGS = flags.FLAGS

def visualize(relevances, images_tensor):
    n, dim = relevances.shape
    heatmap = relevances.reshape([n,28,28,1])
    input_images = images_tensor.reshape([n,28,28,1])
    heatmaps = []
    for h,heat in enumerate(heatmap):
        input_image = input_images[h]
        maps = render.hm_to_rgb(heat, input_image, scaling = 3, sigma = 2)
        heatmaps.append(maps)
    R = np.array(heatmaps)
    with tf.name_scope('input_reshape'):
        img = tf.image_summary('input', tf.cast(R, tf.float32), n)
    return img.eval()

def layers(x):
    # Define the layers of your network here 
    my_network = Sequential([Linear(784,500, input_shape=(FLAGS.batch_size,784)), 
                             Relu(),
                             Linear(500, 100), 
                             Relu(),
                             Linear(100, 10),
                             Softmax()])
    return my_network, my_network.forward(x)

def reload(saver, sess):
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    try: 
        if ckpt and ckpt.model_checkpoint_path:
            print('Reloading from -- '+FLAGS.checkpoint_dir+'/model.ckpt')
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            raise ValueError('No model found!')
    except:
        raise ValueError('Layer definition and model layers mismatch!')
            
def test():

    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    with tf.Session() as sess:
        x = tf.placeholder(tf.float32, [None, 784], name='input')
        with tf.variable_scope('model'):
            my_netowrk, output = layers(x)
            if FLAGS.relevance_bool:
                RELEVANCE = my_netowrk.lrp(output, 'simple', 1.0)
                
        # Merge all the summaries and write them out 
        merged = tf.merge_all_summaries()
        test_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/my_model')

        # Intialize variables and reload your model
        saver = tf.train.Saver()
        tf.initialize_all_variables().run()
        reload(saver, sess)

        # Extract testing data 
        xs, ys = mnist.test.next_batch(FLAGS.batch_size)
        # Pass the test data to the restored model
        summary, relevance_test= sess.run([merged, RELEVANCE], feed_dict={x:xs})
        test_writer.add_summary(summary, 0)

        # Save the images as heatmaps to visualize on tensorboard
        test_img_summary = visualize(relevance_test, xs)
        test_writer.add_summary(test_img_summary)
        test_writer.flush()
        test_writer.close()
    
def main(_):
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)
    test()


if __name__ == '__main__':
    tf.app.run()
