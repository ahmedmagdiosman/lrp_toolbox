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
from modules.tconvolution import Tconvolution
import modules.render as render
import input_data

import argparse
import tensorflow as tf
import numpy as np
import pdb

flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("max_steps", 10000,'Number of steps to run trainer.')
flags.DEFINE_integer("batch_size", 100,'Number of steps to run trainer.')
flags.DEFINE_integer("test_batch_size", 100,'Number of steps to run trainer.')
flags.DEFINE_integer("test_every", 100,'Number of steps to run trainer.')
flags.DEFINE_integer("hidden_size", 10,'Number of steps to run trainer.')


flags.DEFINE_float("learning_rate", 0.001,'Initial learning rate')
flags.DEFINE_float("dropout", 0.9, 'Keep probability for training dropout.')
flags.DEFINE_string("data_dir", 'data','Directory for storing data')
flags.DEFINE_string("summaries_dir", 'mnist_ae_logs','Summaries directory')
flags.DEFINE_boolean("relevance_bool", False,'Compute relevances')
flags.DEFINE_string("relevance_method", 'simple','relevance methods: simple/eps/w^2/alphabeta')


FLAGS = flags.FLAGS


def autoencoder(x):

    encoder = [Linear(784,500, input_shape=(FLAGS.batch_size,28)), 
                     Relu(),
                     Linear(500, 100), 
                     Relu(),
                     Linear(100, 10), 
                     ]
    decoder = [Linear(10,100), 
                     Relu(),
                     Linear(100, 500), 
                     Relu(),
                     Linear(500, 784),
                     Relu()
                     ]
    
    nn = Sequential(encoder+decoder)
    
    return nn, nn.forward(x)

def seq_conv_nn(x):

    encoder = [Convolution(input_dim=1,output_dim=32, input_shape=(FLAGS.batch_size,28)), 
                     Tanh(), 
                     Convolution(input_dim=32,output_dim=64),
                     Tanh(),  
                     Convolution(input_dim=64,output_dim=16),
                     Tanh(), 
                     Linear(256, 10)]
    # decoder = [Linear(10,100), 
    #                  Relu(),
    #                  Linear(100, 500), 
    #                  Relu(),
    #                  Linear(500, 784),
    #                  Relu()]
    decoder = [Tconvolution(input_dim=10,output_dim=128, kernel_size=(3,3), stride_size=(1,1)), 
                     Relu(),
               Tconvolution(input_dim=128,output_dim=64, kernel_size=(5,5)), 
                     Relu(),
               Tconvolution(input_dim=64,output_dim=32, kernel_size=(5,5), pad='VALID'), 
                     Relu(),
               Tconvolution(input_dim=32,output_dim=16, kernel_size=(5,5)), 
                     Relu(),
               Tconvolution(input_dim=16,output_dim=1, kernel_size=(5,5)), 
                     Relu()]
    
    nn = Sequential(encoder+decoder)
    return nn, nn.forward(x)


def noise(inp):
    noise = tf.get_variable(tf.truncated_normal(shape=[-1]+inp.get_shape().as_list()[1:], stddev = 0.1))
    return tf.addd(inp,noise)


def visualize(relevances, images_tensor=None):
    n,w,h, dim = relevances.shape
    heatmap = relevances
    #heatmap = relevances.reshape([n,28,28,1])
    heatmaps = []

    if images_tensor is not None:
        input_images = images_tensor.reshape([n,28,28,1])
            
    for h,heat in enumerate(heatmap):
        
        if images_tensor is not None:
            input_image = input_images[h]
    
            maps = render.hm_to_rgb(heat, input_image, scaling = 3, sigma = 2)
        else:
            maps = render.hm_to_rgb(heat, scaling = 3, sigma = 2)
        heatmaps.append(maps)
    R = np.array(heatmaps)
    with tf.name_scope('input_reshape'):
        img = tf.image_summary('input', tf.cast(R, tf.float32), n)
    return img.eval()

def train():
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  with tf.Session() as sess:
    # Input placeholders
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
        keep_prob = tf.placeholder(tf.float32)

    
    #noisy_x = noise(x)
    with tf.variable_scope('model'):
        #nn, y = autoencoder(x)
        nn, y = seq_conv_nn(x)
        output_shape = y.get_shape().as_list()
        #pdb.set_trace()
        y = tf.reshape(y, [FLAGS.batch_size, output_shape[1]*output_shape[2]*output_shape[3]])
        if FLAGS.relevance_bool:
            RELEVANCE = nn.lrp(y, FLAGS.relevance_method, 1.0)
        
    with tf.name_scope('l2_loss'):
        epsilon = 1e-8
        l2_loss = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(x,y))))
    tf.scalar_summary('l2_loss', l2_loss)
        #l2_loss = tf.reduce_sum(-x * tf.log(y + epsilon) -
        #                (1.0 - x) * tf.log(1.0 - y + epsilon))
        #l2_loss = tf.nn.l2_loss(y, x)
        # with tf.name_scope('total'):
        #     cross_entropy = tf.reduce_mean(diff)
        # tf.scalar_summary('cross entropy', cross_entropy)

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(l2_loss)

    # with tf.name_scope('accuracy'):
    #     with tf.name_scope('correct_prediction'):
    #         correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    #     with tf.name_scope('accuracy'):
    #         accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #     tf.scalar_summary('accuracy', accuracy)

    # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
    merged = tf.merge_all_summaries()
    train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/train', sess.graph)
    test_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/test')
    tf.initialize_all_variables().run()


    def feed_dict(train):
        """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
        if train:
            xs, ys = mnist.train.next_batch(FLAGS.batch_size)
            k = FLAGS.dropout
        else:
            xs, ys = mnist.test.next_batch(FLAGS.test_batch_size)
            k = 1.0
        return {x: xs, y_: ys, keep_prob: k}

    for i in range(FLAGS.max_steps):
        if i % FLAGS.test_every == 0:  # test-set accuracy
            test_inp = feed_dict(False)
            if FLAGS.relevance_bool:
                summary, loss , relevance_test, test_op_imgs= sess.run([merged, l2_loss, RELEVANCE,y], feed_dict=test_inp)
            else:
                summary, loss, test_op_imgs = sess.run([merged, l2_loss,y], feed_dict=test_inp)
            test_writer.add_summary(summary, i)
            print('Loss at step %s: %f' % (i, loss))
        else:  
            inp = feed_dict(True)
            if FLAGS.relevance_bool:
                summary, _ , relevance_train, op_imgs= sess.run([merged, train_step, RELEVANCE, y], feed_dict=inp)
            else:
                summary, _ , op_imgs = sess.run([merged, train_step, y], feed_dict=inp)
            train_writer.add_summary(summary, i)
    #pdb.set_trace()
    if FLAGS.relevance_bool:
        test_img_summary = visualize(relevance_test, test_inp[test_inp.keys()[0]])
        test_writer.add_summary(test_img_summary)
        test_writer.flush()

        #train_img_summary = visualize(relevance_train, inp[inp.keys()[0]])
        #train_img_summary = visualize(test_op_imgs)
        with tf.name_scope('input_reshape'):
            img = tf.image_summary('input', tf.cast(test_op_imgs.reshape([FLAGS.test_batch_size, 28,28,1]), tf.float32), test_op_imgs.shape[0])
        
        train_writer.add_summary(img.eval())
        train_writer.flush()

    train_writer.close()
    test_writer.close()


def main(_):
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
