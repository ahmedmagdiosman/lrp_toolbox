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
import modules.render as render
import input_data

import argparse
import tensorflow as tf
import numpy as np
import pdb
import os

flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("max_steps", 3500,'Number of steps to run trainer.')
flags.DEFINE_integer("batch_size", 100,'Number of steps to run trainer.')
flags.DEFINE_integer("test_batch_size", 100,'Number of steps to run trainer.')
flags.DEFINE_integer("test_every", 100,'Number of steps to run trainer.')
flags.DEFINE_float("learning_rate", 0.001,'Initial learning rate')
flags.DEFINE_float("dropout", 0.9, 'Keep probability for training dropout.')
flags.DEFINE_string("data_dir", 'data','Directory for storing data')
flags.DEFINE_string("summaries_dir", 'mnist_linear_logs','Summaries directory')
flags.DEFINE_boolean("relevance_bool", False,'Compute relevances')
flags.DEFINE_boolean("save_model", False,'Save the trained model')
flags.DEFINE_boolean("reload_model", False,'Restore the trained model')
flags.DEFINE_string("checkpoint_dir", 'mnist_linear_model','Checkpoint dir')


FLAGS = flags.FLAGS


def nn():
    return Sequential([Linear(784,500, input_shape=(FLAGS.batch_size,784)), 
                     Relu(),
                     Linear(500, 100), 
                     Relu(),
                     Linear(100, 10), 
                     Softmax()])


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


def init_vars(sess):
    saver = tf.train.Saver()
    tf.initialize_all_variables().run()
    if FLAGS.reload_model:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print('Reloading from -- '+FLAGS.checkpoint_dir+'/model.ckpt')
            saver.restore(sess, ckpt.model_checkpoint_path)
    return saver

def save_model(sess, saver):
    if FLAGS.save_model:
        if not os.path.exists(FLAGS.checkpoint_dir):
            os.system('mkdir '+FLAGS.checkpoint_dir)
        save_path = saver.save(sess, FLAGS.checkpoint_dir+'/model.ckpt',write_meta_graph=False)

def plot_relevances(rel, img, writer):
    img_summary = visualize(rel, img)
    writer.add_summary(img_summary)
    writer.flush()

def train():
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  with tf.Session() as sess:
    # Input placeholders
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
        keep_prob = tf.placeholder(tf.float32)

    # Model definition along with training and relevances
    with tf.variable_scope('model'):
        net = nn()
        y = net.forward(x)
        train = net.fit(output=y,ground_truth=y_,loss='softmax_crossentropy',optimizer='adam', opt_params=[FLAGS.learning_rate])
        
        if FLAGS.relevance_bool:
            RELEVANCE = net.lrp(y, 'simple', 1.0)
        else:
            RELEVANCE = None
        
    # Accuracy computation
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.scalar_summary('accuracy', accuracy)

    # Merge all the summaries and write them out 
    merged = tf.merge_all_summaries()
    train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/train', sess.graph)
    test_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/test')

    saver = init_vars(sess)

    # input dict creation as per tensorflow source code
    def feed_dict(train):
        """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
        if train:
            xs, ys = mnist.train.next_batch(FLAGS.batch_size)
            k = FLAGS.dropout
        else:
            xs, ys = mnist.test.next_batch(FLAGS.test_batch_size)
            k = 1.0
        return {x: xs, y_: ys, keep_prob: k}

    # iterate over train and test data
    for i in range(FLAGS.max_steps):
        if i % FLAGS.test_every == 0:  
            test_inp = feed_dict(False)
            summary, acc , relevance_test= sess.run([merged, accuracy, RELEVANCE], feed_dict=test_inp)
            test_writer.add_summary(summary, i)
            print('Accuracy at step %s: %f' % (i, acc))
        else:  
            inp = feed_dict(True)
            summary, _ , relevance_train= sess.run([merged, train, RELEVANCE], feed_dict=inp)
            train_writer.add_summary(summary, i)

    # save model if required
    save_model(sess, saver)

    # relevances plotted with visually pleasing color schemes
    if FLAGS.relevance_bool:
        # plot test images with relevances overlaid
        plot_relevances(relevance_test, test_inp[test_inp.keys()[0]], test_writer )
        # plot train images with relevances overlaid
        plot_relevances(relevance_train, inp[inp.keys()[0]], train_writer )

    train_writer.close()
    test_writer.close()


def main(_):
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
