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
#from modules.convolution import Convolution
from modules.conv import Convolution

from modules.avgpool import AvgPool
from modules.maxpool import MaxPool


import modules.render as render
import input_data

import argparse
import tensorflow as tf
import numpy as np
import pdb

flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("max_steps", 3001,'Number of steps to run trainer.')
flags.DEFINE_integer("batch_size", 100,'Number of steps to run trainer.')
flags.DEFINE_integer("test_batch_size", 100,'Number of steps to run trainer.')
flags.DEFINE_integer("test_every", 500,'Number of steps to run trainer.')
flags.DEFINE_float("learning_rate", 0.01,'Initial learning rate')
flags.DEFINE_float("dropout", 0.9, 'Keep probability for training dropout.')
flags.DEFINE_string("data_dir", 'data','Directory for storing data')
flags.DEFINE_string("summaries_dir", 'mnist_convolution_logs','Summaries directory')
flags.DEFINE_boolean("relevance_bool", False,'Compute relevances')
flags.DEFINE_string("relevance_method", 'simple','relevance methods: simple/eps/w^2/alphabeta')
flags.DEFINE_boolean("save_model", False,'Save the trained model')
flags.DEFINE_boolean("reload_model", False,'Restore the trained model')
flags.DEFINE_string("checkpoint_dir", 'mnist_convolution_model','Checkpoint dir')

FLAGS = flags.FLAGS


def seq_conv_nn(x):
    #pdb.set_trace()
    nn = Sequential([Convolution(input_dim=1,output_dim=32,input_shape=(FLAGS.batch_size, 28)), 
                     MaxPool(),
                     Tanh(),
                     Convolution(input_dim=32,output_dim=64),
                     MaxPool(),
                     Tanh(),  
                     Linear(256, 10), 
                     Softmax()])
    return nn, nn.forward(x)





def visualize_conv(relevances, images_tensor):
    n, w,h,c = relevances.shape
    heatmap = relevances
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


def train():
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  with tf.Session() as sess:
    # Input placeholders
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
        keep_prob = tf.placeholder(tf.float32)
    
    with tf.variable_scope('model'):
        #nn, y = seq_nn(x)
        nn, y = seq_conv_nn(x)
        if FLAGS.relevance_bool:
            RELEVANCE = nn.lrp(y, FLAGS.relevance_method, 1.0)
        
    with tf.name_scope('cross_entropy'):
        diff = tf.nn.softmax_cross_entropy_with_logits(y, y_)
        with tf.name_scope('total'):
            cross_entropy = tf.reduce_mean(diff)
        tf.scalar_summary('cross entropy', cross_entropy)

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.scalar_summary('accuracy', accuracy)

    # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
    merged = tf.merge_all_summaries()
    train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/train', sess.graph)
    test_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/test')

    saver = tf.train.Saver()
    tf.initialize_all_variables().run()
    if FLAGS.reload_model:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print('Reloading from -- '+FLAGS.checkpoint_dir+'/model.ckpt')
            saver.restore(sess, ckpt.model_checkpoint_path)

    def feed_dict(train):
        """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
        if train:
            xs, ys = mnist.train.next_batch(FLAGS.batch_size)
            k = FLAGS.dropout
        else:
            xs, ys = mnist.test.next_batch(FLAGS.test_batch_size)
            k = 1.0
        return {x: xs, y_: ys, keep_prob: k}
    #import pdb; pdb.set_trace()
    for i in range(FLAGS.max_steps):
        if i % FLAGS.test_every == 0:  # test-set accuracy
            test_inp = feed_dict(False)
            if FLAGS.relevance_bool:
                summary, acc , relevance_test= sess.run([merged, accuracy, RELEVANCE], feed_dict=test_inp)
            else:
                summary, acc = sess.run([merged, accuracy], feed_dict=test_inp)
            test_writer.add_summary(summary, i)
            print('Accuracy at step %s: %f' % (i, acc))
        else:  
            inp = feed_dict(True)
            if FLAGS.relevance_bool:
                summary, _ , relevance_train= sess.run([merged, train_step, RELEVANCE], feed_dict=inp)
            else:
                summary, _ = sess.run([merged, train_step], feed_dict=inp)
            train_writer.add_summary(summary, i)

    if FLAGS.save_model:
        if not os.path.exists(FLAGS.checkpoint_dir):
            os.system('mkdir '+FLAGS.checkpoint_dir)
        save_path = saver.save(sess, FLAGS.checkpoint_dir+'/model.ckpt',write_meta_graph=False)
        
    if FLAGS.relevance_bool:
        test_img_summary = visualize_conv(relevance_test, test_inp[test_inp.keys()[0]])
        test_writer.add_summary(test_img_summary)
        test_writer.flush()

        train_img_summary = visualize_conv(relevance_train, inp[inp.keys()[0]])
        train_writer.add_summary(train_img_summary)
        train_writer.flush()

    train_writer.close()
    test_writer.close()
    print('Accuracy at step %s: %f' % (i, acc))

def main(_):
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
