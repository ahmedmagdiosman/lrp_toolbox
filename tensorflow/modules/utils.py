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
import modules.render as render
import numpy as np
import tensorflow as tf

def visualize(relevances, images_tensor=None):
    n,w,h, dim = relevances.shape
    heatmaps = []

    if images_tensor is not None:
        assert relevances.shape==images_tensor.shape, 'Relevances shape != Images shape'
    for h,heat in enumerate(relevances):
        if images_tensor is not None:
            input_image = images_tensor[h]
            maps = render.hm_to_rgb(heat, input_image, scaling = 3, sigma = 2)
        else:
            maps = render.hm_to_rgb(heat, scaling = 3, sigma = 2)
        heatmaps.append(maps)
    R = np.array(heatmaps)
    with tf.name_scope('input'):
        img = tf.summary.image('input', tf.cast(R, tf.float32), n)
    return img.eval()

def plot_relevances(rel, img, writer):
    img_summary = visualize(rel, img)
    writer.add_summary(img_summary)
    writer.flush()

class Summaries():
    def __init__(self, summaries_dir, sub_dir=None, graph=None, name="summaries"):
        self.summaries_dir = summaries_dir
        self.sub_dir = sub_dir
        self.writer = self.create_writer(graph)

    def create_writer(self, graph=None):
        return tf.train.SummaryWriter(self.summaries_dir + '/' + self.sub_dir, graph)
        






class Utils():
    def __init__(self, session, checkpoint_dir=None, name="utils"):
        self.name = name
        self.session = session
        self.checkpoint_dir = checkpoint_dir
        

    def init_vars(self):
        self.saver = tf.train.Saver()
        #tf.initialize_all_variables().run()
        tf.global_variables_initializer().run()
        if self.checkpoint_dir is not None:
            ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                print('Reloading from -- '+self.checkpoint_dir+'/model.ckpt')
                self.saver.restore(self.session, ckpt.model_checkpoint_path)

    def save_model(self, step=0):
        import os
        if not os.path.exists(self.checkpoint_dir):
            os.system('mkdir '+self.checkpoint_dir)
        save_path = self.saver.save(self.session, self.checkpoint_dir+'/model.ckpt',write_meta_graph=False)
