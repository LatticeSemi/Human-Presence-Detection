# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""SqueezeDet model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import joblib
from utils import util
from easydict import EasyDict as edict
import numpy as np
import tensorflow as tf
from nn_skeleton import ModelSkeleton

class SqueezeDet(ModelSkeleton):
  def __init__(self, mc, gpu_id=0):
    with tf.device('/gpu:{}'.format(gpu_id)):
      ModelSkeleton.__init__(self, mc)

      self._add_forward_graph()
      self._add_interpretation_graph()
      self._add_loss_graph()
      self._add_train_graph()
      self._add_viz_graph()

  def _add_forward_graph(self):
    """NN architecture."""

    mc = self.mc
    bin_k = 1 # K for BNN

    if mc.LOAD_PRETRAINED_MODEL:
      assert tf.gfile.Exists(mc.PRETRAINED_MODEL_PATH), \
          'Cannot find pretrained model at the given path:' \
          '  {}'.format(mc.PRETRAINED_MODEL_PATH)
      self.caffemodel_weight = joblib.load(mc.PRETRAINED_MODEL_PATH)

    depth = [16, 16, 16, 20, 20, 32]  #  T+ #3
    ####################################################################
    # Binarized layers
    ####################################################################
    ml_w_bin = False
    ml_a_bin = False

    fire1 = self._fire_layer('fire1', self.image_input, oc=depth[0], freeze=False, w_bin=False,    a_bin=False)
    fire2 = self._fire_layer('fire2', fire1,    oc=depth[1], freeze=False, w_bin=ml_w_bin, a_bin=ml_a_bin, pool_en=False)
    fire3 = self._fire_layer('fire3', fire2,    oc=depth[2], freeze=False, w_bin=ml_w_bin, a_bin=ml_a_bin)
    fire4 = self._fire_layer('fire4', fire3,    oc=depth[3], freeze=False, w_bin=ml_w_bin, a_bin=ml_a_bin, pool_en=False)
    fire5 = self._fire_layer('fire5', fire4,    oc=depth[4], freeze=False, w_bin=ml_w_bin, a_bin=ml_a_bin)
    fire6 = self._fire_layer('fire6', fire5,    oc=depth[5], freeze=False, w_bin=False,    a_bin=False)
    ####################################################################

    num_output = mc.ANCHOR_PER_GRID * (mc.CLASSES + 1 + 4) # 6 * (1 + 1 + 4) * 4 * 4
    self.preds = self._conv_layer('conv12', fire6, filters=num_output, size=3, stride=1,
        padding='SAME', xavier=False, relu=False, stddev=0.0001, w_bin=False)
    print('self.preds:', self.preds)

  def _fire_layer(self, layer_name, inputs, oc, stddev=0.01, freeze=False, w_bin=False, a_bin=False, pool_en=True):
    with tf.variable_scope(layer_name):
      ex3x3 = self._conv_layer('conv3x3', inputs, filters=oc, size=3, stride=1,
        padding='SAME', stddev=stddev, freeze=freeze, relu=False, w_bin=w_bin)

      ex3x3 = self._batch_norm('bn', ex3x3)
      ex3x3 = self.binary_wrapper(ex3x3, a_bin=a_bin)
      tf.summary.histogram('after_relu', ex3x3)
      if pool_en:
        pool = self._pooling_layer('pool', ex3x3, size=2, stride=2, padding='SAME')
      else:
        pool = ex3x3
      tf.summary.histogram('pool', pool)

    return pool
