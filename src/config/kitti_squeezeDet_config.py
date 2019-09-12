# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""Model configuration for pascal dataset"""

import numpy as np

from .config import base_model_config

def kitti_squeezeDet_config():
  """Specify the parameters to tune below."""
  mc                       = base_model_config('KITTI')

  # Traffic signal
  # aiu - 1024x522
  # vid - 704x480, 640x480
  mc.IMAGE_WIDTH           = 64 #128 #224 #280 #160 #224 #704 # half width 621
  mc.IMAGE_HEIGHT          = 64 #128 #224 #210 #120 #224 #480 # half height 187
  #mc.IMAGE_WIDTH           = 1248
  #mc.IMAGE_HEIGHT          = 384

  mc.BATCH_SIZE            = 20

  mc.WEIGHT_DECAY          = 0.0001
  mc.LEARNING_RATE         = 0.01
  mc.DECAY_STEPS           = 10000
  mc.MAX_GRAD_NORM         = 1.0
  mc.MOMENTUM              = 0.9
  mc.LR_DECAY_FACTOR       = 0.5

  mc.LOSS_COEF_BBOX        = 5.0
  mc.LOSS_COEF_CONF_POS    = 75.0
  mc.LOSS_COEF_CONF_NEG    = 100.0
  mc.LOSS_COEF_CLASS       = 1.0

  mc.PLOT_PROB_THRESH      = 0.55 #0.6
  mc.NMS_THRESH            = 0.4
  mc.PROB_THRESH           = 0.005
  mc.TOP_N_DETECTION       = 2 #64

  mc.DATA_AUGMENTATION     = True
  mc.DRIFT_X               = 32 #150 # 60 #150; for original image size
  mc.DRIFT_Y               = 32 #100 # 42 #100
  mc.EXCLUDE_HARD_EXAMPLES = False

  mc.ANCHOR_BOX            = set_anchors(mc)
  mc.ANCHORS               = len(mc.ANCHOR_BOX)
  mc.ANCHOR_PER_GRID       = 6 #9

  return mc

def set_anchors(mc):
  H, W, B = 4, 4, 6 # 4 layer; 6 layer with 4 pool; 6 anchors
  div_scale = 8.0
  anchor_shapes = np.reshape(
      [np.array(
          [
           # Human presence detection
	   [int(366./div_scale), int(174./div_scale)], 
	   [int(115./div_scale), int( 59./div_scale)],
           [int(162./div_scale), int( 87./div_scale)], 
	   [int(258./div_scale), int(173./div_scale)],
           [int(224./div_scale), int(108./div_scale)], 
	   [int( 78./div_scale), int(170./div_scale)], 
          ])] * H * W,
      (H, W, B, 2)
  )

  center_x = np.reshape(
      np.transpose(
          np.reshape(
              np.array([np.arange(1, W+1)*float(mc.IMAGE_WIDTH)/(W+1)]*H*B), 
              (B, H, W)
          ),
          (1, 2, 0)
      ),
      (H, W, B, 1)
  )
  center_y = np.reshape(
      np.transpose(
          np.reshape(
              np.array([np.arange(1, H+1)*float(mc.IMAGE_HEIGHT)/(H+1)]*W*B),
              (B, W, H)
          ),
          (2, 1, 0)
      ),
      (H, W, B, 1)
  )
  anchors = np.reshape(
      np.concatenate((center_x, center_y, anchor_shapes), axis=3),
      (-1, 4)
  )

  return anchors
