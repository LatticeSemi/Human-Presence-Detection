# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""The data base wrapper class"""

import os
import random
import shutil
import sys

from PIL import Image, ImageFont, ImageDraw
import cv2
import numpy as np
from utils.util import iou, batch_iou

sys.stdout.flush()

class imdb(object):
  """Image database."""

  def __init__(self, name, mc):
    self._name = name
    self._classes = []
    self._image_set = []
    self._image_idx = []
    self._data_root_path = []
    self._rois = {}
    self.mc = mc

    # batch reader
    self._perm_idx = None
    self._cur_idx = 0

  @property
  def name(self):
    return self._name

  @property
  def classes(self):
    return self._classes

  @property
  def num_classes(self):
    return len(self._classes)

  @property
  def image_idx(self):
    return self._image_idx

  @property
  def image_set(self):
    return self._image_set

  @property
  def data_root_path(self):
    return self._data_root_path

  @property
  def year(self):
    return self._year

  def _shuffle_image_idx(self):
    self._perm_idx = [self._image_idx[i] for i in
        np.random.permutation(np.arange(len(self._image_idx)))]
    self._cur_idx = 0

  def read_image_batch(self, shuffle=True):
    """Only Read a batch of images
    Args:
      shuffle: whether or not to shuffle the dataset
    Returns:
      images: length batch_size list of arrays [height, width, 3]
    """
    mc = self.mc
    if shuffle:
      if self._cur_idx + mc.BATCH_SIZE >= len(self._image_idx):
        self._shuffle_image_idx()
      batch_idx = self._perm_idx[self._cur_idx:self._cur_idx+mc.BATCH_SIZE]
      self._cur_idx += mc.BATCH_SIZE
    else:
      if self._cur_idx + mc.BATCH_SIZE >= len(self._image_idx):
        batch_idx = self._image_idx[self._cur_idx:] \
            + self._image_idx[:self._cur_idx + mc.BATCH_SIZE-len(self._image_idx)]
        self._cur_idx += mc.BATCH_SIZE - len(self._image_idx)
      else:
        batch_idx = self._image_idx[self._cur_idx:self._cur_idx+mc.BATCH_SIZE]
        self._cur_idx += mc.BATCH_SIZE

    images, scales = [], []
    for i in batch_idx:
      im = cv2.imread(self._image_path_at(i))
      im = im.astype(np.float32, copy=False)
      im -= mc.BGR_MEANS
      orig_h, orig_w, _ = [float(v) for v in im.shape]
      im = cv2.resize(im, (mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT), interpolation=CV_INTER_AREA)

      #im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) # gray color instead of RGB <----------------
      #im = im - np.array([[[128]]]) <----------------
      #im = im.reshape((mc.IMAGE_HEIGHT, mc.IMAGE_WIDTH, 1)) <----------------

      x_scale = mc.IMAGE_WIDTH/orig_w
      y_scale = mc.IMAGE_HEIGHT/orig_h
      images.append(im)
      scales.append((x_scale, y_scale))

    return images, scales

  def read_batch(self, shuffle=True):
    """Read a batch of image and bounding box annotations.
    Args:
      shuffle: whether or not to shuffle the dataset
    Returns:
      image_per_batch: images. Shape: batch_size x width x height x [b, g, r]
      label_per_batch: labels. Shape: batch_size x object_num
      delta_per_batch: bounding box deltas. Shape: batch_size x object_num x 
          [dx ,dy, dw, dh]
      aidx_per_batch: index of anchors that are responsible for prediction.
          Shape: batch_size x object_num
      bbox_per_batch: scaled bounding boxes. Shape: batch_size x object_num x 
          [cx, cy, w, h]
    """
    mc = self.mc

    if shuffle:
      if self._cur_idx + mc.BATCH_SIZE >= len(self._image_idx):
        self._shuffle_image_idx()
      batch_idx = self._perm_idx[self._cur_idx:self._cur_idx+mc.BATCH_SIZE]
      self._cur_idx += mc.BATCH_SIZE
    else:
      if self._cur_idx + mc.BATCH_SIZE >= len(self._image_idx):
        batch_idx = self._image_idx[self._cur_idx:] \
            + self._image_idx[:self._cur_idx + mc.BATCH_SIZE-len(self._image_idx)]
        self._cur_idx += mc.BATCH_SIZE - len(self._image_idx)
      else:
        batch_idx = self._image_idx[self._cur_idx:self._cur_idx+mc.BATCH_SIZE]
        self._cur_idx += mc.BATCH_SIZE

    image_per_batch = []
    label_per_batch = []
    bbox_per_batch  = []
    delta_per_batch = []
    aidx_per_batch  = []
    if mc.DEBUG_MODE:
      avg_ious = 0.
      num_objects = 0.
      max_iou = 0.0
      min_iou = 1.0
      num_zero_iou_obj = 0

    for idx in batch_idx:
      # load the image
      #im = cv2.imread(self._image_path_at(idx), cv2.IMREAD_GRAYSCALE).astype(np.float32, copy=False)
      #print('file name:', self._image_path_at(idx))
      #im = cv2.imread(self._image_path_at(idx)).astype(np.float32, copy=False)
      im = cv2.imread(self._image_path_at(idx))
      if im is None:
        print('failed file read:'+self._image_path_at(idx))

      im = im.astype(np.float32, copy=False)

      # random brightness control
      hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
      h, s, v = cv2.split(hsv)
      add_v = np.random.randint(0, 255)
      v += (add_v - 128)
      final_hsv = cv2.merge((h, s, v))
      im = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
      if True: # add random noise to BGR
          b,g,r = cv2.split(im)
          add_b = np.random.random_integers(0, 32, b.shape) # characteristics of Himax module
          add_g = np.random.random_integers(0, 12, g.shape)
          add_r = np.random.random_integers(0, 12, r.shape)
	  #print("add_bgr={} {} {}".format(add_b, add_g, add_r))
	  #print("shape of b={}".format(b.shape))
	  #print("shape of add_b={}".format(add_b.shape))
          b += (add_b - 16)
          g += (add_g -  6)
          r += (add_r -  6)
          im = cv2.merge((b, g, r))

      im -= mc.BGR_MEANS # <-------------------------------
      #im -= np.array([[128]]) # <-------------------------------
      #print('im_shae',im.shape) # <-------------------------------
      orig_h, orig_w, _ = [float(v) for v in im.shape]
      #print('orig_h:', orig_h)
      #print('orig_w:', orig_w)

      # load annotations
      label_per_batch.append([b[4] for b in self._rois[idx][:]])
      gt_bbox = np.array([[(b[0]+b[2])/2, (b[1]+b[3])/2, b[2]-b[0], b[3]-b[1]] for b in self._rois[idx][:]])
      assert np.any(gt_bbox[:,0] > 0), 'less than 0 gt_bbox[0]'
      assert np.any(gt_bbox[:,1] > 0), 'less than 0 gt_bbox[1]'
      assert np.any(gt_bbox[:,2] > 0), 'less than 0 gt_bbox[2]'
      assert np.any(gt_bbox[:,3] > 0), 'less than 0 gt_bbox[3]'
      '''

      b[0] = b[0]*orig_w # cx
      b[1] = b[1]*orig_h # cy
      b[2] = b[2]*orig_w # w
      b[3] = b[3]*orig_h # h


      # b = center of x(xmin+width/2), center of y(ymin+height/2), width, height
      #print('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa')
      #print('image path:', self._image_path_at(idx))
      #print('bs=', b[0], b[1], b[2], b[3])
      #print('orig_w=', orig_w)
      #print('orig_h=', orig_h)
      #print('1st value', b[0]-b[2]/2+mc.IMAGE_WIDTH-orig_w)
      #print('2nd value', b[0]-b[2]/2)
      #print('3rd value', mc.IMAGE_WIDTH-b[2])
      assert max(0, b[0]-b[2]/2+mc.IMAGE_WIDTH*2-orig_w) < min(b[0]-b[2]/2+1, mc.IMAGE_WIDTH*2-b[2]+1), 'incorrect lo/hi' 
      disp_x = np.random.randint(max(0, b[0]-b[2]/2+mc.IMAGE_WIDTH*2-orig_w), 
                                 min(b[0]-b[2]/2+1, mc.IMAGE_WIDTH*2-b[2]+1)) 
      disp_y = np.random.randint(max(0, b[1]-b[3]/2+mc.IMAGE_HEIGHT*2-orig_h), 
                                 min(b[1]-b[3]/2+1, mc.IMAGE_HEIGHT*2-b[3]+1))
      crop_x = int(b[0]-b[2]/2 - disp_x) # width =b[2]
      crop_y = int(b[1]-b[3]/2 - disp_y) # height=b[3]
      #print('crops', crop_x, crop_y)
      im = im[crop_y:crop_y+mc.IMAGE_HEIGHT*2, crop_x:crop_x+mc.IMAGE_WIDTH*2]
      #b[0] = disp_x+b[2]/2 #crop_x
      #b[1] = disp_y+b[3]/2 #crop_y

      gt_bbox[:, 0] = disp_x+b[2]/2 #crop_x
      gt_bbox[:, 1] = disp_y+b[3]/2 #crop_y
      #print('new bs=', gt_bbox[0], b[2], b[3])
      assert crop_x >= 0 and crop_y >= 0, 'crop_x and crop_y incorrect'
      orig_h, orig_w, _ = [float(v) for v in im.shape]
      #print('size of cropped image', orig_h, orig_w)
      assert orig_h == mc.IMAGE_HEIGHT*2 and orig_w == mc.IMAGE_WIDTH*2, 'incorrect size of cropped image'
      '''


      #b[0] = b[0] 
      if mc.DATA_AUGMENTATION:
        assert mc.DRIFT_X >= 0 and mc.DRIFT_Y > 0, \
            'mc.DRIFT_X and mc.DRIFT_Y must be >= 0'

        if mc.DRIFT_X > 0 or mc.DRIFT_Y > 0:
          # Ensures that gt boundibg box is not cutted out of the image
	  #print('gt_bbox', gt_bbox)
	  #print('gt_bbox[:, 0]', gt_bbox[:, 0])
	  #print('gt_bbox[:, 2]', gt_bbox[:, 2])
          max_drift_x = min(gt_bbox[:, 0] - gt_bbox[:, 2]/2.0+1)
          max_drift_y = min(gt_bbox[:, 1] - gt_bbox[:, 3]/2.0+1)
	  #print('max_drift_x=', max_drift_x)
	  #print('max_drift_y=', max_drift_y)
          assert max_drift_x >= 0 and max_drift_y >= 0, 'bbox out of image'

          dy = np.random.randint(-mc.DRIFT_Y, min(mc.DRIFT_Y+1, max_drift_y))
          dx = np.random.randint(-mc.DRIFT_X, min(mc.DRIFT_X+1, max_drift_x))

          # shift bbox
          gt_bbox[:, 0] = gt_bbox[:, 0] - dx
          gt_bbox[:, 1] = gt_bbox[:, 1] - dy

          # distort image
          orig_h -= dy
          orig_w -= dx
          orig_x, dist_x = max(dx, 0), max(-dx, 0)
          orig_y, dist_y = max(dy, 0), max(-dy, 0)

          distorted_im = np.zeros(
              (int(orig_h), int(orig_w), 3)).astype(np.float32)
          distorted_im[dist_y:, dist_x:, :] = im[orig_y:, orig_x:, :]
	  #print(distorted_im.shape)
	  #print(im.shape)
          im = distorted_im

        # Flip image with 50% probability
        if np.random.randint(2) > 0.5:
          im = im[:, ::-1, :]
          gt_bbox[:, 0] = orig_w - 1 - gt_bbox[:, 0]

      # scale image
      #im = cv2.resize(im, (mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT))
      im = cv2.resize(im, (mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)
      #im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) # gray color instead of RGB # <-------------------------------
      #im = im - np.array([[[128]]]) # <----------------------------------------------????????????????
      #im = im.reshape((mc.IMAGE_HEIGHT, mc.IMAGE_WIDTH, 1))
      image_per_batch.append(im)

      # scale annotation
      x_scale = mc.IMAGE_WIDTH/orig_w
      y_scale = mc.IMAGE_HEIGHT/orig_h
      gt_bbox[:, 0::2] = gt_bbox[:, 0::2]*x_scale
      gt_bbox[:, 1::2] = gt_bbox[:, 1::2]*y_scale
      bbox_per_batch.append(gt_bbox)

      aidx_per_image, delta_per_image = [], []
      aidx_set = set()
      for i in range(len(gt_bbox)):
        overlaps = batch_iou(mc.ANCHOR_BOX, gt_bbox[i])

        aidx = len(mc.ANCHOR_BOX)
        for ov_idx in np.argsort(overlaps)[::-1]:
          if overlaps[ov_idx] <= 0:
            if mc.DEBUG_MODE:
              min_iou = min(overlaps[ov_idx], min_iou)
              num_objects += 1
              num_zero_iou_obj += 1
            break
          if ov_idx not in aidx_set:
            aidx_set.add(ov_idx)
            aidx = ov_idx
            if mc.DEBUG_MODE:
              max_iou = max(overlaps[ov_idx], max_iou)
              min_iou = min(overlaps[ov_idx], min_iou)
              avg_ious += overlaps[ov_idx]
              num_objects += 1
            break

        if aidx == len(mc.ANCHOR_BOX): 
          # even the largeset available overlap is 0, thus, choose one with the
          # smallest square distance
          dist = np.sum(np.square(gt_bbox[i] - mc.ANCHOR_BOX), axis=1)
          for dist_idx in np.argsort(dist):
            if dist_idx not in aidx_set:
              aidx_set.add(dist_idx)
              aidx = dist_idx
              break

        box_cx, box_cy, box_w, box_h = gt_bbox[i]
        #print(box_cx, box_cy, box_w, box_h)
        delta = [0]*4
        delta[0] = (box_cx - mc.ANCHOR_BOX[aidx][0])/mc.ANCHOR_BOX[aidx][2]
        delta[1] = (box_cy - mc.ANCHOR_BOX[aidx][1])/mc.ANCHOR_BOX[aidx][3]
        delta[2] = np.log(box_w/mc.ANCHOR_BOX[aidx][2])
        delta[3] = np.log(box_h/mc.ANCHOR_BOX[aidx][3])

        aidx_per_image.append(aidx)
        delta_per_image.append(delta)

      delta_per_batch.append(delta_per_image)
      aidx_per_batch.append(aidx_per_image)

    if mc.DEBUG_MODE:
      print ('max iou: {}'.format(max_iou))
      print ('min iou: {}'.format(min_iou))
      print ('avg iou: {}'.format(avg_ious/num_objects))
      print ('number of objects: {}'.format(num_objects))
      print ('number of objects with 0 iou: {}'.format(num_zero_iou_obj))

    return image_per_batch, label_per_batch, delta_per_batch, \
        aidx_per_batch, bbox_per_batch

  def evaluate_detections(self):
    raise NotImplementedError

  def visualize_detections(
      self, image_dir, image_format, det_error_file, output_image_dir,
      num_det_per_type=10):

    # load detections
    with open(det_error_file) as f:
      lines = f.readlines()
      random.shuffle(lines)
    f.close()

    dets_per_type = {}
    for line in lines:
      obj = line.strip().split(' ')
      error_type = obj[1]
      if error_type not in dets_per_type:
        dets_per_type[error_type] = [{
            'im_idx':obj[0], 
            'bbox':[float(obj[2]), float(obj[3]), float(obj[4]), float(obj[5])],
            'class':obj[6],
            'score': float(obj[7])
        }]
      else:
        dets_per_type[error_type].append({
            'im_idx':obj[0], 
            'bbox':[float(obj[2]), float(obj[3]), float(obj[4]), float(obj[5])],
            'class':obj[6],
            'score': float(obj[7])
        })

    out_ims = []
    # Randomly select some detections and plot them
    COLOR = (200, 200, 0)
    for error_type, dets in dets_per_type.iteritems():
      det_im_dir = os.path.join(output_image_dir, error_type)
      if os.path.exists(det_im_dir):
        shutil.rmtree(det_im_dir)
      os.makedirs(det_im_dir)

      for i in range(min(num_det_per_type, len(dets))):
        det = dets[i]
        im = Image.open(
            os.path.join(image_dir, det['im_idx']+image_format))
        draw = ImageDraw.Draw(im)
        draw.rectangle(det['bbox'], outline=COLOR)
        draw.text((det['bbox'][0], det['bbox'][1]), 
                  '{:s} ({:.2f})'.format(det['class'], det['score']),
                  fill=COLOR)
        out_im_path = os.path.join(det_im_dir, str(i)+image_format)
        im.save(out_im_path)
        im = np.array(im)
        out_ims.append(im[:,:,::-1]) # RGB to BGR
    return out_ims

