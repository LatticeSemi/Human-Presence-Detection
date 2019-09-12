
# Kishan Go.


import tensorflow as tf

import argparse
from config import *
from nets import *

def genpb(ckpt_dir):
    mc = kitti_squeezeDet_config()
    model = SqueezeDet(mc)

    saver = tf.train.Saver(model.model_params)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        ckpt = tf.train.latest_checkpoint(ckpt_dir)
        print("Using checkpoint: " + ckpt)
        saver.restore(sess, ckpt)
        tf.train.write_graph(sess.graph_def, ckpt_dir, 'graph.pb', as_text=False)
        tf.train.write_graph(sess.graph_def, ckpt_dir, 'graph.pbtxt', as_text=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True, default="", help="Checkpoint directory")
    args = parser.parse_args()
    genpb(args.ckpt_dir)
