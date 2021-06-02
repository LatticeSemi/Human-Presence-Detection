"""
Convert model.ckpt to model.pb
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from google.protobuf import text_format

#import tensorflow.compat.v1 as tf
import tensorflow as tf
from tensorflow.python.framework import graph_util
tf.compat.v1.disable_v2_behavior()

def convert(check_point_dir, check_point_file, eval_pbtxt_file, output_node_names, output_dir, output_file):

    # create a session
    sess = tf.compat.v1.Session()

    # import model
    saver = tf.compat.v1.train.import_meta_graph(check_point_dir + '/' + check_point_file + '.meta', clear_devices=True)
    #tf.compat.v1.get_default_session().run(tf.global_variables_initializer())
    #tf.compat.v1.get_default_session().run(tf.local_variables_initializer())
    saver.restore(sess, check_point_dir + '/' + check_point_file)


    if True: # This assumes .pbtxt made in non-training mode (eval/demo mode)
        with open(check_point_dir + '/' + eval_pbtxt_file) as f:
            text_graph = f.read()
        graph_def = text_format.Parse(text_graph, tf.compat.v1.GraphDef())
        gd = graph_def
    else: # This is not correct. The .meta file used here came from training and BN has
          # all training related functions. Below code tries to remove it manually and 
          # it breakes the BN function. The correct one is to generate .pbtxt in non-training 
          # mode so that all measured values become constant. Then below problem itself
          # does not happen. Use above method
        gd = sess.graph.as_graph_def()
                    
        # for fixing the bug of batch norm
        for node in gd.node:            
            if node.op == 'RefSwitch':
                node.op = 'Switch'
                for index in xrange(len(node.input)):
                    if 'moving_' in node.input[index]:
                        node.input[index] = node.input[index] + '/read'
            elif node.op == 'AssignSub':
                node.op = 'Sub'
                if 'use_locking' in node.attr: del node.attr['use_locking']
            elif node.op == 'AssignAdd':
                node.op = 'Add'
                if 'use_locking' in node.attr: del node.attr['use_locking']

    converted_graph_def = graph_util.convert_variables_to_constants(sess, gd, output_node_names.split(","))
    tf.compat.v1.train.write_graph(converted_graph_def, output_dir, output_file, as_text=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CheckPoint to .pb converter')
    parser.add_argument('--check_point_dir', type=str, default="./trained_model", help='Checkpoint directory')
    parser.add_argument('--check_point_file', type=str, default="./model.ckpt-249999", help='Checkpoint file')
    parser.add_argument('--eval_pbtxt_file', type=str, default="./model.pbtxt", help='pbtxt generated in evaluation mode')
    parser.add_argument('--output_node_names', type=str, default="interpret_output/Reshape_1", help='output node name')
    parser.add_argument('--output_dir', type=str, default="./demo", help='output directory')
    parser.add_argument('--output_file', type=str, default="model.pb", help='output .pb file name')
    args = parser.parse_args()
    convert(args.check_point_dir, args.check_point_file, args.eval_pbtxt_file, args.output_node_names, args.output_dir, args.output_file)


