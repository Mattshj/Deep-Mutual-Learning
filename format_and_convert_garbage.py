"""
    Format Market-1501 training images and convert all the splits into TFRecords
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from datasets import convert_to_tfrecords, convert_to_tfrecords_garbage, make_filename_list_garbage
from datasets import format_market_train
from datasets import make_filename_list
from datasets.utils import *
import my_env

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('image_dir', f'{my_env.path}/datasets/garbage/garbage-classification', 'path to the raw images')
tf.app.flags.DEFINE_string('output_dir', f'{my_env.path}/datasets/results/garbage/tfrecords', 'path to the list and tfrecords ')
tf.app.flags.DEFINE_string('split_name', 'train', 'split name')



def main(_):
    mkdir_if_missing(FLAGS.output_dir)
    #
    # make_filename_list_garbage.run(image_dir=FLAGS.image_dir,
    #                        output_dir=FLAGS.output_dir,
    #                        split_name=FLAGS.split_name)

    convert_to_tfrecords_garbage.run(image_dir=FLAGS.image_dir,
                             output_dir=FLAGS.output_dir,
                             split_name=FLAGS.split_name)


if __name__ == '__main__':
    tf.app.run()
