"""
    Format Market-1501 training images and convert all the splits into TFRecords
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from datasets import convert_to_tfrecords
from datasets import format_market_train
from datasets import make_filename_list
from datasets.utils import *

my_flag = 'train'

FLAGS = tf.app.flags.FLAGS

if my_flag == 'train':
    tf.app.flags.DEFINE_string('image_dir', '/home/hamid/PycharmProjects/Deep-Mutual-Learning/datasets/Market-1501/bounding_box_train', 'path to the raw images')
    tf.app.flags.DEFINE_string('output_dir', '/home/hamid/PycharmProjects/Deep-Mutual-Learning/datasets/results/market-1501/tfrecords', 'path to the list and tfrecords ')
    tf.app.flags.DEFINE_string('split_name', 'bounding_box_train', 'split name')
elif my_flag == 'test':
    tf.app.flags.DEFINE_string('image_dir', '/home/hamid/PycharmProjects/Deep-Mutual-Learning/datasets/Market-1501/bounding_box_test', 'path to the raw images')
    tf.app.flags.DEFINE_string('output_dir', '/home/hamid/PycharmProjects/Deep-Mutual-Learning/datasets/results/market-1501/tfrecords', 'path to the list and tfrecords ')
    tf.app.flags.DEFINE_string('split_name', 'bounding_box_test', 'split name')
elif my_flag == 'gtbox':
    tf.app.flags.DEFINE_string('image_dir', '/home/hamid/PycharmProjects/Deep-Mutual-Learning/datasets/Market-1501/gt_bbox', 'path to the raw images')
    tf.app.flags.DEFINE_string('output_dir', '/home/hamid/PycharmProjects/Deep-Mutual-Learning/datasets/results/market-1501/tfrecords', 'path to the list and tfrecords ')
    tf.app.flags.DEFINE_string('split_name', 'gt_bbox', 'split name')
elif my_flag == 'query':
    tf.app.flags.DEFINE_string('image_dir', '/home/hamid/PycharmProjects/Deep-Mutual-Learning/datasets/Market-1501/query', 'path to the raw images')
    tf.app.flags.DEFINE_string('output_dir', '/home/hamid/PycharmProjects/Deep-Mutual-Learning/datasets/results/market-1501/tfrecords', 'path to the list and tfrecords ')
    tf.app.flags.DEFINE_string('split_name', 'query', 'split name')


def main(_):
    mkdir_if_missing(FLAGS.output_dir)

    if FLAGS.split_name == 'bounding_box_train':
        format_market_train.run(image_dir=FLAGS.image_dir)

    make_filename_list.run(image_dir=FLAGS.image_dir,
                           output_dir=FLAGS.output_dir,
                           split_name=FLAGS.split_name)

    convert_to_tfrecords.run(image_dir=FLAGS.image_dir,
                             output_dir=FLAGS.output_dir,
                             split_name=FLAGS.split_name)


if __name__ == '__main__':
    tf.app.run()
