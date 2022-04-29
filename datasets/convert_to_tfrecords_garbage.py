"""
    Convert Market-1501 to TFRecords of TF-Example protos.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import sys
from scipy import misc
from datasets.dataset_utils import *

# resize all the re-id images into the same size
_IMAGE_HEIGHT = 94
_IMAGE_WIDTH = 128
_IMAGE_CHANNELS = 3

labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']


def _add_to_tfrecord(image_dir, tfrecord_writer, split_name):
    """Loads images and writes files to a TFRecord.

    Args:
      image_dir: The image directory where the raw images are stored.
      tfrecord_writer: The TFRecord writer to use for writing.
    """

    num_images = 0

    for label in labels:
        directory = os.path.join(image_dir, label)
        value = len(os.listdir(directory))
        print("Images of label \"" + label + "\":\t", value)
        num_images += value

    shape = (_IMAGE_HEIGHT, _IMAGE_WIDTH, _IMAGE_CHANNELS)

    with tf.Graph().as_default():
        image = tf.placeholder(dtype=tf.uint8, shape=shape)
        encoded_png = tf.image.encode_png(image)
        j = 0
        with tf.Session('') as sess:
            label_number = -1
            for label in labels:
                label_number += 1
                directory = os.path.join(image_dir, label)
                for imagename in os.listdir(directory):
                # for line in tf.gfile.FastGFile(list_filename, 'r').readlines():
                    sys.stdout.write('\r>> Converting %s image %d/%d' % (split_name, j + 1, num_images))
                    sys.stdout.flush()
                    j += 1

                    file_path = os.path.join(directory, imagename)
                    image_data = misc.imread(file_path)
                    image_data = misc.imresize(image_data, [_IMAGE_HEIGHT, _IMAGE_WIDTH])
                    png_string = sess.run(encoded_png, feed_dict={image: image_data})
                    example = image_to_tfexample(png_string, label_number, imagename, _IMAGE_HEIGHT, _IMAGE_WIDTH, 'png')
                    tfrecord_writer.write(example.SerializeToString())


def run(image_dir, output_dir, split_name):
    """Convert images to tfrecords.
    Args:
    image_dir: The image directory where the raw images are stored.
    output_dir: The directory where the lists and tfrecords are stored.
    split_name: The split name of dataset.
    """
    tf_filename = os.path.join(output_dir, '%s.tfrecord' % split_name)

    if tf.gfile.Exists(tf_filename):
        print('Dataset files already exist. Exiting without re-creating them.')
        return

    with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
        _add_to_tfrecord(image_dir, tfrecord_writer, split_name)

    print(" Done! \n")

