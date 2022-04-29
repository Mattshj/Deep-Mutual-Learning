"""
   Format Market-1501 training images with consecutive labels.

   This code modifies the data preparation method of
   "Learning Deep Feature Representations with Domain Guided Dropout for Person Re-identification".

"""

import shutil
from glob import glob
from datasets.utils import *

limit = 100000

labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

def _format_train_data(in_dir, output_dir):
    images = list()
    for label in labels:
        directory = osp.join(in_dir, label)
        temp = glob(osp.join(directory, '*.jpg'))[:limit]
        images.append(temp)
    images.sort()

    num_images = len(images)
    meta = {'name': 'garbage', 'count': num_images}
    write_json(meta, osp.join(output_dir, 'meta.json'))
    print("Training data has %d images of %d classes" % (num_images, len(labels)))


def run(image_dir):
    """Format the datasets with consecutive labels.

    Args:
        image_dir: The dataset directory where the raw images are stored.

    """
    in_dir = image_dir + "_raw"
    os.rename(image_dir, in_dir)
    mkdir_if_missing(image_dir)
    _format_train_data(in_dir, image_dir)
