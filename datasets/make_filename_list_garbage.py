# """
#     Make a list for image files. This provides guidance for checking the data preparation.
# """
# import numpy as np
# from glob import glob
# from datasets.utils import *
#
# limit = 100000
#
# labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
#
#
# def _save(file_label_list, file_path):
#     content = ['{} {}'.format(x, y) for x, y in file_label_list]
#     write_list(content, file_path)
#
#
# def _get_train_list(files):
#     ret = []
#     for views in files:
#         for v in views:
#             for f in v:
#                 # camID = int(osp.basename(f)[4:6])
#                 label = int(osp.basename(f)[7:12])
#                 ret.append((f, label))
#     return np.asarray(ret)
#
#
# def _make_train_list(image_dir, output_dir, split_name):
#     meta = read_json(osp.join(image_dir, 'meta.json'))
#     identities = np.asarray(meta['count'])
#     images = _get_train_list(identities)
#     _save(images, os.path.join(output_dir, '%s.txt' % split_name))
#
#
# def run(image_dir, output_dir, split_name):
#     """Make list file for images.
#
#     Args:
#     image_dir: The image directory where the raw images are stored.
#     output_dir: The directory where the lists and tfrecords are stored.
#     split_name: The split name of dataset.
#     """
#     _make_train_list(image_dir, output_dir, split_name)
