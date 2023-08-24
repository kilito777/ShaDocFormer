import os
from .dataset_RGB import DataReader


def check_folder(img_dir, inp, tar, mode):
    assert os.path.exists(os.path.join(img_dir, mode, inp))
    assert os.path.exists(os.path.join(img_dir, mode, tar))


def get_data(img_dir, inp, tar, mode='train', img_options=None):
    check_folder(img_dir, inp, tar, mode)
    return DataReader(img_dir, inp, tar, mode, img_options)
