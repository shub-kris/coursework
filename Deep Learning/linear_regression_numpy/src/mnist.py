import os
import struct
import numpy as np

"""
from https://gist.github.com/akesling/5358964, in turn loosely inspired by
http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
which is GPL licensed.
"""

def read(dataset = "training", path = "../data"):
    """
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    """

    if dataset is "training":
        fname_img = os.path.join(path, 'train-imagesidx3-ubyte.sec')
        fname_lbl = os.path.join(path, 'train-labelsidx1-ubyte.sec')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-imagesidx3-ubyte.sec')
        fname_lbl = os.path.join(path, 't10k-labelsidx1-ubyte.sec')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    return img, lbl
