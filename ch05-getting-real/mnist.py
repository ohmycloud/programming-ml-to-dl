import numpy as np
import gzip
import struct

def load_images(filename):
    # open and unzip the file images:
    with gzip.open(filename, 'rb') as f:
        # read the header information into a bunch of varables
        _ignored, n_images, columns, rows = struct.unpack('>IIII', f.read(16))
        # read all the pixels into a Numpy array of bytes:
        all_pixels = np.frombuffer(f.read(), dtype=np.uint8)
        # reshape the pixels into a matrix where each line is an image:
        return all_pixels.reshape(n_images, columns * rows)

def prepend_bias(X):
    # insert a column of 1s in the position 0 of X
    # ("axis=1" stands for: "insert a column, not a row")
    return np.insert(X, 0, 1, axis=1)

# 60000 images, each 785 elements (1 bias + 28 * 28 pixels)
X_train = prepend_bias(load_images("../data/mnist/train-images-idx3-ubyte.gz"))

# 10000 images, each 785 elements, with the same structure as X_train
X_test = prepend_bias(load_images("../data/mnist/t10k-images-idx3-ubyte.gz"))

def load_labels(filename):
    # open and unzip the file of images:
    with gzip.open(filename, 'rb') as f:
        # skip the header bytes
        f.read(8)
        # read all the labels into a list
        all_labels = f.read()
        # reshape the list of labels into a one-column matrix:
        return np.frombuffer(all_labels, dtype=np.uint8).reshape(-1, 1)

def encode_fives(Y):
    # convert all 5s to 1, and everything else yo 0
    return (Y == 5).astype(int)

# 60K labels, each with value 1 if the digit is a five, and 0 otherwise
Y_train = encode_fives(load_labels("../data/mnist/train-labels-idx1-ubyte.gz"))

# 10000 images, each 785 elements, with the same structure as X_train
Y_test = encode_fives(load_labels("../data/mnist/t10k-labels-idx1-ubyte.gz"))
