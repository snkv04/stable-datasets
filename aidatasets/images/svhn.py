import scipy.io as sio
import os
from ..utils import download_dataset
import numpy as np
import time
from pathlib import Path

_name = "svhn"
_urls = {
    "train_32x32.mat": "http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
    "test_32x32.mat": "http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
}
SHAPE = (3, 32, 32)
N_SAMPLES = 73257


def load(path=None):
    """Street number classification.

    The `SVHN <http://ufldl.stanford.edu/housenumbers/>`_
    dataset is a real-world
    image dataset for developing machine learning and object
    recognition algorithms with minimal requirement on data
    preprocessing and formatting. It can be seen as similar in flavor
    to MNIST (e.g., the images are of small cropped digits), but
    incorporates an order of magnitude more labeled data (over 600,000
    digit images) and comes from a significantly harder, unsolved,
    real world problem (recognizing digits and numbers in natural
    scene images). SVHN is obtained from house numbers in Google
    Street View images.

    Parameters
    ----------
        path: str (optional)
            default $DATASET_PATH, the path to look for the data and
            where the data will be downloaded if not present

    Returns
    -------

        train_images: array

        train_labels: array

        test_images: array

        test_labels: array


    """

    download_dataset(_name, _urls, path=path)

    # Load the dataset (download if necessary) and set
    # the class attributess.
    print("Loading svhn")

    t0 = time.time()

    # Train set
    data = sio.loadmat(Path(path) / "svhn/train_32x32.mat")
    train_images = data["X"].transpose([3, 0, 1, 2])
    train_labels = np.squeeze(data["y"]) - 1

    # Test set
    data = sio.loadmat(Path(path) / "svhn/test_32x32.mat")
    test_images = data["X"].transpose([3, 0, 1, 2])
    test_labels = np.squeeze(data["y"]) - 1

    print("Dataset svhn loaded in", "{0:.2f}".format(time.time() - t0), "s.")

    dataset = {
        "train": {"X": np.array(train_images), "y": np.array(train_labels)},
        "val": {"X": np.array(test_images), "y": np.array(test_labels)},
    }

    return dataset
