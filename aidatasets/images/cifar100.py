import os
import pickle
import tarfile
import time
from ..utils import download_dataset

import numpy as np


labels_list = [
    "apple",
    "aquarium_fish",
    "baby",
    "bear",
    "beaver",
    "bed",
    "bee",
    "beetle",
    "bicycle",
    "bottle",
    "bowl",
    "boy",
    "bridge",
    "bus",
    "butterfly",
    "camel",
    "can",
    "castle",
    "caterpillar",
    "cattle",
    "chair",
    "chimpanzee",
    "clock",
    "cloud",
    "cockroach",
    "couch",
    "crab",
    "crocodile",
    "cup",
    "dinosaur",
    "dolphin",
    "elephant",
    "flatfish",
    "forest",
    "fox",
    "girl",
    "hamster",
    "house",
    "kangaroo",
    "keyboard",
    "lamp",
    "lawn_mower",
    "leopard",
    "lion",
    "lizard",
    "lobster",
    "man",
    "maple_tree",
    "motorcycle",
    "mountain",
    "mouse",
    "mushroom",
    "oak_tree",
    "orange",
    "orchid",
    "otter",
    "palm_tree",
    "pear",
    "pickup_truck",
    "pine_tree",
    "plain",
    "plate",
    "poppy",
    "porcupine",
    "possum",
    "rabbit",
    "raccoon",
    "ray",
    "road",
    "rocket",
    "rose",
    "sea",
    "seal",
    "shark",
    "shrew",
    "skunk",
    "skyscraper",
    "snail",
    "snake",
    "spider",
    "squirrel",
    "streetcar",
    "sunflower",
    "sweet_pepper",
    "table",
    "tank",
    "telephone",
    "television",
    "tiger",
    "tractor",
    "train",
    "trout",
    "tulip",
    "turtle",
    "wardrobe",
    "whale",
    "willow_tree",
    "wolf",
    "woman",
    "worm",
]


_name = "cifar100"
_urls = {"cifar100.tar.gz": "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"}

N_SAMPLES = 50000
SHAPE = (3, 32, 32)


def load(path=None):
    """Image classification.

    The `CIFAR-100 < https: // www.cs.toronto.edu/~kriz/cifar.html >`_ dataset is
    just like the CIFAR-10, except it has 100 classes containing 600 images
    each. There are 500 training images and 100 testing images per class.
    The 100 classes in the CIFAR-100 are grouped into 20 superclasses. Each
    image comes with a "fine" label(the class to which it belongs) and a
    "coarse" label(the superclass to which it belongs)."""

    if path is None:
        path = os.environ["DATASET_PATH"]

    download_dataset(_name, _urls, path=path)

    t0 = time.time()
    # Loading the file
    tar = tarfile.open(os.path.join(path, _name, "cifar100.tar.gz"), "r:gz")

    # Loading training set
    f = tar.extractfile("cifar-100-python/train").read()
    data = pickle.loads(f, encoding="latin1")
    train_images = data["data"].reshape((-1, 3, 32, 32)).astype("float32")
    train_fine = np.array(data["fine_labels"])
    train_coarse = np.array(data["coarse_labels"])

    # Loading test set
    f = tar.extractfile("cifar-100-python/test").read()
    data = pickle.loads(f, encoding="latin1")
    test_images = data["data"].reshape((-1, 3, 32, 32)).astype("float32")
    test_fine = np.array(data["fine_labels"])
    test_coarse = np.array(data["coarse_labels"])

    dataset = {
        "train": {
            "X": np.transpose(train_images, (0, 2, 3, 1)),
            "y": train_fine,
            "y_coarse": train_coarse,
        },
        "val": {
            "X": np.transpose(test_images, (0, 2, 3, 1)),
            "y": test_fine,
            "y_coarse": test_coarse,
        },
    }
    print("Dataset cifar100 loaded in {0:.2f}s.".format(time.time() - t0))
    return dataset
