import os
import pickle
import tarfile
import time
from ..utils import Dataset
from io import BytesIO
import numpy as np


class CIFAR100(Dataset):
    """Image classification.

    The `CIFAR-100 < https: // www.cs.toronto.edu/~kriz/cifar.html >`_ dataset is
    just like the CIFAR-10, except it has 100 classes containing 600 images
    each. There are 500 training images and 100 testing images per class.
    The 100 classes in the CIFAR-100 are grouped into 20 superclasses. Each
    image comes with a "fine" label(the class to which it belongs) and a
    "coarse" label(the superclass to which it belongs)."""

    @property
    def label_to_name(label):
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

    @property
    def urls(self):
        return {
            "cifar100.tar.gz": "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
        }

    @property
    def md5(self):
        return {"cifar100.tar.gz": "eb9058c3a382ffc7106e4002c42a8d85"}

    @property
    def num_classes(self):
        return 100

    @property
    def image_shape(self):
        return (32, 32, 3)

    @property
    def modalities(self):
        return dict(
            train_X="image",
            test_X="image",
            train_y=int,
            test_y=int,
            train_y_coarse=int,
            test_y_coarse=int,
        )

    def load(self):
        t0 = time.time()
        # Loading the file
        tar = tarfile.open(self.path / self.name / list(self.urls.keys())[0], "r:gz")

        # Loading training set
        f = tar.extractfile("cifar-100-python/train").read()
        data = pickle.loads(f, encoding="latin1")
        train_images = data["data"].reshape((-1, 3, 32, 32))
        train_fine = np.array(data["fine_labels"])
        train_coarse = np.array(data["coarse_labels"])

        # Loading test set
        f = tar.extractfile("cifar-100-python/test").read()
        data = pickle.loads(f, encoding="latin1")
        test_images = data["data"].reshape((-1, 3, 32, 32))
        test_fine = np.array(data["fine_labels"])
        test_coarse = np.array(data["coarse_labels"])

        self["train_X"] = np.transpose(train_images, (0, 2, 3, 1))
        self["train_y"] = train_fine
        self["train_y_coarse"] = train_coarse
        self["test_X"] = np.transpose(test_images, (0, 2, 3, 1))
        self["test_y"] = test_fine
        self["test_y_coarse"] = test_coarse
        print("Dataset cifar100 loaded in {0:.2f}s.".format(time.time() - t0))
        return self


class CIFAR100C(CIFAR100):
    """Image classification.
    The `CIFAR-10 < https: // www.cs.toronto.edu/~kriz/cifar.html >`_ dataset
    was collected by Alex Krizhevsky, Vinod Nair, and Geoffrey
    Hinton. It consists of 60000 32x32 colour images in 10 classes, with
    6000 images per class. There are 50000 training images and 10000 test images.
    The dataset is divided into five training batches and one test batch,
    each with 10000 images. The test batch contains exactly 1000 randomly
    selected images from each class. The training batches contain the
    remaining images in random order, but some training batches may
    contain more images from one class than another. Between them, the
    training batches contain exactly 5000 images from each class.

    Parameters
    ----------

    path: str
        default ($DATASET_PATH), the path to look for the data and
        where the data will be downloaded if not present

    corruption: str or list
        which corruption version to use

    Returns
    -------

    train_images: array

    train_labels: array

    test_images: array

    test_labels: array

    """

    @property
    def corruptions(self):
        return [
            "zoom_blur",
            "speckle_noise",
            "spatter",
            "snow",
            "shot_noise",
            "saturate",
            "pixelate",
            "motion_blur",
            "jpeg_compression",
            "impulse_noise",
            "glass_blur",
            "gaussian_noise",
            "gaussian_blur",
            "frost",
            "fog",
            "elastic_transform",
            "defocus_blur",
            "contrast",
            "bightness",
        ]

    @property
    def urls(self):
        return {
            "CIFAR-100-C.tar": "https://zenodo.org/records/3555552/files/CIFAR-100-C.tar?download=1"
        }

    @property
    def md5(self):
        return {"CIFAR-100-C.tar": "11f0ed0f1191edbf9fa23466ae6021d3"}

    @property
    def webpage(self):
        return "https://zenodo.org/records/3555552"

    @property
    def modalities(self):
        return dict(X="image", y=int, corruption_name=str, corruption_level=int)

    def load(self, corruption=None):
        t0 = time.time()

        tar = tarfile.open(self.path / self.name / "CIFAR-100-C.tar", "r")

        # Load train set
        array_file = BytesIO()
        array_file.write(tar.extractfile("CIFAR-100-C/labels.npy").read())
        array_file.seek(0)
        labels = np.load(array_file)
        if type(corruption) == str:
            corruptions = [corruption]
        elif type(corruption) in [list, tuple]:
            corruptions = corruption
        else:
            corruptions = self.corruptions
        images = []
        names = []
        for c in corruptions:
            assert c in self.corruptions
            print(f"Loading corruption {c}")
            array_file = BytesIO()
            array_file.write(tar.extractfile(f"CIFAR-100-C/{c}.npy").read())
            array_file.seek(0)
            images.append(np.load(array_file))
            names.extend([c] * 10000)

        self["X"] = np.concatenate(images).transpose((0, 2, 3, 1))
        self["y"] = np.concatenate([labels for i in range(len(corruptions))])
        self["corruption_name"] = np.array(names)
        self["corruption_level"] = (
            np.arange(1, 6).repeat(10000).repeat(len(corruptions))
        )
        print("Dataset cifar100-C loaded in{0:.2f}s.".format(time.time() - t0))
        return self
