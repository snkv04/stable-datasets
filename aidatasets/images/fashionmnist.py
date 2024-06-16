import os
import gzip
import numpy as np
import time
from ..utils import Dataset
from pathlib import Path


class FashionMNIST(Dataset):
    """Grayscale image classification

    `Zalando <https://jobs.zalando.com/tech/>`_ 's article image classification.
    `Fashion-MNIST <https://github.com/zalandoresearch/fashion-mnist>`_ is
    a dataset of `Zalando <https://jobs.zalando.com/tech/>`_ 's article
    images consisting of a training set of 60,000 examples and a test set
    of 10,000 examples. Each example is a 28x28 grayscale image, associated
    with a label from 10 classes. We intend Fashion-MNIST to serve as a direct
    drop-in replacement for the original MNIST dataset for benchmarking
    machine learning algorithms. It shares the same image size and structure
    of training and testing splits.
    """

    @property
    def urls(self):
        return {
            "train-images.gz": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",
            "train-labels.gz": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz",
            "test-images.gz": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
            "test-labels.gz": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz",
        }

    @property
    def md5(self):
        return {
            "train-images.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
            "train-labels.gz": "25c81989df183df01b3e8a0aad5dffbe",
            "test-images.gz": "bef4ecab320f06d8554ea6380940ec79",
            "test-labels.gz": "bb300cfdad3c16e7a12a480ee83cd310",
        }

    @property
    def modalities(self):
        return dict(
            train_X="image",
            test_X="image",
            val_X="image",
            train_y="integer",
            val_y="integer",
        )

    @property
    def image_shape(self):
        return (28, 28, 1)

    def load(self):
        t0 = time.time()
        print("\tLoading fashionmnist")
        with gzip.open(self.path / self.name / "train-labels.gz", "rb") as lbpath:
            train_labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

        with gzip.open(self.path / self.name / "train-images.gz", "rb") as lbpath:
            train_images = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=16)
        train_images = train_images.reshape((-1, 28, 28, 1)).astype("float32")

        with gzip.open(self.path / self.name / "test-labels.gz", "rb") as lbpath:
            test_labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

        with gzip.open(self.path / self.name / "test-images.gz", "rb") as lbpath:
            test_images = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=16)
        self["test_X"] = test_images.reshape((-1, 28, 28, 1))

        self["train_X"] = np.array(train_images)
        self["train_y"] = np.array(train_labels)
        self["val_X"] = np.array(test_images)
        self["val_y"] = np.array(test_labels)
        print(f"Dataset {self.name} loaded in {0:.2f}s.".format(time.time() - t0))
        return self
