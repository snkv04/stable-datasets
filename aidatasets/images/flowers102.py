import os
from io import BytesIO
from tqdm import tqdm
import pickle
import tarfile
import time
from ..utils import Dataset
import scipy.io
import numpy as np
from PIL import Image
from pathlib import Path

class Flowers102(Dataset):
    """Image classification.

    We have created a 102 category dataset, consisting of 102 flower categories. The flowers chosen to be flower commonly occuring in the United Kingdom. Each class consists of between 40 and 258 images. The details of the categories and the number of images for each class can be found on this category statistics page.

The images have large scale, pose and light variations. In addition, there are categories that have large variations within the category and several very similar categories. The dataset is visualized using isomap with shape and colour features. """

    @property
    def website(self):
        return "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/"

    @property
    def urls(self):
        return {
                "102flowers.tgz": "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz",
                "imagelabels.mat":"https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat",
                "setid.mat":"https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat",
                "102segmentations.tgz":"https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102segmentations.tgz"
                }

    @property
    def num_classes(self):
        return 102

    @property
    def extract(self):
        return ["102flowers.tgz"]
    
    def load(self):
    
        t0 = time.time()
        # Loading the file
        # labels are 1-indexed so we shift them
        labels = scipy.io.loadmat(self.path / self.name / "imagelabels.mat")["labels"][0] - 1
        ids = scipy.io.loadmat(self.path / self.name / "setid.mat")
        train_ids = list(ids["trnid"][0])
        test_ids = list(ids["tstid"][0])
        valid_ids = list(ids["valid"][0])
        print(set(train_ids) - set(test_ids))
        train_images, test_images, valid_images, train_labels, test_labels, valid_labels = [], [], [], [], [], []
        for id_ in tqdm(train_ids, desc= "Loading train images"):
            name = self.path / self.name / "extracted_102flowers" / "jpg" / f"image_{id_:05d}.jpg"
            im = Image.open(name)
            train_images.append(im.copy())
            im.close()
            train_labels.append(labels[id_-1])
        for id_ in tqdm(test_ids, desc= "Loading test images"):
            name = self.path / self.name / "extracted_102flowers" / "jpg" / f"image_{id_:05d}.jpg"
            im = Image.open(name)
            test_images.append(im.copy())
            im.close()
            test_labels.append(labels[id_-1])
        for id_ in tqdm(valid_ids, desc= "Loading valid images"):
            name = self.path / self.name / "extracted_102flowers" / "jpg" / f"image_{id_:05d}.jpg"
            im = Image.open(name)
            valid_images.append(im.copy())
            im.close()
            valid_labels.append(labels[id_-1])

        self["train_X"] = train_images
        self["test_X"] = test_images
        self["valid_X"] = valid_images
        self["train_y"] = train_labels
        self["test_y"] = test_labels
        self["valid_y"] = valid_labels
        print(self["train_y"])
        print("Dataset Flowers102 loaded in {0:.2f}s.".format(time.time() - t0))
