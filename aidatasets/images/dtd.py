import os
import pickle
import tarfile
import time
from ..utils import Dataset
from io import BytesIO
from PIL import Image
import numpy as np
from tqdm import tqdm



class DTD(Dataset):
    """Image classification.

    """


    @property
    def urls(self):
        return {
        "dtd-r1.0.1.tar.gz": "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz",
        "dtd-r1.0.1-labels.tar.gz": "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1-labels.tar.gz",
        }

    @property
    def cite(self):
        return """
	    @InProceedings{cimpoi14describing,
	      Author    = {M. Cimpoi and S. Maji and I. Kokkinos and S. Mohamed and and A. Vedaldi},
	      Title     = {Describing Textures in the Wild},
	      Booktitle = {Proceedings of the {IEEE} Conf. on Computer Vision and Pattern Recognition ({CVPR})},
	      Year      = {2014}}"""


    @property
    def num_classes(self):
        return 47
 
    @property
    def webpage(self):
        return "https://www.robots.ox.ac.uk/~vgg/data/dtd/"

    def load(self, split=1):
        t0 = time.time()
    
        tar = tarfile.open(self.path / self.name / "dtd-r1.0.1-labels.tar.gz", "r:gz")
        train_names = tar.extractfile("labels/train{split}.txt").read().decode("utf-8")
        test_names = tar.extractfile("labels/train{split}.txt").read().decode("utf-8")
        val_names = tar.extractfile("labels/train{split}.txt").read().decode("utf-8")



    
        # Load train set
        print(tar.getmembers())
        val_names = tar.extractfile("fgvc-aircraft-2013b/data/images_val.txt").read().decode("utf-8")
        train_names = tar.extractfile("fgvc-aircraft-2013b/data/images_train.txt").read().decode("utf-8")
        test_names = tar.extractfile("fgvc-aircraft-2013b/data/images_test.txt").read().decode("utf-8")
        train_names = train_names.split("\n")
        train_images, test_images, val_images = [], [], []
        for name in tqdm(train_names, desc="Train images"):
            im = BytesIO(tar.extractfile(f"fgvc-aircraft-2013b/data/images/{name}.jpg").read())
            train_images.append(Image.open(im))
        for name in tqdm(val_names, desc="Val images"):
            im = BytesIO(tar.extractfile(f"fgvc-aircraft-2013b/data/images/{name}.jpg").read())
            val_images.append(Image.open(im))
        for name in tqdm(test_names, desc="Test images"):
            im = BytesIO(tar.extractfile(f"fgvc-aircraft-2013b/data/images/{name}.jpg").read())
            test_images.append(Image.open(im))



        asdf
        train_images = list()
        train_labels = list()
        for k in tqdm(range(1, 6), desc="Loading cifar10", ascii=True):
            f = tar.extractfile("cifar-10-batches-py/data_batch_" + str(k)).read()
            data_dic = pickle.loads(f, encoding="latin1")
            train_images.append(data_dic["data"].reshape((-1, 3, 32, 32)))
            train_labels.append(data_dic["labels"])
        train_images = np.concatenate(train_images, 0)
        train_labels = np.concatenate(train_labels, 0)
    
        # Load test set
        f = tar.extractfile("cifar-10-batches-py/test_batch").read()
        data_dic = pickle.loads(f, encoding="latin1")
        test_images = data_dic["data"].reshape((-1, 3, 32, 32))
        test_labels = np.array(data_dic["labels"])
    
        self["train_X"] =  np.transpose(train_images, (0, 2, 3, 1))
        self["train_y"] =  train_labels
        self["test_X"] =  np.transpose(test_images, (0, 2, 3, 1))
        self["test_y"] = test_labels
        print("Dataset cifar10 loaded in{0:.2f}s.".format(time.time() - t0))
