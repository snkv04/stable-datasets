import os
import pickle
import tarfile
import time
from ..utils import Dataset

import numpy as np
from tqdm import tqdm



class Country211(Dataset):
    """Image classification.
    In the paper, we used an image classification dataset called Country211, to evaluate the model's capability on geolocation. To do so, we filtered the YFCC100m dataset that have GPS coordinate corresponding to a ISO-3166 country code and created a balanced dataset by sampling 150 train images, 50 validation images, and 100 test images images for each country.

    Parameters
    ----------

    path: str (optional)
        default ($DATASET_PATH), the path to look for the data and
        where the data will be downloaded if not present

    Returns
    -------

    train_images: array

    train_labels: array

    test_images: array

    test_labels: array

    """


    @property
    def urls(self):
        return {
        "country211.tgz": "https://openaipublic.azureedge.net/clip/data/country211.tgz"
        }

    @property
    def num_classes(self):
        return 211
 

    @property
    def website(self):
        return "https://github.com/openai/CLIP/blob/main/data/country211.md"


    def load(self):
        t0 = time.time()
    
        tar = tarfile.open(self.path / self.name / list(self.urls.keys())[0], 
                "r:gz")
    
        # Load train set
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
