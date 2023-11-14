import os
from tqdm import tqdm
import urllib.request
import numpy as np
import imageio
import tarfile
from ..utils import download_dataset
from PIL import Image

_urls = {
    "CUB_200_2011.tgz": "http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz"
}

N_SAMPLES = 11788
N_CLASSES = 200
_name = "cub200"


def load(path=None):
    """Image classification of bird species.
    The `CUB-200 <http://www.vision.caltech.edu/visipedia/CUB-200-2011.html>`_.
    dataset  contains  11,788  images of  200  bird  species
    This dataset contains 8732 labeled sound excerpts (<=4s) of urban sounds
    from 10 classes: air_conditioner, car_horn, children_playing,
    dog_bark, drilling, enginge_idling, gun_shot, jackhammer, siren,
    and street_music. The classes are drawn from the
    `urban sound taxonomy <https://urbansounddataset.weebly.com/taxonomy.html>`_.
    The dataset is obtained from `Kaggle <https://www.kaggle.com/pavansanagapati/urban-sound-classification>_`
    """

    download_dataset(_name, _urls, path=path)

    tar = tarfile.open(path + "cub200/CUB_200_2011.tgz", "r:gz")

    # Load the class names
    f = tar.extractfile("CUB_200_2011/classes.txt")
    names = np.loadtxt(f, dtype="str")
    classes = dict([[c, n.split(".")[1]] for c, n in enumerate(names[:, 1])])

    # Load Bounding boxes
    f = tar.extractfile("CUB_200_2011/bounding_boxes.txt")
    boxes = np.loadtxt(f, dtype="int32")
    bounding_boxes = dict()
    for i in range(boxes.shape[0]):
        bounding_boxes[str(boxes[i, 0])] = boxes[i, 1:]

    # Load dataset
    labels = list()
    boxes = list()
    for member in tqdm(tar.getmembers()):
        if "CUB_200_2011/images/" in member.name and "jpg" in member.name:
            class_ = member.name.split("/")[2].split(".")[0]
            image_id = member.name.split("_")[-1][:-4]
            f = tar.extractfile(member)
            data.append(Image.open(f).convert("RGB"))
            labels.append(int(class_))
            boxes.append(bounding_boxes[image_id])
    labels = np.array(labels).astype("int32")

    # data = {
    #     "images": data,
    #     "labels": labels,
    #     "boxes": boxes,
    #     "classes": classes,
    # }

    dataset = {"train": {"X": data, "y": labels}}

    return dataset
