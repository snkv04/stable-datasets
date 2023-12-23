import os
from io import BytesIO
import tarfile
from zipfile import ZipFile
import urllib.request
import numpy as np
import imageio
from ..utils import Dataset
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

from PIL import Image


class TinyImagenet(Dataset):
    """
    Tiny Imagenet has 200 classes. Each class has 500 training images, 50
    validation images, and 50 test images. We have released the training and
    validation sets with images and annotations. We provide both class labels an
    bounding boxes as annotations; however, you are asked only to predict the
    class label of each image without localizing the objects. The test set is
    released without labels. You can download the whole tiny ImageNet dataset
    here.
    """

    @property
    def urls(self):
        return {"tiny-imagenet-200.zip": "http://cs231n.stanford.edu/tiny-imagenet-200.zip"}

    @property
    def md5(self):
        return {"tiny-imagenet-200.zip": "90528d7ca1a48142e341f4ef8d21d0de"}

    @property
    def image_shape(self):
        return (3, 64, 64)

    @property
    def num_classes(self):
        return 200


    def load(self):
        # Loading the file
        f = ZipFile(self.path / self.name/ "tiny-imagenet-200.zip", "r")
        names = [name for name in f.namelist() if name.endswith("JPEG")]
        val_classes = np.loadtxt(
            f.open("tiny-imagenet-200/val/val_annotations.txt"),
            dtype=str,
            delimiter="\t",
        )
        val_classes = dict([(a, b) for a, b in zip(val_classes[:, 0], val_classes[:, 1])])
        x_train, x_test, x_valid, y_train, y_test, y_valid = [], [], [], [], [], []
        for name in tqdm(names, desc=f"Loading {self.name}"):
            im = Image.open(f.open(name)).convert("RGB")
            if "train" in name:
                classe = name.split("/")[-1].split("_")[0]
                x_train.append(im)
                y_train.append(classe)
            if "val" in name:
                x_valid.append(im)
                arg = name.split("/")[-1]
                y_valid.append(val_classes[arg])
            if "test" in name:
                x_test.append(im)
        labels = LabelEncoder().fit(y_train)
        self["train_X"] = x_train
        self["train_y"] = labels.transform(y_train)
        self["test_X"] = x_valid
        self["test_y"] = labels.transform(y_valid)




class TinyImagenetC(TinyImagenet):

    @property
    def urls(self):
        return {"Tiny-ImageNet-C.tar": "https://zenodo.org/records/2536630/files/Tiny-ImageNet-C.tar?download=1"}

    @property
    def webpage(self):
        return "https://zenodo.org/records/2536630"

    @property
    def corruptions(self):
        return [
                "zoom_blur",
                "snow",
                "shot_noise",
                "pixelate",
                "motion_blur",
                "jpeg_compression",
                "impulse_noise",
                "glass_blur",
                "gaussian_noise",
                "frost",
                "fog",
                "elastic_transform",
                "defocus_blur",
                "contrast",
                "brightness"
                ]

    def load(self):
        # Loading the file
        f = tarfile.open(self.path / self.name / "Tiny-ImageNet-C.tar", "r")
        names = [name.name for name in f.getmembers() if name.name.endswith("JPEG")]
        if type(self.corruption) == str:
            corruption = [self.corruption]
        else:
            corruption = self.corruption

        names = [n for n in names if n.split("/")[1] in corruption]
        x, y, level, corruption = [], [], [], []
        for name in tqdm(names, desc=f"Loading {self.name}"):
            x.append(Image.open(BytesIO(f.extractfile(name).read())).convert("RGB"))
            y.append(name.split("/")[3])
            level.append(name.split("/")[2])
            corruption.append(name.split("/")[1])
        self["X"] = x
        self["y"] = y
        self["level"] = level
        self["corruption_name"] = corruption
