import os
from tqdm import tqdm
import numpy as np
import imageio
import tarfile
from ..utils import Dataset
from PIL import Image


class CUB200(Dataset):
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

    @property
    def urls(self):
        return {
            "CUB_200_2011.tgz": "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"
        }

    @property
    def md5(self):
        return {"CUB_200_2011.tgz": "97eceeb196236b17998738112f37df78"}

    @property
    def extract(self):
        return ["CUB_200_2011.tgz"]

    @property
    def num_classes(self):
        return 200

    @property
    def modalities(self):
        return dict(train_X="image", train_y=int)

    def load(self, path=None):

        base = self.path / self.name / "extracted_CUB_200_2011/CUB_200_2011"

        # Load the class names
        names = np.loadtxt(base / "classes.txt", dtype=str)
        classes = dict([[c, n.split(".")[1]] for c, n in enumerate(names[:, 1])])

        # Load Bounding boxes
        boxes = np.loadtxt(base / "bounding_boxes.txt", dtype="int32")
        bounding_boxes = dict()
        for i in range(boxes.shape[0]):
            bounding_boxes[str(boxes[i, 0])] = boxes[i, 1:]

        # Load dataset
        labels = list()
        boxes = list()
        data = list()
        for member in tqdm(list(base.rglob("./images/*/*.jpg"))):
            class_ = int(member.parent.name.split(".")[0]) - 1
            # image_id = member.name.split("_")[-1][:-4]
            # f = tar.extractfile(member)
            data.append(Image.open(member))
            labels.append(int(class_))
            # if image_id in bounding_boxes:
            #     boxes.append(bounding_boxes[image_id])
            # else:
            #     print(image_id)
            #     boxes.append(np.nan)
        # print(bounding_boxes)
        labels = np.array(labels).astype("int32")

        self["train_X"] = data
        self["train_y"] = labels
        self["boxes"] = boxes
        self["classes"] = classes
