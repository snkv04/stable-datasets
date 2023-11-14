import os
import numpy as np
from ..utils import download_dataset
import time
import io
from tqdm import tqdm
import matplotlib.image as mpimg
from zipfile import ZipFile


_source = "https://github.com/mloey/Arabic-Handwritten-Digits-Dataset"

cite = """
@inproceedings{el2016cnn,
  title={CNN for handwritten arabic digits recognition based on LeNet-5},
  author={El-Sawy, Ahmed and Hazem, EL-Bakry and Loey, Mohamed},
  booktitle={International conference on advanced intelligent systems and informatics},
  pages={566--575},
  year={2016},
  organization={Springer}
}"""

_name = "arabic_digits"

_urls = {
    "TestImages.zip": "https://github.com/mloey/Arabic-Handwritten-Digits-Dataset/raw/master/Test%20Images.zip",
    "TrainImages.zip": "https://github.com/mloey/Arabic-Handwritten-Digits-Dataset/raw/master/Train%20Images.zip",
}

SHAPE = (1, 28, 28)
N_SAMPLES = 60000


def load(path=None):
    """Arabic Handwritten Digits Dataset


    Parameters
    ----------

        path: str (optional)
            default ($DATASET_PATH), the path to look for the data and
            where the data will be downloaded if not present

    Returns
    -------

        train_images: array

        train_labels: array

        valid_images: array

        valid_labels: array

        test_images: array

        test_labels: array

    """

    t0 = time.time()
    download_dataset(_name, _urls, path=path)
    train_images = []
    test_images = []
    train_labels = []
    test_labels = []
    with ZipFile(os.path.join(path, _name, "TestImages.zip")) as archive:
        for entry in tqdm(archive.infolist()):
            if ".png" not in entry.filename:
                continue
            content = archive.read(entry)
            test_images.append(mpimg.imread(io.BytesIO(content), "png"))
            test_labels.append(int(entry.filename.split("_")[-1][:-4]))

    with ZipFile(os.path.join(path, _name, "TrainImages.zip")) as archive:
        for entry in tqdm(archive.infolist(), ascii=True):
            if ".png" not in entry.filename:
                continue
            content = archive.read(entry)
            train_images.append(mpimg.imread(io.BytesIO(content), "png"))
            train_labels.append(int(entry.filename.split("_")[-1][:-4]))
    dataset = {
        "train": {
            "X": np.array(train_images)[:, None].astype("float32"),
            "y": np.array(train_labels).astype("int"),
        },
        "val": {
            "X": np.array(test_images)[:, None].astype("float32"),
            "y": np.array(test_labels).astype("int"),
        },
    }

    print("Dataset arabic_digits loaded in {0:.2f}s.".format(time.time() - t0))

    return dataset
