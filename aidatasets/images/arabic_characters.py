import numpy as np
import io
from tqdm import tqdm
from PIL import Image
from zipfile import ZipFile
from ..utils import Dataset


class ArabicCharacters(Dataset):
    """Arabic Handwritten Characters Dataset

    Astract
    Handwritten Arabic character recognition systems face several challenges, including the unlimited variation in human handwriting and large public databases. In this work, we model a deep learning architecture that can be effectively apply to recognizing Arabic handwritten characters. A Convolutional Neural Network (CNN) is a special type of feed-forward multilayer trained in supervised mode. The CNN trained and tested our database that contain 16800 of handwritten Arabic characters. In this paper, the optimization methods implemented to increase the performance of CNN. Common machine learning methods usually apply a combination of feature extractor and trainable classifier. The use of CNN leads to significant improvements across different machine-learning classification algorithms. Our proposed CNN is giving an average 5.1% misclassification error on testing data.

    Context
    The motivation of this study is to use cross knowledge learned from multiple works to enhancement the performance of Arabic handwritten character recognition. In recent years, Arabic handwritten characters recognition with different handwriting styles as well, making it important to find and work on a new and advanced solution for handwriting recognition. A deep learning systems needs a huge number of data (images) to be able to make a good decisions.

    Content
    The data-set is composed of 16,800 characters written by 60 participants, the age range is between 19 to 40 years, and 90% of participants are right-hand. Each participant wrote each character (from ’alef’ to ’yeh’) ten times on two forms as shown in Fig. 7(a) & 7(b). The forms were scanned at the resolution of 300 dpi. Each block is segmented automatically using Matlab 2016a to determining the coordinates for each block. The database is partitioned into two sets: a training set (13,440 characters to 480 images per class) and a test set (3,360 characters to 120 images per class). Writers of training set and test set are exclusive. Ordering of including writers to test set are randomized to make sure that writers of test set are not from a single institution (to ensure variability of the test set).


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

    _source = "https://github.com/mloey/Arabic-Handwritten-Characters-Dataset"

    cite = """
    @article{el2017arabic,
    title={Arabic handwritten characters recognition using convolutional neural network},
    author={El-Sawy, Ahmed and Loey, Mohamed and El-Bakry, Hazem},
    journal={WSEAS Transactions on Computer Research},
    volume={5},
    pages={11--19},
    year={2017}
    }"""

    @property
    def md5(self):
        return {
            "TestImages32x32.zip": "71b492694895e3c39660d023c077779c",
            "TrainImages32x32.zip": "a4b80c95fc07ff69d219daa58cf71a34",
        }

    @property
    def urls(self):
        return {
            "TestImages32x32.zip": "https://github.com/mloey/Arabic-Handwritten-Characters-Dataset/raw/master/Test%20Images%203360x32x32.zip",
            "TrainImages32x32.zip": "https://github.com/mloey/Arabic-Handwritten-Characters-Dataset/raw/master/Train%20Images%2013440x32x32.zip",
        }

    @property
    def image_shape(self):
        return (28, 28, 1)

    @property
    def modalities(self):
        return dict(train_X="image", test_X="image", train_y=int, test_y=int)

    def load(self):

        train_images = []
        test_images = []
        train_labels = []
        test_labels = []
        with ZipFile(self.path / self.name / "TestImages32x32.zip") as archive:
            for entry in tqdm(archive.infolist()):
                if ".png" not in entry.filename:
                    continue
                content = archive.read(entry)
                test_images.append(Image.open(io.BytesIO(content)))
                test_labels.append(int(entry.filename.split("_")[-1][:-4]) - 1)

        with ZipFile(self.path / self.name / "TrainImages32x32.zip") as archive:
            for entry in tqdm(archive.infolist()):
                if ".png" not in entry.filename:
                    continue
                content = archive.read(entry)
                train_images.append(Image.open(io.BytesIO(content)))
                train_labels.append(int(entry.filename.split("_")[-1][:-4]) - 1)

        self["train_X"] = np.array(train_images)
        self["train_y"] = np.array(train_labels)
        self["test_X"] = np.array(test_images)
        self["test_y"] = np.array(test_labels)
        return self
