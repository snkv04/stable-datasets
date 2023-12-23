import os
import pickle
import tarfile
import time
from ..utils import Dataset
from io import BytesIO
from PIL import Image
import numpy as np
from tqdm import tqdm



class FGVCAircraft(Dataset):
    """Image classification.

    """


    @property
    def urls(self):
        return {
        "fgvc-aircraft-2013b.tar.gz": "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz"
        }

    @property
    def extract(self):
        return ["fgvc-aircraft-2013b.tar.gz"]

    @property
    def num_classes(self):
        return 102
 
    @property
    def num_samples(self):
        return 10200

    @property
    def webpage(self):
        return "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/"

    def load(self):
        t0 = time.time()
    
        tar = tarfile.open(self.path / self.name / list(self.urls.keys())[0], 
                "r:gz")
    
        # Load train set
        print(tar.getmembers())
        base = self.path / self.name / "extracted_fgvc-aircraft-2013b.tar/fgvc-aircraft-2013b/data"
        for n in ["variants", "manufacturers", "families"]:
            self[n] = (base / (n + ".txt")).read_text().split("\n")
        for n in ["variant", "manufacturer", "family"]:
            for p in ["train", "test", "val"]:
                self[p+"_"+n] = (base / f"images_{n}_{p}.txt").read_text()

        val_names = (base / "images_val.txt").read_text().split("\n")
        train_names = (base / "images_train.txt").read_text().split("\n")
        test_names = (base / "images_test.txt").read_text().split("\n")

        train_images, test_images, val_images = [], [], []
        for images, names, desc in zip([train_images, test_images, val_images], [train_names, test_names, val_names], ["Train", "Test", "Val"]):
            for name in tqdm(names, desc=f"{desc} images"):
                im = Image.open(base / "images" / f"{name}.jpg")
                images.append(im.copy())
                im.close()
        print(f"Dataset {self.name} loaded in {0:.2f}s.".format(time.time() - t0))
