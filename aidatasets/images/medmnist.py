import os
import pickle
import tarfile
import time
from ..utils import Dataset
from io import BytesIO
import numpy as np
from tqdm import tqdm



class MedMNIST(Dataset):
    """Image classification.

    """

    @property
    def md5(self):
        return {
                "adrenalmnist3d.npz": "bbd3c5a5576322bc4cdfea780653b1ce",
                "bloodmnist.npz":"7053d0359d879ad8a5505303e11de1dc",
                "breastmnist.npz":"750601b1f35ba3300ea97c75c52ff8f6",
                "chestmnist.npz":"02c8a6516a18b556561a56cbdd36c4a8",
                "dermamnist.npz":"0744692d530f8e62ec473284d019b0c7",
                "fracturemnist3d.npz":"6aa7b0143a6b42da40027a9dda61302f",
                "nodulemnist3d":"8755a7e9e05a4d9ce80a24c3e7a256f3",
                "octmnist.npz":"c68d92d5b585d8d81f7112f81e2d0842",
                "organamnist.npz": "866b832ed4eeba67bfb9edee1d5544e6",
                "organcmnist.npz":"0afa5834fb105f7705a7d93372119a21",
                "organmnist3d.npz":"21f0a239e7f502e6eca33c3fc453c0b6",
                "organsmnist.npz":"e5c39f1af030238290b9557d9503af9d",
                "pathmnist.npz":"a8b06965200029087d5bd730944a56c1",
                "pneumoniamnist.npz":"28209eda62fecd6e6a2d98b1501bb15f",
                "retinamnist.npz":"bd4c0672f1bba3e3a89f0e4e876791e4",
                "synapsemnist3d.npz":"1235b78a3cd6280881dd7850a78eadb6",
                "tissuemnist.npz":"ebe78ee8b05294063de985d821c1c34b",
                "vesselmnist3d.npz":"2ba5b80617d705141f3f85627108fce8"
        }


    @property
    def urls(self):
        return {
                "adrenalmnist3d.npz": "https://zenodo.org/records/6496656/files/adrenalmnist3d.npz?download=1",
                "bloodmnist.npz":"https://zenodo.org/records/6496656/files/bloodmnist.npz?download=1",
                "breastmnist.npz":"https://zenodo.org/records/6496656/files/breastmnist.npz?download=1",
                "chestmnist.npz":"https://zenodo.org/records/6496656/files/chestmnist.npz?download=1",
                "dermamnist.npz":"https://zenodo.org/records/6496656/files/dermamnist.npz?download=1",
                "fracturemnist3d.npz":"https://zenodo.org/records/6496656/files/fracturemnist3d.npz?download=1",
                "nodulemnist3d":"https://zenodo.org/records/6496656/files/nodulemnist3d.npz?download=1",
                "octmnist.npz":"https://zenodo.org/records/6496656/files/octmnist.npz?download=1",
                "organamnist.npz": "https://zenodo.org/records/6496656/files/organamnist.npz?download=1",
                "organcmnist.npz":"https://zenodo.org/records/6496656/files/organcmnist.npz?download=1",
                "organmnist3d.npz":"https://zenodo.org/records/6496656/files/organmnist3d.npz?download=1",
                "organsmnist.npz":"https://zenodo.org/records/6496656/files/organsmnist.npz?download=1",
                "pathmnist.npz":"https://zenodo.org/records/6496656/files/pathmnist.npz?download=1",
                "pneumoniamnist.npz":"https://zenodo.org/records/6496656/files/pneumoniamnist.npz?download=1",
                "retinamnist.npz":"https://zenodo.org/records/6496656/files/retinamnist.npz?download=1",
                "synapsemnist3d.npz":"https://zenodo.org/records/6496656/files/synapsemnist3d.npz?download=1",
                "tissuemnist.npz":"https://zenodo.org/records/6496656/files/tissuemnist.npz?download=1",
                "vesselmnist3d.npz":"https://zenodo.org/records/6496656/files/vesselmnist3d.npz?download=1"
        }

    @property
    def names(self):
        return [i[:-4] for i in self.urls.keys()]

    @property
    def num_classes(self):
        if self._loaded_name == "bloodmnist":
            return 8
        elif self._loaded_name == "pathmnist":
            return 9
        elif self._loaded_name == "chestmnist":
            return 14
        elif self._loaded_name == "dermamnist":
            return 7
        elif self._loaded_name == "octmnist":
            return 4
        elif self._loaded_name == "pneumoniamnist":
            return 2
        elif self._loaded_name == "retinamnist":
            return 5
        elif self._loaded_name == "breastmnist":
            return 2
        elif self._loaded_name == "tissuemnist":
            return 8
        elif self._loaded_name == "organamnist":
            return 11
        elif self._loaded_name == "organcmnist":
            return 11
        elif self._loaded_name == "organmnist3d":
            return 11
        elif self._loaded_name == "nodulemnist3d":
            return 2
        elif self._loaded_name == "adrenalmnist3d":
            return 2
        elif self._loaded_name == "fracturemnist3d":
            return 3
        elif self._loaded_name == "vesselmnist3d":
            return 2
        elif self._loaded_name == "synapsemnist3d":
            return 2

    def load(self, name):
        assert name in self.names
        self._loaded_name = name
        data = np.load(self.path / self.name / f"{name}.npz")
        for n in ["train", "test", "val"]:
            self[f"{n}_X"] = data[f"{n}_images"]
            self[f"{n}_y"] = data[f"{n}_labels"].squeeze()

