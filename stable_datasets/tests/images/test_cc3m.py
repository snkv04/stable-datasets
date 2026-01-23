import os

from stable_datasets.images.cc3m import CC3M

def test_cc3m_dataset():
    validation_dataset = CC3M(
        split="validation",
        download_dir=f"/cs/data/people/{os.environ['USER']}/.stable_datasets/downloads",
        processed_cache_dir=f"/cs/data/people/{os.environ['USER']}/.stable_datasets/processed",
    )
    sample = validation_dataset[0]
    print(sample.keys())  # {"image", "caption"}

if __name__ == "__main__":
    test_cc3m_dataset()
