from stable_datasets.images.cc3m import CC3M

def test_cc3m_dataset():
    # Large dataset, so using large temporary directory
    validation_dataset = CC3M(
        split="validation",
        download_dir=f"/ltmp/.stable_datasets/downloads",
        processed_cache_dir=f"/ltmp/.stable_datasets/processed",
    )
    sample = validation_dataset[0]
    print(sample.keys())  # {"image", "caption"}

if __name__ == "__main__":
    test_cc3m_dataset()
