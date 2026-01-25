import numpy as np
from loguru import logger as logging
from PIL import Image
from tqdm import tqdm
from stable_datasets.images.cc3m import CC3M

def test_cc3m_dataset():
    # Test 1: Checks that the validation dataset is not empty
    # Large dataset, so using large temporary directory
    download_dir = "/ltmp/.stable_datasets/downloads"
    processed_cache_dir = "/ltmp/.stable_datasets/processed"
    validation_dataset = CC3M(
        split="validation",
        download_dir=download_dir,
        processed_cache_dir=processed_cache_dir,
    )
    assert len(validation_dataset) > 0, f"Validation dataset should not be empty"
    logging.info(f"Loaded {len(validation_dataset)} examples from validation split.")

    # Test 2: Checks that the keys are correct
    first_sample = validation_dataset[0]
    actual_keys = set(first_sample.keys())
    expected_keys = {"image", "caption"}
    assert actual_keys == expected_keys, f"Expected keys {expected_keys}, got {actual_keys}"

    # Test 3: Checks that all the captions are non-empty
    for sample in tqdm(validation_dataset, desc="Checking captions"):
        caption = sample["caption"]
        assert isinstance(caption, str), f"Caption should be a string, got {type(caption)}"
        assert caption, f"Caption should be non-empty"

    # Test 4: Checks that the image is a PIL image
    image = first_sample["image"]
    assert isinstance(image, Image.Image), f"Image should be a PIL image, got {type(image)}"

    # Test 5: Casts image to numpy ndarray and checks properties
    image_np = np.asarray(image)
    assert image_np.ndim == 3, f"Image should be a 3D numpy array, got shape {image_np.shape}"
    assert image_np.shape[2] == 3, f"Image should have 3 channels, got {image_np.shape[2]} channels"
    assert image_np.dtype == np.uint8, f"Image should have dtype uint8, got {image_np.dtype}"

    # Test 6: Checks that the training dataset is not empty
    train_dataset = CC3M(
        split="train",
        download_dir=download_dir,
        processed_cache_dir=processed_cache_dir,
    )
    assert len(train_dataset) > 0, f"Training dataset should not be empty"
    logging.info(f"Loaded {len(train_dataset)} examples from train split.")

    # Can't be tested deterministically:
    # - Image labels, because the link to the dataset containing image labels (located on the
    # CC3M website at https://ai.google.com/research/ConceptualCaptions/download) does not
    # authorize public access
    # - Number of samples, because the owners of the images can make the images inaccessible
    # at any time by invalidating the URLs, so the number of available images is not constant

    logging.info("All CC3M dataset tests passed successfully!")

if __name__ == "__main__":
    test_cc3m_dataset()
