# Python Standard Library imports
import csv
import os
from functools import partial

# Third-party imports
from pathlib import Path
from urllib.parse import urlparse
import datasets
from loguru import logger as logging
from PIL import Image
from tqdm import tqdm
from joblib import Parallel, delayed
from concurrent.futures import ProcessPoolExecutor
from typing import Any

# Local imports
from stable_datasets.utils import download, _default_dest_folder, BaseDatasetBuilder


def call_with_timeout(func, timeout=10, log_timeout=False, *args, **kwargs) -> Any:
    with ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            if log_timeout:
                logging.warning(f"{func.__name__} timed out after {timeout} seconds")
            return None


def safe_download(url, dest_folder, log_failure=False) -> Path | None:
    try:
        return call_with_timeout(
            download,
            url=url,
            dest_folder=dest_folder,
            progress_bar=False,
            disable_logging=True,
        )
    except Exception as e:
        if log_failure:
            logging.warning(f"Failed to download {url}: {e}")
        return None


def safe_bulk_download(
    urls,
    dest_folder,
    num_processes=None,
    log_failure=False,
    parallel=False,
) -> list[Path | None]:
    if num_processes is None:
        num_processes = os.cpu_count() // 2
    
    download_func = partial(
        safe_download,
        dest_folder=dest_folder,
        log_failure=log_failure
    )

    if parallel:
        results = Parallel(n_jobs=num_processes)(
            delayed(download_func)(url)
            for url in tqdm(urls, desc=f"Downloading {len(urls)} images")
        )
    else:
        results = [download_func(url) for url in tqdm(urls, desc=f"Downloading {len(urls)} images")]
    return results


class CC3M(BaseDatasetBuilder):
    """CC3M"""

    VERSION = datasets.Version("1.0.0")

    SOURCE = {
        "homepage": "https://ai.google.com/research/ConceptualCaptions/download",
        "assets": {
            # "train": "https://storage.googleapis.com/gcc-data/Train/GCC-training.tsv",
            "val": "https://storage.googleapis.com/gcc-data/Validation/GCC-1.1.0-Validation.tsv",
        },
        "citation": """@inproceedings{sharma-etal-2018-conceptual,
                        title = "Conceptual Captions: A Cleaned, Hypernymed, Image Alt-text Dataset For Automatic Image Captioning",
                        author = "Sharma, Piyush  and
                        Ding, Nan  and
                        Goodman, Sebastian  and
                        Soricut, Radu",
                        editor = "Gurevych, Iryna  and
                        Miyao, Yusuke",
                        booktitle = "Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
                        month = jul,
                        year = "2018",
                        address = "Melbourne, Australia",
                        publisher = "Association for Computational Linguistics",
                        url = "https://aclanthology.org/P18-1238/",
                        doi = "10.18653/v1/P18-1238",
                        pages = "2556--2565",
                    }""",
    }

    def _info(self):
        return datasets.DatasetInfo(
            description="""Conceptual Captions, colloquially known as CC3M, is a dataset of 3 million images with their corresponding captions constructed by a purely automated pipeline developed at Google AI. It is a large-scale dataset typically used for image captioning.""",
            features=datasets.Features(
                {"image": datasets.Image(), "caption": datasets.Value("string")}
            ),
            supervised_keys=("image", "caption"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    def _generate_examples(self, data_path, split, download_batch_size=1024, log_failures=False):        
        # Creates a subfolder for this split's images
        download_dir = getattr(self, "_raw_download_dir", _default_dest_folder())
        images_dir = Path(download_dir) / f"cc3m_{split}_images"
        images_dir.mkdir(parents=True, exist_ok=True)

        # Computes batch info
        total_lines = sum(1 for _ in open(data_path))
        assert total_lines != 0, f"No {split} images found in {data_path}"
        logging.info(f"Total number of {split} images: {total_lines}")
        num_batches = (total_lines + download_batch_size - 1) // download_batch_size
        last_batch_size = total_lines % download_batch_size
        if last_batch_size == 0:
            last_batch_size = download_batch_size
        logging.info(f"Total number of {split} download batches: {num_batches}")
        logging.info(f"Size of last download batch: {last_batch_size}")

        # First pass: downloads all images in batches, where the images in
        # each batch are downloaded in parallel
        successful_downloads = 0
        skipped_downloads = 0
        failed_downloads = 0
        with open(data_path) as f:
            reader = csv.reader(f, delimiter='\t')
            for batch_idx in tqdm(range(num_batches), desc=f"Processing {split} download batches"):
                # Finds batch size
                if batch_idx == num_batches - 1:
                    curr_batch_size = last_batch_size
                else:
                    curr_batch_size = download_batch_size

                # Gets all URLs for this batch by iterating through the rows of the file
                urls_to_download = []
                for _ in tqdm(range(curr_batch_size), desc=f"Getting URLs for batch {batch_idx} of split {split}"):
                    # Gets row
                    row = next(reader)
                    caption, image_url = row

                    # Checks if image file already exists
                    try:
                        if (images_dir / os.path.basename(urlparse(image_url).path)).is_file():
                            skipped_downloads += 1
                            continue
                    except OSError as e:
                        if e.errno == 36:  # "File name too long" to check if the path is a file
                            skipped_downloads += 1
                            continue
                        else:
                            raise e

                    # Adds URL to batch
                    urls_to_download.append(image_url)

                # Downloads batch
                logging.info(
                    f"In batch {batch_idx} of split {split}, "
                    f"there are {len(urls_to_download)} non-skipped images to download "
                    f"and {skipped_downloads} already-downloaded images"
                )
                results = safe_bulk_download(
                    urls_to_download,
                    dest_folder=images_dir,
                    log_failure=log_failures,
                )
                successful_downloads += sum(1 for r in results if r is not None)
                failed_downloads += sum(1 for r in results if r is None)
        logging.info(f"Successfully downloaded {successful_downloads} {split} images.")
        logging.info(f"Failed downloads: {failed_downloads}")

        # Second pass: opens and yields examples
        example_idx = 0
        invalid_images = 0
        with open(data_path) as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:  # Re-reads file instead of using `results` to simplify logic
                caption, image_url = row
                
                # Skips if the image wasn't downloaded
                filename = os.path.basename(urlparse(image_url).path)
                image_path = images_dir / filename
                if not image_path.is_file():
                    continue
                
                # Tries to open the image
                try:
                    image = Image.open(image_path)
                except Exception as e:
                    if log_failures:
                        logging.warning(f"Failed to open {image_path}: {e}")
                    invalid_images += 1
                    continue

                # Yields the example
                yield example_idx, {"image": image, "caption": caption}
                example_idx += 1
        logging.info(f"Successfully loaded {example_idx} examples from {split} split.")
        logging.info(f"Invalid images (couldn't be opened): {invalid_images}")
