import os
import shutil
import requests
import tempfile
import os.path as osp
from tqdm import tqdm
from utils.logger import get_logger as logger


def check_disk_space(path, required_size_mb):
    """Check if there's enough disk space"""
    free_space = shutil.disk_usage(path).free
    required_space = required_size_mb * 1024 * 1024  # Convert MB to bytes
    return free_space >= required_space


def download_file(url, local_path, chunk_size=8192):
    """Download a file with progress bar and proper error handling"""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        with (
            open(local_path, "wb") as f,
            tqdm(
                desc=osp.basename(local_path),
                total=total_size,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar,
        ):
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    size = f.write(chunk)
                    pbar.update(size)
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading {url}: {str(e)}")
        return False


def download_dataset(dataset_name, cache_dir):
    """Download dataset files manually"""
    logger.info(f"Attempting to download dataset {dataset_name} manually...")

    # Create a temporary directory for downloads
    with tempfile.TemporaryDirectory() as temp_dir:
        # Download the dataset files
        base_url = f"https://huggingface.co/datasets/{dataset_name}/resolve/main"
        files_to_download = ["README.md", "data/train-00000-of-00001.parquet"]

        for file in files_to_download:
            url = f"{base_url}/{file}"
            local_path = osp.join(temp_dir, file)

            # Create directory if it doesn't exist
            os.makedirs(osp.dirname(local_path), exist_ok=True)

            logger.info(f"Downloading {file}...")
            if download_file(url, local_path):
                # Move to cache directory
                cache_path = osp.join(cache_dir, dataset_name.replace("/", "--"), file)
                os.makedirs(osp.dirname(cache_path), exist_ok=True)
                shutil.move(local_path, cache_path)
                logger.info(f"Successfully downloaded {file}")
            else:
                raise Exception(f"Failed to download {file}")
