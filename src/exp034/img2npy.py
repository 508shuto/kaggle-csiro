import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import tyro
from PIL import Image
from tqdm import tqdm


def process_one_image(file_path: Path, output_dir: Path, image_size: int) -> tuple[Path, bool]:
    """Process a single image and save as numpy array.

    Args:
        file_path: Path to input image file
        output_dir: Output directory for numpy files
        image_size: Target image size (width, height)

    Returns:
        Tuple of (file_path, success_flag)
    """
    try:
        with Image.open(file_path) as image:
            image = image.resize((image_size, image_size))
            image_array = np.array(image).astype(np.float32)
        output_path = output_dir / (file_path.stem + ".npy")
        np.save(output_path, image_array)
        return (file_path, True)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return (file_path, False)


def main(
    input_dir: Path = Path("./input/train"),
    output_dir: Path = Path("./output/exp000/train"),
    image_size: int = 512,
    num_workers: int | None = None,
):
    """Convert images to numpy arrays with parallel processing.

    Args:
        input_dir: Directory containing input images
        output_dir: Output directory for numpy files
        image_size: Target image size (width, height)
        num_workers: Number of parallel workers. If None, uses min(4, cpu_count())
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all image files once
    image_files = sorted(input_dir.glob("*.jpg"))
    if not image_files:
        print(f"No .jpg files found in {input_dir}")
        return

    # Determine number of workers
    if num_workers is None:
        num_workers = min(4, os.cpu_count() or 1)

    # Process images in parallel
    success_count = 0
    fail_count = 0
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_one_image, file_path, output_dir, image_size) for file_path in image_files]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Converting images"):
            _, success = future.result()
            if success:
                success_count += 1
            else:
                fail_count += 1

    print(f"Completed: {success_count} succeeded, {fail_count} failed")


if __name__ == "__main__":
    tyro.cli(main)
