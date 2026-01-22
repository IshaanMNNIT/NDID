"""
pHash Computation Script
-----------------------

This script computes perceptual hashes (pHash) for images from
two datasets:

1. Copydays dataset
   - original/
   - strong/

2. Landmarks dataset
   - nested directories (0/, 1/, 2/, ...)

The resulting mapping:
    image_path (str) -> phash (int)

is serialized to disk for later use in near-duplicate image detection.

IMPORTANT:
- Core logic is intentionally simple and unchanged.
- Extra structure and comments are added for clarity, safety, and maintainability.
"""

from pathlib import Path
from typing import Dict
import pickle

# Local import: perceptual hash computation
# This function is assumed to:
#   - accept a Path
#   - return a hashable integer (usually 64-bit)
from features.phash import compute_phash


# -------------------------------------------------------------------
# Dataset root directories
# -------------------------------------------------------------------

# Copydays dataset root
COPYDAYS_ROOT = Path("data/raw/copydays")

# Landmarks dataset root
LANDMARKS_ROOT = Path("data/raw/landmarks")

# Output path where computed hashes will be stored
OUTPUT_PATH = Path("data/processed/phashes.pkl")


# -------------------------------------------------------------------
# Core computation logic
# -------------------------------------------------------------------

def compute_all_phashes() -> Dict[str, int]:
    """
    Walk through all supported datasets and compute pHash values.

    Returns
    -------
    Dict[str, int]
        A dictionary mapping absolute image paths (as strings)
        to their perceptual hash values.

    Design notes:
    - Paths are stored as strings to make the output pickle-safe
      across platforms and Python versions.
    - Errors are handled per-image so one corrupted file does not
      kill the entire preprocessing run.
    - Traversal order is deterministic (sorted paths).
    """

    # Main storage dictionary
    # Key   -> image path as string
    # Value -> perceptual hash (int)
    phashes: Dict[str, int] = {}

    # ---------------------------------------------------------------
    # 1. Process Copydays dataset
    # ---------------------------------------------------------------
    # Copydays has two important splits:
    #   - original : untouched images
    #   - strong   : heavily transformed versions
    #
    # Both are useful for evaluating robustness of perceptual hashing.
    # ---------------------------------------------------------------

    for split in ("original", "strong"):

        # Construct directory path: data/raw/copydays/<split>
        img_dir = COPYDAYS_ROOT / split

        # Defensive check: skip if directory does not exist
        if not img_dir.exists():
            print(f"[INFO] Skipping missing directory: {img_dir}")
            continue

        # Iterate over all .jpg files (non-recursive)
        # Sorting ensures deterministic ordering across runs
        for img_path in sorted(img_dir.glob("*.jpg")):

            try:
                # Compute perceptual hash
                # This is the core operation of the entire script
                phash_value = compute_phash(img_path)

                # Store result using string path as key
                phashes[str(img_path)] = phash_value

            except Exception as e:
                # Catch *any* exception:
                # - corrupted image
                # - unexpected format
                # - library issues
                #
                # We log and continue instead of crashing.
                print(f"[WARN] Failed to process {img_path}: {e}")

    # ---------------------------------------------------------------
    # 2. Process Landmarks dataset
    # ---------------------------------------------------------------
    # Landmarks dataset structure:
    #   data/raw/landmarks/
    #       0/
    #       1/
    #       2/
    #       ...
    #
    # Each subdirectory may contain thousands of images.
    # We traverse recursively.
    # ---------------------------------------------------------------

    if LANDMARKS_ROOT.exists():

        # rglob allows recursive search for .jpg files
        # Again, sorting for determinism
        for img_path in sorted(LANDMARKS_ROOT.rglob("*.jpg")):

            try:
                # Compute perceptual hash
                phash_value = compute_phash(img_path)

                # Store in shared dictionary
                phashes[str(img_path)] = phash_value

            except Exception as e:
                # Same philosophy as above:
                # fail soft, not hard
                print(f"[WARN] Failed to process {img_path}: {e}")

    else:
        print(f"[INFO] Landmarks root not found: {LANDMARKS_ROOT}")

    # Final dictionary containing all computed hashes
    return phashes


# -------------------------------------------------------------------
# Serialization utilities
# -------------------------------------------------------------------

def save_phashes(phashes: Dict[str, int], output_path: Path) -> None:
    """
    Serialize computed pHashes to disk using pickle.

    Parameters
    ----------
    phashes : Dict[str, int]
        Mapping from image paths to perceptual hashes.

    output_path : Path
        Destination file path (e.g., data/processed/phashes.pkl)

    Notes
    -----
    - Uses highest available pickle protocol for efficiency.
    - Ensures parent directories exist before writing.
    """

    # Create parent directories if they do not exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Open file in binary write mode
    with output_path.open("wb") as f:
        pickle.dump(
            phashes,
            f,
            protocol=pickle.HIGHEST_PROTOCOL
        )


# -------------------------------------------------------------------
# Script entry point
# -------------------------------------------------------------------

if __name__ == "__main__":
    """
    Entry point when script is executed directly.

    Typical usage:
        python compute_phashes.py

    This will:
    1. Traverse datasets
    2. Compute perceptual hashes
    3. Save results to disk
    """

    # Step 1: Compute hashes
    phashes = compute_all_phashes()

    # Step 2: Persist hashes to disk
    save_phashes(phashes, OUTPUT_PATH)

    # Step 3: Simple sanity log
    print(f"[DONE] Computed pHash for {len(phashes)} images")
