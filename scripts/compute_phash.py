from pathlib import Path
from typing import Dict
import pickle

from features.phash import compute_phash

COPYDAYS_ROOT = Path("data/raw/copydays")
LANDMARKS_ROOT = Path("data/raw/landmarks")
OUTPUT_PATH = Path("data/processed/phashes.pkl")


def compute_all_phashes() -> Dict[str, int]:
    """
    Compute perceptual hashes for all images in Copydays and Landmarks datasets.

    Returns:
        Dict[str, int]: Mapping from image path (string) to pHash value
    """
    phashes: Dict[str, int] = {}

    # Copydays: original + strong
    for split in ("original", "strong"):
        img_dir = COPYDAYS_ROOT / split
        if not img_dir.exists():
            continue

        for img_path in sorted(img_dir.glob("*.jpg")):
            try:
                phashes[str(img_path)] = compute_phash(img_path)
            except Exception as e:
                print(f"[WARN] Failed to process {img_path}: {e}")

    # Landmarks: recursive (0/, 1/, 2/, ...)
    if LANDMARKS_ROOT.exists():
        for img_path in sorted(LANDMARKS_ROOT.rglob("*.jpg")):
            try:
                phashes[str(img_path)] = compute_phash(img_path)
            except Exception as e:
                print(f"[WARN] Failed to process {img_path}: {e}")

    return phashes


def save_phashes(phashes: Dict[str, int], output_path: Path) -> None:
    """
    Serialize pHashes to disk using pickle.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(phashes, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    phashes = compute_all_phashes()
    save_phashes(phashes, OUTPUT_PATH)
    print(f"Computed pHash for {len(phashes)} images")
    print(f"Saved pHashes to {OUTPUT_PATH}")