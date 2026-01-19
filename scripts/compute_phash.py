from pathlib import Path
from features.phash import compute_phash
import pickle

COPYDAYS_ROOT = Path("data/raw/copydays")
LANDMARKS_ROOT = Path("data/raw/landmarks")

def compute_all_phashes():
    phashes = {}

    # Copydays: original + strong
    for split in ["original", "strong"]:
        img_dir = COPYDAYS_ROOT / split
        for img_path in img_dir.glob("*.jpg"):
            phashes[str(img_path)] = compute_phash(img_path)

    # Landmarks: recursive (0/,1/,2/,...)
    if LANDMARKS_ROOT.exists():
        for img_path in LANDMARKS_ROOT.rglob("*.jpg"):
            phashes[str(img_path)] = compute_phash(img_path)

    return phashes


if __name__ == "__main__":
    phashes = compute_all_phashes()

    out_path = Path("data/processed/phashes.pkl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "wb") as f:
        pickle.dump(phashes, f)

    print(f"Computed pHash for {len(phashes)} images")
