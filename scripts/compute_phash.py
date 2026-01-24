from pathlib import Path
from typing import Dict
import pickle

from features.phash import compute_phash


PROJECT_ROOT = Path(__file__).resolve().parents[1]

COPYDAYS_ROOT = PROJECT_ROOT / "data/raw/copydays"
LANDMARKS_ROOT = PROJECT_ROOT / "data/raw/landmarks"
AIRBNB_ROOT = PROJECT_ROOT / "data/raw/airbnb"

OUTPUT_PATH = PROJECT_ROOT / "data/processed/phashes.pkl"


def normalize_path(p: Path) -> str:
    return p.resolve().relative_to(PROJECT_ROOT).as_posix()


def compute_all_phashes() -> Dict[str, int]:
    phashes: Dict[str, int] = {}

    # -------- Copydays --------
    for split in ("original", "strong"):
        d = COPYDAYS_ROOT / split
        if not d.exists():
            continue
        for img in sorted(d.glob("*.jpg")):
            phashes[normalize_path(img)] = compute_phash(img)

    # -------- Landmarks --------
    if LANDMARKS_ROOT.exists():
        for img in sorted(LANDMARKS_ROOT.rglob("*.jpg")):
            phashes[normalize_path(img)] = compute_phash(img)

    # -------- Airbnb --------
    if AIRBNB_ROOT.exists():
        for img in sorted(AIRBNB_ROOT.rglob("*.jpg")):
            phashes[normalize_path(img)] = compute_phash(img)

    return phashes


def save_phashes(phashes: Dict[str, int], out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        pickle.dump(phashes, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    ph = compute_all_phashes()
    save_phashes(ph, OUTPUT_PATH)
    print(f"[DONE] pHash computed for {len(ph)} images")
