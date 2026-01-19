import pickle
from pathlib import Path
import torch

from embedding.resnet_embedder import ResNetEmbedder

ROOTS = [
    Path("data/raw/copydays/original"),
    Path("data/raw/copydays/strong"),
    Path("data/raw/landmarks"),
]

def collect_images():
    imgs = []
    for root in ROOTS:
        if root.exists():
            imgs.extend(root.rglob("*.jpg"))
    return imgs


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = ResNetEmbedder(device=device)

    embeddings = {}
    images = collect_images()

    print(f"Computing ResNet embeddings for {len(images)} images")

    for img_path in images:
        embeddings[str(img_path)] = embedder.embed(img_path).numpy()

    out = Path("data/processed/resnet_embeddings.pkl")
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "wb") as f:
        pickle.dump(embeddings, f)

    print("Saved embeddings to", out)
