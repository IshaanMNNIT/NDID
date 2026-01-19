import numpy as np
from evaluation.gating import gate

def cosine(a, b):
    return float(np.dot(a, b))


def decide(phash_dist, resnet_sim, clip_sim,
           T_hash=10,
           T_high=0.80,
           T_low=0.55,
           T_clip=0.85):
    """
    Final decision after gating + optional CLIP.
    Returns:
      - 1 (duplicate)
      - 0 (not duplicate)
    """

    g = gate(phash_dist, resnet_sim, T_hash, T_high, T_low)

    if g == "ACCEPT":
        return 1

    if g == "REJECT":
        return 0

    # AMBIGUOUS → CLIP decides
    return int(clip_sim >= T_clip)
