def gate(phash_dist, resnet_sim,
         T_hash=10,
         T_high=0.80,
         T_low=0.55):
    """
    Returns one of:
    - 'ACCEPT'
    - 'REJECT'
    - 'AMBIGUOUS'
    """

    if phash_dist > T_hash:
        return "REJECT"

    if resnet_sim >= T_high:
        return "ACCEPT"

    if resnet_sim <= T_low:
        return "REJECT"

    return "AMBIGUOUS"
