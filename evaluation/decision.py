import pickle
import numpy as np

with open("data/processed/decider.pkl", "rb") as f:
    clf = pickle.load(f)

def decide(phash_dist, resnet_sim, clip_sim):
    x = np.array([[phash_dist, resnet_sim, clip_sim]])
    return int(clf.predict(x)[0])
