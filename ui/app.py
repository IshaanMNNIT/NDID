# ui/app.py

import sys
import os
from pathlib import Path
import streamlit as st
import numpy as np
from PIL import Image
import torch

# --------------------------------------------------
# Fix Streamlit import path
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --------------------------------------------------
# Imports from project
# --------------------------------------------------
from embedding.resnet_embedder import ResNetEmbedder
from embedding.clip_embedder import CLIPEmbedder
from features.phash import compute_phash, hamming_distance
from index.retrieve import retrieve
from evaluation.gating import gate
from evaluation.decision import decide

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(page_title="EigenSoul NDID", layout="wide")
st.title("🌀 EigenSoul — Near Duplicate Image Detection")

# --------------------------------------------------
# Load models (cached)
# --------------------------------------------------
@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return ResNetEmbedder(device=device), CLIPEmbedder(device=device)

resnet_model, clip_model = load_models()

def cosine(a, b):
    return float(np.dot(a, b))  # vectors already L2-normalized

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
mode = st.sidebar.radio(
    "Mode",
    ["Search Index", "Compare Two Images"]
)

# ==================================================
# MODE 1 — SEARCH INDEX
# ==================================================
if mode == "Search Index":
    st.header("🔍 Search for Near-Duplicates")

    uploaded = st.file_uploader(
        "Upload Query Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Query Image", width=300)

        tmp = Path("ui/_query.jpg")
        img.save(tmp)

        if st.button("Find Matches"):
            with st.spinner("Processing..."):
                # ---- Feature extraction ----
                q_ph = compute_phash(tmp)
                q_rs = resnet_model.embed(tmp).numpy()

                # ---- Retrieval ----
                results = retrieve(q_rs, k=10)

            st.subheader("Results")

            for path, resnet_sim in results:
                if not os.path.exists(path):
                    continue

                # ---- Pairwise signals ----
                p_ph = compute_phash(path)
                ph_dist = hamming_distance(q_ph, p_ph)

                g = gate(ph_dist, resnet_sim)

                # ---- CLIP only if ambiguous ----
                clip_sim = -1.0
                if g == "AMBIGUOUS":
                    q_cl = clip_model.embed(tmp).numpy()
                    p_cl = clip_model.embed(path).numpy()
                    clip_sim = cosine(q_cl, p_cl)

                # ---- FINAL decision (ALWAYS logistic regression) ----
                decision = decide(ph_dist, resnet_sim, clip_sim)

                label = "MATCH" if decision == 1 else "NO MATCH"
                color = "green" if decision == 1 else "red"

                with st.container():
                    c1, c2, c3 = st.columns([1, 2, 1])
                    c1.image(path, width=150)
                    c2.markdown(f"**Path:** `{path}`")
                    c2.markdown(
                        f"""
                        - pHash distance: `{ph_dist}`
                        - ResNet similarity: `{resnet_sim:.3f}`
                        - CLIP similarity: `{clip_sim if clip_sim >= 0 else 'N/A'}`
                        - Gate: `{g}`
                        """
                    )
                    c3.markdown(f"## :{color}[{label}]")
                    st.divider()

# ==================================================
# MODE 2 — COMPARE TWO IMAGES
# ==================================================
else:
    st.header("🆚 Compare Two Images")

    c1, c2 = st.columns(2)
    f1 = c1.file_uploader("Image A", type=["jpg", "jpeg", "png"], key="a")
    f2 = c2.file_uploader("Image B", type=["jpg", "jpeg", "png"], key="b")

    if f1 and f2:
        img1 = Image.open(f1).convert("RGB")
        img2 = Image.open(f2).convert("RGB")

        c1.image(img1, caption="Image A", use_column_width=True)
        c2.image(img2, caption="Image B", use_column_width=True)

        if st.button("Compare"):
            p1 = Path("ui/_a.jpg")
            p2 = Path("ui/_b.jpg")
            img1.save(p1)
            img2.save(p2)

            # ---- Signals ----
            ph1 = compute_phash(p1)
            ph2 = compute_phash(p2)
            ph_dist = hamming_distance(ph1, ph2)

            rs1 = resnet_model.embed(p1).numpy()
            rs2 = resnet_model.embed(p2).numpy()
            resnet_sim = cosine(rs1, rs2)

            g = gate(ph_dist, resnet_sim)

            clip_sim = -1.0
            if g == "AMBIGUOUS":
                cl1 = clip_model.embed(p1).numpy()
                cl2 = clip_model.embed(p2).numpy()
                clip_sim = cosine(cl1, cl2)
                st.metric("CLIP Similarity", f"{clip_sim:.4f}")

            # ---- FINAL decision ----
            final = decide(ph_dist, resnet_sim, clip_sim)
            verdict = "DUPLICATE" if final == 1 else "DIFFERENT"

            st.success(f"Final Verdict: **{verdict}**")
            st.markdown(
                f"""
                **Signals**
                - pHash distance: `{ph_dist}`
                - ResNet similarity: `{resnet_sim:.4f}`
                - Gate: `{g}`
                """
            )
            if clip_sim >= 0:
                st.markdown(f"- CLIP similarity: `{clip_sim:.4f}`")