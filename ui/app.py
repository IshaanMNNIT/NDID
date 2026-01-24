# ui/app.py

import sys
from pathlib import Path
import json
import streamlit as st
import matplotlib.pyplot as plt

# --------------------------------------------------
# Ensure project root is on PYTHONPATH (Streamlit quirk)
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --------------------------------------------------
# Load frozen metrics
# --------------------------------------------------
METRICS_PATH = PROJECT_ROOT / "data/processed/metrics.json"

with METRICS_PATH.open() as f:
    METRICS = json.load(f)

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="EigenSoul – NDID Results",
    layout="centered"
)

st.title("🌀 EigenSoul")
st.subheader("Near-Duplicate Image Detection — Final Evaluation Results")

st.markdown(
    """
This dashboard displays **final, frozen evaluation metrics**.  
No models are executed. No thresholds are tuned.  
Metrics were computed offline using dataset-appropriate evaluation protocols.
"""
)

st.divider()

# --------------------------------------------------
# Dataset selector
# --------------------------------------------------
dataset = st.selectbox(
    "Select Dataset",
    ["Copydays", "Airbnb"]
)

key = dataset.lower()
m = METRICS[key]

# --------------------------------------------------
# Metric cards
# --------------------------------------------------
c1, c2, c3 = st.columns(3)

c1.metric("Precision", f"{m['precision']:}")
c2.metric("Recall", f"{m['recall']:}")
c3.metric("F1 Score", f"{m['f1']:}")

st.divider()

# --------------------------------------------------
# Simple bar chart
# --------------------------------------------------
fig, ax = plt.subplots()
ax.bar(
    ["Precision", "Recall", "F1"],
    [m["precision"], m["recall"], m["f1"]],
)
ax.set_ylim(0, 1)
ax.set_ylabel("Score")
ax.set_title(f"{dataset} Performance")

st.pyplot(fig)

st.divider()

# --------------------------------------------------
# Comparison view
# --------------------------------------------------
st.subheader("Copydays vs Airbnb")

labels = ["Precision", "Recall", "F1"]
copy = METRICS["copydays"]
air = METRICS["airbnb"]

fig2, ax2 = plt.subplots()
x = range(len(labels))

ax2.bar(
    [i - 0.2 for i in x],
    [copy[l.lower()] for l in labels],
    width=0.4,
    label="Copydays"
)
ax2.bar(
    [i + 0.2 for i in x],
    [air[l.lower()] for l in labels],
    width=0.4,
    label="Airbnb"
)

ax2.set_xticks(list(x))
ax2.set_xticklabels(labels)
ax2.set_ylim(0, 1)
ax2.legend()
ax2.set_title("Cross-Dataset Generalization")

st.pyplot(fig2)

st.divider()

st.markdown(
    """
### ✅ Summary
- Same pipeline evaluated on **academic (Copydays)** and **real-world (Airbnb)** data
- No retraining between datasets
- Evaluation aligned with production NDID systems
"""
)
