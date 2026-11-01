"""Standalone Streamlit demo for the Iris classifier (loads model directly, no API needed)."""
from pathlib import Path

import streamlit as st

from src.inference import MODEL_PATH, predict
from src.train import train

st.set_page_config(page_title="Iris Classifier", page_icon="🌸")
st.title("🌸 Iris Species Classifier")
st.caption("A simple MLOps demo project — RandomForest trained on the classic Iris dataset.")

if not Path(MODEL_PATH).exists():
    with st.spinner("No trained model found — training one now..."):
        train()

col1, col2 = st.columns(2)
with col1:
    sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.1)
    sepal_width = st.slider("Sepal width (cm)", 2.0, 4.5, 3.5)
with col2:
    petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 1.4)
    petal_width = st.slider("Petal width (cm)", 0.1, 2.5, 0.2)

if st.button("Predict species", type="primary"):
    result = predict(sepal_length, sepal_width, petal_length, petal_width)
    st.success(f"Predicted species: **{result['predicted_class']}**")
    st.bar_chart(result["probabilities"])
