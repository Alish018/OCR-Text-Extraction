# app.py
"""
Simple Streamlit app to upload one image and extract the token containing '_1_'.
Run: streamlit run app.py
"""

import streamlit as st
from src.text_extraction import extract_target_line_from_image
from PIL import Image
import os

st.set_page_config(page_title="OCR _1_ Extractor", layout="centered")
st.title("OCR Task — Extract the token containing `_1_`")
st.markdown("Upload an image of a label / waybill. The app will try to find a token that includes `_1_`.")

uploaded = st.file_uploader("Choose an image", type=["jpg","jpeg","png","tiff"])
if uploaded:
    # Save temporarily
    tmp_path = "tmp_uploaded.jpg"
    with open(tmp_path, "wb") as f:
        f.write(uploaded.getbuffer())
    st.image(Image.open(tmp_path), caption="Uploaded image", use_column_width=True)

    if st.button("Run OCR"):
        with st.spinner("Running OCR..."):
            result, info = extract_target_line_from_image(tmp_path)
        if result:
            st.success("Extracted: " + result)
            st.write("Debug:", info)
        else:
            st.error("No matching token found. Try a clearer/cropped image.")
            st.write("Debug:", info)

st.sidebar.header("Tips")
st.sidebar.write("""
- If detector fails: crop the part of image with the text and upload that crop.
- For best results: high-resolution, straight images work best.
""")
