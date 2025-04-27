import streamlit as st
from PIL import Image
import numpy as np
from enhancer.enhancer import Enhancer

st.header('Image Enhancer App')
st.divider()

image_path = st.file_uploader("Choose file: ", type=['.png', '.jpg', '.jpeg'])

# app settings
st.sidebar.header("App Settings:")
method = st.sidebar.selectbox("Method:", ["gfpgan", "codeformer", "RestoreFormer"])
background_enhancement = st.sidebar.checkbox("Background enhancement", value=False)
upscale = st.sidebar.selectbox("Upscale enhancement:", [None, 2, 4])
picture_width = st.sidebar.slider('Picture Width', min_value=100, max_value=500)

if image_path is not None:
    image = Image.open(image_path)
    image_array = np.array(image)

    # Display original image
    col1, col2 = st.columns(2)
    with col1:
        st.header("Input Image")
        st.image(image, width=picture_width)

    # Create enhancer
    enhancer = Enhancer(method=method, background_enhancement=background_enhancement, upscale=upscale)
    restored_image = enhancer.enhance(image_array)

    # Convert the result to a PIL Image
    final_image = Image.fromarray(restored_image)

    with col2:
        st.header("Enhanced Image")
        st.image(final_image, width=picture_width)
