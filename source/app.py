import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas

page = st.sidebar.radio("Go to", ["Draw", "Analytics"])

canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    background_color="#eee",
    stroke_width=10,
    height=500,
    width=500,
    drawing_mode="freedraw",
    key="canvas",
)