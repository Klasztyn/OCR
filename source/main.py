import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from source.app import draw_page, analytics_page

st.set_page_config(page_title="OCR")

page = st.sidebar.radio("Page", ["Draw", "Analytics"])

if page == "Draw":
    draw_page()
else:
    analytics_page()