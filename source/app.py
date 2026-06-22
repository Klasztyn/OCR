import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import json
import os
import source.model as model

def draw_page():
    digit = st.radio("Digit", list(range(10)))
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 1.0)",
        background_color="white",
        stroke_color="black",
        stroke_width=15,
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas"
    )
    if st.button("Add"):
        img = cv2.cvtColor(canvas_result.image_data, cv2.COLOR_RGBA2GRAY)
        img = cv2.resize(img, (28, 28))
        img = 255 - img
        folder = f"data/{digit}/"
        count = len(os.listdir(folder))
        cv2.imwrite(f"{folder}img_{count}.png", img)
    folder = f"data/{digit}/"
    st.write(len(os.listdir(folder)))
    correct = st.number_input("Correct number", 0, 9)
    if st.button("Guess"):
        img = cv2.cvtColor(canvas_result.image_data, cv2.COLOR_RGBA2GRAY)
        img = cv2.resize(img, (28, 28))
        img = 255 - img
        x = img.reshape(1, 28, 28, 1) / 255.0
        m = tf.keras.models.load_model("analytics/model.h5")
        p = m.predict(x)
        pred = np.argmax(p)
        st.write(pred)
        folder = f"data/{int(correct)}/"
        count = len(os.listdir(folder))
        cv2.imwrite(f"{folder}img_{count}.png", img)

def analytics_page():
    if st.button("Train Model"):
        model.train_model()
        st.success("Training done")
    with open("analytics/history.json", "r") as f:
        h = json.load(f)
    x = [d["total_images"] for d in h]
    y = [d["accuracy"] for d in h]
    plt.plot(x, y)
    plt.xlabel("Total Images")
    plt.ylabel("Accuracy")
    st.pyplot(plt)