import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import json
import os

def train_model():
    X = []
    y = []
    for i in range(10):
        folder = f"data/{i}/"
        files = os.listdir(folder)
        for f in files:
            img = cv2.imread(os.path.join(folder, f), cv2.IMREAD_GRAYSCALE)
            X.append(img)
            y.append(i)
    X = np.array(X)
    y = np.array(y)
    X = X.reshape(-1, 28, 28, 1) / 255.0
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=5, batch_size=10)
    loss, acc = model.evaluate(X, y)
    with open("analytics/history.json", "r") as f:
        h = json.load(f)
    h.append({"total_images": len(X), "accuracy": float(acc)})
    with open("analytics/history.json", "w") as f:
        json.dump(h, f)
    model.save("analytics/model.h5")