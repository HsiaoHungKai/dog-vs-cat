import os
import sys
import numpy as np
import random
from keras.preprocessing.image import load_img
from keras.models import load_model

PROJECT_ROOT = "/Users/Repositories/dog-vs-cat"
sys.path.append(PROJECT_ROOT)
from src.dataset import df

model = load_model(f"{PROJECT_ROOT}/models/dog_vs_cat_model.h5")

# Test the prediction of individual image
image_path = input("Enter image path: ")
img = load_img(image_path, target_size=(128, 128))
img = np.array(img)
img = img / 255.0
img = img.reshape(1, 128, 128, 3)
pred = model.predict(img)
if pred[0] > 0.5:
    print("Dog")
else:
    print("Cat")

# Test the model with a random sample of 100 dogs
success = 0
start = random.randint(0, len(df[df["label"] == "1"]) - 1)
test_size = 1000
for image in df[df["label"] == "1"]["images"][start : start + test_size]:
    img = load_img(image, target_size=(128, 128))
    img = np.array(img)
    img = img / 255.0
    img = img.reshape(1, 128, 128, 3)
    pred = model.predict(img)
    if pred[0] > 0.5:
        success += 1
print(f"Success: {success} / {test_size}")

# Test the model with a random sample of 100 cats
success = 0
start = random.randint(0, len(df[df["label"] == "0"]) - 1)
test_size = 1000
for image in df[df["label"] == "0"]["images"][start : start + test_size]:
    img = load_img(image, target_size=(128, 128))
    img = np.array(img)
    img = img / 255.0
    img = img.reshape(1, 128, 128, 3)
    pred = model.predict(img)
    if pred[0] <= 0.5:
        success += 1
print(f"Success: {success} / {test_size}")
