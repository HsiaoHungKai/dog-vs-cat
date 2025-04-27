import pandas as pd
import numpy as np
import os
from PIL import Image

input_path = []
labels = []

pet_images_path = "/Users/Repositories/dog-vs-cat-images"
for class_name in os.listdir(pet_images_path):
    class_path = os.path.join(pet_images_path, class_name)
    if not os.path.isdir(class_path):  # Skip non-directory entries
        continue
    for path in os.listdir(class_path):
        file_path = os.path.join(class_path, path)
        if not os.path.isfile(file_path):  # Skip non-file entries
            continue
        if class_name == "Cat":
            labels.append("0")
        else:
            labels.append("1")
        input_path.append(file_path)     

df = pd.DataFrame()
df["images"] = input_path
df["label"] = labels
df = df.sample(frac=1).reset_index(drop=True)

invalid_images = []
for image in df["images"]:
    try:
        img = Image.open(image)
        img.verify()  # Verify that it is an image
    except (IOError, SyntaxError) as e:
        invalid_images.append(image)
invalid_images

for image in invalid_images:
    df = df[df["images"] != image]