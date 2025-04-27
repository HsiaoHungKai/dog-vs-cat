import sys
import matplotlib.pyplot as plt
import random
import numpy as np
import pickle
from keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator

PROJECT_ROOT = "/Users/Repositories/dog-vs-cat"
sys.path.append(PROJECT_ROOT)

from src.dataset import df
from src.features import train_iterator

# Display 25 images of Dogs
plt.figure(figsize=(25, 25))
dogs = df[df["label"] == "1"]["images"]
start = random.randint(0, len(dogs))
dog_images = dogs[start : start + 25]
for i, image in enumerate(dog_images):
    plt.subplot(5, 5, i + 1)
    img = load_img(image)
    img = np.array(img)
    plt.imshow(img)
    plt.title("Dog")
    plt.axis("off")
plt.show()

# Display 25 images of Cats
cats = df[df["label"] == "0"]["images"]
start = random.randint(0, len(cats))
cat_images = cats[start : start + 25]
for i, image in enumerate(cat_images):
    plt.subplot(5, 5, i + 1)
    img = load_img(image)
    img = np.array(img)
    plt.imshow(img)
    plt.title("Cat")
    plt.axis("off")
plt.show()

# Display 25 augmented images
images, label = next(train_iterator)
start = random.randint(0, len(images))
images = images[start : start + 25]
label = label[start : start + 25]
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.imshow(images[i])
    plt.title("Dog" if label[i] == 1 else "Cat")
    plt.axis("off")
plt.show()

# Display accuracy
with open(f"{PROJECT_ROOT}/models/history.pkl", "rb") as f:
    history = pickle.load(f)
acc = history["accuracy"]
val_acc = history["val_accuracy"]
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, "b", label="Training Accuracy")
plt.plot(epochs, val_acc, "r", label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.figure()

# Display loss
loss = history["loss"]
val_loss = history["val_loss"]
plt.plot(epochs, loss, "b", label="Training Loss")
plt.plot(epochs, val_loss, "r", label="Validation Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.show()