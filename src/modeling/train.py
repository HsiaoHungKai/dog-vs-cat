import sys
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import pickle
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = "/Users/Repositories/dog-vs-cat"
sys.path.append(PROJECT_ROOT)

from src.features import train_iterator, val_iterator

model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(train_iterator, epochs=10, validation_data=val_iterator)
with open(f"{PROJECT_ROOT}/models/history.pkl", "wb") as f:
    pickle.dump(history.history, f)

# model.save(f"{PROJECT_ROOT}/models/dog_vs_cat_model.h5")
# from keras.models import load_model
# model = load_model(f"{PROJECT_ROOT}/models/dog_vs_cat_model.h5")