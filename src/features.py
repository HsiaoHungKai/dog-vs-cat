import sys
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

PROJECT_ROOT = "/Users/Repositories/dog-vs-cat"
sys.path.append(PROJECT_ROOT)

from src.dataset import df

train, test = train_test_split(df, test_size=0.2, random_state=42)

# Data augmentation
train_generator = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

train_iterator = train_generator.flow_from_dataframe(
    train, 
    x_col="images",
    y_col="label",
    target_size=(128, 128),
    batch_size=512,
    class_mode="binary"
)

val_generator = ImageDataGenerator(rescale=1.0 / 255)

val_iterator = val_generator.flow_from_dataframe(
    test, 
    x_col="images",
    y_col="label",
    target_size=(128, 128),
    batch_size=512,
    class_mode="binary"
)


next(train_iterator)
next(val_iterator)