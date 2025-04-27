# Dog vs Cat Classification
This project implements a deep learning pipeline to classify images of dogs and cats using a Convolutional Neural Network (CNN). The project includes data preprocessing, model training, evaluation, and visualization of results.


### Dataset Preparation
The dataset.py script:

- Reads images from the data/pet_images/ folder.
- Assigns labels (0 for Cats, 1 for Dogs).
- Removes invalid or corrupted images.

### Data Augmentation
The features.py script:

- Splits the dataset into training and validation sets.
- Applies data augmentation (e.g., rotation, zoom, flipping) to the training set using ImageDataGenerator.

### Model Training
The train.py script:

- Defines a CNN model using Keras.
- Trains the model on the augmented dataset.
- Saves the trained model (dog_vs_cat_model.h5) and training history (history.pkl) in the models/ folder.

### Prediction
The predict.py script:

- Loads the trained model.
- Predicts the label for a single image provided by the user.
- Evaluates the model on random samples of 1000 dog and 1000 cat images.
#### Accuracy:

- **Dog Images**: About **68.6%** of dog images are predicted correctly.
- **Cat Images**: About **85.7%** of cat images are predicted correctly.

### Visualization
The plots.py script:

- Displays sample images of dogs, cats, and augmented images.
- Plots training and validation accuracy and loss.

### Dataset

- The dataset used for this project is too large to be stored in the repository. You can download it from the following link:

[Download Dog vs Cat Dataset](https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip)


## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── README.md          <- The top-level README for developers using this project
├── data
│   ├── external       <- Data from third party sources
│   ├── interim        <- Intermediate data that has been transformed
│   ├── processed      <- The final, canonical data sets for modeling
│   └── raw            <- The original, immutable data dump
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
└── src                         <- Source code for this project
    │
    ├── __init__.py             <- Makes src a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    │    
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    ├── plots.py                <- Code to create visualizations 
    │
    └── services                <- Service classes to connect with external platforms, tools, or APIs
        └── __init__.py 
```

--------