import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models 
import matplotlib.pyplot as plt
import os

DATA_PATH="processed_data"
MODEL_PATH="models"
EPOCHS = 30
BATCH_SIZE =8

def load_data():
    X_train = np.load(os.path.join(DATA_PATH, "X_train.npy"))
    X_test = np.load(os.path.join(DATA_PATH, "X_test.npy"))
    y_train = np.load(os.path.join(DATA_PATH, "y_train.npy"))
    y_test = np.load(os.path.join(DATA_PATH, "y_test.npy"))
    
    X_train = X_train[...,np.newaxis]
    X_test= X_test[..., np.newaxis]
    
    return X_train, y_train, X_test, y_test

def build_model(input_shape , num_classes):
    model  = models.Sequential()
    
    model.add(layers.Conv2D(32,(3,3), activation = 'relu', input_shape = input_shape))
    model.add(layers.MaxPooling2D((2,2)))
    
    model.add(layers.Conv2D(64, (3,3), activation = 'relu'))
    model.add(layers.MaxPooling2D((2,2)))
    
    model.add(layers.Flatten())
    
    model.add(layers.Dense(64, activation = 'relu'))
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Dense(num_classes, activation = 'softmax'))
    
    model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy',metrics=['accuracy'])
    return model


    