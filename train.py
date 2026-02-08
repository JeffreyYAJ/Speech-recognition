import numpy as np
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os

DATA_PATH = "processed_data"
MODEL_PATH = "models"
EPOCHS = 30
BATCH_SIZE = 8

def load_data():
    """Load preprocessed training and test data.
    
    Returns:
        Tuple of (X_train, y_train, X_test, y_test) with added channel dimension
    """
    X_train = np.load(os.path.join(DATA_PATH, "X_train.npy"))
    X_test = np.load(os.path.join(DATA_PATH, "X_test.npy"))
    y_train = np.load(os.path.join(DATA_PATH, "y_train.npy"))
    y_test = np.load(os.path.join(DATA_PATH, "y_test.npy"))
    
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    
    return X_train, y_train, X_test, y_test

def build_model(input_shape, num_classes):
    """Build CNN model for audio classification.
    
    Args:
        input_shape: Input shape (height, width, channels)
        num_classes: Number of output classes
        
    Returns:
        Compiled Keras Sequential model
    """
    model = models.Sequential()
    
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Flatten())
    
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def plot_history(history):
    """Plot training history curves for accuracy and loss.
    
    Args:
        history: Keras training history object
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Precision')
    plt.plot(epochs_range, val_acc, label='Validation')
    plt.legend(loc='lower right')
    plt.title('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Loss training')
    plt.plot(epochs_range, val_loss, label='Loss Validation')
    plt.legend(loc='upper right')
    plt.title('Loss')
    
    plt.savefig("results/training_curves.png")
    plt.show()

if __name__ == "__main__":
    print("Loading data...")
    X_train, y_train, X_test, y_test = load_data()
    
    num_classes = len(np.unique(y_train)) 
    input_shape = X_train.shape[1:] 
    
    print(f"Input Shape: {input_shape}")
    print(f"Classes: {num_classes}")

    model = build_model(input_shape, num_classes)
    model.summary() 
    
    print("\n--- Begin training ---")
    history = model.fit(X_train, y_train, epochs=EPOCHS, validation_data=(X_test, y_test), batch_size=BATCH_SIZE)
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    model.save(os.path.join(MODEL_PATH, "camer_digit_model.h5"))

    plot_history(history)
