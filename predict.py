import time
import os
import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf 
from recorder import record

MODEL_PATH = "models/camer_digit_model.h5"
SAMPLE_RATE = 22050
DURATION = 1.0
SAMPLE_PER_TRACK = SAMPLE_RATE * DURATION

def preprocess_live_audio(signal):
    length = len(signal)
    target_len = int(SAMPLE_PER_TRACK)
    
    if length > target_len:
        signal = signal[:target_len]
    elif length < target_len:
        pad_width = target_len - length 
        signal = np.pad(signal, (0, pad_width), mode = 'constant')
        
    mfcc = librosa.feature.mfcc(y=signal, sr = SAMPLE_RATE, n_mfcc=13, n_fft =2048, hop_length = 512)
    mfcc = mfcc.T
    
    mfcc = mfcc[np.newaxis,..., np.newaxis]
    
    return mfcc

def predict(model):
    audio = record()
    process_audio = preprocess_live_audio(audio)
    prediction  = model.predict(process_audio, verbose = 0)
    
    predicted = np.argmax(prediction)
    accuracy = np.max(predicted) * 100
    
    return predicted, accuracy, prediction[0]

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print("Model not existing\n run training first")
        exit()
    
    model = tf.keras.models.load_model(MODEL_PATH)
    
    print("Press Enter to start recording")
    
    while True:
        user = input(">>")
        if user.lower() == 'q':
            break
        
        digit, accuracy, score = predict(model)
        print(f"NUmber is {digit}")
        print(f"Accuracy = {accuracy}")
        
        print("Détail des probabilités :")
        for i, score in enumerate(score):
            if score > 0.01:
                print(f"   Chiffre {i} : {score*100:.1f}%")