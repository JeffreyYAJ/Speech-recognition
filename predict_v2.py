import time
import os
import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf
from scipy.io.wavfile import write

# --- CONFIGURATION ---
MODEL_PATH = "models/camer_digit_model.h5"
SAMPLE_RATE = 22050
DURATION = 1.0
SEUIL_SILENCE = 0.02  

def record_live_audio():
    print("\nParlez (1 sec)...", end="", flush=True)
    # Enregistrement
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()
    print(" Stop.")
    
    audio = audio.flatten()
    
    # --- DEBUG : Sauvegarder pour écouter ce que l'IA entend ---
    write("dernier_enregistrement_debug.wav", SAMPLE_RATE, (audio * 32767).astype(np.int16))
    
    return audio

def preprocess_live_audio(signal):
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal = signal / max_val
    
    if max_val < SEUIL_SILENCE:
        return None  

    mfcc = librosa.feature.mfcc(y=signal, sr=SAMPLE_RATE, n_mfcc=13, n_fft=2048, hop_length=512)
    mfcc = mfcc.T
    mfcc = mfcc[np.newaxis, ..., np.newaxis]
    return mfcc

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(" Model not found.")
        exit()

    print("Loading model")
    model = tf.keras.models.load_model(MODEL_PATH)
    print(" Prêt !")

    while True:
        input("Press enter to speak (q to exit)... ")        
        audio = record_live_audio()
        processed_audio = preprocess_live_audio(audio)
    
        if processed_audio is None:
            print(" Speak louder please !")
            continue
            
        prediction = model.predict(processed_audio, verbose=0)
        chiffre = np.argmax(prediction)
        confiance = np.max(prediction) * 100
        
        print(f" Predicted number : {chiffre} ({confiance:.1f}%)")
        print("-" * 30)