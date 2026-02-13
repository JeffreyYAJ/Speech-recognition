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
    print("\n🎤 Parlez (1 sec)...", end="", flush=True)
    # Enregistrement
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()
    print(" Stop.")
    
    # Aplatir le signal (de 2D à 1D)
    audio = audio.flatten()
    
    # --- DEBUG : Sauvegarder pour écouter ce que l'IA entend ---
    # Multiplié par 32767 pour que ce soit audible sur VLC/Lecteur média
    write("dernier_enregistrement_debug.wav", SAMPLE_RATE, (audio * 32767).astype(np.int16))
    
    return audio

def preprocess_live_audio(signal):
    # 1. NORMALISATION (Le correctif le plus important !)
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal = signal / max_val
    
    # 2. Vérification du silence (Anti-bruit)
    if max_val < SEUIL_SILENCE:
        return None  

    # 3. MFCC (Comme à l'entraînement)
    mfcc = librosa.feature.mfcc(y=signal, sr=SAMPLE_RATE, n_mfcc=13, n_fft=2048, hop_length=512)
    mfcc = mfcc.T
    
    # 4. Reshape pour le CNN (Batch, Temps, MFCC, Canal)
    mfcc = mfcc[np.newaxis, ..., np.newaxis]
    return mfcc

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(" Modèle introuvable.")
        exit()

    print("Chargement du modèle...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print(" Prêt !")

    while True:
        input("Appuie sur Entrée pour parler (q pour quitter)... ")
        
        # 1. Enregistrement
        audio = record_live_audio()
        
        # 2. Traitement
        processed_audio = preprocess_live_audio(audio)
        
        if processed_audio is None:
            print(" Volume trop faible. Parlez plus fort !")
            continue
            
        prediction = model.predict(processed_audio, verbose=0)
        chiffre = np.argmax(prediction)
        confiance = np.max(prediction) * 100
        
        print(f" Je pense que c'est : {chiffre} ({confiance:.1f}%)")
        print("-" * 30)