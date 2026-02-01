import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=1.0) # On coupe à 1 seconde max
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    return mfccs


mon_fichier = "mon_enregistrement.wav" 

try:
    filename = librosa.ex('trumpet')
    y, sr = librosa.load(filename)
    
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    print(f"Forme du signal brut : {y.shape}") # (N_samples,)
    print(f"Forme des MFCC (L'image) : {mfccs.shape}") # (13, Time_steps)
    
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.title('MFCC (La "Photo" de ta voix)')
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"Erreur: {e}")