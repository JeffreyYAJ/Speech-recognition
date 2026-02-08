import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split

DATASET_PATH = "datasets"
OUTPUT_PATH = 'processed_data'
SAMPLE_RATE = 22050
DURATION = 1.0
SAMPLE_PER_TRACK = SAMPLE_RATE * DURATION

def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512):
    """Extract MFCC features from audio dataset and save processed data.
    
    Args:
        dataset_path: Path to the audio dataset directory
        json_path: Path for output JSON (for future use)
        n_mfcc: Number of MFCC coefficients
        n_fft: FFT window size
        hop_length: Number of samples between successive frames
    """
    mapping = []
    X = []
    y = []
    
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        if dirpath != dataset_path:
            label = dirpath.split("/")[-1]
            
            if os.name == 'nt':
                label = dirpath.split("\\")[-1]
                
            print("Data processing")
            mapping.append(label)
            
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                
                # loading audio
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                
                length = len(signal)
                target_length = int(SAMPLE_PER_TRACK)
                
                if length > target_length:
                    signal = signal[:target_length]
                elif length < target_length:
                    pad_width = target_length - length
                    signal = np.pad(signal, (0, pad_width), mode='constant')
                
                mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
                mfcc = mfcc.T
                
                X.append(mfcc.tolist())
                y.append(int(label))
    X = np.array(X)
    y = np.array(y)
    
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    np.save(os.path.join(OUTPUT_PATH, "X_train.npy"), X_train)
    np.save(os.path.join(OUTPUT_PATH, "X_test.npy"), X_test)
    np.save(os.path.join(OUTPUT_PATH, "y_train.npy"), y_train)
    np.save(os.path.join(OUTPUT_PATH, "y_test.npy"), y_test)
    
    print("\n--- Done Processing ---")
    print(f"Size X_train : {X_train.shape}") 
    print(f"Size X_test  : {X_test.shape}")

if __name__ == "__main__":
    save_mfcc(DATASET_PATH, OUTPUT_PATH)