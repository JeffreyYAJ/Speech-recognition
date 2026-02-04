import librosa
import numpy as np


def load_audio(file_path, duration=1.0, sr=22050):
    """Load audio file and return waveform and sample rate.
    
    Args:
        file_path: Path to audio file
        duration: Maximum duration in seconds
        sr: Sample rate
        
    Returns:
        y: Audio waveform
        sr: Sample rate
    """
    y, sr = librosa.load(file_path, duration=duration, sr=sr)
    return y, sr


if __name__ == "__main__":
    print("Speech Recognition Project - African Accent Bias")
