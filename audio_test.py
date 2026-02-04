import librosa
import librosa.display
import numpy as np
import pytest

def extract_features(file_path):
    """Extract MFCC features from audio file."""
    y, sr = librosa.load(file_path, duration=1.0)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return mfccs


def test_extract_features_shape():
    """Test that extract_features returns correct MFCC shape."""
    filename = librosa.ex('trumpet')
    mfccs = extract_features(filename)
    
    # MFCC output should have shape (n_mfcc, time_steps)
    assert mfccs.shape[0] == 13, "MFCC should have 13 coefficients"
    assert len(mfccs.shape) == 2, "MFCC should be 2D array"


def test_extract_features_not_empty():
    """Test that MFCC extraction produces non-empty output."""
    filename = librosa.ex('trumpet')
    mfccs = extract_features(filename)
    
    assert mfccs.size > 0, "MFCC features should not be empty"
    assert not np.isnan(mfccs).any(), "MFCC features should not contain NaN values"