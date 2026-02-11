# Speech-recognition: Dialect Digit Recognition (Low-Resource Speech AI)

This project is an experimental Speech-to-Intent system designed to identify spoken digits (0-9) while specifically addressing **linguistic biases** found in Cameroonian accents and dialects. 

Standard Speech-to-Text models (trained on European or American datasets) often fail to accurately recognize African accents. This project serves as a Proof of Concept (PoC) for building more inclusive AI.

##  Project Overview

The system uses **Mel-Frequency Cepstral Coefficients (MFCCs)** to convert audio signals into visual representations (spectrograms) and a **Convolutional Neural Network (CNN)** to classify them.

### Key Features:
- **Custom Dataset**: Recorded audio samples featuring neutral and Cameroonian-accented speech.
- **MFCC Feature Extraction**: Captures the unique "spectral signature" of the speaker's voice.
- **CNN Architecture**: A deep learning model optimized for small-scale audio classification.
- **Real-time Prediction**: Live microphone testing with confidence scoring.

---

##  Project Structure

```text
.
├── dataset/                # Raw .wav recordings organized by digit (0-9)
├── processed_data/         # Extracted MFCC features (NumPy format)
├── models/                 # Trained CNN model (.h5)
├── results/                # Accuracy/Loss curves and bias analysis
├── recorder.py             # Data collection tool
├── preprocessing.py        # Signal processing (WAV -> MFCC)
├── train.py                # CNN training script
├── predict.py              # Real-time microphone inference
└── requirements.txt        # Project dependencies
```
## Installation & Setup

- Clone the repository:
```bash
git clone [https://github.com/your-username/cameroon-speech-recognition.git](https://github.com/your-username/cameroon-speech-recognition.git)
cd Speech-recognition
```
- Install Dependencies:
```bash
pip install -r requirements.txt
```
- Data Collection:
Use the built-in recorder to build your local dataset:
```bash
python recorder.py
```
- Preprocessing & Training:

```bash
python preprocessing.py
python train.py
```
- Live Testing:

```bash
python predict.py
```
## Addressing Bias
The core of this research focuses on the Confidence Gap.
By comparing the model's output on "Standard French/English" vs. "Cameroonian Accented" speech, we visualize how intonation and phonetics affect AI decision-making.

## Tech Stack
Language: Python 3.12
Signal Processing: Librosa, SciPy
Deep Learning: TensorFlow / Keras
Numerical Computing: NumPy, Scikit-learn
Audio I/O: SoundDevice

