import sounddevice as sd
from scipy.io.wavfile import write
import os
import time
import numpy as np

OUTPUT_FOLDER = "datasets"
SAMPLE_RATE = 22050
DURATION = 1.0 
CHANNELS = 1

def create_folder():
    """Create dataset folder structure for recording audio samples."""
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        
    for i in range(10):
        path = os.path.join(OUTPUT_FOLDER, str(i))
        if not os.path.exists(path):
            os.makedirs(path)
            print("Dataset folder created")
            
def record_audio(label):
    """Record audio sample for a given label.
    
    Args:
        label: Label for the audio sample (0-9)
    """
    folder_path = os.path.join(OUTPUT_FOLDER, str(label))
    existing_files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]
    
    index = len(existing_files)
    filename = os.path.join(folder_path, f"{label}_{index}.wav")
    
    print(f"\nTime to say : '{label}' in 1 second")
    print("3...")
    time.sleep(0.5)
    print("2...")
    time.sleep(0.5)
    print("1...")
    time.sleep(0.5)
    print("Recording (Speak now)")
    
    audio_data = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS)
    sd.wait()
    
    print("Done.")
    
    write(filename, SAMPLE_RATE, audio_data)
    print(f"Recording saved as {filename}")

def record():
    print("\nSpeak now...", end="", flush=True)
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()
    print(" Stop.")
    return audio.flatten()

if __name__ == "__main__":
    print("--- Sound recorder ---")
    create_folder()
    
    print("\nINSTRUCTIONS :")
    print(f"1. Every record is {DURATION} seconds long.")
    print("2. Speak clearly after the timer.")
    print("3. Try to change tone variation and speed")
    
    while True:
        user_input = input("\nWhat digit do you want to record (0-9) ? ('q' to quit) : ")
        
        if user_input.lower() == 'q':
            print("Closing recorder.")
            break
        
        if user_input.isdigit() and 0 <= int(user_input) <= 9:
            record_audio(user_input)
        else:
            print("Enter a number between 0 and 9.")