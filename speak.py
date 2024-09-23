import soundfile as sf
import numpy as np
import os

# Path to the directory containing .aif files
input_directory = r"D:\Private Server\Projects\Emma\voice"

for filename in os.listdir(input_directory):
    if filename.endswith(".mp3"):
        aif_file = os.path.join(input_directory, filename)
        wav_filename = filename.replace(".mp3", ".wav")
        wav_file = os.path.join(input_directory, wav_filename)
        
        # Read .aif file
        data, samplerate = sf.read(aif_file, dtype='float32')
        
        # Save as .wav file
        sf.write(wav_file, data, samplerate)
        print(f"Converted {filename} to {wav_filename}")

print("Conversion complete.")
