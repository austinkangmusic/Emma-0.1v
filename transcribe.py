import pyaudio
import os
import wave
import numpy as np
import torch
# from initialize_whisper import initialize_whisper_model
from silero_vad import (load_silero_vad, read_audio, get_speech_timestamps, save_audio, VADIterator, collect_chunks)# from initialize_whisper import initialize_whisper_model

# Initialize Whisper model
# whisper_model = initialize_whisper_model()

# Load VAD model
model = load_silero_vad(onnx=False)
sampling_rate = 16000
chunk_size = 512
speech_threshold = 0.9

# Function to process audio chunks with VAD
def process_chunk(chunk):
    tensor_chunk = torch.from_numpy(chunk).float()
    speech_prob = model(tensor_chunk, sampling_rate).item()
    return speech_prob

# Function to process full audio and convert segments to time
def process_full_audio(audio_file):
    wav = read_audio(audio_file, sampling_rate=sampling_rate)
    # Get speech timestamps from full audio file
    speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=sampling_rate)
    
    # Print speech segments with human-readable time
    print("Speech segments:")
    for segment in speech_timestamps:
        start_time = segment['start'] / sampling_rate
        end_time = segment['end'] / sampling_rate
        print(f"Start: {start_time:.2f} seconds, End: {end_time:.2f} seconds")
    
    # Merge all speech chunks to one audio
    save_audio('audios/user_audio.wav', collect_chunks(speech_timestamps, wav), sampling_rate=sampling_rate)
    print("Speech segments saved to 'audios/user_audio.wav'")

# Function to record audio from the microphone and save it to a file
def record_audio(file_path, silence_duration=2):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=sampling_rate, input=True, frames_per_buffer=chunk_size)
    frames = []
    audio_buffer = []
    silence_counter = 0
    voice_detected = False
    silence_threshold = sampling_rate / chunk_size * silence_duration

    print("Monitoring voice...")
    try:
        while True:
            data = stream.read(chunk_size, exception_on_overflow=False)  # Avoid exceptions on overflow
            frames.append(data)
            audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0  # Normalize
            
            audio_buffer.extend(audio_data)
            while len(audio_buffer) >= chunk_size:
                chunk = np.array(audio_buffer[:chunk_size])
                audio_buffer = audio_buffer[chunk_size:]

                speech_prob = process_chunk(chunk)
                percentage = speech_prob* 100
                rounded_value = round(percentage, 2)  # Rounds to 2 decimal places (nearest 0.01)

                if speech_prob >= speech_threshold:
                    print(f"Voice detected at {rounded_value}%")

                    voice_detected = True
                    silence_counter = 0
                else:
                    if voice_detected:
                        silence_counter += 1

                        if silence_counter >= silence_threshold:
                            print("Silence detected after voice, stopping recording.")
                            break
            if silence_counter >= silence_threshold:
                break
    except KeyboardInterrupt:
        print("Recording interrupted.")
    finally:
        # Ensure the stream is properly closed
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        # Save recorded audio to a WAV file
        wf = wave.open(file_path, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sampling_rate)
        wf.writeframes(b''.join(frames))
        wf.close()


# Function to transcribe audio using Whisper model
def transcribe_with_whisper(audio_file, whisper_model):
    segments, info = whisper_model.transcribe(audio_file, beam_size=5)
    transcription = ""
    for segment in segments:
        transcription += segment.text + " "
    return transcription.strip()

# Function to manage the whole process
def start(whisper_model, use_recording=True):
    audio_file = "audios/user_audio.wav"
    

        
    # Record new audio
    record_audio(audio_file)


    process_full_audio(audio_file)    # Process full audio

    # Transcribe the recorded or default audio
    user_input = transcribe_with_whisper(audio_file, whisper_model)
    
    print("User: ", user_input)

    # Save transcription to a file
    with open("transcription/user_audio.txt", "w") as file:
        file.write(user_input)

# while True:
    # start(whisper_model)

