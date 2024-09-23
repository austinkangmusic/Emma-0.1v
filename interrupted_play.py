import threading
import pyaudio
import wave
import numpy as np
import os
import torch
from silero_vad import (load_silero_vad, read_audio, get_speech_timestamps, save_audio, VADIterator, collect_chunks)# from initialize_whisper import initialize_whisper_model
# from initialize_whisper import initialize_whisper_model
import time
from robotic_voice import robotic, apply_vocoder

# Initialize Whisper model
# whisper_model = initialize_whisper_model()

# Load VAD model
vad_model = load_silero_vad(onnx=False)
sampling_rate = 16000
chunk_size = 512
speech_threshold = 0.9

# Global variables
interrupted = False
buffer = b''  # Buffer to store audio data
wf_global = None  # To keep a reference to the wave file object
playback_active = True  # Default value, will be controlled

def process_full_audio(audio_file):
    global interrupted
    wav = read_audio(audio_file, sampling_rate=sampling_rate)
    
    # Get speech timestamps from full audio file
    speech_timestamps = get_speech_timestamps(wav, vad_model, sampling_rate=sampling_rate)
    
    # Print speech segments with human-readable time
    print("Speech segments:")
    if not speech_timestamps:
        print("No speech segments detected.")
        return

    for segment in speech_timestamps:
        start_time = segment['start'] / sampling_rate
        end_time = segment['end'] / sampling_rate
        print(f"Start: {start_time:.2f} seconds, End: {end_time:.2f} seconds")

    # Merge all speech chunks to one audio
    chunks = collect_chunks(speech_timestamps, wav)

    if chunks.numel() == 0:  # Check for empty chunks
        print("No audio chunks to save.")
        return

    # Check if chunks is a single tensor or a list of tensors
    if isinstance(chunks, torch.Tensor):
        audio_data = chunks  # It's a single tensor
    else:
        audio_data = torch.cat(chunks)  # It's a list of tensors

    save_audio('audios/interrupted_user_audio.wav' if interrupted else 'audios/user_audio.wav', audio_data, sampling_rate=sampling_rate)
    print(f"Speech segments saved to '{'audios/interrupted_user_audio.wav' if interrupted else 'audios/user_audio.wav'}'")



def process_chunk(chunk):
    tensor_chunk = torch.from_numpy(chunk).float()
    speech_prob = vad_model(tensor_chunk, sampling_rate).item()
    return speech_prob

def play_audio(file_path, whisper_model):
    global interrupted, buffer, wf_global, playback_active
    playback_active = True

    if not os.path.exists(file_path):
        print(f"Audio file not found: {file_path}")
        return

    wf = wave.open(file_path, 'rb')
    wf_global = wf
    p = pyaudio.PyAudio()
    
    try:
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)
        
        data = wf.readframes(512)
        while data and not interrupted:
            buffer += data
            stream.write(data)
            data = wf.readframes(512)
        
        playback_active = False
        print("AI has stopped talking.")

        if interrupted:
            save_interrupted_file(file_path, whisper_model)

        buffer = b''  # Clear buffer after saving if needed
        
        stream.stop_stream()
    except Exception as e:
        print(f"Error playing audio: {e}")
    finally:
        stream.close()
        wf.close()
        p.terminate()


def save_interrupted_file(original_file_path, whisper_model):
    global wf_global, buffer
    if wf_global is None:
        print("Wave file object is not available.")
        return

    output_file = f'audios/interrupted_{os.path.basename(original_file_path)}'
    with wave.open(output_file, 'wb') as out_wf:
        out_wf.setnchannels(wf_global.getnchannels())
        out_wf.setsampwidth(wf_global.getsampwidth())
        out_wf.setframerate(wf_global.getframerate())
        out_wf.writeframes(buffer)
    print(f"Interrupted audio saved as {output_file}")

def monitor_voice(silence_duration=2):
    global interrupted, playback_active
    interrupted = False
    playback_active = True

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=sampling_rate, input=True, frames_per_buffer=chunk_size)
    frames = []
    silence_counter = 0
    voice_detected = False
    silence_threshold = sampling_rate / chunk_size * silence_duration


    print("Monitoring for voice...")
    try:
        while True:
            data = stream.read(chunk_size)
            frames.append(data)
            audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

            if len(audio_data) >= chunk_size:
                chunk = audio_data[:chunk_size]
                speech_prob = process_chunk(chunk)
                interrupted = interrupted
                percentage = speech_prob* 100
                rounded_value = round(percentage, 2)  # Rounds to 2 decimal places (nearest 0.01)

                if speech_prob >= speech_threshold:
                    voice_detected = True
                    silence_counter = 0
                    print(f"Voice detected at {rounded_value}%")
                    if playback_active:
                        silence_counter = 0
                        print("AI got interrupted...")
                        interrupted = True
                        playback_active = False
                else:
                    if voice_detected:
                        silence_counter += 1

                if voice_detected and silence_counter >= silence_threshold:
                    print("Silence detected after voice, stopping recording.")
                    break
    except KeyboardInterrupt:
        print("Recording interrupted.")
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    temp_audio_file = 'audios/interrupted_user_audio.wav' if interrupted else 'audios/user_audio.wav'
    with wave.open(temp_audio_file, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sampling_rate)
        wf.writeframes(b''.join(frames))
    print(f"Audio saved as {temp_audio_file}")

    process_full_audio(temp_audio_file)

    status = 'True' if interrupted else 'False'
    with open('interrupted_status.txt', 'w') as file:
        file.write(status)

def transcribe_and_save_all(whisper_model, interrupted):
    audio_files = [
        'audios/interrupted_output_0.wav' if interrupted else 'audios/output_0.wav',
        'audios/interrupted_user_audio.wav' if interrupted else 'audios/user_audio.wav'
    ]
    
    output_folder = 'transcription'
    os.makedirs(output_folder, exist_ok=True)
    
    for file_path in audio_files:
        if os.path.exists(file_path):
            segments, _ = whisper_model.transcribe(file_path, beam_size=5)
            transcription = ' '.join(segment.text for segment in segments)
            txt_file_name = os.path.basename(file_path).replace('.wav', '.txt')
            txt_file_path = os.path.join(output_folder, txt_file_name)
            with open(txt_file_path, 'w') as file:
                file.write(transcription)
            print(f"Transcription saved to {txt_file_path}")
        else:
            print(f"Audio file not found: {file_path}")

def play_interruptable_audio(whisper_model):
    global playback_active
    audio_file = 'audios/output_0.wav'
    
    # Start the voice monitoring thread
    voice_thread = threading.Thread(target=monitor_voice)
    voice_thread.start()
    
    time.sleep(1)

    if interrupted:
         apply_vocoder("audios/output_0.wav", "audios/interrupted_output_0.wav", "audios/carrier.wav")
    else:
        apply_vocoder("audios/output_0.wav", "audios/output_0.wav", "audios/carrier.wav")   
    # Play the audio
    play_audio(audio_file, whisper_model)
    
    # Wait for the voice monitoring thread to complete
    voice_thread.join()
    
    transcribe_and_save_all(whisper_model, interrupted)

# while True:
#     play_interruptable_audio(whisper_model)