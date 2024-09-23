import torch
import pyaudio
import numpy as np
from silero_vad import load_silero_vad

def live_speech_detection(sampling_rate=16000, chunk_size=512, speech_threshold=0.8):
    """
    Function to perform live speech detection using Voice Activity Detection (VAD).
    
    Parameters:
        sampling_rate (int): The sampling rate of the audio.
        chunk_size (int): The number of audio frames per buffer.
        speech_threshold (float): The threshold probability to consider as speech.
    """
    torch.set_num_threads(1)

    # Load VAD model
    model = load_silero_vad(onnx=False)

    # Initialize PyAudio
    p = pyaudio.PyAudio()
    
    def process_chunk(chunk):
        # Convert numpy array to PyTorch tensor
        tensor_chunk = torch.from_numpy(chunk).float()
        
        # Run VAD
        speech_prob = model(tensor_chunk, sampling_rate).item()
        return speech_prob

    def callback(in_data, frame_count, time_info, status):
        global audio_buffer
        
        try:
            # Convert audio buffer to numpy array
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            audio_buffer.extend(audio_data)
            
            # Process chunks of CHUNK_SIZE samples
            while len(audio_buffer) >= chunk_size:
                chunk = np.array(audio_buffer[:chunk_size])
                audio_buffer = audio_buffer[chunk_size:]
                
                # Run VAD
                speech_prob = process_chunk(chunk)
                
                if speech_prob >= speech_threshold:
                    print("Speech detected")
                else:
                    print(f"Speech probability: {speech_prob}")

        except Exception as e:
            print(f"Error in callback: {e}")
        
        return (in_data, pyaudio.paContinue)

    # Create audio buffer
    global audio_buffer
    audio_buffer = []

    # Open audio stream
    stream = None

    try:
        stream = p.open(format=pyaudio.paFloat32,
                        channels=1,
                        rate=sampling_rate,
                        input=True,
                        frames_per_buffer=chunk_size,
                        stream_callback=callback)

        # Start the stream
        stream.start_stream()
        print("Streaming... Press Ctrl+C to stop.")

        try:
            while stream.is_active():
                pass  # Keep the stream open
        except KeyboardInterrupt:
            print("Stopping...")

    finally:
        # Ensure resources are cleaned up
        if stream is not None:
            stream.stop_stream()
            stream.close()
        p.terminate()

# Example usage
live_speech_detection(sampling_rate=16000, chunk_size=512, speech_threshold=0.8)
