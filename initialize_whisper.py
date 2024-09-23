from faster_whisper import WhisperModel
from cuda_device import get_device  # Import the get_device function

def initialize_whisper_model():
    device = get_device()  # Dynamically select device using cuda_device.py
    model_size = "base.en"
    
    # Initialize Whisper model
    whisper_model = WhisperModel(model_size, device=device, compute_type="float32")
    
    return whisper_model
