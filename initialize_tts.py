import os
import glob
import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from cuda_device import get_device  # Import the get_device function

# Singleton-like behavior
_initialized = False
_model = None
_gpt_cond_latent = None
_speaker_embedding = None

def initialize_tts_model():
    global _initialized, _model, _gpt_cond_latent, _speaker_embedding
    if not _initialized:
        device = get_device()

        print("Loading model...")
        config = XttsConfig()
        config.load_json("XTTS-v2\config.json")
        _model = Xtts.init_from_config(config)
        _model.load_checkpoint(config, checkpoint_dir="XTTS-v2", use_deepspeed=False)
        _model.device

        print("Computing speaker latents...")
        
        # Collect all .wav files from the specified directory
        audio_files = glob.glob("voice\*.wav")
        
        # Compute conditioning latents for all collected audio files
        _gpt_cond_latent, _speaker_embedding = _model.get_conditioning_latents(audio_path=audio_files)

        _initialized = True
    
    return _model, _gpt_cond_latent, _speaker_embedding
