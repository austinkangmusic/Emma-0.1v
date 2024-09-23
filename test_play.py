from initialize_whisper import initialize_whisper_model
from interrupted_play import handle_interrupted_audio


whisper_model = initialize_whisper_model()


handle_interrupted_audio(whisper_model)
