
import warnings
import os
import torch
import pyaudio
import wave
from faster_whisper import WhisperModel
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import soundfile as sf
import chat_utils

# ANSI escape codes for colors
PINK = '\033[95m'
CYAN = '\033[96m'
RESET_COLOR = '\033[0m'

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*torch.load.*")
warnings.filterwarnings("ignore", message=".*attention mask is not set.*")
torch.cuda.is_available()

# Set device for models
device = "cuda" 

# Set up the faster-whisper model
model_size = "tiny.en"
whisper_model = WhisperModel(model_size, device=device, compute_type="float32")

# Initialize chat_utils models
chat_llm, utility_llm, embedding_llm = chat_utils.initialize()

# Function to play audio using PyAudio
def play_audio(file_path):
    wf = wave.open(file_path, 'rb')
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)
    data = wf.readframes(1024)
    while data:
        stream.write(data)
        data = wf.readframes(1024)
    stream.stop_stream()
    stream.close()
    p.terminate()

# Model and device setup
output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)

# Load XTTS configuration
xtts_config = XttsConfig()
xtts_config.load_json(r"D:\Private Server\Projects\friday\XTTS-v2\config.json")

# Initialize XTTS model
xtts_model = Xtts.init_from_config(xtts_config)
xtts_model.load_checkpoint(xtts_config, checkpoint_dir=r"D:\Private Server\Projects\friday\XTTS-v2", eval=True)
xtts_model.to(device)  # Move model to the appropriate device

# Global cache for the cloned voice model and speaker embedding
cloned_voice_model = None
cloned_speaker_embedding = None

# Function to synthesize speech using XTTS
def process_and_play(prompt, audio_file_pth):
    global cloned_voice_model, cloned_speaker_embedding
    
    # Load XTTS model only once if it's not already loaded
    if cloned_voice_model is None:
        cloned_voice_model = xtts_model
    
    # Load speaker embedding only once if it's not already loaded
    if cloned_speaker_embedding is None:
        cloned_speaker_embedding = audio_file_pth

    tts_model = cloned_voice_model
    speaker_embedding = cloned_speaker_embedding

    try:
        # Clean and split prompt if necessary
        clean_prompt = ' '.join(prompt.split())
        chunks = split_text(clean_prompt)  # Assuming split_text is defined
        for i, chunk in enumerate(chunks):
            outputs = tts_model.synthesize(
                chunk,
                xtts_config,
                speaker_wav=speaker_embedding,  # Use cached speaker embedding
                gpt_cond_len=24,
                temperature=0.6,
                language='en',
                speed=1.2
            )
            synthesized_audio = outputs['wav']
            src_path = f'{output_dir}/output_{i}.wav'
            sample_rate = xtts_config.audio.sample_rate
            sf.write(src_path, synthesized_audio, sample_rate)
            play_audio(src_path)
    except Exception as e:
        print(f"Error during audio generation: {e}")

def split_text(text, max_tokens=400):
    tokens = text.split()  # Simple tokenization
    return [' '.join(tokens[i:i + max_tokens]) for i in range(0, len(tokens), max_tokens)]

def trim_text(text, max_length=400):
    return text[:max_length]

def chatgpt_streamed(user_input, system_message, conversation_history):
    """
    Function to stream responses from the chat model and handle long responses.
    """
    messages = [{"role": "system", "content": system_message}] + conversation_history + [{"role": "user", "content": user_input}]
    
    full_response = ""
    chunks = []

    response_generator = chat_llm.stream(messages)
    for chunk in response_generator:
        delta_content = chunk.content
        if delta_content is not None:
            chunks.append(delta_content)
            print(delta_content, end="", flush=True)  # Print chunk as it arrives
    
    print('\n')

    full_response = ''.join(chunks)
    return full_response.strip()

def user_chatbot_conversation():
    conversation_history = []
    system_message = "You are Emma, a helpful AI assistant. Keep it casual and concise. Focus on short, clear responses."
    
    while True:
        user_input = input(CYAN + "You:\n" + RESET_COLOR)
        
        if user_input.lower() == "exit":
            break
        
        conversation_history.append({"role": "user", "content": user_input})
        print(PINK + "Emma:" + RESET_COLOR)
        chatbot_response = chatgpt_streamed(user_input, system_message, conversation_history)
        conversation_history.append({"role": "assistant", "content": chatbot_response})
        
        prompt2 = chatbot_response
        audio_file_pth2 = r"D:\Private Server\Projects\friday\XTTS-v2\samples\emma.wav"
        audio_paths = process(prompt, audio_file_pth)
        play(audio_paths)
   

        if len(conversation_history) > 20:
            conversation_history = conversation_history[-20:]

user_chatbot_conversation()  # Start the conversation