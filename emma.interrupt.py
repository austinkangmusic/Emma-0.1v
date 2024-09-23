import chat_utils
import files
from initialize_whisper import initialize_whisper_model
from interrupted_play import play_interruptable_audio
from transcribe import start
from initialize_tts import initialize_tts_model

# Initialize the TTS model and latents
model, gpt_cond_latent, speaker_embedding = initialize_tts_model()

import generate_voice
from intro import initialize_emma

# Initialize LLM models
chat_llm, utility_llm, embedding_llm = chat_utils.initialize()

# Initialize Whisper model
whisper_model = initialize_whisper_model()

def save_response(text):
    with open('ai_response.txt', 'w', encoding='utf-8') as file:
        file.write(text)

def stream_response(model, system_prompt, conversation_history):
    # Combine system prompt, conversation history, and user input
    full_conversation = [{"role": "system", "content": system_prompt}] + conversation_history
    print("\n\n\n\nFULL CONVERSATION\n\n\n\n\n", full_conversation)
    
    chunks = []
    response_generator = model.stream(full_conversation)
    for chunk in response_generator:
        delta_content = chunk.content
        if delta_content is not None:
            chunks.append(delta_content)
            print(delta_content, end="", flush=True)
    print('\n')
    return ''.join(chunks)

first = True

def main():
    global first
    conversation_history = []
    initialize_emma()

    while True:
        # DETECTION
        if first:
            start(whisper_model)
            first = False
            with open("transcription/user_audio.txt", "r") as file:
                user_input = file.read() 
        else:
                        # Define the file path
            file_path = "interrupted_status.txt"

            # Read the content of the file
            with open(file_path, 'r') as file:
                interrupted_status = file.read()

            # Print the value of interrupted_status (optional)
            print(interrupted_status)

            if interrupted_status.lower() == 'true':
                with open("transcription/interrupted_user_audio.txt", "r") as file:
                    user_input_before = file.read()   

                with open("transcription/interrupted_output_0.txt", "r") as file:
                    interrupted_ai_response = file.read()   
                    user_input = f'You said: "{interrupted_ai_response}-" and have been interrupted by the user with User: "{user_input_before}"'              
            else:
                with open("transcription/user_audio.txt", "r") as file:
                    user_input = file.read()                

            
        # Read the system prompt from the file
        system_prompt = files.read_file("./prompts/detector.md")

            
        conversation_history.append({"role": "user", "content": user_input})

        # Stream the response with the system prompt, user input, and conversation history
        chatbot_response = stream_response(chat_llm, system_prompt, conversation_history)
        
        conversation_history.append({"role": "assistant", "content": chatbot_response})

        tool_name, tool_args = chat_utils.extract_tool_info(chatbot_response)

        if len(conversation_history) > 20:
            conversation_history = conversation_history[-20:]

        if tool_name in ('response', 'intervene'):
            text_to_save = tool_args.get('text', '')
            save_response(text_to_save)
            generate_voice.run()
            play_interruptable_audio(whisper_model)

        elif tool_name in ('ignore', 'listen'):
            start(whisper_model)
            status = 'False'
            with open('interrupted_status.txt', 'w') as file:
                file.write(status)                 

main()

