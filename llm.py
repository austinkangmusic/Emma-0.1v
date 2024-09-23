import os
os.environ['TRANSFORMERS_NO_TF'] = '1'


import chat_utils
import files

# Initialize LLM models
chat_utils.initialize()

chat_llm = chat_utils.use_chat_llm()

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

def run():
    conversation_history = []

    while True:
        # Read the system prompt from the file
        system_prompt = files.read_file("./prompts/detector.md")
        
        user_input = input('\nUser:\n')
            
        conversation_history.append({"role": "user", "content": user_input})

        # Stream the response with the system prompt, user input, and conversation history
        chatbot_response = stream_response(chat_llm, system_prompt, conversation_history)
        
        conversation_history.append({"role": "assistant", "content": chatbot_response})

        if len(conversation_history) > 20:
            conversation_history = conversation_history[-20:]

run()

