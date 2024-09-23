import chat_utils
import files

chat_utils.initialize()

chat_llm = chat_utils.use_chat_llm()

def stream_response(prompt, model, system_prompt):
    # Prepend the system prompt to the user input
    conversation = system_prompt + "\n\n" + prompt
    chunks = []
    for chunk in model.stream(conversation):  # Use .stream instead of .astream
        chunks.append(chunk.content)
        print(chunk.content, end="", flush=True)
    return ''.join(chunks)

while True:
    # Read the system prompt from the file
    system_prompt = files.read_file("./prompts/detector.md")
    
    user_input = input("\nYou: ")
    
    # Stream the response with both the system prompt and user input
    response_content = stream_response(user_input, chat_llm, system_prompt)

    
    tool_name, tool_args = chat_utils.extract_tool_info(chatbot_response)

    chatbot_response = tool_name.get('tool_name')
