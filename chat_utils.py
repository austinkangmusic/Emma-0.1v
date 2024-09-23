import llms
import json
import re

def initialize():
    global chat_llm, utility_llm, embedding_llm

    # main chat model used by agents (smarter, more accurate)
    # chat_llm = models.get_openai_chat(model_name="gpt-4o-mini", temperature=0)
    # chat_llm = models.get_ollama_chat(model_name="gemma2:latest", temperature=0)
    # chat_llm = models.get_lmstudio_chat(model_name="TheBloke/Mistral-7B-Instruct-v0.2-GGUF", temperature=0)
    # chat_llm = models.get_openrouter(model_name="meta-llama/llama-3-8b-instruct:free")
    # chat_llm = models.get_azure_openai_chat(deployment_name="gpt-4o-mini", temperature=0)
    # chat_llm = models.get_anthropic_chat(model_name="claude-3-5-sonnet-20240620", temperature=0)
    # chat_llm = models.get_google_chat(model_name="gemini-1.5-flash", temperature=0)
    chat_llm = llms.get_groq_chat(model_name="llama-3.1-70b-versatile", temperature=0)
    
    # utility model used for helper functions (cheaper, faster)
    utility_llm = chat_llm # change if you want to use a different utility model

    # embedding model used for memory
    # embedding_llm = models.get_openai_embedding(model_name="text-embedding-3-small")
    # embedding_llm = models.get_ollama_embedding(model_name="nomic-embed-text")
    embedding_llm = llms.get_huggingface_embedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    return chat_llm, utility_llm, embedding_llm

def use_chat_llm():
    return chat_llm

def use_utility_llm():
    return utility_llm

def use_embedding_llm():
    return embedding_llm

def remove_trailing_commas(json_string):
    # Remove trailing commas before closing brackets or braces
    json_string = re.sub(r',\s*(\]|\})', r'\1', json_string)
    return json_string

def extract_tool_info(response_json):
    # Clean the JSON string by removing invalid elements
    cleaned_response = re.sub(r'\.\.\.|//.*', '', response_json).strip()
    
    # Remove trailing commas
    cleaned_response = remove_trailing_commas(cleaned_response)

    try:
        response = json.loads(cleaned_response)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON after cleanup: {e}")
        print("Cleaned response content:", cleaned_response)
        return None, None  # Return None if JSON parsing fails

    tool_name = response.get("tool_name", "")
    tool_args = response.get("tool_args", {})

    return tool_name, tool_args

