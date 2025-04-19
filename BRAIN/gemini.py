import google.generativeai as genai
from TOOLS.DATABASE.data import ConversationDB

def ask_gemini(prompt: str, api_key: str, context: str = "default_context"):
    """Asks the Gemini model a question, incorporating conversation history."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash') # Or use 'gemini-pro' or other models

    db = ConversationDB()

    # Fetch and format history
    history_tuples = db.get_conversation_history(context)
    formatted_history = []
    for sender, message, _ in history_tuples:
        role = "user" if sender.lower() == "user" else "model"
        formatted_history.append({'role': role, 'parts': [message]})

    # Combine history and current prompt
    contents = formatted_history + [{'role': 'user', 'parts': [prompt]}]

    try:
        # Start a chat session if needed (for multi-turn)
        # For single turn, generate_content might be sufficient
        # chat = model.start_chat(history=formatted_history) 
        # response = chat.send_message(prompt)
        
        # Using generate_content for simplicity, adjust if multi-turn chat is needed
        response = model.generate_content(contents)

        # Save user prompt and assistant response
        db.save_message(context=context, sender="user", message=prompt)
        if response.text:
             db.save_message(context=context, sender="assistant", message=response.text)
             return response.text
        else:
            # Handle cases where response might be empty or blocked
            error_message = "Gemini response was empty or blocked."
            # Optionally save this error state to DB if needed
            # db.save_message(context=context, sender="system_error", message=error_message)
            return error_message

    except Exception as e:
        error_message = f"Error calling Gemini API: {str(e)}"
        # Optionally save this error state to DB
        # db.save_message(context=context, sender="system_error", message=error_message)
        return error_message

