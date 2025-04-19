from google import genai
from google.genai import types
import pathlib
import httpx
from TOOLS.DATABASE.data import ConversationDB
from TOOLS.AUDIO.tts import speak

def analyze_pdf(doc_path, prompt="Summarize this document", api_key=None):
    """
    Analyze a PDF document using Gemini model
    
    Args:
        doc_path (str): URL or local path to PDF file
        prompt (str): Analysis prompt for the model
        api_key (str): Google Generative AI API key
        
    Returns:
        str: Model response text
    """
    client = genai.Client(api_key=api_key)
    db = ConversationDB()
    
    try:
        # Handle URL or local file paths
        if doc_path.startswith(('http', 'https')):
            response = httpx.get(doc_path)
            response.raise_for_status()
            pdf_bytes = response.content
        else:
            filepath = pathlib.Path(doc_path.strip('"').strip())
        filepath = filepath.resolve()
        pdf_bytes = filepath.read_bytes()
        
        # Create content parts
        pdf_part = types.Part.from_bytes(
            data=pdf_bytes,
            mime_type='application/pdf'
        )
        text_part = types.Part.from_text(text=prompt)
        
        # Generate response using client
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=[pdf_part, text_part]
        )
        
        # Store the processed data in database
        result_text = response.text if response.text else "No response generated"
        db.save_message(
            context=f"pdf_processing:{filepath.name}",
            sender="assistant",
            message=result_text
        )
        sender="system",
        message=f"Prompt: {prompt}\nResponse: {result_text}"
        db.save_message(
            context=f"pdf_processing:{filepath.name}",
            sender=sender,
            message=message
        )
        speak(result_text)
        return result_text

    except Exception as e:
        return f"Error analyzing PDF: {str(e)}\nFile path: {doc_path}"