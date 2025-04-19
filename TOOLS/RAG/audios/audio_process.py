from google import genai
import os
import mimetypes
from pathlib import Path
from TOOLS.DATABASE.data import ConversationDB
from TOOLS.AUDIO.tts import speak

def process_audio(audio_file_path, prompt, api_key="GOOGLE_API_KEY"):
    # Initialize the client
    client = genai.Client(api_key=api_key)
    db = ConversationDB()
    
    try:
        # Clean and normalize the file path
        audio_path = Path(audio_file_path.strip('"')).resolve()
        
        # Check if file exists and is accessible
        if not audio_path.is_file():
            return f"Error: File not found or inaccessible: {audio_path}"
        
        # Determine the MIME type of the audio file
        mime_type, _ = mimetypes.guess_type(str(audio_path))
        if not mime_type or not mime_type.startswith('audio/'):
            mime_type = 'audio/mpeg'  # Default to audio/mpeg if type can't be determined
        
        # Upload the audio file
        myfile = client.files.upload(file=audio_path)
        
        # Generate content from the audio
        response = client.models.generate_content(
            model="gemini-2.0-flash", # Or the appropriate model for audio
            contents=[myfile, prompt]
        )
        
        # Store the processed data in database
        result_text = response.text if response.text else "No response generated"
        db.save_message(
            context=f"audio_processing:{audio_path.name}",
            sender="assistant",
            message=result_text,
        )
        db.save_message(
            context=f"audio_processing:{audio_path.name}",
            sender="system",
            message=f"Prompt: {prompt}\nResponse: {result_text}"
        )
        speak(result_text)

        # Return the respons   
        return result_text
            
    except FileNotFoundError:
        return f"Error: Could not find audio file at {audio_file_path}"
    except PermissionError:
        return f"Error: Permission denied accessing {audio_file_path}"
    except Exception as e:
        return f"Error processing audio: {str(e)}"
