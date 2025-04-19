from google import genai
from pathlib import Path
import mimetypes
from TOOLS.DATABASE.data import ConversationDB
from TOOLS.AUDIO.tts import speak

def analyze_video(video_path, api_key, prompt, max_size_mb=100):
    client = genai.Client(api_key=api_key)
    db = ConversationDB()
    max_bytes = max_size_mb * 1024 * 1024  # Convert MB to bytes

    try:
        # Validate and prepare file path
        video_path = Path(video_path.strip('"')).resolve()
        
        if not video_path.is_file():
            return f"Error: File not found - {video_path}"
            
        file_size = video_path.stat().st_size
        if file_size > max_bytes:
            return f"Error: File exceeds {max_size_mb}MB limit ({file_size/1e6:.1f}MB)"

        # Determine MIME type
        mime_type, _ = mimetypes.guess_type(str(video_path))
        if not mime_type or not mime_type.startswith('video/'):
            mime_type = 'video/mp4'  # Default to MP4 format

        # Read video bytes
        with open(video_path, 'rb') as f:
            video_bytes = f.read()

        # Prepare request content
        content = genai.types.Content(
            parts=[
                genai.types.Part(
                    inline_data=genai.types.Blob(
                        data=video_bytes,
                        mime_type=mime_type
                    )
                ),
                genai.types.Part(text=prompt)
            ]
        )

        # Generate response
        response = client.models.generate_content(
            model='models/gemini-2.0-flash',
            contents=content
        )

        # Store the processed data in database
        result_text = response.text if response.text else "No response generated"
        db.save_message(
            context=f"video_processing:{video_path.name}",
            sender="assistant",
            message=result_text,
        )
        db.save_message(
            context=f"video_processing:{video_path.name}",
            sender="system",
            message=f"Prompt: {prompt}\nResponse: {result_text}"
        )
        speak(result_text)
        return result_text

    except FileNotFoundError:
        return f"Error: Could not find video file at {video_path}"
    except PermissionError:
        return f"Error: Permission denied accessing {video_path}"
    except Exception as e:
        return f"Error analyzing video: {str(e)}"