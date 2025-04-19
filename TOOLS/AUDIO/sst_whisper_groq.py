import os
import json
from groq import Groq

def transcribe_audio(audio_file_path, api_key=None):
    """
    Transcribe an audio file using Groq's API.
    
    Args:
        audio_file_path (str): Path to the audio file
        api_key (str, optional): Groq API key. If not provided, it should be set as environment variable.
    
    Returns:
        dict: Transcription response from Groq
    """
    if api_key:
        client = Groq(api_key=api_key)
    else:
        client = Groq()

    # Open the audio file
    with open(audio_file_path, "rb") as file:
        # Create a transcription of the audio file
        transcription = client.audio.transcriptions.create(
            file=file,
            model="whisper-large-v3-turbo",
            prompt="Specify context or spelling",
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"],
            language="en",
            temperature=0.0
        )
        return transcription