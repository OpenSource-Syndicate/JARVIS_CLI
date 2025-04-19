from a4f_local import A4F
from TOOLS.AUDIO.Interrupted_Playsound import play_audio
from pathlib import Path
import os
import pygame

def text_to_speech(text, voice="alloy", output_file="output.wav"):
    """
    Convert text to speech using A4F
    
    Args:
        text (str): The text to convert to speech
        voice (str): The voice to use (default: "alloy")
        output_file (str): The output file path (default: "output.wav")
    """
    client = A4F()
    
    try:
        # Play processing sound
        pygame.mixer.init()
        pygame.mixer.music.load("ASSETS\\SOUNDS\\calculating_sound.mp3")
        pygame.mixer.music.play()
        
        # Normalize output file path
        output_path = Path(output_file).resolve()
        
        audio_bytes = client.audio.speech.create(
            model="tts-1",
            input=text,
            voice=voice
        )
        with open(output_path, "wb") as f:
            f.write(audio_bytes)
        print(f"Generated {output_path}")
        
        # Stop processing sound
        pygame.mixer.music.stop()
        pygame.mixer.quit()
        
        return True
    except Exception as e:
        print(f"An error occurred: {e}")
        # Ensure sound is stopped even on error
        if pygame.mixer.get_init():
            pygame.mixer.music.stop()
            pygame.mixer.quit()
        return False


def speak(text, voice="shimmer"):
    text_to_speech(text=text, voice=voice)
    # Use absolute path for audio file
    audio_path = Path("output.wav").resolve()
    play_audio(audio_file_path=str(audio_path), prints=True)