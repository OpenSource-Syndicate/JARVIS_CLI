import threading
import json
from vosk import Model, KaldiRecognizer
import wave
from langdetect import detect
import os
from pydub import AudioSegment

def convert_to_vosk_format(input_path):
    """Convert audio file to Vosk-compatible format (16kHz, mono, 16-bit PCM WAV)"""
    temp_path = input_path + "_converted.wav"
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)
    audio.export(temp_path, format="wav", parameters=["-acodec", "pcm_s16le"])
    return temp_path

def transcribe_audio(audio_path):
    # Convert audio if needed
    try:
        wf = wave.open(audio_path, "rb")
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
            wf.close()
            audio_path = convert_to_vosk_format(audio_path)
        else:
            wf.close()
    except Exception as e:
        print(f"Converting audio to compatible format: {str(e)}")
        audio_path = convert_to_vosk_format(audio_path)

    # Get absolute paths to models
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Go up 3 levels to JARVIS root
    model_en_path = os.path.join(base_path, "models", "vosk-model-en-us-0.22")
    model_te_path = os.path.join(base_path, "models", "vosk-model-small-te-0.42")

    # Load language models
    try:
        model_en = Model(model_en_path)  # English model path
        model_te = Model(model_te_path)  # Telugu model path
    except Exception as e:
        raise Exception(f"Failed to load models. Make sure they exist at: \n{model_en_path}\n{model_te_path}\nError: {str(e)}")

    # Read and validate audio file
    wf = wave.open(audio_path, "rb")
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
        raise ValueError("Audio must be mono PCM 16kHz 16-bit WAV format")
    
    audio_data = wf.readframes(wf.getnframes())
    wf.close()

    # Thread-safe results storage
    results = [None, None]
    lock = threading.Lock()

    # Worker function for model processing
    def process_model(model, index):
        rec = KaldiRecognizer(model, 16000)
        rec.AcceptWaveform(audio_data)
        res = json.loads(rec.Result())
        with lock:
            results[index] = res.get('text', '').strip()

    # Create and start threads
    thread_en = threading.Thread(target=process_model, args=(model_en, 0))
    thread_te = threading.Thread(target=process_model, args=(model_te, 1))
    
    thread_en.start()
    thread_te.start()
    
    thread_en.join()
    thread_te.join()

    # Language detection and selection
    en_text, te_text = results
    
    try:
        # Check English result
        if en_text:
            lang = detect(en_text)
            if lang == 'en':
                return en_text
        
        # Check Telugu result
        if te_text:
            lang = detect(te_text)
            if lang == 'te':
                return te_text
            
        # Fallback if detection fails
        return en_text or te_text or "No transcription found"
    
    except Exception as e:
        print(f"Language detection error: {str(e)}")
        return en_text or te_text or "No transcription found"
    
if __name__=="__main__":
    print(transcribe_audio("output.wav"))