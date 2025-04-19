import speech_recognition as sr

def record_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        r.adjust_for_ambient_noise(source)  # Adjust for ambient noise
        try:
            # phrase_time_limit=None means it will listen until silence is detected
            audio = r.listen(source, phrase_time_limit=None)
            # Save the audio to WAV file
            with open("output.wav", "wb") as f:
                f.write(audio.get_wav_data())
            return audio
        except sr.RequestError as e:
            print(f"Error occurred: {e}")
            return None