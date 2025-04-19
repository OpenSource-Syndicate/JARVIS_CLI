import speech_recognition as sr
from TOOLS.AUDIO.record import record_audio
import threading
from googletrans import Translator

async def translation(audio_file_path="output.wav", language_codes=['en-US', 'te-IN']):
    def transcribe_audio_with_fallback(audio_file_path, language_codes):
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_file_path) as source:
            audio = recognizer.record(source)

        results = {}
        threads = []
        lock = threading.Lock()

        def transcribe(lang):
            try:
                text = recognizer.recognize_google(audio, language=lang)
                with lock:
                    results[lang] = text
            except sr.UnknownValueError:
                pass

        for lang in language_codes:
            t = threading.Thread(target=transcribe, args=(lang,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        return results
    
    record_audio()
    transcriptions = transcribe_audio_with_fallback(audio_file_path, language_codes)
    if not transcriptions:
        print("No transcription succeeded")
    else:
        translator = Translator()
        combined_results = {}
        for lang, text in transcriptions.items():
            combined_results[lang] = {'original': text}
            if lang.startswith('te'):
                try:
                    translated = translator.translate(text, src='te', dest='en')
                    combined_results[lang]['translated'] = translated.text
                except Exception as e:
                    combined_results[lang]['translated'] = f"Translation error: {e}"
        return combined_results

if __name__ == "__main__":
    import asyncio
    print(asyncio.run(translation()))  # Changed from main() to translation()