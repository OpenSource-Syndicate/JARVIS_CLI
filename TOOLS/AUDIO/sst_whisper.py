from faster_whisper import WhisperModel

# Load the model (choose size: tiny, base, small, medium, large)
model = WhisperModel("medium")

# Transcribe audio (supports various formats like WAV, MP3, etc.)
def sst(path):
    segments, info = model.transcribe(path)

    transcription = " ".join([segment.text for segment in segments])
    return transcription