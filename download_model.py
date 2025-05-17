from TTS.api import TTS

print("Downloading YourTTS model... (This will take 20-40 minutes)")
TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=True)
print("\nModel downloaded successfully!")