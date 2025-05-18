from TTS.utils.manage import ModelManager
import os

def download_model():
    """Robust model downloader for TTS v0.20.x"""
    print("🚚 Downloading YourTTS Model...")
    
    manager = ModelManager()
    model_path, config_path, model_item = manager.download_model("tts_models/multilingual/multi-dataset/your_tts")
    
    print(f"✅ Model downloaded to: {model_path}")

if __name__ == "__main__":
    download_model()