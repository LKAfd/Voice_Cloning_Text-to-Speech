from TTS.utils.manage import ModelManager
import os
import time
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

def download_model():
    """Robust model downloader with enhanced reliability"""
    MODEL_NAME = "tts_models/multilingual/multi-dataset/your_tts"
    
    logging.info("🚀 Starting YourTTS Model Download Process")
    logging.info(f"• Model: {MODEL_NAME}")
    logging.info("⚠️ Note: This may take 20-40 minutes depending on your internet connection")
    
    try:
        start_time = time.time()
        
        # Initialize model manager
        manager = ModelManager()
        
        # Check if model already exists
        if manager.model_exists(MODEL_NAME):
            logging.info("✅ Model already exists. Skipping download.")
            return True
            
        logging.info("🔍 Starting model download...")
        
        # Download model
        model_path, config_path, model_item = manager.download_model(MODEL_NAME)
        
        # Verify download
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
            
        # Calculate download stats
        download_time = time.time() - start_time
        logging.info(f"🎉 Successfully downloaded in {download_time/60:.2f} minutes")
        logging.info(f"📁 Model location: {model_path}")
        
        return True

    except Exception as e:
        logging.error(f"❌ Download failed: {str(e)}")
        logging.error("💡 Troubleshooting tips:")
        logging.error("- Check internet connection")
        logging.error("- Ensure sufficient disk space (>5GB free)")
        logging.error("- Try: pip install --upgrade TTS")
        return False

if __name__ == "__main__":
    print("=" * 70)
    print("COQUI TTS MODEL DOWNLOADER".center(70))
    print("=" * 70)
    
    if download_model():
        print("\n✅ Download completed successfully!")
        print("You can now run the Streamlit app: streamlit run app.py")
    else:
        print("\n❌ Download failed. See log above for details")
        sys.exit(1)