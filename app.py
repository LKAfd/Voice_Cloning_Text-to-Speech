import streamlit as st
from TTS.api import TTS
import librosa
import soundfile as sf
import numpy as np
from io import BytesIO
import gc
import psutil
import torch
from datetime import datetime

# Configuration
MODEL_NAME = "tts_models/multilingual/multi-dataset/your_tts"
TARGET_SAMPLE_RATE = 16000
MAX_FILE_SIZE_MB = 3
MAX_TEXT_LENGTH = 1000  # Increased to 1000 characters
MEMORY_LIMIT_MB = 800

@st.cache_resource(max_entries=1, ttl=600, show_spinner=False)
def load_model():
    """Load optimized TTS model"""
    model = TTS(model_name=MODEL_NAME, progress_bar=False, gpu=False)
    return model.to('cpu')

def memory_guard():
    """Monitor and protect against memory overflows"""
    process = psutil.Process()
    mem_usage = process.memory_info().rss / (1024 ** 2)
    
    if mem_usage > MEMORY_LIMIT_MB * 0.85:
        st.error(f"🛑 Memory Usage High ({mem_usage:.1f}MB/{MEMORY_LIMIT_MB}MB)\n"
                 "1. Use shorter text/audio\n2. Refresh page\n3. Wait 5 minutes")
        st.stop()

def process_audio(uploaded_file):
    """Efficient audio processing with safety checks"""
    try:
        with BytesIO(uploaded_file.getvalue()) as audio_buffer:
            y, sr = librosa.load(
                audio_buffer,
                sr=TARGET_SAMPLE_RATE,
                mono=True,
                duration=30  # Keep 30s limit for memory safety
            )
        
        y = librosa.effects.trim(y, top_db=25)[0]
        y = (y * 32767).astype(np.int16)
        
        with BytesIO() as processed_buffer:
            sf.write(processed_buffer, y, TARGET_SAMPLE_RATE, 
                    format='WAV', subtype='PCM_16')
            return processed_buffer.getvalue()
            
    except Exception as e:
        st.error(f"Audio Processing Error: {str(e)}")
        return None

def clean_memory():
    """Comprehensive memory cleanup"""
    if 'tts' in st.session_state:
        del st.session_state.tts
    if 'audio_data' in globals():
        del globals()['audio_data']
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    st.cache_data.clear()
    st.cache_resource.clear()

def main():
    st.title("🔊 Enhanced Voice Cloning")
    
    # Memory safety check
    memory_guard()
    
    # File upload section
    voice_file = st.file_uploader(
        "Upload Voice Sample (max 30s)", 
        type=["wav", "mp3"],
        help=f"Maximum size: {MAX_FILE_SIZE_MB}MB"
    )
    
    # Text input with expanded limits
    text_input = st.text_area(
        "Enter Text to Convert", 
        height=150,
        max_chars=MAX_TEXT_LENGTH,
        help=f"Up to {MAX_TEXT_LENGTH} characters allowed"
    )
    
    if st.button("Generate Speech") and voice_file and text_input:
        with st.spinner("Processing..."):
            try:
                memory_guard()
                
                # Lazy load model
                if 'tts' not in st.session_state:
                    st.session_state.tts = load_model()
                
                # Process audio
                audio_data = process_audio(voice_file)
                if not audio_data:
                    return
                
                # Generate speech
                with BytesIO() as output_buffer:
                    st.session_state.tts.tts_to_file(
                        text=text_input[:MAX_TEXT_LENGTH],
                        speaker_wav=BytesIO(audio_data),
                        language="en",
                        file_path=output_buffer
                    )
                    
                    # Display audio
                    st.audio(output_buffer.getvalue())
                    st.success("Generation Complete!")
                    
                    # Enhanced download button
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    st.download_button(
                        label="Download Audio",
                        data=output_buffer.getvalue(),
                        file_name=f"cloned_voice_{timestamp}.wav",
                        mime="audio/wav",
                        help="Click to download generated audio"
                    )
                    
                    # Cleanup
                    clean_memory()
                    memory_guard()

            except Exception as e:
                st.error(f"Generation Error: {str(e)}")
                clean_memory()

    # Memory monitor
    mem_usage = psutil.Process().memory_info().rss / (1024 ** 2)
    st.sidebar.progress(
        min(mem_usage / MEMORY_LIMIT_MB, 1.0),
        text=f"Memory Usage: {mem_usage:.1f}MB / {MEMORY_LIMIT_MB}MB"
    )

if __name__ == "__main__":
    main()