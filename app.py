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
import os

# Enhanced Configuration with 8GB resources
MODEL_NAME = "tts_models/multilingual/multi-dataset/your_tts"
TARGET_SAMPLE_RATE = 22050
MAX_FILE_SIZE_MB = 50
MAX_TEXT_LENGTH = 3000
MEMORY_LIMIT_MB = 8192
MAX_AUDIO_DURATION = 60
MODEL_TTL = 1800

@st.cache_resource(max_entries=1, ttl=MODEL_TTL, show_spinner=False)
def load_model():
    """Load model with enhanced capabilities and memory optimization"""
    model = TTS(model_name=MODEL_NAME, progress_bar=False, gpu=False)
    return model.to('cpu')

def memory_guard():
    """Proactive memory monitoring with detailed feedback"""
    process = psutil.Process()
    mem_usage = process.memory_info().rss / (1024 ** 2)
    
    # Detailed memory usage feedback
    if mem_usage > MEMORY_LIMIT_MB * 0.90:
        st.error(f"🚨 Critical Memory Usage ({mem_usage:.1f}MB/{MEMORY_LIMIT_MB}MB)\n"
                 "• Refresh the page to reset\n• Use shorter audio/text\n• Contact support")
        st.stop()
    elif mem_usage > MEMORY_LIMIT_MB * 0.75:
        st.warning(f"⚠️ High Memory Usage ({mem_usage:.1f}MB/{MEMORY_LIMIT_MB}MB)\n"
                  "Consider reducing input size for better performance")

def process_audio(uploaded_file):
    """Memory-efficient audio processing with validation"""
    try:
        # Validate file size first
        file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
        if file_size > MAX_FILE_SIZE_MB:
            raise ValueError(f"File size {file_size:.1f}MB exceeds {MAX_FILE_SIZE_MB}MB limit")
        
        # Process with duration limit
        with BytesIO(uploaded_file.getvalue()) as audio_buffer:
            y, sr = librosa.load(
                audio_buffer,
                sr=TARGET_SAMPLE_RATE,
                mono=True,
                duration=MAX_AUDIO_DURATION
            )
        
        # Audio enhancement
        y = librosa.effects.trim(y, top_db=20)[0]
        y = (y * 32767).astype(np.int16)
        
        # Memory-efficient output
        processed_buffer = BytesIO()
        sf.write(processed_buffer, y, TARGET_SAMPLE_RATE, 
                format='WAV', subtype='PCM_16')
        return processed_buffer.getvalue()
            
    except Exception as e:
        st.error(f"Audio Processing Error: {str(e)}")
        return None

def optimized_generation(text, audio_data):
    """Memory-optimized speech generation"""
    with BytesIO() as output_buffer:
        st.session_state.tts.tts_to_file(
            text=text[:MAX_TEXT_LENGTH],
            speaker_wav=BytesIO(audio_data),
            language="en",
            file_path=output_buffer
        )
        return output_buffer.getvalue()

def memory_cleanup():
    """Safe and efficient memory cleanup"""
    # Clear large objects
    for var in ['audio_data', 'output_data', 'processed_buffer']:
        if var in globals():
            del globals()[var]
    
    # Python garbage collection
    gc.collect()
    
    # PyTorch cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Clear temporary files
    for f in ["output.wav"]:
        if os.path.exists(f):
            try:
                os.remove(f)
            except:
                pass

def main():
    st.title("🔊 Premium Voice Cloning")
    st.caption("Now with enhanced 8GB memory capacity")
    
    # Initial memory check
    memory_guard()
    
    # File upload section
    voice_file = st.file_uploader(
        "Upload Voice Sample", 
        type=["wav", "mp3", "ogg"],
        help=f"Max {MAX_FILE_SIZE_MB}MB, {MAX_AUDIO_DURATION}s duration"
    )
    
    # Text input with expanded limits
    text_input = st.text_area(
        "Text to Convert to Speech", 
        height=200,
        max_chars=MAX_TEXT_LENGTH,
        placeholder="Enter up to 3000 characters...",
        help="For best results, use 1-3 paragraphs of text"
    )
    
    # Language selection
    language_options = ["English", "French", "Portuguese", "Spanish", "German"]
    selected_lang = st.selectbox("Output Language", language_options, index=0)
    
    # Quality settings
    with st.expander("Advanced Settings"):
        speed_factor = st.slider("Speech Speed", 0.8, 1.2, 1.0, 0.05)
        emphasis_level = st.slider("Word Emphasis", 1, 3, 2)
    
    if st.button("Generate Speech") and voice_file and text_input:
        with st.spinner("Processing with premium resources..."):
            try:
                # Continuous memory monitoring
                memory_guard()
                
                # Lazy load model
                if 'tts' not in st.session_state:
                    with st.spinner("Loading AI model (first time may take longer)..."):
                        st.session_state.tts = load_model()
                
                # Process audio
                audio_data = process_audio(voice_file)
                if not audio_data:
                    return
                
                # Generate speech
                output_data = optimized_generation(text_input, audio_data)
                
                # Display results
                st.audio(output_data)
                st.success("Premium Quality Speech Generated!")
                
                # Download button (FIXED: removed 'type' parameter)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(
                    label="Download Audio",
                    data=output_data,
                    file_name=f"premium_clone_{selected_lang}_{timestamp}.wav",
                    mime="audio/wav"
                )
                
                # Cleanup
                memory_cleanup()

            except Exception as e:
                st.error(f"Generation Error: {str(e)}")
                memory_cleanup()

    # Resource monitoring dashboard
    mem_usage = psutil.Process().memory_info().rss / (1024 ** 2)
    mem_percent = min(mem_usage / MEMORY_LIMIT_MB, 1.0)
    
    st.sidebar.subheader("Resource Monitor")
    st.sidebar.metric("Memory Usage", f"{mem_usage:.1f}MB", f"{mem_percent*100:.1f}%")
    st.sidebar.progress(mem_percent)
    
    # System information
    with st.sidebar.expander("System Info"):
        st.write(f"**Model**: {MODEL_NAME}")
        st.write(f"**Max Text Length**: {MAX_TEXT_LENGTH} chars")
        st.write(f"**Max Audio Duration**: {MAX_AUDIO_DURATION}s")
        st.write(f"**Memory Allocation**: {MEMORY_LIMIT_MB}MB")
        
    # Add resource documentation link
    st.sidebar.markdown("[Memory Optimization Tips](https://blog.streamlit.io/3-steps-to-fix-app-memory-leaks/)")

if __name__ == "__main__":
    # Set environment variables for stability
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    main()