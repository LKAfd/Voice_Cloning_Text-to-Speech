import streamlit as st
from TTS.api import TTS
import librosa
import soundfile as sf
import numpy as np
from io import BytesIO

# Configuration
MODEL_NAME = "tts_models/multilingual/multi-dataset/your_tts"
SUPPORTED_LANGUAGES = {
    "English": "en",
    "French": "fr-fr",
    "Portuguese": "pt-br"
}
TARGET_SAMPLE_RATE = 22050  # Define the target sample rate (e.g., 22050 Hz)

@st.cache_resource
def load_model():
    return TTS(model_name=MODEL_NAME, progress_bar=False, gpu=False)

def process_audio(uploaded_file):
    """Process audio file to correct format"""
    try:
        # Read audio file
        audio_bytes = uploaded_file.read()
        
        # Get file extension
        file_ext = uploaded_file.name.split('.')[-1].lower()
        if file_ext not in ['wav', 'mp3', 'ogg', 'flac']:
            st.error("Unsupported file format. Please use WAV, MP3, OGG, or FLAC.")
            return None

        # Load with librosa (resample to target sr)
        y, sr = librosa.load(BytesIO(audio_bytes), sr=TARGET_SAMPLE_RATE)
        
        # Convert to proper format
        y = librosa.effects.trim(y, top_db=20)[0]  # Remove silence
        y = (y * 32767).astype(np.int16)  # Convert to 16-bit PCM
        
        # Create in-memory WAV file with explicit format
        processed_audio = BytesIO()
        sf.write(
            processed_audio,
            y,
            TARGET_SAMPLE_RATE,
            format='WAV',  # Explicitly set format
            subtype='PCM_16'
        )
        processed_audio.seek(0)  # Reset pointer to start of file
        
        return processed_audio
        
    except Exception as e:
        st.error(f"Audio processing error: {str(e)}")
        return None

def main():
    st.title("🌍 Working Voice Cloning")
    
    if 'tts' not in st.session_state:
        with st.spinner("Loading model..."):
            st.session_state.tts = load_model()
    
    selected_lang = st.selectbox("Choose Language", list(SUPPORTED_LANGUAGES.keys()))
    lang_code = SUPPORTED_LANGUAGES[selected_lang]
    
    voice_file = st.file_uploader("Upload voice sample", type=["wav", "mp3"])
    text_input = st.text_area("Text to speak", height=150)
    
    if st.button("Generate") and voice_file and text_input:
        with st.spinner("Processing..."):
            try:
                processed_audio = process_audio(voice_file)
                if not processed_audio:
                    return
                
                output_path = "output.wav"
                st.session_state.tts.tts_to_file(
                    text=text_input,
                    speaker_wav=processed_audio,
                    language=lang_code,
                    file_path=output_path
                )
                
                st.audio(output_path)
                st.success("Done!")
                
                with open(output_path, "rb") as f:
                    st.download_button("Download", f, "cloned.wav")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()