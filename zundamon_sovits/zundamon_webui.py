import streamlit as st
import tempfile
import os
import soundfile as sf

import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_dir)
sys.path.insert(0, repo_root)
sys.path.insert(0, current_dir)

sys.path.append(os.path.join(current_dir, 'GPT_SoVITS'))

# Import your inference functions and required packages (adjust import paths as needed)
from tools.i18n.i18n import I18nAuto
from GPT_SoVITS.inference_webui import change_gpt_weights, change_sovits_weights, get_tts_wav

# Fixed model file paths (please modify as needed)
GPT_MODEL_PATH =  os.path.join(current_dir, 'GPT_weights_v2', 'zudamon_style_1-e15.ckpt') # "/GPT_weights_v2/zudamon_style_1-e15.ckpt"
SOVITS_MODEL_PATH = os.path.join(current_dir, 'SoVITS_weights_v2', 'zudamon_style_1_e8_s96.pth')

# Define the inference function
def synthesize(GPT_model_path, SoVITS_model_path, ref_audio_path, ref_text_path, ref_language, target_text_path, target_language, output_path):
    # i18n = I18nAuto()

    # Read reference text
    with open(ref_text_path, 'r', encoding='utf-8') as file:
        ref_text = file.read()

    # Read target text
    with open(target_text_path, 'r', encoding='utf-8') as file:
        target_text = file.read()

    # Change model weights
    change_gpt_weights(gpt_path=GPT_model_path)
    change_sovits_weights(sovits_path=SOVITS_MODEL_PATH)

    # Generate audio
    synthesis_result = get_tts_wav(ref_wav_path=ref_audio_path, 
                                   prompt_text=ref_text, 
                                   prompt_language=ref_language, 
                                   text=target_text, 
                                   text_language=target_language, 
                                   top_p=1, temperature=1)
    
    result_list = list(synthesis_result)

    if result_list:
        last_sampling_rate, last_audio_data = result_list[-1]
        output_wav_path = os.path.join(output_path, "output.wav")
        sf.write(output_wav_path, last_audio_data, last_sampling_rate)
        print(f"Audio saved to {output_wav_path}")

# Page configuration
st.set_page_config(page_title="Zundamon TTS WebUI", layout="wide")

# Use session_state to store generated audio data to prevent clearing on rerun
if "audio_bytes" not in st.session_state:
    st.session_state.audio_bytes = None

# Top tabs section, reserved for future features
tabs = st.tabs(["TTS", "Other Features (Coming Soon)"])
with tabs[0]:
    # TTS Module
    st.markdown("<h1 style='text-align: center;'>Zundamon TTS WebUI</h1>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("Please follow the steps below to upload a reference audio file, enter text, select the corresponding languages, and then click the **Generate Speech** button.")

    # First row: Reference Audio File and Target Text (side by side)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Step1 Reference Audio File")
        uploaded_audio = st.file_uploader("Please upload a reference audio file (supports WAV, FLAC, MP3)", type=["wav", "flac", "mp3"])
    with col2:
        st.markdown("### Step3 Target Text")
        default_target_text = "Please enter the text content to generate speech"
        target_text_input = st.text_area("", value=default_target_text, height=150, label_visibility="hidden")

    st.markdown("---")

    # Second row: Reference Text and Language Selection (side by side)
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("### Step2 Reference Text")
        default_ref_text = "Please enter the reference text, or leave blank to use the default."
        ref_text_input = st.text_area("", value=default_ref_text, height=150, label_visibility="hidden")
    with col4:
        st.markdown("### Step4 Language Selection")
        # 修改为支持所有模型支持的语言选项（显示为英文）
        ref_language = st.selectbox("Reference Language", [
            "Chinese",
            "English",
            "Japanese",
            "Cantonese",
            "Korean"
            # "Chinese-English Mixed",
            # "Japanese-English Mixed",
            # "Cantonese-English Mixed",
            # "Korean-English Mixed",
            # "Multilingual Mixed",
            # "Multilingual Mixed (Cantonese)"
        ])
        target_language = st.selectbox("Target Language", [
            "Chinese",
            "English",
            "Japanese",
            "Cantonese",
            "Korean",
            "Chinese-English Mixed",
            "Japanese-English Mixed",
            "Cantonese-English Mixed",
            "Korean-English Mixed"
            # "Multilingual Mixed",
            # "Multilingual Mixed (Cantonese)"
        ])

    # Center the Generate Speech button
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    generate_button = st.button("Generate Speech")
    st.markdown("</div>", unsafe_allow_html=True)

    if generate_button:
        if uploaded_audio is None:
            st.error("Please upload a reference audio file!")
        else:
            try:
                # Save the uploaded reference audio as a temporary file (preserving its original extension)
                audio_suffix = os.path.splitext(uploaded_audio.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=audio_suffix) as tmp_audio:
                    tmp_audio.write(uploaded_audio.read())
                    tmp_audio_path = tmp_audio.name

                # Write the reference text to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode='w', encoding='utf-8') as tmp_ref_text:
                    tmp_ref_text.write(ref_text_input)
                    tmp_ref_text_path = tmp_ref_text.name

                # Write the target text to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode='w', encoding='utf-8') as tmp_target_text:
                    tmp_target_text.write(target_text_input)
                    tmp_target_text_path = tmp_target_text.name

                # Create a temporary output directory
                tmp_output_dir = tempfile.mkdtemp()

                st.info("Generating speech, please wait...")
                # Call the inference function
                synthesize(GPT_MODEL_PATH, SOVITS_MODEL_PATH, tmp_audio_path, tmp_ref_text_path, ref_language,
                           tmp_target_text_path, target_language, tmp_output_dir)

                # The inference function generates an audio file named output.wav
                output_wav_path = os.path.join(tmp_output_dir, "output.wav")
                if os.path.exists(output_wav_path):
                    with open(output_wav_path, "rb") as f:
                        st.session_state.audio_bytes = f.read()

                    st.success("Speech generated successfully!")
                else:
                    st.error("Failed to generate audio. Please check model configuration or logs.")

            except Exception as e:
                st.error(f"An error occurred during inference: {e}")

    # If audio has been generated previously, display the preview and download button
    if st.session_state.audio_bytes is not None:
        st.audio(st.session_state.audio_bytes, format="audio/wav")
        st.download_button("Download Generated Audio", st.session_state.audio_bytes, file_name="output.wav", mime="audio/wav")
