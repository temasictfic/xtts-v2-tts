import torch
import os
import gradio as gr
import traceback
import gc
import numpy as np
import tempfile
from pydub import AudioSegment
from pydub.effects import normalize
from TTS.api import TTS

# Available languages for XTTS v2
LANGUAGES = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Polish": "pl",
    "Turkish": "tr",
    "Russian": "ru",
    "Dutch": "nl",
    "Czech": "cs",
    "Arabic": "ar",
    "Chinese": "zh-cn",
    "Japanese": "ja",
    "Korean": "ko",
    "Hindi": "hi",
    "Hungarian": "hu",
}

# Global model instance
tts_model = None

def initialize_model():
    """Initialize XTTS v2 model."""
    global tts_model
    print("Initializing XTTS v2 model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    print(f"Model loaded successfully on {device}!")
    return tts_model

# Initialize model on startup
initialize_model()

def reset_model():
    """Reset the TTS model to recover from errors."""
    global tts_model
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print("Reinitializing XTTS v2 model...")
        initialize_model()
        print("Model reinitialized successfully!")
        return True
    except Exception as e:
        print(f"Failed to reinitialize model: {e}")
        return False

def cleanup_memory():
    """Clean up GPU and system memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def preprocess_audio(audio_path, target_sr=22050, max_duration=30):
    """Preprocess audio for voice cloning."""
    try:
        # Load with pydub for robust format handling
        audio = AudioSegment.from_file(audio_path)

        # Convert to mono if stereo
        if audio.channels > 1:
            audio = audio.set_channels(1)

        # Limit duration to prevent memory issues
        if len(audio) > max_duration * 1000:
            audio = audio[:max_duration * 1000]

        # Normalize audio to prevent clipping
        audio = normalize(audio)

        # Convert to target sample rate
        audio = audio.set_frame_rate(target_sr)

        # Export to temporary WAV file
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, "reference_processed.wav")
        audio.export(
            temp_path,
            format="wav",
            parameters=["-acodec", "pcm_s16le", "-ac", "1", "-ar", str(target_sr)]
        )

        return temp_path

    except Exception as e:
        print(f"Audio preprocessing failed: {e}")
        raise ValueError(f"Failed to process audio: {str(e)}")

def generate_speech(reference_audio, text, language):
    """Generate speech with voice cloning using XTTS v2."""
    global tts_model

    if not reference_audio or not text:
        gr.Warning("Please provide both reference audio and text to generate.")
        return None

    try:
        print(f"Generating speech: '{text[:50]}...' in {language}")

        # Check CUDA availability
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"CUDA device: {torch.cuda.get_device_name()}")

        # Preprocess reference audio
        try:
            processed_audio_path = preprocess_audio(reference_audio)
        except Exception as audio_error:
            gr.Warning(f"Audio preprocessing failed: {str(audio_error)}")
            return None

        # Get language code
        lang_code = LANGUAGES.get(language, "en")

        # Create temporary output file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            output_path = tmp_file.name

        # Generate speech with XTTS v2
        try:
            tts_model.tts_to_file(
                text=text,
                speaker_wav=processed_audio_path,
                language=lang_code,
                file_path=output_path,
            )

            cleanup_memory()
            return output_path

        except RuntimeError as cuda_error:
            if "CUDA" in str(cuda_error):
                print(f"CUDA error detected: {cuda_error}")
                if reset_model():
                    gr.Warning("CUDA error occurred. Model has been reset. Please try again.")
                else:
                    gr.Warning("CUDA error occurred and model reset failed. Please restart the application.")
                return None
            else:
                raise cuda_error

    except Exception as e:
        traceback.print_exc()
        gr.Warning(f"Speech generation failed: {str(e)}")
        cleanup_memory()
        return None


# Build Gradio Interface
with gr.Blocks(title="XTTS v2 Voice Cloning", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# XTTS v2 Voice Cloning")
    gr.Markdown("""
    **XTTS v2** is a powerful multilingual text-to-speech model with zero-shot voice cloning capabilities.

    Using the community-maintained [coqui-tts](https://github.com/idiap/coqui-ai-TTS) fork.

    Upload a reference audio clip (at least 6 seconds recommended) and enter text to generate speech with the cloned voice.

    **Supported Languages:** English, Spanish, French, German, Italian, Portuguese, Polish, Turkish, Russian, Dutch, Czech, Arabic, Chinese, Japanese, Korean, Hindi, Hungarian
    """)
    gr.Markdown("**Please use responsibly for research and educational purposes only!**")

    with gr.Row():
        with gr.Column(scale=1):
            reference_audio = gr.Audio(
                label="Reference Audio (6+ seconds recommended)",
                type="filepath",
                sources=["upload", "microphone"]
            )
            text_input = gr.Textbox(
                label="Text to Generate",
                placeholder="Enter the text you want to synthesize...",
                lines=4
            )
            language = gr.Dropdown(
                label="Language",
                choices=list(LANGUAGES.keys()),
                value="English"
            )

            generate_btn = gr.Button("Generate Speech", variant="primary", size="lg")

        with gr.Column(scale=1):
            output_audio = gr.Audio(label="Generated Audio", type="filepath")

            gr.Markdown("""
            ### Tips for Best Results:
            - Use **clear, noise-free** reference audio
            - Reference audio should be **6-30 seconds** long
            - Match the **language** to your reference speaker
            - Single speaker, minimal background noise
            """)

    generate_btn.click(
        fn=generate_speech,
        inputs=[reference_audio, text_input, language],
        outputs=[output_audio]
    )

if __name__ == '__main__':
    demo.launch(server_name='0.0.0.0', server_port=7860, debug=True, share=True)
