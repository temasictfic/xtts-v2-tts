# XTTS v2 Voice Cloning

A Gradio-based web interface for voice cloning using XTTS v2 model.

Uses the community-maintained [coqui-tts](https://github.com/idiap/coqui-ai-TTS) fork that supports Python 3.12.

## Features

- Zero-shot voice cloning from short audio samples
- Support for 17 languages
- Easy-to-use web interface

## Supported Languages

English, Spanish, French, German, Italian, Portuguese, Polish, Turkish, Russian, Dutch, Czech, Arabic, Chinese, Japanese, Korean, Hindi, Hungarian

## Usage in Google Colab (GPU Runtime Required)

```python
%cd /content
!git clone https://github.com/YOUR_USERNAME/xtts-v2-tts
%cd /content/xtts-v2-tts
!pip install coqui-tts gradio pydub --quiet
!python app.py
```

**Note:** Make sure to select a GPU runtime (Runtime > Change runtime type > T4 GPU)

## Local Installation

```bash
git clone https://github.com/YOUR_USERNAME/xtts-v2-tts
cd xtts-v2-tts
pip install -r requirements.txt
python app.py
```

## Requirements

- Python 3.9-3.12
- CUDA-compatible GPU (recommended)
- ~6GB VRAM

## Tips for Best Results

- Use clear, noise-free reference audio
- Reference audio should be 6-30 seconds long
- Match the language to your reference speaker

## License

This project uses the coqui-tts library (Idiap fork). Please refer to their license for usage terms.

## Disclaimer

This tool is for research and educational purposes only. Please use responsibly.
