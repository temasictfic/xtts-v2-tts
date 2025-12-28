# XTTS v2 Voice Cloning

A Gradio-based web interface for voice cloning using Coqui's XTTS v2 model.

## Features

- Zero-shot voice cloning from short audio samples
- Support for 16 languages
- Adjustable speech speed, temperature, and generation parameters
- Easy-to-use web interface

## Supported Languages

English, Spanish, French, German, Italian, Portuguese, Polish, Turkish, Russian, Dutch, Czech, Arabic, Chinese, Japanese, Korean, Hindi

## Usage in Google Colab

```python
%cd /content
!git clone https://github.com/YOUR_USERNAME/xtts-v2-tts
%cd /content/xtts-v2-tts
!pip install -r requirements.txt
!python app.py
```

## Local Installation

```bash
git clone https://github.com/YOUR_USERNAME/xtts-v2-tts
cd xtts-v2-tts
pip install -r requirements.txt
python app.py
```

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)
- ~6GB VRAM

## Tips for Best Results

- Use clear, noise-free reference audio
- Reference audio should be 6-30 seconds long
- Match the language to your reference speaker
- Adjust temperature for more/less variation

## License

This project uses the Coqui TTS library. Please refer to their license for usage terms.

## Disclaimer

This tool is for research and educational purposes only. Please use responsibly.
