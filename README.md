# Sign Language Translator - Arabic & English

This project detects both **Arabic** and **English** sign language gestures in real-time using a **CNN-LSTM deep learning model**, and converts them into **text** and **speech**.

## Features

- ✅ Real-time hand tracking using **MediaPipe**
- ✅ Detects **Arabic** and **English** sign languages
- ✅ Converts recognized signs into **text**
- ✅ Converts recognized text into **speech** (Text-to-Speech)
- ✅ Handles repeated letters correctly
- ✅ Supports clearing text and speaking on-demand

## Architecture

The model uses a **CNN-LSTM** architecture:

1. **CNN** layers extract spatial features from hand landmarks.
2. **LSTM** layers capture temporal sequences for gesture recognition.
3. Output layer predicts the sign among **Arabic and English classes**.

Achieved **99.6% accuracy** for English sign language.

## Requirements

- Python 3.8+
- TensorFlow
- OpenCV
- MediaPipe
- pyttsx3
- Pillow
- arabic_reshaper
- python-bidi
- numpy
- scikit-learn

Install dependencies with:

```bash
pip install -r requirements.txt