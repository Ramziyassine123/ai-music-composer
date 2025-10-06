# 🎵 AI Music Composer

An AI-powered music composition tool that generates beautiful melodies using machine learning.

## Features

- 🤖 LSTM-based music generation
- 🎹 Real-time MIDI playback
- 💾 Download generated compositions
- 🎛️ Adjustable creativity and length settings
- 🌐 Clean, responsive web interface

## Quick Start

### Backend Setup
```bash
cd backend
pip install -r requirements.txt

# Add MIDI files to data/midi_files/ (see Data Setup below)
python train.py  # Train the model
python -m uvicorn api:app --reload  # Start API server