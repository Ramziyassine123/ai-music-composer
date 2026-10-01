# 🎵 AI Music Composer

An AI-powered music composition tool that generates beautiful melodies using machine learning.

## Features

- 🤖 LSTM-based music generation
- 🎹 Real-time MIDI playback
- 💾 Download generated compositions
- 🎛️ Adjustable creativity and length settings
- 🌐 Clean, responsive web interface

## Quick Start

Requires **Python 3.11** (the pinned torch/numpy versions don't support newer
Pythons) and **Node.js**.

### Backend Setup
Run all backend commands from inside `backend/` — paths are relative to it.

```bash
cd backend
py -3.11 -m venv venv           # Windows; use python3.11 on macOS/Linux
venv\Scripts\activate           # Windows; use source venv/bin/activate on macOS/Linux
pip install -r requirements.txt

# Add MIDI files to data/midi_files/ (see Data Setup below)
python train.py                 # Train the model
python -m uvicorn api:app --reload   # Start API server on http://localhost:8000
```

`train.py` options: `--epochs` (default 15), `--batch-size` (128), `--stride`
(4, the step between training windows; 1 uses every position but is ~4x slower),
`--lr` (0.001). The model is saved after every epoch, so you can stop training
early with Ctrl+C and still use the latest checkpoint.

Check `http://localhost:8000/health` shows `"model_loaded": true`. Interactive
API docs are at `http://localhost:8000/docs`.

### Frontend Setup
In a second terminal:

```bash
cd frontend
npm install
npm start                       # Opens http://localhost:3000
```

The dev server proxies API calls to `http://localhost:8000`.

### Data Setup
Put `.mid`/`.midi` files in `backend/data/midi_files/`. The model learns a
single melody line (the highest note at each 1/8-second step), so melody-only
files work best. A good starting set is the Nottingham folk tune dataset
(~1000 melodies):

```bash
git clone --depth 1 https://github.com/jukedeck/nottingham-dataset.git
mkdir -p backend/data/midi_files
cp nottingham-dataset/MIDI/melody/*.mid backend/data/midi_files/
```

On a CPU, 15 epochs over that dataset takes roughly 1–2 hours.