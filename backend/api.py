"""Web API that turns a trained model into MIDI melodies.

Flow of POST /generate:
  1. Pick the key, mode (major/minor/...) and style from the request.
  2. The model always writes in C (every training piece was shifted to C, see
     midi_utils.py); it is told the tonality and style at every step.
  3. Notes outside the chosen scale are banned while sampling.
  4. The finished token sequence is transposed to the requested key and
     returned as a MIDI file.

The model is loaded once at startup, so restart the server after retraining.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import torch
import pickle
import io
import os
from typing import Literal, Optional
from pydantic import BaseModel, Field
from model import MusicLSTM
from midi_utils import MidiProcessor, tonality_of, STYLES

# Pitch the melody starts on (before transposing to the key): folk tunes sit
# around middle C, the top line of piano music an octave higher
SEED_PITCH = {'folk': 60, 'classical': 72}

app = FastAPI(title="AI Music Composer API")

# Allow any origin so the React dev server (port 3000) can call this API
# directly if it is not going through its proxy. Fine for local development;
# restrict allow_origins before exposing this publicly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and processor
model = None
processor = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GenerationRequest(BaseModel):
    # The limits stop a request from asking for an absurdly long (slow) melody
    # or a temperature that breaks the softmax
    length: int = Field(32, ge=1, le=1024)  # time steps of 0.125s
    temperature: float = Field(1.0, gt=0, le=5.0)
    key: str = "C"  # root note, e.g. "C", "F#", "Bb"; the melody starts on it
    style: Literal["folk", "classical"] = "folk"
    # "major" conditions the model on major pieces; every other mode on minor ones
    # (there are only two tonalities in training). The mode also picks the scale
    # that stay_in_key enforces.
    mode: Literal["major", "minor", "harmonic_minor", "melodic_minor", "chromatic"] = "major"
    stay_in_key: bool = True  # only allow notes from the mode's scale

@app.on_event("startup")
async def load_model():
    global model, processor

    try:
        # The processor holds the tokenizer settings used in training; loading
        # the same one guarantees tokens mean the same thing here
        with open('saved_models/processor.pkl', 'rb') as f:
            processor = pickle.load(f)

        # A model trained with an older token format would load but produce
        # garbage (or fail on shape mismatches), so refuse it with a clear message
        if getattr(processor, 'token_format_version', 1) != MidiProcessor.TOKEN_FORMAT_VERSION:
            print("Saved model uses an old token format! Please retrain: python train.py")
            model = None
            processor = None
            return

        # Load model
        model = MusicLSTM(vocab_size=processor.vocab_size)
        model.load_state_dict(torch.load('saved_models/music_model.pth',
                                         map_location=device))
        model.to(device)
        model.eval()

        print("Model loaded successfully!")

    except FileNotFoundError:
        print("Model not found! Please train the model first.")
        model = None
        processor = None
    except Exception as e:
        print(f"Failed to load model ({e}). Please retrain: python train.py")
        model = None
        processor = None

@app.get("/")
async def root():
    return {"message": "AI Music Composer API", "status": "running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device)
    }

@app.post("/generate")
async def generate_music(request: GenerationRequest):
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        root = processor.key_root(request.key)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    try:
        # The model writes in C and the result is transposed to the key, so
        # the melody starts on the tonic (C4 shifted by at most a tritone)
        transpose = processor.shift_for_root(root)
        start_sequence = [processor.START_TOKEN,
                          processor.pitch_to_token(SEED_PITCH[request.style])]

        # Sampling is steered by banning tokens (their probability becomes 0):
        # START/END so the melody runs the full length, notes that would leave
        # the piano range once transposed, and (if stay_in_key) every note
        # outside the scale. The scale is built in C because the model writes
        # in C; transposing afterwards moves it to the requested key.
        banned = [processor.START_TOKEN, processor.END_TOKEN]
        banned += processor.out_of_range_tokens(transpose)
        if request.stay_in_key:
            banned += processor.out_of_key_tokens('C', request.mode)

        generated_tokens = model.generate(
            start_sequence=start_sequence,
            length=request.length,
            temperature=request.temperature,
            device=device,
            banned_tokens=banned,
            mode=tonality_of(request.mode),
            style=STYLES.index(request.style)
        )

        # Convert to MIDI
        midi = processor.tokens_to_midi(generated_tokens, transpose=transpose)

        # Convert MIDI to bytes
        midi_bytes = io.BytesIO()
        midi.write(midi_bytes)
        midi_bytes.seek(0)

        return Response(
            content=midi_bytes.getvalue(),
            media_type="audio/midi",
            headers={"Content-Disposition": "attachment; filename=generated.mid"}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.get("/vocab_info")
async def get_vocab_info():
    if processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "vocab_size": processor.vocab_size,
        "pitch_range": f"{processor.min_pitch}-{processor.max_pitch}",
        "special_tokens": {
            "REST": processor.REST_TOKEN,
            "START": processor.START_TOKEN,
            "END": processor.END_TOKEN,
            "HOLD": processor.HOLD_TOKEN
        }
    }