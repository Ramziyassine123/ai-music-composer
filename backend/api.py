from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import torch
import pickle
import io
import os
from typing import Optional
from pydantic import BaseModel, Field
from model import MusicLSTM

app = FastAPI(title="AI Music Composer API")

# Enable CORS
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
    length: int = Field(32, ge=1, le=1024)  # time steps of 0.125s
    temperature: float = Field(1.0, gt=0, le=5.0)
    key: str = "C"  # For future use

@app.on_event("startup")
async def load_model():
    global model, processor

    try:
        # Load processor
        with open('saved_models/processor.pkl', 'rb') as f:
            processor = pickle.load(f)

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
        # Create a simple starting sequence
        c4_token = 60 - processor.min_pitch + processor.SPECIAL_TOKENS
        start_sequence = [processor.START_TOKEN, c4_token]  # Start with C4

        # Generate sequence; START/END are banned so the melody runs the full length
        generated_tokens = model.generate(
            start_sequence=start_sequence,
            length=request.length,
            temperature=request.temperature,
            device=device,
            banned_tokens=[processor.START_TOKEN, processor.END_TOKEN]
        )

        # Convert to MIDI
        midi = processor.tokens_to_midi(generated_tokens)

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
            "END": processor.END_TOKEN
        }
    }