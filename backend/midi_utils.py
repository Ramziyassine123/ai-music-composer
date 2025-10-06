import pretty_midi
import numpy as np
import os
from typing import List, Tuple, Dict
import pickle


class MidiProcessor:
    def __init__(self, min_pitch: int = 21, max_pitch: int = 108):
        self.min_pitch = min_pitch  # A0
        self.max_pitch = max_pitch  # C8
        self.pitch_range = max_pitch - min_pitch + 1

        # Special tokens
        self.REST_TOKEN = 0
        self.START_TOKEN = 1
        self.END_TOKEN = 2
        self.SPECIAL_TOKENS = 3

        self.vocab_size = self.pitch_range + self.SPECIAL_TOKENS

    def midi_to_tokens(self, midi_path: str, time_step: float = 0.125) -> List[int]:
        """Convert MIDI file to sequence of tokens"""
        try:
            midi_data = pretty_midi.PrettyMIDI(midi_path)

            # Get piano roll representation
            piano_roll = midi_data.get_piano_roll(fs=1 / time_step)

            # Convert to token sequence
            tokens = [self.START_TOKEN]

            for time_step_idx in range(piano_roll.shape[1]):
                # Find active notes at this time step
                active_notes = np.where(piano_roll[:, time_step_idx] > 0)[0]

                if len(active_notes) == 0:
                    tokens.append(self.REST_TOKEN)
                else:
                    # Take the highest note (melody line)
                    note = active_notes[-1]
                    if self.min_pitch <= note <= self.max_pitch:
                        token = note - self.min_pitch + self.SPECIAL_TOKENS
                        tokens.append(token)
                    else:
                        tokens.append(self.REST_TOKEN)

            tokens.append(self.END_TOKEN)
            return tokens

        except Exception as e:
            print(f"Error processing {midi_path}: {e}")
            return []

    def tokens_to_midi(self, tokens: List[int], tempo: int = 120,
                       time_step: float = 0.125) -> pretty_midi.PrettyMIDI:
        """Convert token sequence back to MIDI"""
        midi = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)  # Piano

        current_time = 0.0

        for token in tokens:
            if token == self.START_TOKEN or token == self.END_TOKEN:
                continue
            elif token == self.REST_TOKEN:
                current_time += time_step
            else:
                # Convert token back to pitch
                pitch = token - self.SPECIAL_TOKENS + self.min_pitch
                if self.min_pitch <= pitch <= self.max_pitch:
                    note = pretty_midi.Note(
                        velocity=80,
                        pitch=int(pitch),
                        start=current_time,
                        end=current_time + time_step
                    )
                    instrument.notes.append(note)
                current_time += time_step

        midi.instruments.append(instrument)
        return midi

    def process_dataset(self, midi_folder: str) -> Tuple[List[List[int]], Dict]:
        """Process all MIDI files in folder"""
        all_tokens = []
        stats = {'processed': 0, 'failed': 0, 'total_tokens': 0}

        for filename in os.listdir(midi_folder):
            if filename.endswith('.mid') or filename.endswith('.midi'):
                file_path = os.path.join(midi_folder, filename)
                tokens = self.midi_to_tokens(file_path)

                if len(tokens) > 10:  # Minimum length filter
                    all_tokens.append(tokens)
                    stats['processed'] += 1
                    stats['total_tokens'] += len(tokens)
                else:
                    stats['failed'] += 1

        print(f"Processed {stats['processed']} files, failed {stats['failed']}")
        print(f"Total tokens: {stats['total_tokens']}")

        return all_tokens, stats

    def create_sequences(self, token_sequences: List[List[int]],
                         seq_length: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        """Create training sequences from token data"""
        X, y = [], []

        for tokens in token_sequences:
            for i in range(len(tokens) - seq_length):
                X.append(tokens[i:i + seq_length])
                y.append(tokens[i + 1:i + seq_length + 1])

        return np.array(X), np.array(y)
