import json
import pretty_midi
import numpy as np
import os
from typing import List, Tuple, Dict, Optional
import pickle

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
FLAT_NAMES = {'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#'}

# Scales the generator can be restricted to (semitones above the key's root)
SCALES = {
    'major': [0, 2, 4, 5, 7, 9, 11],
    'minor': [0, 2, 3, 5, 7, 8, 10],           # natural minor
    'harmonic_minor': [0, 2, 3, 5, 7, 8, 11],
    'melodic_minor': [0, 2, 3, 5, 7, 9, 11],   # ascending form
    'chromatic': list(range(12)),
}

# The model is conditioned on one of two tonalities
MAJOR, MINOR = 0, 1

# ... and on a style (the index is the id the model sees)
STYLES = ['folk', 'classical']
# Piano performances are full of ornaments and trills; notes shorter than this
# (seconds) are ignored when pulling out the melody line
STYLE_MIN_NOTE = {'folk': 0.0, 'classical': 0.09}
# Pianists lift keys between notes of a legato line (the pedal carries the
# sound), so a rest of at most this many steps between two notes becomes a hold
STYLE_FILL_GAP = {'folk': 0, 'classical': 3}

# Krumhansl-Schmuckler key profiles, used only when a tune has no known key
_KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def tonality_of(scale_mode: str) -> int:
    """Which tonality conditions the model for a scale: major, else minor"""
    return MAJOR if scale_mode == 'major' else MINOR


class MidiProcessor:
    # Bump when the token format or model inputs change, so stale saved
    # processors and checkpoints are rejected
    TOKEN_FORMAT_VERSION = 4

    def __init__(self, min_pitch: int = 21, max_pitch: int = 108):
        self.min_pitch = min_pitch  # A0
        self.max_pitch = max_pitch  # C8
        self.pitch_range = max_pitch - min_pitch + 1
        self.token_format_version = self.TOKEN_FORMAT_VERSION

        # Special tokens
        self.REST_TOKEN = 0
        self.START_TOKEN = 1
        self.END_TOKEN = 2
        self.HOLD_TOKEN = 3  # previous note keeps sounding
        self.SPECIAL_TOKENS = 4

        self.vocab_size = self.pitch_range + self.SPECIAL_TOKENS

    def pitch_to_token(self, pitch: int) -> int:
        return pitch - self.min_pitch + self.SPECIAL_TOKENS

    def token_to_pitch(self, token: int) -> int:
        return token - self.SPECIAL_TOKENS + self.min_pitch

    @staticmethod
    def key_root(key: str) -> int:
        """Pitch class (0-11) of a key name like 'C', 'F#' or 'Bb'"""
        name = key.strip()
        name = name[:1].upper() + name[1:]
        name = FLAT_NAMES.get(name, name)
        if name not in NOTE_NAMES:
            raise ValueError(f"Unknown key: {key}")
        return NOTE_NAMES.index(name)

    @staticmethod
    def shift_for_root(root: int) -> int:
        """Smallest transposition (-5..+6 semitones) that moves C to the root"""
        return root if root <= 6 else root - 12

    def out_of_key_tokens(self, key: str, mode: str = 'major') -> List[int]:
        """Pitch tokens whose notes are outside the key's scale"""
        if mode not in SCALES:
            raise ValueError(f"Unknown mode: {mode}")
        root = self.key_root(key)
        scale = {(root + step) % 12 for step in SCALES[mode]}
        return [self.pitch_to_token(p)
                for p in range(self.min_pitch, self.max_pitch + 1)
                if p % 12 not in scale]

    def out_of_range_tokens(self, transpose: int) -> List[int]:
        """Pitch tokens that would leave the piano range once transposed"""
        return [self.pitch_to_token(p)
                for p in range(self.min_pitch, self.max_pitch + 1)
                if not self.min_pitch <= p + transpose <= self.max_pitch]

    @staticmethod
    def detect_key(pitch_class_weights: np.ndarray) -> Tuple[int, int]:
        """Best-guess (tonic pitch class, MAJOR/MINOR) from note durations.

        Unreliable (often picks the relative minor of a major tune), so
        known keys from keys.json are preferred.
        """
        best = None
        for tonic in range(12):
            for mode, profile in ((MAJOR, _KS_MAJOR), (MINOR, _KS_MINOR)):
                corr = np.corrcoef(pitch_class_weights, np.roll(profile, tonic))[0, 1]
                if best is None or corr > best[0]:
                    best = (corr, tonic, mode)
        return best[1], best[2]

    def midi_to_tokens(self, midi_path: str, time_step: float = 0.125,
                       key: Optional[Tuple[int, int]] = None,
                       style: int = 0) -> Tuple[List[int], int]:
        """Convert a MIDI file to (tokens, tonality), transposed to C.

        The tune is shifted so its tonic lands on C (key is a known
        (tonic pitch class, MAJOR/MINOR), else it is detected), so the model
        learns each tonality once and generation transposes to any key.

        Each time step is the highest sounding note (the melody line):
        its pitch token on the step the note starts, HOLD while it keeps
        sounding, or REST when nothing is playing.
        """
        try:
            midi_data = pretty_midi.PrettyMIDI(midi_path)
            min_length = STYLE_MIN_NOTE[STYLES[style]]
            notes = [note for instrument in midi_data.instruments
                     if not instrument.is_drum for note in instrument.notes
                     if note.end - note.start >= min_length]

            if key is None:
                weights = np.zeros(12)
                for note in notes:
                    weights[note.pitch % 12] += note.end - note.start
                key = self.detect_key(weights)
            tonic, tonality = key
            shift = -self.shift_for_root(tonic)

            num_steps = int(np.ceil(midi_data.get_end_time() / time_step))
            top_pitch = np.full(num_steps, -1)
            is_onset = np.zeros(num_steps, dtype=bool)

            for note in notes:
                pitch = note.pitch + shift
                if not self.min_pitch <= pitch <= self.max_pitch:
                    continue
                start = int(round(note.start / time_step))
                end = max(start + 1, int(round(note.end / time_step)))
                for step in range(start, min(end, num_steps)):
                    if pitch > top_pitch[step]:
                        top_pitch[step] = pitch
                        is_onset[step] = step == start
                    elif pitch == top_pitch[step] and step == start:
                        is_onset[step] = True  # re-struck same pitch

            fill_gap = STYLE_FILL_GAP[STYLES[style]]
            step = 0
            while fill_gap and step < num_steps:
                if top_pitch[step] < 0 and step > 0 and top_pitch[step - 1] >= 0:
                    end = step
                    while end < num_steps and top_pitch[end] < 0:
                        end += 1
                    if end < num_steps and end - step <= fill_gap:
                        top_pitch[step:end] = top_pitch[step - 1]
                    step = end
                else:
                    step += 1

            # Convert to token sequence
            tokens = [self.START_TOKEN]
            previous = -1

            for step in range(num_steps):
                pitch = top_pitch[step]
                if pitch < 0:
                    tokens.append(self.REST_TOKEN)
                elif is_onset[step] or pitch != previous:
                    tokens.append(self.pitch_to_token(int(pitch)))
                else:
                    tokens.append(self.HOLD_TOKEN)
                previous = pitch

            tokens.append(self.END_TOKEN)
            return tokens, tonality

        except Exception as e:
            print(f"Error processing {midi_path}: {e}")
            return [], MAJOR

    def tokens_to_midi(self, tokens: List[int], tempo: int = 120,
                       time_step: float = 0.125,
                       transpose: int = 0) -> pretty_midi.PrettyMIDI:
        """Convert token sequence back to MIDI, shifted by `transpose` semitones"""
        midi = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)  # Piano

        current_time = 0.0
        current_note = None

        for token in tokens:
            if token == self.START_TOKEN or token == self.END_TOKEN:
                continue

            if token == self.HOLD_TOKEN:
                # Extend the sounding note; a HOLD after a rest stays silent
                if current_note is not None:
                    current_note.end = current_time + time_step
            elif token == self.REST_TOKEN:
                current_note = None
            else:
                pitch = self.token_to_pitch(token) + transpose
                if self.min_pitch <= pitch <= self.max_pitch:
                    current_note = pretty_midi.Note(
                        velocity=80,
                        pitch=int(pitch),
                        start=current_time,
                        end=current_time + time_step
                    )
                    instrument.notes.append(current_note)
                else:
                    current_note = None

            current_time += time_step

        midi.instruments.append(instrument)
        return midi


    @staticmethod
    def load_key_labels(path: str) -> Dict[str, Tuple[int, int, int]]:
        """Read {filename: [key name, 'major'|'minor'(, style)]} into
        {filename: (tonic pitch class, MAJOR/MINOR, style id)}.

        Written by label_keys.py, abc_to_midi.py and prepare_maestro.py; a
        missing style means 'folk'.
        """
        if not os.path.exists(path):
            return {}
        with open(path, encoding='utf-8') as f:
            raw = json.load(f)
        labels = {}
        for name, entry in raw.items():
            key, mode = entry[0], entry[1]
            style = entry[2] if len(entry) > 2 else 'folk'
            labels[name] = (MidiProcessor.key_root(key),
                            MINOR if mode == 'minor' else MAJOR,
                            STYLES.index(style))
        return labels

    def process_dataset(self, midi_folder: str,
                        labels: Optional[Dict[str, Tuple[int, int, int]]] = None
                        ) -> Tuple[List[List[int]], List[int], List[int], Dict]:
        """Process all MIDI files in folder.

        Returns (token sequences, tonality of each, style of each, stats).
        """
        labels = labels or {}
        all_tokens, tonalities, styles = [], [], []
        stats = {'processed': 0, 'failed': 0, 'total_tokens': 0,
                 'labeled': 0, 'detected': 0, 'major': 0, 'minor': 0,
                 **{style: 0 for style in STYLES}}

        for filename in sorted(os.listdir(midi_folder)):
            if filename.lower().endswith(('.mid', '.midi')):
                file_path = os.path.join(midi_folder, filename)
                label = labels.get(filename)
                key = label[:2] if label else None
                style = label[2] if label else 0
                tokens, tonality = self.midi_to_tokens(file_path, key=key, style=style)

                if len(tokens) > 10:  # Minimum length filter
                    all_tokens.append(tokens)
                    tonalities.append(tonality)
                    styles.append(style)
                    stats['processed'] += 1
                    stats['total_tokens'] += len(tokens)
                    stats['labeled' if label else 'detected'] += 1
                    stats['minor' if tonality == MINOR else 'major'] += 1
                    stats[STYLES[style]] += 1
                else:
                    stats['failed'] += 1

        print(f"Processed {stats['processed']} files, failed {stats['failed']}")
        print(f"Total tokens: {stats['total_tokens']}")
        print(f"Keys: {stats['labeled']} from labels, {stats['detected']} detected "
              f"| {stats['major']} major, {stats['minor']} minor")
        print("Styles: " + ", ".join(f"{stats[s]} {s}" for s in STYLES))

        return all_tokens, tonalities, styles, stats

    def create_sequences(self, token_sequences: List[List[int]],
                         tonalities: List[int], styles: List[int],
                         seq_length: int = 32,
                         strides: Optional[Dict[Tuple[int, int], int]] = None,
                         default_stride: int = 1
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create training windows (inputs, targets, tonality, style of each).

        strides maps (style id, tonality) to the step between windows, so rare
        kinds of music (minor tunes) can be sampled more densely and long
        pieces (piano performances) less densely.
        """
        strides = strides or {}
        X, y, modes, style_ids = [], [], [], []

        for tokens, tonality, style in zip(token_sequences, tonalities, styles):
            step = strides.get((style, tonality), default_stride)
            for i in range(0, len(tokens) - seq_length, step):
                X.append(tokens[i:i + seq_length])
                y.append(tokens[i + 1:i + seq_length + 1])
                modes.append(tonality)
                style_ids.append(style)

        return np.array(X), np.array(y), np.array(modes), np.array(style_ids)
