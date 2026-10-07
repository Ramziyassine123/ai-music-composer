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

`train.py` options:

| Option | Default | Meaning |
|---|---|---|
| `--epochs` | 15 | epochs to run (with `--resume`: additional epochs) |
| `--seq-length` | 64 | training window length in 1/8-second steps (64 = 8 s) |
| `--stride` | 32 | step between training windows for major folk tunes; smaller = more data, slower |
| `--minor-stride` | 12 | same, for minor folk tunes (rarer, so denser) |
| `--classical-stride` | 64 | same, for major piano pieces (long, so sparser) |
| `--classical-minor-stride` | 32 | same, for minor piano pieces |
| `--batch-size` | 128 | |
| `--lr` | 0.001 | learning rate |
| `--resume` | off | continue from `saved_models/checkpoint.pth` |

The model is saved after every epoch, so you can stop training with Ctrl+C
and still use the latest checkpoint, then pick up later with
`python train.py --resume --epochs 5`. The API loads the model at startup, so
restart it after training.

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
Put `.mid`/`.midi` files in `backend/data/midi_files/`, and a `data/keys.json`
giving each tune's key (see below). The model learns a single melody line (the
highest note at each 1/8-second step), so melody-only files work best.

**1. Nottingham folk tunes** (~1000 melodies, 96 of them minor):

```bash
git clone --depth 1 https://github.com/jukedeck/nottingham-dataset.git
mkdir -p backend/data/midi_files
cp nottingham-dataset/MIDI/melody/*.mid backend/data/midi_files/

# Read each tune's true key from the ABC files (writes backend/data/keys.json)
cd backend
python label_keys.py ../nottingham-dataset/ABC_cleaned
```

**2. O'Neill's 1850 Irish tunes** (optional, ~1900 more tunes, ~300 of them
minor). These ship with music21 as ABC text, so they are converted to MIDI
with `abc_to_midi.py`. music21 needs a newer numpy than the training
environment, so use a separate virtual environment for this step:

```bash
cd backend
py -3.11 -m venv abcvenv
abcvenv\Scripts\pip install music21
abcvenv\Scripts\python -c "import music21; print(music21.__path__[0])"   # note this path
abcvenv\Scripts\python abc_to_midi.py "<that path>/corpus/oneills1850/*.abc" --prefix oneills
```

This takes about 15 minutes, adds `oneills_*.mid` files and merges their keys
into `data/keys.json`. It skips tunes in modes other than major/minor
(Dorian, Mixolydian) and a few that fail to parse.

Keys matter: training needs to know whether each tune is major or minor, and
without `data/keys.json` it falls back to key detection, which often mistakes
a major tune for its relative minor (about 10% of tunes). Minor tunes are rare
in folk collections, so more minor MIDI files is the best way to improve the
minor melodies.

**3. Romantic piano from MAESTRO** (for the classical style): recorded
performances of Chopin, Schubert, Liszt, Schumann, Brahms, Rachmaninoff and
others. Download `maestro-v3.0.0-midi.zip` (56 MB) from
https://magenta.tensorflow.org/datasets/maestro, unzip it, then:

```bash
cd backend
python prepare_maestro.py path/to/maestro-v3.0.0
```

This keeps the pieces whose title names a single key ("Ballade No. 1 in G
Minor"), skipping sets like "24 Preludes" that have no single key: about 366
performances, 201 of them minor. It copies them to `data/midi_files/` and adds their
keys, marked as classical, to `data/keys.json`. A piece is labelled with its
home key throughout, so pieces that modulate a lot are labelled noisily.

Piano music needs different melody extraction from folk tunes: notes shorter
than 0.09 s (ornaments, trills) are ignored, and gaps of up to 3 steps
between notes are filled, since pianists lift keys during a legato line.

With all three sets (~3300 pieces, ~87k training windows, about half classical
and half folk, 48% minor), 15 epochs on a CPU takes roughly 2 hours.

**Licences:** MAESTRO is CC BY-NC-SA 4.0 (non-commercial, share-alike; cite
Hawthorne et al., "Enabling Factorized Piano Music Modeling and Generation
with the MAESTRO Dataset", ICLR 2019). The O'Neill's tunes are a public-domain
collection transcribed to ABC by volunteers and distributed with music21's
corpus, which asks that you check each collection's licence before commercial
use. None of the data is committed to this repository.

## How melodies are represented
Music is cut into 1/8-second steps, and each step keeps only the highest
sounding note. Each step becomes one token: a **pitch** (a note starts),
**HOLD** (the previous note keeps sounding), or **REST**, plus START/END
markers. The LSTM learns to predict the next token; generation samples one
token at a time.

Every training piece is labelled (or detected) as major or minor and shifted so
its tonic is C. The model also receives the tonality (major/minor) and the
**style** (folk or classical) as extra inputs at every step, so it learns each
once and can write any combination.
Generation happens in C and the result is shifted to the chosen **Key**, so
every melody starts on that key's root.

**Mode** picks the tonality and the scale used by **Stay in key**:

| Mode | Scale | Model trained on |
|---|---|---|
| Major | major | major tunes |
| Natural minor | 1 2 ♭3 4 5 ♭6 ♭7 | minor tunes |
| Harmonic minor | 1 2 ♭3 4 5 ♭6 7 | minor tunes |
| Melodic minor | 1 2 ♭3 4 5 6 7 (ascending form) | minor tunes |
| Chromatic | all 12 notes | minor tunes |

The folk tunes use natural minor almost exclusively, so for folk, harmonic and
melodic minor are scale restrictions only: the model hasn't learned how the
raised 6th/7th are normally used. The classical pieces do use them (about
6-7% of notes), so those modes should sound more natural in the classical
style. Unticking **Stay in key** removes the scale restriction (the model
still follows the chosen tonality).

**Style** picks folk tunes or Romantic piano. Classical melodies start on the
key's root an octave higher, and, being the top line of piano music, cover a
wide range.