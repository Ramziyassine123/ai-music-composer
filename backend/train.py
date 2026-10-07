"""Train the music LSTM on the MIDI files in data/midi_files/.

Each file is turned into tokens (midi_utils.py), shifted so its tonic is C,
and cut into fixed-length windows. The model learns to predict, at every
position in a window, the token that comes next. It is also told whether
the piece is major or minor and whether it is folk or classical.

data/keys.json (written by label_keys.py, abc_to_midi.py and
prepare_maestro.py) supplies each piece's key and style. Run this from
backend/: all paths are relative to it.
"""
import argparse
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from model import MusicLSTM
from midi_utils import MidiProcessor, STYLES, MAJOR, MINOR

MODEL_PATH = 'saved_models/music_model.pth'      # weights only, loaded by api.py
CHECKPOINT_PATH = 'saved_models/checkpoint.pth'  # weights + optimizer, for --resume
PROCESSOR_PATH = 'saved_models/processor.pkl'
KEY_LABELS_PATH = 'data/keys.json'               # written by label_keys.py


class MusicDataset(Dataset):
    def __init__(self, sequences_x, sequences_y, modes, styles):
        self.x = torch.tensor(sequences_x, dtype=torch.long)
        self.y = torch.tensor(sequences_y, dtype=torch.long)
        self.modes = torch.tensor(modes, dtype=torch.long)
        self.styles = torch.tensor(styles, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.modes[idx], self.styles[idx]


def train_model(epochs: int = 15, batch_size: int = 128, stride: int = 32,
                minor_stride: int = 12, classical_stride: int = 64,
                classical_minor_stride: int = 32, seq_length: int = 64,
                lr: float = 0.001, resume: bool = False):
    # Initialize processor
    processor = MidiProcessor()

    # Load the checkpoint first so an incompatible one fails before preprocessing
    checkpoint = None
    if resume:
        if not os.path.exists(CHECKPOINT_PATH):
            print(f"No checkpoint at {CHECKPOINT_PATH}; train without --resume first.")
            return
        checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
        if (checkpoint.get('vocab_size') != processor.vocab_size
                or checkpoint.get('format') != MidiProcessor.TOKEN_FORMAT_VERSION):
            print("Checkpoint uses a different token format; train without --resume.")
            return

    # Known keys (from label_keys.py) are far more reliable than detection
    labels = MidiProcessor.load_key_labels(KEY_LABELS_PATH)
    if not labels:
        print(f"No {KEY_LABELS_PATH}: keys will be detected, which often mistakes "
              "major tunes for their relative minor. See README (Data Setup).")

    # Process MIDI files
    print("Processing MIDI files...")
    token_sequences, tonalities, styles, stats = processor.process_dataset(
        'data/midi_files', labels)

    if len(token_sequences) == 0:
        print("No MIDI files found! Please add MIDI files to data/midi_files/")
        return

    # Create training sequences. A larger stride skips heavily overlapping
    # windows: piano pieces are long, so they get a larger stride, while minor
    # music is rarer, so it gets a smaller one (= more windows)
    folk, classical = STYLES.index('folk'), STYLES.index('classical')
    strides = {(folk, MAJOR): stride, (folk, MINOR): minor_stride,
               (classical, MAJOR): classical_stride,
               (classical, MINOR): classical_minor_stride}
    print("Creating training sequences...")
    X, y, modes, style_ids = processor.create_sequences(
        token_sequences, tonalities, styles, seq_length=seq_length,
        strides=strides)

    print(f"Training data shape: {X.shape} "
          f"({int(modes.sum())} minor windows, {int(len(modes) - modes.sum())} major; "
          f"{int((style_ids == classical).sum())} classical, "
          f"{int((style_ids == folk).sum())} folk)")

    # Create dataset and dataloader
    dataset = MusicDataset(X, y, modes, style_ids)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = MusicLSTM(vocab_size=processor.vocab_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    start_epoch = 0
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        print(f"Resuming after epoch {start_epoch} "
              f"(last loss {checkpoint['loss']:.4f})")

    # The learning rate is one Adam step size for all weights. Cross-entropy
    # measures how much probability the model gave to the true next token
    # (lower is better; ~4.5 would be random guessing over 92 tokens).

    # Save processor up front; the model is saved after every epoch so
    # training can be stopped at any point with a usable checkpoint
    os.makedirs('saved_models', exist_ok=True)
    with open(PROCESSOR_PATH, 'wb') as f:
        pickle.dump(processor, f)

    # Training loop
    model.train()

    for epoch in range(start_epoch, start_epoch + epochs):
        total_loss = 0
        start = time.time()

        for batch_idx, (data, targets, mode, style) in enumerate(dataloader):
            data, targets = data.to(device), targets.to(device)
            mode, style = mode.to(device), style.to(device)

            # Forward pass
            outputs, _ = model(data, mode=mode, style=style)
            loss = criterion(outputs.reshape(-1, processor.vocab_size),
                             targets.reshape(-1))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch + 1}/{start_epoch + epochs}], Loss: {avg_loss:.4f}, '
              f'Time: {time.time() - start:.0f}s', flush=True)

        # Two files on purpose: the weights alone are what the API loads, while
        # the checkpoint also keeps the optimizer state, which --resume needs
        # to continue training exactly where it stopped
        torch.save(model.state_dict(), MODEL_PATH)
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'loss': avg_loss,
            'vocab_size': processor.vocab_size,
            'format': MidiProcessor.TOKEN_FORMAT_VERSION,
        }, CHECKPOINT_PATH)

    print("Training completed! Model saved to saved_models/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the music LSTM")
    parser.add_argument('--epochs', type=int, default=15,
                        help="epochs to run (with --resume: additional epochs)")
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--seq-length', type=int, default=64,
                        help="training window length in 1/8-second steps")
    parser.add_argument('--stride', type=int, default=32,
                        help="step between training windows (1 = every position)")
    parser.add_argument('--minor-stride', type=int, default=12,
                        help="stride for minor folk tunes; smaller = more minor windows")
    parser.add_argument('--classical-stride', type=int, default=64,
                        help="stride for major piano pieces (long, so sparser)")
    parser.add_argument('--classical-minor-stride', type=int, default=32,
                        help="stride for minor piano pieces")
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--resume', action='store_true',
                        help="continue from saved_models/checkpoint.pth")
    args = parser.parse_args()

    train_model(epochs=args.epochs, batch_size=args.batch_size,
                stride=args.stride, minor_stride=args.minor_stride,
                classical_stride=args.classical_stride,
                classical_minor_stride=args.classical_minor_stride,
                seq_length=args.seq_length, lr=args.lr, resume=args.resume)
