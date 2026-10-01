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
from midi_utils import MidiProcessor


class MusicDataset(Dataset):
    def __init__(self, sequences_x, sequences_y):
        self.x = torch.tensor(sequences_x, dtype=torch.long)
        self.y = torch.tensor(sequences_y, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def train_model(epochs: int = 15, batch_size: int = 128, stride: int = 4,
                seq_length: int = 32, lr: float = 0.001):
    # Initialize processor
    processor = MidiProcessor()

    # Process MIDI files
    print("Processing MIDI files...")
    token_sequences, stats = processor.process_dataset('data/midi_files')

    if len(token_sequences) == 0:
        print("No MIDI files found! Please add MIDI files to data/midi_files/")
        return

    # Create training sequences (stride > 1 skips heavily overlapping windows)
    print("Creating training sequences...")
    X, y = processor.create_sequences(token_sequences, seq_length=seq_length,
                                      stride=stride)

    print(f"Training data shape: {X.shape}")

    # Create dataset and dataloader
    dataset = MusicDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = MusicLSTM(vocab_size=processor.vocab_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Save processor up front; the model is saved after every epoch so
    # training can be stopped at any point with a usable checkpoint
    os.makedirs('saved_models', exist_ok=True)
    with open('saved_models/processor.pkl', 'wb') as f:
        pickle.dump(processor, f)

    # Training loop
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        start = time.time()

        for batch_idx, (data, targets) in enumerate(dataloader):
            data, targets = data.to(device), targets.to(device)

            # Forward pass
            outputs, _ = model(data)
            loss = criterion(outputs.reshape(-1, processor.vocab_size),
                             targets.reshape(-1))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}, '
              f'Time: {time.time() - start:.0f}s', flush=True)

        torch.save(model.state_dict(), 'saved_models/music_model.pth')

    print("Training completed! Model saved to saved_models/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the music LSTM")
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--stride', type=int, default=4,
                        help="step between training windows (1 = every position)")
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()

    train_model(epochs=args.epochs, batch_size=args.batch_size,
                stride=args.stride, lr=args.lr)
