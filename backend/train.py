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


def train_model():
    # Initialize processor
    processor = MidiProcessor()

    # Process MIDI files
    print("Processing MIDI files...")
    token_sequences, stats = processor.process_dataset('data/midi_files')

    if len(token_sequences) == 0:
        print("No MIDI files found! Please add MIDI files to data/midi_files/")
        return

    # Create training sequences
    print("Creating training sequences...")
    X, y = processor.create_sequences(token_sequences, seq_length=32)

    print(f"Training data shape: {X.shape}")

    # Create dataset and dataloader
    dataset = MusicDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = MusicLSTM(vocab_size=processor.vocab_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 50
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0

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
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

    # Save model and processor
    os.makedirs('saved_models', exist_ok=True)
    torch.save(model.state_dict(), 'saved_models/music_model.pth')

    # Save processor for later use
    import pickle
    with open('saved_models/processor.pkl', 'wb') as f:
        pickle.dump(processor, f)

    print("Training completed! Model saved to saved_models/")


if __name__ == "__main__":
    train_model()
