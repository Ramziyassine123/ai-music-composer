"""The neural network: an LSTM that predicts the next music token.

Pipeline: token -> embedding (+ tonality embedding + style embedding) ->
2-layer LSTM -> one score per token. An LSTM reads a sequence step by step
and carries a memory of what it has seen, which is how it can keep a key,
rhythm and phrase in mind.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional

class MusicLSTM(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int = 128,
                 hidden_size: int = 256, num_layers: int = 2, dropout: float = 0.3,
                 num_modes: int = 2, num_styles: int = 2):
        super(MusicLSTM, self).__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_size)
        # Tonality (major/minor) and style (folk/classical) are added to every
        # step's input, so the model keeps them in mind however long the
        # melody runs
        self.mode_embedding = nn.Embedding(num_modes, embed_size)
        self.style_embedding = nn.Embedding(num_styles, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden=None, mode=None, style=None):
        # x: (batch, steps) of token ids; mode/style: (batch,) of ids.
        # Adding the mode/style embedding to every step (rather than only
        # the first) means they still apply late in a long melody, where a
        # single starting token would have faded from the LSTM's memory.
        embedded = self.embedding(x)
        if mode is not None:
            embedded = embedded + self.mode_embedding(mode).unsqueeze(1)
        if style is not None:
            embedded = embedded + self.style_embedding(style).unsqueeze(1)
        lstm_out, hidden = self.lstm(embedded, hidden)
        lstm_out = self.dropout(lstm_out)
        output = self.fc(lstm_out)
        return output, hidden

    def generate(self, start_sequence: List[int], length: int = 32,
                 temperature: float = 1.0, device: str = 'cpu',
                 banned_tokens: Optional[List[int]] = None,
                 mode: int = 0, style: int = 0) -> List[int]:
        self.eval()
        generated = start_sequence.copy()

        with torch.no_grad():
            hidden = None
            input_seq = torch.tensor([start_sequence], dtype=torch.long).to(device)
            mode_tensor = torch.tensor([mode], dtype=torch.long).to(device)
            style_tensor = torch.tensor([style], dtype=torch.long).to(device)

            for _ in range(length):
                output, hidden = self.forward(input_seq, hidden, mode_tensor, style_tensor)
                logits = output[0, -1, :] / temperature
                # Temperature below 1 sharpens the distribution (safer notes),
                # above 1 flattens it (more surprising notes)
                if banned_tokens:
                    # -inf becomes probability 0 after the softmax
                    logits[banned_tokens] = float('-inf')
                probabilities = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probabilities, 1).item()
                generated.append(next_token)
                input_seq = torch.tensor([[next_token]], dtype=torch.long).to(device)

        return generated
