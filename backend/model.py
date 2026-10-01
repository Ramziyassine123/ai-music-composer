import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional

class MusicLSTM(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int = 128,
                 hidden_size: int = 256, num_layers: int = 2, dropout: float = 0.3):
        super(MusicLSTM, self).__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        lstm_out, hidden = self.lstm(embedded, hidden)
        lstm_out = self.dropout(lstm_out)
        output = self.fc(lstm_out)
        return output, hidden

    def generate(self, start_sequence: List[int], length: int = 32,
                 temperature: float = 1.0, device: str = 'cpu',
                 banned_tokens: Optional[List[int]] = None) -> List[int]:
        self.eval()
        generated = start_sequence.copy()

        with torch.no_grad():
            hidden = None
            input_seq = torch.tensor([start_sequence], dtype=torch.long).to(device)

            for _ in range(length):
                output, hidden = self.forward(input_seq, hidden)
                logits = output[0, -1, :] / temperature
                if banned_tokens:
                    logits[banned_tokens] = float('-inf')
                probabilities = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probabilities, 1).item()
                generated.append(next_token)
                input_seq = torch.tensor([[next_token]], dtype=torch.long).to(device)

        return generated