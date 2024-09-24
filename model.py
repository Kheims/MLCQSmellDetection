import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass


@dataclass
class LSTMConfig(nn.Module):
    vocab_size: int = 50257 
    embedding_dim: int = 768
    hidden_dim: int = 512
    num_layers: int = 2  
    dropout: float = 0.1  
    max_seq_len: int = 1024  
    num_classes: int = 5  

class LSTMModel(nn.Module):
    def __init__(self, config: LSTMConfig):
        super(LSTMModel, self).__init__()

        
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        
        self.lstm = nn.LSTM(input_size=config.embedding_dim,
                            hidden_size=config.hidden_dim,
                            num_layers=config.num_layers,
                            batch_first=True,
                            dropout=config.dropout)
        
        self.dropout = nn.Dropout(config.dropout)
        
        self.fc = nn.Linear(config.hidden_dim, config.num_classes)
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        
        x = self.embedding(x)  # (batch_size, sequence_length, embedding_dim)
        
        lstm_out, (h_n, c_n) = self.lstm(x)  # (batch_size, sequence_length, hidden_dim)
        
        lstm_out = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
        
        output = self.dropout(lstm_out)
        output = self.fc(output)  # (batch_size, num_classes)
        
        return output