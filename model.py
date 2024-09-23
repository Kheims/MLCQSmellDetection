import torch.nn as nn
import torch.optim as optim

class LSTMCodeSmellClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes):
        super(LSTMCodeSmellClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # x shape: (batch_size, seq_length)
        embedded = self.embedding(x)  # shape: (batch_size, seq_length, embed_size)
        lstm_out, _ = self.lstm(embedded)  # shape: (batch_size, seq_length, hidden_size)
        lstm_out = lstm_out[:, -1, :]  # Take the output from the last LSTM cell
        out = self.fc(lstm_out)  # shape: (batch_size, num_classes)
        return out