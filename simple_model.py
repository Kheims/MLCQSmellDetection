import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class SimpleLSTMConfig:
    vocab_size: int = 10000  
    embedding_dim: int = 100 
    hidden_dim: int = 128    
    num_layers: int = 1      
    dropout: float = 0.0     
    max_seq_len: int = 512   
    num_classes: int = 5     

class SimpleLSTM(nn.Module):
    def __init__(self, config: SimpleLSTMConfig):
        super(SimpleLSTM, self).__init__()
        
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        
        self.lstm = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True
        )
        
        self.fc = nn.Linear(config.hidden_dim, config.num_classes)
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        last_hidden = lstm_out[:, -1, :]  
        output = self.fc(last_hidden)
        return output

class SimpleBILSTM(nn.Module):
    def __init__(self, config: SimpleLSTMConfig):
        super(SimpleBILSTM, self).__init__()
        
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        
        self.lstm = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=True
        )
        

        self.fc = nn.Linear(2*config.hidden_dim, config.num_classes)


    def forward(self, x):

        embedded = self.embedding(x)
        hidden_dim = self.lstm.hidden_size
        lstm_out, _ = self.lstm(embedded)
        last_hidden = torch.cat((lstm_out[:, -1, :hidden_dim],lstm_out[:,0,hidden_dim:]), dim=1)  
        output = self.fc(last_hidden)
        return output



class SimpleBILSTMAttn(nn.Module):
    def __init__(self, config: SimpleLSTMConfig):
        super(SimpleBILSTMAttn, self).__init__()
        
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        
        self.lstm = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        self.attention = nn.Linear(config.hidden_dim * 2, 1)

        self.fc = nn.Linear(2*config.hidden_dim, config.num_classes)

    def attention_net(self, lstm_output):
        # lstm_output shape: (batch_size, seq_len, hidden_dim * 2)
        attention_weights = torch.tanh(self.attention(lstm_output))
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # context vector shape : ((batch_size, seq_len,1) * (batch_size, seq_len, hidden_dim*2)) 
        #                           -> (batch_size, seq_len, hidden_dim*2)  'elem wise matmul'
        #                           -> (batch_size, hidden_dim*2) ie context vector repr of each snippet
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        return context_vector 

    def forward(self, x):

        embedded = self.embedding(x)
        hidden_dim = self.lstm.hidden_size
        lstm_out, _ = self.lstm(embedded)
        attention_output = self.attention_net(lstm_out)
        output = self.fc(attention_output)
        return output