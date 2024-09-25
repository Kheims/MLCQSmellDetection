import torch.nn as nn
import torch.optim as optim
import json
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, classification_report
from torch.utils.data import DataLoader, Dataset
from time import time 
import re

from model import LSTMModel, LSTMConfig, LSTMMLCQ
from config import (LSTMConfig, OptimizerConfig, SchedulerConfig, TrainingConfig, 
                    get_optimizer, get_scheduler)
from custom_tokenizers import SimpleTokenizer, BertTokenizer, TokenizerContext

# ----------------------------- Data Prep -------------------------------------------------------------
with open("MLCQCodeSmellSamples.json", "r") as f:
    data = json.load(f)

codebert_tokenizer = BertTokenizer()
simple_tokenizer = SimpleTokenizer()


tokenizer_context = TokenizerContext(codebert_tokenizer)

tokenized_data = [tokenizer_context.tokenize(entry["code_snippet"]) for entry in data]

label_encoder = LabelEncoder()
labels = [entry["smell"] for entry in data]
encoded_labels = label_encoder.fit_transform(labels)  



## Create vocabulary from tokenized data
#vocab = set(token for snippet in tokenized_data for token in snippet)
#vocab.add("<UNK>")
#vocab.add("<PAD>")
#vocab_size = len(vocab)
#token_to_idx = {token: idx for idx, token in enumerate(vocab)}
#pad_idx = token_to_idx["<PAD>"]
#
## Convert tokenized data to indexed sequences
#def encode_snippet(snippet):
#    return [token_to_idx.get(token, token_to_idx["<UNK>"]) for token in snippet]
#
#encoded_data = [encode_snippet(snippet) for snippet in tokenized_data]
#
## Padding and truncating to a fixed length
#max_len = 1000  
#def pad_sequence(seq, max_len):
#    if len(seq) < max_len:
#        return seq + [pad_idx] * (max_len - len(seq))  # Padding with 0
#    else:
#        return seq[:max_len]  # Truncate to max_len
#
#padded_data = [pad_sequence(seq, max_len) for seq in encoded_data]

X_train, X_test, y_train, y_test = train_test_split(tokenized_data, encoded_labels, test_size=0.2, random_state=42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train_tensor = torch.tensor(X_train, dtype=torch.long).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.long).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

class CodeSmellDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = CodeSmellDataset(X_train_tensor, y_train_tensor)
test_dataset = CodeSmellDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)



# ---------------------------------------------------------------------------------------------------#


model_config = LSTMConfig(
    vocab_size=codebert_tokenizer.tokenizer.vocab_size if tokenizer_context.strategy.provides_embeddings() else None,  
    embedding_dim=256, 
    hidden_dim=512,    
    num_layers=1,      
    dropout=0.2,       
    max_seq_len=512,   
    num_classes=4      
)
model = LSTMMLCQ(model_config, tokenizer_context.strategy.provides_embeddings())
model.to(device)

optimizer_config = OptimizerConfig(optimizer_type="Adam", learning_rate=1e-4, weight_decay=0.01)
scheduler_config = SchedulerConfig(scheduler_type="step", step_size=10, gamma=0.5)


optimizer = get_optimizer(model, optimizer_config)
scheduler = get_scheduler(optimizer, scheduler_config)

training_config = TrainingConfig(batch_size=8, epochs=10, clip_grad=None)

criterion = nn.CrossEntropyLoss()

def train_model(model, train_loader, optimizer, scheduler=None, training_config=None):
    model.train()
    for epoch in range(training_config.epochs):
        start_time = time()
        running_loss = 0.0
        for inputs, labels in train_loader:

            inputs, labels = inputs.to(device), labels.to(device)


            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            if tokenizer_context.strategy.provides_embeddings():
                inputs = tokenizer_context.strategy.get_embeddings(inputs).to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()

            if training_config.clip_grad is not None:
                nn.utils.clip_grad_norm_(model.parameters(), training_config.clip_grad)
            

            if training_config.clip_grad is not None:
                nn.utils.clip_grad_norm_(model.parameters(), training_config.clip_grad)
            
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            if scheduler is not None:
                scheduler.step()
            
            running_loss += loss.item()

        epoch_time = time() - start_time
        print(f"Epoch [{epoch+1}/{training_config.epochs}], Loss: {running_loss/len(train_loader):.4f} | duration: {epoch_time:.3f}")


train_model(model, train_loader, optimizer, scheduler, training_config)

# -----------------------------------------------------------------------------------------------#
# -----------------------------------------------------------------------------------------------#


def evaluate_model(model, test_loader):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            if tokenizer_context.strategy.provides_embeddings():
                inputs = tokenizer_context.strategy.get_embeddings(inputs).to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    return y_true, y_pred

y_true, y_pred = evaluate_model(model, test_loader)
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)

for i, label in enumerate(label_encoder.classes_):  # label_encoder.classes_ contains the names of the smells
    print(f"Class: {label}")
    print(f"  Precision: {precision[i]:.4f}")
    print(f"  Recall: {recall[i]:.4f}")
    print(f"  F1-Score: {f1[i]:.4f}")

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))