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

from model import LSTMCodeSmellClassifier

with open("MLCQCodeSmellSamples.json", "r") as f:
    data = json.load(f)

# Tokenizer: split by whitespace and punctuation
def simple_tokenizer(code_snippet):
    tokens = re.findall(r'\w+|\S', code_snippet)  # Basic split by non-alphanumeric chars
    return tokens

label_encoder = LabelEncoder()
labels = [entry["smell"] for entry in data]
encoded_labels = label_encoder.fit_transform(labels)  

tokenized_data = [simple_tokenizer(entry["code_snippet"]) for entry in data]

# Create vocabulary from tokenized data
vocab = set(token for snippet in tokenized_data for token in snippet)
vocab_size = len(vocab)
token_to_idx = {token: idx for idx, token in enumerate(vocab)}

# Convert tokenized data to indexed sequences
def encode_snippet(snippet):
    return [token_to_idx[token] for token in snippet]

encoded_data = [encode_snippet(snippet) for snippet in tokenized_data]

# Padding and truncating to a fixed length
max_len = 1000  
def pad_sequence(seq, max_len):
    if len(seq) < max_len:
        return seq + [0] * (max_len - len(seq))  # Padding with 0
    else:
        return seq[:max_len]  # Truncate to max_len

padded_data = [pad_sequence(seq, max_len) for seq in encoded_data]

X_train, X_test, y_train, y_test = train_test_split(padded_data, encoded_labels, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.long)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

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

vocab_size = len(vocab)
embed_size = 128  
hidden_size = 256 
num_classes = len(label_encoder.classes_)  

model = LSTMCodeSmellClassifier(vocab_size, embed_size, hidden_size, num_classes)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        start_time = time()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        epoch_time = time() - start_time
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f} | duration: {epoch_time:.3f}")

train_model(model, train_loader, criterion, optimizer)


def evaluate_model(model, test_loader):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in test_loader:
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