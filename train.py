import torch.nn as nn
import torch.optim as optim
import torch
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, classification_report
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import numpy as np
from time import time 
import re
import logging

from model import LSTMModel, LSTMConfig, LSTMMLCQ
from config import (LSTMConfig, OptimizerConfig, SchedulerConfig, TrainingConfig, 
                    get_optimizer, get_scheduler)
from custom_tokenizers import SimpleTokenizer, BertTokenizer, TokenizerContext


logging.basicConfig(
    filename='training_log.txt',
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
# ----------------------------- Data Prep -------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("MLCQCodeSmellSamples.json", "r") as f:
    data = json.load(f)



tokenizer_context = TokenizerContext(SimpleTokenizer())

tokenized_data = [tokenizer_context.tokenize(entry["code_snippet"]) for entry in data]


label_encoder = LabelEncoder()
labels = [entry["smell"] if entry["severity"] != "none" else "no_smell" for entry in data]
encoded_labels = label_encoder.fit_transform(labels)  




X_train, X_test, y_train, y_test = train_test_split(tokenizer_context.get_preprocessed_data(tokenized_data),
                                                     encoded_labels, test_size=0.2, random_state=42)

y_train_tensor = torch.tensor(y_train, dtype=torch.long)
unique_classes = torch.unique(y_train_tensor)
class_sample_count = torch.tensor([torch.sum(y_train_tensor == t).item() for t in unique_classes])
weight = 1.0 / class_sample_count.double()

samples_weight = weight[y_train_tensor]

samples_weight = samples_weight.double()

sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)

class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)


class CodeSmellDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

class CodeSmellDatasetBert(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        tokenized_input = self.X[idx]
        input_ids = tokenized_input['input_ids'].squeeze(0)  # Shape [seq_len]
        attention_mask = tokenized_input['attention_mask'].squeeze(0)  # Shape [seq_len]
        return input_ids, attention_mask, self.y[idx]
    



train_dataset = CodeSmellDataset(X_train, y_train)
test_dataset = CodeSmellDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False,collate_fn=tokenizer_context.get_collate_fn(), sampler=sampler)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=tokenizer_context.get_collate_fn())



# ---------------------------------------------------------------------------------------------------#


model_config = LSTMConfig(
    vocab_size= tokenizer_context.get_vocab_size(),
    embedding_dim=256, 
    hidden_dim=512,    
    num_layers=3,      
    dropout=0.5,       
    max_seq_len=1028,   
    num_classes=5     
)
model = LSTMMLCQ(model_config, tokenizer_context.strategy.provides_embeddings())
model.to(device)

optimizer_config = OptimizerConfig(optimizer_type="SGD", learning_rate=1e-4, momentum=0.9)
scheduler_config = SchedulerConfig(scheduler_type="step", step_size=10, gamma=0.5)


optimizer = get_optimizer(model, optimizer_config)
scheduler = get_scheduler(optimizer, scheduler_config)

training_config = TrainingConfig(batch_size=16, epochs=10, clip_grad=1.0)

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)


#---------------------------- Training ---------------------------------------------------------------#

logging.info("------------------------------------------------------------------------------")
logging.info(f"Training Model with Config: {model_config}")
logging.info(f"Optimizer Config: {optimizer_config}")
logging.info(f"Scheduler Config: {scheduler_config}")
logging.info(f"Training Config: {training_config}")



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
            

            
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            
            running_loss += loss.item()

        epoch_time = time() - start_time
        print(f"Epoch [{epoch+1}/{training_config.epochs}], Loss: {running_loss/len(train_loader):.4f} | duration: {epoch_time:.3f}")
        logging.info(f"Epoch [{epoch+1}/{training_config.epochs}], Loss: {running_loss/len(train_loader):.4f} | duration: {epoch_time:.3f}")

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
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(predicted.cpu().tolist())
    
    return y_true, y_pred



if __name__ == '__main__':

    train_model(model, train_loader, optimizer, scheduler, training_config)

    # -----------------------------------------------------------------------------------------------#


    y_true, y_pred = evaluate_model(model, test_loader)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)

    for i, label in enumerate(label_encoder.classes_):  # label_encoder.classes_ contains the names of the smells
        print(f"Class: {label}")
        print(f"  Precision: {precision[i]:.4f}")
        print(f"  Recall: {recall[i]:.4f}")
        print(f"  F1-Score: {f1[i]:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))
    logging.info(classification_report(y_true, y_pred, target_names=label_encoder.classes_))