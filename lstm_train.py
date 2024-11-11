import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset, Subset
import json
import logging
from simple_model import SimpleLSTM, SimpleLSTMConfig
from simple_tokenizer import SimpleTokenizer
from simple_config import SimpleTrainingConfig
from tqdm import tqdm
from sklearn.metrics import classification_report, precision_recall_fscore_support
import time

# Configure logging
logging.basicConfig(
    filename='simple_training_log.txt',
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class SimpleCodeSmellDataset(Dataset):
    def __init__(self, json_file, tokenizer):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
            #data = json.load(f)
            #self.data = data[:2000]
        self.tokenizer = tokenizer
        
        # Map smells to numeric labels
        self.smell_to_label = {
            "blob": 0,
            "feature envy": 1,
            "long method": 2,
            "data class": 3,
            "no smell": 4

        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        code = item['code_snippet']
        label = item['smell'] if item['severity'] != "none" else "no smell"
        label = self.smell_to_label[label]
        return code, label
    
    def get_smell_repartition(self):
        self.smell_count = {}
        for data in self.data:
            smell = data['smell'] if data['severity'] != 'none' else 'no smell'
            self.smell_count[smell] = self.smell_count.get(smell,0) + 1
        
        for smell, instances in self.smell_count.items():
            print(f"{smell} has {instances} occurence")
        


def train_epoch(model, dataloader, criterion, optimizer, device, pbar=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    epoch_start_time = time.time()

    for code, labels in dataloader:
        if pbar:
            pbar.update(1)
            
        code = code.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(code)
        loss = criterion(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    accuracy = 100. * correct / total
    avg_loss = total_loss / len(dataloader)
    epoch_time = time.time() - epoch_start_time
    
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    
    logging.info(f"Training - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, "
                f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, "
                f"Time: {epoch_time:.2f}s")
    return avg_loss, accuracy, precision, recall, f1

def evaluate(model, dataloader, criterion, device, pbar=None):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for code, labels in dataloader:
            if pbar:
                pbar.update(1)
                
            code = code.to(device)
            labels = labels.to(device)

            outputs = model(code)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100. * correct / total
    avg_loss = total_loss / len(dataloader)
    
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    
    logging.info(f"Validation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, "
                f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    logging.info("\nClassification Report:")
    report = classification_report(all_labels, all_preds, target_names=[
        "Blob", "Feature Envy", "Long Method", "Data Class", "No Smell"
    ], labels=[0,1,2,3,4])
    logging.info(f"\n{report}")
    
    return avg_loss, accuracy, precision, recall, f1

def main():
    logging.info("\n" + "="*50)
    
    config = SimpleTrainingConfig()
    model_config = SimpleLSTMConfig()
    
    model_config.num_classes = 5      


    logging.info("\nTraining Configuration:")
    logging.info(f"Batch Size: {config.batch_size}")
    logging.info(f"Epochs: {config.epochs}")
    logging.info(f"Learning Rate: {config.learning_rate}")
    logging.info(f"Train Split: {config.train_split}")
    logging.info(f"Number of Classes: {model_config.num_classes}")
    
    #device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    device = 'cpu'
    logging.info(f"Using device: {device}")

    tokenizer = SimpleTokenizer(max_len=model_config.max_seq_len)
    
    dataset = SimpleCodeSmellDataset("MLCQCodeSmellSamples.json", tokenizer)
    smelly_indices = [i for i, (_, label) in enumerate(dataset) if label!=4]
    smelly_dataset = Subset(dataset, smelly_indices)

    oversampled_smelly_dataset = ConcatDataset([smelly_dataset]*3)
    balanced_dataset = ConcatDataset([dataset, oversampled_smelly_dataset])

    logging.info(f"Original dataset size: {len(dataset)}")
    logging.info(f"balanced dataset size: {len(dataset)}")

    
    texts = [item[0] for item in dataset]
    vocab_size = tokenizer.build_vocab(texts)
    logging.info(f"Vocabulary size: {vocab_size}")
    
    model_config.vocab_size = vocab_size
    
    dataset.get_smell_repartition()
    train_size = int(config.train_split * len(balanced_dataset))
    val_size = len(balanced_dataset) - train_size
    train_dataset, val_dataset = random_split(balanced_dataset, [train_size, val_size])
    logging.info(f"Train size: {train_size}, Validation size: {val_size}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        collate_fn=tokenizer.get_collate_fn()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=tokenizer.get_collate_fn()
    )

    model = SimpleLSTM(model_config).to(device)
    logging.info(f"\nModel Architecture:\n{model}")
    weights = torch.tensor([10.0,10.0,10.0,10.0,1.0], device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    best_val_accuracy = 0
    training_start_time = time.time()
    
    total_steps = len(train_loader) * config.epochs + len(val_loader) * config.epochs
    
    with tqdm(total=total_steps, desc="Training Progress") as pbar:
        for epoch in range(config.epochs):
            logging.info(f"\nEpoch {epoch+1}/{config.epochs}")
            
            train_loss, train_acc, train_prec, train_rec, train_f1 = train_epoch(
                model, train_loader, criterion, optimizer, device, pbar
            )
            
            val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(
                model, val_loader, criterion, device, pbar
            )
            
            if config.save_model and val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                torch.save(model.state_dict(), config.model_path)
                logging.info(f"Saved best model with validation accuracy: {val_acc:.2f}%")
    
    total_time = time.time() - training_start_time
    logging.info(f"\nTraining completed in {total_time:.2f} seconds")
    logging.info(f"Best validation accuracy: {best_val_accuracy:.2f}%")
    logging.info("="*50 + "\n")

if __name__ == "__main__":
    main()
