from abc import ABC, abstractmethod
import re
import torch
from transformers import RobertaTokenizer, RobertaModel

MAX_LEN = 4000

def encode_snippet(token_to_idx, snippet):
        return [token_to_idx.get(token, token_to_idx["<UNK>"]) for token in snippet]

def pad_sequence(seq, max_len, pad_idx):
    if len(seq) < max_len:
        return seq + [pad_idx] * (max_len - len(seq))  # Padding with 0
    else:
        return seq[:max_len]  # Truncate to max_len


class TokenizerStrategy(ABC):
    @abstractmethod
    def tokenize(self, text):
        pass

    @abstractmethod
    def provides_embeddings(self):
        pass

class SimpleTokenizer(TokenizerStrategy):
    def __init__(self):
        self.vocab = None
        self.vocab_size = None 

    def tokenize(self, text):
        tokens = re.findall(r'\w+|\S', text)  # Basic split by non-alphanumeric chars
        return tokens 
    def build_vocab(self, tokenized_data):
        self.vocab = set(token for snippet in tokenized_data for token in snippet)
        self.vocab.add("<UNK>")
        self.vocab.add("<PAD>")
        self.vocab_size = len(self.vocab)
        return self.vocab_size

    def get_vocab_size(self):
        return self.vocab_size
    
    def get_preprocessed_data(self, tokens):
        
        if self.vocab is None:
            self.build_vocab(tokens)

        token_to_idx = {token: idx for idx, token in enumerate(self.vocab)}
        pad_idx = token_to_idx["<PAD>"]

        encoded_data = [encode_snippet(token_to_idx, snippet) for snippet in tokens]
    
        padded_data = [pad_sequence(seq, MAX_LEN, pad_idx) for seq in encoded_data]

        return padded_data
    
    def get_collate_fn(self):
        def simple_collate_fn(batch):
            
            inputs, labels = zip(*batch)
            # Convert each input (which is a list) into a tensor
            inputs_tensor = [torch.tensor(input_seq, dtype=torch.long) for input_seq in inputs]
        
            # Pad sequences to have the same length within the batch if necessary
            inputs_tensor = torch.nn.utils.rnn.pad_sequence(inputs_tensor, batch_first=True, padding_value=0)
        
            # Convert labels into a tensor
            labels_tensor = torch.tensor(labels, dtype=torch.long)
        
            return inputs_tensor, labels_tensor
        return simple_collate_fn
    
    def provides_embeddings(self):
        return False

class BertTokenizer(TokenizerStrategy):
    def __init__(self):
        self.tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        self.model = RobertaModel.from_pretrained("microsoft/codebert-base")

    def tokenize(self, text):
        tokens = self.tokenizer.encode_plus(
            text, 
            add_special_tokens=True, 
            padding='max_length', 
            truncation=True, 
            max_length=512,
            return_tensors='pt'
        )
        return tokens
    
    def get_preprocessed_data(self, tokens):
        return tokens

    def get_embeddings(self, tokenized_input):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_ids = tokenized_input["input_ids"].to(device)
        attention_mask = tokenized_input["attention_mask"].to(device)
        self.model.to(device)

        with torch.no_grad():
            outputs = self.model(tokenized_input, attention_mask=attention_mask)
        return outputs.last_hidden_state  # CodeBERT embeddings

    def get_vocab_size(self):
        return self.tokenizer.vocab_size    

    def get_collate_fn(self):
            
        def bert_collate_fn(batch):
            input_ids = [item[0] for item in batch]
            attention_masks = [item[1] for item in batch]
            labels = [item[2] for item in batch]
            
            input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
            attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)
            labels = torch.tensor(labels, dtype=torch.long)

            return {'input_ids': input_ids, 'attention_mask': attention_masks}, labels
        
        return bert_collate_fn

    def provides_embeddings(self):
        return True
    
class TokenizerContext:
    def __init__(self, strategy: TokenizerStrategy):
        self.strategy = strategy

    def set_strategy(self, strategy: TokenizerStrategy):
        self.strategy = strategy

    def tokenize(self, text):
        return self.strategy.tokenize(text)
    
    def get_preprocessed_data(self, tokens):
        return self.strategy.get_preprocessed_data(tokens)
    
    def get_vocab_size(self):
        return self.strategy.get_vocab_size()
    def get_collate_fn(self):
        return self.strategy.get_collate_fn()