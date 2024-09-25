from abc import ABC, abstractmethod
import re
import torch
from transformers import RobertaTokenizer, RobertaModel


class TokenizerStrategy(ABC):
    @abstractmethod
    def tokenize(self, text):
        pass

    @abstractmethod
    def provides_embeddings(self):
        pass

class SimpleTokenizer(TokenizerStrategy):
    def tokenize(self, text):
        tokens = re.findall(r'\w+|\S', text)  # Basic split by non-alphanumeric chars
        return tokens 
    def provides_embeddings(self):
        return False


class BertTokenizer(TokenizerStrategy):
    def __init__(self):
        self.tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        self.model = RobertaModel.from_pretrained("microsoft/codebert-base")

    def tokenize(self, text):
        return self.tokenizer.encode(text, add_special_tokens=True, padding='max_length', truncation=True, max_length=512)
    
    def get_embeddings(self, tokenized_input):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenized_input = tokenized_input.to(device)
        self.model.to(device)

        with torch.no_grad():
            outputs = self.model(tokenized_input)
        return outputs.last_hidden_state  # CodeBERT embeddings

    def provides_embeddings(self):
        return True
    
class TokenizerContext:
    def __init__(self, strategy: TokenizerStrategy):
        self.strategy = strategy

    def set_strategy(self, strategy: TokenizerStrategy):
        self.strategy = strategy

    def tokenize(self, text):
        return self.strategy.tokenize(text)