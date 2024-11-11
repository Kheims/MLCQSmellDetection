from transformers import RobertaTokenizer
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors
import torch
from typing import List, Tuple

class CodeTokenizer:
    def __init__(self, max_length: int = 512):
        self.tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
        self.max_length = max_length
        
    def encode(self, code: str) -> Tuple[torch.Tensor, int]:
        # Tokenize with special tokens
        tokens = self.tokenizer.encode(
            code,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        actual_length = (tokens != self.tokenizer.pad_token_id).sum().item()
        
        return tokens.squeeze(), actual_length

class HierarchicalCodeTokenizer:
    def __init__(self, chunk_size: int = 128, max_chunks: int = 4):
        self.base_tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
        self.chunk_size = chunk_size
        self.max_chunks = max_chunks
        
    def split_code_into_chunks(self, code: str) -> List[str]:
        lines = code.split('\n')
        chunks = []
        current_chunk = []
        
        for line in lines:
            current_chunk.append(line)
            if len(current_chunk) >= self.chunk_size or \
               any(line.strip().startswith(keyword) for keyword in ['class', 'public', 'private', 'protected', 'void', 'int', 'String']):
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
            
        return chunks[:self.max_chunks]
    
    def encode(self, code: str) -> Tuple[torch.Tensor, List[int]]:
        chunks = self.split_code_into_chunks(code)
        encoded_chunks = []
        chunk_lengths = []
        
        for chunk in chunks:
            tokens = self.base_tokenizer.encode(
                chunk,
                add_special_tokens=True,
                max_length=self.chunk_size,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            encoded_chunks.append(tokens)
            chunk_lengths.append((tokens != self.base_tokenizer.pad_token_id).sum().item())
        
        # Pad to max_chunks if necessary
        while len(encoded_chunks) < self.max_chunks:
            encoded_chunks.append(torch.full((1, self.chunk_size), 
                                          self.base_tokenizer.pad_token_id))
            chunk_lengths.append(0)
            
        return torch.cat(encoded_chunks, dim=0), chunk_lengths
