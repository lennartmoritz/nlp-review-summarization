import pandas as pd
import torch
from transformers import DistilBertModel, DistilBertTokenizer


class SentenceEncoder:
    def __init__(self):
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = DistilBertModel.from_pretrained('distilbert-base-uncased')

    def encode_sentences(self, data: torch.Tensor):
        encoded = self.tokenizer(data)
        return self.model(encoded)
