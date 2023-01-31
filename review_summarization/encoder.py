import pandas as pd
import torch
from transformers import DistilBertModel, DistilBertTokenizer


class SentenceEncoder:
    def __init__(self):
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = DistilBertModel.from_pretrained('distilbert-base-uncased')

    def encode_sentences(self, data: torch.Tensor):
        encoded = self.tokenizer(data, return_tensors='pt', padding=True, truncation=True)
        model_output = self.model(**encoded)
        last_model_layer = model_output[0]
        # cls token pooling
        pooled_model_output = last_model_layer[:, 0]
        return pooled_model_output
