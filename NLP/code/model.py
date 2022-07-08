import torch
from torch import nn
from transformers import AutoModel

class RumourCLS(nn.Module):
    def __init__(self):
        super(RumourCLS, self).__init__()
        self.encoder = AutoModel.from_pretrained('bert-base-uncased')
        hidden_size = self.encoder.config.hidden_size
        self.clser = nn.Linear(hidden_size, 1)

    def forward(self, inputs, masks):
        texts_emb = self.encoder(input_ids=inputs, attention_mask=masks).last_hidden_state
        texts_emb = texts_emb[:, 0, :]
        logits = self.clser(texts_emb)
        return logits