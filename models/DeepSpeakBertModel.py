from config import Config
from transformers import BertModel
import torch
import torch.nn as nn

class DeepSpeakBertModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Linear(self.hidden_size, cfg.max_group_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, group_id: torch.Tensor, member_mask: torch.Tensor):
        outputs = self.bert(input_ids, attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        logits[~member_mask] = -torch.inf
        return logits
