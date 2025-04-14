from torch import nn
import torch

"""
Baseline model that randomly chooses among the valid member_ids.
"""
class BaselineModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, group_id: torch.Tensor, member_mask: torch.Tensor):
        logits = torch.rand_like(member_mask, dtype=torch.float)
        logits[~member_mask] = -torch.inf
        return logits
