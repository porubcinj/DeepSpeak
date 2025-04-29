from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch
import torch.nn as nn

def run_epoch(device: str, model: nn.Module, dataloader: DataLoader, criterion: nn.CrossEntropyLoss, optimizer: Optimizer | None = None):
    total_loss = 0.0
    correct = 0
    total = 0

    if is_training := optimizer is not None:
        model.train()
        context = torch.enable_grad()
    else:
        model.eval()
        context = torch.no_grad()

    with context:
        progress_bar = tqdm(dataloader, leave=False)
        for samples, labels in progress_bar:
            samples = {k: v.to(device) for k, v in samples.items()}
            labels = labels.to(device)

            if is_training:
                optimizer.zero_grad()

            logits = model(**samples)
            loss = criterion(logits, labels)

            if is_training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            avg_loss = total_loss / total
            accuracy = correct / total

            progress_bar.set_postfix(loss=avg_loss, accuracy=accuracy)

    return avg_loss, accuracy
