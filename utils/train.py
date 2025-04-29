from config import Config
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from utils import run_epoch
import logging
import os
import torch
import torch.nn as nn

def train(cfg: Config, model: nn.Module, train_dl: DataLoader, val_dl: DataLoader, criterion: nn.CrossEntropyLoss, optimizer: Optimizer | None = None):
    start_epoch = 0

    checkpoint_path = os.path.join(cfg.output_dir, cfg.checkpoint)
    if cfg.resume and os.path.isfile(checkpoint_path) and optimizer is not None:
        checkpoint = torch.load(checkpoint_path, map_location=cfg.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_val_loss = checkpoint['best_val_loss']
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from checkpoint at epoch {checkpoint['epoch'] + 1}")

    best_val_loss = torch.inf
    epochs_without_improvement = 0
    best_model_path = os.path.join(cfg.output_dir, cfg.best_model_path)

    model.to(cfg.device)

    for epoch in range(start_epoch, cfg.num_epochs):
        log_msg = f"Epoch {epoch + 1}/{cfg.num_epochs}"
        logging.info(log_msg)
        print(log_msg)

        for is_training in (True, False):
            avg_loss, accuracy = run_epoch(
                cfg.device,
                model,
                train_dl if is_training else val_dl,
                criterion,
                optimizer if is_training else None,
            )

            mode = "Train" if is_training else "Val"
            log_msg = f"{mode} Loss: {avg_loss:.4f}, {mode} Acc: {accuracy:.4f}"
            logging.info(log_msg)
            print(log_msg)

        if avg_loss < best_val_loss:
            best_val_loss = avg_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.patience:
                log_msg = f"Stopping early after {epoch + 1} epochs (no improvement for {cfg.patience} epochs)."
                logging.info(log_msg)
                print(log_msg)
                break

        if optimizer is not None:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
            }
            torch.save(checkpoint, checkpoint_path)
