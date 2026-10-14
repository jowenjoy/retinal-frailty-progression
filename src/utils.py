"""
utils.py
"""

import os
import random
import logging
import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(log_file: str = None, name: str = 'survival') -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if logger.handlers: return logger

    fmt = logging.Formatter('[%(asctime)s] %(levelname)s — %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file:
        os.makedirs(os.path.dirname(log_file) or '.', exist_ok=True)
        fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


class EarlyStopping:
    def __init__(self, patience: int = 10, mode: str = 'max', delta: float = 1e-4, path: str = 'best_model.pth'):
        self.patience, self.mode, self.delta, self.path = patience, mode, delta, path
        self.counter, self.best_score, self.early_stop = 0, None, False

    def __call__(self, score: float, model: torch.nn.Module) -> bool:
        if self.best_score is None:
            self._save(score, model)
            return True
        improved = (self.mode == 'max' and score > self.best_score + self.delta) or \
                   (self.mode == 'min' and score < self.best_score - self.delta)
        if improved:
            self._save(score, model)
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience: self.early_stop = True
        return False

    def _save(self, score: float, model: torch.nn.Module):
        torch.save(model.state_dict(), self.path)
        self.best_score = score