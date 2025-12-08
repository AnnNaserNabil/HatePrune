# utils.py
import torch
import random
import numpy as np
import pandas as pd

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_model_stats(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    zeros = sum((p == 0).sum().item() for p in model.parameters() if p.ndim > 1)
    total_weights = sum(p.numel() for p in model.parameters() if p.ndim > 1)
    sparsity = zeros / total_weights if total_weights > 0 else 0
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
    return {
        'total_params': total,
        'trainable': trainable,
        'size_mb': round(size_mb, 2),
        'sparsity_%': round(sparsity * 100, 2)
    }
