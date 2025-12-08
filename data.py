# data.py
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold, KFold
import random

class HateSpeechDataset(Dataset):
    def __init__(self, comments, labels, tokenizer, max_length=128):
        self.comments = comments
        self.labels = labels.astype(np.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self): return len(self.comments)

    def __getitem__(self, idx):
        comment = str(self.comments[idx])
        encoding = self.tokenizer(
            comment,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float)
        }

def load_and_preprocess_data(path):
    df = pd.read_csv(path)
    if 'Comments' not in df.columns or 'HateSpeech' not in df.columns:
        raise ValueError("CSV must have 'Comments' and 'HateSpeech' columns")
    comments = df['Comments'].values
    labels = df['HateSpeech'].values
    print(f"Loaded {len(comments)} samples | Hate: {labels.sum()} ({labels.mean()*100:.2f}%)")
    return comments, labels

def get_calibration_samples(comments, labels, num_samples=512, seed=42):
    rng = random.Random(seed)
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    pos_sel = rng.sample(list(pos_idx), min(num_samples//2, len(pos_idx)))
    neg_sel = rng.sample(list(neg_idx), min(num_samples//2, len(neg_idx)))
    sel_idx = pos_sel + neg_sel
    return [comments[i] for i in sel_idx]

def get_kfold_splits(comments, labels, n_folds=5, stratify=True, seed=42):
    if stratify:
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    else:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return kf.split(comments, labels)

def get_class_weight(labels):
    pos = labels.sum()
    neg = len(labels) - pos
    return torch.tensor([neg / pos if pos > 0 else 1.0])
