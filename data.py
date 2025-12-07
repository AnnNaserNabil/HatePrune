# data.py (minor addition: function to get calib samples for Wanda)
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold, KFold
import random

LABEL_COLUMN = 'HateSpeech'

class HateSpeechDataset(Dataset):
    def __init__(self, comments, labels, tokenizer, max_length=128):
        self.comments = comments
        self.labels = labels.astype(np.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self): return len(self.comments)

    def __getitem__(self, idx):
        comment = str(self.comments[idx])
        label = self.labels[idx]
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
            'labels': torch.tensor(label, dtype=torch.float)
        }

def load_and_preprocess_data(dataset_path):
    df = pd.read_csv(dataset_path)
    if 'Comments' not in df.columns or 'HateSpeech' not in df.columns:
        raise ValueError("Need 'Comments' and 'HateSpeech' columns")
    comments = df['Comments'].values
    labels = df['HateSpeech'].values
    pos = np.sum(labels)
    print(f"\nDataset: {len(comments)} samples | Hate: {pos} ({pos/len(labels)*100:.2f}%)")
    return comments, labels

def get_calibration_samples(comments, labels, num_samples=512, seed=42):
    """Random stratified sample for Wanda calibration."""
    random.seed(seed)
    indices = list(range(len(comments)))
    pos_indices = [i for i, l in enumerate(labels) if l == 1]
    neg_indices = [i for i, l in enumerate(labels) if l == 0]
    
    pos_sample = random.sample(pos_indices, min(num_samples // 2, len(pos_indices)))
    neg_sample = random.sample(neg_indices, min(num_samples // 2, len(neg_indices)))
    calib_indices = pos_sample + neg_sample
    return [comments[i] for i in calib_indices]

def prepare_kfold_splits(comments, labels, num_folds=5, stratification_type='binary', seed=42):
    if stratification_type == 'binary':
        kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    else:
        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    return kfold.split(comments, labels)

def calculate_class_weights(labels):
    pos = np.sum(labels)
    neg = len(labels) - pos
    weight = neg / pos if pos > 0 else 1.0
    print(f"Class weight (Hate): {weight:.3f}")
    return torch.FloatTensor([weight])
