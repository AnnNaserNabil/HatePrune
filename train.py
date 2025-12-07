# train.py (updated: gradual prune in loop; Wanda is pre-applied)
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

def calculate_metrics(y_true, y_pred):
    thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    try:
        auc = roc_auc_score(y_true, y_pred)
    except:
        auc = 0.0

    best_macro = -1
    best_metrics = {}
    for th in thresholds:
        y_bin = (y_pred > th).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_bin, average=None, zero_division=0)
        macro = (f1[0] + f1[1]) / 2
        if macro > best_macro:
            best_macro = macro
            best_metrics = {
                'accuracy': accuracy_score(y_true, y_bin),
                'precision': p[1], 'recall': r[1], 'f1': f1[1],
                'precision_negative': p[0], 'recall_negative': r[0], 'f1_negative': f1[0],
                'macro_f1': macro, 'roc_auc': auc, 'best_threshold': th
            }
    return best_metrics

def distillation_loss(s_logits, t_logits, labels, T=4.0, alpha=0.7, pos_weight=None):
    kl = F.kl_div(
        F.log_softmax(s_logits / T, dim=-1),
        F.softmax(t_logits / T, dim=-1),
        reduction='batchmean'
    ) * (T * T)

    bce = F.binary_cross_entropy_with_logits(
        s_logits, labels.float(), pos_weight=pos_weight
    )

    return alpha * kl + (1 - alpha) * bce

def train_epoch(model, loader, optimizer, scheduler, device, class_weights=None,
                T=4.0, alpha=0.7, max_norm=1.0, distill=True, prune_method='baseline',
                prune_freq=100, prune_sparsity=0.6):
    model.train()
    total_loss = 0
    scaler = GradScaler()
    pos_weight = class_weights.to(device) if class_weights is not None else None
    total_steps = len(loader) * 100  # Approximate total for gradual

    global_step = 0
    for batch_idx, batch in enumerate(tqdm(loader, desc="Train")):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        with autocast():
            out = model(input_ids, attention_mask, labels=labels)
            s_logits = out['logits']
            if distill and 'teacher_logits' in out:
                loss = distillation_loss(
                    s_logits, out['teacher_logits'],
                    labels, T=T, alpha=alpha, pos_weight=pos_weight
                )
            else:
                loss = F.binary_cross_entropy_with_logits(
                    s_logits, labels.float(), pos_weight=pos_weight
                )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # Gradual prune if applicable
        global_step += 1
        if prune_method == 'gradual_magnitude' and global_step % prune_freq == 0:
            model.gradual_prune_step(global_step, total_steps)

        total_loss += loss.item()

    return {'loss': total_loss / len(loader)}

def evaluate(model, loader, device):
    model.eval()
    preds, trues, total_loss = [], [], 0.0
    num_batches = len(loader)
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            out = model(input_ids, attention_mask)
            logits = out['logits']
            loss = F.binary_cross_entropy_with_logits(logits, labels.float())
            total_loss += loss.item()
            preds.extend(torch.sigmoid(logits).cpu().numpy())
            trues.extend(labels.cpu().numpy())
    metrics = calculate_metrics(trues, preds)
    metrics['loss'] = total_loss / num_batches
    return metrics
