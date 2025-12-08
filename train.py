# train.py
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

def compute_metrics(y_true, y_pred_prob):
    thresholds = np.arange(0.3, 0.7, 0.05)
    y_true = np.array(y_true)
    best_f1 = -1
    best = {}
    for th in thresholds:
        y_pred = (y_pred_prob > th).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
        macro = (f1[0] + f1[1]) / 2
        if macro > best_f1:
            best_f1 = macro
            auc = roc_auc_score(y_true, y_pred_prob) if len(np.unique(y_true)) > 1 else 0
            best = {
                'threshold': th, 'macro_f1': macro, 'f1': f1[1], 'precision': p[1], 'recall': r[1],
                'accuracy': accuracy_score(y_true, y_pred), 'roc_auc': auc,
                'f1_neg': f1[0], 'precision_neg': p[0], 'recall_neg': r[0]
            }
    return best

def train_one_epoch(model, loader, optimizer, scheduler, device, config, class_weight, global_step_ref):
    model.train()
    total_loss = 0
    scaler = GradScaler()

    for batch in tqdm(loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        with autocast():
            out = model(input_ids, attention_mask, labels)
            if config.distill and 'teacher_logits' in out:
                loss = distillation_loss(out['logits'], out['teacher_logits'], labels,
                                        config.temperature, config.alpha, class_weight.to(device))
            else:
                loss = F.binary_cross_entropy_with_logits(out['logits'], labels, pos_weight=class_weight.to(device))

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # Gradual pruning
        global_step_ref[0] += 1
        if config.prune and config.prune_method == 'gradual_magnitude' and global_step_ref[0] % config.prune_freq == 0:
            total_steps = len(loader) * config.epochs
            model.gradual_prune_step(global_step_ref[0], total_steps)

        total_loss += loss.item()

    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    total_loss = 0
    for batch in tqdm(loader, desc="Eval"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        out = model(input_ids, attention_mask)
        loss = F.binary_cross_entropy_with_logits(out['logits'], labels)
        total_loss += loss.item()
        preds.extend(torch.sigmoid(out['logits']).cpu().numpy())
        trues.extend(labels.cpu().numpy())

    metrics = compute_metrics(trues, preds)
    metrics['loss'] = total_loss / len(loader)
    return metrics

def distillation_loss(s_logits, t_logits, labels, T=4.0, alpha=0.7, pos_weight=None):
    soft_loss = F.kl_div(
        F.log_softmax(s_logits / T, dim=-1),
        F.softmax(t_logits / T, dim=-1),
        reduction='batchmean'
    ) * (T * T)
    hard_loss = F.binary_cross_entropy_with_logits(s_logits, labels, pos_weight=pos_weight)
    return alpha * soft_loss + (1 - alpha) * hard_loss
