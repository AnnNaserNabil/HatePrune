# utils.py (modified to include pruning metrics: total params, non-zero weights, sparsity)
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed: {seed}")

def get_model_metrics(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = sum(p.nelement() * p.element_size() for p in model.parameters()) / (1024**2)

    # Pruning-specific: count non-zero weights (only in 'weight' params, ignoring biases etc.)
    total_weights = 0
    non_zero_weights = 0
    for name, param in model.named_parameters():
        if 'weight' in name:  # Focus on weights (e.g., linear.weight)
            total_weights += param.numel()
            non_zero_weights += (param != 0).sum().item()

    sparsity = 1 - (non_zero_weights / total_weights) if total_weights > 0 else 0.0

    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': round(size_mb, 2),
        'total_weights': total_weights,
        'non_zero_weights': non_zero_weights,
        'sparsity': round(sparsity * 100, 2)
    }

def print_fold_summary(fold_num, best_metrics, best_epoch):
    print("\n" + "-"*60)
    print(f"FOLD {fold_num + 1} SUMMARY")
    print("-"*60)
    print(f"Best epoch: {best_epoch}")
    print(f"Val F1 (Hate): {best_metrics['f1']:.4f}")
    print(f"Val Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"Val Precision (Hate): {best_metrics['precision']:.4f}")
    print(f"Val Recall (Hate): {best_metrics['recall']:.4f}")
    print(f"Val Macro F1: {best_metrics['macro_f1']:.4f}")
    print(f"Val ROC-AUC: {best_metrics['roc_auc']:.4f}")
    print(f"Val Loss: {best_metrics['loss']:.4f}")
    print(f"Train F1 (Hate): {best_metrics['train_f1']:.4f}")
    print(f"Train Macro F1: {best_metrics['train_macro_f1']:.4f}")
    print(f"Train Loss: {best_metrics['train_loss']:.4f}")
    print("-"*60 + "\n")

def print_experiment_summary(best_fold_idx, best_metrics, model_metrics):
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)
    print(f"Best Fold: {best_fold_idx + 1}")
    print(f"Val Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"Val Precision (Hate): {best_metrics['precision']:.4f}")
    print(f"Val Recall (Hate): {best_metrics['recall']:.4f}")
    print(f"Val F1 (Hate): {best_metrics['f1']:.4f}")
    print(f"Val Precision (Non-Hate): {best_metrics['precision_negative']:.4f}")
    print(f"Val Recall (Non-Hate): {best_metrics['recall_negative']:.4f}")
    print(f"Val F1 (Non-Hate): {best_metrics['f1_negative']:.4f}")
    print(f"Val Macro F1: {best_metrics['macro_f1']:.4f}")
    print(f"Val ROC-AUC: {best_metrics['roc_auc']:.4f}")
    print(f"Val Loss: {best_metrics['loss']:.4f}")
    print(f"Best Threshold: {best_metrics['best_threshold']}")
    print(f"Train Accuracy: {best_metrics['train_accuracy']:.4f}")
    print(f"Train Precision (Hate): {best_metrics['train_precision']:.4f}")
    print(f"Train Recall (Hate): {best_metrics['train_recall']:.4f}")
    print(f"Train F1 (Hate): {best_metrics['train_f1']:.4f}")
    print(f"Train Macro F1: {best_metrics['train_macro_f1']:.4f}")
    print(f"Train ROC-AUC: {best_metrics['train_roc_auc']:.4f}")
    print(f"Train Loss: {best_metrics['train_loss']:.4f}")
    print("\nModel Size:")
    print(f"  Total params: {model_metrics['total_parameters']:,}")
    print(f"  Trainable: {model_metrics['trainable_parameters']:,}")
    print(f"  Size: {model_metrics['model_size_mb']} MB")
    print(f"  Total weights: {model_metrics['total_weights']:,}")
    print(f"  Non-zero weights: {model_metrics['non_zero_weights']:,}")
    print(f"  Sparsity: {model_metrics['sparsity']}%")
    print("="*60)
