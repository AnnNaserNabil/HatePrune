# main.py - FIXED PERMANENT PRUNING BEFORE SAVE
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from config import parse_arguments
from data import load_and_preprocess_data, get_kfold_splits, get_calibration_samples, get_class_weight, HateSpeechDataset
from model import HateSpeechModel
from train import train_one_epoch, evaluate
from utils import set_seed, get_model_stats
import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")  # Suppress traitlets warnings

def print_experiment_summary(best_fold_idx, best_metrics, model_metrics):
    print("\n" + "="*70)
    print("                       EXPERIMENT COMPLETE")
    print("="*70)
    print(f"Best Fold              : {best_fold_idx + 1}")
    print(f"Best Threshold         : {best_metrics.get('threshold', 0.5):.3f}")
    print(f"Val Accuracy           : {best_metrics.get('accuracy', 0):.4f}")
    print(f"Val Precision (Hate)   : {best_metrics.get('precision', 0):.4f}")
    print(f"Val Recall (Hate)      : {best_metrics.get('recall', 0):.4f}")
    print(f"Val F1 (Hate)          : {best_metrics.get('f1', 0):.4f}")
    print(f"Val Precision (Non-Hate): {best_metrics.get('precision_neg', 0):.4f}")
    print(f"Val Recall (Non-Hate)  : {best_metrics.get('recall_neg', 0):.4f}")
    print(f"Val F1 (Non-Hate)      : {best_metrics.get('f1_neg', 0):.4f}")
    print(f"Val Macro F1           : {best_metrics.get('macro_f1', 0):.4f}")
    print(f"Val ROC-AUC            : {best_metrics.get('roc_auc', 0):.4f}")
    print(f"Val Loss               : {best_metrics.get('loss', 0):.4f}")
    print("-"*70)
    print(f"Train Accuracy         : {best_metrics.get('train_accuracy', 0):.4f}")
    print(f"Train Precision (Hate) : {best_metrics.get('train_precision', 0):.4f}")
    print(f"Train Recall (Hate)    : {best_metrics.get('train_recall', 0):.4f}")
    print(f"Train F1 (Hate)        : {best_metrics.get('train_f1', 0):.4f}")
    print(f"Train Macro F1         : {best_metrics.get('train_macro_f1', 0):.4f}")
    print(f"Train ROC-AUC          : {best_metrics.get('train_roc_auc', 0):.4f}")
    print(f"Train Loss             : {best_metrics.get('train_loss', 0):.4f}")
    print("="*70)
    print("Model Size & Pruning Stats")
    print("="*70)
    print(f"Total params           : {model_metrics.get('total_params', 0):,}")
    print(f"Trainable params       : {model_metrics.get('trainable', 0):,}")
    print(f"Model size (MB)        : {model_metrics.get('size_mb', 0)}")
    print(f"Sparsity               : {model_metrics.get('sparsity_%', 0)}%")
    print(f"Non-zero weights       : {model_metrics.get('non_zero', 0):,}")
    print(f"Total weights          : {model_metrics.get('total_weights', 0):,}")
    print("="*70 + "\n")

def run_kfold(config):
    set_seed(config.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)

    comments, labels = load_and_preprocess_data(config.dataset_path)
    class_weight = get_class_weight(labels)

    all_fold_results = []
    best_macro_f1 = -1
    best_model_state = None
    best_fold_idx = -1
    best_train_metrics = None

    print(f"\nStarting {config.num_folds}-Fold Cross Validation...\n")

    for fold, (train_idx, val_idx) in enumerate(get_kfold_splits(comments, labels, config.num_folds, stratify=True)):
        print(f"\n{'='*20} FOLD {fold+1}/{config.num_folds} {'='*20}")

        train_comments, val_comments = comments[train_idx], comments[val_idx]
        train_labels, val_labels = labels[train_idx], labels[val_idx]

        train_ds = HateSpeechDataset(train_comments, train_labels, tokenizer, config.max_length)
        val_ds = HateSpeechDataset(val_comments, val_labels, tokenizer, config.max_length)

        train_loader = DataLoader(train_ds, batch_size=config.batch, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=config.batch, shuffle=False, num_workers=2, pin_memory=True)

        # Wanda calibration
        calib_texts = None
        if config.prune and config.prune_method == 'wanda':
            calib_texts = get_calibration_samples(comments, labels, config.calib_samples, config.seed)

        model = HateSpeechModel(
            student_name=config.model_path,
            teacher_name=config.teacher if config.distill else None,
            dropout=config.dropout,
            prune=config.prune,
            prune_method=config.prune_method,
            sparsity=config.prune_sparsity,
            global_prune=config.prune_global,
            calib_texts=calib_texts,
            device=device
        ).to(device)

        if config.freeze_base:
            for p in model.student.parameters():
                p.requires_grad = False

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], 
            lr=config.lr, weight_decay=config.weight_decay
        )
        total_steps = len(train_loader) * config.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=int(total_steps * config.warmup_ratio), 
            num_training_steps=total_steps
        )

        best_val_macro = -1
        best_state = None
        patience_counter = 0
        global_step = [0]

        best_train_metrics_this_fold = None

        for epoch in range(1, config.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, config, class_weight, global_step)
            
            # Evaluate train and val
            train_metrics = evaluate(model, train_loader, device)
            val_metrics = evaluate(model, val_loader, device)

            print(f"Epoch {epoch:2d} | Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | Val Macro F1: {val_metrics['macro_f1']:.4f}")

            if val_metrics['macro_f1'] > best_val_macro:
                best_val_macro = val_metrics['macro_f1']
                
                # FIXED: Make permanent BEFORE saving state_dict
                if config.prune and config.prune_method == 'gradual_magnitude':
                    print("Making pruning permanent before saving best state...")
                    model.make_pruning_permanent()
                
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                best_train_metrics_this_fold = train_metrics.copy()
                best_train_metrics_this_fold.update({'train_loss': train_loss})
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.early_stopping_patience:
                    print("Early stopping triggered")
                    break

        # Fold result (use last eval for train, but best for val)
        result = val_metrics.copy()
        result.update({
            'fold': fold + 1,
            'train_loss': train_loss,
            'train_accuracy': train_metrics['accuracy'],
            'train_precision': train_metrics['precision'],
            'train_recall': train_metrics['recall'],
            'train_f1': train_metrics['f1'],
            'train_macro_f1': train_metrics['macro_f1'],
            'train_roc_auc': train_metrics['roc_auc'],
            'threshold': val_metrics['threshold']  # Fixed key
        })
        all_fold_results.append(result)

        if best_val_macro > best_macro_f1:
            best_macro_f1 = best_val_macro
            best_model_state = best_state
            best_fold_idx = fold
            best_train_metrics = best_train_metrics_this_fold

        print(f"Fold {fold+1} Best Val Macro F1: {best_val_macro:.4f}\n")

    # FINAL BEST MODEL
    print(f"\nBEST MODEL FROM FOLD {best_fold_idx + 1} (Macro F1 = {best_macro_f1:.4f})")

    # Create final model WITHOUT pruning (will load permanent state)
    final_model = HateSpeechModel(
        student_name=config.model_path,
        teacher_name=None,  # No teacher for inference
        dropout=config.dropout,
        prune=False  # Key: No re-pruning
    ).to(device)
    
    # Load the permanent state (no more error!)
    final_model.load_state_dict(best_model_state)

    # Save for Hugging Face
    save_dir = "./bangla-hate-distill-pruned"
    os.makedirs(save_dir, exist_ok=True)
    final_model.student.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    torch.save(final_model.classifier.state_dict(), os.path.join(save_dir, "classifier.pt"))
    print(f"Model saved successfully to {save_dir}")

    # Final stats
    model_stats = get_model_stats(final_model)
    model_stats['non_zero'] = sum((p != 0).sum().item() for p in final_model.parameters() if p.ndim > 1)
    model_stats['total_weights'] = sum(p.numel() for p in final_model.parameters() if p.ndim > 1)

    # Combine best metrics
    best_result = all_fold_results[best_fold_idx].copy()
    best_result.update({
        'train_accuracy': best_train_metrics['accuracy'],
        'train_precision': best_train_metrics['precision'],
        'train_recall': best_train_metrics['recall'],
        'train_f1': best_train_metrics['f1'],
        'train_macro_f1': best_train_metrics['macro_f1'],
        'train_roc_auc': best_train_metrics['roc_auc'],
        'train_loss': best_train_metrics['train_loss'],
        'threshold': best_result.get('threshold', 0.5),  # Ensure key exists
        'precision_neg': best_result.get('precision_neg', 0),
        'recall_neg': best_result.get('recall_neg', 0),
        'f1_neg': best_result.get('f1_neg', 0),
    })

    # PRINT THE FULL TABLE
    print_experiment_summary(best_fold_idx, best_result, model_stats)

    # SAVE CSVs
    df = pd.DataFrame(all_fold_results)
    df.to_csv("5_fold_cv_results.csv", index=False)
    print(f"\n5-fold results saved to: 5_fold_cv_results.csv")
    print(df[['fold', 'macro_f1', 'f1', 'accuracy', 'roc_auc']].round(4))

    best_df = pd.Series(best_result)
    best_df.to_csv("BEST_MODEL_FULL_METRICS.csv")
    print(f"\nBest model metrics saved to: BEST_MODEL_FULL_METRICS.csv")

    # MLflow note (if you want it later)
    if hasattr(config, 'mlflow_experiment_name'):
        print(f"MLflow experiment '{config.mlflow_experiment_name}' ready for logging (add mlflow code if needed)")

if __name__ == "__main__":
    config = parse_arguments()
    run_kfold(config)
