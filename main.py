# main.py (final fixed version: correct pruning handling, clean saving, consistent metrics)
import os
import time
import torch
from torch.utils.data import DataLoader
import mlflow
import pandas as pd
from transformers import AutoTokenizer
from config import parse_arguments, print_config
from data import load_and_preprocess_data, prepare_kfold_splits, HateSpeechDataset, calculate_class_weights, get_calibration_samples
from model import Model
from train import train_epoch, evaluate
from utils import set_seed, get_model_metrics, print_experiment_summary, print_fold_summary
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup


def run_kfold(config, comments, labels, tokenizer, device):
    os.makedirs('./outputs', exist_ok=True)
    os.makedirs('./cache', exist_ok=True)

    mlflow.set_tracking_uri("file://./mlruns")
    mlflow.set_experiment(config.mlflow_experiment_name)

    class_weights = calculate_class_weights(labels)
    splits = list(prepare_kfold_splits(comments, labels, config.num_folds,
                                       config.stratification_type, config.seed))

    # For Wanda: Get calib samples once (outside folds)
    calib_texts = None
    if config.prune and config.prune_method == 'wanda':
        calib_texts = get_calibration_samples(comments, labels, config.calib_samples, config.seed)

    fold_results = []
    best_macro_f1 = -1
    best_fold_idx = -1
    best_overall_metrics = {}
    best_overall_epoch = -1
    best_state_dict = None  # Will hold permanently pruned (clean) weights

    run_name = f"{config.author_name}_{'Distill' if config.distill else 'Train'}{'_Prune' if config.prune else ''}_{config.prune_method}"
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        mlflow.log_params(vars(config))

        for fold, (train_idx, val_idx) in enumerate(splits):
            print(f"\n{'='*30} FOLD {fold+1}/{config.num_folds} {'='*30}")

            train_ds = HateSpeechDataset(comments[train_idx], labels[train_idx], tokenizer, config.max_length)
            val_ds = HateSpeechDataset(comments[val_idx], labels[val_idx], tokenizer, config.max_length)

            train_loader = DataLoader(train_ds, batch_size=config.batch, shuffle=True, num_workers=4, pin_memory=True)
            val_loader = DataLoader(val_ds, batch_size=config.batch, shuffle=False, num_workers=4, pin_memory=True)

            model = Model(
                student_name=config.model_path,
                teacher_name=config.teacher if config.distill else None,
                dropout=config.dropout,
                prune=config.prune,
                prune_method=config.prune_method,
                prune_sparsity=config.prune_sparsity,
                prune_global=config.prune_global,
                calib_texts=calib_texts,
                device=device
            ).to(device)

            if config.freeze_base:
                for p in model.student.parameters():
                    p.requires_grad = False

            optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=config.lr, weight_decay=config.weight_decay)
            total_steps = len(train_loader) * config.epochs
            scheduler = get_linear_schedule_with_warmup(optimizer,
                num_warmup_steps=int(config.warmup_ratio * total_steps),
                num_training_steps=total_steps)

            # Per-fold early stopping
            best_val_macro = -1
            best_fold_state = None
            best_epoch = 0
            patience_counter = 0

            for epoch in range(config.epochs):
                train_metrics = train_epoch(model, train_loader, optimizer, scheduler, device,
                                            class_weights, config.temperature, config.alpha,
                                            config.gradient_clip_norm, config.distill,
                                            config.prune_method, config.prune_freq,
                                            config.prune_sparsity, config.epochs)
                val_metrics = evaluate(model, val_loader, device)

                if val_metrics['macro_f1'] > best_val_macro:
                    best_val_macro = val_metrics['macro_f1']
                    best_fold_state = {k: v.cpu() for k, v in model.state_dict().items()}
                    best_epoch = epoch + 1
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= config.early_stopping_patience:
                        print(f"Early stopping triggered at epoch {epoch+1}")
                        break

            # Load the best checkpoint for this fold
            if best_fold_state is not None:
                model.load_state_dict(best_fold_state)

            # === CRITICAL: Make pruning permanent BEFORE saving or updating global best ===
            if config.prune:
                model.make_pruning_permanent()

            # Final evaluation on the permanently pruned model
            val_metrics = evaluate(model, val_loader, device)
            train_metrics = evaluate(model, train_loader, device)

            # Combine metrics
            final_metrics = val_metrics.copy()
            for k, v in train_metrics.items():
                if k != 'best_threshold':
                    final_metrics[f'train_{k}'] = v
            final_metrics['best_epoch'] = best_epoch

            fold_results.append(final_metrics)
            print_fold_summary(fold, final_metrics, best_epoch)

            # Update global best — using the clean, permanently pruned state dict
            if val_metrics['macro_f1'] > best_macro_f1:
                best_macro_f1 = val_metrics['macro_f1']
                best_fold_idx = fold
                best_overall_metrics = final_metrics
                best_overall_epoch = best_epoch
                best_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}  # Clean weights (no _orig/_mask)

        # ====================== SAVE BEST MODEL ======================
        model_name_safe = config.model_path.split('/')[-1]
        mode_str = f"{config.prune_method}_pruned" if config.prune else 'distilled' if config.distill else 'trained'
        best_model_dir = f"./best_{mode_str}_model_{config.author_name.replace(' ', '_')}_{model_name_safe}"
        os.makedirs(best_model_dir, exist_ok=True)

        # Create final model WITHOUT any pruning logic (clean load)
        final_model = Model(
            student_name=config.model_path,
            teacher_name=config.teacher if config.distill else None,
            dropout=config.dropout,
            prune=False,               # No pruning preparation
            prune_method='baseline',   # Safe default
            prune_sparsity=config.prune_sparsity,
            prune_global=config.prune_global,
            calib_texts=None,
            device=device
        ).to(device)

        # Load the clean state dict (already has real zeros)
        final_model.load_state_dict(best_state_dict)

        # Get final model metrics (pruning is permanent)
        model_metrics = get_model_metrics(final_model)

        # Save the pruned model
        final_model.student.save_pretrained(best_model_dir)
        tokenizer.save_pretrained(best_model_dir)

        print(f"\nBEST MODEL SAVED!")
        print(f"   → {os.path.abspath(best_model_dir)}")
        print(f"   → Val Macro F1: {best_macro_f1:.4f} (Fold {best_fold_idx+1}, Epoch {best_overall_epoch})")

        # ====================== SAVE METRICS CSVs ======================
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        best_csv_name = f"{mode_str}_best_metrics_batch{config.batch}_lr{config.lr}_epochs{config.epochs}_{timestamp}.csv"
        best_csv_path = os.path.join("./outputs", best_csv_name)

        best_metrics_data = {
            'Epochs': [config.epochs],
            'Best Fold': [f'Fold {best_fold_idx + 1}'],
            'Best Epoch': [best_overall_epoch],
            'Val Accuracy': [best_overall_metrics['accuracy']],
            'Val Precision (Hate)': [best_overall_metrics['precision']],
            'Val Recall (Hate)': [best_overall_metrics['recall']],
            'Val F1 (Hate)': [best_overall_metrics['f1']],
            'Val Precision (Non-Hate)': [best_overall_metrics['precision_negative']],
            'Val Recall (Non-Hate)': [best_overall_metrics['recall_negative']],
            'Val F1 (Non-Hate)': [best_overall_metrics['f1_negative']],
            'Val Macro F1': [best_overall_metrics['macro_f1']],
            'Val ROC-AUC': [best_overall_metrics['roc_auc']],
            'Val Loss': [best_overall_metrics['loss']],
            'Best Threshold': [best_overall_metrics['best_threshold']],
            'Train Accuracy': [best_overall_metrics['train_accuracy']],
            'Train Precision (Hate)': [best_overall_metrics['train_precision']],
            'Train Recall (Hate)': [best_overall_metrics['train_recall']],
            'Train F1 (Hate)': [best_overall_metrics['train_f1']],
            'Train Precision (Non-Hate)': [best_overall_metrics['train_precision_negative']],
            'Train Recall (Non-Hate)': [best_overall_metrics['train_recall_negative']],
            'Train F1 (Non-Hate)': [best_overall_metrics['train_f1_negative']],
            'Train Macro F1': [best_overall_metrics['train_macro_f1']],
            'Train ROC-AUC': [best_overall_metrics['train_roc_auc']],
            'Train Loss': [best_overall_metrics['train_loss']],
            'Total Parameters': [model_metrics['total_parameters']],
            'Trainable Parameters': [model_metrics['trainable_parameters']],
            'Model Size (MB)': [model_metrics['model_size_mb']],
            'Total Weights': [model_metrics['total_weights']],
            'Non-zero Weights': [model_metrics['non_zero_weights']],
            'Sparsity (%)': [model_metrics['sparsity_percent']]
        }

        pd.DataFrame(best_metrics_data).to_csv(best_csv_path, index=False)
        mlflow.log_artifact(best_csv_path)

        # All folds summary
        all_folds_csv = f"{mode_str}_all_folds_summary_{timestamp}.csv"
        all_folds_path = f"./outputs/{all_folds_csv}"
        all_folds_data = []
        for fold_idx, metrics in enumerate(fold_results):
            row = {
                'Fold': fold_idx + 1,
                'Best Epoch': metrics['best_epoch'],
                'Val Accuracy': metrics['accuracy'],
                'Val Precision (Hate)': metrics['precision'],
                'Val Recall (Hate)': metrics['recall'],
                'Val F1 (Hate)': metrics['f1'],
                'Val Precision (Non-Hate)': metrics['precision_negative'],
                'Val Recall (Non-Hate)': metrics['recall_negative'],
                'Val F1 (Non-Hate)': metrics['f1_negative'],
                'Val Macro F1': metrics['macro_f1'],
                'Val ROC-AUC': metrics['roc_auc'],
                'Val Loss': metrics['loss'],
                'Best Threshold': metrics['best_threshold'],
                'Train Accuracy': metrics['train_accuracy'],
                'Train Precision (Hate)': metrics['train_precision'],
                'Train Recall (Hate)': metrics['train_recall'],
                'Train F1 (Hate)': metrics['train_f1'],
                'Train Precision (Non-Hate)': metrics['train_precision_negative'],
                'Train Recall (Non-Hate)': metrics['train_recall_negative'],
                'Train F1 (Non-Hate)': metrics['train_f1_negative'],
                'Train Macro F1': metrics['train_macro_f1'],
                'Train ROC-AUC': metrics['train_roc_auc'],
                'Train Loss': metrics['train_loss']
            }
            all_folds_data.append(row)
        pd.DataFrame(all_folds_data).to_csv(all_folds_path, index=False)
        mlflow.log_artifact(all_folds_path)

        mlflow.log_metric("best_val_macro_f1", best_macro_f1)
        mlflow.log_metric("best_fold", best_fold_idx + 1)
        mlflow.log_metric("best_epoch", best_overall_epoch)

        print_experiment_summary(best_fold_idx, best_overall_metrics, model_metrics)

        print(f"\n{'='*70}")
        print("TRAINING COMPLETED!")
        print(f"   Best Model → {best_model_dir}")
        print(f"   Best Metrics CSV → {best_csv_name}")
        print(f"   All Folds CSV → {all_folds_csv}")
        print(f"   MLflow Run ID: {run_id}")
        print("="*70)


if __name__ == "__main__":
    config = parse_arguments()
    print_config(config)
    set_seed(config.seed)

    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    comments, labels = load_and_preprocess_data(config.dataset_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    run_kfold(config, comments, labels, tokenizer, device)
