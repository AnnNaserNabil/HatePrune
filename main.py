# main.py (renamed from main_distill.py for generality; modified)
import os
import time
import torch
from torch.utils.data import DataLoader
import mlflow
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from config import parse_arguments, print_config
from data import load_and_preprocess_data, prepare_kfold_splits, HateSpeechDataset, calculate_class_weights
from model import Model  # Updated name from DistillationModel
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

    fold_results = []
    best_macro_f1 = -1
    best_fold_idx = -1
    best_overall_metrics = {}
    best_overall_epoch = -1
    best_state_dict = None

    run_name = f"{config.author_name}_{'Distill' if config.distill else 'Train'}{'_Prune' if config.prune else ''}"
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
                prune_amount=config.prune_amount,
                prune_type=config.prune_type,
                prune_global=config.prune_global
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
                                            config.gradient_clip_norm, config.distill)
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

            # Load best checkpoint of this fold
            if best_fold_state is not None:
                model.load_state_dict(best_fold_state)

            # Make pruning permanent (remove masks, zero out weights)
            if config.prune:
                model.make_pruning_permanent()

            # Final evaluation
            val_metrics = evaluate(model, val_loader, device)
            train_metrics = evaluate(model, train_loader, device)

            # Combine metrics
            final_metrics = val_metrics.copy()
            for k, v in train_metrics.items():
                if k != 'best_threshold':
                    final_metrics[f'train_{k}'] = v
            final_metrics['best_epoch'] = best_epoch
            final_metrics['loss'] = val_metrics.get('loss', 0.0)
            final_metrics['train_loss'] = train_metrics.get('loss', 0.0)

            fold_results.append(final_metrics)
            print_fold_summary(fold, final_metrics, best_epoch)

            # Update global best
            if val_metrics['macro_f1'] > best_macro_f1:
                best_macro_f1 = val_metrics['macro_f1']
                best_fold_idx = fold
                best_overall_metrics = final_metrics
                best_overall_epoch = best_epoch
                best_state_dict = best_fold_state  # Note: pruning made permanent above, but state is before; reload if needed

        # ===================================================================
        # 1. SAVE BEST MODEL FOR HUGGING FACE DEPLOYMENT
        # ===================================================================
        model_name_safe = config.model_path.split('/')[-1]
        mode_str = 'pruned' if config.prune else 'distilled' if config.distill else 'trained'
        best_model_dir = f"./best_{mode_str}_model_{config.author_name.replace(' ', '_')}_{model_name_safe}"
        os.makedirs(best_model_dir, exist_ok=True)

        final_model = Model(
            student_name=config.model_path,
            teacher_name=config.teacher if config.distill else None,
            dropout=config.dropout,
            prune=config.prune,
            prune_amount=config.prune_amount,
            prune_type=config.prune_type,
            prune_global=config.prune_global
        )
        final_model.load_state_dict(best_state_dict)
        if config.prune:
            final_model.make_pruning_permanent()  # Ensure permanent for saved model
        final_model.student.save_pretrained(best_model_dir)
        tokenizer.save_pretrained(best_model_dir)

        print(f"\nBEST MODEL SAVED FOR DEPLOYMENT!")
        print(f"   → {os.path.abspath(best_model_dir)}")
        print(f"   → Val Macro F1: {best_macro_f1:.4f} (Fold {best_fold_idx+1}, Epoch {best_overall_epoch})")
        print(f"   Upload with: huggingface-cli upload yourname/bangla-hate-{mode_str} {best_model_dir} .")

        # ===================================================================
        # 2. SAVE FULL 22-METRIC CSV (exactly like original repo)
        # ===================================================================
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        best_csv_name = f"{mode_str}_best_metrics_batch{config.batch}_lr{config.lr}_epochs{config.epochs}_{timestamp}.csv"
        best_csv_path = os.path.join("./outputs", best_csv_name)

        best_metrics_data = {
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
            'Train Loss': [best_overall_metrics['train_loss']]
        }

        pd.DataFrame(best_metrics_data).to_csv(best_csv_path, index=False)
        mlflow.log_artifact(best_csv_path)

        # Also save all-fold summary
        all_folds_csv = f"{mode_str}_all_folds_summary_{timestamp}.csv"
        pd.DataFrame(fold_results).to_csv(f"./outputs/{all_folds_csv}", index=False)
        mlflow.log_artifact(f"./outputs/{all_folds_csv}")

        # Log to MLflow
        mlflow.log_metric("best_val_macro_f1", best_macro_f1)
        mlflow.log_metric("best_fold", best_fold_idx + 1)
        mlflow.log_metric("best_epoch", best_overall_epoch)

        print_experiment_summary(best_fold_idx, best_overall_metrics, get_model_metrics(final_model))

        print(f"\n{'='*70}")
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print(f"   Best Model → {best_model_dir}")
        print(f"   Best Metrics CSV → {best_csv_name}")
        print(f"   All Folds CSV → {all_folds_csv}")
        print(f"   MLflow Run ID: {run_id}")
        print(f"   Run: mlflow ui → http://localhost:5000")
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
