# config.py (modified)
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Knowledge Distillation or Pruning for Bangla Hate Speech Detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Core training
    parser.add_argument('--batch', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=12, help='Max epochs')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to CSV')
    parser.add_argument('--model_path', type=str, default='neuropark/sahajBERT',
                        help='Student model (HuggingFace name/path); for pruning, use path to distilled model')
    parser.add_argument('--teacher', type=str, default='google-bert/bert-base-multilingual-cased',
                        help='Teacher model (HuggingFace name/path); ignored if --distill False')
    parser.add_argument('--max_length', type=int, default=128, help='Max sequence length')
    parser.add_argument('--num_folds', type=int, default=5, help='K-fold CV')
    parser.add_argument('--freeze_base', action='store_true', help='Freeze student base')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--stratification_type', type=str, default='binary',
                        choices=['binary', 'none'], help='Stratification type')

    # Distillation
    parser.add_argument('--distill', action='store_true', help='Enable distillation (use teacher)')
    parser.add_argument('--alpha', type=float, default=0.7,
                        help='Weight of soft loss (0=hard only, 1=soft only)')
    parser.add_argument('--temperature', type=float, default=4.0,
                        help='Temperature for soft labels')

    # Pruning (new options)
    parser.add_argument('--prune', action='store_true', help='Enable pruning on student model')
    parser.add_argument('--prune_amount', type=float, default=0.5,
                        help='Fraction of weights to prune (e.g., 0.5 for 50% sparsity)')
    parser.add_argument('--prune_type', type=str, default='l1_unstructured',
                        choices=['l1_unstructured', 'random_unstructured'],
                        help='Pruning method (from torch.nn.utils.prune)')
    parser.add_argument('--prune_global', action='store_true', help='Apply global pruning instead of per-layer')

    # MLflow & others
    parser.add_argument('--author_name', type=str, required=True, help='Your name')
    parser.add_argument('--mlflow_experiment_name', type=str, default='Bangla-HateSpeech-Distill-Prune')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--gradient_clip_norm', type=float, default=1.0)
    parser.add_argument('--early_stopping_patience', type=int, default=5)

    args = parser.parse_args()

    # Validation
    if args.batch <= 0: raise ValueError("Batch size must be positive")
    if args.lr <= 0: raise ValueError("Learning rate must be positive")
    if args.epochs <= 0: raise ValueError("Epochs must be positive")
    if args.num_folds < 2: raise ValueError("Folds >= 2")
    if not (0 <= args.dropout < 1): raise ValueError("Dropout in [0,1)")
    if not (0 <= args.warmup_ratio <= 1): raise ValueError("Warmup ratio in [0,1]")
    if args.prune and not (0 < args.prune_amount < 1): raise ValueError("Prune amount in (0,1)")
    if args.distill and args.prune: print("Warning: Distillation and pruning enabled together—pruning applied to student.")

    return args


def print_config(config):
    print("\n" + "="*60)
    print("CONFIGURATION")
    print("="*60)
    print(f"Mode: {'Distillation' if config.distill else 'Standard Training'}{' + Pruning' if config.prune else ''}")
    print(f"Student: {config.model_path}")
    if config.distill:
        print(f"Teacher: {config.teacher}")
        print(f"Alpha (soft weight): {config.alpha}")
        print(f"Temperature: {config.temperature}")
    if config.prune:
        print(f"Prune: Yes | Amount: {config.prune_amount} | Type: {config.prune_type} | Global: {config.prune_global}")
    print(f"Batch: {config.batch} | LR: {config.lr} | Epochs: {config.epochs}")
    print(f"K-Folds: {config.num_folds} | Stratify: {config.stratification_type}")
    print(f"Freeze Base: {config.freeze_base}")
    print(f"Author: {config.author_name}")
    print("="*60 + "\n")
