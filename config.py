# config.py
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Bangla Hate Speech - Distill + Prune")
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, default='neuropark/sahajBERT')
    parser.add_argument('--teacher', type=str, default='google-bert/bert-base-multilingual-cased')
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--num_folds', type=int, default=5)
    parser.add_argument('--freeze_base', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--stratification_type', type=str, default='binary', choices=['binary', 'none'])

    # Distillation
    parser.add_argument('--distill', action='store_true')
    parser.add_argument('--alpha', type=float, default=0.7)
    parser.add_argument('--temperature', type=float, default=4.0)

    # Pruning
    parser.add_argument('--prune', action='store_true')
    parser.add_argument('--prune_method', type=str, default='baseline', choices=['baseline', 'wanda', 'gradual_magnitude'])
    parser.add_argument('--prune_sparsity', type=float, default=0.6)
    parser.add_argument('--calib_samples', type=int, default=512)
    parser.add_argument('--prune_freq', type=int, default=100)
    parser.add_argument('--prune_global', action='store_true')

    # Others
    parser.add_argument('--author_name', type=str, required=True)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--gradient_clip_norm', type=float, default=1.0)
    parser.add_argument('--early_stopping_patience', type=int, default=5)

    args = parser.parse_args()

    if args.prune and args.prune_method == 'baseline':
        print("Warning: --prune is enabled but prune_method=baseline → no pruning will be applied.")
    if args.distill and args.prune:
        print("Running Distillation + Pruning")

    return args
