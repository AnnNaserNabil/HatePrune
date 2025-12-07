# config.py
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Distillation + Pruning Framework for Bangla Hate Speech")
    
    # Core
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, default='neuropark/sahajBERT')
    parser.add_argument('--teacher', type=str, default=None)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--num_folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dropout', type=float, default=0.1)

    # Modes
    parser.add_argument('--distill', action='store_true')
    parser.add_argument('--prune', action='store_true')
    parser.add_argument('--prune_method', type=str, default='gradual_magnitude',
                        choices=['gradual_magnitude', 'wanda', 'none'])
    parser.add_argument('--prune_sparsity', type=float, default=0.6)
    parser.add_argument('--prune_freq', type=int, default=100)

    # Others
    parser.add_argument('--author_name', type=str, required=True)
    parser.add_argument('--mlflow_experiment_name', type=str, default='Bangla-HateSpeech')

    args = parser.parse_args()
    if not args.prune:
        args.prune_method = 'none'
    return args
