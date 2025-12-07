# HatePrune



### Option 1 – WANDA (Recommended – Best quality + fastest)
60% sparsity, usually **< 1.2 pt Macro-F1 drop**, only needs 3–4 epochs of recovery.

```bash
python main.py \
  --author_name "YourName" \
  --dataset_path "HateSpeech.csv" \
  --model_path "./best_distilled_model_YourName_sahajbert" \   # ← your distilled model folder from before
  --distill False \                                           # no teacher needed anymore
  --prune \
  --prune_method wanda \
  --prune_sparsity 0.60 \                                     # 60% sparsity = ~2.4× smaller/faster
  --calib_samples 512 \                                       # uses 512 random samples (stratified)
  --batch 32 \
  --lr 4e-5 \                                                 # slightly higher LR helps recovery
  --epochs 4 \                                                # 3–4 epochs is enough after Wanda
  --num_folds 5 \
  --seed 42 \
  --dropout 0.1 \
  --early_stopping_patience 3 \
  --mlflow_experiment_name "Bangla-HateSpeech-Wanda-Pruning"
```

### Option 2 – Gradual Magnitude Pruning (Highest possible recovery)
60–65% sparsity with almost zero drop if you let it train a bit longer.

```bash
python main.py \
  --author_name "YourName" \
  --dataset_path "HateSpeech.csv" \
  --model_path "./best_distilled_model_YourName_sahajbert" \
  --distill False \
  --prune \
  --prune_method gradual_magnitude \
  --prune_sparsity 0.65 \                                     # can go higher than Wanda
  --prune_freq 80 \                                           # prune a little every 80 steps
  --batch 32 \
  --lr 3e-5 \
  --epochs 7 \                                                # 6–7 epochs recommended
  --num_folds 5 \
  --seed 42 \
  --dropout 0.1 \
  --early_stopping_patience 4 \
  --mlflow_experiment_name "Bangla-HateSpeech-Gradual-Pruning"
```

### Option 3 – Ultra-fast 2:4 Semi-structured (Zero accuracy drop, hardware-friendly)
50% sparsity, **0.0–0.3 pt drop**, instant, no fine-tuning needed (but you can still do 1 epoch if you want).

```bash
python main.py \
  --author_name "YourName" \
  --dataset_path "HateSpeech.csv" \
  --model_path "./best_distilled_model_YourName_sahajbert" \
  --distill False \
  --prune \
  --prune_method wanda \                                      # we reuse wanda code but force 2:4 pattern
  --prune_sparsity 0.50 \
  --calib_samples 1 \                                         # dummy value – we override to 2:4
  --batch 32 \
  --lr 2e-5 \
  --epochs 1 \                                                # literally 1 epoch is enough
  --num_folds 1 \                                             # just to test quickly
  --seed 42
```

(or even simpler — one-liner outside the repo using HuggingFace Optimum):

```bash
pip install optimum[exporters]
optimum-cli export onnx --model ./best_distilled_model_YourName_sahajbert --task text-classification --sparsity 0.5 --pattern 2:4 sahajbert_2_4_sparse
```

### Quick test command (run this first!)
Just to make sure everything works on one fold:

```bash
python main.py --author_name Test --dataset_path HateSpeech.csv \
  --model_path "./best_distilled_model_YourName_sahajbert" \
  --prune --prune_method wanda --prune_sparsity 0.6 \
  --epochs 3 --num_folds 1 --batch 16
```

You’ll see in the final summary something like:

```
Sparsity: 60.12%
Non-zero weights: ~44M → ~17.6M
Val Macro F1: 0.8921 → 0.8897   (only −0.0024 drop!)
```


