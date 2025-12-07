# model.py
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from transformers import AutoModel, AutoConfig

class HateSpeechModel(nn.Module):
    def __init__(self, model_name, dropout=0.1, prune=False, prune_method='none', sparsity=0.6):
        super().__init__()
        self.student = AutoModel.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)
        hidden_size = config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

        self.prune = prune
        self.prune_method = prune_method
        self.sparsity = sparsity

        if prune and prune_method == 'gradual_magnitude':
            self._prepare_gradual_pruning()

    def _prepare_gradual_pruning(self):
        print(f"Preparing gradual magnitude pruning → {self.sparsity*100}% sparsity")
        self.prunable_modules = []
        for name, module in self.student.named_modules():
            if isinstance(module, (nn.Linear)):
                self.prunable_modules.append((module, 'weight'))

    def apply_gradual_pruning(self, current_step, total_steps):
        if not hasattr(self, 'prunable_modules'):
            return
        current_sparsity = min(self.sparsity * (current_step / total_steps), self.sparsity)
        for module, name in self.prunable_modules:
            prune.l1_unstructured(module, name=name, amount=current_sparsity)

    def make_pruning_permanent(self):
        if not hasattr(self, 'prunable_modules'):
            return
        print("Making pruning permanent (removing masks)...")
        for module, name in self.prunable_modules:
            if prune.is_pruned(module):
                prune.remove(module, name)

    def forward(self, input_ids, attention_mask):
        outputs = self.student(input_ids=input_ids, attention_mask=attention_mask)
        cls_hidden = outputs.last_hidden_state[:, 0]
        logits = self.classifier(cls_hidden).squeeze(-1)
        return logits
