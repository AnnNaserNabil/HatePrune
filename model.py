# model.py
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from transformers import AutoModel, AutoConfig, AutoTokenizer

class HateSpeechModel(nn.Module):
    def __init__(self, student_name, teacher_name=None, dropout=0.1,
                 prune=False, prune_method='baseline', sparsity=0.6, global_prune=False,
                 calib_texts=None, device='cuda'):
        super().__init__()
        self.device = device
        self.prune_method = prune_method if prune else 'baseline'

        # Student
        self.student = AutoModel.from_pretrained(student_name)
        config = AutoConfig.from_pretrained(student_name)
        hidden_size = config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 1)

        # Teacher (frozen)
        self.teacher = None
        self.proj = None
        if teacher_name:
            self.teacher = AutoModel.from_pretrained(teacher_name)
            t_config = AutoConfig.from_pretrained(teacher_name)
            if t_config.hidden_size != hidden_size:
                self.proj = nn.Linear(t_config.hidden_size, hidden_size)
            else:
                self.proj = nn.Identity()
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad = False
            if hasattr(self.proj, 'weight'):
                for p in self.proj.parameters():
                    p.requires_grad = False

        # Pruning setup
        if prune and prune_method != 'baseline':
            if prune_method == 'wanda':
                self._apply_wanda(calib_texts, sparsity)
            elif prune_method == 'gradual_magnitude':
                self._prepare_gradual(sparsity, global_prune)

    def _apply_wanda(self, calib_texts, sparsity):
        print(f"Applying Wanda pruning (sparsity={sparsity})...")
        from wanda import measure_importance, sparsify
        tokenizer = AutoTokenizer.from_pretrained(self.student.config._name_or_path)
        enc = tokenizer(calib_texts, padding=True, truncation=True, return_tensors='pt').to(self.device)

        importance = measure_importance(self.student, enc.input_ids, enc.attention_mask)
        sparsify(self.student, importance, sparsity, group_type='columns')
        print("Wanda pruning applied.")

    def _prepare_gradual(self, sparsity, global_prune):
        print(f"Preparing gradual magnitude pruning (target sparsity={sparsity})")
        self.pruning_modules = []
        for name, module in self.student.named_modules():
            if isinstance(module, (nn.Linear)):
                self.pruning_modules.append((module, 'weight'))
        # Start with 0 sparsity
        if global_prune:
            prune.global_unstructured(self.pruning_modules, pruning_method=prune.L1Unstructured, amount=0.0)

    def gradual_prune_step(self, current_step, total_steps):
        if self.prune_method != 'gradual_magnitude':
            return
        target = getattr(self, 'target_sparsity', 0.6)
        current_sparsity = min(target * (current_step / total_steps), target)
        if hasattr(self, 'global_prune') and self.global_prune:
            prune.global_unstructured(self.pruning_modules, pruning_method=prune.L1Unstructured, amount=current_sparsity)
        else:
            for module, name in self.pruning_modules:
                prune.l1_unstructured(module, name, amount=current_sparsity)

    def make_pruning_permanent(self):
        """Call before saving model"""
        print("Making pruning permanent...")
        for module, name in getattr(self, 'pruning_modules', []):
            if prune.is_pruned(module):
                prune.remove(module, name)
        print("Pruning is now permanent.")

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.student(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state[:, 0]
        logits = self.classifier(self.dropout(hidden)).squeeze(-1)

        result = {'logits': logits}

        if self.teacher is not None:
            with torch.no_grad():
                t_out = self.teacher(input_ids=input_ids, attention_mask=attention_mask)
                t_hidden = t_out.last_hidden_state[:, 0]
                t_hidden = self.proj(t_hidden)
                t_logits = self.classifier(self.dropout(t_hidden)).squeeze(-1)
            result['teacher_logits'] = t_logits

        if labels is not None:
            result['labels'] = labels
        return result
