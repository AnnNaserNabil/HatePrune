# model.py (core changes: Wanda integration + gradual prune logic)
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from transformers import AutoModel, AutoConfig
import wandb  # From locuslab/wanda; assumes installed

class Model(nn.Module):
    def __init__(self, student_name, teacher_name=None, dropout=0.1,
                 prune=False, prune_method='baseline', prune_sparsity=0.6, prune_global=False,
                 calib_texts=None, device='cuda'):
        super().__init__()
        self.device = device
        self.prune_method = prune_method
        self.prune_sparsity = prune_sparsity
        self.prune_global = prune_global

        # Student
        self.student = AutoModel.from_pretrained(student_name)
        student_cfg = AutoConfig.from_pretrained(student_name)
        self.student_hidden = student_cfg.hidden_size

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.student_hidden, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

        # Teacher (frozen, unchanged)
        self.teacher = None
        self.teacher_projection = None
        if teacher_name:
            self.teacher = AutoModel.from_pretrained(teacher_name)
            teacher_cfg = AutoConfig.from_pretrained(teacher_name)
            teacher_hidden = teacher_cfg.hidden_size
            if teacher_hidden != self.student_hidden:
                print(f"Warning: Hidden size mismatch: Teacher {teacher_hidden} → Student {self.student_hidden}")
                self.teacher_projection = nn.Linear(teacher_hidden, self.student_hidden)
            else:
                self.teacher_projection = nn.Identity()
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad = False
            if self.teacher_projection is not None:
                for p in self.teacher_projection.parameters():
                    p.requires_grad = False

        # Apply pruning if enabled
        if prune and prune_method != 'baseline':
            if prune_method == 'wanda':
                self.apply_wanda_pruning(calib_texts)
            elif prune_method == 'gradual_magnitude':
                self.prepare_gradual_pruning()

    def apply_wanda_pruning(self, calib_texts):
        print(f"Applying Wanda pruning (sparsity={self.prune_sparsity}) with {len(calib_texts)} calib samples...")
        from wanda import measure_importance, sparsify  # From locuslab/wanda
        tokenizer = AutoTokenizer.from_pretrained(self.student.name_or_path)  # Reuse
        calib_encodings = tokenizer(calib_texts, return_tensors='pt', padding=True, truncation=True).to(self.device)

        # Measure importance (activations on calib data)
        importance = measure_importance(self.student, calib_encodings.input_ids, calib_encodings.attention_mask)

        # Sparsify (one-shot)
        sparsify(self.student, importance, self.prune_sparsity, group_type='columns')  # Per-output, unstructured
        print("Wanda pruning complete.")

    def prepare_gradual_pruning(self):
        print(f"Preparing gradual magnitude pruning (target={self.prune_sparsity}, global={self.prune_global})...")
        self.modules_to_prune = []
        for name, module in self.student.named_modules():
            if isinstance(module, nn.Linear):
                self.modules_to_prune.append((module, 'weight'))
        # Add classifier layers to prune
        for name, module in self.classifier.named_modules():
            if isinstance(module, nn.Linear):
                self.modules_to_prune.append((module, 'weight'))
        # Initial mask (start at 0 sparsity)
        if self.prune_global:
            prune.global_unstructured(self.modules_to_prune, pruning_method=prune.L1Unstructured, amount=0.0)

    def gradual_prune_step(self, current_step, total_steps):
        if self.prune_method != 'gradual_magnitude':
            return
        # Cubic schedule for gradual increase
        progress = min(1.0, current_step / total_steps)
        current_sparsity = self.prune_sparsity * (progress ** 3)  # Cubic ramp-up
        current_sparsity = min(current_sparsity, self.prune_sparsity)
        print(f"Gradual prune at step {current_step}/{total_steps}: current sparsity {current_sparsity:.4f}")

        if self.prune_global:
            prune.global_unstructured(self.modules_to_prune, pruning_method=prune.L1Unstructured, amount=current_sparsity)
        else:
            for module, param in self.modules_to_prune:
                prune.l1_unstructured(module, name=param, amount=current_sparsity)

    def make_pruning_permanent(self):
        print("Making pruning permanent...")
        for module in self.student.modules():
            if hasattr(module, 'weight') and prune.is_pruned(module):
                prune.remove(module, 'weight')
        for module in self.classifier.modules():
            if hasattr(module, 'weight') and prune.is_pruned(module):
                prune.remove(module, 'weight')
        # For Wanda, already permanent (direct zeroing)

    def forward(self, input_ids, attention_mask, labels=None):
        s_out = self.student(input_ids=input_ids, attention_mask=attention_mask)
        s_cls = s_out.last_hidden_state[:, 0]
        s_logits = self.classifier(s_cls).squeeze(-1)

        out = {'logits': s_logits}

        if self.teacher is not None:
            with torch.no_grad():
                t_out = self.teacher(input_ids=input_ids, attention_mask=attention_mask)
                t_cls = t_out.last_hidden_state[:, 0]
                t_cls_proj = self.teacher_projection(t_cls)
                t_logits = self.classifier(t_cls_proj).squeeze(-1)
            out['teacher_logits'] = t_logits

        if labels is not None:
            out['labels'] = labels
        return out
