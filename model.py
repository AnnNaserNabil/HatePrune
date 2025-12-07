# model.py (renamed from model_distill.py; modified for pruning)
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from transformers import AutoModel, AutoConfig

class Model(nn.Module):
    def __init__(self, student_name, teacher_name=None, dropout=0.1,
                 prune=False, prune_amount=0.5, prune_type='l1_unstructured', prune_global=False):
        super().__init__()
        # Student
        self.student = AutoModel.from_pretrained(student_name)
        student_cfg = AutoConfig.from_pretrained(student_name)
        self.student_hidden = student_cfg.hidden_size

        # Classification head (on student hidden size)
        self.classifier = nn.Sequential(
            nn.Linear(self.student_hidden, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

        # Teacher (frozen)
        self.teacher = None
        self.teacher_projection = None
        if teacher_name:
            self.teacher = AutoModel.from_pretrained(teacher_name)
            teacher_cfg = AutoConfig.from_pretrained(teacher_name)
            teacher_hidden = teacher_cfg.hidden_size

            # Only add projection if sizes differ
            if teacher_hidden != self.student_hidden:
                print(f"Warning: Hidden size mismatch: Teacher {teacher_hidden} → Student {self.student_hidden}")
                print("Adding projection layer...")
                self.teacher_projection = nn.Linear(teacher_hidden, self.student_hidden)
            else:
                self.teacher_projection = nn.Identity()

            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad = False
            if self.teacher_projection is not None:
                for p in self.teacher_projection.parameters():
                    p.requires_grad = False  # Freeze projection

        # Apply pruning to student if enabled (one-shot before training)
        if prune:
            self.apply_pruning(amount=prune_amount, prune_type=prune_type, global_prune=prune_global)

    def apply_pruning(self, amount, prune_type, global_prune):
        print(f"Applying {prune_type} pruning (amount={amount}, global={global_prune}) to student model...")
        modules_to_prune = []
        for name, module in self.student.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):  # Target linear/conv layers in BERT (attention, FFN)
                modules_to_prune.append((module, 'weight'))

        if global_prune:
            prune.global_unstructured(
                modules_to_prune,
                pruning_method=getattr(prune, prune_type),
                amount=amount
            )
        else:
            for module, param in modules_to_prune:
                getattr(prune, prune_type)(module, param, amount=amount)

    def make_pruning_permanent(self):
        print("Making pruning permanent (removing masks, zeroing weights)...")
        for module in self.student.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                if prune.is_pruned(module):
                    prune.remove(module, 'weight')

    def forward(self, input_ids, attention_mask, labels=None):
        # Student forward
        s_out = self.student(input_ids=input_ids, attention_mask=attention_mask)
        s_cls = s_out.last_hidden_state[:, 0]  # [B, student_hidden]
        s_logits = self.classifier(s_cls).squeeze(-1)

        out = {'logits': s_logits}

        if self.teacher is not None:
            with torch.no_grad():
                t_out = self.teacher(input_ids=input_ids, attention_mask=attention_mask)
                t_cls = t_out.last_hidden_state[:, 0]  # [B, teacher_hidden]
                t_cls_proj = self.teacher_projection(t_cls)  # → [B, student_hidden]
                t_logits = self.classifier(t_cls_proj).squeeze(-1)
            out['teacher_logits'] = t_logits

        if labels is not None:
            out['labels'] = labels
        return out
