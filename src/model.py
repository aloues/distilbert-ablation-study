"""
src/model.py
Define las cabezas de clasificación y el modelo completo.
Compatible con Hugging Face Trainer: forward devuelve (loss, logits).
No aplica congelamiento (eso es freezing.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
from transformers import AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput

from src.config import DatasetConfig, ExperimentConfig


# ─── Cabezas de clasificación ─────────────────────────────────────────────────

class LinearHead(nn.Module):
    """
    C1, C2, C3: cabeza lineal simple.
    hidden_size → Dropout → Linear → num_labels
    """

    def __init__(self, hidden_size: int, num_labels: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(x))


class BottleneckHead(nn.Module):
    """
    C4: cabeza no lineal con compresión progresiva.
    hidden_size → hidden_dims[0] → ... → hidden_dims[-1] → num_labels
    Con ReLU y Dropout entre capas.
    """

    def __init__(
        self,
        hidden_size: int,
        hidden_dims: List[int],
        num_labels: int,
        dropout: float,
    ):
        super().__init__()
        layers = []
        in_dim = hidden_size
        for out_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, num_labels))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ─── Modelo completo ──────────────────────────────────────────────────────────

class ClassificationModel(nn.Module):
    """
    Modelo completo: backbone (DistilBERT o BERT) + cabeza de clasificación.

    Compatible con HF Trainer:
    - forward(input_ids, attention_mask, labels=None) → SequenceClassifierOutput
    - Si labels se provee, calcula cross-entropy loss.
    - Siempre devuelve logits.
    """

    def __init__(self, backbone: nn.Module, head: nn.Module, num_labels: int):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.num_labels = num_labels
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> SequenceClassifierOutput:
        # DistilBERT y BERT ambos exponen last_hidden_state
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]   # [batch, hidden_size]
        logits = self.head(cls_token)                     # [batch, num_labels]

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )


# ─── Factory ──────────────────────────────────────────────────────────────────

def build_model(exp_cfg: ExperimentConfig, dataset_cfg: DatasetConfig) -> ClassificationModel:
    """
    Construye backbone + cabeza según la configuración del experimento.
    No aplica congelamiento; eso lo hace freezing.apply_freeze_strategy().
    """
    print(f"[model] Cargando backbone: {exp_cfg.model_name}")
    backbone = AutoModel.from_pretrained(exp_cfg.model_name)
    hidden_size = backbone.config.hidden_size
    print(f"[model] hidden_size del backbone: {hidden_size}")

    num_labels = dataset_cfg.num_labels

    if exp_cfg.classifier_type == "linear":
        head = LinearHead(hidden_size, num_labels, exp_cfg.dropout)
        print(f"[model] Cabeza: LinearHead ({hidden_size} → {num_labels})")

    elif exp_cfg.classifier_type == "bottleneck":
        head = BottleneckHead(hidden_size, exp_cfg.hidden_dims, num_labels, exp_cfg.dropout)
        dims_str = " → ".join(
            [str(hidden_size)] + [str(d) for d in exp_cfg.hidden_dims] + [str(num_labels)]
        )
        print(f"[model] Cabeza: BottleneckHead ({dims_str})")

    else:
        raise ValueError(f"classifier_type desconocido: {exp_cfg.classifier_type}")

    model = ClassificationModel(backbone, head, num_labels)
    return model
