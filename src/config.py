"""
src/config.py
Carga y valida los YAMLs de dataset y experimento.
Expone DatasetConfig y ExperimentConfig como dataclasses tipados.
Toda la lógica del pipeline accede a la configuración solo a través de estos objetos.
"""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# ─── Dataset Config ──────────────────────────────────────────────────────────

@dataclass
class DatasetConfig:
    dataset_name: str
    text_column: str
    label_column: str
    num_labels: int
    train_split: str
    max_length: int
    batch_size: int
    output_dataset_name: str
    dataset_subset: Optional[str] = None
    validation_split: Optional[str] = None
    test_split: Optional[str] = None

    def __post_init__(self):
        self._validate()

    def _validate(self):
        assert self.dataset_name, "dataset_name no puede estar vacío"
        assert self.text_column, "text_column no puede estar vacío"
        assert self.label_column, "label_column no puede estar vacío"
        assert self.num_labels >= 2, f"num_labels debe ser >= 2, got {self.num_labels}"
        assert self.max_length > 0, f"max_length debe ser > 0, got {self.max_length}"
        assert self.batch_size > 0, f"batch_size debe ser > 0, got {self.batch_size}"


# ─── Experiment Config ────────────────────────────────────────────────────────

VALID_FREEZE_STRATEGIES = {"frozen_all", "partial", "unfrozen_all"}
VALID_CLASSIFIER_TYPES = {"linear", "bottleneck"}
VALID_BASE_MODEL_TYPES = {"distilbert", "bert"}
VALID_EVAL_STRATEGIES = {"epoch", "steps"}


@dataclass
class ExperimentConfig:
    experiment_name: str
    base_model_type: str
    model_name: str
    freeze_strategy: str
    classifier_type: str
    dropout: float
    learning_rate: float
    weight_decay: float
    num_epochs: int
    seed: int
    fp16: bool
    gradient_accumulation_steps: int
    evaluation_strategy: str
    save_strategy: str
    logging_steps: int
    hidden_dims: List[int] = field(default_factory=list)
    trainable_layers: Optional[List[int]] = None
    limit_train_samples: Optional[int] = None
    limit_eval_samples: Optional[int] = None

    def __post_init__(self):
        self._validate()

    def _validate(self):
        assert self.experiment_name, "experiment_name no puede estar vacío"

        assert self.base_model_type in VALID_BASE_MODEL_TYPES, (
            f"base_model_type inválido: '{self.base_model_type}'. "
            f"Válidos: {VALID_BASE_MODEL_TYPES}"
        )
        assert self.freeze_strategy in VALID_FREEZE_STRATEGIES, (
            f"freeze_strategy inválida: '{self.freeze_strategy}'. "
            f"Válidas: {VALID_FREEZE_STRATEGIES}"
        )
        assert self.classifier_type in VALID_CLASSIFIER_TYPES, (
            f"classifier_type inválido: '{self.classifier_type}'. "
            f"Válidos: {VALID_CLASSIFIER_TYPES}"
        )
        assert self.evaluation_strategy in VALID_EVAL_STRATEGIES, (
            f"evaluation_strategy inválida: '{self.evaluation_strategy}'"
        )

        if self.freeze_strategy == "partial":
            assert self.trainable_layers is not None and len(self.trainable_layers) > 0, (
                "trainable_layers debe estar definido cuando freeze_strategy = 'partial'"
            )

        if self.classifier_type == "bottleneck":
            assert len(self.hidden_dims) > 0, (
                "hidden_dims debe tener al menos una dimensión para classifier_type = 'bottleneck'"
            )

        assert 0.0 <= self.dropout < 1.0, f"dropout debe estar en [0, 1), got {self.dropout}"
        assert self.learning_rate > 0, f"learning_rate debe ser > 0"
        assert self.num_epochs > 0, f"num_epochs debe ser > 0"
        assert self.gradient_accumulation_steps >= 1

    @property
    def effective_batch_size(self) -> int:
        """Batch efectivo considerando gradient accumulation."""
        # Se calcula en contexto con dataset batch_size
        return self.gradient_accumulation_steps  # multiplicador; dataset_config provee batch_size


# ─── Loaders ─────────────────────────────────────────────────────────────────

def load_dataset_config(path: str | Path) -> DatasetConfig:
    """Carga y valida el YAML de dataset."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset config no encontrado: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    # Normalizar null → None
    for key in ("dataset_subset", "validation_split", "test_split",
                "limit_train_samples", "limit_eval_samples"):
        if key in raw and raw[key] is None:
            raw[key] = None

    return DatasetConfig(**raw)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Carga y valida el YAML de experimento."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Experiment config no encontrado: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    # Verificar que no haya campos ??? sin completar
    for key, val in raw.items():
        if val == "???":
            raise ValueError(
                f"El campo '{key}' en {path} contiene '???' — "
                "completa bert_base_best_config.yaml con los valores del ablation study."
            )

    # Normalizar
    raw.setdefault("hidden_dims", [])
    raw.setdefault("trainable_layers", None)
    raw.setdefault("limit_train_samples", None)
    raw.setdefault("limit_eval_samples", None)

    if raw.get("hidden_dims") is None:
        raw["hidden_dims"] = []

    return ExperimentConfig(**raw)
