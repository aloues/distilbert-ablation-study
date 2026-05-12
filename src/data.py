"""
src/data.py
Carga datasets desde Hugging Face, tokeniza y devuelve DataLoaders.
No contiene lógica de modelo ni de métricas.
"""

from __future__ import annotations

from typing import Dict, Tuple, Optional

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from src.config import DatasetConfig, ExperimentConfig


# ─── Resolución de splits ─────────────────────────────────────────────────────

def _resolve_splits(
    raw_dataset: DatasetDict,
    dataset_cfg: DatasetConfig,
    seed: int,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Devuelve (train, val, test) resolviendo los casos especiales:
    - validation_split null → crear val desde train con split 90/10
    - test_split null (SST-2) → usar val como test
    """
    # Train
    train_ds = raw_dataset[dataset_cfg.train_split]

    # Validation
    if dataset_cfg.validation_split is not None:
        val_ds = raw_dataset[dataset_cfg.validation_split]
    else:
        # Crear val desde train con split 90/10
        print(
            f"[data] '{dataset_cfg.output_dataset_name}' no tiene validation nativo. "
            "Creando 90/10 split desde train..."
        )
        split = train_ds.train_test_split(test_size=0.1, seed=seed)
        train_ds = split["train"]
        val_ds = split["test"]

    # Test
    if dataset_cfg.test_split is not None:
        test_ds = raw_dataset[dataset_cfg.test_split]
    else:
        # SST-2: test no tiene labels, usar val
        print(
            f"[data] '{dataset_cfg.output_dataset_name}' no tiene test con labels. "
            "Usando validation como test."
        )
        test_ds = val_ds

    return train_ds, val_ds, test_ds


# ─── Validación de columnas ───────────────────────────────────────────────────

def _validate_columns(dataset: Dataset, dataset_cfg: DatasetConfig) -> None:
    cols = dataset.column_names
    if dataset_cfg.text_column not in cols:
        raise ValueError(
            f"text_column '{dataset_cfg.text_column}' no encontrada. "
            f"Columnas disponibles: {cols}"
        )
    if dataset_cfg.label_column not in cols:
        raise ValueError(
            f"label_column '{dataset_cfg.label_column}' no encontrada. "
            f"Columnas disponibles: {cols}"
        )

    # Validar rango de labels
    sample_labels = dataset[dataset_cfg.label_column][:100]
    min_label = min(sample_labels)
    max_label = max(sample_labels)
    if min_label < 0 or max_label >= dataset_cfg.num_labels:
        raise ValueError(
            f"Labels fuera de rango [0, {dataset_cfg.num_labels - 1}]: "
            f"encontrados min={min_label}, max={max_label}"
        )


# ─── Tokenización ─────────────────────────────────────────────────────────────

def _tokenize_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    dataset_cfg: DatasetConfig,
) -> Dataset:
    """Tokeniza el dataset y devuelve un Dataset con input_ids, attention_mask, labels."""

    text_col = dataset_cfg.text_column
    label_col = dataset_cfg.label_column

    def tokenize_fn(examples):
        encoding = tokenizer(
            examples[text_col],
            padding="max_length",
            truncation=True,
            max_length=dataset_cfg.max_length,
        )
        encoding["labels"] = examples[label_col]
        return encoding

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
    )
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized


# ─── Función principal ────────────────────────────────────────────────────────

def load_data(
    dataset_cfg: DatasetConfig,
    exp_cfg: ExperimentConfig,
) -> Dict[str, DataLoader]:
    """
    Pipeline completo de datos:
    1. Carga el dataset desde HF.
    2. Aplica límites de muestras.
    3. Resuelve splits.
    4. Valida columnas.
    5. Tokeniza.
    6. Devuelve dict con 'train', 'val', 'test' DataLoaders.
    """
    print(f"\n[data] Cargando dataset: {dataset_cfg.dataset_name} "
          f"(subset={dataset_cfg.dataset_subset})")

    raw = load_dataset(dataset_cfg.dataset_name, dataset_cfg.dataset_subset)

    # Aplicar límites antes de hacer splits
    if exp_cfg.limit_train_samples is not None:
        n = min(exp_cfg.limit_train_samples, len(raw[dataset_cfg.train_split]))
        raw[dataset_cfg.train_split] = raw[dataset_cfg.train_split].select(range(n))
        print(f"[data] limit_train_samples={n}")

    train_ds, val_ds, test_ds = _resolve_splits(raw, dataset_cfg, exp_cfg.seed)

    if exp_cfg.limit_eval_samples is not None:
        n_val = min(exp_cfg.limit_eval_samples, len(val_ds))
        n_test = min(exp_cfg.limit_eval_samples, len(test_ds))
        val_ds = val_ds.select(range(n_val))
        test_ds = test_ds.select(range(n_test))
        print(f"[data] limit_eval_samples={exp_cfg.limit_eval_samples}")

    _validate_columns(train_ds, dataset_cfg)

    print(f"[data] Splits — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    # Tokenizer
    print(f"[data] Cargando tokenizer: {exp_cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(exp_cfg.model_name)

    train_tok = _tokenize_dataset(train_ds, tokenizer, dataset_cfg)
    val_tok   = _tokenize_dataset(val_ds, tokenizer, dataset_cfg)
    test_tok  = _tokenize_dataset(test_ds, tokenizer, dataset_cfg)

    bs = dataset_cfg.batch_size

    loaders = {
        "train": DataLoader(train_tok, batch_size=bs, shuffle=True,  num_workers=2, pin_memory=True),
        "val":   DataLoader(val_tok,   batch_size=bs, shuffle=False, num_workers=2, pin_memory=True),
        "test":  DataLoader(test_tok,  batch_size=bs, shuffle=False, num_workers=2, pin_memory=True),
    }

    print(f"[data] DataLoaders creados (batch_size={bs})")
    return loaders
