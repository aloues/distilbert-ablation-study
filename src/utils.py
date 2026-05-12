"""
src/utils.py
Funciones de utilidad compartidas: semilla aleatoria, directorios, timestamps.
Sin dependencias cruzadas entre módulos de src/.
"""

from __future__ import annotations

import os
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Union

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Fija la semilla en Python, NumPy, PyTorch y CUDA para reproducibilidad total."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Para ops deterministas (puede afectar rendimiento)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[utils] Semilla fijada: {seed}")


def get_device() -> torch.device:
    """Devuelve el dispositivo disponible (CUDA si hay GPU, sino CPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"[utils] Dispositivo: {torch.cuda.get_device_name(0)}")
    else:
        print("[utils] Dispositivo: CPU")
    return device


def make_run_name(dataset_name: str, experiment_name: str) -> str:
    """Crea un nombre único para el run: dataset_experiment_timestamp."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{dataset_name}_{experiment_name}_{ts}"


def make_output_dirs(base_output: Union[str, Path] = "outputs") -> dict:
    """Crea y devuelve todos los directorios de salida."""
    base = Path(base_output)
    dirs = {
        "results": base / "results",
        "models": base / "models",
        "logs": base / "logs",
        "plots": base / "plots",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def empty_cuda_cache() -> None:
    """Libera memoria CUDA no utilizada."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("[utils] CUDA cache liberado.")


def count_trainable_params_summary(model: torch.nn.Module) -> str:
    """Devuelve un string resumen de parámetros totales y entrenables."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100.0 * trainable / total if total > 0 else 0.0
    return (
        f"Parámetros totales:      {total:,}\n"
        f"Parámetros entrenables:  {trainable:,} ({pct:.1f}%)"
    )


def save_yaml_copy(source_path: Union[str, Path], dest_path: Union[str, Path]) -> None:
    """Copia un YAML a dest_path para preservar la configuración usada."""
    shutil.copy2(str(source_path), str(dest_path))
