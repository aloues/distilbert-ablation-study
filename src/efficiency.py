"""
src/efficiency.py
Mide métricas de eficiencia: parámetros, latencia, memoria GPU.
Independiente del entrenamiento; se ejecuta al final del pipeline.
"""

from __future__ import annotations

import time
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Cuenta parámetros totales y entrenables."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
    }


def measure_latency(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    n_warmup: int = 10,
    n_measure: int = 100,
) -> Dict[str, float]:
    """
    Mide la latencia de inferencia promedio por muestra.

    Ejecuta n_warmup iteraciones para estabilizar CUDA,
    luego n_measure iteraciones con timer sincronizado.

    Returns:
        {"latency_ms_per_sample": float}
    """
    model.eval()

    # Obtener un batch fijo de referencia
    batch = next(iter(dataloader))
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    batch_size = input_ids.shape[0]

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            model(input_ids=input_ids, attention_mask=attention_mask)

    # Medición con sincronización CUDA
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_measure):
            model(input_ids=input_ids, attention_mask=attention_mask)

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    ms_per_batch = (elapsed / n_measure) * 1000.0
    ms_per_sample = ms_per_batch / batch_size

    return {"latency_ms_per_sample": round(ms_per_sample, 4)}


def measure_gpu_memory() -> Dict[str, float]:
    """
    Devuelve el pico de memoria GPU usada durante el entrenamiento.
    Debe llamarse DESPUÉS del entrenamiento para capturar el pico real.

    Returns:
        {"gpu_memory_mb": float}  (0.0 si no hay GPU)
    """
    if torch.cuda.is_available():
        gpu_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        gpu_mb = 0.0
    return {"gpu_memory_mb": round(gpu_mb, 2)}
