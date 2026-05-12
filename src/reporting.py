"""
src/reporting.py
Serializa resultados en JSON/CSV y genera curvas de loss como PNG.
Toda la lógica de escritura a disco está aquí.
"""

from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")  # backend sin GUI para servidores/Colab
import matplotlib.pyplot as plt


# ─── JSON de resultado final ──────────────────────────────────────────────────

def build_result_dict(
    dataset_cfg,
    exp_cfg,
    performance_metrics: Dict[str, float],
    efficiency_metrics: Dict[str, Any],
    training_time: float,
    run_name: str,
    loss_curves_path: str,
    model_path: str,
) -> Dict:
    """Construye el diccionario de resultado con el esquema estándar del proyecto."""
    effective_batch = dataset_cfg.batch_size * exp_cfg.gradient_accumulation_steps

    return {
        # Identificación
        "dataset": dataset_cfg.output_dataset_name,
        "experiment": exp_cfg.experiment_name,
        "base_model_type": exp_cfg.base_model_type,
        "base_model": exp_cfg.model_name,
        "classifier_type": exp_cfg.classifier_type,
        "freeze_strategy": exp_cfg.freeze_strategy,
        "trainable_layers": exp_cfg.trainable_layers,
        "hidden_dims": exp_cfg.hidden_dims,

        # Métricas de desempeño
        "accuracy":            performance_metrics.get("accuracy", 0.0),
        "precision_macro":     performance_metrics.get("precision_macro", 0.0),
        "precision_weighted":  performance_metrics.get("precision_weighted", 0.0),
        "recall_macro":        performance_metrics.get("recall_macro", 0.0),
        "recall_weighted":     performance_metrics.get("recall_weighted", 0.0),
        "f1_macro":            performance_metrics.get("f1_macro", 0.0),
        "f1_weighted":         performance_metrics.get("f1_weighted", 0.0),
        "eval_loss":           performance_metrics.get("eval_loss", 0.0),

        # Métricas de eficiencia
        "total_parameters":      efficiency_metrics.get("total_parameters", 0),
        "trainable_parameters":  efficiency_metrics.get("trainable_parameters", 0),
        "latency_ms_per_sample": efficiency_metrics.get("latency_ms_per_sample", 0.0),
        "gpu_memory_mb":         efficiency_metrics.get("gpu_memory_mb", 0.0),
        "training_time_seconds": round(training_time, 2),

        # Hiperparámetros del experimento
        "seed":                         exp_cfg.seed,
        "max_length":                   dataset_cfg.max_length,
        "batch_size":                   dataset_cfg.batch_size,
        "effective_batch_size":         effective_batch,
        "gradient_accumulation_steps":  exp_cfg.gradient_accumulation_steps,
        "learning_rate":                exp_cfg.learning_rate,
        "weight_decay":                 exp_cfg.weight_decay,
        "num_epochs":                   exp_cfg.num_epochs,
        "dropout":                      exp_cfg.dropout,
        "fp16":                         exp_cfg.fp16,

        # Rutas
        "timestamp":        datetime.now().isoformat(timespec="seconds"),
        "run_name":         run_name,
        "loss_curves_path": str(loss_curves_path),
        "model_path":       str(model_path),
    }


def save_result_json(result: Dict, path: Path) -> None:
    """Guarda el resultado en formato JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[reporting] Resultado guardado: {path}")


# ─── Loss curves JSON ─────────────────────────────────────────────────────────

def save_loss_curves_json(history: Dict, path: Path) -> None:
    """Guarda el historial de curvas de loss en JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[reporting] Curvas de loss guardadas: {path}")


# ─── Gráfico de curvas de loss ────────────────────────────────────────────────

def save_loss_curves_plot(history: Dict, path: Path, title: str = "") -> None:
    """
    Genera y guarda el gráfico de training loss vs validation loss como PNG.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    train_steps = [entry["step"] for entry in history.get("train_loss_by_step", [])]
    train_losses = [entry["loss"] for entry in history.get("train_loss_by_step", [])]

    val_epochs = [entry["epoch"] for entry in history.get("val_loss_by_epoch", [])]
    val_losses = [entry["loss"] for entry in history.get("val_loss_by_epoch", [])]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title or "Loss Curves", fontsize=13, fontweight="bold")

    # Training loss por step
    ax1 = axes[0]
    if train_steps:
        ax1.plot(train_steps, train_losses, color="#2563EB", linewidth=1.5, label="Train Loss")
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training Loss (por step)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # Validation loss por época
    ax2 = axes[1]
    if val_epochs:
        ax2.plot(val_epochs, val_losses, color="#DC2626", linewidth=2,
                 marker="o", markersize=6, label="Val Loss")
        ax2.set_xlabel("Época")
        ax2.set_ylabel("Loss")
        ax2.set_title("Validation Loss (por época)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(val_epochs)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[reporting] Gráfico de loss guardado: {path}")


# ─── CSV acumulativo ──────────────────────────────────────────────────────────

RESULTS_CSV_PATH = Path("outputs/results/all_results.csv")

# Campos que se escriben en el CSV (subconjunto del JSON para comparación rápida)
CSV_FIELDS = [
    "dataset", "experiment", "base_model_type", "classifier_type", "freeze_strategy",
    "accuracy", "f1_macro", "f1_weighted", "eval_loss",
    "total_parameters", "trainable_parameters",
    "latency_ms_per_sample", "gpu_memory_mb", "training_time_seconds",
    "learning_rate", "batch_size", "effective_batch_size",
    "num_epochs", "seed", "timestamp",
]


def append_to_results_csv(result: Dict, csv_path: Path = RESULTS_CSV_PATH) -> None:
    """
    Añade el resultado al CSV acumulativo de todos los experimentos.
    Crea el archivo con encabezados si no existe.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not csv_path.exists()
    row = {k: result.get(k, "") for k in CSV_FIELDS}

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"[reporting] Resultado añadido a CSV acumulativo: {csv_path}")


# ─── Guardado de configuración ────────────────────────────────────────────────

def save_config_copy(
    dataset_yaml_path: str,
    experiment_yaml_path: str,
    dest_dir: Path,
    run_name: str,
) -> None:
    """Copia los YAMLs usados al directorio de resultados para reproducibilidad."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(dataset_yaml_path, dest_dir / f"{run_name}_dataset_config.yaml")
    shutil.copy2(experiment_yaml_path, dest_dir / f"{run_name}_experiment_config.yaml")
    print(f"[reporting] Configs copiadas a: {dest_dir}")


# ─── Resumen en consola ───────────────────────────────────────────────────────

def print_summary(result: Dict) -> None:
    """Imprime un resumen visual del experimento en consola."""
    sep = "═" * 60
    print(f"\n{sep}")
    print(f"  RESUMEN: {result['dataset'].upper()} × {result['experiment'].upper()}")
    print(sep)
    print(f"  Accuracy:          {result['accuracy']:.4f}")
    print(f"  F1 Macro:          {result['f1_macro']:.4f}")
    print(f"  F1 Weighted:       {result['f1_weighted']:.4f}")
    print(f"  Eval Loss:         {result['eval_loss']:.4f}")
    print(f"  ─────────────────────────────────────────────")
    print(f"  Params totales:    {result['total_parameters']:,}")
    print(f"  Params entren.:    {result['trainable_parameters']:,}")
    print(f"  Latencia/muestra:  {result['latency_ms_per_sample']:.2f} ms")
    print(f"  GPU Memory:        {result['gpu_memory_mb']:.1f} MB")
    print(f"  Tiempo total:      {result['training_time_seconds']:.1f}s")
    print(sep + "\n")
