"""
main.py
Orquesta el pipeline completo.
Conecta todos los módulos en el orden correcto.
No contiene lógica de negocio propia.

Uso:
  python main.py --dataset configs/datasets/ag_news.yaml \
                 --experiment configs/experiments/c4_bottleneck.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# ── Módulos del proyecto ──────────────────────────────────────────────────────
from src.config import load_dataset_config, load_experiment_config
from src.data import load_data
from src.efficiency import count_parameters, measure_gpu_memory, measure_latency
from src.freezing import apply_freeze_strategy
from src.metrics import compute_classification_metrics
from src.model import build_model
from src.reporting import (
    append_to_results_csv,
    build_result_dict,
    print_summary,
    save_config_copy,
    save_loss_curves_json,
    save_loss_curves_plot,
    save_result_json,
)
from src.trainer import evaluate, train_and_evaluate
from src.utils import empty_cuda_cache, get_device, make_output_dirs, make_run_name, set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline de clasificación de texto con DistilBERT/BERT."
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Ruta al YAML de configuración del dataset (ej: configs/datasets/ag_news.yaml)"
    )
    parser.add_argument(
        "--experiment", required=True,
        help="Ruta al YAML de configuración del experimento (ej: configs/experiments/c4_bottleneck.yaml)"
    )
    parser.add_argument(
        "--output_dir", default="outputs",
        help="Directorio base para guardar resultados (default: outputs/)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "═" * 60)
    print("  PIPELINE DE CLASIFICACIÓN DE TEXTO")
    print("═" * 60)

    # ── FASE DE CONFIGURACIÓN ────────────────────────────────────────────────
    print("\n[main] Cargando configuraciones...")
    dataset_cfg = load_dataset_config(args.dataset)
    exp_cfg = load_experiment_config(args.experiment)

    print(f"  Dataset:    {dataset_cfg.output_dataset_name}")
    print(f"  Experimento: {exp_cfg.experiment_name}")
    print(f"  Modelo:     {exp_cfg.model_name}")
    print(f"  Estrategia: {exp_cfg.freeze_strategy} | Cabeza: {exp_cfg.classifier_type}")

    run_name = make_run_name(dataset_cfg.output_dataset_name, exp_cfg.experiment_name)
    dirs = make_output_dirs(args.output_dir)

    # ── FASE DE REPRODUCIBILIDAD ─────────────────────────────────────────────
    set_seed(exp_cfg.seed)

    # Limpiar cache CUDA antes de empezar
    empty_cuda_cache()

    # ── DISPOSITIVO ───────────────────────────────────────────────────────────
    device = get_device()

    # ── FASE DE DATOS ─────────────────────────────────────────────────────────
    loaders = load_data(dataset_cfg, exp_cfg)

    # ── FASE DE MODELO ────────────────────────────────────────────────────────
    print(f"\n[main] Construyendo modelo...")
    model = build_model(exp_cfg, dataset_cfg)

    # Aplicar estrategia de congelamiento ANTES de crear el optimizer
    apply_freeze_strategy(model, exp_cfg)

    # ── FASE DE ENTRENAMIENTO ─────────────────────────────────────────────────
    torch.cuda.reset_peak_memory_stats()  # resetear pico para capturar solo el entrenamiento
    model, history, training_time = train_and_evaluate(
        model, loaders, exp_cfg, dataset_cfg, device
    )

    # ── FASE DE EVALUACIÓN ────────────────────────────────────────────────────
    use_fp16 = exp_cfg.fp16 and device.type == "cuda"
    print("\n[main] Evaluando en test set...")
    test_results = evaluate(model, loaders["test"], device, use_fp16=use_fp16)

    performance_metrics = compute_classification_metrics(
        y_true=test_results["y_true"],
        y_pred=test_results["y_pred"],
        loss_value=test_results["loss"],
    )

    print(f"[main] accuracy={performance_metrics['accuracy']:.4f} | "
          f"f1_macro={performance_metrics['f1_macro']:.4f}")

    # ── FASE DE EFICIENCIA ────────────────────────────────────────────────────
    print("\n[main] Midiendo métricas de eficiencia...")
    param_metrics = count_parameters(model)
    latency_metrics = measure_latency(model, loaders["test"], device)
    gpu_metrics = measure_gpu_memory()

    efficiency_metrics = {**param_metrics, **latency_metrics, **gpu_metrics}

    # ── FASE DE GUARDADO ──────────────────────────────────────────────────────

    # Paths
    loss_curves_path = dirs["logs"] / f"{run_name}_loss_curves.json"
    loss_plot_path   = dirs["plots"] / f"{run_name}_loss_curves.png"
    result_json_path = dirs["results"] / f"{run_name}.json"
    model_save_path  = dirs["models"] / run_name
    config_save_dir  = dirs["results"]

    # Guardar curvas de loss (JSON + PNG)
    save_loss_curves_json(history, loss_curves_path)
    save_loss_curves_plot(
        history, loss_plot_path,
        title=f"{dataset_cfg.output_dataset_name} | {exp_cfg.experiment_name}"
    )

    # Construir y guardar resultado final
    result = build_result_dict(
        dataset_cfg=dataset_cfg,
        exp_cfg=exp_cfg,
        performance_metrics=performance_metrics,
        efficiency_metrics=efficiency_metrics,
        training_time=training_time,
        run_name=run_name,
        loss_curves_path=loss_curves_path,
        model_path=model_save_path,
    )
    save_result_json(result, result_json_path)
    append_to_results_csv(result)

    # Guardar modelo
    model_save_path.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(str(model_save_path / "backbone"))
    torch.save(model.head.state_dict(), str(model_save_path / "head.pt"))
    print(f"[main] Modelo guardado: {model_save_path}")

    # Guardar copia de configs usadas
    save_config_copy(args.dataset, args.experiment, config_save_dir, run_name)

    # Resumen final
    print_summary(result)

    print(f"[main] ✓ Experimento completado. Run: {run_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
