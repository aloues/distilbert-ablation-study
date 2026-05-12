"""
compare_results.py
Lee todos los JSONs de outputs/results/, genera tabla comparativa
y el gráfico de burbujas para el reporte.

Uso:
  python compare_results.py
  python compare_results.py --results_dir outputs/results --output_dir outputs/plots
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ─── Carga de resultados ──────────────────────────────────────────────────────

def load_all_results(results_dir: Path) -> List[Dict]:
    """Lee todos los JSONs del directorio de resultados."""
    results = []
    json_files = sorted(results_dir.glob("*.json"))
    if not json_files:
        print(f"[compare] No se encontraron JSONs en {results_dir}")
        return results

    for fp in json_files:
        try:
            with open(fp) as f:
                data = json.load(f)
            # Verificar campos mínimos requeridos
            required = ["dataset", "experiment", "accuracy", "f1_macro",
                        "total_parameters", "trainable_parameters"]
            missing = [k for k in required if k not in data]
            if missing:
                print(f"[compare] ⚠ {fp.name}: campos faltantes {missing}, se omite.")
                continue
            results.append(data)
        except json.JSONDecodeError as e:
            print(f"[compare] ⚠ Error leyendo {fp.name}: {e}")

    print(f"[compare] {len(results)} resultados cargados.")
    return results


# ─── Tabla comparativa ────────────────────────────────────────────────────────

DISPLAY_FIELDS = [
    ("experiment",            "Experimento"),
    ("dataset",               "Dataset"),
    ("freeze_strategy",       "Freeze"),
    ("classifier_type",       "Cabeza"),
    ("accuracy",              "Accuracy"),
    ("f1_macro",              "F1 Macro"),
    ("f1_weighted",           "F1 Weighted"),
    ("eval_loss",             "Eval Loss"),
    ("total_parameters",      "Params Total"),
    ("trainable_parameters",  "Params Entren."),
    ("latency_ms_per_sample", "Latencia (ms)"),
    ("gpu_memory_mb",         "GPU Mem (MB)"),
    ("training_time_seconds", "Tiempo (s)"),
]


def print_comparison_table(results: List[Dict]) -> None:
    """Imprime la tabla comparativa por dataset."""
    datasets = sorted({r["dataset"] for r in results})

    for ds in datasets:
        ds_results = [r for r in results if r["dataset"] == ds]
        ds_results.sort(key=lambda r: r.get("f1_macro", 0), reverse=True)

        print(f"\n{'═' * 100}")
        print(f"  DATASET: {ds.upper()}")
        print(f"{'═' * 100}")

        header = f"{'Experimento':<25} {'Freeze':<14} {'Cabeza':<12} "
        header += f"{'Acc':>7} {'F1 M':>7} {'F1 W':>7} {'ELoss':>7} "
        header += f"{'Params':>12} {'Entren.':>12} {'Lat(ms)':>9} {'GPU(MB)':>9}"
        print(header)
        print("─" * 100)

        for r in ds_results:
            line = (
                f"{r.get('experiment','?'):<25} "
                f"{r.get('freeze_strategy','?'):<14} "
                f"{r.get('classifier_type','?'):<12} "
                f"{r.get('accuracy', 0):.4f} "
                f"{r.get('f1_macro', 0):.4f} "
                f"{r.get('f1_weighted', 0):.4f} "
                f"{r.get('eval_loss', 0):.4f} "
                f"{r.get('total_parameters', 0):>12,} "
                f"{r.get('trainable_parameters', 0):>12,} "
                f"{r.get('latency_ms_per_sample', 0):>9.2f} "
                f"{r.get('gpu_memory_mb', 0):>9.1f}"
            )
            print(line)


def find_best_config(results: List[Dict], metric: str = "f1_macro") -> None:
    """Identifica la mejor configuración global por dataset."""
    datasets = sorted({r["dataset"] for r in results})
    print(f"\n{'═' * 60}")
    print(f"  MEJOR CONFIGURACIÓN POR DATASET (métrica: {metric})")
    print(f"{'═' * 60}")
    for ds in datasets:
        ds_results = [r for r in results if r["dataset"] == ds]
        if not ds_results:
            continue
        best = max(ds_results, key=lambda r: r.get(metric, 0))
        print(f"  {ds:<20} → {best['experiment']:<25} "
              f"({metric}={best.get(metric, 0):.4f})")

    # Mejor global (promedio de f1_macro en todos los datasets)
    experiments = sorted({r["experiment"] for r in results})
    print(f"\n  Promedio de {metric} por experimento:")
    avg_scores = {}
    for exp in experiments:
        exp_results = [r for r in results if r["experiment"] == exp]
        if exp_results:
            avg = np.mean([r.get(metric, 0) for r in exp_results])
            avg_scores[exp] = avg
            print(f"    {exp:<30} avg={avg:.4f}")

    if avg_scores:
        best_exp = max(avg_scores, key=avg_scores.get)
        print(f"\n  ★ MEJOR CONFIGURACIÓN GLOBAL: {best_exp} "
              f"(avg {metric}={avg_scores[best_exp]:.4f})")
        print(f"    → Usar esta config en bert_base_best_config.yaml")


# ─── Gráfico de burbujas ──────────────────────────────────────────────────────

EXPERIMENT_COLORS = {
    "c1_linear_probing":   "#2563EB",
    "c2_partial_finetuning": "#16A34A",
    "c3_full_finetuning":  "#DC2626",
    "c4_bottleneck":       "#9333EA",
    "bert_base_best_config": "#EA580C",
}


def plot_bubble_chart(results: List[Dict], output_path: Path) -> None:
    """
    Gráfico de burbujas:
      X = trainable_parameters
      Y = accuracy
      Tamaño = latency_ms_per_sample (escalado)
      Color = experimento
    """
    if not results:
        print("[compare] Sin datos para el gráfico de burbujas.")
        return

    datasets = sorted({r["dataset"] for r in results})
    n_datasets = len(datasets)

    fig, axes = plt.subplots(1, n_datasets, figsize=(7 * n_datasets, 6), squeeze=False)
    fig.suptitle("Parámetros entrenables vs Accuracy\n(tamaño = latencia ms/muestra)",
                 fontsize=14, fontweight="bold")

    for col, ds in enumerate(datasets):
        ax = axes[0][col]
        ds_results = [r for r in results if r["dataset"] == ds]

        for r in ds_results:
            x = r.get("trainable_parameters", 0)
            y = r.get("accuracy", 0)
            lat = r.get("latency_ms_per_sample", 1)
            exp = r.get("experiment", "?")
            color = EXPERIMENT_COLORS.get(exp, "#6B7280")
            size = max(lat * 200, 50)  # escalar tamaño

            ax.scatter(x, y, s=size, color=color, alpha=0.75, edgecolors="white", linewidth=1.5)
            ax.annotate(
                exp.replace("_", "\n"),
                (x, y),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=7,
                color=color,
            )

        ax.set_xlabel("Parámetros entrenables", fontsize=10)
        ax.set_ylabel("Accuracy", fontsize=10)
        ax.set_title(ds, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Formatear eje X con notación compacta
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{v/1e6:.1f}M" if v >= 1e6 else f"{int(v):,}")
        )

    # Leyenda
    legend_patches = [
        mpatches.Patch(color=color, label=exp)
        for exp, color in EXPERIMENT_COLORS.items()
        if any(r.get("experiment") == exp for r in results)
    ]
    if legend_patches:
        fig.legend(
            handles=legend_patches,
            loc="lower center",
            ncol=max(1, len(legend_patches)),
            bbox_to_anchor=(0.5, -0.06),
            fontsize=9
    )

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[compare] Gráfico de burbujas guardado: {output_path}")


# ─── CSV final de comparación ─────────────────────────────────────────────────

def save_comparison_csv(results: List[Dict], output_path: Path) -> None:
    """Guarda todos los resultados en un CSV ordenado para el reporte."""
    if not results:
        return
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Ordenar por dataset y f1_macro descendente
    sorted_results = sorted(results, key=lambda r: (r.get("dataset", ""), -r.get("f1_macro", 0)))

    fields = list(sorted_results[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted_results)

    print(f"[compare] CSV de comparación guardado: {output_path}")


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Comparar resultados de experimentos.")
    parser.add_argument("--results_dir", default="outputs/results",
                        help="Directorio con los JSONs de resultados")
    parser.add_argument("--output_dir", default="outputs/plots",
                        help="Directorio para guardar gráficos y CSV de comparación")
    parser.add_argument("--metric", default="f1_macro",
                        choices=["f1_macro", "accuracy", "f1_weighted"],
                        help="Métrica principal para ranking (default: f1_macro)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_all_results(results_dir)
    if not results:
        print("[compare] Sin resultados para comparar. "
              "Asegúrate de haber ejecutado al menos un experimento.")
        return

    print_comparison_table(results)
    find_best_config(results, metric=args.metric)
    plot_bubble_chart(results, output_dir / "bubble_chart.png")
    save_comparison_csv(results, output_dir / "comparison_table.csv")

    print("\n[compare] ✓ Comparación completada.")


if __name__ == "__main__":
    main()
