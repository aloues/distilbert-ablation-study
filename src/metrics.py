"""
src/metrics.py
Calcula todas las métricas de desempeño a partir de y_true y y_pred.
No depende de PyTorch ni de Hugging Face; usa solo scikit-learn.
"""

from __future__ import annotations

from typing import Dict, List

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def compute_classification_metrics(
    y_true: List[int],
    y_pred: List[int],
    loss_value: float,
) -> Dict[str, float]:
    """
    Calcula el conjunto completo de métricas de clasificación.

    Args:
        y_true:     Labels reales (lista de enteros).
        y_pred:     Predicciones del modelo (lista de enteros).
        loss_value: Eval loss promedio sobre el split (float).

    Returns:
        Diccionario con todas las métricas estándar del proyecto.

    Por qué macro y weighted:
        - macro: trata todas las clases por igual (importante en AG News 4 clases,
          Yelp 5 clases). Detecta si el modelo falla en clases minoritarias.
        - weighted: pondera por frecuencia de clase (mejor resumen cuando hay desbalance).
    """
    # zero_division=0 evita warnings cuando una clase no aparece en y_pred
    return {
        "accuracy":            float(accuracy_score(y_true, y_pred)),
        "precision_macro":     float(precision_score(y_true, y_pred, average="macro",    zero_division=0)),
        "precision_weighted":  float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_macro":        float(recall_score(y_true, y_pred, average="macro",       zero_division=0)),
        "recall_weighted":     float(recall_score(y_true, y_pred, average="weighted",    zero_division=0)),
        "f1_macro":            float(f1_score(y_true, y_pred, average="macro",           zero_division=0)),
        "f1_weighted":         float(f1_score(y_true, y_pred, average="weighted",        zero_division=0)),
        "eval_loss":           float(loss_value),
    }
