"""
src/freezing.py
Aplica la estrategia de congelamiento sobre el backbone del modelo.
Módulo separado porque el congelamiento es ortogonal a la arquitectura.

Nota de indexación:
  DistilBERT tiene 6 transformer layers indexadas 0–5.
  "Capas 5 y 6 entrenables" en el reporte = índices 4 y 5 en código.
  trainable_layers en el YAML usa indexación 0-based (ej: [4, 5]).
"""

from __future__ import annotations

from src.config import ExperimentConfig
from src.model import ClassificationModel


def apply_freeze_strategy(model: ClassificationModel, exp_cfg: ExperimentConfig) -> None:
    """
    Modifica requires_grad del backbone según la estrategia configurada.
    Siempre preserva los parámetros de la cabeza como entrenables.
    Modifica el modelo in-place.
    """
    strategy = exp_cfg.freeze_strategy

    if strategy == "frozen_all":
        _freeze_all_backbone(model)

    elif strategy == "partial":
        _freeze_partial_backbone(model, exp_cfg)

    elif strategy == "unfrozen_all":
        _unfreeze_all_backbone(model)

    else:
        raise ValueError(f"freeze_strategy desconocida: '{strategy}'")

    # Verificación de seguridad: la cabeza siempre debe ser entrenable
    assert any(p.requires_grad for p in model.head.parameters()), (
        "ERROR CRÍTICO: La cabeza de clasificación no tiene parámetros entrenables. "
        "Revisar freezing.py."
    )

    _print_trainable_summary(model, strategy)


def _freeze_all_backbone(model: ClassificationModel) -> None:
    """C1, C4: congela todos los parámetros del backbone."""
    for param in model.backbone.parameters():
        param.requires_grad = False


def _unfreeze_all_backbone(model: ClassificationModel) -> None:
    """C3: todos los parámetros del backbone son entrenables."""
    for param in model.backbone.parameters():
        param.requires_grad = True


def _freeze_partial_backbone(model: ClassificationModel, exp_cfg: ExperimentConfig) -> None:
    """
    C2: congela todas las capas del backbone primero,
    luego descongela las capas especificadas en trainable_layers.

    Para DistilBERT: las capas transformer están en
      model.backbone.transformer.layer[i]

    Para BERT: las capas están en
      model.backbone.encoder.layer[i]
    """
    # Congelar todo el backbone
    for param in model.backbone.parameters():
        param.requires_grad = False

    # Identificar el módulo de capas según el tipo de modelo
    backbone = model.backbone
    if hasattr(backbone, "transformer"):
        # DistilBERT
        layer_module = backbone.transformer.layer
        model_type_str = "DistilBERT (backbone.transformer.layer)"
    elif hasattr(backbone, "encoder"):
        # BERT
        layer_module = backbone.encoder.layer
        model_type_str = "BERT (backbone.encoder.layer)"
    else:
        raise RuntimeError(
            "No se pudo identificar el módulo de capas del backbone. "
            "Solo se soportan DistilBERT y BERT."
        )

    n_layers = len(layer_module)
    trainable_indices = exp_cfg.trainable_layers

    # Validar índices
    for idx in trainable_indices:
        if idx < 0 or idx >= n_layers:
            raise ValueError(
                f"trainable_layers contiene índice {idx} inválido para un backbone "
                f"con {n_layers} capas (índices válidos: 0–{n_layers - 1})."
            )

    # Descongelar las capas especificadas
    for idx in trainable_indices:
        for param in layer_module[idx].parameters():
            param.requires_grad = True

    print(
        f"[freezing] {model_type_str}: {n_layers} capas totales. "
        f"Capas entrenables (0-based): {trainable_indices}"
    )


def _print_trainable_summary(model: ClassificationModel, strategy: str) -> None:
    """Imprime un resumen de qué partes del modelo son entrenables."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100.0 * trainable / total if total > 0 else 0.0

    print(f"\n[freezing] Estrategia aplicada: '{strategy}'")
    print(f"  Parámetros totales:     {total:,}")
    print(f"  Parámetros entrenables: {trainable:,} ({pct:.2f}%)")

    # Desglose por módulo
    backbone_trainable = sum(
        p.numel() for p in model.backbone.parameters() if p.requires_grad
    )
    head_trainable = sum(
        p.numel() for p in model.head.parameters() if p.requires_grad
    )
    print(f"  → Backbone entrenable:  {backbone_trainable:,}")
    print(f"  → Cabeza entrenable:    {head_trainable:,}\n")
