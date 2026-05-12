# Framework de Clasificación de Texto: Ablation Study DistilBERT

Proyecto para comparar cuatro estrategias de transfer learning (C1–C4) usando DistilBERT en tres datasets de clasificación de texto, con comparación final contra BERT base.

---

## Objetivo del Proyecto

Evaluar cómo diferentes estrategias de congelamiento y arquitecturas de cabeza de clasificación afectan el rendimiento y la eficiencia en tareas de clasificación de texto. El proyecto está diseñado como un **ablation study** controlado: la única diferencia entre C1, C2, C3 y C4 es la arquitectura del clasificador y la estrategia de congelamiento.

---

## Las Cuatro Configuraciones

| Config | Backbone | Estrategia | Cabeza | Qué entrena |
|--------|----------|------------|--------|-------------|
| **C1** | DistilBERT | Congelado total | Linear 768→N | Solo cabeza lineal |
| **C2** | DistilBERT | Parcial (capas 5–6) | Linear 768→N | Cabeza + últimas 2 capas |
| **C3** | DistilBERT | Libre total | Linear 768→N | Todo el modelo |
| **C4** | DistilBERT | Congelado total | Bottleneck 768→128→64→N | Solo cabeza no lineal |

### ¿Qué hace C4?

C4 explora la hipótesis: *¿puede una cabeza de clasificación más expresiva compensar el backbone completamente congelado?*

Su cabeza bottleneck aplica compresión progresiva (768 → 128 → 64 → num_labels) con ReLU y Dropout entre capas. Si C4 supera a C1, la hipótesis se confirma. Si se acerca a C3 con muchos menos parámetros entrenables, el resultado tiene valor de eficiencia.

---

## Datasets Soportados

| Dataset | Clases | Split Train | Val | Test |
|---------|--------|-------------|-----|------|
| AG News | 4 | 120,000 | auto 90/10 | 7,600 |
| SST-2 (GLUE) | 2 | 67,349 | 872 | val (sin labels) |
| Yelp Reviews Full | 5 | 650,000 | auto 90/10 | 50,000 |

---

## Instalación

```bash
# Clonar el repositorio
git clone <url-del-repo>
cd project

# Instalar dependencias
pip install -r requirements.txt
```

Para Google Colab (T4):
```bash
!pip install -r requirements.txt
```

---

## Cómo Ejecutar

### Sintaxis general

```bash
python main.py --dataset configs/datasets/<dataset>.yaml \
               --experiment configs/experiments/<experiment>.yaml
```

### Ejecutar C1 (Linear Probing)

```bash
# AG News
python main.py --dataset configs/datasets/ag_news.yaml \
               --experiment configs/experiments/c1_linear_probing.yaml

# SST-2
python main.py --dataset configs/datasets/sst2.yaml \
               --experiment configs/experiments/c1_linear_probing.yaml

# Yelp Reviews
python main.py --dataset configs/datasets/yelp_reviews.yaml \
               --experiment configs/experiments/c1_linear_probing.yaml
```

### Ejecutar C2 (Partial Finetuning)

```bash
python main.py --dataset configs/datasets/ag_news.yaml \
               --experiment configs/experiments/c2_partial_finetuning.yaml
```

### Ejecutar C3 (Full Finetuning)

```bash
python main.py --dataset configs/datasets/ag_news.yaml \
               --experiment configs/experiments/c3_full_finetuning.yaml
```

### Ejecutar C4 (Bottleneck Head)

```bash
python main.py --dataset configs/datasets/ag_news.yaml \
               --experiment configs/experiments/c4_bottleneck.yaml

python main.py --dataset configs/datasets/sst2.yaml \
               --experiment configs/experiments/c4_bottleneck.yaml

python main.py --dataset configs/datasets/yelp_reviews.yaml \
               --experiment configs/experiments/c4_bottleneck.yaml
```

---

## Pruebas Rápidas (Colab / Pocas Muestras)

Para probar que el pipeline funciona antes de entrenar con el dataset completo, modifica temporalmente los YAMLs de experimento:

```yaml
# En cualquier configs/experiments/*.yaml
limit_train_samples: 500
limit_eval_samples: 100
num_epochs: 1
```

O usa la versión de línea de comandos sobreescribiendo el YAML manualmente antes de ejecutar.

> **Tip Colab:** Para Yelp + C3 con T4, usa `batch_size: 8` y `gradient_accumulation_steps: 4` para mantener el batch efectivo en 32.

---

## Comparación Final (DistilBERT vs BERT Base)

Una vez que el ablation study (C1–C4) esté completo:

1. Identificar la mejor config con `compare_results.py`.
2. Editar `configs/experiments/bert_base_best_config.yaml` con los valores de la config ganadora (solo cambiar `base_model_type` y `model_name`).
3. Ejecutar:

```bash
# Ejemplo si C4 es la mejor config
python main.py --dataset configs/datasets/ag_news.yaml \
               --experiment configs/experiments/bert_base_best_config.yaml

python main.py --dataset configs/datasets/sst2.yaml \
               --experiment configs/experiments/bert_base_best_config.yaml

python main.py --dataset configs/datasets/yelp_reviews.yaml \
               --experiment configs/experiments/bert_base_best_config.yaml
```

---

## Generar Comparación Final

```bash
python compare_results.py
```

Esto genera:
- Tabla comparativa por dataset en consola.
- Identificación de la mejor configuración.
- `outputs/plots/bubble_chart.png`: parámetros entrenables vs accuracy.
- `outputs/plots/comparison_table.csv`: CSV listo para el reporte.

```bash
# Especificar directorio y métrica de ranking
python compare_results.py --results_dir outputs/results \
                           --output_dir outputs/plots \
                           --metric f1_macro
```

---

## Interpretación de Métricas

| Métrica | Descripción |
|---------|-------------|
| `accuracy` | % de muestras clasificadas correctamente |
| `f1_macro` | F1 promedio por clase (sin ponderar por frecuencia) |
| `f1_weighted` | F1 promedio ponderado por frecuencia de clase |
| `eval_loss` | Cross-entropy loss en el set de evaluación |
| `total_parameters` | Todos los parámetros del modelo |
| `trainable_parameters` | Solo los que se actualizan en el entrenamiento |
| `latency_ms_per_sample` | Tiempo de inferencia por muestra (ms) |
| `gpu_memory_mb` | Pico de memoria GPU durante entrenamiento |
| `training_time_seconds` | Tiempo total de entrenamiento |

**¿Cuándo usar f1_macro?** En datasets con múltiples clases (AG News, Yelp) donde quieres saber si el modelo falla en alguna clase específica.

**¿Cuándo usar f1_weighted?** Para un resumen global que considera la distribución de clases.

---

## Estructura de Salidas

```
outputs/
├── results/
│   ├── ag_news_c4_bottleneck_<timestamp>.json   ← métricas completas
│   ├── all_results.csv                           ← acumulativo de todos los runs
│   └── <run_name>_*_config.yaml                 ← copia de configs usadas
├── models/
│   └── ag_news_c4_bottleneck_<timestamp>/        ← backbone + head.pt
├── logs/
│   └── <run_name>_loss_curves.json               ← curvas de loss
└── plots/
    ├── <run_name>_loss_curves.png                ← gráfico train vs val loss
    └── bubble_chart.png                          ← gráfico comparativo
```

---

## Reglas para el Equipo

**Regla de oro: No modificar archivos dentro de `src/`.** Si hay un bug, reportarlo al arquitecto del pipeline para que lo corrija para todos.

### División de trabajo

| Integrante | Tarea |
|-----------|-------|
| **Aleksander** | Implementa el framework completo + ejecuta C4 en los 3 datasets |
| **Andrea** | Ejecuta C1 y C2 en los 3 datasets (solo `python main.py ...`) |
| **Ana** | Ejecuta C3 en los 3 datasets; monitorea OOM en Yelp |

### Archivos a subir al repositorio (todos)

```
configs/           ← YAMLs de datasets y experimentos (sin cambios)
src/               ← código fuente (sin cambios)
outputs/results/   ← los JSONs de tus experimentos
main.py
compare_results.py
requirements.txt
README.md
```

**No subir:** `outputs/models/` (son grandes; usar `.gitignore`).

---

## CUDA Out of Memory

Si aparece `torch.cuda.OutOfMemoryError`:

1. Reducir `batch_size` en el YAML del dataset (ej: `16` → `8`).
2. Aumentar `gradient_accumulation_steps` en el YAML del experimento para compensar (ej: `2` → `4`).
3. El **batch efectivo** = `batch_size × gradient_accumulation_steps` debe mantenerse igual.
4. Ejecutar `torch.cuda.empty_cache()` entre experimentos en Colab.

---

## Nota sobre Indexación de Capas (C2)

DistilBERT tiene 6 capas transformer (índices 0–5).

En el reporte se habla de "capas 5 y 6 entrenables" → en el código esto es `trainable_layers: [4, 5]` (indexación base 0).

Esta correspondencia está documentada en `configs/experiments/c2_partial_finetuning.yaml` y en `src/freezing.py`.
