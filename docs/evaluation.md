# Evaluation

All evaluation logic lives in the package (`flux.evaluation`), exposed via
the **`flux-eval`** CLI. The notebooks under `notebooks/` are executed demos
of the same pipeline — nothing requires running them.

## CLI

```bash
flux-eval --dataset ml-100k --epochs 40 --output results.json
flux-eval --dataset ml-1m --epochs 20 --fairness-weight 5.0
flux-eval --help
```

The pipeline: load the dataset (cached download) → train the hyperbolic
model and an equal-dimension Euclidean baseline (identical optimizer and
negative sampling) → leave-latest-out HR@k / NDCG@k → head/tail exposure
report before and after `FairReranker`. Results print to stdout and can be
written as JSON with `--output`.

## Programmatic use

```python
from flux.evaluation import EvalConfig, run_evaluation

results = run_evaluation(EvalConfig(dataset="ml-1m", epochs=20))
print(results["hyperbolic"], results["fairness"]["after"])
```

`ranking_metrics(model, data, k)` and `exposure_report(model, data, config)`
are importable on their own for custom models.

## Reference results

MovieLens 100K (dim 16, 40 epochs, leave-latest-out):

| Model | HR@10 | NDCG@10 |
|---|---|---|
| Hyperbolic | **0.077** | **0.038** |
| Euclidean | 0.067 | 0.031 |

Fairness (same trained scores, weight 5.0, 50/50 head-tail target):
tail exposure **5% → 42%**.

Numbers come from a simple uniform-negative triplet objective at small
dimension — they demonstrate the geometry comparison, not state-of-the-art
tuning.

On **MovieLens 1M** (6040 users, 3883 movies, ~570k train pairs) the same
recipe at 20 epochs is *under-trained for the hyperbolic model*: Euclidean
reaches HR@10 0.032 while hyperbolic sits at 0.009. Hyperbolic embeddings
start near the origin where distances are compressed, so at larger catalog
scale they need substantially more epochs (or a curvature/lr schedule) to
spread out — increase `--epochs` well beyond 20 before drawing conclusions
at this scale.

## Notebooks (demos)

- `notebooks/evaluation.ipynb` — synthetic hierarchical data, plus IPS vs
  naive CTR under simulated position bias
- `notebooks/movielens_evaluation.ipynb` — the ML-100K study above with plots

Both are committed executed, so the outputs are viewable on GitHub without
running anything.
