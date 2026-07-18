# Changelog

## 0.3.0 (2026-07-18)

- Evaluation pipeline moved into the package (`flux.evaluation`) with a
  `flux-eval` CLI — notebooks are now demos only, nothing requires them
- MovieLens 1M support (`load_ml1m`, sparse-id re-indexing, named genres)
- FastAPI serving layer (`flux.serving.create_app`) with `[api]` extra and
  a runnable example (`examples/serve_api.py`)
- MkDocs documentation site (`docs/`, mkdocs-material) with auto-deploy to
  GitHub Pages on pushes to main
- Tag-triggered PyPI release workflow using trusted publishing (OIDC) —
  see docs/releasing.md for the one-time PyPI setup

## 0.2.0 (2026-07-18)

- MovieLens 100K loader (`flux.datasets`): cached download, offline-testable
  parser, leave-latest-out implicit-feedback split, genre labels, and a
  head/tail popularity split for fairness experiments
- Real-dataset evaluation notebook (`notebooks/movielens_evaluation.ipynb`,
  executed): hyperbolic vs Euclidean ranking on ML-100K, tail-item exposure
  before/after fair re-ranking
- PyPI packaging via `pyproject.toml` (`pip install flux-ranking`), with
  `torch`, `eval`, and `dev` extras
- GitHub Actions CI: unit tests (core + torch) and package build check
- Fixed `IPSDebiaser.estimate_ctr` to use the IPS mean estimator (0.1.x
  releases used self-normalized weighting, which underestimates CTR under
  position bias)

## 0.1.0 (2026-07-18)

- Initial release: NumPy core (hyperbolic embeddings, quantum-inspired
  attention, IPS debiasing, fair exposure re-ranking, `FluxEngine`),
  PyTorch training backend, synthetic-data evaluation notebook
