# Feature Scaling

This package experiments with scaling strategies for tabular market data. Each case script evaluates how different transformers influence downstream models on both public benchmarks and our in-house datasets.

Inside the package:
- Baseline demos on open datasets to validate the tooling before applying it to finance data.
- Comparative studies of standardisation, robust scaling, power transforms, and polynomial feature expansions.
- Shared helpers that keep preprocessing pipelines and metadata consistent across experiments.
- Result folders in `outputs/` storing transformed datasets, diagnostic plots, and performance summaries.

Start here when you want to choose an appropriate scaling pipeline or when you need reference implementations for integrating scaling into other research packages.
