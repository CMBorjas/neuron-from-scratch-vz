# Neuron from Scratch (Victor Zhou – Reproduction)

A faithful, **from-scratch** implementation of the 2–2–1 neural network described by Victor Zhou — with **tests, types, CI, docs**, and a minimal notebook so recruiters can run and skim quickly.

[![CI](https://github.com/<user>/neuron-from-scratch-vz/actions/workflows/ci.yml/badge.svg)](https://github.com/<user>/neuron-from-scratch-vz/actions/workflows/ci.yml)
[![Docs – MkDocs Material](https://img.shields.io/badge/docs-mkdocs--material-informational)](https://<user>.github.io/neuron-from-scratch-vz)

> **Attribution:** This repo **reproduces** the math and code structure from Victor Zhou’s tutorial
> “Machine Learning for Beginners: An Introduction to Neural Networks.” Read the original here.  
> https://victorzhou.com/blog/intro-to-neural-networks/  :contentReference[oaicite:2]{index=2}

---

## What’s inside

- **Network:** 2 inputs → 2 hidden units (sigmoid) → 1 output (sigmoid).
- **Loss:** mean squared error (MSE).
- **Training:** stochastic gradient descent (per-sample) with explicit backprop partials, as in the tutorial. :contentReference[oaicite:3]{index=3}
- **Engineering polish:** pytest + coverage, Ruff, MyPy, pre-commit, MkDocs Material, VS Code settings, and (optional) Codespaces.

## Quickstart (local)

```bash
python -m venv .venv
# Windows: .venv\Scripts\Activate ; macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"

# run quality checks
pre-commit install
pre-commit run --all-files

# run tests
pytest

# open the minimal demo notebook
jupyter notebook notebooks/01_reproduce_vz_post.ipynb
