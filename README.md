# Neuron from Scratch (Victor Zhou – Reproduction)

## Live demo

[![Live Demo](https://img.shields.io/badge/Live-Demo-fuchsia.svg)](https://cmborjas.github.io/ResumeWebsite/projects/neuron-from-scratch)

A faithful, **from-scratch** implementation of the 2–2–1 neural network described by Victor Zhou — with **tests, types, CI, docs**, and a minimal notebook so recruiters can run and skim quickly.

[![CI](https://github.com/CMBorjas/neuron-from-scratch-vz/actions/workflows/ci.yml/badge.svg)](https://github.com/<user>/neuron-from-scratch-vz/actions/workflows/ci.yml)
[![Docs – MkDocs Material](https://img.shields.io/badge/docs-mkdocs--material-informational)](https://<user>.github.io/neuron-from-scratch-vz)

> **Attribution:** This repo **reproduces** the math and code structure from Victor Zhou’s tutorial
> “Machine Learning for Beginners: An Introduction to Neural Networks.” Read the original here.
> https://victorzhou.com/blog/intro-to-neural-networks/

---

## What's inside

- **Network:** 2 inputs → 2 hidden units (sigmoid) → 1 output (sigmoid).
- **Loss:** mean squared error (MSE).
- **Training:** stochastic gradient descent (per-sample) with explicit backprop partials, as in the tutorial.
- **Engineering polish:** pytest + coverage, Ruff, MyPy, pre-commit, MkDocs Material, VS Code settings, and (optional) Codespaces.

## Feature Extraction (Preprocessing)

In the tutorial, we try to predict a person's gender based on their weight and height. The original data is:

| Name | Weight (lb) | Height (in) | Gender |
| :--- | :--- | :--- | :--- |
| Alice | 133 | 65 | F (1) |
| Bob | 160 | 72 | M (0) |
| Charlie | 152 | 70 | M (0) |
| Diana | 120 | 60 | F (1) |

To make it easier for our neural network to train, we perform **feature extraction** by shifting our data (centering it).
We subtract an arbitrary value (135 pounds for weight, 66 inches for height) from each measurement to center the dataset around zero:

- **Weight shift:** `Weight - 135`
- **Height shift:** `Height - 66`

After extraction, the data looks like this:

| Name | Shifted Weight | Shifted Height |
| :--- | :--- | :--- |
| Alice | -2 | -1 |
| Bob | 25 | 6 |
| Charlie | 17 | 4 |
| Diana | -15 | -6 |

This preprocessing step ensures the features are roughly centered around zero. It prevents the network's initial sigmoid activations from saturating too early, allowing the gradients to flow properly and the model to learn efficiently!

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
```

## Future goals

---

[x] 1. Add visualization section for website integration
[x] 2. Verfiy that the neuron outline visulization ring outpuit node in the yellow output color, and the inner disk is the way that it is currently, and the inner disk will be shrunken and based on the predictive outcome. 
	* Male: yellow outer ring, blue inner disk; 
	* Female: blue outer ring, yellow inner disk.
[x] 3. Update code with more comments and cleaner variable names
[x] 4. Update ReadMe with more detailed information
[x] 5. Add Feature Extraction section 
[ ] 6. 

---