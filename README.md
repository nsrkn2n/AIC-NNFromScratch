# AIC-NNFromScratch

This repository is for **educational purposes** as part of an **AIC workshop**. It demonstrates how to build a very simple and basic artificial neural network **from scratch**, without relying on any Python libraries, in order to avoid any abstractions and keep every step of the process transparent and easy to understand.

The only dependency used is **pydantic**, and solely because Python does not natively have its own type system. It is used to define typed data models for the network's components (neurons, layers, and the network itself) and nothing more — no machine learning, no tensor operations, and no training frameworks are involved.

## What's inside

- `main.py` — A minimal, self-contained neural network implementation:
  - `Neuron` — forward pass with ReLU activation and a manual backward (gradient) update.
  - `Layer` — a collection of neurons sharing the same input.
  - `Network` — a stack of layers with a forward pass and a single training step.
  - `mse` / `mse_grad` — a simple mean-squared-error loss and its gradient.
- `pyproject.toml` — project metadata; the only runtime dependency is `pydantic`.

## Running

```bash
uv run main.py
```

## Goal

The goal is to make the mechanics of a neural network visible: weights, biases, forward propagation, ReLU, loss, and backpropagation are all implemented by hand so learners can see exactly how each piece contributes, free of the magic that larger frameworks hide.