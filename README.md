# ai-color-workflow

Exploring AI-assisted color grading workflows with Nuke CopyCat and PyTorch.

## Motivation

Dailies color is the fast first-pass grading applied to raw camera footage so production can review it the next day. The per-shot balancing work is repetitive and time-pressured — the kind of pattern-based task ML can assist with.

This repo explores two AI approaches to that problem:

1. **Nuke CopyCat** - Foundry's built-in ML tool, trained on pairs of ungraded and graded frames. Stays inside the Nuke pipeline and respects OCIO.
2. **Custom PyTorch** - a small neural network built from scratch in Python. More tailored can learn per-shot CDL values or match to a reference look. Portable across environments.

## Structure

- `copycat/` — Nuke project files, node graph screenshots, CopyCat training results
- `pytorch/` — PyTorch model code (training, inference, architecture)
- `notebooks/` — Jupyter notebooks for exploration
- `data/` — sample frames (not tracked in git — see `data/README.md` for sourcing)
- `scripts/` — helper scripts (e.g., generating synthetic training pairs)
- `tests/` — pytest suite

## Setup

```bash