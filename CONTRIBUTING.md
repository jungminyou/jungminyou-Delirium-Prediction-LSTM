# Contributing Guide

## Scope

This repository accompanies an SCI manuscript. Contributions should preserve:

- patient-level split integrity,
- leakage-free preprocessing,
- reproducible random seeds,
- manuscript-consistent evaluation metrics.

## Development Setup

1. Create a Python environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Run stage scripts to validate local changes.

## Coding Rules

- Keep changes modular and stage-specific.
- Avoid hard-coded absolute paths.
- Do not commit private data, patient identifiers, or confidential artifacts.
- Add concise comments only where logic is non-trivial.

## Pull Request Checklist

- Explain scientific impact of the change.
- Confirm no data leakage introduced.
- Confirm random seeds and deterministic behavior where applicable.
- Attach before/after metric tables if model behavior changed.
