# Contributing to AI-ML-Methodology

Thanks for your interest in contributing! This guide helps preserve code quality and reproducibility.

## Getting started

1. Fork the repository and create a feature branch.
2. Install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r REQUIREMENTS.txt
   ```

3. Run tests:

   ```bash
   pytest -q
   ```

## Coding best practices

- Keep functions focused and modular.
- Add or update tests for new behavior.
- Keep the README updated with usage info.

## Pull request process

1. Open a PR against `main`.
2. Include short description of changes and why they are needed.
3. Ensure CI passes.
