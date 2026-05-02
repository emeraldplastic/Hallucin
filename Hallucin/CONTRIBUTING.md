# Contributing to Hallucin Studio

Thanks for your interest in improving Hallucin Studio! Here's how to get started.

## Development Setup

```bash
git clone https://github.com/your-username/Hallucin.git
cd Hallucin
python -m venv venv
venv\Scripts\activate   # Windows
pip install -e ".[full]"
pip install pytest
```

## Running Tests

```bash
pytest -q
```

All tests must pass before opening a pull request.

## Code Style

- Follow PEP 8 conventions.
- Keep functions focused and well-documented.
- Add tests for new features or bug fixes.

## Pull Requests

1. Fork the repo and create a feature branch from `main`.
2. Make your changes with clear, descriptive commit messages.
3. Run the full test suite and confirm everything passes.
4. Open a PR with a concise description of what changed and why.

## Reporting Issues

Open a GitHub Issue with:
- A clear title and description
- Steps to reproduce (if applicable)
- Expected vs. actual behavior
- Environment details (Python version, OS)

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
