# Contributing to Evret

Thank you for your interest in contributing to Evret! We welcome contributions from the community and are excited to have you here.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for everyone. Please be kind and courteous in all interactions.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Create a new branch for your feature or bug fix
4. Make your changes
5. Submit a pull request

## Development Setup

### Prerequisites

- Python 3.10 or higher
- Git
- Docker (for integration tests)

### Local Setup

We recommend using `uv` for dependency management:

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/evret.git
cd evret

# Create virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in editable mode with all dependencies
uv pip install -e ".[all,dev]"
```

### Alternative Setup with pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[all,dev]"
```

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:

- A clear, descriptive title
- Detailed steps to reproduce the issue
- Expected behavior vs actual behavior
- Your environment (OS, Python version, etc.)
- Any relevant code snippets or error messages

### Suggesting Enhancements

We welcome feature suggestions! Please create an issue with:

- A clear description of the feature
- The motivation and use case
- Any examples or mockups if applicable

### Code Contributions

We accept contributions for:

- Bug fixes
- New features
- Performance improvements
- Documentation improvements
- Test coverage improvements
- New metrics implementations
- New vector database adapters

## Pull Request Process

1. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following our coding standards

3. **Add tests** for any new functionality

4. **Run the test suite** to ensure everything passes:
   ```bash
   pytest
   ```

5. **Run integration tests** if you modified retriever adapters:
   ```bash
   EVRET_RUN_INTEGRATION=1 pytest -m integration
   ```

6. **Update documentation** if needed

7. **Commit your changes** with clear commit messages:
   ```bash
   git commit -m "Add feature: brief description"
   ```

8. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

9. **Create a Pull Request** on GitHub with:
   - A clear title and description
   - Reference to any related issues
   - Summary of changes
   - Screenshots or examples if applicable

### Pull Request Review

- Maintainers will review your PR and may request changes
- Address any feedback by pushing new commits to your branch
- Once approved, your PR will be merged

## Coding Standards

### Python Style

- Follow [PEP 8](https://pep8.org/) style guidelines
- Use type hints for all function signatures
- Maximum line length: 100 characters
- Use descriptive variable and function names

### Code Formatting

We use `black` for code formatting (recommended but not enforced):

```bash
pip install black
black src/ tests/
```

### Type Checking

We encourage type checking with `mypy`:

```bash
pip install mypy
mypy src/
```

### Imports

- Use absolute imports
- Group imports: standard library, third-party, local
- Sort imports alphabetically within groups

Example:
```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from evret.metrics.base import BaseMetric
from evret.utils import validate_k
```

## Testing Guidelines

### Writing Tests

- Write unit tests for all new functions and classes
- Place tests in the `tests/` directory mirroring the `src/` structure
- Use descriptive test names: `test_<function>_<scenario>_<expected_result>`

Example:
```python
def test_hit_rate_with_relevant_docs_returns_one():
    """Test that HitRate returns 1.0 when relevant docs are retrieved."""
    metric = HitRate(k=3)
    retrieved = [["doc_1", "doc_2"]]
    relevant = [{"doc_1"}]
    score = metric.score(retrieved, relevant)
    assert score == 1.0
```

### Running Tests

```bash
# Run all unit tests
pytest

# Run specific test file
pytest tests/metrics/test_hit_rate.py

# Run with coverage
pytest --cov=evret --cov-report=html

# Run integration tests (requires Docker)
EVRET_RUN_INTEGRATION=1 pytest -m integration
```

### Test Coverage

- Aim for high test coverage (>80%)
- All new features must include tests
- Bug fixes should include a regression test

## Documentation

### Docstrings

Use Google-style docstrings for all public functions and classes:

```python
def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
    """Retrieve top-k documents for a given query.

    Args:
        query: The search query string.
        k: Number of documents to retrieve.

    Returns:
        List of RetrievalResult objects ordered by relevance score.

    Raises:
        ValueError: If k is less than 1.
    """
```

### README Updates

- Update the README if you add new features
- Include code examples for new functionality
- Keep examples simple and clear

### Examples

- Add examples in the `examples/` directory if appropriate
- Include comments explaining key steps
- Ensure examples are self-contained and runnable

## Adding New Metrics

If you're contributing a new metric:

1. Create a new file in `src/evret/metrics/`
2. Inherit from `BaseMetric`
3. Implement the `_score` method
4. Add comprehensive tests
5. Update the README with usage examples
6. Add the metric to `src/evret/metrics/__init__.py`

Example structure:
```python
from evret.metrics.base import BaseMetric

class NewMetric(BaseMetric):
    """Brief description of the metric."""

    def _score(
        self,
        retrieved_by_query: Sequence[Sequence[str]],
        relevant_by_query: Sequence[set[str]],
        relevance_scores_by_query: Sequence[dict[str, float]] | None = None,
    ) -> float:
        """Implementation of metric calculation."""
        # Your implementation here
        pass
```

## Adding New Vector Database Adapters

If you're contributing a new vector database adapter:

1. Create a new file in `src/evret/retrievers/`
2. Inherit from `BaseRetriever`
3. Implement the `retrieve` method
4. Handle optional dependencies properly
5. Add integration tests
6. Update the README with usage examples
7. Add the retriever to `src/evret/retrievers/__init__.py`

## Questions?

If you have questions about contributing, feel free to:

- Open an issue with the `question` label
- Reach out to the maintainers

Thank you for contributing to Evret!
