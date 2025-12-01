# Maya4 Tests

This directory contains the test suite for the Maya4 package.

## Test Structure

```
tests/
├── conftest.py              # Pytest fixtures and configuration
├── test_installation.py     # Basic installation and import tests
├── test_dataloader_clean.py # Tests for refactored dataloader components
├── test_normalization.py    # Tests for normalization modules
└── test_utils.py           # Tests for utility functions
```

## Running Tests

### Install development dependencies

```bash
pip install -e ".[dev]"
```

### Run all tests

```bash
pytest
```

### Run with coverage report

```bash
pytest --cov=maya4 --cov-report=html
```

Then open `htmlcov/index.html` in your browser to view the coverage report.

### Run specific test files

```bash
pytest tests/test_dataloader_clean.py
```

### Run specific test classes or functions

```bash
pytest tests/test_dataloader_clean.py::TestLazyCoordinateRange
pytest tests/test_dataloader_clean.py::TestLazyCoordinateRange::test_initialization
```

### Run with verbose output

```bash
pytest -v
```

### Run tests matching a pattern

```bash
pytest -k "test_initialization"
```

### Run tests with specific markers

```bash
# Run only unit tests
pytest -m unit

# Skip slow tests
pytest -m "not slow"
```

## Using the Makefile

For convenience, you can use the provided Makefile:

```bash
# Run tests with coverage
make test

# Run tests without coverage (faster)
make test-fast

# Run with verbose output
make test-verbose

# Run linting checks
make lint

# Auto-format code
make format

# View coverage report in browser
make test-cov
```

## CI/CD Integration

Tests run automatically on GitHub Actions for:
- Every push to `main`, `pypi`, and `feature/*` branches
- Every pull request to `main` and `pypi`
- Multiple Python versions (3.9, 3.10, 3.11, 3.12)
- Multiple operating systems (Ubuntu, macOS, Windows)

See `.github/workflows/tests.yml` for the full CI configuration.

## Writing New Tests

### Test Structure

Follow the existing pattern:

```python
class TestMyComponent:
    """Tests for MyComponent class."""
    
    def test_initialization(self):
        """Test basic initialization."""
        # Test code here
        assert True
    
    def test_specific_behavior(self):
        """Test specific behavior."""
        # Test code here
        pass
```

### Using Fixtures

Define reusable fixtures in `conftest.py`:

```python
@pytest.fixture
def my_fixture():
    """Description of fixture."""
    # Setup
    obj = create_test_object()
    yield obj
    # Teardown (optional)
    cleanup(obj)
```

Use fixtures in tests:

```python
def test_with_fixture(my_fixture):
    """Test using the fixture."""
    assert my_fixture is not None
```

### Test Markers

Mark tests with custom markers:

```python
@pytest.mark.slow
def test_slow_operation():
    """This test takes a long time."""
    pass

@pytest.mark.integration
def test_full_pipeline():
    """Integration test for full pipeline."""
    pass
```

## Coverage Goals

We aim for:
- **80%+** overall code coverage
- **90%+** coverage for core dataloader components
- **100%** coverage for critical utility functions

## Common Test Patterns

### Testing with temporary files

```python
def test_with_temp_files(temp_dir):
    """Use the temp_dir fixture from conftest.py."""
    test_file = temp_dir / "test.txt"
    test_file.write_text("test content")
    assert test_file.exists()
```

### Testing exceptions

```python
def test_invalid_input():
    """Test that invalid input raises appropriate error."""
    with pytest.raises(ValueError, match="Invalid"):
        function_that_should_raise("invalid")
```

### Testing numerical arrays

```python
import numpy as np

def test_array_output():
    """Test numerical array output."""
    result = compute_array()
    expected = np.array([1, 2, 3])
    np.testing.assert_array_almost_equal(result, expected)
```

## Troubleshooting

### Import errors

If you get import errors, make sure you've installed the package in editable mode:

```bash
pip install -e .
```

### Missing dependencies

Install all development dependencies:

```bash
pip install -e ".[dev]"
```

### Tests failing in CI but passing locally

- Check that you're testing with the same Python version
- Ensure all test dependencies are in `pyproject.toml`
- Check for platform-specific issues (path separators, etc.)
