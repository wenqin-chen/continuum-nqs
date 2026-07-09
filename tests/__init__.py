"""Unit tests for continuum NQS modules.

Each src/ module should have a corresponding test_*.py here. Tests are
designed to run on CPU in seconds — they validate API contracts,
small-tolerance numerical agreement against analytic references, and
shape/dtype invariants. Production-grade convergence tests live in the
notebooks themselves.

Run all tests:
    pytest -v

Run only a specific module's tests:
    pytest tests/test_kinetic.py -v
"""
