"""Root conftest — makes the repo root a Python path root so that
``import backend.vision.pipeline`` resolves correctly during pytest runs
without requiring an editable install.

It also pins ``DATABASE_URL`` to an in-memory SQLite database *before* any
test module is imported.  ``backend.database`` evaluates ``os.environ`` at
import time, so the URL must be set here — at conftest load time — rather
than inside individual test modules.  Without this guard, whichever test
file is collected first (``backend/tests/test_analyze_endpoint.py`` in the
default testpaths order) triggers the ``backend.database`` import with the
real ``kime.db`` path, and all subsequent ``os.environ`` assignments in test
modules are too late.
"""
import os
import sys
import pathlib

# Must be set before the first ``import backend.database`` anywhere in the
# test suite.  Use setdefault so an explicit DATABASE_URL in the environment
# (e.g. a CI integration-test matrix) is not silently overridden.
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

# Insert the repository root at the front of sys.path
sys.path.insert(0, str(pathlib.Path(__file__).parent))
