"""Root conftest — makes the repo root a Python path root so that
``import backend.vision.pipeline`` resolves correctly during pytest runs
without requiring an editable install.
"""
import sys
import pathlib

# Insert the repository root at the front of sys.path
sys.path.insert(0, str(pathlib.Path(__file__).parent))
