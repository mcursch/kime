"""Application configuration resolved from environment variables."""

import os
from pathlib import Path

# Directory where uploaded video files are persisted.
# Configurable so tests and production can point to different locations.
UPLOAD_DIR: Path = Path(os.getenv("UPLOAD_DIR", "uploads"))
