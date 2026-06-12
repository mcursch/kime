"""Allow `python -m data_pipeline` as an alias for `python -m data_pipeline.cli`."""
import sys

from data_pipeline.cli import main

if __name__ == "__main__":
    result = main()
    if result is not None:
        sys.exit(result)
