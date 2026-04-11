#!/usr/bin/env python3
"""Compatibility wrapper to semantic train_pipeline entrypoint.

Deprecated: use thesis/scripts/train_pipeline.py directly.
"""

from thesis.scripts.train_pipeline import main


if __name__ == "__main__":
    main()
