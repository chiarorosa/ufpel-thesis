#!/usr/bin/env python3
"""Compatibility wrapper to semantic prepare_data entrypoint.

Deprecated: use thesis/scripts/prepare_data.py directly.
"""

from thesis.scripts.prepare_data import main


if __name__ == "__main__":
    main()
