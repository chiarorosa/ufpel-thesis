#!/usr/bin/env python3
"""Compatibility wrapper to semantic run_pipeline_end_to_end entrypoint.

Deprecated: use thesis/scripts/run_pipeline_end_to_end.py directly.
"""

from thesis.scripts.run_pipeline_end_to_end import main


if __name__ == "__main__":
    main()
