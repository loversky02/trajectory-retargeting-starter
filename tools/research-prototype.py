#!/usr/bin/env python3
"""Run the separately versioned offline IL research package from this checkout."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_pipeline.cli import main

if __name__ == "__main__":
    main()
