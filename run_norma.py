#!/usr/bin/env python3
"""Entry point: python run_norma.py <command> [...]; see `python run_norma.py -h`."""

import sys

from norma.cli import main

if __name__ == "__main__":
    sys.exit(main())
