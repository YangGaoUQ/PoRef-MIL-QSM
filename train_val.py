"""Run training from the source tree: python train_val.py --help."""

import sys

from poref.cli import main

if __name__ == "__main__":
    main(["train", *sys.argv[1:]])
