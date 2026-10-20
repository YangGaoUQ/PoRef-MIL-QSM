"""Compatibility entry: final evaluation is patient-level PDM, not slice AUC."""
from poref.experiment import main

if __name__ == "__main__":
    import sys
    from poref.cli import main as cli_main
    cli_main(["train", "--mode", "poref", *sys.argv[1:]])
