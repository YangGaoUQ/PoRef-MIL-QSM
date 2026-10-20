"""Iterative training now uses the patient-split experiment engine."""
from poref.experiment import main

if __name__ == "__main__":
    import sys
    from poref.cli import main as cli_main
    cli_main(["train", "--mode", "poref", *sys.argv[1:]])
