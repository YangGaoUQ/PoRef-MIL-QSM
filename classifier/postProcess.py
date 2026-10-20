"""Compatibility import for the final forward/backward PDM (no Viterbi)."""
from poref.postprocess import main

if __name__ == "__main__":
    import sys
    from poref.cli import main as cli_main
    cli_main(["postprocess", *sys.argv[1:]])
