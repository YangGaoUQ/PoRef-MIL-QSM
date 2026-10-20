"""Explicit-path preprocessing entry; importing this file never modifies data."""
if __name__ == "__main__":
    import sys
    from poref.cli import main
    main(["preprocess", *sys.argv[1:]])
