"""Legacy name for safe preprocessing; no in-place image overwrite."""
if __name__ == "__main__":
    import sys
    from poref.cli import main
    main(["preprocess", *sys.argv[1:]])
