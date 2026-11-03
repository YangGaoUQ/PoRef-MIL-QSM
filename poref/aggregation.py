"""Patient prediction from raw slice probabilities without PDM."""
import numpy as np

from poref.pdm import probabilities

WINDOW_MEAN_FORMULA = "max_j mean(p[j:j+L])"


def sliding_window_mean(values, *, window_length, window_threshold):
    p = probabilities(values)
    if not isinstance(window_length, (int, np.integer)) or not 1 <= window_length <= len(p):
        raise ValueError("window_length must be an integer between 1 and the slice count")
    if not np.isfinite(window_threshold) or not 0 <= window_threshold <= 1:
        raise ValueError("window_threshold must be finite and in [0,1]")
    windows = np.convolve(p, np.ones(window_length) / window_length, mode="valid")
    start = int(np.argmax(windows))
    score = float(windows[start])
    return {"score": score, "prediction": int(score > window_threshold),
            "window_scores": windows.tolist(), "score_formula": WINDOW_MEAN_FORMULA,
            "best_window_start": start,
            "best_window_slices": list(range(start + 1, start + window_length + 1))}
