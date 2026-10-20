"""One forward/backward pass; no learned parameters or path decoding."""
import numpy as np


def probabilities(values):
    p = np.asarray(values, dtype=np.float64)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Expected a nonempty finite ordered probability vector in [0,1]")
    return p


def forward_backward(values, transition=None, prior=None):
    p = probabilities(values)
    a = np.asarray(transition if transition is not None else [[0.9, 0.1], [0.1, 0.9]], dtype=float)
    pi = np.asarray(prior if prior is not None else [0.5, 0.5], dtype=float)
    if a.shape != (2, 2) or not np.isfinite(a).all() or (a < 0).any() or not np.allclose(a.sum(1), 1):
        raise ValueError("transition must be a finite nonnegative 2x2 row-stochastic matrix")
    if pi.shape != (2,) or not np.isfinite(pi).all() or (pi < 0).any() or not np.isclose(pi.sum(), 1):
        raise ValueError("prior must be a finite probability pair")
    p = np.clip(p, 1e-12, 1 - 1e-12)
    emit = np.log(np.column_stack((1 - p, p)))
    log_a = np.full_like(a, -np.inf)
    np.log(a, out=log_a, where=a > 0)
    log_pi = np.full_like(pi, -np.inf)
    np.log(pi, out=log_pi, where=pi > 0)
    alpha = np.empty_like(emit)
    alpha[0] = log_pi + emit[0]
    alpha[0] -= np.logaddexp.reduce(alpha[0])
    for i in range(1, len(p)):
        alpha[i] = np.logaddexp.reduce(alpha[i - 1, :, None] + log_a, axis=0) + emit[i]
        alpha[i] -= np.logaddexp.reduce(alpha[i])
    beta = np.zeros_like(emit)
    for i in range(len(p) - 2, -1, -1):
        beta[i] = np.logaddexp.reduce(log_a + emit[i + 1] + beta[i + 1], axis=1)
        beta[i] -= np.logaddexp.reduce(beta[i])
    log_post = alpha + beta
    log_post -= np.logaddexp.reduce(log_post, axis=1, keepdims=True)
    return np.exp(log_post[:, 1])


def entropy_confidence(values):
    p = probabilities(values)
    q = np.clip(p, 1e-15, 1 - 1e-15)
    entropy = -(q * np.log2(q) + (1 - q) * np.log2(1 - q))
    entropy[(p == 0) | (p == 1)] = 0
    return entropy, np.clip(1 - entropy, 0, 1)


def decide(values, *, window_length, window_threshold, transition=None, prior=None):
    p = probabilities(values)
    if not isinstance(window_length, (int, np.integer)) or not 1 <= window_length <= len(p):
        raise ValueError("window_length must be an integer between 1 and the slice count")
    if not np.isfinite(window_threshold) or not 0 <= window_threshold <= 1:
        raise ValueError("window_threshold must be finite and in [0,1]")
    post = forward_backward(p, transition, prior)
    entropy, confidence = entropy_confidence(post)
    support = post * confidence  # User-confirmed R_i = posterior * (1 - H_i).
    windows = np.convolve(support, np.ones(window_length) / window_length, mode="valid")
    score = float(windows.max())
    return {"score": score, "prediction": int(score > window_threshold),
            "posterior": post.tolist(), "entropy": entropy.tolist(),
            "confidence": confidence.tolist(), "support": support.tolist(),
            "window_scores": windows.tolist(), "best_window_start": int(windows.argmax())}
