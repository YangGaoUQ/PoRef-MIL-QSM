"""Patient-level metrics with explicit confusion matrix and std convention."""
import numpy as np

METRIC_NAMES = ("auc", "acc", "sen", "spe", "pre", "f1")


def evaluate(labels, scores, predictions):
    y = np.asarray(labels)
    s = np.asarray(scores, dtype=float)
    pred = np.asarray(predictions)
    if y.ndim != 1 or len(y) == 0 or s.shape != y.shape or pred.shape != y.shape:
        raise ValueError("labels, scores and predictions must be equal-length nonempty vectors")
    if not np.isin(y, [0, 1]).all() or not np.isin(pred, [0, 1]).all() or not np.isfinite(s).all():
        raise ValueError("Expected binary labels/predictions and finite scores")
    tp = int(((y == 1) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    ratio = lambda a, b: float(a / b) if b else None
    positive, negative = s[y == 1], s[y == 0]
    auc = None
    if len(positive) and len(negative):
        auc = float(((positive[:, None] > negative).sum() + 0.5 * (positive[:, None] == negative).sum()) / (len(positive) * len(negative)))
    thresholds = np.r_[np.inf, np.unique(s)[::-1]]
    roc = [{"threshold": None if np.isinf(t) else float(t),
            "tpr": ratio(int(((s >= t) & (y == 1)).sum()), len(positive)),
            "fpr": ratio(int(((s >= t) & (y == 0)).sum()), len(negative))} for t in thresholds]
    return {"auc": auc, "acc": (tp + tn) / len(y), "sen": ratio(tp, tp + fn),
            "spe": ratio(tn, tn + fp), "pre": ratio(tp, tp + fp),
            "f1": ratio(2 * tp, 2 * tp + fp + fn), "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "confusion_matrix": [[tp, fn], [fp, tn]], "n": len(y), "roc": roc}


def summarize(fold_metrics, ddof=1):
    if ddof not in (0, 1) or not fold_metrics:
        raise ValueError("Require fold metrics and ddof=0 or ddof=1")
    summary = {"ddof": ddof, "folds": len(fold_metrics)}
    for key in METRIC_NAMES:
        vals = [r[key] for r in fold_metrics]
        complete = all(v is not None for v in vals)
        summary[key] = {"mean": float(np.mean(vals)) if complete else None,
                        "std": float(np.std(vals, ddof=ddof)) if complete and len(vals) > ddof else None,
                        "defined_folds": sum(v is not None for v in vals)}
    return summary
