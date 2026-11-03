
"""Calculate evaluation metrics for binary attack detection."""


def calculate_metrics(actual, predicted):
    """1 = attack, 0 = normal. Predicted 1 means attack detected."""

    if len(actual) != len(predicted):
        raise ValueError("Actual and predicted lists must have equal length")

    if not actual:
        raise ValueError("Input lists cannot be empty")

    if any(x not in (0, 1) for x in actual + predicted):
        raise ValueError("Values must be 0 or 1")

    tp = sum(a == 1 and p == 1 for a, p in zip(actual, predicted))
    tn = sum(a == 0 and p == 0 for a, p in zip(actual, predicted))
    fp = sum(a == 0 and p == 1 for a, p in zip(actual, predicted))
    fn = sum(a == 1 and p == 0 for a, p in zip(actual, predicted))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    fpr = fp / (fp + tn) if fp + tn else 0.0

    return {
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall_attack": recall,
        "f1_score": f1,
        "false_positive_rate": fpr,
    }


if __name__ == "__main__":
    # Small hand-checkable example; not real project results.
    actual =    [1, 1, 0, 0, 1]
    predicted = [1, 0, 0, 1, 1]

    results = calculate_metrics(actual, predicted)

    print("DEMO ONLY - NOT REAL MODEL RESULTS")
    for name, value in results.items():
        print(f"{name}: {value}")

def calculate_asr(attack_objective_achieved):
    """Calculate attack success rate from known outcomes."""
    known = [x for x in attack_objective_achieved if x is not None]

    if not known:
        raise ValueError("At least one known outcome is required")

    if any(not isinstance(x, bool) for x in known):
        raise ValueError("Outcomes must be True, False, or None")

    return sum(known) / len(known)