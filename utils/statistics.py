"""Statistical hypothesis testing functions."""

from typing import Dict, Sequence
import numpy as np
from scipy import stats


def perform_paired_t_test(
    scores_model_a: Sequence[float],
    scores_model_b: Sequence[float],
) -> Dict[str, float]:
    """Calculate two-sided paired Student's t-test between fold results."""
    array_a = np.array(scores_model_a)
    array_b = np.array(scores_model_b)
    t_statistic, p_value = stats.ttest_rel(array_a, array_b)
    return {
        "t_statistic": float(t_statistic),
        "p_value": float(p_value),
        "mean_difference": float(np.mean(array_a - array_b)),
    }

