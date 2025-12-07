"""Statistical testing utilities."""
import numpy as np
from scipy import stats
from typing import Dict


def ks_test_two_sample(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Two-sample Kolmogorov-Smirnov test for distribution equality."""
    stat, p_value = stats.ks_2samp(x, y)
    return {"statistic": stat, "p_value": p_value, "distributions_equal": p_value > 0.05}


def levene_test(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Levene's test for equality of variances."""
    stat, p_value = stats.levene(x, y)
    return {"statistic": stat, "p_value": p_value, "equal_variance": p_value > 0.05}


def bootstrap_ci(data: np.ndarray, stat_func=np.mean,
                 n_bootstrap: int = 10000, alpha: float = 0.05,
                 seed: int = 42) -> Dict[str, float]:
    """Bootstrap confidence interval for any statistic."""
    rng = np.random.default_rng(seed)
    boot_stats = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=len(data), replace=True)
        boot_stats.append(stat_func(sample))

    boot_stats = np.array(boot_stats)
    return {
        "estimate": stat_func(data),
        "ci_lower": np.percentile(boot_stats, 100 * alpha / 2),
        "ci_upper": np.percentile(boot_stats, 100 * (1 - alpha / 2)),
        "se": np.std(boot_stats),
    }


def permutation_test(treatment: np.ndarray, control: np.ndarray,
                     n_permutations: int = 10000,
                     seed: int = 42) -> Dict[str, float]:
    """Permutation test for difference in means (non-parametric).

    Does not assume any distributional form — valid even when
    t-test assumptions (normality, equal variance) are violated.
    """
    rng = np.random.default_rng(seed)
    observed_diff = np.mean(treatment) - np.mean(control)
    combined = np.concatenate([treatment, control])
    n_treat = len(treatment)

    count = 0
    for _ in range(n_permutations):
        rng.shuffle(combined)
        perm_diff = np.mean(combined[:n_treat]) - np.mean(combined[n_treat:])
        if abs(perm_diff) >= abs(observed_diff):
            count += 1

    p_value = count / n_permutations
    return {"observed_diff": observed_diff, "p_value": p_value, "n_permutations": n_permutations}
