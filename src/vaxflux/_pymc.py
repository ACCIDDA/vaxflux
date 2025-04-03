import pymc as pm
import pytensor.tensor as pt


def _modified_gamma_logp(x, mu_s, sigma_s):
    """
    Custom modified log-probability function for the gamma distribution.

    This function computes a modified log-probability for the sum of gamma distributions
    with different means and standard deviations for a singular observation.

    Args:
        x: The input value.
        mu_s: The means of the distributions.
        sigma_s: The standard deviations of the distributions.
    """
    alpha_sum = pt.pow(pm.math.sum(mu_s), 2.0) / pm.math.sum(pt.pow(sigma_s, 2.0))
    beta_sum = pm.math.sum(pt.pow(sigma_s, 2.0)) / pm.math.sum(mu_s)
    return (
        (alpha_sum * pm.math.log(beta_sum))
        + (alpha_sum - 1) * pm.math.log(x)
        - (beta_sum * x)
        - pt.gammaln(alpha_sum)
    )
