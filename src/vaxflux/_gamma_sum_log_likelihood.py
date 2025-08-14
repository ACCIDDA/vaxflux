"""
Utilities for computing the log-likelihood for the sum of gamma-distributed variables.

The utilities in this module can be used to compute the log-likelihood of a sum of
gamma-distributed random variables, which is challenging due to the lack of a simple
formula for the distribution of the sum.

The implementation is based on the closed-form solution for the PDF of the sum of
gamma-distributed variables from P. G. Moschopolulos's "The Distribution Of The Sum Of
Independent Gamma Variables" (1984, https://doi.org/10.1007%2FBF02481123).
"""

__all__: tuple[str, ...] = ()

import jax.numpy as jnp
from jax import Array
from jax.lax import scan
from jax.scipy.special import gammaln
from jax.typing import ArrayLike


def _log_scaling_constant(alpha: Array, beta: Array, min_beta: ArrayLike) -> ArrayLike:
    r"""
    Compute the log scaling constant for the sum of gamma-distributed variables.

    This function computes the log scaling constant used in the PDF of the sum of
    independent gamma-distributed random variables with shape parameters `alpha` and
    scale parameters `beta`. The scaling constant is given by:

    .. math::

        C = \\prod_{i=1}^n\\left(\frac{\\min(\beta_i)}{\beta_i}\right)^{\alpha_i}

    Args:
        alpha: The shape parameters of the summed gamma distributions.
        beta: The scale parameters of the summed gamma distributions.
        min_beta: The minimum value of the scale parameters `beta`.

    Returns:
        The log scaling constant.

    Examples:
        >>> import jax.numpy as jnp
        >>> from vaxflux._gamma_sum_log_likelihood import _log_scaling_constant
        >>> alpha = jnp.arange(3, 6)
        >>> beta = jnp.arange(7, 10)
        >>> min_beta = jnp.min(beta)
        >>> _log_scaling_constant(alpha, beta, min_beta)
        Array(-1.7906973, dtype=float32)
        >>> alpha = beta = min_beta = jnp.array(1.0)
        >>> _log_scaling_constant(alpha, beta, min_beta)
        Array(0., dtype=float32)
    """
    return jnp.sum(alpha * (jnp.log(min_beta) - jnp.log(beta)))


def _gamma_k(
    alpha: Array, beta: Array, min_beta: ArrayLike, k: Array, len_k: int
) -> Array:
    r"""
    Compute the k-th gamma constant for the sum of gamma-distributed variables.

    .. math::

        \gamma_k = \sum_{i=1}^n\frac{\alpha_i}{k}\left(1-\frac{\min(\beta_i)}{\beta_i}\right)^k

    Args:
        alpha: The shape parameters of the summed gamma distributions.
        beta: The scale parameters of the summed gamma distributions.
        min_beta: The minimum value of the scale parameters `beta`.
        k: The k values to compute the gamma constants for.
        len_k: The length of the `k` array.

    Returns:
        A jax array the same length as `k` containing the gamma constants.

    Examples:
        >>> import jax.numpy as jnp
        >>> from vaxflux._gamma_sum_log_likelihood import _gamma_k
        >>> alpha = jnp.arange(3, 6)
        >>> beta = jnp.arange(8, 11)
        >>> min_beta = jnp.min(beta)
        >>> k = jnp.arange(1, 11)
        >>> _gamma_k(alpha, beta, min_beta, k, len(k))
        Array([1.4444444e+00, 1.2469134e-01, 1.5162320e-02, 2.1524152e-03,
               3.3354797e-04, 5.4587766e-05, 9.2623250e-06, 1.6116145e-06,
               2.8559148e-07, 5.1314686e-08], dtype=float32)
        >>> alpha = beta = min_beta = jnp.array(1.0)
        >>> _gamma_k(alpha, beta, min_beta, k, len(k))
        Array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)

    """  # noqa: E501
    len_k = len(k)
    k = k.reshape((-1, 1))
    alpha_expanded = jnp.tile(alpha, (len_k, 1))
    reduce_beta_expanded = 1.0 - (min_beta / jnp.tile(beta, (len_k, 1)))
    return jnp.sum((alpha_expanded / k) * jnp.power(reduce_beta_expanded, k), axis=1)


def _delta_k(alpha: Array, beta: Array, min_beta: ArrayLike, k: Array) -> Array:
    r"""
    Compute the k-th delta constant for the sum of gamma-distributed variables.

    .. math::

        \delta_{k+1} = \frac{1}{k+1}\sum_{i=1}^{k+1}i\gamma_i\delta_{k+1-i},\,\delta_0=1

    Args:
        alpha: The shape parameters of the summed gamma distributions.
        beta: The scale parameters of the summed gamma distributions.
        min_beta: The minimum value of the scale parameters `beta`.
        k: The k values to compute the delta constants for.

    Returns:
        A jax array the same length as `k` containing the delta constants.

    Examples:
        >>> import jax.numpy as jnp
        >>> from vaxflux._gamma_sum_log_likelihood import _delta_k
        >>> alpha = jnp.arange(3, 6)
        >>> beta = jnp.arange(8, 11)
        >>> min_beta = jnp.min(beta)
        >>> k = jnp.arange(11)
        >>> _delta_k(alpha, beta, min_beta, k)
        Array([1.0000000e+00, 7.2222221e-01, 4.3086419e-01, 2.1198902e-01,
               9.1023549e-02, 3.5304897e-02, 1.2654284e-02, 4.2585409e-03,
               1.3612178e-03, 4.1688309e-04, 1.2314973e-04], dtype=float32)
    """
    len_k = len(k)
    i_gamma_k = (k + 1.0) * _gamma_k(alpha, beta, min_beta, k + 1.0, len_k)
    initial_carry = jnp.ones((len_k,), dtype=jnp.float32)
    indices = jnp.arange(len_k)

    def _scan_body(carry: Array, idx: Array) -> tuple[Array, Array]:
        mask = jnp.arange(len_k) < idx
        flipped_carry_slice = jnp.roll(jnp.flip(mask * carry), shift=idx)
        calculated_value = jnp.sum((mask * i_gamma_k) * flipped_carry_slice) / (
            1.0 + idx
        )
        current_value = jnp.where(idx == 0, 1.0, calculated_value)
        new_carry = carry.at[idx].set(current_value)
        return new_carry, current_value

    final_carry, _ = scan(_scan_body, initial_carry, xs=indices)
    return final_carry


def _gamma_sum_log_likelihood(
    y: ArrayLike, alpha: Array, beta: Array, n_k: int = 30
) -> ArrayLike:
    """
    Compute the log-likelihood of a sum of gamma-distributed variables.

    This function computes the log-likelihood of observing a value `y` given that it is
    the sum of independent gamma-distributed random variables with shape parameters
    `alpha` and scale parameters `beta`.

    Args:
        y: The observed value.
        alpha: The shape parameters of the summed gamma distributions.
        beta: The scale parameters of the summed gamma distributions.
        n_k: The number of terms to use in the series expansion.

    Returns:
        The log-likelihood of observing `y`.

    Examples:
        >>> import jax.numpy as jnp
        >>> from vaxflux._gamma_sum_log_likelihood import _gamma_sum_log_likelihood
        >>> alpha = jnp.arange(3, 6)
        >>> beta = jnp.arange(8, 11)
        >>> y = jnp.sum(alpha * beta).item()
        >>> _gamma_sum_log_likelihood(y, alpha, beta)
        Array(-5.0765705, dtype=float32)

    """
    k = jnp.arange(n_k)
    min_beta = jnp.min(beta)
    rho = jnp.sum(alpha)
    scaling_constant = _log_scaling_constant(alpha, beta, min_beta)
    log_delta_k_term = jnp.log(_delta_k(alpha, beta, min_beta, k))
    log_power_y_term = (rho + k - 1.0) * jnp.log(y)
    y_const_term = y / min_beta
    log_gamma_term = gammaln(rho + k)
    log_power_min_beta_term = (rho + k) * jnp.log(min_beta)
    return scaling_constant + jnp.log(
        jnp.sum(
            jnp.exp(
                log_delta_k_term
                + log_power_y_term
                - y_const_term
                - log_gamma_term
                - log_power_min_beta_term
            )
        )
    )
