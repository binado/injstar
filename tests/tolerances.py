"""Dtype-aware default tolerances for array comparisons in tests.

The defaults follow scipy's ``xp_assert_close``: the relative tolerance is
``sqrt(eps) * 4`` for the inexact dtype under test and the absolute tolerance
is zero, so a comparison cannot pass only because the arrays are small. The
tolerance may be a number or a ``{dtype: value}`` mapping, as in
``jax.test_util.tolerance``, and the less precise of the two dtypes governs.
"""

from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike

type Tolerance = float | Mapping[type | np.dtype, float] | None


def default_rtol(dtype: np.dtype) -> float:
    """Return the default relative tolerance for ``dtype``.

    Parameters
    ----------
    dtype
        Dtype of one of the compared arrays.

    Returns
    -------
    float
        ``sqrt(eps) * 4`` for floating and complex dtypes, and zero for the
        rest, which must match exactly.
    """
    if dtype.kind in "fc":
        return float(np.finfo(dtype).eps) ** 0.5 * 4
    return 0.0


def tolerance(dtype: np.dtype, tol: Tolerance = None) -> float:
    """Resolve a tolerance for ``dtype``.

    Parameters
    ----------
    dtype
        Dtype of one of the compared arrays.
    tol
        A number used for every dtype, a mapping from dtype to value, or
        ``None`` for the default. A dtype missing from the mapping gets the
        default.

    Returns
    -------
    float
        The relative tolerance to use for ``dtype``.
    """
    if tol is None:
        return default_rtol(dtype)
    if isinstance(tol, Mapping):
        table = {np.dtype(key): value for key, value in tol.items()}
        return table.get(dtype, default_rtol(dtype))
    return tol


def assert_close(
    actual: ArrayLike,
    desired: ArrayLike,
    *,
    rtol: Tolerance = None,
    atol: float = 0.0,
    promote_dtypes: bool = False,
) -> None:
    """Assert that two arrays have the same shape and dtype and are close.

    Parameters
    ----------
    actual, desired
        Arrays to compare.
    rtol
        Relative tolerance as accepted by :func:`tolerance`. When the dtypes
        differ the larger resolved value is used. Omit it to use the dtype
        default.
    atol
        Absolute tolerance. It is zero unless a test asks for one.
    promote_dtypes
        Compare after promoting both arrays to their common dtype instead of
        requiring equal dtypes. Use it against references stored at lower
        precision, such as LAL's float32 response tensors. The tolerance still
        comes from the original dtypes, so the less precise array governs.

    Raises
    ------
    AssertionError
        If the shapes differ, the dtypes differ and ``promote_dtypes`` is
        false, or the values are not close.
    """
    actual, desired = np.asarray(actual), np.asarray(desired)
    assert actual.shape == desired.shape, (actual.shape, desired.shape)
    rtol = max(tolerance(actual.dtype, rtol), tolerance(desired.dtype, rtol))
    if promote_dtypes:
        common = np.result_type(actual, desired)
        actual, desired = actual.astype(common), desired.astype(common)
    assert actual.dtype == desired.dtype, (actual.dtype, desired.dtype)
    np.testing.assert_allclose(actual, desired, rtol=rtol, atol=atol)
