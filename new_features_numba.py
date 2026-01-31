import numpy as np
try:
    import numba as nb  # type: ignore
except ModuleNotFoundError:
    # Define a dummy "nb" object with a no-op njit decorator.  When numba
    # is unavailable the functions will execute in pure Python.  This
    # preserves API compatibility for downstream code that expects an
    # ``@nb.njit`` decorator.
    class _DummyNumba:
        def njit(self, *args, **kwargs):  # type: ignore
            def decorator(func):
                return func
            return decorator

    nb = _DummyNumba()  # type: ignore


# =========================
# Rolling Z-score (Numba)
# =========================
@nb.njit
def rolling_z_numba(x: np.ndarray, window: int) -> np.ndarray:
    """Compute a rolling Z-score over a window.

    Parameters
    ----------
    x : ndarray
        Input array of values.  NaNs are ignored and propagate into the
        output until enough non-NaN observations have been seen.
    window : int
        Number of observations to include in the rolling window.

    Returns
    -------
    ndarray
        Array of the same length as ``x`` containing Z-scores.  Values
        where the variance is zero or the window is incomplete are set
        to NaN.
    """
    n = len(x)
    out = np.full(n, np.nan)
    buf = np.zeros(window)
    s = 0.0
    ss = 0.0
    cnt = 0
    for i in range(n):
        xi = x[i]
        if np.isnan(xi):
            continue
        if cnt < window:
            buf[cnt] = xi
            s += xi
            ss += xi * xi
            cnt += 1
        else:
            j = i % window
            old = buf[j]
            buf[j] = xi
            s += xi - old
            ss += xi * xi - old * old
        if cnt == window:
            mean = s / window
            var = ss / window - mean * mean
            if var > 0:
                out[i] = (xi - mean) / np.sqrt(var)
    return out


# =========================
# Run-length (Numba)
# =========================
@nb.njit
def run_length_numba(sign: np.ndarray) -> np.ndarray:
    """Compute the run length of consecutive equal non-zero signs.

    sign : ndarray of ints
        Input array with values -1, 0, or 1.  Consecutive non-zero
        values produce increasing run lengths.  A zero resets the
        count.  A change in sign resets the count to 1.

    Returns
    -------
    ndarray of ints
        Run length at each position.
    """
    n = len(sign)
    out = np.zeros(n, dtype=np.int32)
    prev = 0
    cnt = 0
    for i in range(n):
        sgn = sign[i]
        if sgn == 0:
            cnt = 0
            out[i] = 0
        elif sgn == prev:
            cnt += 1
            out[i] = cnt
        else:
            cnt = 1
            out[i] = 1
            prev = sgn
    return out


# =========================
# Tick spread in bp
# =========================
@nb.njit
def tick_spread_bp(price: np.ndarray) -> np.ndarray:
    """Compute the absolute change in price per tick in basis points."""
    n = len(price)
    out = np.full(n, np.nan)
    for i in range(1, n):
        if price[i] > 0 and price[i - 1] > 0:
            out[i] = abs(price[i] - price[i - 1]) / price[i - 1] * 10_000.0
    return out