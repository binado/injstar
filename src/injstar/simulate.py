"""SNR-level simulation.

Batched JAX code that draws each detector's matched-filter SNR time series
directly: expected complex SNR, whitened template autocorrelation and correlated
Gaussian noise from one inverse FFT per detector, plus detection flags.

Design: "SNR-level simulation" in docs/design.md.
Not implemented yet.
"""
