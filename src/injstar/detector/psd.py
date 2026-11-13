"""Noise spectra.

Resolves a PSD reference to a file and interpolates it onto the frequency grid
of a run. A reference is a gwmock-noise bundled preset, a file in the packaged
``data/noise_curves/`` directory, a path on disk, or an HTTP(S) URL. Whether a
file holds a PSD or an ASD is stated explicitly, never guessed.

``gwmock_noise`` clips frequencies outside the curve's grid to PSD ``0``, an
infinite contribution to an inverse-variance sum. Calling a
:class:`Sensitivity` maps them to ``inf`` instead, so out-of-band bins
contribute nothing.

Nothing here imports JAX.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import numpy as np
from gwmock_noise.gaussian.psd import resolve_bundled_psd_preset
from gwmock_noise.spectral import interpolate_real_spectral_series, load_spectral_series
from numpy.typing import ArrayLike, NDArray

NOISE_CURVES_BASE_DIR = Path(__file__).parent.parent / "data" / "noise_curves"

PsdKind = Literal["psd", "asd"]


def resolve_psd_path(reference: str | Path) -> Path | str:
    """Resolve a PSD reference to something gwmock-noise can load.

    Resolution order: gwmock-noise bundled preset -> packaged
    ``noise_curves/`` file -> absolute/relative path on disk -> HTTP(S) URL.
    Bundled presets and on-disk files return a ``Path``; URLs return the
    original ``str`` (gwmock-noise fetches them directly).

    Parameters
    ----------
    reference
        PSD reference to resolve.

    Raises
    ------
    FileNotFoundError
        If the reference matches none of the above.
    """
    reference_str = str(reference)

    # Only bare preset names need gwmock-noise's bundled-resource lookup.
    if not Path(reference_str).suffix and not any(
        separator in reference_str for separator in ("/", "\\")
    ):
        bundled = resolve_bundled_psd_preset(reference_str)
        if bundled is not None:
            return bundled

    packaged = NOISE_CURVES_BASE_DIR / reference_str
    if packaged.exists():
        return packaged

    path = Path(reference)
    if path.exists():
        return path

    parsed = urlparse(reference_str)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return reference_str

    raise FileNotFoundError(f"Could not resolve PSD reference: {reference!r}")


@dataclass(frozen=True, slots=True)
class Sensitivity:
    """Noise curve of one detector, read on first use.

    Parameters
    ----------
    psd_reference
        Reference accepted by :func:`resolve_psd_path`.
    kind
        ``"psd"`` if the file holds a one-sided power spectral density,
        ``"asd"`` if it holds its square root.
    """

    psd_reference: str | Path
    kind: PsdKind = "psd"
    _curve: tuple[NDArray[np.float64], NDArray[np.float64]] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def _load_curve(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """The PSD grid and values, read on first use and kept afterwards."""
        curve = self._curve
        if curve is None:
            # ``kind`` only labels gwmock-noise's error messages; the file is
            # two real columns either way, and an ASD is squared here.
            grid, values = load_spectral_series(
                resolve_psd_path(self.psd_reference), kind=self.kind.upper()
            )
            if self.kind == "asd":
                values = values**2
            curve = (np.asarray(grid, dtype=float), np.asarray(values, dtype=float))
            object.__setattr__(self, "_curve", curve)
        return curve

    def __call__(self, frequencies: ArrayLike) -> NDArray[np.float64]:
        """Interpolate the one-sided PSD onto ``frequencies``.

        Frequencies outside the curve's grid, and bins where the curve is not
        positive, are ``inf``.

        Parameters
        ----------
        frequencies
            Frequency grid in Hz.
        """
        frequencies = np.asarray(frequencies, dtype=float)
        grid, grid_values = self._load_curve()
        values = interpolate_real_spectral_series(grid, grid_values, frequencies)
        unusable = (
            (frequencies < grid.min()) | (frequencies > grid.max()) | (values <= 0.0)
        )
        return np.where(unusable, np.inf, values)
