"""Batched injection results and HDF5 persistence."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import h5py
import jax
import numpy as np
from numpy.typing import NDArray

Array = jax.Array | NDArray
DETECTOR_FIELDS = ("expected_snr", "observed_snr", "horizon", "bandwidth", "epochs")
ARRAY_FIELDS = (*DETECTOR_FIELDS, "network_snr", "detected", "row_index", "series")


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class InjectionResult:
    """Complete results with a leading injection axis, including singletons.

    Attributes
    ----------
    expected_snr, observed_snr, horizon, bandwidth, epochs
        Arrays of shape ``(injections, detectors)``. Horizons are in Mpc,
        bandwidths in rad/s, and epochs in seconds relative to geocentric
        coalescence time (the first sample of each series).
    network_snr, detected, row_index
        Arrays of shape ``(injections,)``. Original population row indices seed
        the noise, independently of input order and batch size.
    series
        Complex SNR, shape ``(injections, detectors, samples)``.
    detectors, sampling_frequency
        Static pytree metadata: detector order and sample rate in Hz.

    Notes
    -----
    Leaves may be NumPy or JAX arrays. HDF5 reads return NumPy arrays and writes
    preserve their numerical precision. Selections use increasing file row
    positions, independently of ``row_index``, and retain the injection axis.
    """

    expected_snr: Array
    observed_snr: Array
    horizon: Array
    bandwidth: Array
    epochs: Array
    network_snr: Array
    detected: Array
    row_index: Array
    series: Array
    detectors: tuple[str, ...] = field(metadata={"static": True})
    sampling_frequency: float = field(metadata={"static": True})

    def write(self, path: str | Path, *, append: bool = False) -> None:
        """Write chunked, uncompressed datasets, optionally appending injections.

        Parameters
        ----------
        path
            Destination HDF5 file.
        append
            Append along the injection axis; shared metadata must match.
        """
        with h5py.File(path, "a" if append else "w") as file:
            if "detectors" in file.attrs:
                if (
                    tuple(file.attrs["detectors"]) != self.detectors
                    or file.attrs["sampling_frequency"] != self.sampling_frequency
                ):
                    raise ValueError("incompatible injection result metadata")
            else:
                file.attrs["detectors"] = self.detectors
                file.attrs["sampling_frequency"] = self.sampling_frequency
            for name in ARRAY_FIELDS:
                values = np.asarray(getattr(self, name))
                if name in file:
                    dataset = file[name]
                    start = len(dataset)
                    dataset.resize(start + len(values), axis=0)
                    dataset[start:] = values
                else:
                    file.create_dataset(
                        name,
                        data=values,
                        maxshape=(None, *values.shape[1:]),
                        chunks=True,
                    )

    @classmethod
    def read(
        cls, path: str | Path, *, selection: slice | NDArray | list[int] | None = None
    ) -> Self:
        """Read all injections or selected increasing row positions into NumPy."""
        with h5py.File(path, "r") as file:
            return cls(
                **{
                    name: file[name][
                        selection if selection is not None else slice(None)
                    ]
                    for name in ARRAY_FIELDS
                },
                detectors=tuple(file.attrs["detectors"]),
                sampling_frequency=float(file.attrs["sampling_frequency"]),
            )
