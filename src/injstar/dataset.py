"""Tabular SNR products, independent of the simulation runtime."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
import polars as pl
from numpy.typing import NDArray

DETECTOR_FIELDS = ("expected_snr", "observed_snr", "horizon", "bandwidth")
SCALAR_FIELDS = (*DETECTOR_FIELDS, "network_snr", "detected")


@dataclass(frozen=True)
class SNRDataset:
    """One row per injection, with detector columns in network order.

    Attributes
    ----------
    frame
        SNR table. ``row_index`` is the original zero-based population row.
        Horizons are in Mpc and bandwidths in rad/s.
    detectors
        Detector names defining the order of the NumPy detector axis.
    """

    frame: pl.DataFrame
    detectors: tuple[str, ...]

    @staticmethod
    def schema(detectors: Sequence[str]) -> pl.Schema:
        """Generate the column types for a detector network."""
        return pl.Schema(
            {
                "row_index": pl.UInt64,
                **{
                    f"{field}_{name}": pl.Float64
                    for field in DETECTOR_FIELDS
                    for name in detectors
                },
                "network_snr": pl.Float64,
                "detected": pl.Boolean,
            }
        )

    @classmethod
    def from_arrays(
        cls,
        arrays: Mapping[str, NDArray],
        detectors: Sequence[str],
        rows: NDArray | None = None,
    ) -> Self:
        """Build a table from simulation arrays with an event-leading axis."""
        if rows is None:
            rows = np.arange(len(arrays["detected"]), dtype=np.uint64)
        columns = {"row_index": rows}
        for field in DETECTOR_FIELDS:
            for index, name in enumerate(detectors):
                columns[f"{field}_{name}"] = arrays[field][:, index]
        columns.update({field: arrays[field] for field in ("network_snr", "detected")})
        return cls(
            pl.DataFrame(columns, schema=cls.schema(detectors)), tuple(detectors)
        )

    @classmethod
    def read_csv(cls, path: Path) -> Self:
        """Read an SNR table, inferring network order from expected-SNR columns."""
        frame = pl.read_csv(path)
        detectors = tuple(
            name.removeprefix("expected_snr_")
            for name in frame.columns
            if name.startswith("expected_snr_")
        )
        return cls(
            frame.select(cls.schema(detectors).names()).cast(cls.schema(detectors)),
            detectors,
        )

    def write_csv(self, path: Path) -> None:
        """Write the SNR table without rounding floating-point values."""
        self.frame.write_csv(path)

    def _detector_array(self, field: str) -> NDArray[np.float64]:
        return self.frame.select(
            f"{field}_{name}" for name in self.detectors
        ).to_numpy()

    @property
    def expected_snr(self) -> NDArray[np.float64]:
        """Optimal injected SNR, shape ``(events, detectors)``."""
        return self._detector_array("expected_snr")

    @property
    def observed_snr(self) -> NDArray[np.float64]:
        """Peak matched-filter SNR, shape ``(events, detectors)``."""
        return self._detector_array("observed_snr")

    @property
    def horizon(self) -> NDArray[np.float64]:
        """SNR = 1 distance in Mpc, shape ``(events, detectors)``."""
        return self._detector_array("horizon")

    @property
    def bandwidth(self) -> NDArray[np.float64]:
        """RMS angular-frequency bandwidth, shape ``(events, detectors)``."""
        return self._detector_array("bandwidth")

    @property
    def network_snr(self) -> NDArray[np.float64]:
        """Network SNR, shape ``(events,)``."""
        return self.frame["network_snr"].to_numpy()

    @property
    def detected(self) -> NDArray[np.bool_]:
        """Detection flags, shape ``(events,)``."""
        return self.frame["detected"].to_numpy()
