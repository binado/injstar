from pathlib import Path

import numpy as np
import pytest
from polars.testing import assert_frame_equal

from injstar.dataset import DETECTOR_FIELDS, SNRDataset


@pytest.mark.parametrize("detectors", [("H1",), ("L1", "H1")])
def test_csv_roundtrip(tmp_path: Path, detectors: tuple[str, ...]) -> None:
    values = np.arange(3 * len(detectors), dtype=float).reshape(3, -1) + 0.123456789
    arrays = {
        **{field: values for field in DETECTOR_FIELDS},
        "network_snr": np.linalg.norm(values, axis=1),
        "detected": np.array([True, False, True]),
    }
    dataset = SNRDataset.from_arrays(arrays, detectors, np.array([7, 2, 9]))
    path = tmp_path / "snr.csv"
    dataset.write_csv(path)
    restored = SNRDataset.read_csv(path)

    assert restored.detectors == detectors
    assert restored.frame.schema == SNRDataset.schema(detectors)
    assert_frame_equal(restored.frame, dataset.frame)
    np.testing.assert_array_equal(restored.frame["row_index"].to_numpy(), [7, 2, 9])
    for field in arrays:
        np.testing.assert_allclose(getattr(restored, field), arrays[field])
