from pathlib import Path

import h5py
import numpy as np
from gwmock_pop.loaders.file_loader import read_population_catalogue

from injstar.io import write_table


def test_run_table_is_a_gwmock_catalogue(tmp_path: Path) -> None:
    path = tmp_path / "table.h5"
    write_table(
        path, {"distance": np.array([10.0, 20.0]), "label": np.array(["é", "a"])}
    )

    with h5py.File(path) as file:
        assert isinstance(file["data"], h5py.Dataset)
    columns = read_population_catalogue(path)
    np.testing.assert_array_equal(columns["distance"], [10.0, 20.0])
    np.testing.assert_array_equal(columns["label"], ["é".encode(), b"a"])
