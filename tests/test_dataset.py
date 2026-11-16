from dataclasses import replace
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from injstar.dataset import ARRAY_FIELDS, DETECTOR_FIELDS, InjectionResult


@pytest.fixture(params=[1, 3])
def result(request: pytest.FixtureRequest) -> InjectionResult:
    count = request.param
    values = np.arange(count * 2, dtype=float).reshape(count, 2) + 0.123456789
    return InjectionResult(
        **{field: values for field in DETECTOR_FIELDS},
        network_snr=np.linalg.norm(values, axis=1),
        detected=np.arange(count) % 2 == 0,
        row_index=np.array([7, 2, 2**32 + 9], dtype=np.uint64)[:count],
        series=np.broadcast_to((values + 1j * values)[..., None], (count, 2, 5)),
        detectors=("L1", "H1"),
        sampling_frequency=4096.0,
    )


def assert_result_equal(actual: InjectionResult, expected: InjectionResult) -> None:
    assert actual.detectors == expected.detectors
    assert actual.sampling_frequency == expected.sampling_frequency
    for name in ARRAY_FIELDS:
        actual_values, expected_values = getattr(actual, name), getattr(expected, name)
        assert actual_values.shape == expected_values.shape
        assert actual_values.dtype == expected_values.dtype
        np.testing.assert_array_equal(actual_values, expected_values)


@pytest.mark.parametrize("on_device", [False, True])
def test_hdf5_roundtrip(
    tmp_path: Path, result: InjectionResult, on_device: bool
) -> None:
    path = tmp_path / "results.h5"
    if on_device:
        result = jax.tree.map(jnp.asarray, result)
    result.write(path)
    restored = InjectionResult.read(path)
    assert_result_equal(restored, result)
    assert all(isinstance(leaf, np.ndarray) for leaf in jax.tree.leaves(restored))
    with h5py.File(path) as file:
        assert set(file) == set(ARRAY_FIELDS)
        for name in ARRAY_FIELDS:
            assert file[name].chunks is not None
            assert file[name].compression is None


def test_append(tmp_path: Path, result: InjectionResult) -> None:
    path = tmp_path / "results.h5"
    first = jax.tree.map(lambda values: values[:1], result)
    rest = jax.tree.map(lambda values: values[1:], result)
    first.write(path, append=True)
    rest.write(path, append=True)
    assert_result_equal(InjectionResult.read(path), result)


@pytest.mark.parametrize("selection", [slice(0, 1), slice(None, None, 2), [0], []])
def test_selected_reads(
    tmp_path: Path, result: InjectionResult, selection: slice | list[int]
) -> None:
    path = tmp_path / "results.h5"
    result.write(path)
    expected = jax.tree.map(lambda values: values[selection], result)
    assert_result_equal(InjectionResult.read(path, selection=selection), expected)


@pytest.mark.parametrize(
    "metadata", [{"detectors": ("H1", "L1")}, {"sampling_frequency": 2048.0}]
)
def test_append_requires_matching_metadata(
    tmp_path: Path, result: InjectionResult, metadata: dict
) -> None:
    path = tmp_path / "results.h5"
    result.write(path)
    with pytest.raises(ValueError, match="metadata"):
        replace(result, **metadata).write(path, append=True)
    assert_result_equal(InjectionResult.read(path), result)


def test_pytree_transforms_preserve_metadata(result: InjectionResult) -> None:
    device = jax.tree.map(jnp.asarray, result)
    assert_result_equal(jax.jit(lambda batch: batch)(device), result)
    stacked = jax.tree.map(lambda values: jnp.stack([values, values]), device)
    mapped = jax.vmap(lambda batch: batch)(stacked)
    assert_result_equal(jax.tree.map(lambda values: values[1], mapped), result)
    assert len(jax.tree.leaves(result)) == len(ARRAY_FIELDS)
