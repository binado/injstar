# Test guidelines

- Use pytest fixtures and `pytest.mark.parametrize` for testing
- Don't create tests that re-implement functions by hand
- Use `np.testing.assert_allclose` for array comparison
- Don't add tests that compare arrays with fine-tuned tolerances
- Tests marked `slow` call BAYESTAR; CI runs them only on Python 3.14 (matrix is 3.12–3.14).
