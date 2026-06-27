from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse

from arc_impute.results import deconvolve_assignment, impute_expression, validate_assignment


def make_curated_reference(*, sparse_matrix: bool = False) -> AnnData:
    expression = np.array(
        [
            [1.0, 10.0, 100.0],
            [2.0, 20.0, 200.0],
            [4.0, 40.0, 400.0],
        ],
        dtype=np.float32,
    )
    if sparse_matrix:
        expression = sparse.csr_matrix(expression)
    return AnnData(
        X=expression,
        obs=pd.DataFrame({"cell_type": ["A", "B", "A"]}, index=["c1", "c2", "c3"]),
        var=pd.DataFrame(index=["g1", "g2", "g3"]),
    )


def make_assignment() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0],
            [0.25, 0.75],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )


@pytest.mark.parametrize("sparse_matrix", [False, True])
def test_impute_returns_spot_by_gene_dataframe(sparse_matrix):
    imputed = impute_expression(
        assignment=make_assignment(),
        curated_reference=make_curated_reference(sparse_matrix=sparse_matrix),
        spot_ids=["s1", "s2"],
        genes=["g3", "g1"],
    )

    assert imputed.index.tolist() == ["s1", "s2"]
    assert imputed.columns.tolist() == ["g3", "g1"]
    expected = np.array(
        [
            [150.0, 1.5],
            [550.0, 5.5],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(imputed.to_numpy(), expected)


def test_impute_requires_nonempty_unique_genes():
    with pytest.raises(ValueError, match="genes must be non-empty"):
        impute_expression(
            assignment=make_assignment(),
            curated_reference=make_curated_reference(),
            spot_ids=["s1", "s2"],
            genes=[],
        )

    with pytest.raises(ValueError, match="genes must be unique"):
        impute_expression(
            assignment=make_assignment(),
            curated_reference=make_curated_reference(),
            spot_ids=["s1", "s2"],
            genes=["g1", "g1"],
        )


def test_impute_unknown_gene_fails_fast():
    with pytest.raises(ValueError, match="curated reference is missing gene 'missing'"):
        impute_expression(
            assignment=make_assignment(),
            curated_reference=make_curated_reference(),
            spot_ids=["s1", "s2"],
            genes=["missing"],
        )


def test_deconvolve_returns_normalised_spot_composition():
    composition = deconvolve_assignment(
        assignment=make_assignment(),
        cell_types=np.array(["A", "B", "A"], dtype=object),
        spot_ids=["s1", "s2"],
    )

    assert composition.index.tolist() == ["s1", "s2"]
    assert composition.columns.tolist() == ["A", "B"]
    expected = np.array(
        [
            [1.0 / 1.25, 0.25 / 1.25],
            [1.0 / 1.75, 0.75 / 1.75],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(composition.to_numpy(), expected)
    np.testing.assert_allclose(composition.sum(axis=1).to_numpy(), np.ones(2))


def test_deconvolve_sorts_cell_type_columns_for_stable_composition():
    composition = deconvolve_assignment(
        assignment=np.ones((3, 2), dtype=np.float32),
        cell_types=np.array(["B", "A", "B"], dtype=object),
        spot_ids=["s1", "s2"],
    )

    assert composition.columns.tolist() == ["A", "B"]


def test_validate_assignment_rejects_bad_values():
    with pytest.raises(ValueError, match="shape reference cells x spatial spots"):
        validate_assignment(np.ones((2, 2)), n_reference_cells=3, n_spatial_spots=2)

    nonfinite = np.ones((3, 2), dtype=np.float32)
    nonfinite[0, 0] = np.nan
    with pytest.raises(ValueError, match="contain only finite values"):
        validate_assignment(nonfinite, n_reference_cells=3, n_spatial_spots=2)

    negative = np.ones((3, 2), dtype=np.float32)
    negative[1, 1] = -0.1
    with pytest.raises(ValueError, match="must be non-negative"):
        validate_assignment(negative, n_reference_cells=3, n_spatial_spots=2)


def test_deconvolve_rejects_spots_with_zero_assignment_mass():
    assignment = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )

    with pytest.raises(ValueError, match="every spot must receive positive assignment mass"):
        deconvolve_assignment(
            assignment=assignment,
            cell_types=np.array(["A", "B", "A"], dtype=object),
            spot_ids=["s1", "s2"],
        )
