from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse

from arc_impute import ArcImpute, ArcImputeConfig, ReferenceCurationConfig, TwinPanelConfig
from arc_impute.data import ReferenceDataset, SpatialDataset, prepare_datasets_from_anndata


def make_reference(*, sparse_matrix: bool = False) -> AnnData:
    matrix = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
        ],
        dtype=np.float32,
    )
    if sparse_matrix:
        matrix = sparse.csr_matrix(matrix)
    return AnnData(
        X=matrix,
        obs=pd.DataFrame(
            {"cell_type": ["A", "B", "A"], "batch": ["b1", "b1", "b2"]},
            index=["cell_1", "cell_2", "cell_3"],
        ),
        var=pd.DataFrame(index=["g1", "g2", "g3", "g4"]),
    )


def make_spatial(*, sparse_matrix: bool = False) -> AnnData:
    matrix = np.array(
        [
            [2.0, 0.0, 1.0, 3.0],
            [1.0, 5.0, 2.0, 4.0],
        ],
        dtype=np.float32,
    )
    if sparse_matrix:
        matrix = sparse.csr_matrix(matrix)
    adata = AnnData(
        X=matrix,
        obs=pd.DataFrame(
            {"region": ["R1", "R2"], "quality": [0.9, 0.7]},
            index=["spot_1", "spot_2"],
        ),
        var=pd.DataFrame(index=["g1", "g2", "g3", "g4"]),
    )
    adata.obsm["spatial"] = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    return adata


def make_config() -> ArcImputeConfig:
    return ArcImputeConfig(
        cell_type_key="cell_type",
        region_key="region",
        reference_curation=ReferenceCurationConfig(q_min=1, budget_fraction=1.0),
        twin_panel=TwinPanelConfig(lambda_agree=0.0),
    )


def test_prepare_dense_anndata_preserves_shape_dtype_and_metadata():
    prepared = prepare_datasets_from_anndata(
        reference=make_reference(),
        spatial=make_spatial(),
        config=make_config(),
        train_genes=["g3", "g1"],
    )

    assert prepared.train_genes == ["g3", "g1"]
    assert prepared.reference.expression.dtype == np.float32
    assert prepared.spatial.expression.dtype == np.float32
    assert prepared.reference.expression.shape == (3, 2)
    assert prepared.spatial.expression.shape == (2, 2)
    assert prepared.reference.genes == ["g3", "g1"]
    assert prepared.spatial.genes == ["g3", "g1"]
    assert prepared.curated_reference.var_names.tolist() == ["g1", "g2", "g3", "g4"]
    assert len(prepared.reference.cell_ids) == 3
    assert all("__src" in cell_id for cell_id in prepared.reference.cell_ids)
    assert prepared.spatial.spot_ids == ["spot_1", "spot_2"]
    assert sorted(prepared.reference.cell_types.tolist()) == ["A", "A", "B"]
    assert prepared.spatial.regions.tolist() == ["R1", "R2"]
    np.testing.assert_allclose(prepared.spatial.coordinates, np.array([[0.0, 1.0], [2.0, 3.0]]))
    assert "source_cell_id" in prepared.reference.cell_metadata.columns
    assert prepared.spatial.spot_metadata.loc["spot_2", "quality"] == 0.7


def test_prepare_sparse_anndata_matches_dense_values():
    dense = prepare_datasets_from_anndata(
        reference=make_reference(),
        spatial=make_spatial(),
        config=make_config(),
        train_genes=["g2", "g4"],
    )
    sparse_prepared = prepare_datasets_from_anndata(
        reference=make_reference(sparse_matrix=True),
        spatial=make_spatial(sparse_matrix=True),
        config=make_config(),
        train_genes=["g2", "g4"],
    )

    np.testing.assert_allclose(sparse_prepared.reference.expression, dense.reference.expression)
    np.testing.assert_allclose(sparse_prepared.spatial.expression, dense.spatial.expression)
    np.testing.assert_allclose(sparse_prepared.spatial.density, dense.spatial.density)
    assert sparse.issparse(sparse_prepared.curated_reference.X)


def test_spatial_density_uses_training_genes_and_sums_to_one():
    prepared = prepare_datasets_from_anndata(
        reference=make_reference(),
        spatial=make_spatial(),
        config=make_config(),
        train_genes=["g1"],
    )

    expected = np.array([2.0, 1.0], dtype=np.float32)
    expected = expected / expected.sum()
    np.testing.assert_allclose(prepared.spatial.density, expected)
    assert float(prepared.spatial.density.sum()) == pytest.approx(1.0)


def test_model_prepare_stores_prepared_input():
    model = ArcImpute(make_config()).prepare(
        make_reference(),
        make_spatial(),
        train_genes=["g1", "g2"],
    )

    prepared = model.prepared_
    assert prepared.train_genes == ["g1", "g2"]
    assert prepared.reference.genes == ["g1", "g2"]


@pytest.mark.parametrize(
    ("train_genes", "message"),
    [
        ([], "train_genes must be non-empty"),
        (["g1", "g1"], "must be unique"),
    ],
)
def test_gene_request_validation(train_genes, message):
    with pytest.raises(ValueError, match=message):
        prepare_datasets_from_anndata(
            reference=make_reference(),
            spatial=make_spatial(),
            config=make_config(),
            train_genes=train_genes,
        )


def test_missing_reference_gene_fails_fast():
    with pytest.raises(ValueError, match="reference.var_names is missing required gene 'missing'"):
        prepare_datasets_from_anndata(
            reference=make_reference(),
            spatial=make_spatial(),
            config=make_config(),
            train_genes=["g1", "missing"],
        )


def test_missing_spatial_gene_fails_fast():
    spatial = make_spatial()
    spatial = spatial[:, ["g1", "g2", "g3"]].copy()
    with pytest.raises(ValueError, match="spatial.var_names is missing required gene 'g4'"):
        prepare_datasets_from_anndata(
            reference=make_reference(),
            spatial=spatial,
            config=make_config(),
            train_genes=["g1", "g4"],
        )


def test_missing_cell_type_key_fails_fast():
    with pytest.raises(ValueError, match="reference.obs is missing required column 'unknown'"):
        prepare_datasets_from_anndata(
            reference=make_reference(),
            spatial=make_spatial(),
            config=ArcImputeConfig(cell_type_key="unknown", region_key="region"),
            train_genes=["g1"],
        )


def test_missing_region_key_fails_fast():
    with pytest.raises(ValueError, match="spatial.obs is missing required column 'unknown'"):
        prepare_datasets_from_anndata(
            reference=make_reference(),
            spatial=make_spatial(),
            config=ArcImputeConfig(cell_type_key="cell_type", region_key="unknown"),
            train_genes=["g1"],
        )


def test_missing_cell_type_values_fail_fast():
    reference = make_reference()
    reference.obs.loc["cell_2", "cell_type"] = None
    with pytest.raises(ValueError, match="reference.obs\\['cell_type'\\] must not contain missing values"):
        prepare_datasets_from_anndata(
            reference=reference,
            spatial=make_spatial(),
            config=make_config(),
            train_genes=["g1"],
        )


def test_missing_region_values_fail_fast():
    spatial = make_spatial()
    spatial.obs.loc["spot_1", "region"] = None
    with pytest.raises(ValueError, match="spatial.obs\\['region'\\] must not contain missing values"):
        prepare_datasets_from_anndata(
            reference=make_reference(),
            spatial=spatial,
            config=make_config(),
            train_genes=["g1"],
        )


def test_duplicate_reference_obs_names_fail_fast():
    reference = make_reference()
    reference.obs_names = ["cell_1", "cell_1", "cell_3"]
    with pytest.raises(ValueError, match="reference.obs_names must be unique"):
        prepare_datasets_from_anndata(
            reference=reference,
            spatial=make_spatial(),
            config=make_config(),
            train_genes=["g1"],
        )


def test_duplicate_spatial_var_names_fail_fast():
    spatial = make_spatial()
    spatial.var_names = ["g1", "g1", "g3", "g4"]
    with pytest.raises(ValueError, match="spatial.var_names must be unique"):
        prepare_datasets_from_anndata(
            reference=make_reference(),
            spatial=spatial,
            config=make_config(),
            train_genes=["g1"],
        )


def test_nonfinite_reference_expression_fails_fast():
    reference = make_reference()
    reference.X[0, 0] = np.nan
    with pytest.raises(ValueError, match="reference.X must contain only finite values"):
        prepare_datasets_from_anndata(
            reference=reference,
            spatial=make_spatial(),
            config=make_config(),
            train_genes=["g1"],
        )


def test_zero_spatial_density_fails_fast():
    spatial = make_spatial()
    spatial.X[:, :] = 0
    with pytest.raises(ValueError, match="spatial density must have positive total mass"):
        prepare_datasets_from_anndata(
            reference=make_reference(),
            spatial=spatial,
            config=make_config(),
            train_genes=["g1"],
        )


def test_bad_coordinates_shape_fails_fast():
    spatial = make_spatial()
    spatial.obsm["spatial"] = np.ones((2, 1, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="spatial.obsm\\['spatial'\\] must be 2D"):
        prepare_datasets_from_anndata(
            reference=make_reference(),
            spatial=spatial,
            config=make_config(),
            train_genes=["g1"],
        )


def test_coordinates_can_be_disabled():
    prepared = prepare_datasets_from_anndata(
        reference=make_reference(),
        spatial=make_spatial(),
        config=ArcImputeConfig(
            cell_type_key="cell_type",
            region_key="region",
            spatial_coordinates_key=None,
            reference_curation=ReferenceCurationConfig(q_min=1, budget_fraction=1.0),
        ),
        train_genes=["g1"],
    )

    assert prepared.spatial.coordinates is None


def test_reference_dataset_constructor_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="expression shape must match cell_ids x genes"):
        ReferenceDataset(
            expression=np.ones((2, 2), dtype=np.float32),
            genes=["g1"],
            cell_ids=["c1", "c2"],
            cell_types=np.array(["A", "B"], dtype=object),
            cell_metadata=pd.DataFrame(index=["c1", "c2"]),
        )


def test_spatial_dataset_constructor_rejects_bad_density_shape():
    with pytest.raises(ValueError, match="density must match spot_ids"):
        SpatialDataset(
            expression=np.ones((2, 1), dtype=np.float32),
            genes=["g1"],
            spot_ids=["s1", "s2"],
            regions=np.array(["R1", "R2"], dtype=object),
            density=np.array([[0.5, 0.5]], dtype=np.float32),
            spot_metadata=pd.DataFrame(index=["s1", "s2"]),
        )
