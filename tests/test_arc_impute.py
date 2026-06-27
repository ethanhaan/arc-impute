from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from arc_impute import (
    AnchorResidualConfig,
    ArcImpute,
    ArcImputeConfig,
    OptimisationConfig,
    ReferenceCurationConfig,
    TwinPanelConfig,
)


def make_reference() -> AnnData:
    return AnnData(
        X=np.array(
            [
                [1.0, 2.0, 3.0],
                [3.0, 4.0, 5.0],
                [5.0, 1.0, 2.0],
                [2.0, 5.0, 1.0],
            ],
            dtype=np.float32,
        ),
        obs=pd.DataFrame({"cell_type": ["A", "A", "B", "B"]}, index=["cell_a", "cell_b", "cell_c", "cell_d"]),
        var=pd.DataFrame(index=["g1", "g2", "g3"]),
    )


def make_spatial() -> AnnData:
    return AnnData(
        X=np.array([[2.0, 1.0], [5.0, 6.0]], dtype=np.float32),
        obs=pd.DataFrame({"region": ["R1", "R2"]}, index=["spot_1", "spot_2"]),
        var=pd.DataFrame(index=["g1", "g2"]),
    )


def small_config(*, num_epochs: int = 2, twin_lambda: float = 0.0) -> ArcImputeConfig:
    return ArcImputeConfig(
        cell_type_key="cell_type",
        region_key="region",
        device="cpu",
        reference_curation=ReferenceCurationConfig(q_min=1, budget_fraction=1.0),
        anchor_residual=AnchorResidualConfig(coarse_optimisation_epochs=2, coarse_optimisation_runs=1),
        twin_panel=TwinPanelConfig(lambda_agree=twin_lambda),
        optimisation=OptimisationConfig(num_epochs=num_epochs, print_each=None),
    )


def test_config_requires_cell_type_key():
    with pytest.raises(ValueError, match="cell_type_key is required"):
        ArcImputeConfig(cell_type_key="", region_key="region")


def test_config_requires_region_key():
    with pytest.raises(ValueError, match="region_key is required"):
        ArcImputeConfig(cell_type_key="cell_type", region_key="")


def test_config_requires_device():
    with pytest.raises(ValueError, match="device is required"):
        ArcImputeConfig(cell_type_key="cell_type", region_key="region", device="")


def test_config_defaults_to_cpu():
    config = ArcImputeConfig(cell_type_key="cell_type", region_key="region")

    assert config.device == "cpu"


def test_config_requires_nonnegative_seed():
    with pytest.raises(ValueError, match="seed must be >= 0"):
        ArcImputeConfig(cell_type_key="cell_type", region_key="region", seed=-1)


def test_arc_impute_requires_config():
    with pytest.raises(ValueError, match="config is required"):
        ArcImpute(None)


def test_prepare_is_stateful_and_returns_model():
    model = ArcImpute(small_config())

    returned = model.prepare(make_reference(), make_spatial(), train_genes=["g1", "g2"])

    assert returned is model
    assert model.prepared_ is not None
    assert model.prepared_.train_genes == ["g1", "g2"]
    assert model.prepared_.reference.expression.shape == (4, 2)
    assert model.prepared_.spatial.expression.shape == (2, 2)
    assert model.prepared_.curated_reference.var_names.tolist() == ["g1", "g2", "g3"]
    assert model.prepared_.diagnostics["reference_curation"]["curated_cells"] == 4
    assert model.assignment_ is None


def test_fit_requires_prepare():
    with pytest.raises(ValueError, match=r"call prepare\(\) before fit"):
        ArcImpute(small_config()).fit()


@pytest.mark.parametrize("method_name", ["impute", "deconvolve"])
def test_outputs_require_fit(method_name):
    model = ArcImpute(small_config()).prepare(make_reference(), make_spatial(), train_genes=["g1", "g2"])

    with pytest.raises(ValueError, match=r"call prepare\(\) and fit\(\) first"):
        if method_name == "impute":
            model.impute(genes=["g3"])
        else:
            model.deconvolve()


def test_prepare_fit_impute_and_deconvolve_lifecycle():
    model = ArcImpute(small_config())
    model.prepare(make_reference(), make_spatial(), train_genes=["g1", "g2"])

    returned = model.fit()
    imputed = model.impute(genes=["g3"])
    composition = model.deconvolve()

    assert returned is model
    assert model.assignment_.shape == (4, 2)
    np.testing.assert_allclose(model.assignment_.sum(axis=1), np.ones(4), rtol=1e-5)
    assert imputed.index.tolist() == ["spot_1", "spot_2"]
    assert imputed.columns.tolist() == ["g3"]
    assert composition.index.tolist() == ["spot_1", "spot_2"]
    assert composition.columns.tolist() == ["A", "B"]
    assert "assignment_routing" in model.diagnostics_
    assert "anchor_residual" in model.diagnostics_


def test_prepare_again_clears_fitted_state():
    model = ArcImpute(small_config())
    model.prepare(make_reference(), make_spatial(), train_genes=["g1", "g2"]).fit()

    model.prepare(make_reference(), make_spatial(), train_genes=["g1"])

    assert model.assignment_ is None
    assert model.training_history_ is None
    assert model.diagnostics_ is None
    with pytest.raises(ValueError, match=r"call prepare\(\) and fit\(\) first"):
        model.impute(genes=["g3"])


def test_failed_prepare_clears_previous_state():
    model = ArcImpute(small_config())
    model.prepare(make_reference(), make_spatial(), train_genes=["g1", "g2"]).fit()

    with pytest.raises(ValueError, match="missing required gene 'missing'"):
        model.prepare(make_reference(), make_spatial(), train_genes=["missing"])

    assert model.prepared_ is None
    assert model.assignment_ is None


def test_twin_fit_records_panel_diagnostics():
    spatial = make_spatial()
    spatial = AnnData(
        X=np.column_stack([spatial.X, np.array([3.0, 1.0], dtype=np.float32)]),
        obs=spatial.obs.copy(),
        var=pd.DataFrame(index=["g1", "g2", "g3"]),
    )
    model = ArcImpute(small_config(num_epochs=1, twin_lambda=0.05))
    model.prepare(make_reference(), spatial, train_genes=["g1", "g2", "g3"]).fit()

    assert model.assignment_.shape == (4, 2)
    assert model.diagnostics_["twin_panel"]["fusion_mode"] == "mean"
    assert "final_mean_panel_disagreement" in model.training_history_
