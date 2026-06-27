from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from anndata import AnnData

from arc_impute import (
    AnchorResidualConfig,
    ArcImputeConfig,
    AssignmentRoutingConfig,
    OptimisationConfig,
    ReferenceCurationConfig,
    TwinPanelConfig,
)
from arc_impute.modules.anchor_residual import (
    aggregate_reference_clusters,
    build_anchor_residual,
    compute_optimised_anchor,
)
from arc_impute.data import ReferenceDataset, SpatialDataset, prepare_datasets_from_anndata
from arc_impute.modules.reference_curation_routing import (
    build_assignment_routing,
    build_centroid_curated_reference,
    compute_curation_data,
    repair_quotas,
)
from arc_impute.modules.twin_panel_consistency import (
    build_twin_panel_split,
    composition_from_assignment,
    fuse_branch_assignments,
    twin_composition_loss,
)


def make_reference_adata() -> AnnData:
    return AnnData(
        X=np.array(
            [
                [1.0, 2.0, 4.0, 8.0],
                [1.2, 2.2, 4.2, 8.2],
                [7.0, 3.0, 2.0, 1.0],
                [7.2, 3.2, 2.2, 1.2],
            ],
            dtype=np.float32,
        ),
        obs=pd.DataFrame({"cell_type": ["A", "A", "B", "B"]}, index=["c1", "c2", "c3", "c4"]),
        var=pd.DataFrame(index=["g1", "g2", "g3", "g4"]),
    )


def make_spatial_adata() -> AnnData:
    return AnnData(
        X=np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [2.0, 3.0, 4.0, 5.0],
                [7.0, 2.0, 1.0, 2.0],
            ],
            dtype=np.float32,
        ),
        obs=pd.DataFrame({"region": ["R1", "R1", "R2"]}, index=["s1", "s2", "s3"]),
        var=pd.DataFrame(index=["g1", "g2", "g3", "g4"]),
    )


def make_spatial_adata_with_zero_train_spot() -> AnnData:
    return AnnData(
        X=np.array(
            [
                [0.0, 0.0, 0.0, 4.0],
                [2.0, 3.0, 4.0, 5.0],
                [7.0, 2.0, 1.0, 2.0],
            ],
            dtype=np.float32,
        ),
        obs=pd.DataFrame({"region": ["R1", "R1", "R2"]}, index=["s1", "s2", "s3"]),
        var=pd.DataFrame(index=["g1", "g2", "g3", "g4"]),
    )


def make_prepared():
    config = ArcImputeConfig(
        cell_type_key="cell_type",
        region_key="region",
        device="cpu",
        reference_curation=ReferenceCurationConfig(q_min=1, budget_fraction=1.0),
        twin_panel=TwinPanelConfig(lambda_agree=0.0),
    )
    return prepare_datasets_from_anndata(
        reference=make_reference_adata(),
        spatial=make_spatial_adata(),
        config=config,
        train_genes=["g1", "g2", "g3"],
    )


def test_repair_quotas_preserves_budget_and_minimum():
    repaired = repair_quotas(
        labels=["A", "B", "C"],
        initial_quota=np.array([4, 4, 4]),
        target_float=np.array([3.4, 1.3, 1.3]),
        budget=6,
        q_min=1,
    )

    assert repaired.sum() == 6
    assert repaired.min() >= 1


def test_centroid_curated_reference_oversamples_with_copy_ids():
    quotas = pd.Series([3, 1], index=pd.Index(["A", "B"], name="label"))
    curated, selected = build_centroid_curated_reference(
        make_reference_adata(),
        quotas=quotas,
        label_key="cell_type",
        genes=["g1", "g2"],
    )

    assert curated.n_obs == 4
    assert curated.obs_names.is_unique
    assert selected["reference_curation_copy_index"].max() == 1


def test_curation_data_outputs_probabilities_and_quota_table():
    curation_data = compute_curation_data(
        make_reference_adata(),
        make_spatial_adata(),
        genes=["g1", "g2"],
        label_key="cell_type",
        similarity="pearson",
        representation="centroid",
        tau=0.1,
        gamma=1.0,
        budget=4,
        q_min=1,
        max_programs=4,
    )

    assert curation_data["probabilities"].shape == (3, 2)
    np.testing.assert_allclose(curation_data["probabilities"].sum(axis=1).to_numpy(), np.ones(3))
    assert int(curation_data["quotas"].sum()) == 4


def test_assignment_routing_bias_shapes_and_clipping():
    prepared = make_prepared()
    routing = build_assignment_routing(
        prepared.reference,
        prepared.spatial,
        train_genes=prepared.train_genes,
        config=AssignmentRoutingConfig(bias_clip_min=-0.5, bias_clip_max=0.5),
    )

    assert routing.spot_celltype_logit_bias.shape == (2, 3)
    assert routing.spot_cell_logit_bias.shape == (4, 3)
    assert routing.cell_label_indices.shape == (4,)
    assert routing.labels == ["A", "B"]
    assert float(routing.spot_cell_logit_bias.min()) >= -0.075001
    assert float(routing.spot_cell_logit_bias.max()) <= 0.075001


def test_assignment_routing_accepts_zero_expression_train_spots():
    config = ArcImputeConfig(
        cell_type_key="cell_type",
        region_key="region",
        device="cpu",
        reference_curation=ReferenceCurationConfig(q_min=1, budget_fraction=1.0),
        twin_panel=TwinPanelConfig(lambda_agree=0.0),
    )
    prepared = prepare_datasets_from_anndata(
        reference=make_reference_adata(),
        spatial=make_spatial_adata_with_zero_train_spot(),
        config=config,
        train_genes=["g1", "g2", "g3"],
    )

    routing = build_assignment_routing(
        prepared.reference,
        prepared.spatial,
        train_genes=prepared.train_genes,
        config=AssignmentRoutingConfig(),
    )

    assert np.isfinite(routing.spot_cell_logit_bias).all()
    assert routing.summary["cell_fit_min"] == 0.5


def test_optimised_anchor_shape_and_gate_groups():
    prepared = make_prepared()
    optimised_logits, optimised_summary = compute_optimised_anchor(
        prepared.reference,
        prepared.spatial,
        train_genes=prepared.train_genes,
        anchor_config=AnchorResidualConfig(coarse_optimisation_epochs=1, coarse_optimisation_runs=1),
        optimisation_config=OptimisationConfig(num_epochs=1, print_each=None),
        device="cpu",
    )
    anchor = build_anchor_residual(
        prepared.reference,
        prepared.spatial,
        train_genes=prepared.train_genes,
        anchor_config=AnchorResidualConfig(
            gate_mode="region",
            coarse_optimisation_epochs=1,
            coarse_optimisation_runs=1,
        ),
        optimisation_config=OptimisationConfig(num_epochs=1, print_each=None),
        device="cpu",
    )

    assert optimised_logits.shape == (4, 3)
    assert optimised_summary["n_reference_groups"] == 2
    assert anchor.gate_groups.tolist() == [0, 0, 1]
    assert anchor.gate_group_labels == ["R1", "R2"]


def test_optimised_anchor_uses_cluster_sum_frequency_order_and_fixed_learning_rate():
    prepared = make_prepared()
    labels = np.array(["B", "A", "A", "B"], dtype=object)
    cluster_names, cluster_expression, cluster_density = aggregate_reference_clusters(
        prepared.reference,
        train_genes=["g1", "g2"],
        cluster_labels=labels,
    )
    _logits, summary = compute_optimised_anchor(
        prepared.reference,
        prepared.spatial,
        train_genes=prepared.train_genes,
        anchor_config=AnchorResidualConfig(coarse_optimisation_epochs=1, coarse_optimisation_runs=1),
        optimisation_config=OptimisationConfig(num_epochs=1, learning_rate=0.5, print_each=None),
        device="cpu",
    )

    assert cluster_names == ["B", "A"]
    np.testing.assert_allclose(cluster_expression[0], prepared.reference.expression[[0, 3]][:, [0, 1]].sum(axis=0))
    np.testing.assert_allclose(cluster_density, np.array([0.5, 0.5]))
    assert summary["coarse_optimisation_learning_rate"] == 0.1


def test_twin_panel_helpers_compute_composition_loss_and_fusion():
    prepared = make_prepared()
    split = build_twin_panel_split(
        prepared.reference,
        train_genes=prepared.train_genes,
        config=TwinPanelConfig(lambda_agree=0.05),
    )
    assignment_a = torch.tensor(
        [
            [0.9, 0.1, 0.0],
            [0.8, 0.1, 0.1],
            [0.1, 0.2, 0.7],
            [0.0, 0.2, 0.8],
        ],
        dtype=torch.float32,
    )
    assignment_b = torch.tensor(
        [
            [0.7, 0.2, 0.1],
            [0.7, 0.1, 0.2],
            [0.2, 0.2, 0.6],
            [0.1, 0.3, 0.6],
        ],
        dtype=torch.float32,
    )
    codes = torch.tensor(split.cell_type_codes, dtype=torch.float32)
    composition = composition_from_assignment(assignment_a, codes)
    loss, reg, disagreement = twin_composition_loss(
        assignment_a,
        assignment_b,
        cell_type_codes=codes,
        lambda_agree=0.05,
    )
    fused = fuse_branch_assignments(assignment_a, assignment_b)

    assert composition.shape == (3, 2)
    assert float(loss.item()) >= 0
    assert float(reg.item()) >= 0
    assert disagreement >= 0
    np.testing.assert_allclose(fused.numpy(), ((assignment_a + assignment_b) * 0.5).numpy())


def test_arc_impute_source_has_no_runtime_mapping_dependency():
    root = Path(__file__).resolve().parents[1] / "src" / "arc_impute"
    source = "\n".join(path.read_text() for path in root.rglob("*.py"))

    assert "import tangram" not in source
    assert "from tangram" not in source
