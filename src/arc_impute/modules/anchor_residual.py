from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import cosine_similarity, softmax

from arc_impute._utils import normalise, probabilities_to_logits, unique_in_order, validate_labels
from arc_impute.config import AnchorResidualConfig, OptimisationConfig
from arc_impute.data import ReferenceDataset, SpatialDataset


log = logging.getLogger(__name__)

# The coarse P optimiser adapts Tangram 1.0.4's softmax reconstruction and
# density objective to cell-type and spatial-region profiles (BSD-3-Clause).
# https://github.com/broadinstitute/Tangram/blob/d537ff936a1a26445a0f196c714d100d10e3e5ea/tangram/mapping_optimizer.py


@dataclass(slots=True)
class AnchorResidual:
    anchor_logits: np.ndarray
    gate_groups: np.ndarray | None
    gate_group_labels: list[str]
    summary: dict[str, object]


def build_anchor_residual(
    reference: ReferenceDataset,
    spatial: SpatialDataset,
    *,
    train_genes: list[str],
    anchor_config: AnchorResidualConfig,
    optimisation_config: OptimisationConfig,
    device: str,
) -> AnchorResidual:
    """Fit and expand the coarse mapping P into the persistent anchor A."""

    anchor_logits, coarse_summary = compute_optimised_anchor(
        reference,
        spatial,
        train_genes=train_genes,
        anchor_config=anchor_config,
        optimisation_config=optimisation_config,
        device=device,
    )

    if anchor_config.anchor_strength_scalar != 1.0:
        anchor_logits = anchor_logits * np.float32(anchor_config.anchor_strength_scalar)

    gate_groups, gate_group_labels = resolve_gate_groups(
        spatial,
        gate_mode=anchor_config.gate_mode,
    )
    summary = {
        **coarse_summary,
        "anchor_strength_scalar": float(anchor_config.anchor_strength_scalar),
        "gate_mode": anchor_config.gate_mode,
        "gate_group_labels": gate_group_labels,
        "n_gate_groups": len(gate_group_labels),
    }
    return AnchorResidual(
        anchor_logits=anchor_logits.astype(np.float32, copy=False),
        gate_groups=gate_groups,
        gate_group_labels=gate_group_labels,
        summary=summary,
    )


def compute_optimised_anchor(
    reference: ReferenceDataset,
    spatial: SpatialDataset,
    *,
    train_genes: list[str],
    anchor_config: AnchorResidualConfig,
    optimisation_config: OptimisationConfig,
    device: str,
) -> tuple[np.ndarray, dict[str, object]]:
    """Estimate P between cell-type and spatial-region profiles."""

    cluster_labels = pd.Series(reference.cell_types).astype(str).to_numpy()
    spatial_regions = pd.Series(spatial.regions).astype(str).to_numpy()
    validate_labels("reference.cell_types", cluster_labels)
    validate_labels("spatial.regions", spatial_regions)

    cluster_names, cluster_expression, cluster_density = aggregate_reference_clusters(
        reference,
        train_genes=train_genes,
        cluster_labels=cluster_labels,
    )
    region_names, region_expression, region_density = aggregate_spatial_regions(
        spatial,
        train_genes=train_genes,
        density_prior=optimisation_config.density_prior,
    )

    P_runs = []
    for seed_offset in range(anchor_config.coarse_optimisation_runs):
        optimiser = CoarseAssignmentOptimiser(
            X_bar=cluster_expression,
            Y=region_expression,
            target_density=region_density,
            source_density=cluster_density,
            lambda_d=1.0,
            reconstruction_weight=1.0,
            device=torch.device(device),
            random_state=anchor_config.coarse_optimisation_seed + seed_offset,
        )
        P_run, _history = optimiser.train(
            num_epochs=anchor_config.coarse_optimisation_epochs,
            learning_rate=0.1,
        )
        P_runs.append(P_run)

    P = np.median(np.stack(P_runs, axis=0), axis=0)
    A_region = expand_cluster_probabilities_to_cells(
        P,
        cluster_names=cluster_names,
        cluster_labels=cluster_labels,
    )
    spot_weights = None
    if anchor_config.density_based_spot_expansion:
        spot_weights = resolve_spot_expansion_weights(spatial, density_prior=optimisation_config.density_prior)
    A = expand_region_probabilities_to_spots(
        region_probabilities=A_region,
        region_labels=region_names,
        spot_regions=spatial_regions,
        spot_weights=spot_weights,
    )
    A = A / A.sum(axis=1, keepdims=True)
    return probabilities_to_logits(A), {
        "n_reference_groups": len(cluster_names),
        "n_spatial_regions": len(region_names),
        "coarse_optimisation_learning_rate": 0.1,
        "coarse_optimisation_epochs": int(anchor_config.coarse_optimisation_epochs),
        "coarse_optimisation_runs": int(anchor_config.coarse_optimisation_runs),
    }


class CoarseAssignmentOptimiser:
    """Optimise the coarse cell-type-to-region mapping P."""

    def __init__(
        self,
        *,
        X_bar: np.ndarray,
        Y: np.ndarray,
        target_density: np.ndarray,
        source_density: np.ndarray,
        lambda_d: float,
        reconstruction_weight: float,
        device: torch.device,
        random_state: int,
    ):
        self.device = device
        self.random_state = random_state
        self.X_bar = torch.tensor(X_bar, device=device, dtype=torch.float32)
        self.Y = torch.tensor(Y, device=device, dtype=torch.float32)
        self.target_density = torch.tensor(target_density, device=device, dtype=torch.float32)
        self.source_density = torch.tensor(source_density, device=device, dtype=torch.float32)
        self.lambda_d = float(lambda_d)
        self.reconstruction_weight = float(reconstruction_weight)
        self._density_criterion = torch.nn.KLDivLoss(reduction="sum")
        if random_state:
            np.random.seed(seed=random_state)
        self.P_logits = torch.tensor(
            np.random.normal(0, 1, (X_bar.shape[0], Y.shape[0])).astype(np.float32),
            device=device,
            requires_grad=True,
            dtype=torch.float32,
        )

    def _loss_fn(self):
        """Combine the coarse reconstruction and spatial-density terms."""

        P = softmax(self.P_logits, dim=1)
        Y_hat = torch.matmul(P.t(), self.X_bar)
        reconstruction_term = self.reconstruction_weight * cosine_similarity(Y_hat, self.Y, dim=0).mean()
        predicted_density_log = torch.log(self.source_density @ P)
        density_term = self.lambda_d * self._density_criterion(predicted_density_log, self.target_density)
        total_loss = -reconstruction_term + density_term
        return total_loss, float((reconstruction_term / self.reconstruction_weight).detach().cpu().item()), float(
            (density_term / self.lambda_d).detach().cpu().item()
        )

    def train(self, *, num_epochs: int, learning_rate: float) -> tuple[np.ndarray, dict[str, list[float]]]:
        """Optimise P with Adam and return its row-wise softmax."""

        if self.random_state:
            torch.manual_seed(seed=self.random_state)
        optimiser = torch.optim.Adam([self.P_logits], lr=learning_rate)
        history = {"total_loss": [], "main_loss": [], "kl_reg": []}
        for _epoch in range(num_epochs):
            loss, main_loss, kl_reg = self._loss_fn()
            history["total_loss"].append(float(loss.detach().cpu().item()))
            history["main_loss"].append(main_loss)
            history["kl_reg"].append(kl_reg)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        with torch.no_grad():
            return softmax(self.P_logits, dim=1).detach().cpu().numpy().astype(np.float32), history


def aggregate_reference_clusters(
    reference: ReferenceDataset,
    *,
    train_genes: list[str],
    cluster_labels: np.ndarray,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Aggregate the curated reference X_bar by annotated cell type."""

    gene_indices = reference.gene_indices(train_genes)
    expression = reference.expression[:, gene_indices].astype(np.float64, copy=False)
    cluster_names = pd.Series(cluster_labels).value_counts(normalize=True).index.astype(str).tolist()
    cluster_expression = np.zeros((len(cluster_names), len(train_genes)), dtype=np.float32)
    cluster_counts = np.zeros(len(cluster_names), dtype=np.float64)
    for index, cluster in enumerate(cluster_names):
        mask = cluster_labels == cluster
        cluster_expression[index] = expression[mask].sum(axis=0)
        cluster_counts[index] = mask.sum()
    return cluster_names, cluster_expression, normalise(cluster_counts, label="reference cluster density")


def aggregate_spatial_regions(
    spatial: SpatialDataset,
    *,
    train_genes: list[str],
    density_prior,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Aggregate the spatial target Y by annotated region."""

    gene_indices = spatial.gene_indices(train_genes)
    expression = spatial.expression[:, gene_indices].astype(np.float64, copy=False)
    regions = pd.Series(spatial.regions).astype(str).to_numpy()
    region_names = unique_in_order(regions.tolist())
    region_expression = np.zeros((len(region_names), len(train_genes)), dtype=np.float32)
    region_density = np.zeros(len(region_names), dtype=np.float64)
    spot_density = resolve_spot_expansion_weights(spatial, density_prior=density_prior)
    for index, region in enumerate(region_names):
        mask = regions == region
        region_expression[index] = expression[mask].mean(axis=0)
        region_density[index] = spot_density[mask].sum()
    return region_names, region_expression, normalise(region_density, label="spatial region density")


def resolve_spot_expansion_weights(spatial: SpatialDataset, *, density_prior) -> np.ndarray:
    if density_prior == "rna_count_based":
        return np.asarray(spatial.density, dtype=np.float64)
    if density_prior == "uniform" or density_prior is None:
        return np.ones(len(spatial.spot_ids), dtype=np.float64) / len(spatial.spot_ids)
    density = np.asarray(density_prior, dtype=np.float64)
    if density.shape != (len(spatial.spot_ids),):
        raise ValueError("resolve_spot_expansion_weights(...): custom density_prior must have one value per spot.")
    if not np.isfinite(density).all() or (density < 0).any() or not np.isclose(float(density.sum()), 1.0):
        raise ValueError(
            "resolve_spot_expansion_weights(...): custom density_prior must be finite, non-negative, and sum to 1."
        )
    return density


def expand_cluster_probabilities_to_cells(
    cluster_probabilities: np.ndarray,
    *,
    cluster_names: list[str],
    cluster_labels: np.ndarray,
) -> np.ndarray:
    """Expand P from cell types to individual reference cells."""

    cluster_lookup = {cluster: index for index, cluster in enumerate(cluster_names)}
    expanded = np.zeros((len(cluster_labels), cluster_probabilities.shape[1]), dtype=np.float64)
    for cell_index, cluster in enumerate(cluster_labels):
        expanded[cell_index] = cluster_probabilities[cluster_lookup[str(cluster)]]
    return expanded


def expand_region_probabilities_to_spots(
    *,
    region_probabilities: np.ndarray,
    region_labels: list[str],
    spot_regions: np.ndarray,
    spot_weights: np.ndarray | None,
) -> np.ndarray:
    """Expand regional probabilities into the cell-to-spot anchor A."""

    region_to_col = {region: index for index, region in enumerate(region_labels)}
    weights = None if spot_weights is None else np.asarray(spot_weights, dtype=np.float64)
    spot_probabilities = np.zeros((region_probabilities.shape[0], len(spot_regions)), dtype=np.float64)
    for region in unique_in_order(spot_regions.tolist()):
        if region not in region_to_col:
            raise ValueError(f"expand_region_probabilities_to_spots(...): region {region!r} missing from region map.")
        spot_idx = np.where(spot_regions == region)[0]
        region_column = region_probabilities[:, region_to_col[region]].reshape(-1, 1)
        if weights is None:
            spot_probabilities[:, spot_idx] = region_column / len(spot_idx)
        else:
            region_weights = weights[spot_idx]
            region_weight_sum = region_weights.sum()
            if region_weight_sum <= 0:
                raise ValueError(
                    f"expand_region_probabilities_to_spots(...): spot weights sum to zero for region {region!r}."
                )
            spot_probabilities[:, spot_idx] = region_column * (region_weights / region_weight_sum).reshape(1, -1)
    return spot_probabilities


def resolve_gate_groups(
    spatial: SpatialDataset,
    *,
    gate_mode: str,
) -> tuple[np.ndarray | None, list[str]]:
    if gate_mode == "scalar":
        return None, ["global"]
    if gate_mode == "spot":
        return np.arange(len(spatial.spot_ids), dtype=np.int64), [str(value) for value in spatial.spot_ids]

    region_labels = pd.Series(spatial.regions).astype(str).to_numpy()
    region_to_index = {}
    gate_groups = []
    for label in region_labels:
        if label not in region_to_index:
            region_to_index[label] = len(region_to_index)
        gate_groups.append(region_to_index[label])
    return np.asarray(gate_groups, dtype=np.int64), list(region_to_index.keys())
