from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from anndata import AnnData
import numpy as np
import pandas as pd

from arc_impute.config import ArcImputeConfig
from arc_impute._utils import dense_float32_matrix


@dataclass(slots=True)
class ReferenceDataset:
    expression: np.ndarray
    genes: list[str]
    cell_ids: list[str]
    cell_types: np.ndarray
    cell_metadata: pd.DataFrame

    def __post_init__(self) -> None:
        self.expression = np.asarray(self.expression, dtype=np.float32)
        if self.expression.ndim != 2:
            raise ValueError("ReferenceDataset(...): expression must be 2D.")
        if self.expression.shape != (len(self.cell_ids), len(self.genes)):
            raise ValueError(
                "ReferenceDataset(...): expression shape must match cell_ids x genes."
            )
        if len(set(self.cell_ids)) != len(self.cell_ids):
            raise ValueError("ReferenceDataset(...): cell_ids must be unique.")
        if len(set(self.genes)) != len(self.genes):
            raise ValueError("ReferenceDataset(...): genes must be unique.")
        if len(self.cell_types) != len(self.cell_ids):
            raise ValueError("ReferenceDataset(...): cell_types must match cell_ids.")
        if len(self.cell_metadata.index) != len(self.cell_ids):
            raise ValueError("ReferenceDataset(...): cell_metadata must match cell_ids.")
        if not np.isfinite(self.expression).all():
            raise ValueError("ReferenceDataset(...): expression must contain only finite values.")

    def gene_indices(self, genes: list[str]) -> list[int]:
        gene_to_index = {gene: index for index, gene in enumerate(self.genes)}
        missing = [gene for gene in genes if gene not in gene_to_index]
        if missing:
            raise ValueError(f"ReferenceDataset.gene_indices(...): unknown gene {missing[0]!r}.")
        return [gene_to_index[gene] for gene in genes]


@dataclass(slots=True)
class SpatialDataset:
    expression: np.ndarray
    genes: list[str]
    spot_ids: list[str]
    regions: np.ndarray
    density: np.ndarray
    spot_metadata: pd.DataFrame
    coordinates: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.expression = np.asarray(self.expression, dtype=np.float32)
        self.density = np.asarray(self.density, dtype=np.float32)
        if self.expression.ndim != 2:
            raise ValueError("SpatialDataset(...): expression must be 2D.")
        if self.expression.shape != (len(self.spot_ids), len(self.genes)):
            raise ValueError("SpatialDataset(...): expression shape must match spot_ids x genes.")
        if len(set(self.spot_ids)) != len(self.spot_ids):
            raise ValueError("SpatialDataset(...): spot_ids must be unique.")
        if len(set(self.genes)) != len(self.genes):
            raise ValueError("SpatialDataset(...): genes must be unique.")
        if len(self.regions) != len(self.spot_ids):
            raise ValueError("SpatialDataset(...): regions must match spot_ids.")
        if self.density.shape != (len(self.spot_ids),):
            raise ValueError("SpatialDataset(...): density must match spot_ids.")
        if len(self.spot_metadata.index) != len(self.spot_ids):
            raise ValueError("SpatialDataset(...): spot_metadata must match spot_ids.")
        if not np.isfinite(self.expression).all():
            raise ValueError("SpatialDataset(...): expression must contain only finite values.")
        if not np.isfinite(self.density).all():
            raise ValueError("SpatialDataset(...): density must contain only finite values.")
        if (self.density < 0).any():
            raise ValueError("SpatialDataset(...): density must be non-negative.")
        if self.density.sum() <= 0:
            raise ValueError("SpatialDataset(...): density must have positive total mass.")
        self.density = self.density / self.density.sum()
        if self.coordinates is not None:
            self.coordinates = np.asarray(self.coordinates, dtype=np.float32)
            if self.coordinates.ndim != 2 or self.coordinates.shape[0] != len(self.spot_ids):
                raise ValueError(
                    "SpatialDataset(...): coordinates must be 2D with one row per spot."
                )

    def gene_indices(self, genes: list[str]) -> list[int]:
        gene_to_index = {gene: index for index, gene in enumerate(self.genes)}
        missing = [gene for gene in genes if gene not in gene_to_index]
        if missing:
            raise ValueError(f"SpatialDataset.gene_indices(...): unknown gene {missing[0]!r}.")
        return [gene_to_index[gene] for gene in genes]


@dataclass(slots=True)
class PreparedArcImputeInput:
    reference: ReferenceDataset
    spatial: SpatialDataset
    curated_reference: AnnData
    train_genes: list[str]
    diagnostics: dict[str, Any] = field(default_factory=dict)


def prepare_datasets_from_anndata(
    *,
    reference,
    spatial,
    config: ArcImputeConfig,
    train_genes: list[str],
) -> PreparedArcImputeInput:
    from arc_impute.modules.reference_curation_routing import curate_reference

    train_genes = list(train_genes)
    _validate_gene_request(train_genes=train_genes)

    _validate_axis_names(reference, dataset_label="reference")
    _validate_axis_names(spatial, dataset_label="spatial")
    _validate_obs_column(reference, key=config.cell_type_key, dataset_label="reference")
    _validate_obs_column(spatial, key=config.region_key, dataset_label="spatial")
    _validate_genes_present(reference, genes=train_genes, dataset_label="reference")
    _validate_genes_present(spatial, genes=train_genes, dataset_label="spatial")

    curation = curate_reference(
        reference,
        spatial,
        genes=train_genes,
        label_key=config.cell_type_key,
        config=config.reference_curation,
    )
    curated_reference = curation.reference

    reference_expression = dense_float32_matrix(
        curated_reference[:, train_genes].X,
        dataset_label="reference.X",
    )
    spatial_expression = _dense_float32_matrix(spatial[:, train_genes].X, dataset_label="spatial")
    density = _normalised_spot_density(spatial_expression)
    coordinates = _optional_spatial_coordinates(spatial, config=config)

    reference_dataset = ReferenceDataset(
        expression=reference_expression,
        genes=list(train_genes),
        cell_ids=[str(value) for value in curated_reference.obs_names.tolist()],
        cell_types=curated_reference.obs[config.cell_type_key].astype(str).to_numpy(dtype=object),
        cell_metadata=curated_reference.obs.copy(),
    )
    spatial_dataset = SpatialDataset(
        expression=spatial_expression,
        genes=list(train_genes),
        spot_ids=[str(value) for value in spatial.obs_names.tolist()],
        regions=spatial.obs[config.region_key].astype(str).to_numpy(dtype=object),
        density=density,
        spot_metadata=spatial.obs.copy(),
        coordinates=coordinates,
    )
    return PreparedArcImputeInput(
        reference=reference_dataset,
        spatial=spatial_dataset,
        curated_reference=curated_reference,
        train_genes=train_genes,
        diagnostics={
            "reference_curation": curation.summary,
            "reference_curation_data": curation.curation_data,
            "selected_cells": curation.selected_cells,
        },
    )


def _validate_gene_request(*, train_genes: list[str]) -> None:
    if len(train_genes) == 0:
        raise ValueError("ArcImpute.prepare(...): train_genes must be non-empty.")
    if len(set(train_genes)) != len(train_genes):
        raise ValueError("ArcImpute.prepare(...): train_genes must be unique.")


def _validate_axis_names(adata, *, dataset_label: str) -> None:
    if not adata.obs_names.is_unique:
        raise ValueError(f"{dataset_label}.obs_names must be unique.")
    if not adata.var_names.is_unique:
        raise ValueError(f"{dataset_label}.var_names must be unique.")


def _validate_obs_column(adata, *, key: str, dataset_label: str) -> None:
    if key not in adata.obs.columns:
        raise ValueError(f"{dataset_label}.obs is missing required column {key!r}.")
    if adata.obs[key].isna().any():
        raise ValueError(f"{dataset_label}.obs[{key!r}] must not contain missing values.")


def _validate_genes_present(adata, *, genes: list[str], dataset_label: str) -> None:
    available = set(adata.var_names.tolist())
    missing = [gene for gene in genes if gene not in available]
    if missing:
        raise ValueError(f"{dataset_label}.var_names is missing required gene {missing[0]!r}.")


def _dense_float32_matrix(matrix, *, dataset_label: str) -> np.ndarray:
    return dense_float32_matrix(matrix, dataset_label=f"{dataset_label}.X")


def _normalised_spot_density(spatial_expression: np.ndarray) -> np.ndarray:
    density = spatial_expression.sum(axis=1).astype(np.float32)
    if not np.isfinite(density).all():
        raise ValueError("spatial density must contain only finite values.")
    if density.sum() <= 0:
        raise ValueError("spatial density must have positive total mass.")
    return density / density.sum()


def _optional_spatial_coordinates(spatial, *, config: ArcImputeConfig) -> np.ndarray | None:
    key = config.spatial_coordinates_key
    if key is None or key not in spatial.obsm:
        return None
    coordinates = np.asarray(spatial.obsm[key], dtype=np.float32)
    if coordinates.ndim != 2 or coordinates.shape[0] != spatial.n_obs:
        raise ValueError(
            f"spatial.obsm[{key!r}] must be 2D with one row per spatial spot."
        )
    if not np.isfinite(coordinates).all():
        raise ValueError(f"spatial.obsm[{key!r}] must contain only finite values.")
    return coordinates
