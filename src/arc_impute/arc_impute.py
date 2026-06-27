from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from arc_impute.config import ArcImputeConfig
from arc_impute.data import PreparedArcImputeInput, prepare_datasets_from_anndata
from arc_impute.modules.anchor_residual import build_anchor_residual
from arc_impute.modules.reference_curation_routing import build_assignment_routing
from arc_impute.modules.twin_panel_consistency import build_twin_panel_split
from arc_impute.optimiser import ArcOptimiser
from arc_impute.results import deconvolve_assignment, impute_expression, validate_assignment

log = logging.getLogger(__name__)


class ArcImpute:
    """Curate a reference, optimise its spatial mapping and derive outputs."""

    def __init__(self, config: ArcImputeConfig):
        if config is None:
            raise ValueError("ArcImpute(...): config is required.")
        self.config = config
        self.prepared_: PreparedArcImputeInput | None = None
        self.assignment_: np.ndarray | None = None
        self.training_history_: dict | None = None
        self.diagnostics_: dict | None = None

    def prepare(
        self,
        reference,
        spatial,
        *,
        train_genes: list[str],
    ) -> ArcImpute:
        """Validate the inputs and construct the curated reference X_bar."""

        self.prepared_ = None
        self._clear_fitted_state()
        prepared = prepare_datasets_from_anndata(
            reference=reference,
            spatial=spatial,
            config=self.config,
            train_genes=train_genes,
        )
        self.prepared_ = prepared
        log.info(
            "Prepared ARC-Impute inputs: cells=%d spots=%d train_genes=%d reference_genes=%d",
            len(prepared.reference.cell_ids),
            len(prepared.spatial.spot_ids),
            len(prepared.train_genes),
            prepared.curated_reference.n_vars,
        )
        return self

    def fit(self) -> ArcImpute:
        """Fit the anchored cell-to-spot mapping M."""

        if self.prepared_ is None:
            raise ValueError("ArcImpute.fit(...): call prepare() before fit().")
        self._clear_fitted_state()
        prepared = self.prepared_

        anchor = build_anchor_residual(
            prepared.reference,
            prepared.spatial,
            train_genes=prepared.train_genes,
            anchor_config=self.config.anchor_residual,
            optimisation_config=self.config.optimisation,
            device=self.config.device,
        )
        routing = build_assignment_routing(
            prepared.reference,
            prepared.spatial,
            train_genes=prepared.train_genes,
            config=self.config.assignment_routing,
        )
        twin_panel = None
        if self.config.twin_panel.lambda_agree > 0:
            twin_panel = build_twin_panel_split(
                prepared.reference,
                train_genes=prepared.train_genes,
                config=self.config.twin_panel,
            )
        optimiser = ArcOptimiser(
            reference=prepared.reference,
            spatial=prepared.spatial,
            train_genes=prepared.train_genes,
            anchor=anchor,
            routing=routing,
            twin_panel=twin_panel,
            anchor_config=self.config.anchor_residual,
            twin_config=self.config.twin_panel,
            optimisation_config=self.config.optimisation,
            device=self.config.device,
            seed=self.config.seed,
        )
        output = optimiser.train()
        diagnostics = {
            "reference_curation": prepared.diagnostics.get("reference_curation", {}),
            "assignment_routing": routing.summary,
            "anchor_residual": anchor.summary,
            "twin_panel": {} if twin_panel is None else twin_panel.summary,
            "training_history": output.training_history,
        }
        diagnostics["reference_curation_data"] = prepared.diagnostics.get("reference_curation_data")
        diagnostics["selected_cells"] = prepared.diagnostics.get("selected_cells")
        diagnostics["optimisation"] = output.diagnostics

        self.assignment_ = validate_assignment(
            output.assignment,
            n_reference_cells=len(prepared.reference.cell_ids),
            n_spatial_spots=len(prepared.spatial.spot_ids),
        )
        self.training_history_ = output.training_history
        self.diagnostics_ = diagnostics
        log.info(
            "Fitted ARC-Impute: cells=%d spots=%d train_genes=%d",
            len(prepared.reference.cell_ids),
            len(prepared.spatial.spot_ids),
            len(prepared.train_genes),
        )
        return self

    def impute(self, *, genes: list[str]) -> pd.DataFrame:
        """Project selected reference genes through the fitted mapping."""

        prepared, assignment = self._require_fitted("impute")
        return impute_expression(
            assignment=assignment,
            curated_reference=prepared.curated_reference,
            spot_ids=prepared.spatial.spot_ids,
            genes=genes,
        )

    def deconvolve(self) -> pd.DataFrame:
        """Return spot-wise cell-type compositions from the fitted mapping."""

        prepared, assignment = self._require_fitted("deconvolve")
        return deconvolve_assignment(
            assignment=assignment,
            cell_types=prepared.reference.cell_types,
            spot_ids=prepared.spatial.spot_ids,
        )

    def _clear_fitted_state(self) -> None:
        self.assignment_ = None
        self.training_history_ = None
        self.diagnostics_ = None

    def _require_fitted(self, method_name: str) -> tuple[PreparedArcImputeInput, np.ndarray]:
        if self.prepared_ is None or self.assignment_ is None:
            raise ValueError(f"ArcImpute.{method_name}(...): call prepare() and fit() first.")
        return self.prepared_, self.assignment_
