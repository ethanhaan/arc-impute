from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

import numpy as np
import torch
from torch.nn.functional import cosine_similarity, softmax

from arc_impute.modules.anchor_residual import AnchorResidual
from arc_impute.modules.reference_curation_routing import AssignmentRouting
from arc_impute.config import AnchorResidualConfig, OptimisationConfig, TwinPanelConfig
from arc_impute.data import ReferenceDataset, SpatialDataset
from arc_impute.modules.twin_panel_consistency import (
    TwinPanelSplit,
    composition_from_assignment,
    fuse_branch_assignments,
    twin_composition_loss,
)


log = logging.getLogger(__name__)

# Core softmax mapping, reconstruction and density terms are adapted from
# Tangram 1.0.4 (BSD-3-Clause); ARC-Impute adds A, b, gated residuals and twin panels.
# https://github.com/broadinstitute/Tangram/blob/d537ff936a1a26445a0f196c714d100d10e3e5ea/tangram/mapping_optimizer.py


@dataclass(slots=True)
class OptimisationOutput:
    assignment: np.ndarray
    training_history: dict[str, Any]
    diagnostics: dict[str, Any]


class ArcOptimiser:
    """Optimise the anchored mappings M_k described in the Methods."""

    def __init__(
        self,
        *,
        reference: ReferenceDataset,
        spatial: SpatialDataset,
        train_genes: list[str],
        anchor: AnchorResidual,
        routing: AssignmentRouting,
        twin_panel: TwinPanelSplit | None,
        anchor_config: AnchorResidualConfig,
        twin_config: TwinPanelConfig,
        optimisation_config: OptimisationConfig,
        device: str,
        seed: int,
    ):
        self.reference = reference
        self.spatial = spatial
        self.train_genes = list(train_genes)
        self.anchor = anchor
        self.routing = routing
        self.twin_panel = twin_panel
        self.anchor_config = anchor_config
        self.twin_config = twin_config
        self.optimisation_config = optimisation_config
        self.device = torch.device(device)
        self.seed = seed

        reference_gene_idx = reference.gene_indices(self.train_genes)
        spatial_gene_idx = spatial.gene_indices(self.train_genes)
        self.X_bar = torch.tensor(
            reference.expression[:, reference_gene_idx],
            device=self.device,
            dtype=torch.float32,
        )
        self.Y = torch.tensor(spatial.expression[:, spatial_gene_idx], device=self.device, dtype=torch.float32)
        if not reference.expression[:, reference_gene_idx].any(axis=0).all():
            raise ValueError("ArcOptimiser(...): reference training genes contain all-zero columns.")
        if not spatial.expression[:, spatial_gene_idx].any(axis=0).all():
            raise ValueError("ArcOptimiser(...): spatial training genes contain all-zero columns.")

        expected_shape = (len(reference.cell_ids), len(spatial.spot_ids))
        if anchor.anchor_logits.shape != expected_shape:
            raise ValueError("ArcOptimiser(...): anchor logits must have shape cells x spots.")

        self.anchor_logits = torch.tensor(anchor.anchor_logits, device=self.device, dtype=torch.float32)
        self.spot_celltype_logit_bias = torch.tensor(
            routing.spot_celltype_logit_bias,
            device=self.device,
            dtype=torch.float32,
        )
        self.spot_cell_logit_bias = torch.tensor(
            routing.spot_cell_logit_bias,
            device=self.device,
            dtype=torch.float32,
        )
        self.cell_label_indices = torch.tensor(routing.cell_label_indices, device=self.device, dtype=torch.long)
        self.anchor_probs = softmax(self._base_logits(), dim=1).detach()

        self.lambda_d = self._resolve_lambda_d()
        self.density = self._resolve_density()
        self._density_criterion = torch.nn.KLDivLoss(reduction="sum")

        self.gate_group_index = None
        self.gate_logits = self._initial_gate_logits()
        if twin_config.lambda_agree > 0:
            self._validate_twin_supported_terms()
            if twin_panel is None:
                raise ValueError("ArcOptimiser(...): twin_panel is required when lambda_agree > 0.")
            self.branch_a_delta = torch.zeros_like(self.anchor_logits, requires_grad=True)
            self.branch_b_delta = torch.zeros_like(self.anchor_logits, requires_grad=True)
            self.branch_a_gate_logits = self.gate_logits.detach().clone().requires_grad_()
            self.branch_b_gate_logits = self.gate_logits.detach().clone().requires_grad_()
            self.twin_cell_type_codes = torch.tensor(
                twin_panel.cell_type_codes,
                device=self.device,
                dtype=torch.float32,
            )
            train_gene_to_index = {gene: index for index, gene in enumerate(self.train_genes)}
            self.twin_panel_a_idx = torch.tensor(
                [train_gene_to_index[gene] for gene in twin_panel.panel_a_genes],
                device=self.device,
                dtype=torch.long,
            )
            self.twin_panel_b_idx = torch.tensor(
                [train_gene_to_index[gene] for gene in twin_panel.panel_b_genes],
                device=self.device,
                dtype=torch.long,
            )
        else:
            self.Delta = torch.zeros_like(self.anchor_logits, requires_grad=True)

    def train(self) -> OptimisationOutput:
        """Optimise and return the final cell-to-spot mapping."""

        torch.manual_seed(seed=self.seed)
        if self.twin_config.lambda_agree > 0:
            M, history = self._train_twin()
        else:
            M, history = self._train_single()
        diagnostics = {
            "training_history": history,
            "assignment_shape": [int(M.shape[0]), int(M.shape[1])],
            "assignment_row_sum_min": float(M.sum(axis=1).min()),
            "assignment_row_sum_mean": float(M.sum(axis=1).mean()),
            "assignment_row_sum_max": float(M.sum(axis=1).max()),
        }
        return OptimisationOutput(
            assignment=M.astype(np.float32, copy=False),
            training_history=history,
            diagnostics=diagnostics,
        )

    def _base_logits(self) -> torch.Tensor:
        """Combine the fixed anchor A and routing bias b."""

        return (
            self.anchor_logits
            + self.spot_celltype_logit_bias[self.cell_label_indices, :]
            + self.spot_cell_logit_bias
        )

    def _initial_gate_logits(self) -> torch.Tensor:
        mode = self.anchor_config.gate_mode
        if mode == "scalar":
            return torch.tensor(
                [self.anchor_config.gate_init_logit],
                device=self.device,
                dtype=torch.float32,
                requires_grad=True,
            )
        if mode == "spot":
            gate_logits = torch.full(
                (len(self.spatial.spot_ids),),
                self.anchor_config.gate_init_logit,
                device=self.device,
                dtype=torch.float32,
            )
            gate_logits.requires_grad_()
            return gate_logits

        if self.anchor.gate_groups is None:
            raise ValueError("ArcOptimiser(...): region gate mode requires gate groups.")
        self.gate_group_index = torch.tensor(self.anchor.gate_groups, device=self.device, dtype=torch.long)
        gate_logits = torch.full(
            (len(self.anchor.gate_group_labels),),
            self.anchor_config.gate_init_logit,
            device=self.device,
            dtype=torch.float32,
        )
        gate_logits.requires_grad_()
        return gate_logits

    def _gate_values(self, gate_logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(gate_logits)

    def _mapping_probs_for(self, *, Delta_k: torch.Tensor, gate_logits: torch.Tensor) -> torch.Tensor:
        """Form M_k from A, b and the gated residual Delta_k."""

        gate_values = self._gate_values(gate_logits)
        if self.anchor_config.gate_mode == "scalar":
            return softmax(self._base_logits() + (gate_values[0] * Delta_k), dim=1)
        if self.anchor_config.gate_mode == "spot":
            return softmax(self._base_logits() + (gate_values.unsqueeze(0) * Delta_k), dim=1)
        return softmax(self._base_logits() + (gate_values[self.gate_group_index].unsqueeze(0) * Delta_k), dim=1)

    def _resolve_lambda_d(self) -> float:
        if self.optimisation_config.density_prior is not None and self.optimisation_config.lambda_d == 0:
            return 1.0
        if self.optimisation_config.lambda_d > 0 and self.optimisation_config.density_prior is None:
            raise ValueError("ArcOptimiser(...): density_prior is required when lambda_d > 0.")
        return float(self.optimisation_config.lambda_d)

    def _resolve_density(self) -> torch.Tensor | None:
        density_prior = self.optimisation_config.density_prior
        if density_prior is None:
            return None
        if density_prior == "rna_count_based":
            density = np.asarray(self.spatial.density, dtype=np.float32)
        elif density_prior == "uniform":
            density = np.ones(len(self.spatial.spot_ids), dtype=np.float32) / len(self.spatial.spot_ids)
        else:
            density = np.asarray(density_prior, dtype=np.float32)
            if density.shape != (len(self.spatial.spot_ids),):
                raise ValueError("ArcOptimiser(...): custom density_prior must have one value per spot.")
            if not np.isfinite(density).all() or (density < 0).any() or not np.isclose(float(density.sum()), 1.0):
                raise ValueError("ArcOptimiser(...): custom density_prior must be finite, non-negative, and sum to 1.")
        return torch.tensor(density, device=self.device, dtype=torch.float32)

    def _density_term(self, M_k: torch.Tensor) -> tuple[torch.Tensor, float]:
        """Match predicted and observed spot-density distributions."""

        if self.density is None:
            return torch.tensor(0.0, device=self.device), np.nan
        predicted_density_log = torch.log(M_k.sum(dim=0) / M_k.shape[0])
        density_term = self.lambda_d * self._density_criterion(predicted_density_log, self.density)
        return density_term, float((density_term / self.lambda_d).detach().cpu().item())

    def _anchor_kl(self, M_k: torch.Tensor) -> torch.Tensor:
        """Keep M_k close to its anchored routing distribution."""

        total = M_k.new_tensor(0.0)
        chunk_size = 128
        for start in range(0, M_k.shape[0], chunk_size):
            end = min(start + chunk_size, M_k.shape[0])
            mapping_chunk = M_k[start:end]
            anchor_chunk = self.anchor_probs[start:end]
            total = total + torch.sum(mapping_chunk * (torch.log(mapping_chunk) - torch.log(anchor_chunk)))
        return total / M_k.shape[0]

    def _gate_l1_term(self, gate_logits: torch.Tensor) -> tuple[torch.Tensor, float]:
        if self.anchor_config.gate_l1_lambda <= 0:
            return torch.tensor(0.0, device=self.device), np.nan
        term = self.anchor_config.gate_l1_lambda * self._gate_values(gate_logits).mean()
        return term, float((term / self.anchor_config.gate_l1_lambda).detach().cpu().item())

    def _train_single(self) -> tuple[np.ndarray, dict[str, Any]]:
        optimiser = torch.optim.Adam([self.Delta, self.gate_logits], lr=self.optimisation_config.learning_rate)
        keys = [
            "total_loss",
            "main_loss",
            "vg_reg",
            "kl_reg",
            "entropy_reg",
            "anchor_kl_reg",
            "gate_l1_reg",
            "gate_value",
            "gate_min_value",
            "gate_max_value",
        ]
        history = {key: [] for key in keys}
        for epoch in range(self.optimisation_config.num_epochs):
            run_loss = self._single_loss()
            loss = run_loss[0]
            history["total_loss"].append(float(loss.detach().cpu().item()))
            for index, key in enumerate(keys[1:], start=1):
                history[key].append(run_loss[index])
            if self.optimisation_config.print_each and epoch % self.optimisation_config.print_each == 0:
                log.info("ARC optimisation epoch=%d loss=%.4f", epoch, float(loss.detach().cpu().item()))
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        with torch.no_grad():
            M = self._mapping_probs_for(Delta_k=self.Delta, gate_logits=self.gate_logits).detach().cpu().numpy()
            history["final_gate_values"] = (
                self._gate_values(self.gate_logits).detach().cpu().numpy().astype(np.float32).tolist()
            )
        return M, history

    def _single_loss(self):
        """Evaluate the single-panel reconstruction and regularisation objective."""

        M = self._mapping_probs_for(Delta_k=self.Delta, gate_logits=self.gate_logits)
        Y_hat = torch.matmul(M.t(), self.X_bar)
        reconstruction_term = self.optimisation_config.lambda_g1 * cosine_similarity(Y_hat, self.Y, dim=0).mean()
        expression_term = reconstruction_term
        main_loss = float((reconstruction_term / self.optimisation_config.lambda_g1).detach().cpu().item())

        vg_reg = np.nan
        if self.optimisation_config.lambda_g2 > 0:
            vg_term = self.optimisation_config.lambda_g2 * cosine_similarity(Y_hat, self.Y, dim=1).mean()
            expression_term = expression_term + vg_term
            vg_reg = float((vg_term / self.optimisation_config.lambda_g2).detach().cpu().item())

        density_term, kl_reg = self._density_term(M)
        entropy_term = torch.tensor(0.0, device=self.device)
        entropy_reg = np.nan
        if self.optimisation_config.lambda_r > 0:
            entropy_term = self.optimisation_config.lambda_r * -(torch.log(M) * M).sum()
            entropy_reg = float((entropy_term / self.optimisation_config.lambda_r).detach().cpu().item())

        l1_term = torch.tensor(0.0, device=self.device)
        if self.optimisation_config.lambda_l1 > 0:
            l1_term = self.optimisation_config.lambda_l1 * self.Delta.abs().sum()
        l2_term = torch.tensor(0.0, device=self.device)
        if self.optimisation_config.lambda_l2 > 0:
            l2_term = self.optimisation_config.lambda_l2 * (self.Delta ** 2).sum()

        anchor_kl_term = torch.tensor(0.0, device=self.device)
        anchor_kl_reg = np.nan
        if self.anchor_config.anchor_kl_lambda > 0:
            anchor_kl_term = self.anchor_config.anchor_kl_lambda * self._anchor_kl(M)
            anchor_kl_reg = float((anchor_kl_term / self.anchor_config.anchor_kl_lambda).detach().cpu().item())

        gate_l1_term, gate_l1_reg = self._gate_l1_term(self.gate_logits)
        gate_values = self._gate_values(self.gate_logits).detach().cpu()
        total_loss = -expression_term + density_term + entropy_term + l1_term + l2_term + anchor_kl_term + gate_l1_term
        return (
            total_loss,
            main_loss,
            vg_reg,
            kl_reg,
            entropy_reg,
            anchor_kl_reg,
            gate_l1_reg,
            float(gate_values.mean().item()),
            float(gate_values.min().item()),
            float(gate_values.max().item()),
        )

    def _train_twin(self) -> tuple[np.ndarray, dict[str, Any]]:
        optimiser = torch.optim.Adam(
            [
                self.branch_a_delta,
                self.branch_b_delta,
                self.branch_a_gate_logits,
                self.branch_b_gate_logits,
            ],
            lr=self.optimisation_config.learning_rate,
        )
        keys = [
            "total_loss",
            "branch_a_main_loss",
            "branch_b_main_loss",
            "branch_a_density_reg",
            "branch_b_density_reg",
            "agreement_reg",
            "mean_panel_disagreement",
            "branch_a_anchor_kl_reg",
            "branch_b_anchor_kl_reg",
            "branch_a_gate_l1_reg",
            "branch_b_gate_l1_reg",
            "anchor_kl_reg",
            "gate_l1_reg",
            "gate_value",
            "gate_min_value",
            "gate_max_value",
        ]
        history = {key: [] for key in keys}
        for epoch in range(self.optimisation_config.num_epochs):
            run_loss = self._twin_loss()
            loss = run_loss[0]
            history["total_loss"].append(float(loss.detach().cpu().item()))
            for index, key in enumerate(keys[1:], start=1):
                history[key].append(run_loss[index])
            if self.optimisation_config.print_each and epoch % self.optimisation_config.print_each == 0:
                log.info("ARC twin optimisation epoch=%d loss=%.4f", epoch, float(loss.detach().cpu().item()))
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        with torch.no_grad():
            M_1, M_2 = self._twin_branch_probs()
            M = fuse_branch_assignments(M_1, M_2)
            final_branch_a_gate_values = (
                self._gate_values(self.branch_a_gate_logits).detach().cpu().numpy().astype(np.float32)
            )
            final_branch_b_gate_values = (
                self._gate_values(self.branch_b_gate_logits).detach().cpu().numpy().astype(np.float32)
            )
            history["final_gate_values"] = np.stack(
                [final_branch_a_gate_values, final_branch_b_gate_values],
                axis=0,
            ).tolist()
            Q_1 = composition_from_assignment(M, self.twin_cell_type_codes)
            Q_2 = composition_from_assignment(M_2, self.twin_cell_type_codes)
            history["final_spot_disagreement"] = (
                torch.abs(Q_1 - Q_2).mean(dim=1).detach().cpu().numpy().astype(np.float32).tolist()
            )
            history["final_mean_panel_disagreement"] = float(
                torch.abs(Q_1 - Q_2).mean().detach().cpu().item()
            )
            return M.detach().cpu().numpy(), history

    def _twin_branch_probs(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self._mapping_probs_for(Delta_k=self.branch_a_delta, gate_logits=self.branch_a_gate_logits),
            self._mapping_probs_for(Delta_k=self.branch_b_delta, gate_logits=self.branch_b_gate_logits),
        )

    def _twin_loss(self):
        """Evaluate both panel reconstructions and their composition agreement."""

        M_1, M_2 = self._twin_branch_probs()
        X_bar_1 = self.X_bar[:, self.twin_panel_a_idx]
        X_bar_2 = self.X_bar[:, self.twin_panel_b_idx]
        Y_1 = self.Y[:, self.twin_panel_a_idx]
        Y_2 = self.Y[:, self.twin_panel_b_idx]
        Y_hat_1 = torch.matmul(M_1.t(), X_bar_1)
        Y_hat_2 = torch.matmul(M_2.t(), X_bar_2)

        branch_a_main = self.optimisation_config.lambda_g1 * cosine_similarity(Y_hat_1, Y_1, dim=0).mean()
        branch_b_main = self.optimisation_config.lambda_g1 * cosine_similarity(Y_hat_2, Y_2, dim=0).mean()
        branch_a_main_reg = float((branch_a_main / self.optimisation_config.lambda_g1).detach().cpu().item())
        branch_b_main_reg = float((branch_b_main / self.optimisation_config.lambda_g1).detach().cpu().item())

        branch_a_density, branch_a_density_reg = self._density_term(M_1)
        branch_b_density, branch_b_density_reg = self._density_term(M_2)
        agreement_term, agreement_reg_tensor, mean_panel_disagreement = twin_composition_loss(
            M_1,
            M_2,
            cell_type_codes=self.twin_cell_type_codes,
            lambda_agree=self.twin_config.lambda_agree,
        )
        agreement_reg = float(agreement_reg_tensor.detach().cpu().item())

        branch_a_anchor_kl, branch_a_anchor_kl_reg = self._branch_anchor_kl(M_1)
        branch_b_anchor_kl, branch_b_anchor_kl_reg = self._branch_anchor_kl(M_2)
        branch_a_gate_l1, branch_a_gate_l1_reg = self._gate_l1_term(self.branch_a_gate_logits)
        branch_b_gate_l1, branch_b_gate_l1_reg = self._gate_l1_term(self.branch_b_gate_logits)

        branch_a_gate_values = self._gate_values(self.branch_a_gate_logits).detach().cpu()
        branch_b_gate_values = self._gate_values(self.branch_b_gate_logits).detach().cpu()
        stacked_gate_values = torch.cat([branch_a_gate_values.reshape(-1), branch_b_gate_values.reshape(-1)])
        anchor_kl_reg = float(np.nanmean([branch_a_anchor_kl_reg, branch_b_anchor_kl_reg]))
        gate_l1_reg = float(np.nanmean([branch_a_gate_l1_reg, branch_b_gate_l1_reg]))

        total_loss = (
            -branch_a_main
            - branch_b_main
            + branch_a_density
            + branch_b_density
            + agreement_term
            + branch_a_anchor_kl
            + branch_b_anchor_kl
            + branch_a_gate_l1
            + branch_b_gate_l1
        )
        return (
            total_loss,
            branch_a_main_reg,
            branch_b_main_reg,
            branch_a_density_reg,
            branch_b_density_reg,
            agreement_reg,
            mean_panel_disagreement,
            branch_a_anchor_kl_reg,
            branch_b_anchor_kl_reg,
            branch_a_gate_l1_reg,
            branch_b_gate_l1_reg,
            anchor_kl_reg,
            gate_l1_reg,
            float(stacked_gate_values.mean().item()),
            float(stacked_gate_values.min().item()),
            float(stacked_gate_values.max().item()),
        )

    def _branch_anchor_kl(self, M_k: torch.Tensor) -> tuple[torch.Tensor, float]:
        if self.anchor_config.anchor_kl_lambda <= 0:
            return torch.tensor(0.0, device=self.device), np.nan
        term = self.anchor_config.anchor_kl_lambda * self._anchor_kl(M_k)
        return term, float((term / self.anchor_config.anchor_kl_lambda).detach().cpu().item())

    def _validate_twin_supported_terms(self) -> None:
        unsupported_terms = {
            "lambda_g2": self.optimisation_config.lambda_g2,
            "lambda_r": self.optimisation_config.lambda_r,
            "lambda_l1": self.optimisation_config.lambda_l1,
            "lambda_l2": self.optimisation_config.lambda_l2,
        }
        active_unsupported = [key for key, value in unsupported_terms.items() if value > 0]
        if active_unsupported:
            raise ValueError(
                "ArcOptimiser(...): twin-panel consistency does not support these nonzero terms yet: "
                f"{active_unsupported}."
            )
