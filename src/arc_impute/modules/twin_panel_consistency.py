from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
import torch

from arc_impute.config import TwinPanelConfig
from arc_impute.data import ReferenceDataset


@dataclass(slots=True)
class TwinPanelSplit:
    panel_a_genes: list[str]
    panel_b_genes: list[str]
    shared_core_genes: list[str]
    informative_exclusive_genes: list[str]
    complementary_exclusive_genes: list[str]
    gene_scores: dict[str, float]
    cell_type_labels: list[str]
    cell_type_codes: np.ndarray
    summary: dict[str, object]


def build_twin_panel_split(
    reference: ReferenceDataset,
    *,
    train_genes: list[str],
    config: TwinPanelConfig,
) -> TwinPanelSplit:
    """Form overlapping gene panels G_1 and G_2 around an informative core."""

    labels = pd.Series(reference.cell_types).astype(str)
    if len(train_genes) < 2:
        raise ValueError("build_twin_panel_split(...): need at least 2 training genes to form twin panels.")
    expression = pd.DataFrame(
        reference.expression[:, reference.gene_indices(train_genes)],
        index=pd.Index(reference.cell_ids),
        columns=pd.Index(train_genes),
    )
    group_means = expression.groupby(labels.to_numpy(), sort=False).mean()
    if group_means.shape[0] < 2:
        raise ValueError("build_twin_panel_split(...): need at least 2 non-empty groups.")

    gene_scores = group_means.var(axis=0, ddof=0).sort_values(ascending=False)
    n_shared_core = int(math.ceil(len(train_genes) * config.shared_core_fraction))
    n_shared_core = min(len(train_genes) - 2, max(1, n_shared_core))

    shared_core_genes = gene_scores.index[:n_shared_core].tolist()
    remaining_genes = gene_scores.index[n_shared_core:].tolist()
    if len(remaining_genes) < 2:
        raise ValueError("build_twin_panel_split(...): need at least 2 non-core genes to form exclusive panels.")

    n_informative_exclusive = int(math.ceil(len(remaining_genes) * config.informative_fraction))
    n_informative_exclusive = min(len(remaining_genes) - 1, max(1, n_informative_exclusive))
    informative_exclusive_genes = remaining_genes[:n_informative_exclusive]
    complementary_exclusive_genes = remaining_genes[n_informative_exclusive:]
    panel_a_genes = shared_core_genes + informative_exclusive_genes
    panel_b_genes = shared_core_genes + complementary_exclusive_genes
    cell_type_codes, cell_type_labels = encode_cell_types(reference)
    summary = {
        "panel_a_name": "informative_plus_core",
        "panel_b_name": "complementary_plus_core",
        "n_panel_a_genes": len(panel_a_genes),
        "n_panel_b_genes": len(panel_b_genes),
        "n_shared_core_genes": len(shared_core_genes),
        "shared_core_fraction": config.shared_core_fraction,
        "informative_fraction": config.informative_fraction,
        "fusion_mode": config.fusion_mode,
    }
    return TwinPanelSplit(
        panel_a_genes=panel_a_genes,
        panel_b_genes=panel_b_genes,
        shared_core_genes=shared_core_genes,
        informative_exclusive_genes=informative_exclusive_genes,
        complementary_exclusive_genes=complementary_exclusive_genes,
        gene_scores={gene: float(score) for gene, score in gene_scores.items()},
        cell_type_labels=cell_type_labels,
        cell_type_codes=cell_type_codes,
        summary=summary,
    )


def encode_cell_types(reference: ReferenceDataset) -> tuple[np.ndarray, list[str]]:
    labels = pd.Series(reference.cell_types).astype(str).tolist()
    unique_labels: list[str] = []
    label_to_index: dict[str, int] = {}
    for label in labels:
        if label not in label_to_index:
            label_to_index[label] = len(unique_labels)
            unique_labels.append(label)
    codes = np.zeros((len(reference.cell_ids), len(unique_labels)), dtype=np.float32)
    for row_index, label in enumerate(labels):
        codes[row_index, label_to_index[label]] = 1.0
    return codes, unique_labels


def composition_from_assignment(assignment: torch.Tensor, cell_type_codes: torch.Tensor) -> torch.Tensor:
    """Project M_k into row-normalised spot compositions Q_k."""

    Q_k = assignment.t() @ cell_type_codes
    denominator = Q_k.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return Q_k / denominator


def twin_composition_loss(
    assignment_a: torch.Tensor,
    assignment_b: torch.Tensor,
    *,
    cell_type_codes: torch.Tensor,
    lambda_agree: float,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Penalise disagreement between branch compositions Q_1 and Q_2."""

    Q_1 = composition_from_assignment(assignment_a, cell_type_codes)
    Q_2 = composition_from_assignment(assignment_b, cell_type_codes)
    loss = lambda_agree * ((Q_1 - Q_2) ** 2).mean()
    disagreement = float(torch.abs(Q_1 - Q_2).mean().detach().cpu().item())
    return loss, loss / lambda_agree, disagreement


def fuse_branch_assignments(assignment_a: torch.Tensor, assignment_b: torch.Tensor) -> torch.Tensor:
    """Fuse both branches as M = (M_1 + M_2) / 2."""

    return 0.5 * (assignment_a + assignment_b)
