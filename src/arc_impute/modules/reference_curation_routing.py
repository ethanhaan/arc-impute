from __future__ import annotations

from dataclasses import dataclass
import warnings

from anndata import AnnData
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage

from arc_impute._utils import (
    centre_rows,
    dense_float32_matrix,
    l2_normalise_rows,
    normalise,
    softmax_rows,
    unique_in_order,
)
from arc_impute.config import AssignmentRoutingConfig, ReferenceCurationConfig
from arc_impute.data import ReferenceDataset, SpatialDataset


@dataclass(slots=True)
class ReferenceCurationResult:
    reference: AnnData
    curation_data: dict[str, object]
    selected_cells: pd.DataFrame
    summary: dict[str, object]


def curate_reference(
    reference: AnnData,
    spatial: AnnData,
    *,
    genes: list[str],
    label_key: str,
    config: ReferenceCurationConfig,
) -> ReferenceCurationResult:
    """Resample X into the target-supported curated reference X_bar."""

    budget = int(round(reference.n_obs * float(config.budget_fraction)))
    curation_data = compute_curation_data(
        reference,
        spatial,
        genes=genes,
        label_key=label_key,
        similarity=config.similarity,
        representation=config.representation,
        tau=config.tau,
        gamma=config.gamma,
        budget=budget,
        q_min=config.q_min,
        max_programs=config.max_programs,
    )
    curation_data = apply_source_prior_quota_blend(
        curation_data,
        source_prior_weight=config.source_prior_quota_blend,
        budget=budget,
        q_min=config.q_min,
    )
    curated_reference, selected_cells = build_centroid_curated_reference(
        reference,
        quotas=curation_data["quotas"],
        label_key=label_key,
        genes=genes,
    )
    summary = {
        "raw_cells": int(reference.n_obs),
        "curated_cells": int(curated_reference.n_obs),
        "label_key": label_key,
        "budget": int(budget),
        "quota_sum": int(curation_data["quotas"].sum()),
        "raw_label_counts": reference.obs[label_key].astype(str).value_counts(sort=False).to_dict(),
        "curated_label_counts": curated_reference.obs[label_key].astype(str).value_counts(sort=False).to_dict(),
        "selection_method": "centroid_proximity",
    }
    return ReferenceCurationResult(
        reference=curated_reference,
        curation_data=curation_data,
        selected_cells=selected_cells,
        summary=summary,
    )


def expression_matrix(adata: AnnData, genes: list[str], *, dataset_label: str) -> np.ndarray:
    missing = [gene for gene in genes if gene not in adata.var_names]
    if missing:
        raise ValueError(f"{dataset_label}.var_names is missing required gene {missing[0]!r}.")
    return dense_float32_matrix(adata[:, genes].X, dataset_label=f"{dataset_label}.X").astype(float, copy=False)


def ordered_reference_labels(reference: AnnData, *, label_key: str) -> list[str]:
    if label_key not in reference.obs.columns:
        raise ValueError(f"reference.obs is missing required label column {label_key!r}.")
    labels = reference.obs[label_key].astype(str)
    return labels.drop_duplicates().tolist()


def build_feature_bundle(
    reference: AnnData,
    spatial: AnnData,
    *,
    genes: list[str],
    label_key: str,
    representation: str,
    max_programs: int,
) -> dict[str, object]:
    if len(genes) == 0:
        raise ValueError("build_feature_bundle(...): genes must be non-empty.")

    reference_expression = expression_matrix(reference, genes, dataset_label="reference")
    spatial_expression = expression_matrix(spatial, genes, dataset_label="spatial")
    labels = ordered_reference_labels(reference, label_key=label_key)
    reference_labels = reference.obs[label_key].astype(str)

    if representation == "centroid":
        reference_features = reference_expression
        spatial_features = spatial_expression
        feature_names = list(genes)
        gene_programs = pd.DataFrame(
            {
                "gene": genes,
                "program": np.arange(1, len(genes) + 1, dtype=int),
                "feature_name": feature_names,
            }
        )
    elif representation == "hclust_gene_program":
        assignments = hclust_gene_programs(reference_expression, genes=genes, max_programs=max_programs)
        reference_features, feature_names = program_mean_features(reference_expression, assignments)
        spatial_features, _ = program_mean_features(spatial_expression, assignments)
        gene_programs = pd.DataFrame(
            {
                "gene": genes,
                "program": assignments,
                "feature_name": [feature_names[program - 1] for program in assignments],
            }
        )
    else:
        raise ValueError(f"Unknown representation: {representation!r}")

    prototype_rows = []
    for label in labels:
        mask = reference_labels == label
        prototype_rows.append(reference_features[mask.to_numpy()].mean(axis=0))

    label_prototypes = pd.DataFrame(
        prototype_rows,
        index=pd.Index(labels, name="label"),
        columns=feature_names,
    )
    return {
        "labels": labels,
        "spatial_features": spatial_features,
        "reference_features": reference_features,
        "label_prototypes": label_prototypes,
        "gene_programs": gene_programs,
    }


def hclust_gene_programs(reference_expression: np.ndarray, *, genes: list[str], max_programs: int) -> np.ndarray:
    if max_programs <= 0:
        raise ValueError("hclust_gene_programs(...): max_programs must be positive.")
    if len(genes) != reference_expression.shape[1]:
        raise ValueError("hclust_gene_programs(...): gene count must match expression columns.")
    if len(genes) <= max_programs:
        return np.arange(1, len(genes) + 1, dtype=int)

    clusters = fcluster(
        linkage(reference_expression.T, method="average", metric="euclidean"),
        t=max_programs,
        criterion="maxclust",
    )
    remapped: dict[int, int] = {}
    next_program = 1
    stable_assignments = []
    for cluster in clusters:
        cluster_id = int(cluster)
        if cluster_id not in remapped:
            remapped[cluster_id] = next_program
            next_program += 1
        stable_assignments.append(remapped[cluster_id])
    return np.asarray(stable_assignments, dtype=int)


def program_mean_features(expression: np.ndarray, assignments: np.ndarray) -> tuple[np.ndarray, list[str]]:
    programs = sorted(np.unique(assignments).tolist())
    feature_columns = []
    feature_names = []
    for program in programs:
        feature_columns.append(expression[:, assignments == program].mean(axis=1))
        feature_names.append(f"program_{program:02d}")
    return np.column_stack(feature_columns), feature_names


def similarity_scores(
    spatial_features: np.ndarray,
    label_prototypes: pd.DataFrame,
    *,
    similarity: str,
) -> pd.DataFrame:
    prototype_features = label_prototypes.to_numpy(dtype=float)
    if similarity == "cosine":
        spot_features = l2_normalise_rows(spatial_features)
        prototype_features = l2_normalise_rows(prototype_features)
    elif similarity == "pearson":
        spot_features = l2_normalise_rows(centre_rows(spatial_features), allow_zero_rows=True)
        prototype_features = l2_normalise_rows(centre_rows(prototype_features), allow_zero_rows=True)
    else:
        raise ValueError(f"Unknown similarity: {similarity!r}")

    scores = spot_features @ prototype_features.T
    return pd.DataFrame(scores, columns=label_prototypes.index.astype(str))


def softmax_probabilities(scores: pd.DataFrame, *, tau: float) -> pd.DataFrame:
    if tau <= 0:
        raise ValueError("softmax_probabilities(...): tau must be positive.")
    probabilities = softmax_rows(scores.to_numpy(dtype=float) / tau)
    return pd.DataFrame(probabilities, index=scores.index.copy(), columns=scores.columns.copy())


def compute_curation_data(
    reference: AnnData,
    spatial: AnnData,
    *,
    genes: list[str],
    label_key: str,
    similarity: str,
    representation: str,
    tau: float,
    gamma: float,
    budget: int,
    q_min: int,
    max_programs: int,
) -> dict[str, object]:
    """Estimate local demand, global demand and curated quotas."""

    bundle = build_feature_bundle(
        reference,
        spatial,
        genes=genes,
        label_key=label_key,
        representation=representation,
        max_programs=max_programs,
    )
    scores = similarity_scores(
        bundle["spatial_features"],
        bundle["label_prototypes"],
        similarity=similarity,
    )
    scores.index = spatial.obs_names.copy()
    probabilities = softmax_probabilities(scores, tau=tau)
    demand = probabilities.sum(axis=0).rename("raw_demand")
    source_counts = reference.obs[label_key].astype(str).value_counts(sort=False).reindex(demand.index)
    quotas = build_global_quotas(
        demand=demand,
        source_counts=source_counts,
        budget=budget,
        q_min=q_min,
        gamma=gamma,
    )
    return {
        **bundle,
        "scores": scores,
        "probabilities": probabilities,
        "demand": demand,
        "quota_table": quotas,
        "quotas": quotas["repaired_quota"].astype(int),
    }


def build_global_quotas(
    *,
    demand: pd.Series,
    source_counts: pd.Series,
    budget: int,
    q_min: int,
    gamma: float,
) -> pd.DataFrame:
    """Convert target demand into a budget-constrained cell-type allocation."""

    if budget <= 0:
        raise ValueError("build_global_quotas(...): budget must be positive.")
    if q_min <= 0:
        raise ValueError("build_global_quotas(...): q_min must be positive.")
    if gamma < 0:
        raise ValueError("build_global_quotas(...): gamma must be non-negative.")

    labels = pd.Index(demand.index.astype(str), name="label")
    if len(labels) * q_min > budget:
        raise ValueError(
            "build_global_quotas(...): n_labels * q_min exceeds budget "
            f"({len(labels)} * {q_min} > {budget})."
        )

    aligned_source_counts = source_counts.reindex(labels)
    if aligned_source_counts.isna().any() or (aligned_source_counts <= 0).any():
        raise ValueError("build_global_quotas(...): every demanded label must have positive source_count.")
    aligned_source_counts = aligned_source_counts.astype(int)

    demand_values = demand.reindex(labels).to_numpy(dtype=float)
    if np.any(demand_values < 0):
        raise ValueError("build_global_quotas(...): demand values must be non-negative.")
    weights = np.ones_like(demand_values, dtype=float) if gamma == 0 else np.power(demand_values, gamma)
    if weights.sum() <= 0:
        raise ValueError("build_global_quotas(...): powered demand weights must have positive total.")

    qtilde = weights / weights.sum()
    target_float = budget * qtilde
    initial_quota = np.maximum(q_min, np.rint(target_float).astype(int))
    repaired_quota = repair_quotas(
        labels=labels.tolist(),
        initial_quota=initial_quota,
        target_float=target_float,
        budget=budget,
        q_min=q_min,
    )
    return pd.DataFrame(
        {
            "raw_demand": demand_values,
            "qtilde": qtilde,
            "target_float": target_float,
            "initial_quota": initial_quota,
            "repaired_quota": repaired_quota,
            "source_count": aligned_source_counts.to_numpy(dtype=int),
            "oversampling_factor": repaired_quota / aligned_source_counts.to_numpy(dtype=float),
        },
        index=labels,
    )


def repair_quotas(
    *,
    labels: list[str],
    initial_quota: np.ndarray,
    target_float: np.ndarray,
    budget: int,
    q_min: int,
) -> np.ndarray:
    repaired = np.asarray(initial_quota, dtype=int).copy()

    while int(repaired.sum()) > budget:
        excess = repaired - target_float
        candidates = [idx for idx, quota in enumerate(repaired) if quota > q_min]
        if not candidates:
            raise ValueError("repair_quotas(...): cannot reduce quotas to budget without crossing q_min.")
        candidates.sort(key=lambda idx: (-float(excess[idx]), labels[idx]))
        repaired[candidates[0]] -= 1

    fractional = target_float - np.floor(target_float)
    increment_order = list(range(len(labels)))
    increment_order.sort(key=lambda idx: (-float(fractional[idx]), labels[idx]))
    cursor = 0
    while int(repaired.sum()) < budget:
        repaired[increment_order[cursor % len(increment_order)]] += 1
        cursor += 1
    return repaired


def apply_source_prior_quota_blend(
    curation_data: dict[str, object],
    *,
    source_prior_weight: float,
    budget: int,
    q_min: int,
) -> dict[str, object]:
    if source_prior_weight < 0 or source_prior_weight > 1:
        raise ValueError("apply_source_prior_quota_blend(...): source_prior_weight must be in [0, 1].")
    if source_prior_weight == 0:
        return curation_data

    quota_table = curation_data["quota_table"].copy()
    labels = pd.Index(quota_table.index.astype(str), name="label")
    demand_qtilde = quota_table["qtilde"].to_numpy(dtype=float)
    source_counts = quota_table["source_count"].to_numpy(dtype=float)
    source_qtilde = source_counts / source_counts.sum()
    blended_qtilde = (1.0 - float(source_prior_weight)) * demand_qtilde + float(source_prior_weight) * source_qtilde
    blended_qtilde = blended_qtilde / blended_qtilde.sum()
    target_float = int(budget) * blended_qtilde
    initial_quota = np.maximum(int(q_min), np.rint(target_float).astype(int))
    repaired_quota = repair_quotas(
        labels=labels.tolist(),
        initial_quota=initial_quota,
        target_float=target_float,
        budget=int(budget),
        q_min=int(q_min),
    )

    quota_table["demand_only_qtilde"] = demand_qtilde
    quota_table["source_prior_qtilde"] = source_qtilde
    quota_table["source_prior_blend_weight"] = float(source_prior_weight)
    quota_table["qtilde"] = blended_qtilde
    quota_table["target_float"] = target_float
    quota_table["initial_quota"] = initial_quota
    quota_table["repaired_quota"] = repaired_quota
    quota_table["oversampling_factor"] = repaired_quota / source_counts

    blended_curation_data = dict(curation_data)
    blended_curation_data["quota_table"] = quota_table
    blended_curation_data["quotas"] = quota_table["repaired_quota"].astype(int)
    blended_curation_data["source_prior_quota_blend"] = float(source_prior_weight)
    return blended_curation_data


def build_centroid_curated_reference(
    reference: AnnData,
    *,
    quotas: pd.Series,
    label_key: str,
    genes: list[str],
    family: str = "arc_reference_curation_centroid_quota",
) -> tuple[AnnData, pd.DataFrame]:
    """Select centroid-near cells, repeating them when quotas exceed supply."""

    if label_key not in reference.obs.columns:
        raise ValueError(f"reference.obs is missing required label column {label_key!r}.")
    if len(quotas) == 0:
        raise ValueError("build_centroid_curated_reference(...): quotas must be non-empty.")
    if len(genes) == 0:
        raise ValueError("build_centroid_curated_reference(...): genes must be non-empty.")

    source_labels = reference.obs[label_key].astype(str).to_numpy()
    reference_expression = expression_matrix(reference, genes, dataset_label="reference")
    selected_positions: list[int] = []
    selected_rows: list[dict[str, object]] = []
    copy_counts: dict[int, int] = {}

    for label, quota_value in quotas.items():
        quota = int(quota_value)
        if quota <= 0:
            raise ValueError(f"build_centroid_curated_reference(...): quota must be positive for label {label!r}.")
        source_positions = np.flatnonzero(source_labels == str(label))
        if len(source_positions) == 0:
            raise ValueError(f"build_centroid_curated_reference(...): no source cells for label {label!r}.")

        ranked_positions, ranked_distances = centroid_ranked_label_positions(
            source_positions=source_positions,
            reference_expression=reference_expression,
        )
        rank_by_position = {int(position): rank for rank, position in enumerate(ranked_positions)}
        distance_by_position = {
            int(position): float(distance)
            for position, distance in zip(ranked_positions, ranked_distances)
        }

        for source_position in centroid_label_selection(ranked_positions, quota=quota):
            copy_index = copy_counts.get(int(source_position), 0)
            copy_counts[int(source_position)] = copy_index + 1
            source_cell_id = str(reference.obs_names[source_position])
            curated_obs_name = f"{source_cell_id}__src{int(source_position)}__copy{copy_index}"
            selected_positions.append(int(source_position))
            selected_rows.append(
                {
                    "selected_order": len(selected_rows),
                    "reference_curation_label": str(label),
                    "source_cell_id": source_cell_id,
                    "source_position": int(source_position),
                    "reference_curation_copy_index": int(copy_index),
                    "reference_curation_selection_method": "centroid_proximity",
                    "reference_curation_centroid_rank": int(rank_by_position[int(source_position)]),
                    "reference_curation_centroid_distance": distance_by_position[int(source_position)],
                    "curated_obs_name": curated_obs_name,
                }
            )

    selected_cell_table = pd.DataFrame(selected_rows)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Observation names are not unique.", category=UserWarning)
        curated_reference = reference[selected_positions, :].copy()
    curated_reference.obs["source_cell_id"] = selected_cell_table["source_cell_id"].to_numpy(dtype=object)
    curated_reference.obs["reference_curation_family"] = family
    curated_reference.obs["reference_curation_label"] = selected_cell_table["reference_curation_label"].to_numpy(dtype=object)
    curated_reference.obs["reference_curation_copy_index"] = selected_cell_table[
        "reference_curation_copy_index"
    ].to_numpy(dtype=int)
    curated_reference.obs["reference_curation_selection_method"] = selected_cell_table[
        "reference_curation_selection_method"
    ].to_numpy(dtype=object)
    curated_reference.obs["reference_curation_centroid_rank"] = selected_cell_table[
        "reference_curation_centroid_rank"
    ].to_numpy(dtype=int)
    curated_reference.obs["reference_curation_centroid_distance"] = selected_cell_table[
        "reference_curation_centroid_distance"
    ].to_numpy(dtype=float)
    curated_reference.obs_names = selected_cell_table["curated_obs_name"].astype(str).to_numpy()

    if not curated_reference.obs_names.is_unique:
        raise ValueError("build_centroid_curated_reference(...): curated obs_names are not unique.")
    return curated_reference, selected_cell_table


def centroid_ranked_label_positions(
    *,
    source_positions: np.ndarray,
    reference_expression: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    label_expression = reference_expression[source_positions]
    centroid = label_expression.mean(axis=0)
    distances = np.linalg.norm(label_expression - centroid, axis=1)
    order = np.argsort(distances, kind="stable")
    return source_positions[order].astype(int), distances[order].astype(float)


def centroid_label_selection(ranked_positions: np.ndarray, *, quota: int) -> list[int]:
    n_source = len(ranked_positions)
    if quota <= n_source:
        return ranked_positions[:quota].astype(int).tolist()

    full_cycles, remainder = divmod(quota, n_source)
    selected = ranked_positions.astype(int).tolist() * full_cycles
    if remainder:
        selected.extend(ranked_positions[:remainder].astype(int).tolist())
    return selected


@dataclass(slots=True)
class AssignmentRouting:
    spot_celltype_logit_bias: np.ndarray
    spot_cell_logit_bias: np.ndarray
    cell_label_indices: np.ndarray
    labels: list[str]
    summary: dict[str, object]
    demand_probabilities: pd.DataFrame
    demand_scores: pd.DataFrame
    quota_table: pd.DataFrame


def build_assignment_routing(
    reference: ReferenceDataset,
    spatial: SpatialDataset,
    *,
    train_genes: list[str],
    config: AssignmentRoutingConfig,
) -> AssignmentRouting:
    """Construct the fixed cell-to-spot routing bias b."""

    if len(train_genes) == 0:
        raise ValueError("build_assignment_routing(...): train_genes must be non-empty.")

    curation_data = compute_dataset_curation_data(
        reference,
        spatial,
        genes=train_genes,
        similarity=config.similarity,
        representation=config.representation,
        tau=config.tau,
        gamma=config.gamma,
        budget=len(reference.cell_ids),
        q_min=config.quota_min,
        max_programs=config.max_programs,
    )
    probabilities = curation_data["probabilities"].copy()
    probabilities.index = probabilities.index.astype(str)
    probabilities.columns = probabilities.columns.astype(str)

    spot_names = [str(value) for value in spatial.spot_ids]
    cell_names = [str(value) for value in reference.cell_ids]
    cell_labels = pd.Series(reference.cell_types).astype(str).to_numpy()
    labels = unique_in_order(cell_labels.tolist())

    missing_probability_labels = [label for label in labels if label not in probabilities.columns]
    if missing_probability_labels:
        raise ValueError(
            "build_assignment_routing(...): "
            f"demand probabilities missing label {missing_probability_labels[0]!r}."
        )
    probabilities = probabilities.loc[spot_names, labels]

    label_to_index = {label: index for index, label in enumerate(labels)}
    cell_label_indices = np.asarray([label_to_index[label] for label in cell_labels], dtype=np.int64)
    spot_label_probability = probabilities.to_numpy(dtype=np.float32)
    cell_spot_probability = spot_label_probability[:, cell_label_indices].T

    gene_indices = reference.gene_indices(train_genes)
    spatial_gene_indices = spatial.gene_indices(train_genes)
    reference_expression = reference.expression[:, gene_indices]
    spatial_expression = spatial.expression[:, spatial_gene_indices]

    cell_spot_cosine = (
        l2_normalise_rows(reference_expression, allow_zero_rows=True)
        @ l2_normalise_rows(spatial_expression, allow_zero_rows=True).T
    )
    cell_fit = _sigmoid(cell_spot_cosine / config.temp_cell).astype(np.float32, copy=False)
    d2_allowance = (cell_spot_probability * cell_fit).astype(np.float32, copy=False)

    label_counts = np.asarray([np.sum(cell_labels == label) for label in labels], dtype=np.float64)
    reference_frequency = normalise(label_counts, label="reference_frequency")
    cell_baseline_frequency = reference_frequency[cell_label_indices].astype(np.float32, copy=False)
    d2_raw_bias = np.log((d2_allowance + config.eps) / (cell_baseline_frequency[:, None] + config.eps))
    d2_clipped_bias = np.clip(d2_raw_bias, config.bias_clip_min, config.bias_clip_max)
    d2_scaled_bias = (config.beta_d2 * d2_clipped_bias).astype(np.float32, copy=False)

    s1_raw_bias = np.log((spot_label_probability + config.eps) / (reference_frequency[None, :] + config.eps))
    s1_clipped_bias = np.clip(s1_raw_bias, config.bias_clip_min, config.bias_clip_max)
    s1_scaled_bias = (config.beta_s1 * s1_clipped_bias).astype(np.float32, copy=False)
    spot_celltype_logit_bias = s1_scaled_bias.T.astype(np.float32, copy=False)

    demand_scores = curation_data["scores"].copy()
    demand_scores.index = demand_scores.index.astype(str)
    demand_scores.columns = demand_scores.columns.astype(str)

    summary = {
        "n_labels": len(labels),
        "n_cells": len(cell_names),
        "n_spots": len(spot_names),
        "allowance_min": float(d2_allowance.min()),
        "allowance_mean": float(d2_allowance.mean()),
        "allowance_max": float(d2_allowance.max()),
        "s1_bias_min": float(s1_scaled_bias.min()),
        "s1_bias_mean": float(s1_scaled_bias.mean()),
        "s1_bias_max": float(s1_scaled_bias.max()),
        "d2_bias_min": float(d2_scaled_bias.min()),
        "d2_bias_mean": float(d2_scaled_bias.mean()),
        "d2_bias_max": float(d2_scaled_bias.max()),
        "bias_min": float(min(s1_scaled_bias.min(), d2_scaled_bias.min())),
        "bias_mean": float(s1_scaled_bias.mean() + d2_scaled_bias.mean()),
        "bias_max": float(max(s1_scaled_bias.max(), d2_scaled_bias.max())),
        "cell_fit_min": float(cell_fit.min()),
        "cell_fit_mean": float(cell_fit.mean()),
        "cell_fit_max": float(cell_fit.max()),
    }
    return AssignmentRouting(
        spot_celltype_logit_bias=spot_celltype_logit_bias,
        spot_cell_logit_bias=d2_scaled_bias,
        cell_label_indices=cell_label_indices,
        labels=labels,
        summary=summary,
        demand_probabilities=probabilities,
        demand_scores=demand_scores.loc[spot_names, labels],
        quota_table=curation_data["quota_table"].copy(),
    )


def compute_dataset_curation_data(
    reference: ReferenceDataset,
    spatial: SpatialDataset,
    *,
    genes: list[str],
    similarity: str,
    representation: str,
    tau: float,
    gamma: float,
    budget: int,
    q_min: int,
    max_programs: int,
) -> dict[str, object]:
    reference_adata, spatial_adata = _datasets_to_anndata(reference, spatial)
    bundle = build_feature_bundle(
        reference_adata,
        spatial_adata,
        genes=genes,
        label_key="__arc_label__",
        representation=representation,
        max_programs=max_programs,
    )
    scores = similarity_scores(
        bundle["spatial_features"],
        bundle["label_prototypes"],
        similarity=similarity,
    )
    scores.index = pd.Index(spatial.spot_ids)
    probabilities = softmax_probabilities(scores, tau=tau)
    demand = probabilities.sum(axis=0).rename("raw_demand")
    source_counts = pd.Series(reference.cell_types).astype(str).value_counts(sort=False).reindex(demand.index)
    quotas = build_global_quotas(
        demand=demand,
        source_counts=source_counts,
        budget=budget,
        q_min=q_min,
        gamma=gamma,
    )
    return {
        **bundle,
        "scores": scores,
        "probabilities": probabilities,
        "demand": demand,
        "quota_table": quotas,
        "quotas": quotas["repaired_quota"].astype(int),
    }


def _datasets_to_anndata(reference: ReferenceDataset, spatial: SpatialDataset):
    reference_obs = reference.cell_metadata.copy()
    reference_obs["__arc_label__"] = pd.Series(reference.cell_types, index=reference_obs.index).astype(str).to_numpy()
    spatial_obs = spatial.spot_metadata.copy()
    reference_adata = AnnData(
        X=reference.expression,
        obs=reference_obs,
        var=pd.DataFrame(index=reference.genes),
    )
    spatial_adata = AnnData(
        X=spatial.expression,
        obs=spatial_obs,
        var=pd.DataFrame(index=spatial.genes),
    )
    return reference_adata, spatial_adata


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))
