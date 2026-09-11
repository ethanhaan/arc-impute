from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ReferenceCurationConfig:
    similarity: str = "pearson"
    representation: str = "centroid"
    tau: float = 0.1
    gamma: float = 1.0
    source_prior_quota_blend: float = 0.10
    budget_fraction: float = 0.5
    q_min: int = 32
    max_programs: int = 8

    def __post_init__(self) -> None:
        if self.similarity not in {"pearson", "cosine"}:
            raise ValueError("ReferenceCurationConfig(...): similarity must be one of {'pearson', 'cosine'}.")
        if self.representation not in {"centroid", "hclust_gene_program"}:
            raise ValueError(
                "ReferenceCurationConfig(...): representation must be one of {'centroid', 'hclust_gene_program'}."
            )
        if self.tau <= 0:
            raise ValueError("ReferenceCurationConfig(...): tau must be positive.")
        if self.gamma < 0:
            raise ValueError("ReferenceCurationConfig(...): gamma must be non-negative.")
        if self.source_prior_quota_blend < 0 or self.source_prior_quota_blend > 1:
            raise ValueError("ReferenceCurationConfig(...): source_prior_quota_blend must be in [0, 1].")
        if self.budget_fraction <= 0:
            raise ValueError("ReferenceCurationConfig(...): budget_fraction must be positive.")
        if self.q_min <= 0:
            raise ValueError("ReferenceCurationConfig(...): q_min must be positive.")
        if self.max_programs <= 0:
            raise ValueError("ReferenceCurationConfig(...): max_programs must be positive.")


@dataclass(slots=True)
class AnchorResidualConfig:
    anchor_strength_scalar: float = 0.5
    gate_mode: str = "region"
    gate_init_logit: float = -2.0
    anchor_kl_lambda: float = 0.03
    gate_l1_lambda: float = 0.01
    coarse_optimisation_seed: int = 0
    coarse_optimisation_epochs: int = 500
    coarse_optimisation_runs: int = 5
    density_based_spot_expansion: bool = False

    def __post_init__(self) -> None:
        if self.anchor_strength_scalar < 0:
            raise ValueError("AnchorResidualConfig(...): anchor_strength_scalar must be >= 0.")
        if self.gate_mode not in {"scalar", "spot", "region"}:
            raise ValueError("AnchorResidualConfig(...): gate_mode must be one of {'scalar', 'spot', 'region'}.")
        if self.anchor_kl_lambda < 0:
            raise ValueError("AnchorResidualConfig(...): anchor_kl_lambda must be >= 0.")
        if self.gate_l1_lambda < 0:
            raise ValueError("AnchorResidualConfig(...): gate_l1_lambda must be >= 0.")
        if self.coarse_optimisation_seed < 0:
            raise ValueError("AnchorResidualConfig(...): coarse_optimisation_seed must be >= 0.")
        if self.coarse_optimisation_epochs < 0:
            raise ValueError("AnchorResidualConfig(...): coarse_optimisation_epochs must be >= 0.")
        if self.coarse_optimisation_runs <= 0:
            raise ValueError("AnchorResidualConfig(...): coarse_optimisation_runs must be positive.")


@dataclass(slots=True)
class AssignmentRoutingConfig:
    similarity: str = "pearson"
    representation: str = "centroid"
    tau: float = 0.1
    gamma: float = 1.0
    temp_cell: float = 0.1
    eps: float = 0.01
    bias_clip_min: float = -3.0
    bias_clip_max: float = 3.0
    beta_s1: float = 0.5
    beta_d2: float = 0.15
    quota_min: int = 1
    max_programs: int = 8

    def __post_init__(self) -> None:
        if self.similarity not in {"pearson", "cosine"}:
            raise ValueError("AssignmentRoutingConfig(...): similarity must be one of {'pearson', 'cosine'}.")
        if self.representation not in {"centroid", "hclust_gene_program"}:
            raise ValueError(
                "AssignmentRoutingConfig(...): representation must be one of {'centroid', 'hclust_gene_program'}."
            )
        if self.tau <= 0:
            raise ValueError("AssignmentRoutingConfig(...): tau must be positive.")
        if self.gamma < 0:
            raise ValueError("AssignmentRoutingConfig(...): gamma must be non-negative.")
        if self.temp_cell <= 0:
            raise ValueError("AssignmentRoutingConfig(...): temp_cell must be positive.")
        if self.eps <= 0:
            raise ValueError("AssignmentRoutingConfig(...): eps must be positive.")
        if self.bias_clip_min >= self.bias_clip_max:
            raise ValueError("AssignmentRoutingConfig(...): bias_clip_min must be < bias_clip_max.")
        if self.quota_min <= 0:
            raise ValueError("AssignmentRoutingConfig(...): quota_min must be positive.")
        if self.max_programs <= 0:
            raise ValueError("AssignmentRoutingConfig(...): max_programs must be positive.")


@dataclass(slots=True)
class TwinPanelConfig:
    lambda_agree: float = 0.05
    informative_fraction: float = 0.5
    shared_core_fraction: float = 0.2
    fusion_mode: str = "mean"

    def __post_init__(self) -> None:
        if self.lambda_agree < 0:
            raise ValueError("TwinPanelConfig(...): lambda_agree must be >= 0.")
        if self.informative_fraction <= 0 or self.informative_fraction >= 1:
            raise ValueError("TwinPanelConfig(...): informative_fraction must be strictly between 0 and 1.")
        if self.shared_core_fraction < 0 or self.shared_core_fraction >= 1:
            raise ValueError("TwinPanelConfig(...): shared_core_fraction must be in [0, 1).")
        if self.fusion_mode != "mean":
            raise ValueError("TwinPanelConfig(...): only fusion_mode='mean' is supported.")


@dataclass(slots=True)
class OptimisationConfig:
    density_prior: str | list[float] | None = "rna_count_based"
    num_epochs: int = 500
    learning_rate: float = 0.1
    lambda_g1: float = 1.0
    lambda_d: float = 1.0
    lambda_g2: float = 0.0
    lambda_r: float = 0.0
    lambda_l1: float = 0.0
    lambda_l2: float = 0.0
    print_each: int | None = 100

    def __post_init__(self) -> None:
        if isinstance(self.density_prior, str) and self.density_prior not in {"rna_count_based", "uniform"}:
            raise ValueError("OptimisationConfig(...): density_prior must be 'rna_count_based', 'uniform', a list, or None.")
        if self.num_epochs < 0:
            raise ValueError("OptimisationConfig(...): num_epochs must be >= 0.")
        if self.learning_rate <= 0:
            raise ValueError("OptimisationConfig(...): learning_rate must be positive.")
        if self.lambda_g1 == 0:
            raise ValueError("OptimisationConfig(...): lambda_g1 cannot be 0.")
        if self.lambda_d < 0:
            raise ValueError("OptimisationConfig(...): lambda_d must be >= 0.")
        if self.lambda_g2 < 0 or self.lambda_r < 0 or self.lambda_l1 < 0 or self.lambda_l2 < 0:
            raise ValueError("OptimisationConfig(...): regularisation lambdas must be >= 0.")
        if self.print_each is not None and self.print_each <= 0:
            raise ValueError("OptimisationConfig(...): print_each must be positive or None.")


@dataclass(slots=True)
class ArcImputeConfig:
    cell_type_key: str
    region_key: str
    device: str = "cpu"
    seed: int = 0
    spatial_coordinates_key: str | None = "spatial"
    reference_curation: ReferenceCurationConfig = field(default_factory=ReferenceCurationConfig)
    anchor_residual: AnchorResidualConfig = field(default_factory=AnchorResidualConfig)
    assignment_routing: AssignmentRoutingConfig = field(default_factory=AssignmentRoutingConfig)
    twin_panel: TwinPanelConfig = field(default_factory=TwinPanelConfig)
    optimisation: OptimisationConfig = field(default_factory=OptimisationConfig)

    def __post_init__(self) -> None:
        if not self.cell_type_key:
            raise ValueError("ArcImputeConfig(...): cell_type_key is required.")
        if not self.region_key:
            raise ValueError("ArcImputeConfig(...): region_key is required.")
        if not self.device:
            raise ValueError("ArcImputeConfig(...): device is required.")
        if self.seed < 0:
            raise ValueError("ArcImputeConfig(...): seed must be >= 0.")
