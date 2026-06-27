# Parameters

## `ArcImputeConfig`

| Parameter | Default | Meaning / accepted values |
|---|---:|---|
| `cell_type_key` | required | Reference `.obs` cell-type column. |
| `region_key` | required | Spatial `.obs` region column. |
| `device` | `"cpu"` | PyTorch device, e.g. `"cpu"`, `"cuda:0"`. |
| `seed` | `0` | Non-negative random seed. |
| `spatial_coordinates_key` | `"spatial"` | Spatial `.obsm` key; `None` disables loading. |
| `reference_curation` | defaults below | `ReferenceCurationConfig`. |
| `anchor_residual` | defaults below | `AnchorResidualConfig`. |
| `assignment_routing` | defaults below | `AssignmentRoutingConfig`. |
| `twin_panel` | defaults below | `TwinPanelConfig`. |
| `optimisation` | defaults below | `OptimisationConfig`. |

## `ReferenceCurationConfig`

| Parameter | Default | Meaning / accepted values |
|---|---:|---|
| `similarity` | `"pearson"` | `"pearson"` or `"cosine"`. |
| `representation` | `"centroid"` | `"centroid"` or `"hclust_gene_program"`. |
| `tau` | `0.1` | Positive softmax temperature. |
| `gamma` | `1.0` | Non-negative demand exponent. |
| `source_prior_quota_blend` | `0.10` | Source-prior weight in `[0, 1]`. |
| `budget_fraction` | `0.5` | Positive retained-reference fraction. |
| `q_min` | `32` | Positive minimum cells per type. |
| `max_programs` | `8` | Positive maximum gene programs. |

## `AnchorResidualConfig`

| Parameter | Default | Meaning / accepted values |
|---|---:|---|
| `anchor_strength_scalar` | `0.5` | Non-negative anchor-logit scale. |
| `gate_mode` | `"region"` | `"scalar"`, `"spot"`, or `"region"`. |
| `gate_init_logit` | `-2.0` | Initial gate logit. |
| `anchor_kl_lambda` | `0.03` | Non-negative anchor KL weight. |
| `gate_l1_lambda` | `0.01` | Non-negative gate penalty. |
| `coarse_optimisation_seed` | `0` | Non-negative coarse-map seed. |
| `coarse_optimisation_epochs` | `500` | Non-negative coarse-map epochs. |
| `coarse_optimisation_runs` | `5` | Positive coarse-map runs. |
| `density_based_spot_expansion` | `False` | Use density in anchor expansion. |

## `AssignmentRoutingConfig`

| Parameter | Default | Meaning / accepted values |
|---|---:|---|
| `similarity` | `"pearson"` | `"pearson"` or `"cosine"`. |
| `representation` | `"centroid"` | `"centroid"` or `"hclust_gene_program"`. |
| `tau` | `0.1` | Positive demand temperature. |
| `gamma` | `1.0` | Non-negative demand exponent. |
| `temp_cell` | `0.1` | Positive cell-fit temperature. |
| `eps` | `0.01` | Positive numerical offset. |
| `bias_clip_min` | `-3.0` | Lower bias bound. |
| `bias_clip_max` | `3.0` | Upper bias bound; > lower bound. |
| `beta_s1` | `0.5` | Cell-type routing scale. |
| `beta_d2` | `0.15` | Cell-level routing scale. |
| `quota_min` | `1` | Positive minimum quota. |
| `max_programs` | `8` | Positive maximum gene programs. |

## `TwinPanelConfig`

| Parameter | Default | Meaning / accepted values |
|---|---:|---|
| `lambda_agree` | `0.05` | Non-negative agreement weight; `0` disables twin panels. |
| `informative_fraction` | `0.5` | Fraction strictly between `0` and `1`. |
| `shared_core_fraction` | `0.2` | Fraction in `[0, 1)`. |
| `fusion_mode` | `"mean"` | `"mean"` only. |

## `OptimisationConfig`

| Parameter | Default | Meaning / accepted values |
|---|---:|---|
| `density_prior` | `"rna_count_based"` | `"rna_count_based"`, `"uniform"`, a probability list, or `None`. |
| `num_epochs` | `500` | Non-negative training epochs. |
| `learning_rate` | `0.1` | Positive learning rate. |
| `lambda_g1` | `1.0` | Non-zero gene-similarity weight. |
| `lambda_d` | `0.0` | Non-negative density weight. |
| `lambda_g2` | `0.0` | Non-negative spot-similarity weight. |
| `lambda_r` | `0.0` | Non-negative entropy weight. |
| `lambda_l1` | `0.0` | Non-negative residual L1 weight. |
| `lambda_l2` | `0.0` | Non-negative residual L2 weight. |
| `print_each` | `100` | Positive logging interval or `None`. |

Nested configs use their defaults unless supplied to `ArcImputeConfig`.
