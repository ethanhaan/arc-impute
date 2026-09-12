from __future__ import annotations

from pathlib import Path

from anndata import read_h5ad

from arc_impute import (
    AnchorResidualConfig,
    ArcImpute,
    ArcImputeConfig,
    OptimisationConfig,
    ReferenceCurationConfig,
)


example_dir = Path(__file__).resolve().parent
reference = read_h5ad(example_dir / "arc_impute_test_reference.h5ad")
spatial = read_h5ad(example_dir / "arc_impute_test_spatial.h5ad")

model = ArcImpute(
    ArcImputeConfig(
        cell_type_key="cell_type",
        region_key="region",
        reference_curation=ReferenceCurationConfig(q_min=1, budget_fraction=1.0),
        anchor_residual=AnchorResidualConfig(
            coarse_optimisation_seed=1,
            coarse_optimisation_epochs=2,
            coarse_optimisation_runs=1,
        ),
        optimisation=OptimisationConfig(num_epochs=2, print_each=None),
    )
)
model.prepare(reference, spatial, train_genes=["g1", "g2", "g3"])
model.fit()

imputed = model.impute(genes=["reference_only"])
composition = model.deconvolve()

assert imputed.shape == (3, 1)
assert composition.shape == (3, 2)
print(imputed)
print(composition)
