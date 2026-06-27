from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData

from arc_impute import (
    AnchorResidualConfig,
    ArcImpute,
    ArcImputeConfig,
    OptimisationConfig,
    ReferenceCurationConfig,
)


reference = AnnData(
    X=np.array(
        [
            [5, 4, 1, 8],
            [4, 5, 1, 7],
            [1, 2, 5, 2],
            [1, 1, 6, 3],
        ],
        dtype=np.float32,
    ),
    obs=pd.DataFrame(
        {"cell_type": ["A", "A", "B", "B"]},
        index=["cell_1", "cell_2", "cell_3", "cell_4"],
    ),
    var=pd.DataFrame(index=["g1", "g2", "g3", "reference_only"]),
)
spatial = AnnData(
    X=np.array([[5, 4, 1], [2, 2, 4], [1, 1, 6]], dtype=np.float32),
    obs=pd.DataFrame(
        {"region": ["R1", "R1", "R2"]},
        index=["spot_1", "spot_2", "spot_3"],
    ),
    var=pd.DataFrame(index=["g1", "g2", "g3"]),
)

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
