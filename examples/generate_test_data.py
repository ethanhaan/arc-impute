"""Write the fixed synthetic inputs used by the minimal example."""

from pathlib import Path

import numpy as np
import pandas as pd
from anndata import AnnData


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

# Hand-written expression values are used directly, without normalisation.
example_dir = Path(__file__).resolve().parent
reference.write_h5ad(
    example_dir / "arc_impute_test_reference.h5ad", convert_strings_to_categoricals=False
)
spatial.write_h5ad(
    example_dir / "arc_impute_test_spatial.h5ad", convert_strings_to_categoricals=False
)
print("Wrote synthetic reference (4 cells, 4 genes) and spatial data (3 spots, 3 genes).")
