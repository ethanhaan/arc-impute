# ARC-Impute

ARC-Impute maps an annotated scRNA-seq reference to spatial transcriptomics
spots for spatial gene imputation and cell-type deconvolution.

## Install

```bash
python -m pip install "arc-impute @ git+https://github.com/ethanhaan/arc-impute.git"
```

## Use

```python
from arc_impute import ArcImpute, ArcImputeConfig

model = ArcImpute(
    ArcImputeConfig(cell_type_key="cell_type", region_key="region")
)
model.prepare(reference, spatial, train_genes=["GeneA", "GeneB"])
model.fit()

imputed = model.impute(genes=["GeneC"])
composition = model.deconvolve()
```

`reference` and `spatial` must be `AnnData` objects with unique observation
and gene names, finite expression values, and the configured annotation
columns. Training genes must occur in both datasets; imputed genes need only
occur in the reference. ARC-Impute does not normalise expression matrices.

The default device is CPU. Set `device="cuda:0"` for CUDA. See
[parameters](docs/parameters.md) for all options.

Outputs are available as:

- `assignment_`: reference-cell-by-spot mapping
- `impute(...)`: spot-by-gene `DataFrame`
- `deconvolve()`: spot-by-cell-type `DataFrame`
- `training_history_` and `diagnostics_`

## Example and tests

The example uses two small synthetic test datasets in `examples/`:

- [Reference](examples/arc_impute_test_reference.h5ad): four cells labelled
  `A` or `B` in `.obs["cell_type"]`, with genes `g1`, `g2`, `g3` and `reference_only`.
- [Spatial](examples/arc_impute_test_spatial.h5ad): three spots labelled
  `R1` or `R2` in `.obs["region"]`, with the three shared genes `g1`, `g2` and `g3`.

These hand-written expression values are used directly without normalisation.
This example demonstrates the workflow only; it is not a biological benchmark.
Run it from the repository root after cloning or downloading the repository:

```bash
python -m pip install -e ".[test]"
python examples/minimal_example.py
python -m pytest -q
```

The example fits the mapping and prints a 3-by-1 imputation table for
`reference_only` and a 3-by-2 cell-type proportion table, both indexed by spot.
The bundled inputs can be regenerated with `python examples/generate_test_data.py`.

## Modules

- `modules/anchor_residual.py`
- `modules/reference_curation_routing.py`
- `modules/twin_panel_consistency.py`

## License

BSD 3-Clause. ARC-Impute follows a mapping formulation inspired by Tangram but
does not import or call Tangram at runtime.
