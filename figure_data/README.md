# Figure plotting data

These CSV files contain the final numerical inputs underlying Figures 2--5 of
the ARC-Impute manuscript. They can be plotted without rerunning model fitting.
Missing metric values are excluded rather than replaced with zero.

## Figure 2

- `figure_2/figure_2ab_plot_data.csv`: for each model, calculate the mean
  per-gene `pearson` and `css` values for panel a; the 95% confidence interval
  is `1.96 * sample standard deviation / sqrt(non-missing gene count)`. Plot the
  per-gene `pearson` distributions in panel b and annotate non-missing medians.
- `figure_2/figure_2c_plot_data.csv`: for each displayed gene and model, scatter
  `x` against `y` and colour spots by `intensity`. Intensities are already
  calibrated; do not rescale them. Use the maximum intensity within each gene
  as the common colour limit across that gene's ground-truth and model panels.

## Figure 3

- `figure_3/figure_3_plot_data.csv`: split rows by `platform_group`, then plot
  `relative_rank` for each dataset, metric and model. Scores are ranked within
  each dataset and metric in descending order, using average ranks for ties:
  `relative_rank = (model_count - rank + 1) / model_count`.

## Figure 4

- `figure_4/figure_4a_plot_data.csv`: plot `score` by dataset and model for each
  metric, with `ci95_low` and `ci95_high` as confidence-interval whiskers.
- `figure_4/figure_4b_plot_data.csv`: scatter `x` against `y` for each model and
  colour each spot by its dominant `cell_type` category.

## Figure 5

- `figure_5/figure_5_plot_data.csv`: make one scatter panel per `panel`, using
  `x_mean_relative_rank` and `y_mean_relative_rank` as model coordinates. Axis
  names and contributing dataset counts are supplied by the corresponding
  metric and count columns.

The manuscript and supplement define the metrics, datasets and experimental
design. These tables preserve the values used for plotting; cosmetic styling,
model ordering and panel layout do not change the reported measurements.
