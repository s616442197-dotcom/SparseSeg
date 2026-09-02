# Source data for the v3 manuscript figures

This directory contains the machine-readable values underlying the quantitative
and data-analysis panels in `SparseSeg_final0709_v3.tex` and
`SparseSeg_supplementary_v3.tex`. Schematic panels and representative raw,
prediction-overlay, and three-dimensional rendering panels are image data and
are not duplicated here as tabular source data.

## Figure-to-file map

| Manuscript item | Quantitative content | Source data |
|---|---|---|
| Main Fig. 4a (`fig5.pdf`) | SparseSeg versus StarDist at 50%, 70%, and 95% masking, 10 trials and three cell types | `main_figure5/sparse_label_comparison.csv` |
| Main Fig. 5a (`Fig_add1_v3.pdf`) | Ten-model relative-IoU benchmark | `../benchmark_pipeline/data/benchmark_metrics.csv` |
| Main Fig. 5b (`Fig_add1_v3.pdf`) | Training time per epoch | `../benchmark_pipeline/data/time_per_epoch.csv` |
| Main Fig. 6a (`fig4.pdf`) | Cross-dataset generalization | `main_figure4/cross_dataset_generalization.csv` |
| Main Fig. 6b,c (`fig4.pdf`) | Two- and three-dimensional mitochondrial morphology across HeLa, Jurkat and macrophage data | Not packaged; see **Known source-data gap** below |
| Main Fig. 7c,d (`fig6.pdf`) | Control-versus-patient mitochondrial morphology | `main_figure6/regionprops_2d_control_vs_patient.csv`, `main_figure6/regionprops_3d_control_vs_patient.csv`, and the two statistics tables |
| Supplementary Fig. 3 (`Fig_add2.pdf`) | Threshold, kernel, IoU-threshold, low-weight, patch-size, backbone, loss and feature-group ablations | `supplementary_ablation/ablation_metrics.csv` |
| Supplementary Fig. 5 (`Fig_add_more1_v3.pdf`) | Twelve-model relative IoU, absolute IoU, precision, recall, predicted foreground fraction and wall-clock time | `../benchmark_pipeline/data/benchmark_metrics.csv` and `../benchmark_pipeline/data/wallclock_total.csv` |
| Supplementary Tables 1 and 2 | SparseSeg CPU, RAM, GPU utilization and GPU memory | `supplementary_resource_tables/sparseseg_resource_measurements.csv` |

The figure numbers above follow their order in the v3 manuscript; the TeX file
labels are retained in parentheses where useful because some legacy PDF file
names do not match the displayed figure number.

## Provenance

- `main_figure5/sparse_label_comparison.csv` is a lossless long-form export of
  the three complete evaluator caches written by `evaluation_new.py`. It has
  180 rows: 3 cell types x 2 models x 3 masking ratios x 10 trials.
- `main_figure4/cross_dataset_generalization.csv` is a lossless long-form export
  of the complete dictionaries written by `evaluation_cross_model.py`. It has
  270 rows: 3 training data sets x 3 test data sets x 3 masking ratios x 10
  trials.
- The four Main Fig. 7 tables are the raw object-level measurements and
  statistical summaries produced by `static_property.py`; no plotted samples
  were removed during packaging.
- `supplementary_ablation/ablation_metrics.csv` is a lossless long-form export
  of the complete evaluator caches written by the existing
  `evaluation_cross_trials_ablation*.py` scripts. It has 175 rows and includes
  every plotted setting and all five repeats.
- The main-text ROI benchmark and extended Supplementary Fig. 5 source tables
  are already version-controlled in `benchmark_pipeline/data/`. The same CSVs
  are consumed directly by `benchmark_pipeline/plot_benchmark_figures.py` and
  by the `--plots-only` mode of `benchmark_pipeline/run_benchmark_pipeline.py`.

`SHA256SUMS` records the packaged-file hashes. All CSV files use UTF-8 text,
comma delimiters, a single header row and one observation per row.

## Known source-data gap

The object-level values underlying Main Fig. 6b,c are the only quantitative
figure data not recovered in this archive. The historical analysis is present
in `static_property.py`; the configuration added in Git commit
`b8f86fc5a999604e277a8c64095aeb41adaffb67` reads iteration-2 SparseSeg outputs
from `label_hela2_mito_80/volume_mask_pred_2.tiff`,
`label_jurkat_mito_80/volume_mask_pred_2.tiff`, and
`label_macrophage_mito_80/volume_mask_pred_2.tiff`. Those three prediction
volumes and the CSVs previously derived from them were not found in the local
data archive or the authors' remote project directory.

The similarly named files under `D:/vem_data/benchmark/` are not substitutes:
they are single-channel `uint16` instance-label outputs rather than the
four-channel SparseSeg prediction volumes consumed by the historical script,
and the macrophage file contains no foreground voxels. They were therefore not
used to manufacture replacement source values. Reproduction of these two
panels requires rerunning the released historical `segment_cell.py` workflow
for the three 80% sparse-mask inputs and then running `static_property.py` with
the original threshold (`0.5`), 2D minimum area (`400` pixels), 3D minimum
volume (`500` resampled voxels), and fivefold z-axis resampling.
