# Frozen benchmark environments

The root `requirements.txt` is intentionally only the lightweight plotting/orchestration environment. Exact model environments are archived separately here:

- `vem_module_pip_freeze.txt`: SparseSeg, SparseSeg-ViT, Vanilla U-Net and related PyTorch code used for the new controls (`torch==2.5.1`, CUDA runtime reported by PyTorch as 12.1 in the timing records).
- `nnunetv2_official_pip_freeze.txt`: official nnU-Net v2 environment (`nnunetv2==2.8.1`, `torch==2.5.1`).
- `empanada_pip_freeze.txt`: MitoNet/Empanada environment.
- `stardist_pip_freeze.txt`: legacy StarDist environment (`stardist==0.9.1`, `tensorflow==2.16.2`, `numpy==1.26.4`).
- `deepict_pip_freeze.txt`: legacy DeePiCt environment (`tensorflow-gpu==2.0.0`, `snakemake==5.13.0`).
- `cellmap_csc_pip_freeze.txt`: historical CellMap environment snapshot. The optional architecture package is pinned to `https://github.com/janelia-cellmap/cellmap-segmentation-challenge.git` commit `0300239cd0b4867d4bab008aa9e95161b2442d93`; the released COSEM controls use the repository-tracked explicit sampler, weighted-BCE training loop and background treatment.
- `legacy_workstation_system.txt`: workstation kernel, GPU and driver snapshot.
- `hardware_provenance.csv`: method-level hardware and timing scope. The formal set was not run on a single GPU model, so wall-clock values are descriptive measurements and not hardware-normalized speed rankings.

For a new machine, recreate each environment independently. GPU framework compatibility must be checked against the local driver; do not merge the TensorFlow 2.0 DeePiCt environment with modern PyTorch environments.
