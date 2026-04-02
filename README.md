# starrynite_py

Modern Python pipeline for nuclear detection, tracking, and lineaging in C. elegans embryos.

A reimplementation of [StarryNite](https://github.com/zhirongbaolab/StarryNite) using modern deep learning and optimization libraries, designed to work with [acetree_py](https://github.com/shahlab-ucla/acetree_py) for visualization and curation.

## Architecture

```
TIFF stacks ──→ StarDist 3D ──→ ultrack / btrack ──→ acetree_py
 (dask lazy)     (detection)      (tracking)         (visualization)
```

| Stage | Library | Replaces |
|-------|---------|----------|
| Detection | [StarDist 3D](https://github.com/stardist/stardist) | DoG filtering + ray-based diameter + Bayesian assignment (~90 MATLAB files) |
| Tracking | [ultrack](https://github.com/royerlab/ultrack) (ILP) / [btrack](https://github.com/quantumjot/btrack) (Bayesian) | Greedy linking + MVN classifiers + bifurcation resolution (~100 MATLAB files) |
| Visualization | [acetree_py](https://github.com/shahlab-ucla/acetree_py) | Java AceTree |
| Data loading | tifffile + dask | Custom TIFF/LSM readers |
| Configuration | Pydantic v2 + YAML | eval()-parsed text files |

## Installation

```bash
# CPU only
pip install -e ".[detection,tracking,dev]"

# GPU (requires Python 3.10 + CUDA 11.2 — see GPU_SETUP.md)
pip install tensorflow==2.10.1
pip install -e ".[detection,tracking,dev]"
```

## Quick Start

```python
from starrynite_py.config import load_config
from starrynite_py.pipeline import run_pipeline

config = load_config("configs/nih_dispim.yaml")
detections, tracking, export_path = run_pipeline(config)
# Output: acetree_py-compatible ZIP file
```

## CLI

```bash
starrynite-py detect --config configs/nih_dispim.yaml
starrynite-py run --config configs/nih_dispim.yaml
starrynite-py evaluate --config configs/nih_dispim.yaml
```

## Performance

Tested on NIH diSPIM deconvolved C. elegans dataset (Moyle et al., Nature 2021):

| Stage | Time per volume | Hardware |
|-------|----------------|----------|
| Detection (StarDist 3D) | 8.6s | RTX A5000 GPU |
| Detection (StarDist 3D) | 29s | CPU (AVX2) |
| Tracking (ultrack, 10 frames) | 134s | CPU (ILP solver) |

## Project Status

- [x] Package scaffold + config system
- [x] I/O: TIFF loading, ground truth parsing, acetree_py export
- [x] StarDist 3D detection (pretrained + GPU)
- [x] ultrack tracking (ILP-based)
- [x] btrack tracking (Bayesian)
- [x] Detection evaluation framework
- [x] End-to-end pipeline + CLI
- [ ] StarDist fine-tuning on C. elegans data (in progress)
- [ ] Full-scale evaluation on all 3 embryos
- [ ] Tracking evaluation with lineage comparison

## License

GPL-3.0 (same as original StarryNite)
