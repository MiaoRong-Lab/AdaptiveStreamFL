# AdaptiveStreamFL

Reference implementation for:

> Xiong, T.; Zhang, C.; Rong, M.; Gong, D. W.; Yang, S. X.  
> **AdaptiveStreamFL: A Bayesian-enhanced multi-scale federated learning framework for dynamic data streams with uncertainty quantification.**  
> *Expert Systems with Applications*, 299, 129882, 2026.  
> DOI: [10.1016/j.eswa.2025.129882](https://doi.org/10.1016/j.eswa.2025.129882)

AdaptiveStreamFL extends prototype-based federated stream learning with adaptive
K-neighbor selection, multi-scale prototype learning, uncertainty-aware
aggregation, and uncertainty-aware decision mechanisms for dynamic data streams.

## Repository Status

This repository provides an independent reference implementation of the
AdaptiveStreamFL paper code. The implementation is organized as a Python package
with a backward-compatible command-line wrapper.

The project is released under the MIT License. FedStream is cited as related
research and as a baseline context; this repository does not include FedStream
source files. See `THIRD_PARTY_NOTICES.md`.

## Installation

Python 3.8 or newer is recommended.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Editable package installation is also supported:

```bash
pip install -e .
```

The dependency file is intentionally named `requirements.txt`, following the
standard Python convention.

## Data

Prepared datasets are expected as `.npy` files whose last column is the class
label.

Default location:

```text
dataset/
```

Common dataset filenames used by the command-line examples are:

- `covtype.npy`
- `electricity.npy`
- `occupancy.npy`
- `shuttle.npy`
- `kddcup99.npy`

Large raw or converted datasets should normally not be committed to Git. Prepare
benchmark datasets from their original sources, record the preprocessing steps,
and publish large artifacts via a release archive or an external data repository
when redistribution is permitted.

Dataset source links and preparation notes are listed in `dataset/README.md`.

## Quick Start

Run AdaptiveStreamFL:

```bash
python AdaptiveStreamFL.py --dataset covtype --clients 3 --max_mc 100 --reporting_interval 50
```

Equivalent package-style entry point:

```bash
python -m adaptivestreamfl --dataset covtype --clients 3 --max_mc 100
```

Use custom data and output directories:

```bash
python AdaptiveStreamFL.py ^
  --dataset covtype ^
  --clients 3 ^
  --data_dir dataset ^
  --output_dir results ^
  --seed 42
```

Run the FedStream-style baseline mode exposed by this codebase:

```bash
python AdaptiveStreamFL.py --run_type FedStream --dataset covtype --clients 3
```

## Tests

Run the built-in smoke tests:

```bash
python -m unittest discover -s tests
```

The tests create a tiny synthetic dataset in a temporary directory and verify
that both the FedStream baseline mode and AdaptiveStreamFL mode produce result
CSVs.

## Project Layout

```text
AdaptiveStreamFL.py                  Backward-compatible CLI wrapper
adaptivestreamfl/adaptive.py         AdaptiveStreamFL adaptive components
adaptivestreamfl/cli.py              Command-line argument handling
adaptivestreamfl/core.py             Main training runner
adaptivestreamfl/data.py             Dataset loading and stream partitioning
adaptivestreamfl/microcluster.py     Micro-cluster data structure
adaptivestreamfl/runtime.py          Runtime constants and small utilities
results/                             Generated outputs, ignored by Git
dataset/                             Local datasets, large files ignored by Git
```

## Citation

If you use this repository, please cite:

```bibtex
@article{xiong2026adaptivestreamfl,
  title = {AdaptiveStreamFL: A Bayesian-enhanced multi-scale federated learning framework for dynamic data streams with uncertainty quantification},
  author = {Xiong, Tai and Zhang, Chi and Rong, Miao and Gong, Dunwei and Yang, Shengxiang},
  journal = {Expert Systems with Applications},
  volume = {299},
  pages = {129882},
  year = {2026},
  doi = {10.1016/j.eswa.2025.129882}
}
```

Because this work is related to and compares with FedStream, cite the original
FedStream paper when you discuss that baseline or the relationship:

```bibtex
@article{mawuli2023fedstream,
  title = {FedStream: Prototype-Based Federated Learning on Distributed Concept-Drifting Data Streams},
  author = {Mawuli, Cobbinah B. and Che, Liwei and Kumar, Jay and Din, Salah Ud and Qin, Zhili and Yang, Qinli and Shao, Junming},
  journal = {IEEE Transactions on Systems, Man, and Cybernetics: Systems},
  volume = {53},
  number = {11},
  pages = {7112--7124},
  year = {2023},
  doi = {10.1109/TSMC.2023.3293462}
}
```
