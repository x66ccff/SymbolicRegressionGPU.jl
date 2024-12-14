<!-- prettier-ignore-start -->


SymbolicRegression.jl searches for symbolic expressions which optimize a particular objective.

<!-- https://github.com/MilesCranmer/SymbolicRegression.jl/assets/7593028/f5b68f1f-9830-497f-a197-6ae332c94ee0 -->

| Latest release | Documentation | Forums | Paper |
| :---: | :---: | :---: | :---: |
| [![version](https://juliahub.com/docs/SymbolicRegression/version.svg)](https://juliahub.com/ui/Packages/SymbolicRegression/X2eIS) | [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ai.damtp.cam.ac.uk/symbolicregression/dev/) | [![Discussions](https://img.shields.io/badge/discussions-github-informational)](https://github.com/MilesCranmer/PySR/discussions) | [![Paper](https://img.shields.io/badge/arXiv-2305.01582-b31b1b)](https://arxiv.org/abs/2305.01582) |

| Build status | Coverage |
| :---: | :---: |
| [![CI](https://github.com/MilesCranmer/SymbolicRegression.jl/workflows/CI/badge.svg)](.github/workflows/CI.yml) | [![Coverage Status](https://coveralls.io/repos/github/MilesCranmer/SymbolicRegression.jl/badge.svg?branch=master)](https://coveralls.io/github/MilesCranmer/SymbolicRegression.jl?branch=master) |

Check out [PySR](https://github.com/MilesCranmer/PySR) for
a Python frontend.
[Cite this software](https://arxiv.org/abs/2305.01582)


# How to use SymbolicRegressionGPU.jl?

### 1. clone this repo

### 2. download libtorch and then unzip, place `libtorch` into `THArrays.jl/csrc`
```bash
wget https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu121.zip
```
### 3. install THArrays
see https://github.com/compintell/THArrays.jl/

```
export THARRAYS_DEV=1

(SymbolicRegression) pkg> dev ./THArrays.jl

(SymbolicRegression) pkg> build THArrays
```
### 4. Run
```
export JULIA_NUM_THREADS=4    # allow @spawn for starting PSRN task

julia --project=. example.jl
```

