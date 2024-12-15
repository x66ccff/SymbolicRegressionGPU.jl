# 🚀 SymbolicRegressionGPU.jl 

💻 [PSRN](https://github.com/intell-sci-comput/PTS) (Parallel Symbolic Regression Network) enhanced SymbolicRegression.jl for fast, large-scale parallel symbolic evaluations on GPUs. Based on SymbolicRegression.jl 

<!-- prettier-ignore-start -->


## 🔍 SymbolicRegression.jl

SymbolicRegression.jl searches for symbolic expressions which optimize a particular objective.

SymbolicRegression.jl docs:
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ai.damtp.cam.ac.uk/symbolicregression/dev/)

# How to use SymbolicRegressionGPU.jl?

### 📥 1. clone this repo 

### 📦 2. download libtorch and then unzip, place `libtorch` into `THArrays.jl/csrc`
```bash
wget https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu121.zip
```
### 🔧 3. install THArrays 
see https://github.com/compintell/THArrays.jl/

```
export THARRAYS_DEV=1

(SymbolicRegression) pkg> dev ./THArrays.jl

(SymbolicRegression) pkg> build THArrays
```
### 🏃‍♂️ 4. Run 
```
export JULIA_NUM_THREADS=4    # allow @spawn for starting PSRN task

julia --project=. example.jl
```

# 📚 Citing 

To cite this fork SymbolicRegressionGPU.jl, please use the following BibTeX entry:

```bibtex
@misc{SymbolicRegressionGPU.jl,
  author = {
    Ruan, Kai AND
    Cranmer, Miles AND
    Sun, Hao
  },
  title = {SymbolicRegressionGPU.jl}, 
  year = {2024},
  url = {https://github.com/x66ccff/SymbolicRegressionGPU.jl}
}
```

```bibtex
@misc{cranmerInterpretableMachineLearning2023,
    title = {Interpretable {Machine} {Learning} for {Science} with {PySR} and {SymbolicRegression}.jl},
    url = {http://arxiv.org/abs/2305.01582},
    doi = {10.48550/arXiv.2305.01582},
    urldate = {2023-07-17},
    publisher = {arXiv},
    author = {Cranmer, Miles},
    month = may,
    year = {2023},
    note = {arXiv:2305.01582 [astro-ph, physics:physics]},
    keywords = {Astrophysics - Instrumentation and Methods for Astrophysics, Computer Science - Machine Learning, Computer Science - Neural and Evolutionary Computing, Computer Science - Symbolic Computation, Physics - Data Analysis, Statistics and Probability},
}
```

🎉 Enjoy your symbolic regression journey with SymbolicRegressionGPU.jl! 🎉
