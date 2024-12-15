> [!WARNING]  
> This package is under active development at the moment and may change its API and supported end systems at any time. End-users are advised to wait until a corresponding release with broader availability is made. Package developers are suggested to try out Reactant for integration, but should be advised of the same risks.


# 🚀 SymbolicRegressionGPU.jl 

💻 [PSRN](https://github.com/intell-sci-comput/PTS) (Parallel Symbolic Regression Network) enhanced SymbolicRegression.jl for **faster**, large-scale parallel symbolic evaluations on GPUs. _Based on [SymbolicRegression.jl](https://github.com/MilesCranmer/SymbolicRegression.jl)_.

<!-- prettier-ignore-start -->


## 🔍 SymbolicRegression.jl

SymbolicRegression.jl searches for symbolic expressions which optimize a particular objective.

SymbolicRegression.jl docs:
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ai.damtp.cam.ac.uk/symbolicregression/dev/)

# How to use SymbolicRegressionGPU.jl?

### 📥 1. clone this repo 

```bash
git clone https://github.com/x66ccff/SymbolicRegressionGPU.jl
```

### 📦 2. download libtorch and then unzip, place `libtorch` into `THArrays.jl/csrc`
```bash
wget https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu121.zip
unzip libtorch*.zip
mv libtorch SymbolicRegressionGPU.jl/THArrays.jl/csrc
```
### 🔧 3. install THArrays 
see https://github.com/compintell/THArrays.jl/ for more details

1. Go to the project directory where THArrays.jl is located
2. Set the development environment variable:
```bash
cd SymbolicRegressionGPU.jl
export THARRAYS_DEV=1
```
3. Enter Julia REPL and activate the project environment:
```julia
julia> ]
(@v1.x) pkg> activate .
```
4. Install and build THArrays:

```bash
export CUDAARCHS="native" # For nvidia GPUs
```

```julia
(SymbolicRegression) pkg> instantiate
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
