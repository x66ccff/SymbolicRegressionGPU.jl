
# 🚀 SymbolicRegressionGPU.jl 

💻 [PSRN](https://github.com/intell-sci-comput/PTS) (Parallel Symbolic Regression Network) enhanced SymbolicRegression.jl for **faster**, large-scale parallel symbolic evaluations on GPUs. _Based on [SymbolicRegression.jl](https://github.com/MilesCranmer/SymbolicRegression.jl)_.

In this repository, the implementation of PSRN is based on the high-performance _[Reactant.jl](https://github.com/EnzymeAD/Reactant.jl)_ library, which can automatically recognize device backends, including CPU, GPU, and TPU, without the need for manual device specification.


## 🔍 SymbolicRegression.jl

SymbolicRegression.jl searches for symbolic expressions which optimize a particular objective.

SymbolicRegression.jl docs:
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ai.damtp.cam.ac.uk/symbolicregression/dev/)

# Quickstart

### 📥 Install

```bash
git clone https://github.com/x66ccff/SymbolicRegressionGPU.jl
```

[Install Julia](https://julialang.org/downloads/)

```julia
julia> ]
(@v1.x) pkg> activate .
(SymbolicRegression) pkg> resolve
(SymbolicRegression) pkg> instantiate
```
### 🏃‍♂️ Run 
```
export XLA_REACTANT_GPU_MEM_FRACTION=0.99
julia -t 16 --project=. example.jl
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
