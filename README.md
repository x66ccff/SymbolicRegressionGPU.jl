
# 🚀 SymbolicRegressionGPU.jl 

💻 [PSRN](https://github.com/intell-sci-comput/PTS) (Parallel Symbolic Regression Network) enhanced SymbolicRegression.jl for **faster**, large-scale parallel symbolic evaluations on GPUs. _Based on [SymbolicRegression.jl](https://github.com/MilesCranmer/SymbolicRegression.jl)_.  [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ai.damtp.cam.ac.uk/symbolicregression/dev/)

# Quickstart

### 📥 1. clone this repo 

```bash
git clone https://github.com/x66ccff/SymbolicRegressionGPU.jl
```

### 📦 2. Auto install [libtorch](https://pytorch.org/cppdocs/installing.html) and [THArrays.jl](https://github.com/compintell/THArrays.jl/)

```bash
chmod +x install.sh
./install.sh
```

### 🏃‍♂️ 3. Run 
```bash
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
  title = {SymbolicRegressionGPU.jl: PSRN enhanced SymbolicRegression.jl for fast, large-scale parallel symbolic evaluations on GPUs}, 
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
