#!/bin/bash

# # Step 1: Clone the repository
# echo "📥 Cloning the repository..."
# git clone https://github.com/x66ccff/SymbolicRegressionGPU.jl

# Step 2: Download and set up libtorch
echo "📦 Downloading and setting up libtorch..."
wget https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu121.zip
unzip libtorch*.zip
cp -r libtorch THArrays.jl/csrc

# Step 3: Install THArrays
echo "🔧 Installing THArrays..."

# Set the development environment variable
export THARRAYS_DEV=1

# Enter Julia REPL and activate the project environment
echo "Activating Julia environment..."
julia -e 'using Pkg; Pkg.activate("."); Pkg.develop(path="./THArrays.jl"); Pkg.build("THArrays"); Pkg.instantiate()'

# Set CUDAARCHS for NVIDIA GPUs
export CUDAARCHS="native"

echo "🎉 Installation complete!"