#!/bin/bash

# Check if Julia is installed
if ! command -v julia &> /dev/null; then
    echo "❌ Julia is not installed or not in your PATH. Please install Julia and try again."
    exit 1
fi

# Check if CMake is installed
if ! command -v cmake &> /dev/null; then
    echo "❌ CMake is not installed or not in your PATH. Please install CMake and try again."
    exit 1
fi

# Step 2: Download and set up libtorch
LIBTORCH_ZIP="libtorch-cxx11-abi-shared-with-deps-2.1.0+cu121.zip"
LIBTORCH_URL="https://download.pytorch.org/libtorch/cu121/$LIBTORCH_ZIP"

echo "📦 Downloading and setting up libtorch..."

# Check if the libtorch zip file already exists
if [ -f "$LIBTORCH_ZIP" ]; then
    read -p "The libtorch zip file already exists. Do you want to re-download it? [y/N]: " REDOWNLOAD
    REDOWNLOAD=${REDOWNLOAD:-N}  # Default to N if no input
    if [[ "$REDOWNLOAD" =~ ^[Yy]$ ]]; then
        echo "Re-downloading libtorch..."
        wget_output=$(wget "$LIBTORCH_URL" -O "$LIBTORCH_ZIP" 2>&1)
        if echo "$wget_output" | grep -q "ERROR"; then
            echo "❌ Download failed. Please manually download the file from the following URL:"
            echo "   $LIBTORCH_URL"
            echo "After downloading, place the file in the SymbolicRegressionGPU.jl dir and re-run this script."
            exit 1
        fi
    else
        echo "Skipping download."
    fi
else
    echo "Downloading libtorch..."
    wget_output=$(wget "$LIBTORCH_URL" 2>&1)
    if echo "$wget_output" | grep -q "ERROR"; then
        echo "❌ Download failed. Please manually download the file from the following URL:"
        echo "   $LIBTORCH_URL"
        echo "After downloading, place the file in the SymbolicRegressionGPU.jl dir and re-run this script."
        exit 1
    fi
fi

# Check if the libtorch directory already exists
if [ -d "libtorch" ]; then
    read -p "The libtorch directory already exists. Do you want to re-unzip it? [y/N]: " REUNZIP
    REUNZIP=${REUNZIP:-N}  # Default to N if no input
    if [[ "$REUNZIP" =~ ^[Yy]$ ]]; then
        echo "Re-unzipping libtorch..."
        rm -rf libtorch  # Remove existing directory
        unzip "$LIBTORCH_ZIP"
    else
        echo "Skipping unzip."
    fi
else
    echo "Unzipping libtorch..."
    unzip "$LIBTORCH_ZIP"
fi

# Copy libtorch to THArrays.jl/csrc
echo "Copying libtorch to THArrays.jl/csrc..."
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