using SHA
using Downloads
using ZipFile
using ProgressMeter
using Tar
using CodecZlib
using Pkg

function check_dependencies()
    println("🔍 Checking dependencies...")
    
    # Check CMake
    try
        run(`cmake --version`)
        println("✅ CMake is installed")
    catch
        error("❌ CMake not found. Please install CMake and try again.")
    end
end
using ProgressMeter: Progress, BarGlyphs, update!, next!, finish!

function download_with_retry(url, output_path; max_retries=3, timeout=600)
    for attempt in 1:max_retries
        try
            println("\n📥 Downloading libtorch $attempt/$max_retries...")
            
            p = Progress(1; dt=0.1, barglyphs=BarGlyphs("[=> ]"), barlen=50, color=:cyan)
            
            let total_size = Ref{Int}(0)
                Downloads.download(
                    url,
                    output_path;
                    timeout=timeout,
                    progress = (total, now) -> begin
                        if total_size[] == 0 && total > 0
                            total_size[] = total
                            p.n = total
                        end
                        if total > 0
                            update!(p, now)
                        end
                    end,
                    headers=["User-Agent" => "Julia/1.11"]
                )
            end
            
            finish!(p)
            println("\n✅ Download successful!")
            return true
        catch e
            println("\n❌ Download failed: $e")
            if attempt == max_retries
                rethrow(e)
            else
                println("Waiting 10 seconds before retrying...")
                sleep(10)
            end
        end
    end
    return false
end

function setup_libtorch(temp_dir)
    println("\n📦 Setting up libtorch...")
    
    zip_path = joinpath(temp_dir, "libtorch.zip")
    artifact_dir = joinpath(temp_dir, "artifact")
    mkpath(artifact_dir)
    
    # Download file
    url = "https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu121.zip"
    
    if !download_with_retry(url, zip_path, timeout=600)
        error("Download failed. Please check your network connection or use a proxy")
    end
    
    # Extract files
    println("\n📦 Extracting files...")
    rd = ZipFile.Reader(zip_path)
    
    total_files = length(rd.files)
    p = Progress(total_files; desc="Extraction progress: ", showspeed=true)
    
    for f in rd.files
        fullpath = joinpath(artifact_dir, f.name)
        if endswith(f.name, '/')
            mkpath(fullpath)
        else
            mkpath(dirname(fullpath))
            write(fullpath, read(f))
        end
        next!(p)
    end
    close(rd)
    println("✅ Extraction complete!")
    
    return joinpath(artifact_dir, "libtorch")  # Return the path to the extracted libtorch directory
end


function compile_tharray(libtorch_path)
    println("\n🔧 Compiling THArrays...")
    
    # Set environment variables
    ENV["THARRAYS_DEV"] = "1"
    ENV["CUDAARCHS"] = "native"
    
    # Get correct directory path
    # Go up two levels from deps/build.jl to get project root directory
    root_dir = dirname(dirname(@__FILE__))
    
    # THArrays is in src of main package
    csrc_dir = joinpath(root_dir, "src", "THArrays", "csrc")
    println("Checking directory: $csrc_dir")
    
    if !isfile(joinpath(csrc_dir, "CMakeLists.txt"))
        error("CMakeLists.txt file not found in $csrc_dir")
    end
    
    # Copy libtorch to csrc directory
    println("Copying libtorch to $csrc_dir")
    cp(libtorch_path, joinpath(csrc_dir, "libtorch"), force=true)
    
    # Create build directory
    build_dir = joinpath(csrc_dir, "build")
    mkpath(build_dir)
    
    # Run CMake
    cd(build_dir) do
        println("Running CMake in directory $(pwd())")
        println("Source code directory: $csrc_dir")
        run(`cmake $csrc_dir`)
        run(`cmake --build . --config Release`)
    end
    
    println("✅ Compilation complete!")
end

function main()
    try
        # Check dependencies
        check_dependencies()
        
        # Create temporary directory
        temp_dir = mktempdir()
        
        # Set up libtorch
        libtorch_path = setup_libtorch(temp_dir)
        
        # Compile THArrays
        compile_tharray(libtorch_path)
        
        # Clean up
        println("\n🧹 Cleaning up temporary files...")
        rm(temp_dir, recursive=true)
        println("✅ Cleanup complete!")
        
        println("\n🎉 Build completed successfully!")
        
    catch e
        @error "Build failed" exception=(e, catch_backtrace())
        rethrow(e)
    end
end

# Run main function
main()