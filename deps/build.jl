using Downloads

function check_command(cmd)
    try
        run(`which $cmd`)
        return true
    catch
        return false
    end
end

function main()
    # Check for required commands
    if !check_command("julia")
        error("❌ Julia is not installed or not in your PATH")
    end
    
    if !check_command("cmake")
        error("❌ CMake is not installed or not in your PATH")
    end

    # Libtorch setup
    libtorch_zip = "libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu121.zip"
    libtorch_url = "https://download.pytorch.org/libtorch/cu121/$libtorch_zip"
    
    # Download libtorch if needed
    if isfile(libtorch_zip)
        println("libtorch zip file already exists")
    else
        println("📦 Downloading libtorch...")
        try
            Downloads.download(libtorch_url, libtorch_zip)
        catch e
            @error "❌ Download failed. Please manually download from: $libtorch_url"
            rethrow(e)
        end
    end

    # Unzip libtorch
    if isdir("libtorch")
        println("libtorch directory already exists")
    else
        println("Unzipping libtorch...")
        run(`unzip $libtorch_zip`)
    end

    # Copy to src/THArrays/csrc
    tharray_csrc = joinpath(@__DIR__, "..", "src", "THArrays", "csrc")
    mkpath(tharray_csrc)
    println("Copying libtorch to src/THArrays/csrc...")
    cp("libtorch", joinpath(tharray_csrc, "libtorch"), force=true)

    # Set environment variables
    ENV["THARRAYS_DEV"] = "1"
    ENV["CUDAARCHS"] = "native"

    println("🎉 Build complete!")
end

try
    main()
catch e
    @error "Build failed" exception=(e, catch_backtrace())
    rethrow(e)
end