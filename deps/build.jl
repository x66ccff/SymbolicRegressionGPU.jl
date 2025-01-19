using Pkg.Artifacts

function main()
    # 检查命令
    for cmd in ["julia", "cmake"]
        try
            run(`which $cmd`)
        catch
            error("❌ $cmd is not installed or not in your PATH")
        end
    end

    # 使用 artifact 系统下载 libtorch
    artifact_toml = joinpath(@__DIR__, "..", "Artifacts.toml")
    libtorch_hash = artifact_hash("libtorch", artifact_toml)
    
    if libtorch_hash === nothing || !artifact_exists(libtorch_hash)
        println("📦 Downloading libtorch...")
        libtorch_hash = create_artifact() do artifact_dir
            download_artifact = Downloads.download(
                "https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu121.zip",
                joinpath(artifact_dir, "libtorch.zip")
            )
            run(`unzip $download_artifact -d $artifact_dir`)
        end
        
        bind_artifact!(artifact_toml, "libtorch", libtorch_hash)
    end

    # 复制到目标目录
    libtorch_path = artifact_path(libtorch_hash)
    tharray_csrc = joinpath(@__DIR__, "..", "src", "THArrays", "csrc")
    mkpath(tharray_csrc)
    cp(libtorch_path, joinpath(tharray_csrc, "libtorch"), force=true)

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