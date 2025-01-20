using SHA
using Downloads
using ZipFile
using ProgressMeter
using Tar
using CodecZlib

function download_with_retry(url, output_path; max_retries=3, timeout=600)
    for attempt in 1:max_retries
        try
            println("\n📥 下载尝试 $attempt/$max_retries...")
            
            Downloads.download(
                url, 
                output_path;
                timeout=timeout,
                progress = (total, now) -> begin
                    percentage = round(now/total * 100, digits=1)
                    print("\r下载进度: $percentage% ($(now)/$(total) bytes)")
                end,
                headers=["User-Agent" => "Julia/1.11"]
            )
            println("\n✅ 下载成功!")
            return true
        catch e
            println("\n❌ 下载失败: $e")
            if attempt == max_retries
                rethrow(e)
            else
                println("等待 10 秒后重试...")
                sleep(10)
            end
        end
    end
    return false
end

function create_artifact_tarball()
    println("🚀 开始创建 artifact...")
    
    # 创建临时目录
    temp_dir = mktempdir()
    zip_path = joinpath(temp_dir, "libtorch.zip")
    artifact_dir = joinpath(temp_dir, "artifact")
    mkpath(artifact_dir)
    
    # 下载文件
    url = "https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu121.zip"
    
    if !download_with_retry(url, zip_path, timeout=600)
        error("下载失败，请检查网络连接或使用代理")
    end
    
    # 解压文件
    println("\n📦 正在解压文件...")
    rd = ZipFile.Reader(zip_path)
    
    # 计算总文件数
    total_files = length(rd.files)
    p = Progress(total_files; desc="解压进度: ", showspeed=true)
    
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
    println("✅ 解压完成!")
    
    # 创建 tar.gz
    println("\n📦 正在创建 tar.gz...")
    tarball_path = joinpath(temp_dir, "libtorch.tar.gz")
    open(tarball_path, "w") do io
        gz = GzipCompressorStream(io)
        Tar.create(artifact_dir, gz)
        close(gz)
    end
    
    # 计算 tree-sha1
    println("\n🔍 正在计算 git-tree-sha1...")
    tree_sha = bytes2hex(SHA.sha1(artifact_dir))
    
    # 计算 tarball sha256
    println("🔍 正在计算 tarball sha256...")
    tarball_sha256 = bytes2hex(open(sha256, tarball_path))
    
    println("\n✨ 计算完成!")
    println("\n最终结果:")
    println("git-tree-sha1 = \"$tree_sha\"")
    println("tarball sha256 = \"$tarball_sha256\"")
    
    # 生成 Artifacts.toml
    artifacts_toml = """
    [libtorch]
    git-tree-sha1 = "$tree_sha"
    lazy = true
    
    [[libtorch.download]]
    url = "https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu121.zip"
    sha256 = "$tarball_sha256"
    """
    
    # 保存 Artifacts.toml
    artifacts_path = joinpath(@__DIR__, "Artifacts.toml")
    write(artifacts_path, artifacts_toml)
    println("\n✅ 已生成 Artifacts.toml")
    
    # 清理
    println("\n🧹 清理临时文件...")
    rm(temp_dir, recursive=true)
    println("✅ 清理完成!")
    
    return (tree_sha, tarball_sha256)
end

try
    tree_sha, tarball_sha256 = create_artifact_tarball()
catch e
    @error "创建失败" exception=(e, catch_backtrace())
    rethrow(e)
end