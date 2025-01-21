using SHA
using Downloads
using ZipFile
using ProgressMeter

function compute_tree_sha1()
    println("🚀 开始计算 git-tree-sha1...")
    
    # 创建临时目录
    temp_dir = mktempdir()
    zip_path = joinpath(temp_dir, "libtorch.zip")
    
    # 下载文件
    url = "https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu121.zip"
    println("\n📥 正在下载 libtorch...")
    
    Downloads.download(
        url, 
        zip_path;
        progress = (total, now) -> begin
            percentage = round(now/total * 100, digits=1)
            print("\r下载进度: $percentage% ($(now)/$(total) bytes)")
        end
    )
    println("\n✅ 下载完成!")
    
    # 解压文件
    println("\n📦 正在解压文件...")
    extract_dir = joinpath(temp_dir, "libtorch")
    mkpath(extract_dir)
    rd = ZipFile.Reader(zip_path)
    
    # 计算总文件数
    total_files = length(rd.files)
    p = Progress(total_files; desc="解压进度: ", showspeed=true)
    
    for f in rd.files
        fullpath = joinpath(extract_dir, f.name)
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
    
    # 计算 git-tree-sha1
    println("\n🔍 正在计算 git-tree-sha1...")
    tree_sha = bytes2hex(SHA.sha1(extract_dir))
    println("\n✨ 计算完成!")
    println("\n最终结果:")
    println("git-tree-sha1 = \"$tree_sha\"")
    
    # 清理
    println("\n🧹 清理临时文件...")
    rm(temp_dir, recursive=true)
    println("✅ 清理完成!")
    
    return tree_sha
end

tree_sha = compute_tree_sha1()