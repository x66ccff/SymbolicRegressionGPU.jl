using SHA
using Downloads
using ZipFile
using ProgressMeter
using Tar
using CodecZlib
using Pkg

function check_dependencies()
    println("🔍 检查依赖...")
    
    # 检查CMake
    try
        run(`cmake --version`)
        println("✅ CMake 已安装")
    catch
        error("❌ 未找到CMake。请安装CMake后重试。")
    end
end

function download_with_retry(url, output_path; max_retries=3, timeout=600)
    for attempt in 1:max_retries
        try
            println("\n📥 Downloading libtorch $attempt/$max_retries...")
            
            # Create a progress bar without starting it
            prog = Progress(100; 
                dt=0.1, 
                barglyphs=BarGlyphs("[=> ]"), 
                barlen=50, 
                color=:cyan,
                showspeed=true
            )
            
            let last_percentage = 0
                Downloads.download(
                    url,
                    output_path;
                    timeout=timeout,
                    progress = (total, now) -> begin
                        percentage = round(Int, now/total * 100)
                        if percentage > last_percentage
                            # Update progress bar
                            update!(prog, percentage; showvalues = [
                                (:Size, "$(total) bytes"),
                                (:Downloaded, "$(now) bytes")
                            ])
                            last_percentage = percentage
                        end
                    end,
                    headers=["User-Agent" => "Julia/1.11"]
                )
            end
            
            finish!(prog)
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

function setup_libtorch(temp_dir)
    println("\n📦 设置 libtorch...")
    
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
    
    return joinpath(artifact_dir, "libtorch")  # 返回解压后的libtorch目录路径
end


function compile_tharray(libtorch_path)
    println("\n🔧 编译 THArrays...")
    
    # 设置环境变量
    ENV["THARRAYS_DEV"] = "1"
    ENV["CUDAARCHS"] = "native"
    
    # 获取正确的目录路径
    # 从deps/build.jl向上两级得到项目根目录
    root_dir = dirname(dirname(@__FILE__))
    
    # THArrays在主包的src下
    csrc_dir = joinpath(root_dir, "src", "THArrays", "csrc")
    println("检查目录: $csrc_dir")
    
    if !isfile(joinpath(csrc_dir, "CMakeLists.txt"))
        error("在 $csrc_dir 中未找到 CMakeLists.txt 文件")
    end
    
    # 复制libtorch到csrc目录
    println("复制 libtorch 到 $csrc_dir")
    cp(libtorch_path, joinpath(csrc_dir, "libtorch"), force=true)
    
    # 创建build目录
    build_dir = joinpath(csrc_dir, "build")
    mkpath(build_dir)
    
    # 运行CMake
    cd(build_dir) do
        println("在目录 $(pwd()) 中运行 CMake")
        println("源代码目录: $csrc_dir")
        run(`cmake $csrc_dir`)
        run(`cmake --build . --config Release`)
    end
    
    println("✅ 编译完成!")
end

function main()
    try
        # 检查依赖
        check_dependencies()
        
        # 创建临时目录
        temp_dir = mktempdir()
        
        # 设置libtorch
        libtorch_path = setup_libtorch(temp_dir)
        
        # 编译THArrays
        compile_tharray(libtorch_path)
        
        # 清理
        println("\n🧹 清理临时文件...")
        rm(temp_dir, recursive=true)
        println("✅ 清理完成!")
        
        println("\n🎉 构建成功完成!")
        
    catch e
        @error "构建失败" exception=(e, catch_backtrace())
        rethrow(e)
    end
end

# 运行主函数
main()