module PSRNtharray

# 延迟加载的模块和导出
const THArrays_mod = Ref{Module}()

# 在运行时初始化
function __init__()
    # 使用相对于当前文件的路径
    include(joinpath(@__DIR__, "THArrays/src/THArrays.jl"))
    THArrays_mod[] = THArrays
    
    # 直接从 THArrays 模块重新导出符号
    # 由于 THArrays 已经通过 include 加载，我们可以直接导出它的符号
    eval(quote
        export TorchNumber, Tensor, Scalar, eltype_id,
            THC, THAD, THJIT,
            Device, CPU, CUDA, to, on
    end)
    
    # 将所需的符号导入到当前模块
    for name in [:TorchNumber, :Tensor, :Scalar, :eltype_id,
                 :THC, :THAD, :THJIT,
                 :Device, :CPU, :CUDA, :to, :on]
        @eval $name = $(THArrays_mod[]).$name
    end
end

end