module PSRNfunctions

using KernelAbstractions
const KA = KernelAbstractions

# 一元操作符 kernels
@kernel function identity_kernel!(out, x)
    I = @index(Global)
    @inbounds out[I] = x[I]
end

@kernel function neg_kernel!(out, x)
    I = @index(Global)
    @inbounds out[I] = -x[I]
end

@kernel function inv_kernel!(out, x)
    I = @index(Global)
    @inbounds out[I] = 1 / x[I]
end

@kernel function sin_kernel!(out, x)
    I = @index(Global)
    @inbounds out[I] = sin(x[I])
end

@kernel function cos_kernel!(out, x)
    I = @index(Global)
    @inbounds out[I] = cos(x[I])
end

@kernel function exp_kernel!(out, x)
    I = @index(Global)
    @inbounds out[I] = exp(x[I])
end

@kernel function log_kernel!(out, x)
    I = @index(Global)
    @inbounds out[I] = log(abs(x[I]))
end

# 二元操作符 kernels
@kernel function add_kernel!(out, x1, x2)
    I = @index(Global)
    @inbounds out[I] = x1[I] + x2[I]
end

@kernel function mul_kernel!(out, x1, x2)
    I = @index(Global)
    @inbounds out[I] = x1[I] * x2[I]
end

@kernel function div_kernel!(out, x1, x2)
    I = @index(Global)
    @inbounds out[I] = x1[I] / x2[I]
end

@kernel function sub_kernel!(out, x1, x2)
    I = @index(Global)
    @inbounds out[I] = x1[I] - x2[I]
end

# 添加 pow_kernel! 实现
@kernel function pow_kernel!(out, x1, x2)
    I = @index(Global)
    @inbounds out[I] = x1[I] ^ x2[I]
end

# 添加 sqrt_kernel! 实现
@kernel function sqrt_kernel!(out, x)
    I = @index(Global)
    @inbounds out[I] = sqrt(x[I])
end

# 执行函数
function execute_unary_op(kernel!, x, backend)
    # 根据后端选择合适的工作组大小
    workgroup_size = backend isa KA.CPU ? 64 : 256
    
    # 创建输出数组
    out = similar(x)
    
    # 创建并执行kernel
    kernel = kernel!(backend, workgroup_size)
    event = kernel(out, x, ndrange=size(x))
    wait(event)
    
    return out
end

function execute_binary_op(kernel!, x1, x2, backend)
    # 根据后端选择合适的工作组大小
    workgroup_size = backend isa KA.CPU ? 64 : 256
    
    # 创建输出数组
    out = similar(x1)
    
    # 创建并执行kernel
    kernel = kernel!(backend, workgroup_size)
    event = kernel(out, x1, x2, ndrange=size(x1))
    wait(event)
    
    return out
end

# 导出所有kernel和执行函数
export identity_kernel!, add_kernel!, mul_kernel!, div_kernel!, sub_kernel!,
       neg_kernel!, inv_kernel!, sin_kernel!, cos_kernel!, exp_kernel!,
       log_kernel!, pow_kernel!, sqrt_kernel!, execute_unary_op, execute_binary_op

end