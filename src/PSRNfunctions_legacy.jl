module PSRNfunctions

using KernelAbstractions
const KA = KernelAbstractions

# Generator for unary operator kernels
function generate_unary_kernel(op::F, name::Symbol) where {F<:Function}
    @eval begin
        @kernel function $name(out, x)
            I = @index(Global)
            @inbounds out[I] = $op(x[I])
        end
    end
end

# Generator for binary operator kernels
function generate_binary_kernel(op::F, name::Symbol) where {F<:Function}
    @eval begin
        @kernel function $name(out, x1, x2)
            I = @index(Global)
            @inbounds out[I] = $op(x1[I], x2[I])
        end
    end
end

# Define kernels using generators
generate_unary_kernel(identity, :identity_kernel!)
generate_unary_kernel(-, :neg_kernel!)
generate_unary_kernel(x -> 1/x, :inv_kernel!)
generate_unary_kernel(sin, :sin_kernel!)
generate_unary_kernel(cos, :cos_kernel!)
generate_unary_kernel(exp, :exp_kernel!)
generate_unary_kernel(x -> log(abs(x)), :log_kernel!)
generate_unary_kernel(sqrt, :sqrt_kernel!)

generate_binary_kernel(+, :add_kernel!)
generate_binary_kernel(*, :mul_kernel!)
generate_binary_kernel(/, :div_kernel!)
generate_binary_kernel(-, :sub_kernel!)
generate_binary_kernel(^, :pow_kernel!)

# 执行函数保持不变
function execute_unary_op(kernel!, x, backend)
    workgroup_size = backend isa KA.CPU ? 64 : 256
    out = similar(x)
    kernel = kernel!(backend, workgroup_size)
    event = kernel(out, x, ndrange=size(x))
    wait(event)
    return out
end

function execute_binary_op(kernel!, x1, x2, backend)
    workgroup_size = backend isa KA.CPU ? 64 : 256
    out = similar(x1)
    kernel = kernel!(backend, workgroup_size)
    event = kernel(out, x1, x2, ndrange=size(x1))
    wait(event)
    return out
end

export identity_kernel!, add_kernel!, mul_kernel!, div_kernel!, sub_kernel!,
       neg_kernel!, inv_kernel!, sin_kernel!, cos_kernel!, exp_kernel!,
       log_kernel!, pow_kernel!, sqrt_kernel!, execute_unary_op, execute_binary_op

end