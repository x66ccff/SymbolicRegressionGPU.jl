module PSRNfunctions

using THArrays

# Constants definition
const libtorch_dtype_dict = Dict{Int, DataType}(
    0 => UInt8, 1 => Int8, 2 => Int16, 3 => Int32, 4 => Int64,
    5 => Float16, 6 => Float32, 7 => Float64
)
const libtorch_dtype_reverse_dict = Dict{DataType, Int}(v => k for (k, v) in libtorch_dtype_dict)


function generate_triu_sum_script()
    return """
    def main(x):
        in_dim = x.size(1)
        indices = torch.triu_indices(in_dim, in_dim, offset=0)
        return x[:, indices[0]] + x[:, indices[1]]
    """
end

# 只需要编译一次
const TRIU_SUM_SCRIPT = THJIT.compile(generate_triu_sum_script())

function triu_sum(tensor::Tensor)
    return TRIU_SUM_SCRIPT.main(tensor)
end

function generate_triu_mul_script()
    return """
    def main(x):
        in_dim = x.size(1)
        indices = torch.triu_indices(in_dim, in_dim, offset=0)
        return x[:, indices[0]] * x[:, indices[1]]
    """
end

# 只需要编译一次
const TRIU_MUL_SCRIPT = THJIT.compile(generate_triu_mul_script())

function triu_mul(tensor::Tensor)
    return TRIU_MUL_SCRIPT.main(tensor)
end


function generate_div_script()
    return """
    def main(x):
        num = x.view(1, -1, 1)
        deno = x.view(1, 1, -1)
        return (num / deno).view(1, -1)
    """
end

function generate_sub_script()
    return """
    def main(x):
        num = x.view(1, -1, 1)
        deno = x.view(1, 1, -1)
        return (num - deno).view(1, -1)
    """
end

# 编译两个脚本
const DIV_SCRIPT = THJIT.compile(generate_div_script())
const SUB_SCRIPT = THJIT.compile(generate_sub_script())

function broadcast_div(tensor::Tensor)
    return DIV_SCRIPT.main(tensor)
end

function broadcast_sub(tensor::Tensor)
    return SUB_SCRIPT.main(tensor)
end

# Basic unary operations
function identity_kernel!(x::Tensor)
    return x
end

function neg_kernel!(x::Tensor)
    return -x
end

function inv_kernel!(x::Tensor)
    return 1/x
end

function sin_kernel!(x::Tensor)
    return sin(x)
end

function cos_kernel!(x::Tensor)
    return cos(x)
end

function exp_kernel!(x::Tensor)
    return exp(x)
end

function log_kernel!(x::Tensor)
    return log(x)
end

function sqrt_kernel!(x::Tensor)
    return sqrt(x)
end

# Binary operations
function add_kernel!(x::Tensor)
    return triu_sum(x)
end

function mul_kernel!(x::Tensor)
    return triu_mul(x)
end

function div_kernel!(x::Tensor)
    batch_size = size(x, 1)
    in_dim = size(x, 2)
    
    # num: (batch_size, in_dim, 1)
    num = THC._reshape_alias(x, 
        [batch_size, in_dim, 1], 
        [in_dim, 1, 1])
    
    # deno: (batch_size, 1, in_dim)
    deno = THC._reshape_alias(x, 
        [batch_size, 1, in_dim], 
        [in_dim, in_dim, 1])
    
    # out_temp: (batch_size, in_dim * in_dim)
    out = THC._reshape_alias(num ./ deno, 
        [batch_size, in_dim * in_dim],
        [in_dim * in_dim, 1])
    
    return out
end

function sub_kernel!(x::Tensor)
    batch_size = size(x, 1)
    in_dim = size(x, 2)
    # num: (batch_size, in_dim, 1)
    num = THC._reshape_alias(x, 
        Int64[batch_size, in_dim, 1], 
        Int64[in_dim, 1, 1])
    
    # deno: (batch_size, 1, in_dim)
    deno = THC._reshape_alias(x, 
        Int64[batch_size, 1, in_dim], 
        Int64[in_dim, in_dim, 1])
    
    diff = num .- deno
    # out_temp: (batch_size, in_dim * in_dim)
    out = THC._reshape_alias(diff, 
        Int64[batch_size, in_dim * in_dim],
        Int64[in_dim * in_dim, 1])
    
    return out
end

# Export functions
export identity_kernel!, add_kernel!, mul_kernel!, div_kernel!, sub_kernel!,
       neg_kernel!, inv_kernel!, sin_kernel!, cos_kernel!, exp_kernel!,
       log_kernel!, sqrt_kernel!
end