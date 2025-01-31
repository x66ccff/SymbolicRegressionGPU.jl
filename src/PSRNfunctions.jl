module PSRNfunctions

using ..PSRNtharray

# Constants definition
const libtorch_dtype_dict = Dict{Int,DataType}(
    0 => UInt8,
    1 => Int8,
    2 => Int16,
    3 => Int32,
    4 => Int64,
    5 => Float16,
    6 => Float32,
    7 => Float64,
)
const libtorch_dtype_reverse_dict = Dict{DataType,Int}(
    v => k for (k, v) in libtorch_dtype_dict
)

# 脚本存储和状态管理
mutable struct ScriptManager
    scripts::Dict{Symbol,Any}
    initialized::Bool
    
    ScriptManager() = new(Dict{Symbol,Any}(), false)
end

const SCRIPT_MANAGER = ScriptManager()

const SCRIPT_GENERATORS = Dict{Symbol,String}(
    :triu_sum => """
    def main(x):
        left = x.view(1, -1, 1)
        right = x.view(1, 1, -1)
        return (left + right).view(1, -1)
    """,
    
    :triu_mul => """
    def main(x):
        left = x.view(1, -1, 1)
        right = x.view(1, 1, -1)
        return (left * right).view(1, -1)
    """,
    
    :div => """
    def main(x):
        num = x.view(1, -1, 1)
        deno = x.view(1, 1, -1)
        return (num / deno).view(1, -1)
    """,
    
    :sub => """
    def main(x):
        num = x.view(1, -1, 1)
        deno = x.view(1, 1, -1)
        return (num - deno).view(1, -1)
    """
)

# 延迟编译函数
function ensure_initialized()
    if !SCRIPT_MANAGER.initialized
        SCRIPT_MANAGER.scripts = Dict{Symbol,Any}()
        SCRIPT_MANAGER.initialized = true
    end
end

function compile_script(name::Symbol)
    ensure_initialized()
    if !haskey(SCRIPT_GENERATORS, name)
        error("Unknown script: $name")
    end
    try
        return PSRNtharray.THArrays_mod[].THJIT.compile(SCRIPT_GENERATORS[name])
    catch e
        error("Failed to compile script $name: $e")
    end
end

# 安全获取脚本
function get_script(name::Symbol)
    ensure_initialized()
    if !haskey(SCRIPT_MANAGER.scripts, name)
        SCRIPT_MANAGER.scripts[name] = compile_script(name)
    end
    SCRIPT_MANAGER.scripts[name]
end

function __init__()
    SCRIPT_MANAGER.initialized = false
end

# 使用抽象类型进行分派
abstract type AbstractTensor end

# 操作函数
function triu_sum(tensor::T) where T
    script = get_script(:triu_sum)
    return script.main(tensor)
end

function triu_mul(tensor::T) where T
    script = get_script(:triu_mul)
    return script.main(tensor)
end

function broadcast_div(tensor::T) where T
    script = get_script(:div)
    return script.main(tensor)
end

function broadcast_sub(tensor::T) where T
    script = get_script(:sub)
    return script.main(tensor)
end

# Basic unary operations
function identity_kernel!(x::T) where T
    return x
end

function neg_kernel!(x::T) where T
    return -x
end

function inv_kernel!(x::T) where T
    return 1 / x
end

function sin_kernel!(x::T) where T
    return sin(x)
end

function cos_kernel!(x::T) where T
    return cos(x)
end

function exp_kernel!(x::T) where T
    return exp(x)
end

function log_kernel!(x::T) where T
    return log(x)
end

function sqrt_kernel!(x::T) where T
    return sqrt(x)
end

function square_kernel!(x::T) where T
    return x * x
end

function cube_kernel!(x::T) where T
    return x * x * x
end


# Binary operations
function add_kernel!(x::T) where T
    return triu_sum(x)
end

function mul_kernel!(x::T) where T
    return triu_mul(x)
end

function div_kernel!(x::T) where T
    return broadcast_div(x)
end

function sub_kernel!(x::T) where T
    return broadcast_sub(x)
end

# Export functions
export identity_kernel!,
    add_kernel!,
    mul_kernel!,
    div_kernel!,
    sub_kernel!,
    neg_kernel!,
    inv_kernel!,
    sin_kernel!,
    cos_kernel!,
    exp_kernel!,
    log_kernel!,
    sqrt_kernel!,
    square_kernel!,
    cube_kernel!

end