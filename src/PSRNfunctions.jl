module PSRNfunctions

function __init__()
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