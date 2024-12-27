module PSRNfunctions

# 一元操作的辅助函数
function apply_unary(f::Function, x::AbstractMatrix)
    return f.(x)
end

# 生成上三角矩阵的索引
function get_triu_indices(n::Int)
    indices = Tuple{Int,Int}[]
    for i in 1:n
        for j in i:n
            push!(indices, (i,j))
        end
    end
    return indices
end

# 基本一元运算函数
function identity_kernel!(x::AbstractMatrix)
    return copy(x)
end

function neg_kernel!(x::AbstractMatrix)
    return -x
end

function inv_kernel!(x::AbstractMatrix)
    return 1.0 ./ x
end

function sin_kernel!(x::AbstractMatrix)
    return sin.(x)
end

function cos_kernel!(x::AbstractMatrix)
    return cos.(x)
end

function exp_kernel!(x::AbstractMatrix)
    return exp.(x)
end

function log_kernel!(x::AbstractMatrix)
    return log.(x)
end

function sqrt_kernel!(x::AbstractMatrix)
    return sqrt.(x)
end

# 二元运算函数

# 上三角加法: (n) -> (n*(n+1)/2)
function add_kernel!(x::AbstractMatrix)
    n = size(x, 2)
    indices = get_triu_indices(n)
    result = zeros(eltype(x), size(x, 1), length(indices))
    
    for (idx, (i, j)) in enumerate(indices)
        result[:, idx] = x[:, i] + x[:, j]
    end
    
    return result
end

# 上三角乘法: (n) -> (n*(n+1)/2)
function mul_kernel!(x::AbstractMatrix)
    n = size(x, 2)
    indices = get_triu_indices(n)
    result = zeros(eltype(x), size(x, 1), length(indices))
    
    for (idx, (i, j)) in enumerate(indices)
        result[:, idx] = x[:, i] .* x[:, j]
    end
    
    return result
end

# 广播除法: (n) -> (n*n)
function div_kernel!(x::AbstractMatrix)
    n = size(x, 2)
    result = zeros(eltype(x), size(x, 1), n * n)
    
    for i in 1:n
        for j in 1:n
            idx = (i-1)*n + j
            result[:, idx] = x[:, i] ./ x[:, j]
        end
    end
    
    return result
end

# 广播减法: (n) -> (n*n)
function sub_kernel!(x::AbstractMatrix)
    n = size(x, 2)
    result = zeros(eltype(x), size(x, 1), n * n)
    
    for i in 1:n
        for j in 1:n
            idx = (i-1)*n + j
            result[:, idx] = x[:, i] .- x[:, j]
        end
    end
    
    return result
end

# 导出所有函数
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
    sqrt_kernel!
end
