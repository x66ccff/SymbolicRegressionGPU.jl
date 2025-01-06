module PSRNfunctions


import ..CoreModule.OperatorsModule:
    plus,
    sub,
    mult,
    square,
    cube,
    safe_pow,
    safe_log,
    safe_log2,
    safe_log10,
    safe_sqrt,
    safe_acosh,
    neg,
    greater,
    cond,
    relu,
    logical_or,
    logical_and,
    gamma

using Reactant

const T_kernel_compiling = Float32  # Default Float32
const T_kernel_compiling_idx = Int64  # Default Int64

function apply_unary(f::Function, x::AbstractMatrix)
    return f.(x)
end

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
    return safe_log.(x)
end

function sqrt_kernel!(x::AbstractMatrix)
    return safe_sqrt.(x)
end


function get_triu_indices(n::Int)
    num_indices = n * (n + 1) ÷ 2
    
    indices_matrix = Matrix{T_kernel_compiling_idx}(undef, num_indices, 2)
    
    idx = 1
    for i in 1:n
        for j in i:n
            indices_matrix[idx, 1] = i
            indices_matrix[idx, 2] = j
            idx += 1
        end
    end
    
    return collect(transpose(indices_matrix))
end

function get_squared_indices(n::Int)
    num_indices = n * n
    
    indices_matrix = Matrix{T_kernel_compiling_idx}(undef, num_indices, 2)
    
    idx = 1
    for i in 1:n
        for j in 1:n
            indices_matrix[idx, 1] = i
            indices_matrix[idx, 2] = j
            idx += 1
        end
    end
    
    return collect(transpose(indices_matrix))
end

function add_kernel!(x::AbstractMatrix, n::Int, indices::AbstractMatrix)
    l_idx = indices[1, :]
    r_idx = indices[2, :]
    l_value = x[l_idx]'
    r_value = x[r_idx]'
    res = l_value .+ r_value
    return res
end

function mul_kernel!(x::AbstractMatrix, n::Int, indices::AbstractMatrix)
    l_idx = indices[1, :]
    r_idx = indices[2, :]
    l_value = x[l_idx]'
    r_value = x[r_idx]'
    res = l_value .* r_value
    return res
end

function sub_kernel!(x::AbstractMatrix, n::Int, indices::AbstractMatrix)
    l_idx = indices[1, :]
    r_idx = indices[2, :]
    l_value = x[:, l_idx]
    r_value = x[:, r_idx]
    res = l_value .- r_value
    return res
end

function div_kernel!(x::AbstractMatrix, n::Int, indices::AbstractMatrix)
    l_idx = indices[1, :]
    r_idx = indices[2, :]
    l_value = x[:, l_idx]
    r_value = x[:, r_idx]
    res = l_value ./ r_value
    return res
end




function compile_unary_kernel(input_dim::Int, func::Function)
    input_array = Reactant.ConcreteRArray(ones(T_kernel_compiling, 1, input_dim))
    
    compiled_func = @compile func(input_array)
    
    return compiled_func
end

function compile_binary_triu_kernel(input_dim::Int, func::Function)
    n = input_dim

    x = rand(T_kernel_compiling, 1, n)
    xr = Reactant.to_rarray(x)

    triu = get_triu_indices(n) 
    triur = Reactant.to_rarray(triu)

    @info size(xr)
    @info typeof(xr)
    @info n
    @info size(triur)
    @info typeof(triur)

    compiled_func = @compile func(xr, n, triur)

    @info "compiled_func's name: \n\t\t $compiled_func"
    
    return compiled_func
end

function compile_binary_squared_kernel(input_dim::Int, func::Function)
    n = input_dim

    x = rand(T_kernel_compiling, 1, n)
    xr = Reactant.to_rarray(x)

    squared = get_squared_indices(n) 
    squaredr = Reactant.to_rarray(squared)

    compiled_func = @compile func(xr, n, squaredr)
    
    return compiled_func
end

export compile_unary_kernel, compile_binary_triu_kernel, compile_binary_squared_kernel
export get_triu_indices, get_squared_indices

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


end # module