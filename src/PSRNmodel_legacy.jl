module PSRNmodel

using ..PSRNfunctions
import ..CoreModule.OperatorsModule: plus, sub, mult, square, cube, safe_pow, safe_log,
    safe_log2, safe_log10, safe_sqrt, safe_acosh, neg, greater,
    cond, relu, logical_or, logical_and, gamma

import ..CoreModule: Options, Dataset 


using CUDA


using Printf: @sprintf  
using DynamicExpressions: Node, Expression



using THArrays

# Operator abstractions
abstract type Operator end

struct UnaryOperator <: Operator
    name::String
    kernel::Function
    is_directed::Bool
end

struct BinaryOperator <: Operator
    name::String
    kernel::Function
    is_directed::Bool
end

# Operator dictionary
const OPERATORS = Dict{String, Operator}(
    "Identity" => UnaryOperator("Identity", identity_kernel!, true),
    "Sin" => UnaryOperator("Sin", sin_kernel!, true),
    "Cos" => UnaryOperator("Cos", cos_kernel!, true),
    "Exp" => UnaryOperator("Exp", exp_kernel!, true),
    "Log" => UnaryOperator("Log", log_kernel!, true),
    "Neg" => UnaryOperator("Neg", neg_kernel!, true),
    "Add" => BinaryOperator("Add", add_kernel!, false),
    "Mul" => BinaryOperator("Mul", mul_kernel!, false),
    "Div" => BinaryOperator("Div", div_kernel!, true),
    "Sub" => BinaryOperator("Sub", sub_kernel!, true)
)

# DRLayer implementation
mutable struct DRLayer
    in_dim::Int
    out_dim::Int
    dr_indices::Vector{Int}
    dr_mask::Vector{Bool}

    function DRLayer(in_dim::Int, dr_mask::Vector{Bool})
        out_dim = sum(dr_mask)
        dr_indices = findall(dr_mask)
        new(in_dim, out_dim, dr_indices, dr_mask)
    end
end

function forward(layer::DRLayer, x::Tensor)
    return x[:, layer.dr_mask]
end

function get_op_and_offset(layer::DRLayer, index::Int)
    return layer.dr_indices[index]
end

# SymbolLayer implementation
mutable struct SymbolLayer
    in_dim::Int
    out_dim::Int
    operators::Vector{Operator}
    n_binary_U::Int  # undirected (+ *)
    n_binary_D::Int  # directed (/ -)
    n_unary::Int
    operator_list::Vector{Operator}
    n_triu::Int
    in_dim_square::Int
    out_dim_cum_ls::Union{Vector{Int}, Nothing}
    offset_tensor::Union{Matrix{Int}, Nothing}

    function SymbolLayer(in_dim::Int, operator_names::Vector{String})
        n_binary_U = 0
        n_binary_D = 0
        n_unary = 0
        operator_list = Operator[]
        operators = [OPERATORS[name] for name in operator_names]
        
        n_triu = (in_dim * (in_dim + 1)) ÷ 2
        in_dim_square = in_dim * in_dim

        # Count operators
        for op in operators
            if op isa BinaryOperator
                if op.is_directed
                    n_binary_D += 1
                else
                    n_binary_U += 1
                end
            else
                n_unary += 1
            end
        end

        # Add operators in order
        # 1. Undirected binary operators
        for op in operators
            if op isa BinaryOperator && !op.is_directed
                push!(operator_list, op)
            end
        end
        
        # 2. Directed binary operators
        for op in operators
            if op isa BinaryOperator && op.is_directed
                push!(operator_list, op)
            end
        end
        
        # 3. Unary operators
        for op in operators
            if op isa UnaryOperator
                push!(operator_list, op)
            end
        end

        out_dim = n_unary * in_dim + n_binary_U * n_triu + n_binary_D * in_dim_square

        new(in_dim, out_dim, operators, n_binary_U, n_binary_D, n_unary,
            operator_list, n_triu, in_dim_square, nothing, nothing)
    end
end

function get_out_dim_cum_ls(layer::SymbolLayer)
    if layer.out_dim_cum_ls !== nothing
        return layer.out_dim_cum_ls
    end

    out_dim_ls = Int[]
    for func in layer.operator_list
        if func isa UnaryOperator
            push!(out_dim_ls, layer.in_dim)
        else
            if func.is_directed
                push!(out_dim_ls, layer.in_dim_square)
            else
                push!(out_dim_ls, layer.n_triu)
            end
        end
    end
    
    layer.out_dim_cum_ls = [sum(out_dim_ls[1:i]) for i in 1:length(out_dim_ls)]
    return layer.out_dim_cum_ls
end

function get_offset_tensor(layer::SymbolLayer)
    offset_tensor = zeros(Int, layer.out_dim, 2)
    arange_tensor = collect(1:layer.in_dim)
    
    binary_U_tensor = zeros(Int, layer.n_triu, 2)
    binary_D_tensor = zeros(Int, layer.in_dim_square, 2)
    unary_tensor = zeros(Int, layer.in_dim, 2)
    
    unary_tensor[:, 1] = arange_tensor
    unary_tensor[:, 2] .= layer.in_dim
    
    # Fill binary_U_tensor (index of undirected binary operation)
    start = 1
    for i in 1:layer.in_dim
        len = layer.in_dim - i + 1
        binary_U_tensor[start:start+len-1, 1] .= i
        binary_U_tensor[start:start+len-1, 2] = i:layer.in_dim
        start += len
    end
    
    # Fill binary_D_tensor (index of directed binary operation)
    start = 1
    for i in 1:layer.in_dim
        len = layer.in_dim
        binary_D_tensor[start:start+len-1, 1] .= i
        binary_D_tensor[start:start+len-1, 2] = 1:layer.in_dim
        start += len
    end
    
    # Combine all indexes
    start = 1
    for func in layer.operator_list
        if func isa UnaryOperator
            t = unary_tensor
        else
            t = func.is_directed ? binary_D_tensor : binary_U_tensor
        end
        len = size(t, 1)
        offset_tensor[start:start+len-1, :] = t
        start += len
    end
    
    return offset_tensor
end

function init_offset(layer::SymbolLayer)
    layer.offset_tensor = get_offset_tensor(layer)
end

function get_op_and_offset(layer::SymbolLayer, index::Int)
    out_dim_cum_ls = get_out_dim_cum_ls(layer)
    for i in eachindex(out_dim_cum_ls)
        if index < out_dim_cum_ls[i]
            return layer.operator_list[i], layer.offset_tensor[index, :]
        end
    end
    return layer.operator_list[end], layer.offset_tensor[index, :]
end

function forward(layer::SymbolLayer, x::Tensor)
    results = Tensor[]
    for op in layer.operator_list
        if op isa UnaryOperator
            result = execute_unary_op(op.kernel, x)
            push!(results, result)
        else # BinaryOperator
            result = execute_binary_op(op.kernel, x)
            push!(results, result)
        end
    end
    return cat(results..., dims=2)
end

# PSRN implementation
mutable struct PSRN
    n_variables::Int
    operators::Vector{String}
    n_symbol_layers::Int
    layers::Vector{Union{SymbolLayer, DRLayer}}
    out_dim::Int
    use_dr_mask::Bool
    current_expr_ls::Vector{Any}

    function PSRN(;
        n_variables::Int=1,
        operators::Vector{String}=["Add", "Mul", "Identity", "Sin", "Exp", "Neg"],
        n_symbol_layers::Int=3,
        dr_mask::Union{Vector{Bool}, Nothing}=nothing
    )
        layers = Union{SymbolLayer, DRLayer}[]
        use_dr_mask = !isnothing(dr_mask)

        for i in 1:n_symbol_layers
            if use_dr_mask && i == n_symbol_layers
                push!(layers, DRLayer(layers[end].out_dim, dr_mask))
            end

            if i == 1
                push!(layers, SymbolLayer(n_variables, operators))
            else
                push!(layers, SymbolLayer(layers[end].out_dim, operators))
            end
        end

        new(n_variables, operators, n_symbol_layers, layers, 
            layers[end].out_dim, use_dr_mask, [])
    end
end

function forward(model::PSRN, x::Tensor)
    h = x
    for layer in model.layers
        h = forward(layer, h)
    end
    return h
end

function find_best_indices(outputs::Tensor, y::Tensor; top_k::Int=100)
    n_samples = size(outputs, 1)
    n_expressions = size(outputs, 2)
    
    # Calculate mean squared errors
    sum_squared_errors = sum((outputs .- y).^2, dims=1)
    mean_squared_errors = sum_squared_errors ./ n_samples
    
    # Handle invalid values
    mean_squared_errors = Array(mean_squared_errors)
    mean_squared_errors[isnan.(mean_squared_errors)] .= Inf32
    mean_squared_errors[isinf.(mean_squared_errors)] .= Inf32
    
    # Find indices of top_k smallest MSEs
    sorted_indices = partialsortperm(vec(mean_squared_errors), 1:min(top_k, length(mean_squared_errors)))
    
    return sorted_indices, mean_squared_errors[sorted_indices]
end

 
function _get_expr(model::PSRN, index::Int, layer_idx::Int)
    if layer_idx < 1
        return model.current_expr_ls[index]
    end

    layer = model.layers[layer_idx]
    
    if layer isa DRLayer
        new_index = get_op_and_offset(layer, index)
        return _get_expr(model, new_index, layer_idx-1)
    else
        op, offsets = get_op_and_offset(layer, index)
        if op isa UnaryOperator
            expr1 = _get_expr(model, offsets[1], layer_idx-1)
            return op.kernel(expr1)
        else
            expr1 = _get_expr(model, offsets[1], layer_idx-1)
            expr2 = _get_expr(model, offsets[2], layer_idx-1)
            return op.kernel(expr1, expr2)
        end
    end
end


function Base.show(io::IO, model::PSRN)
    print(io, "PSRN(n_variables=$(model.n_variables), operators=$(model.operators), " *
              "n_layers=$(model.n_symbol_layers))\n")
    print(io, "Layer dimensions: ")
    print(io, join([layer.out_dim for layer in model.layers], " → "))
end

# Export types and functions
export PSRN, SymbolLayer, DRLayer, forward, Operator, UnaryOperator, BinaryOperator,
       find_best_indices, get_expr, get_op_and_offset

end