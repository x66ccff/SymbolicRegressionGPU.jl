module PSRNmodel

using ..PSRNfunctions
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

import ..CoreModule: Options, Dataset

using Printf: @sprintf
using DynamicExpressions: Node, Expression

using ..PSRNtharray

# 使用全局变量
global TensorType = Any

# 首先定义基础类型
abstract type Operator end

struct UnaryOperator <: Operator
    name::String
    kernel::Function
    is_directed::Bool
    expr_gen::Function
end

struct BinaryOperator <: Operator
    name::String
    kernel::Function
    is_directed::Bool
    expr_gen::Function
end

# 然后定义操作字典
const OPERATORS = Dict{String,Operator}(
    "Identity" => UnaryOperator("Identity", identity_kernel!, true, identity),
    "Sin" => UnaryOperator("Sin", sin_kernel!, true, sin),
    "Cos" => UnaryOperator("Cos", cos_kernel!, true, cos),
    "Exp" => UnaryOperator("Exp", exp_kernel!, true, exp),
    "Log" => UnaryOperator("Log", log_kernel!, true, safe_log),
    "Sqrt" => UnaryOperator("Sqrt", sqrt_kernel!, true, safe_sqrt),
    "Add" => BinaryOperator("Add", add_kernel!, true, +),
    "Mul" => BinaryOperator("Mul", mul_kernel!, true, *),
    "Div" => BinaryOperator("Div", div_kernel!, true, /),
    "Sub" => BinaryOperator("Sub", sub_kernel!, true, -),
    "Inv" => UnaryOperator("Inv", inv_kernel!, true, x -> 1 / x),
    "Neg" => UnaryOperator("Neg", neg_kernel!, true, x -> 0 - x),
)

# Script management
mutable struct CompilationManager
    cat_units::Dict{Int,Any}
    topk_units::Dict{Int,Any}
    initialized::Bool
    
    CompilationManager() = new(Dict{Int,Any}(), Dict{Int,Any}(), false)
end

const COMPILATION_MANAGER = CompilationManager()

# Script generation functions
function generate_cat_script(n::Int)
    args = join(('a':'z')[1:n], ", ")
    tensors = "(" * join(('a':'z')[1:n], ", ")  * ")"
    return """
    def main($args):
        result = torch.cat($tensors, dim=1)
        return result
    """
end

function get_cat_unit(n::Int)
    if !haskey(COMPILATION_MANAGER.cat_units, n)
        script = generate_cat_script(n)
        COMPILATION_MANAGER.cat_units[n] = PSRNtharray.THArrays_mod[].THJIT.compile(script)
    end
    return COMPILATION_MANAGER.cat_units[n]
end

function concat_tensors(tensors::Vector{T}) where T
    n = length(tensors)
    if n == 0
        error("Cannot concatenate empty tensor list")
    elseif n == 1
        return tensors[1]
    elseif n > 26
        error("Maximum 26 tensors supported")
    end

    unit = get_cat_unit(n)
    return unit.main(tensors...)
end

# 在 concat_tensors 函数之后添加以下代码

function generate_topk_script(k::Int)
    return """
    def main(tensor):
        _, indices = torch.topk(-tensor, k=$k, dim=1)  # 使用负号来获取最小值
        return indices
    """
end

function get_topk_unit(k::Int)
    if !haskey(COMPILATION_MANAGER.topk_units, k)
        script = generate_topk_script(k)
        COMPILATION_MANAGER.topk_units[k] = PSRNtharray.THArrays_mod[].THJIT.compile(script)
    end
    return COMPILATION_MANAGER.topk_units[k]
end

function topk_indices(tensor::T, k::Int) where T
    unit = get_topk_unit(k)
    return unit.main(tensor)
end

# 之后定义其他结构和函数
mutable struct DRLayer
    in_dim::Int
    out_dim::Int
    dr_indices::TensorType
    dr_mask::TensorType
    device::Int

    function DRLayer(in_dim::Int, dr_mask::Vector{Bool}, device::Int)
        out_dim = sum(dr_mask)
        arange_tensor = collect(0:(length(dr_mask) - 1))
        dr_indices = PSRNtharray.THArrays_mod[].Tensor(arange_tensor[dr_mask])
        dr_mask_tensor = PSRNtharray.THArrays_mod[].Tensor(dr_mask)
        dr_indices = to(dr_indices, CUDA(device))
        dr_mask_tensor = to(dr_mask_tensor, CUDA(device))
        return new(in_dim, out_dim, dr_indices, dr_mask_tensor, device)
    end
end

function forward(layer::DRLayer, x::TensorType)
    return x[:, layer.dr_mask]
end

function get_op_and_offset(layer::DRLayer, index::Int)
    return Int(Array(layer.dr_indices)[index + 1])
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
    out_dim_cum_ls::Union{Vector{Int},Nothing}
    offset_tensor::Union{AbstractArray,Nothing}
    device::Int

    function SymbolLayer(in_dim::Int, operator_names::Vector{String}, device::Int)
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

        layer = new(
            in_dim,
            out_dim,
            operators,
            n_binary_U,
            n_binary_D,
            n_unary,
            operator_list,
            n_triu,
            in_dim_square,
            nothing,
            nothing,
            device,
        )

        init_offset(layer)
        return layer
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
    arange_tensor = collect(0:(layer.in_dim - 1))

    binary_U_tensor = zeros(Int, layer.n_triu, 2)
    binary_D_tensor = zeros(Int, layer.in_dim_square, 2)
    unary_tensor = zeros(Int, layer.in_dim, 2)

    unary_tensor[:, 1] = arange_tensor
    unary_tensor[:, 2] .= layer.in_dim

    # Fill binary_U_tensor (index of undirected binary operation)
    start = 1
    for i in 0:(layer.in_dim - 1)
        len = layer.in_dim - i
        binary_U_tensor[start:(start + len - 1), 1] .= i
        binary_U_tensor[start:(start + len - 1), 2] = i:(layer.in_dim - 1)
        start += len
    end

    # Fill binary_D_tensor (index of directed binary operation)
    start = 1
    for i in 0:(layer.in_dim - 1)
        len = layer.in_dim
        binary_D_tensor[start:(start + len - 1), 1] .= i
        binary_D_tensor[start:(start + len - 1), 2] = 0:(layer.in_dim - 1)
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
        offset_tensor[start:(start + len - 1), :] = t
        start += len
    end
    # convert to TensorType
    # offset_tensor = TensorType(offset_tensor)
    # move to device
    # offset_tensor = to(offset_tensor, CUDA(layer.device))
    return offset_tensor
end

function init_offset(layer::SymbolLayer)
    return layer.offset_tensor = get_offset_tensor(layer)
end

function get_op_and_offset(layer::SymbolLayer, index::Int)
    out_dim_cum_ls = get_out_dim_cum_ls(layer)
    op_idx = 1
    for i in eachindex(out_dim_cum_ls)
        if index < out_dim_cum_ls[i]
            op_idx = i
            break
        end
    end
    offset = layer.offset_tensor[index + 1, :]
    return layer.operator_list[op_idx], offset
end

function forward(layer::SymbolLayer, x::TensorType)
    results = TensorType[]
    for op in layer.operator_list
        result = op.kernel(x)
        push!(results, result)
    end
    res = concat_tensors(results)
    return res
end

# PSRN implementation
mutable struct PSRN
    n_variables::Int
    operators::Vector{String}
    n_symbol_layers::Int
    layers::Vector{Union{SymbolLayer,DRLayer}}
    out_dim::Int
    use_dr_mask::Bool
    current_expr_ls::Vector{Expression}
    device::Int
    options::Options

    function PSRN(;
        n_variables::Int=1,
        operators::Vector{String}=["Add", "Mul", "Identity", "Sin", "Exp", "Neg"],
        n_symbol_layers::Int=3,
        dr_mask::Union{Vector{Bool},Nothing}=nothing,
        device::Int=0,
        options::Options=Options(),
    )
        layers = Union{SymbolLayer,DRLayer}[]
        use_dr_mask = !isnothing(dr_mask)

        for i in 1:n_symbol_layers
            if use_dr_mask && i == n_symbol_layers
                push!(layers, DRLayer(layers[end].out_dim, dr_mask, device))
            end

            if i == 1
                push!(layers, SymbolLayer(n_variables, operators, device))
            else
                push!(layers, SymbolLayer(layers[end].out_dim, operators, device))
            end
        end

        return new(
            n_variables,
            operators,
            n_symbol_layers,
            layers,
            layers[end].out_dim,
            use_dr_mask,
            [],
            device,
            options,
        )
    end
end

function PSRN_forward(model::PSRN, x::TensorType)
    h = x
    # print the device of h
    # @info "h device: $(on(x).index)"
    for layer in model.layers
        h = forward(layer, h)
    end
    return h
end

# function find_best_indices(outputs::TensorType, y::TensorType; top_k::Int=100)
#     n_samples = size(outputs, 1)
#     n_expressions = size(outputs, 2)

#     # Calculate mean squared errors
#     sum_squared_errors = sum((outputs .- y).^2, dims=1)
#     mean_squared_errors = sum_squared_errors ./ n_samples

#     # Handle invalid values
#     mean_squared_errors = Array(mean_squared_errors)
#     mean_squared_errors[isnan.(mean_squared_errors)] .= Inf32
#     mean_squared_errors[isinf.(mean_squared_errors)] .= Inf32

#     # Find indices of top_k smallest MSEs
#     sorted_indices = partialsortperm(vec(mean_squared_errors), 1:min(top_k, length(mean_squared_errors)))

#     return sorted_indices, mean_squared_errors[sorted_indices]
# end

function get_best_expr_and_MSE_topk(
    model::PSRN,
    # X::TensorType{Float32,2}, #TODO 怎么好像变成 Pytorch.Tensor了？？是因为.so的设置不正确吗？改成Any之后就好了
    # X::TensorType{Float32,2},
    X::Any,
    Y::Vector{Float32},
    n_top::Int,
    device_id::Int,
)
    # Calculate MSE for all expressions
    batch_size = size(X, 1)
    Y = Float32.(Y) # for saving memory
    # sum_squared_errors = TensorType(zeros((1, model.out_dim)))

    @time sum_squared_errors = TensorType(zeros(Float32, (1, model.out_dim))) # for saving memory

    @time sum_squared_errors = to(sum_squared_errors, CUDA(device_id))

    # Compute sum of squared errors
    for i in 1:batch_size
        x_sliced = X[i:i, :]

        x_sliced = to(x_sliced, CUDA(device_id))

        H = PSRN_forward(model, x_sliced) #   0.150774 seconds

        diff = H .- Y[i]

        square = diff * diff # don't use ^2, because it will get Float64

        sum_squared_errors += square
    end

    # Calculate mean
    mean_errors = sum_squared_errors ./ batch_size
    # mean_errors = reshape(mean_errors, :)
    # @info "mean_errors shape: $(size(mean_errors))"

    # Get top-k indices and values using THC
    # values, indices = THC.topk(mean_errors, n_top, largest=false, sorted=true)
    indices = topk_indices(mean_errors, n_top) + 1 # add 1 because the index is 0-based

    # Convert to CPU for processing
    # MSE_min_ls = Array(values)
    indices = Array(to(indices, CPU()))

    # Get expressions for best indices
    expr_best_ls = Expression[]
    # @info "Generating best expressions..."
    # @info "indices: $indices"
    # @info "length of indices: $(length(indices))"
    # @info "type of indices: $(typeof(indices))"
    for i in indices
        push!(expr_best_ls, get_expr(model, i))
    end

    # Print results
    # println("Best expressions:")
    # println("-"^20)
    # for expr in expr_best_ls
    # println(expr)
    # end
    # println("-"^20)

    # GC.gc()

    return expr_best_ls
end

function get_expr(model::PSRN, index::Int)
    return _get_expr(model, index, length(model.layers))
end

function _get_expr(model::PSRN, index::Int, layer_idx::Int)
    # @info "\t\tGetting expression for index $index, layer_idx $layer_idx"
    if layer_idx < 1
        return model.current_expr_ls[index + 1]
        # try
        #     return model.current_expr_ls[index + 1]
        # catch e
        #     if isa(e, BoundsError)
        #         @error "BoundsError: length of model.current_expr_ls is $(length(model.current_expr_ls)), however got index $index"
        #         throw(e)
        #     end
        # end
    end

    layer = model.layers[layer_idx]

    if layer isa DRLayer
        new_index = get_op_and_offset(layer, index)
        return _get_expr(model, new_index, layer_idx - 1)
    else
        op, offsets = get_op_and_offset(layer, index)
        if op isa UnaryOperator
            expr1 = _get_expr(model, offsets[1], layer_idx - 1)
            if op.name == "Identity"
                return expr1
            end
            return op.expr_gen(expr1)
        else
            expr1 = _get_expr(model, offsets[1], layer_idx - 1)
            expr2 = _get_expr(model, offsets[2], layer_idx - 1)
            return op.expr_gen(expr1, expr2)
        end
    end
end

function Base.show(io::IO, model::PSRN)
    print(
        io,
        "PSRN(n_variables=$(model.n_variables), operators=$(model.operators), " *
        "n_layers=$(model.n_symbol_layers))\n",
    )
    print(io, "Layer dimensions: ")
    return print(io, join([layer.out_dim for layer in model.layers], " → "))
end

function __init__()
    COMPILATION_MANAGER.initialized = false
    # 更新类型别名
    @eval TensorType = PSRNtharray.THArrays_mod[].Tensor
end

# Export types and functions
export PSRN,
    SymbolLayer,
    DRLayer,
    Operator,
    UnaryOperator,
    BinaryOperator,
    get_best_expr_and_MSE_topk,
    get_expr,
    get_op_and_offset


end
