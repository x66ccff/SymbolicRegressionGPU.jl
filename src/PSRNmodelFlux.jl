module PSRNmodelFlux

import Flux: trainable, gpu, cpu   # 用 import 导入要扩展的函数
using Flux: gpu_device  # 其他不需要扩展的函数可以用 using
using CUDA
using cuDNN
using Statistics
using ProgressMeter
using Zygote: ignore


using ..PSRNfunctionsFlux
import ..CoreModule.OperatorsModule: plus, sub, mult, square, cube, safe_pow, safe_log,
    safe_log2, safe_log10, safe_sqrt, safe_acosh, neg, greater,
    cond, relu, logical_or, logical_and, gamma

import ..CoreModule: Options, Dataset 
using DynamicExpressions: Node, Expression
abstract type Operator end

struct UnaryOperator <: Operator
    name::String
    is_unary::Bool
    is_directed::Bool
    op::Function
end

struct BinaryOperator <: Operator
    name::String
    is_unary::Bool
    is_directed::Bool
    op::Function
end

const OPERATORS = Dict{String, Operator}(
    "Identity" => UnaryOperator("Identity", true, true, identity),
    "Sin" => UnaryOperator("Sin",  true, true, sin),
    "Cos" => UnaryOperator("Cos",  true, true, cos),
    "Exp" => UnaryOperator("Exp",  true, true, exp),
    "Log" => UnaryOperator("Log",  true, true, safe_log),
    "Neg" => UnaryOperator("Neg",  true, true, -),
    "Pow" => BinaryOperator("Pow",  true, true, safe_pow),
    "Sqrt" => UnaryOperator("Sqrt",  true, true, safe_sqrt),

    "Add" => BinaryOperator("Add",  false, false, +),
    "Mul" => BinaryOperator("Mul",  false, false, *),

    "Div" => BinaryOperator("Div",  false, true, /),
    "Sub" => BinaryOperator("Sub",  false, true, -)


)



# Duplicate Removal Layer
struct DRLayer
    in_dim::Int
    out_dim::Int
    dr_indices::Vector{Int}
    dr_mask::Vector{Bool}
end

function DRLayer(in_dim::Int, dr_mask::Vector{Bool})
    out_dim = sum(dr_mask)
    dr_indices = findall(dr_mask)
    DRLayer(in_dim, out_dim, dr_indices, dr_mask)
end

function (l::DRLayer)(x)
    x[:, l.dr_mask]
end

function get_op_and_offset(l::DRLayer, index::Int)
    l.dr_indices[index]
end

# Symbol Layer
mutable struct SymbolLayer
    in_dim::Int
    out_dim::Int
    n_triu::Int
    in_dim_square::Int
    operators::Vector{String}
    operator_list::Vector{Any}
    layers::Vector{Any}
    n_binary_U::Int
    n_binary_D::Int
    n_unary::Int
    offset_tensor::Matrix{Int}
    out_dim_cum_ls::Any
end

function get_offset_tensor(in_dim::Int, out_dim::Int, n_triu::Int, in_dim_square::Int, layers::Vector)
    offset_tensor = zeros(Int, out_dim, 2)
    arange_tensor = collect(1:in_dim)
    
    binary_U_tensor = zeros(Int, n_triu, 2)
    binary_D_tensor = zeros(Int, in_dim_square, 2)
    unary_tensor = zeros(Int, in_dim, 2)

    # 设置一元张量
    unary_tensor[:, 1] = arange_tensor
    unary_tensor[:, 2] .= in_dim

    # 设置无向二元张量
    start = 1
    for i in 1:in_dim
        len_ = in_dim - i + 1
        binary_U_tensor[start:start + len_ - 1, 1] .= i
        binary_U_tensor[start:start + len_ - 1, 2] = arange_tensor[i:end]
        start += len_
    end

    # 设置有向二元张量
    start = 1
    for i in 1:in_dim
        len_ = in_dim
        binary_D_tensor[start:start + len_ - 1, 1] .= i
        binary_D_tensor[start:start + len_ - 1, 2] = arange_tensor[1:end]
        start += len_
    end

    # 填充offset_tensor
    start = 1
    for func in layers
        if !func.is_unary
            if func.is_directed
                t = binary_D_tensor
            else
                t = binary_U_tensor
            end
        else
            t = unary_tensor
        end
        len_ = size(t, 1)
        offset_tensor[start:start + len_ - 1, :] = t
        start += len_
    end
    
    return offset_tensor
end

function init_offset!(in_dim::Int, out_dim::Int, n_triu::Int, in_dim_square::Int, layers::Vector)
    offset_tensor = get_offset_tensor(in_dim, out_dim, n_triu, in_dim_square, layers)
    return offset_tensor
end

function SymbolLayer(in_dim::Int, operator_names=["Add", "Mul", "Identity", "Sin", "Exp", "Neg", "Inv"])
    n_triu = div(in_dim * (in_dim + 1), 2)
    in_dim_square = in_dim * in_dim
    
    # Count operator types
    n_binary_U = 0
    n_binary_D = 0
    n_unary = 0
    layers = []
    
    # First pass to count operators
    for op in operator_names
        func = eval(Symbol(op))(in_dim)
        if !func.is_unary
            if func.is_directed
                n_binary_D += 1
            else
                n_binary_U += 1
            end
        else
            n_unary += 1
        end
    end
    
    # Second pass to order operators
    # Add undirected binary ops
    for op in operator_names
        func = eval(Symbol(op))(in_dim)
        if !func.is_unary && !func.is_directed
            push!(layers, func)
        end
    end
    
    # Add directed binary ops
    for op in operator_names
        func = eval(Symbol(op))(in_dim)
        if !func.is_unary && func.is_directed
            push!(layers, func)
        end
    end
    
    # Add unary ops
    for op in operator_names
        func = eval(Symbol(op))(in_dim)
        if func.is_unary
            push!(layers, func)
        end
    end
    
    operator_list = [OPERATORS[name] for name in operator_names]

    out_dim = n_unary * in_dim + n_binary_U * n_triu + n_binary_D * in_dim_square
    println("out_dim: ", out_dim, " n_unary: ", n_unary, " n_binary_U: ", n_binary_U, " n_binary_D: ", n_binary_D)
    # Initialize offset tensor
    offset_tensor = init_offset!(in_dim, out_dim, n_triu, in_dim_square, layers)
    out_dim_cum_ls = nothing
    SymbolLayer(in_dim, out_dim, n_triu, in_dim_square, operator_names, operator_list,
        layers, n_binary_U, n_binary_D, n_unary, offset_tensor, out_dim_cum_ls)
end

function (l::SymbolLayer)(x)
    outputs = []
    for layer in l.layers
        h = layer(x)
        # @info "h size: ", size(h)
        push!(outputs, h)
    end
    hcat(outputs...)  # Changed from vcat to hcat
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
            if func isa BinaryOperator
                if func.is_directed
                    push!(out_dim_ls, layer.in_dim_square)
                else
                    push!(out_dim_ls, layer.n_triu)
                end
            end
        end
    end
    
    layer.out_dim_cum_ls = [sum(out_dim_ls[1:i]) for i in 1:length(out_dim_ls)]
    return layer.out_dim_cum_ls
end

# 在SymbolLayer结构体定义后添加get_op_and_offset方法
function get_op_and_offset(layer::SymbolLayer, index::Int)
    out_dim_cum_ls = get_out_dim_cum_ls(layer)
    
    # Find the corresponding operator
    op_idx = 1
    for i in eachindex(out_dim_cum_ls)
        if index < out_dim_cum_ls[i]
            op_idx = i
            break
        end
    end
    
    # Get offset
    offset = layer.offset_tensor[index, :]
    return layer.operator_list[op_idx], offset
end

# PSRN Model
mutable struct PSRN
    n_variables::Int
    operators::Vector{String}
    n_symbol_layers::Int
    use_dr_mask::Bool
    layers::Vector{Any}
    out_dim::Int
    current_expr_ls::Vector{Any}
    device::Function  # 添加device字段用于GPU支持
end

function PSRN(;n_variables=1,
              operators=["Add", "Mul", "Identity", "Sin", "Exp", "Neg", "Inv"],
              n_symbol_layers=3,
              options = Options(),
              dr_mask=nothing)
    
    use_dr_mask = !isnothing(dr_mask)
    layers = []
    
    # Build layers
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
    
    out_dim = layers[end].out_dim
    
    PSRN(n_variables, operators, n_symbol_layers, use_dr_mask,
         layers, out_dim, [], gpu)
end

# 添加Flux.trainable接口
trainable(m::PSRN) = (layers=m.layers,)

# 添加GPU支持
function gpu(m::PSRN)
    m.device = gpu
    m.layers = gpu.(m.layers)
    return m
end

function cpu(m::PSRN)
    m.device = cpu  
    m.layers = cpu.(m.layers)
    return m
end

# 实现Flux.Chain的行为
Base.length(m::PSRN) = length(m.layers)
Base.getindex(m::PSRN, i::Int) = m.layers[i]
Base.iterate(m::PSRN) = iterate(m.layers)
Base.iterate(m::PSRN, state) = iterate(m.layers, state)

function (m::PSRN)(x)
    h = x
    for layer in m.layers
        h = layer(h)
    end
    h
end

function _get_expr(psrn::PSRN, index::Int, layer_idx::Int)
    if layer_idx < 1
        return psrn.current_expr_ls[index]
    end
    
    layer = psrn.layers[layer_idx]
    op, offsets = get_op_and_offset(layer, index)
    
    # Get subexpression
    expr1 = _get_expr(psrn, offsets[1], Int(layer_idx-1))
    T = eltype(expr1.tree)  # Gets the type of the expression
    
    if op isa UnaryOperator
        # Create a unary operation expression
        if op.name == "Identity"
            return expr1
        end
        return op.op(expr1)
    else
        # Create a binary operation expression
        expr2 = _get_expr(psrn, offsets[2], Int(layer_idx-1))
        return op.op(expr1, expr2)
    end
end

function get_expr(psrn::PSRN, index::Int64)
    return _get_expr(psrn, Int(index), Int(length(psrn.layers)))
end

function get_best_expr_and_MSE_topk(net::PSRN, X, Y, n_top)
    # times = Dict{String, Float64}()
    # CUDA.allowscalar(true)
    
    # 移动到设备
    # t_start = time()
    X = X |> net.device
    Y = Y |> net.device
    println("net on GPU: ", net.device)
    println("hihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihi")
    @assert X isa CuArray "X is not on GPU"
    @assert Y isa CuArray "Y is not on GPU"
    println("hellohellohellohellohellohellohellohellohellohellohellohellohellohello")
    # times["move_to_device"] = time() - t_start
    
    # 初始化 MSE - 确保在正确的设备上创建
    # t_start = time()
    sum_squared_diff = zeros(eltype(X), net.out_dim) |> net.device
    # times["init_sum_squared_diff"] = time() - t_start
    
    println("here22"^10)
    println("shape of X: ", size(X))
    # 计算 sum_squared_diff
    # t_start = time()
    
    ignore() do
        CUDA.allowscalar() do
            @views for i in axes(X, 1)
                println("i: ", i)
                X_i = reshape(X[i, :], 1, :)
                H = net(X_i)
                H = vec(H)
                diff = H .- Y[i]
                square = diff .^ 2
                square_resized = reshape(square, size(sum_squared_diff))
                sum_squared_diff .+= square_resized  # 改用 .+=
            end
        end
    end
    # times["calculate_sum_squared_diff"] = time() - t_start
    println("here33"^10)
    # 计算平均值
    # t_start = time()
    mean_squared_error = sum_squared_diff ./ size(X, 1)
    mean_squared_error[isnan.(mean_squared_error)] .= Inf
    mean_squared_error[isinf.(mean_squared_error)] .= Inf
    # times["calculate_mse"] = time() - t_start
    println("here44"^10)
    # 找到 top k
    # t_start = time()
    CUDA.allowscalar() do
        sorted_indices = partialsortperm(mean_squared_error, 1:n_top)
    end
    sorted_indices = sorted_indices |> cpu
    # times["sort_indices"] = time() - t_start
    println("here55"^10)
    MSE_min_ls = mean_squared_error[sorted_indices]
    
    # 获取表达式
    # t_start = time()
    expr_best_ls = Expression[]
    @showprogress "Getting expressions..." for i in sorted_indices
        push!(expr_best_ls, get_expr(net, i))
    end
    # times["get_expressions"] = time() - t_start
    println("here66"^10)
    # 打印时间统计
    # println("\nTime statistics:")
    # println("-"^40)
    # for (operation, elapsed) in times
    #     println("$operation: $(round(elapsed, digits=3)) seconds")
    # end
    # println("Total time: $(round(sum(values(times)), digits=3)) seconds")
    # println("-"^40)
    
    println("here77"^10)
    # 打印表达式结果
    println("\nexpr_best_ls:")
    println("-"^20)
    for expr in expr_best_ls
        println(expr)
    end
    println("-"^20)
    println("here88"^10)
    return expr_best_ls, MSE_min_ls
end

# 使用示例:
# expr_best_ls, MSE_min_ls = get_best_expr_and_MSE_topk(model, X_data, Y_data, 10)
# export PSRN, forward, get_expr, to_device, find_best_indices, get_best_expressions
export PSRN, get_best_expr_and_MSE_topk

end # module PSRNmodelFlux