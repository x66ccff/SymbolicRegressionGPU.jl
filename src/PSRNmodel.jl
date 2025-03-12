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
using Reactant: @compile, ConcreteRArray
import Base: copy

using ProgressMeter


using Reactant
const CompiledKernel = Reactant.Compiler.Thunk

const T_kernel_compiling = Float32  # Default Float64
const T_kernel_compiling_idx = Int64  # Default Float64

# Operator abstractions
abstract type Operator end

mutable struct UnaryOperator <: Operator
    name::String
    kernel::Function
    compiled_kernel::Union{CompiledKernel, Nothing}
    expr_gen::Function
end

abstract type BinaryOperator <: Operator end

mutable struct BinaryTriuOperator <: BinaryOperator
    name::String
    kernel::Function
    compiled_kernel::Union{CompiledKernel, Nothing}
    expr_gen::Function
end

mutable struct BinarySquaredOperator <: BinaryOperator
    name::String
    kernel::Function
    compiled_kernel::Union{CompiledKernel, Nothing}
    expr_gen::Function
end

function copy(op::UnaryOperator)
    UnaryOperator(op.name, op.kernel, op.compiled_kernel, op.expr_gen)
end

function copy(op::BinaryTriuOperator)
    BinaryTriuOperator(op.name, op.kernel, op.compiled_kernel, op.expr_gen)
end

function copy(op::BinarySquaredOperator)
    BinarySquaredOperator(op.name, op.kernel, op.compiled_kernel, op.expr_gen)
end

function get_scale(op::Operator, in_dim::Int)
    if op isa UnaryOperator
        return in_dim
    elseif op isa BinaryTriuOperator
        return in_dim * (in_dim + 1) ÷ 2
    elseif op isa BinarySquaredOperator
        return in_dim * in_dim
    else
        error("Unsupported operator type: $(typeof(op))")
    end
end

# Operator dictionary
const OPERATORS = Dict{String,Operator}(
    "Identity" => UnaryOperator("Identity", identity_kernel!, nothing, identity),
    "Sin" => UnaryOperator("Sin", sin_kernel!, nothing, sin),
    "Cos" => UnaryOperator("Cos", cos_kernel!, nothing,  cos),
    "Exp" => UnaryOperator("Exp", exp_kernel!, nothing, exp),
    "Log" => UnaryOperator("Log", log_kernel!, nothing, safe_log),

    "Add" => BinaryTriuOperator("Add", add_kernel!, nothing,  +),
    "Mul" => BinaryTriuOperator("Mul", mul_kernel!, nothing,  *),

    "Div" => BinarySquaredOperator("Div", div_kernel!, nothing,  /),
    "Sub" => BinarySquaredOperator("Sub", sub_kernel!, nothing,  -),

    "SemiDiv" => BinaryTriuOperator("SemiDiv", semidiv_kernel!, nothing,  /),
    "SemiSub" => BinaryTriuOperator("SemiSub", semisub_kernel!, nothing,  -),

    "Inv" => UnaryOperator("Inv", inv_kernel!, nothing, x -> 1 / x),
    "Neg" => UnaryOperator("Neg", neg_kernel!, nothing,  x -> -x),
    "Sqrt" => UnaryOperator("Sqrt", sqrt_kernel!, nothing,  safe_sqrt),

)

function concat_arrays(arrays::Vector{<:AbstractMatrix})
    return hcat(arrays...)
end

function topk_indices(errors::AbstractMatrix, k::Int)
    sorted_indices = partialsortperm(vec(errors), 1:k)
    return sorted_indices
end

# DRLayer implementation
mutable struct DRLayer
    in_dim::Int
    out_dim::Int
    dr_indices::Vector{Int}
    dr_mask::Vector{Bool}

    function DRLayer(in_dim::Int, dr_mask::Vector{Bool})
        out_dim = sum(dr_mask)
        dr_indices = findall(dr_mask)
        return new(in_dim, out_dim, dr_indices, dr_mask)
    end
end

function forward(layer::DRLayer, x::AbstractMatrix)
    return x[:, layer.dr_mask]
end

function get_op_and_offset(layer::DRLayer, index::Int)
    return layer.dr_indices[index + 1]
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
    offset_tensor::Union{Matrix{Int},Nothing}
    triu_idx::Union{ConcreteRArray{T_kernel_compiling_idx, 2},Nothing}
    squared_idx::Union{ConcreteRArray{T_kernel_compiling_idx, 2},Nothing}
    hcat_compiled::Union{CompiledKernel, Nothing}

    function SymbolLayer(in_dim::Int, operator_names::Vector{String})
        n_binary_U = 0
        n_binary_D = 0
        n_unary = 0
        operator_list = Operator[]
        operators = [copy(OPERATORS[name]) for name in operator_names]

        n_triu = (in_dim * (in_dim + 1)) ÷ 2
        in_dim_square = in_dim * in_dim

        for op in operators
            if op isa UnaryOperator
                n_unary += 1
            elseif op isa BinaryTriuOperator
                n_binary_U += 1
            elseif op isa BinarySquaredOperator
                n_binary_D += 1
            else 
                error("op must be UnaryOperator, BinarySquaredOperator or BinaryTriuOperator")
            end
        end

        for op in operators
            if op isa BinaryTriuOperator
                push!(operator_list, op)
            end
        end

        for op in operators
            if op isa BinarySquaredOperator
                push!(operator_list, op)
            end
        end

        for op in operators
            if op isa UnaryOperator
                push!(operator_list, op)
            end
        end

        out_dim = n_unary * in_dim + n_binary_U * n_triu + n_binary_D * in_dim_square

        triu_idx = n_binary_U == 0 ? nothing : ConcreteRArray(get_triu_indices(in_dim))
        squared_idx = n_binary_D == 0 ? nothing : ConcreteRArray(get_squared_indices(in_dim))

        layer = new(
            in_dim, out_dim, operators, n_binary_U, n_binary_D, n_unary,
            operator_list, n_triu, in_dim_square, nothing, nothing, triu_idx, squared_idx, nothing
        )

        init_offset(layer)
        return layer
    end
end

# SymbolLayer methods
function get_out_dim_cum_ls(layer::SymbolLayer)
    if layer.out_dim_cum_ls !== nothing
        return layer.out_dim_cum_ls
    end

    out_dim_ls = Int[]
    for op in layer.operator_list
        if op isa UnaryOperator
            push!(out_dim_ls, layer.in_dim)
        else
            if op isa BinarySquaredOperator
                push!(out_dim_ls, layer.in_dim_square)
            elseif op isa BinaryTriuOperator
                push!(out_dim_ls, layer.n_triu)
            else
                error("Only support BinarySquaredOperator and BinaryTriuOperator in BinaryOperator")
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

    # Fill unary tensor
    unary_tensor[:, 1] = arange_tensor
    unary_tensor[:, 2] .= layer.in_dim

    # Fill binary_U_tensor (undirected binary operations)
    start = 1
    for i in 0:(layer.in_dim - 1)
        len = layer.in_dim - i
        binary_U_tensor[start:(start + len - 1), 1] .= i
        binary_U_tensor[start:(start + len - 1), 2] = i:(layer.in_dim - 1)
        start += len
    end

    # Fill binary_D_tensor (directed binary operations)
    start = 1
    for i in 0:(layer.in_dim - 1)
        len = layer.in_dim
        binary_D_tensor[start:(start + len - 1), 1] .= i
        binary_D_tensor[start:(start + len - 1), 2] = 0:(layer.in_dim - 1)
        start += len
    end

    # Combine all indices
    start = 1
    for func in layer.operator_list
        if func isa UnaryOperator
            t = unary_tensor
        else
            t = (func isa BinarySquaredOperator) ? binary_D_tensor : binary_U_tensor
        end
        len = size(t, 1)
        offset_tensor[start:(start + len - 1), :] = t
        start += len
    end

    return offset_tensor
end

function init_offset(layer::SymbolLayer)
    layer.offset_tensor = get_offset_tensor(layer)
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

function forward(layer::SymbolLayer, xr::Reactant.ConcreteRArray)
    n = layer.in_dim
    results = Reactant.ConcreteRArray{T_kernel_compiling}[]
    for op in layer.operator_list
        if isa(op, UnaryOperator)
            res = op.compiled_kernel(xr)
        elseif isa(op, BinaryTriuOperator)
            res = op.compiled_kernel(xr, n, layer.triu_idx)
        elseif isa(op, BinarySquaredOperator)
            res = op.compiled_kernel(xr, n, layer.squared_idx)
        else
            error("op must be UnaryOperator, BinaryTriuOperator or BinarySquaredOperator")
        end
        push!(results, res)
    end
    if layer.hcat_compiled == nothing
        # compile the hcat
        xr_ls = [Reactant.to_rarray(ones(T_kernel_compiling, 1, get_scale(op, layer.in_dim))) for op in layer.operator_list]
        @info "⏳compiling hcat..."
        layer.hcat_compiled = @compile hcat(xr_ls...)
        @info "hcat compiled!"
    end
    return layer.hcat_compiled(results...)
end

function compile_kernels!(layer::SymbolLayer)
    for op in layer.operators

        if op isa UnaryOperator
            println("⏳compiling unary operator $(op.name) ... $(layer.in_dim)")
            op.compiled_kernel = compile_unary_kernel(layer.in_dim, op.kernel)
        elseif op isa BinaryTriuOperator
            println("⏳compiling binary triu operator $(op.name) ... $(layer.in_dim)")
            op.compiled_kernel = compile_binary_triu_kernel(layer.in_dim, op.kernel)
        elseif op isa BinarySquaredOperator
            println("⏳compiling binary squared operator $(op.name) ... $(layer.in_dim)")
            op.compiled_kernel = compile_binary_squared_kernel(layer.in_dim, op.kernel)
        else
            error("Unsupported operator type: $(typeof(op))")
        end
    end
end

mutable struct PSRN
    n_variables::Int
    operators::Vector{String}
    n_symbol_layers::Int
    layers::Vector{Union{SymbolLayer,DRLayer}}
    out_dim::Int
    use_dr_mask::Bool
    current_expr_ls::Vector{Expression}
    options::Options
    PSRN_topk::Int
    diff_compiled::CompiledKernel
    sum_squared_add_compiled::CompiledKernel
    top_k_compiled::CompiledKernel
    f_select::CompiledKernel
    f_is_finite::CompiledKernel
    f_fill::CompiledKernel

    function PSRN(;
        n_variables::Int,
        operators::Vector{String},
        n_symbol_layers::Int,
        dr_mask::Union{Vector{Bool},Nothing}=nothing,
        options::Options=Options(),
        PSRN_topk::Int,
    )
        layers = Union{SymbolLayer,DRLayer}[]
        use_dr_mask = !isnothing(dr_mask)

        for i in 1:n_symbol_layers
            if use_dr_mask && i == n_symbol_layers
                push!(layers, DRLayer(layers[end].out_dim, dr_mask))
            end

            if i == 1
                layer = SymbolLayer(n_variables, operators)
            else
                layer = SymbolLayer(layers[end].out_dim, operators)
            end
            @info "⏳compiling layer = $i / total $n_symbol_layers ..."
            compile_kernels!(layer)
            push!(layers, layer)
        end

        @info "⏳compiling PSRN.diff_compiled..."
        d(a,b) = a .- b
        x = rand(T_kernel_compiling, 1, layers[end].out_dim)
        
        y::T_kernel_compiling = 666.666
        xr = Reactant.to_rarray(x)

        diff_compiled = @compile d(xr,y)
        @info "👌compiling success!"

        @info "⏳compiling PSRN.sum_squared_add_compiled..."
        sum_squared_add(s,d) = s .+ d.^2
        x1 = ones(T_kernel_compiling, 1, layers[end].out_dim)
        x2 = ones(T_kernel_compiling, 1, layers[end].out_dim)
        x1r = Reactant.to_rarray(x1)
        x2r = Reactant.to_rarray(x2)
        sum_squared_add_compiled = @compile sum_squared_add(x1r, x2r)
        @info "👌compiling success!"

        # https://github.com/EnzymeAD/Reactant.jl/issues/485
        @info "⏳compiling PSRN.top_k_compiled..."
        x3 = ones(T_kernel_compiling, 1, layers[end].out_dim)
        x3r = Reactant.to_rarray(x3)
        top_k_compiled = @compile Reactant.Ops.top_k(x3r, PSRN_topk)
        @info "👌compiling success!"

        # https://github.com/EnzymeAD/Reactant.jl/issues/524
        @info "⏳compiling PSRN.set_nan_to_M_compiled..."
        M::T_kernel_compiling = 1.0*10^9
        x4 = rand(T_kernel_compiling, 1, layers[end].out_dim)
        x4[2] = NaN
        x4r = Reactant.to_rarray(x4)

        f_is_finite = @compile Reactant.Ops.is_finite(x4r)
        f_fill = @compile fill!(similar(x4r), M)
        f_select = @compile Reactant.Ops.select(f_is_finite(x4r), x4r, f_fill(similar(x4r), M))
        
        @info "👌compiling success!"

        return new(
            n_variables,
            operators,
            n_symbol_layers,
            layers,
            layers[end].out_dim,
            use_dr_mask,
            Expression[],
            options,
            PSRN_topk,
            diff_compiled,
            sum_squared_add_compiled,
            top_k_compiled,
            f_select,
            f_is_finite,
            f_fill
        )
    end
end

function PSRN_forward(model::PSRN, x::AbstractMatrix)
    h = x
    h = ConcreteRArray(h)
    for (i, layer) in enumerate(model.layers)
        h = forward(layer, h)
    end
    return h
end

function get_best_expr_and_MSE_topk(
    model::PSRN,
    X::AbstractMatrix,
    Y::Vector{T_kernel_compiling}
)
    batch_size = size(X, 1)
    Y = T_kernel_compiling.(Y)
    sum_squared_errors = zeros(T_kernel_compiling, 1, model.out_dim)
    sum_squared_errors_R = Reactant.to_rarray(sum_squared_errors)

    @info "forwarding time:"
    @time for i in 1:batch_size
        x_sliced = X[i:i, :]
        HR = PSRN_forward(model, x_sliced)
        diffR = model.diff_compiled(HR, Y[i])
        sum_squared_errors_R = model.sum_squared_add_compiled(sum_squared_errors_R, diffR)
    end

    @info "fill nan time:"
    @time sum_squared_errors_R = model.f_select(model.f_is_finite(sum_squared_errors_R),
                                        sum_squared_errors_R,
                                        model.f_fill(similar(sum_squared_errors_R), 1f9))

    mean_errors_R = sum_squared_errors_R

    val_R, idx_R = model.top_k_compiled(-mean_errors_R, model.PSRN_topk)
    indices = vec(convert(Matrix, idx_R))

    @info "Best Expressions:"
    expr_best_ls = Expression[]
    for i in indices
        expr = get_expr(model, Int64(i))
        push!(expr_best_ls, expr)
    end

    @info "GC.......🧹"
    GC.gc()
    @info "GC sucess🧹"

    return expr_best_ls
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

function get_expr(model::PSRN, index::Int)
    return _get_expr(model, index, length(model.layers))
end

function _get_expr(model::PSRN, index::Int, layer_idx::Int)
    if layer_idx < 1
        return model.current_expr_ls[index + 1]
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

end # module
