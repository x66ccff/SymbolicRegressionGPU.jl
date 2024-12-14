module PSRNfunctionsFlux

using ..CoreModule:
    DATA_TYPE,
    LOSS_TYPE,
    RecordType,
    Dataset,
    AbstractOptions,
    Options,
    ComplexityMapping,
    AbstractMutationWeights,
    MutationWeights,
    get_safe_op,
    max_features,
    is_weighted,
    sample_mutation,
    plus,
    sub,
    mult,
    square,
    cube,
    pow,
    safe_pow,
    safe_log,
    safe_log2,
    safe_log10,
    safe_log1p,
    safe_sqrt,
    safe_acosh,
    neg,
    greater,
    cond,
    relu,
    logical_or,
    logical_and,
    gamma,
    erf,
    erfc,
    atanh_clip,
    create_expression,
    has_units

using CUDA
using Flux: gpu

# Base operator abstract types
abstract type BaseOperator end
abstract type UnaryOperator <: BaseOperator end
abstract type BinaryOperator <: BaseOperator end
abstract type DirectedBinaryOperator <: BinaryOperator end
abstract type UndirectedBinaryOperator <: BinaryOperator end

# Operator definitions
struct IdentityOp <: UnaryOperator end
struct SinOp <: UnaryOperator end
struct CosOp <: UnaryOperator end
struct ExpOp <: UnaryOperator end
struct LogOp <: UnaryOperator end
struct NegOp <: UnaryOperator end
struct InvOp <: UnaryOperator end
struct Pow2Op <: UnaryOperator end
struct Pow3Op <: UnaryOperator end
struct SigmoidOp <: UnaryOperator end
struct AbsOp <: UnaryOperator end
struct CoshOp <: UnaryOperator end
struct TanhOp <: UnaryOperator end
struct SqrtOp <: UnaryOperator end

struct AddOp <: UndirectedBinaryOperator end
struct MulOp <: UndirectedBinaryOperator end
struct DivOp <: DirectedBinaryOperator end
struct SubOp <: DirectedBinaryOperator end

# Base layer types
abstract type CustomLayer end
abstract type UnaryLayer <: CustomLayer end
abstract type BinaryLayer <: CustomLayer end

# Generic structures for layers
struct GenericUnaryLayer{T<:UnaryOperator} <: UnaryLayer
    in_dim::Int
    out_dim::Int
    is_unary::Bool
    is_directed::Bool
    operator::T
end

struct GenericBinaryLayer{T<:BinaryOperator} <: BinaryLayer
    in_dim::Int
    out_dim::Int
    is_unary::Bool
    is_directed::Bool
    operator::T
    indices::Vector{Tuple{Int,Int}}
end
# Add direct constructor with type conversion
function GenericBinaryLayer(in_dim::Int, out_dim::Int, is_unary::Bool, is_directed::Bool, 
    operator::T, indices::AbstractArray{Tuple{Int,Int}}) where T<:BinaryOperator
    return GenericBinaryLayer(
        in_dim,
        out_dim,
        is_unary,
        is_directed,
        operator,
        Vector{Tuple{Int,Int}}(indices)  # Convert any array type to Vector
    )
end     

# Constructor helpers
function create_undirected_indices(n::Int)
    [(i,j) for i in 1:n for j in i:n]
end

function create_directed_indices(n::Int)
    [(i,j) for i in 1:n for j in 1:n]
end


# Unary layer constructors
for (op_name, op_type) in [
    (:Identity, :IdentityOp), (:Sin, :SinOp), (:Cos, :CosOp),
    (:Exp, :ExpOp), (:Log, :LogOp), (:Neg, :NegOp),
    (:Inv, :InvOp), (:Pow2, :Pow2Op), (:Pow3, :Pow3Op),
    (:Sigmoid, :SigmoidOp), (:Abs, :AbsOp), (:Cosh, :CoshOp),
    (:Tanh, :TanhOp), (:Sqrt, :SqrtOp)
]
    @eval begin
        const $op_name = GenericUnaryLayer{$op_type}
        function $op_name(in_dim::Int)
            return GenericUnaryLayer(in_dim, in_dim, true, true, $op_type())
        end
        # Add constructor for direct initialization
        function $op_name(in_dim::Int, out_dim::Int, is_unary::Bool, is_directed::Bool, operator::$op_type)
            return GenericUnaryLayer(in_dim, out_dim, is_unary, is_directed, operator)
        end
    end
end

# Binary layer constructors
for (op_name, op_type, indices_fn, is_directed) in [
    (:Add, :AddOp, :create_undirected_indices, false),
    (:Mul, :MulOp, :create_undirected_indices, false),
    (:Div, :DivOp, :create_directed_indices, true),
    (:Sub, :SubOp, :create_directed_indices, true)
]
    @eval begin
        const $op_name = GenericBinaryLayer{$op_type}
        
        function $op_name(in_dim::Int)
            out_dim = if $indices_fn == create_undirected_indices
                div(in_dim * (in_dim + 1), 2)
            else
                in_dim * in_dim
            end
            indices = $indices_fn(in_dim)
            return GenericBinaryLayer(
                in_dim,
                out_dim,
                false,
                $is_directed,
                $op_type(),
                indices
            )
        end
    end
end



# Forward pass implementations
function (l::GenericUnaryLayer{IdentityOp})(x)
    x
end

function (l::GenericUnaryLayer{T})(x) where T <: UnaryOperator
    function element_wise_op(x, op)
        return op.(x)  # 使用广播运算
    end
    
    op_map = Dict(
        SinOp => sin,
        CosOp => cos,
        ExpOp => exp,
        LogOp => safe_log,
        NegOp => -,
        InvOp => x -> 1.0f0 ./ x,
        Pow2Op => x -> x .^ 2,
        Pow3Op => x -> x .^ 3,
        SigmoidOp => x -> 1.0f0 ./ (1.0f0 .+ exp.(-x)),
        AbsOp => abs,
        CoshOp => cosh,
        TanhOp => tanh,
        SqrtOp => sqrt
    )
    
    op = op_map[T]
    return element_wise_op(x, op)
end


# Forward pass implementation
function (l::GenericBinaryLayer{T})(x) where T <: BinaryOperator
    op_map = Dict(
        AddOp => +,
        MulOp => *,
        DivOp => /,
        SubOp => -
    )
    op = op_map[T]
    
    x_gpu = x |> gpu
    # 修改实现方式以正确处理 CUDA arrays
    result = similar(x_gpu, size(x_gpu, 1), length(l.indices))
    for (idx, (i, j)) in enumerate(l.indices)
        result[:, idx] = op.(view(x_gpu, :, i), view(x_gpu, :, j))
    end
    return result
end


# Exports
export Identity, Add, Mul, Sin, Cos, Exp, Log, Neg, Inv,
    Div, Sub, Pow2, Pow3, Sigmoid, Abs, Cosh, Tanh, Sqrt

end # module PSRNfunctionsFlux