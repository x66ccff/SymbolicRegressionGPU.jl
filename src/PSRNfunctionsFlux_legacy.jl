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
# using Flux
using CUDA
# using cuDNN

# Base operator module - equivalent to the operators.jl file that would contain operator definitions
abstract type BaseOperator end

struct IdentityOp <: BaseOperator end
struct SinOp <: BaseOperator end
struct CosOp <: BaseOperator end
struct ExpOp <: BaseOperator end
struct LogOp <: BaseOperator end
struct NegOp <: BaseOperator end
struct InvOp <: BaseOperator end
struct AddOp <: BaseOperator end
struct MulOp <: BaseOperator end
struct DivOp <: BaseOperator end
struct SubOp <: BaseOperator end
struct SemiDivOp <: BaseOperator end
struct SemiSubOp <: BaseOperator end
struct SignOp <: BaseOperator end
struct Pow2Op <: BaseOperator end
struct Pow3Op <: BaseOperator end
struct PowOp <: BaseOperator end
struct SigmoidOp <: BaseOperator end
struct AbsOp <: BaseOperator end
struct CoshOp <: BaseOperator end
struct TanhOp <: BaseOperator end
struct SqrtOp <: BaseOperator end

# Base layer type
abstract type CustomLayer end

####################################### binary operators #######################################

struct Mul <: CustomLayer
    in_dim::Int
    out_dim::Int
    is_unary::Bool
    is_directed::Bool
    operator::MulOp
end

function Mul(in_dim::Int)
    out_dim = div(in_dim * (in_dim + 1), 2)
    Mul(in_dim, out_dim, false, false, MulOp())
end


function (l::Mul)(x)
    n = size(x, 2)
    batch_size = size(x, 1)
    out_dim = (n * (n + 1)) ÷ 2
    
    # 预分配输出矩阵
    out = CUDA.zeros(eltype(x), batch_size, out_dim)
    
    # 创建上三角索引
    indices = [(i,j) for i in 1:n for j in i:n]
    
    # 一次性操作，避免循环
    out = hcat([x[:, i] .* x[:, j] for (i,j) in indices]...)
    
    return out
end

# Add operator
struct Add <: CustomLayer
    in_dim::Int
    out_dim::Int
    is_unary::Bool
    is_directed::Bool
    operator::AddOp
end

function Add(in_dim::Int)
    out_dim = div(in_dim * (in_dim + 1), 2)
    Add(in_dim, out_dim, false, false, AddOp())
end


function (l::Add)(x)
    n = size(x, 2)
    batch_size = size(x, 1)
    out_dim = (n * (n + 1)) ÷ 2
    out = CUDA.zeros(eltype(x), batch_size, out_dim)
    indices = [(i,j) for i in 1:n for j in i:n]
    out = hcat([x[:, i] .+ x[:, j] for (i,j) in indices]...)
    
    return out
end

# Division operator
struct Div <: CustomLayer
    in_dim::Int
    out_dim::Int
    is_unary::Bool
    is_directed::Bool
    operator::DivOp
end

function Div(in_dim::Int)
    out_dim = in_dim * in_dim
    Div(in_dim, out_dim, false, true, DivOp())
end

function (l::Div)(x)
    n = size(x, 2)
    batch_size = size(x, 1)
    out_dim = n * n  # 注意这里是 n*n 而不是 (n*(n+1))÷2
    indices = [(i,j) for i in 1:n for j in 1:n]  # 注意这里是完整遍历
    hcat([x[:, i] ./ x[:, j] for (i,j) in indices]...)
end

# Subtraction operator
struct Sub <: CustomLayer
    in_dim::Int
    out_dim::Int
    is_unary::Bool
    is_directed::Bool
    operator::SubOp
end

function Sub(in_dim::Int)
    out_dim = in_dim * in_dim
    Sub(in_dim, out_dim, false, true, SubOp())
end

function (l::Sub)(x)
    n = size(x, 2)
    batch_size = size(x, 1)
    out_dim = n * n  # 注意这里是 n*n 而不是 (n*(n+1))÷2
    indices = [(i,j) for i in 1:n for j in 1:n]  # 注意这里是完整遍历
    hcat([x[:, i] .- x[:, j] for (i,j) in indices]...)
end

####################################### unary operators #######################################

# Define layers
struct Identity <: CustomLayer
    in_dim::Int
    out_dim::Int
    is_unary::Bool
    is_directed::Bool
    operator::IdentityOp
end

Identity(in_dim::Int) = Identity(in_dim, in_dim, true, true, IdentityOp())

# Forward pass
(l::Identity)(x) = x

struct Sin <: CustomLayer
    in_dim::Int
    out_dim::Int
    is_unary::Bool
    is_directed::Bool
    operator::SinOp
end

Sin(in_dim::Int) = Sin(in_dim, in_dim, true, true, SinOp())

(l::Sin)(x) = sin.(x)

struct Cos <: CustomLayer
    in_dim::Int
    out_dim::Int
    is_unary::Bool
    is_directed::Bool
    operator::CosOp
end

Cos(in_dim::Int) = Cos(in_dim, in_dim, true, true, CosOp())

(l::Cos)(x) = cos.(x)



# Exp operator
struct Exp <: CustomLayer
    in_dim::Int
    out_dim::Int
    is_unary::Bool
    is_directed::Bool
    operator::ExpOp
end

Exp(in_dim::Int) = Exp(in_dim, in_dim, true, true, ExpOp())

(l::Exp)(x) = exp.(x)

# Log operator
struct Log <: CustomLayer
    in_dim::Int
    out_dim::Int
    is_unary::Bool
    is_directed::Bool
    operator::LogOp
end

Log(in_dim::Int) = Log(in_dim, in_dim, true, true, LogOp())

(l::Log)(x) = safe_log.(x)

# Negation operator
struct Neg <: CustomLayer
    in_dim::Int
    out_dim::Int
    is_unary::Bool
    is_directed::Bool
    operator::NegOp
end

Neg(in_dim::Int) = Neg(in_dim, in_dim, true, true, NegOp())

(l::Neg)(x) = -x

# Inverse operator
struct Inv <: CustomLayer
    in_dim::Int
    out_dim::Int
    is_unary::Bool
    is_directed::Bool
    operator::InvOp
end

Inv(in_dim::Int) = Inv(in_dim, in_dim, true, true, InvOp())

(l::Inv)(x) = 1 ./ x



# Power operators
struct Pow2 <: CustomLayer
    in_dim::Int
    out_dim::Int
    is_unary::Bool
    is_directed::Bool
    operator::Pow2Op
end

Pow2(in_dim::Int) = Pow2(in_dim, in_dim, true, true, Pow2Op())

(l::Pow2)(x) = x .^ 2

struct Pow3 <: CustomLayer
    in_dim::Int
    out_dim::Int
    is_unary::Bool
    is_directed::Bool
    operator::Pow3Op
end

Pow3(in_dim::Int) = Pow3(in_dim, in_dim, true, true, Pow3Op())

(l::Pow3)(x) = x .^ 3

# Other unary operators
struct Sigmoid <: CustomLayer
    in_dim::Int
    out_dim::Int
    is_unary::Bool
    is_directed::Bool
    operator::SigmoidOp
end

Sigmoid(in_dim::Int) = Sigmoid(in_dim, in_dim, true, true, SigmoidOp())

(l::Sigmoid)(x) = 1 ./ (1 .+ exp.(-x))

struct Abs <: CustomLayer
    in_dim::Int
    out_dim::Int
    is_unary::Bool
    is_directed::Bool
    operator::AbsOp
end

Abs(in_dim::Int) = Abs(in_dim, in_dim, true, true, AbsOp())

(l::Abs)(x) = abs.(x)

struct Cosh <: CustomLayer
    in_dim::Int
    out_dim::Int
    is_unary::Bool
    is_directed::Bool
    operator::CoshOp
end

Cosh(in_dim::Int) = Cosh(in_dim, in_dim, true, true, CoshOp())

(l::Cosh)(x) = cosh.(x)

struct Tanh <: CustomLayer
    in_dim::Int
    out_dim::Int
    is_unary::Bool
    is_directed::Bool
    operator::TanhOp
end

Tanh(in_dim::Int) = Tanh(in_dim, in_dim, true, true, TanhOp())

(l::Tanh)(x) = tanh.(x)

struct Sqrt <: CustomLayer
    in_dim::Int
    out_dim::Int
    is_unary::Bool
    is_directed::Bool
    operator::SqrtOp
end

Sqrt(in_dim::Int) = Sqrt(in_dim, in_dim, true, true, SqrtOp())

(l::Sqrt)(x) = sqrt.(x)


export Identity, Add, Mul, Sin, Cos, Exp, Log, Neg, Inv, Div, Sub, Pow2, Pow3, Sigmoid, Abs, Cosh, Tanh, Sqrt

end # module PSRNfunctionsFlux