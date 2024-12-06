module PSRNmodel

# 1. 导入基础函数
using ..PSRNfunctions
# 尝试使用完整的导入路径
import ..CoreModule.OperatorsModule: plus, sub, mult, square, cube, safe_pow, safe_log,
    safe_log2, safe_log10, safe_sqrt, safe_acosh, neg, greater,
    cond, relu, logical_or, logical_and, gamma

import ..CoreModule: Options, Dataset  # 添加其他需要的类型


# 2. 只使用KernelAbstractions的抽象接口
# 在文件顶部的导入部分添加
using KernelAbstractions
const KA = KernelAbstractions
using CUDA

@static if Base.find_package("AMDGPU") !== nothing
    using AMDGPU
    using AMDGPU: ROCArray
end

@static if Base.find_package("oneAPI") !== nothing
    using oneAPI
    using oneAPI: oneArray
end


# 3. 其他导入
using Printf: @sprintf  # 直接导入宏
using DynamicExpressions: Node, Expression


# 基础操作符抽象类型
abstract type Operator end

# 操作符类型定义
struct UnaryOperator <: Operator
    name::String
    kernel::Function
    is_directed::Bool
    op::Function  # 实际的操作符函数
end

struct BinaryOperator <: Operator
    name::String
    kernel::Function
    is_directed::Bool
    op::Function  # 实际的操作符函数
end

# 预定义所有支持的操作符
const OPERATORS = Dict{String, Operator}(
    "Identity" => UnaryOperator("Identity", identity_kernel!, true, identity),
    "Sin" => UnaryOperator("Sin", sin_kernel!, true, sin),
    "Cos" => UnaryOperator("Cos", cos_kernel!, true, cos),
    "Exp" => UnaryOperator("Exp", exp_kernel!, true, exp),
    "Log" => UnaryOperator("Log", log_kernel!, true, safe_log),
    "Neg" => UnaryOperator("Neg", neg_kernel!, true, -),
    "Add" => BinaryOperator("Add", add_kernel!, false, +),
    "Mul" => BinaryOperator("Mul", mul_kernel!, false, *),
    "Div" => BinaryOperator("Div", div_kernel!, true, /),
    "Sub" => BinaryOperator("Sub", sub_kernel!, true, -),
    "Pow" => BinaryOperator("Pow", pow_kernel!, true, safe_pow),
    "Sqrt" => UnaryOperator("Sqrt", sqrt_kernel!, true, safe_sqrt)
)

# SymbolLayer结构
mutable struct SymbolLayer
    in_dim::Int
    out_dim::Int
    operators::Vector{Operator}  # 改用Operator而不是String
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
        
        # 计数所有操作符
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
        
        # 按顺序添加操作符：先无向二元，再有向二元，最后一元
        # 1. 无向二元操作符
        for op in operators
            if op isa BinaryOperator && !op.is_directed
                push!(operator_list, op)
            end
        end
        
        # 2. 有向二元操作符
        for op in operators
            if op isa BinaryOperator && op.is_directed
                push!(operator_list, op)
            end
        end
        
        # 3. 一元操作符
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

# 修改索引生成函数使用传入的backend
function get_triu_indices(n::Int, backend)
    if backend isa KA.CPU
        indices = Tuple{Int,Int}[]
        for i in 1:n
            for j in i:n
                push!(indices, (i,j))
            end
        end
        return indices
    else
        # GPU版本返回两个向量
        row_idx = Int[]
        col_idx = Int[]
        for i in 1:n
            for j in i:n
                push!(row_idx, i)
                push!(col_idx, j)
            end
        end
        # 根据backend类型返回对应的数组
        return (to_device(row_idx, backend), to_device(col_idx, backend))
    end
end

function to_device(x::AbstractArray, backend)
    if isa(backend, KA.GPU) && CUDA.functional()
        return CuArray(x)
    elseif @isdefined(ROCArray) && isa(backend, KA.GPU) && AMDGPU.functional()
        return ROCArray(x)
    elseif @isdefined(oneArray) && isa(backend, KA.GPU) && oneAPI.functional()
        return oneArray(x)
    end
    return x
end


# 为SymbolLayer添加获取作符和偏移量的方���
function get_op_and_offset(layer::SymbolLayer, index::Int)
    out_dim_cum_ls = get_out_dim_cum_ls(layer)
    
    # 找到对应的操作符
    op_idx = 1
    for i in eachindex(out_dim_cum_ls)
        if index < out_dim_cum_ls[i]
            op_idx = i
            break
        end
    end
    
    # 获取偏移量
    offset = layer.offset_tensor[index, :]
    return layer.operator_list[op_idx], offset
end

# 添加前向传播函数
function forward(layer::SymbolLayer, x::AbstractArray, backend)
    results = []
    
    for op in layer.operator_list
        if op isa UnaryOperator
            # 获取输入数据的后端
            device_backend = get_backend(x)
            # 创建并执行 kernel
            kernel = op.kernel(device_backend, 256)
            result = similar(x)
            event = kernel(result, x, ndrange=size(x))
            if event !== nothing
                wait(event)
            end
            push!(results, result)
        else # BinaryOperator
            if op.is_directed
                # 有向二元操作 (如除法、减法)
                device_backend = get_backend(x)
                kernel = op.kernel(device_backend, 256)
                
                if device_backend isa KA.CPU
                    for i in 1:layer.in_dim
                        for j in 1:layer.in_dim
                            x1 = view(x, :, i)
                            x2 = view(x, :, j)
                            result = similar(x1)
                            event = kernel(result, x1, x2, ndrange=size(x1))
                            if event !== nothing
                                wait(event)
                            end
                            push!(results, result)
                        end
                    end
                else
                    # GPU版本：批量处理
                    row_idx, col_idx = get_square_indices(layer.in_dim, device_backend)
                    x1 = view(x, :, row_idx)
                    x2 = view(x, :, col_idx)
                    result = similar(x1)
                    event = kernel(result, x1, x2, ndrange=size(x1))
                    if event !== nothing
                        wait(event)
                    end
                    push!(results, result)
                end
            else
                # 无向二元操作 (如加法、乘法)
                device_backend = get_backend(x)
                kernel = op.kernel(device_backend, 256)
                
                if device_backend isa KA.CPU
                    for i in 1:layer.in_dim
                        for j in i:layer.in_dim
                            x1 = view(x, :, i)
                            x2 = view(x, :, j)
                            result = similar(x1)
                            event = kernel(result, x1, x2, ndrange=size(x1))
                            if event !== nothing
                                wait(event)
                            end
                            push!(results, result)
                        end
                    end
                else
                    # GPU版本：批量处理
                    row_idx, col_idx = get_triu_indices(layer.in_dim, device_backend)
                    x1 = view(x, :, row_idx)
                    x2 = view(x, :, col_idx)
                    result = similar(x1)
                    event = kernel(result, x1, x2, ndrange=size(x1))
                    if event !== nothing
                        wait(event)
                    end
                    push!(results, result)
                end
            end
        end
    end
    
    return hcat(results...)
end

# 添加 get_backend 函数
function get_backend(x::AbstractArray)
    if x isa CuArray
        return CUDA.CUDABackend()
    elseif x isa ROCArray
        return AMDGPU.ROCBackend()
    elseif x isa oneArray
        return oneAPI.oneBackend()
    else
        return KA.CPU()
    end
end

# 修改PSRN结构
mutable struct PSRN
    n_variables::Int
    operators::Vector{Operator}
    n_symbol_layers::Int
    layers::Vector{SymbolLayer}
    current_exprs::Vector{Expression}  # 改用Expression而不是String
    out_dim::Int
    backend::Any
    options::Options  # 添加Options用于构建表达式
    
    function PSRN(;
        n_variables::Int=1,
        operators::Vector{String}=["Add", "Mul", "Identity", "Sin", "Exp", "Neg", "Inv"],
        n_symbol_layers::Int=2,
        backend=KA.CPU(),
        initial_expressions=nothing  # 移除类型标注，使其更灵活
    )
        # 设置更完整的Options
        options = Options(;
            binary_operators=[+, -, *, /, ^],  # 添加更多二元操作符
            unary_operators=[cos, exp, sin, log], # 添加更多一元操作符
            populations=20,  # 可以设置种群大小
            parsimony=0.0001  # 添加复杂度惩罚
        )
        
        layers = SymbolLayer[]
        for i in 1:n_symbol_layers
            in_dim = i == 1 ? n_variables : layers[end].out_dim
            layer = SymbolLayer(in_dim, operators)
            init_offset(layer, backend)
            push!(layers, layer)
        end
        
        # 创建初始表达式
        variable_names = ["x$i" for i in 1:n_variables]
        
        # 根据initial_expressions的类型来处理
        current_exprs = if isnothing(initial_expressions)
            # 默认使用变量表达式
            [Expression(
                Node(Float32; feature=i);
                operators=options.operators,
                variable_names=variable_names
            ) for i in 1:n_variables]
        elseif initial_expressions isa Vector{Node}
            # 如果是Node数组，转换为Expression数组
            [Expression(
                node;
                operators=options.operators,
                variable_names=variable_names
            ) for node in initial_expressions]
        elseif initial_expressions isa Vector{Expression}
            # 如果已经是Expression数组，直接使用
            initial_expressions
        else
            throw(ArgumentError("initial_expressions must be Nothing, Vector{Node}, or Vector{Expression}"))
        end
        
        operator_list = [OPERATORS[name] for name in operators]
        
        new(n_variables, operator_list, n_symbol_layers, layers,
            current_exprs, layers[end].out_dim, backend, options)
    end
end


function _get_expr(psrn::PSRN, index::Int, layer_idx::Int)
    if layer_idx < 1
        return psrn.current_exprs[index]
    end
    
    layer = psrn.layers[layer_idx]
    op, offsets = get_op_and_offset(layer, index)
    
    # 获取子表达式
    expr1 = _get_expr(psrn, offsets[1], layer_idx-1)
    T = eltype(expr1.tree)  # 获取表达式的类型
    
    if op isa UnaryOperator
        # 创建一元操作表达式
        return op.op(expr1)
    else
        # 创建二元操作表达式
        expr2 = _get_expr(psrn, offsets[2], layer_idx-1)
        return op.op(expr1, expr2)
    end
end

# 修改 get_expr 方法
function get_expr(psrn::PSRN, index::Int)
    return _get_expr(psrn, index, length(psrn.layers))
end

# 添加PSRN的前向传播函数
function forward(psrn::PSRN, x::AbstractArray{T}) where T
    # 检查输入维度
    size(x, 2) == psrn.n_variables || throw(DimensionMismatch(
        "Input should have $(psrn.n_variables) features, got $(size(x, 2))"
    ))
    
    # 确保数据在正确的设备上
    x_device = to_device(x, psrn.backend)
    
    # 前向传播
    h = x_device
    for layer in psrn.layers
        h = forward(layer, h, psrn.backend)
    end
    return h
end

# 需要添加get_out_dim_cum_ls方法(类似Python版本)
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

# 需要添加init_offset方法
function init_offset(layer::SymbolLayer, backend)
    layer.offset_tensor = get_offset_tensor(layer, backend)
end

# 需要添加get_offset_tensor方法
function get_offset_tensor(layer::SymbolLayer, backend)
    offset_tensor = zeros(Int, layer.out_dim, 2)
    arange_tensor = collect(1:layer.in_dim)
    
    binary_U_tensor = zeros(Int, layer.n_triu, 2)
    binary_D_tensor = zeros(Int, layer.in_dim_square, 2)
    unary_tensor = zeros(Int, layer.in_dim, 2)
    
    unary_tensor[:, 1] = arange_tensor
    unary_tensor[:, 2] .= layer.in_dim
    
    # 填充binary_U_tensor(无向二元操作的索引)
    start = 1
    for i in 1:layer.in_dim
        len = layer.in_dim - i + 1
        binary_U_tensor[start:start+len-1, 1] .= i
        binary_U_tensor[start:start+len-1, 2] = i:layer.in_dim
        start += len
    end
    
    # 填充binary_D_tensor(有向二元操作的索引)
    start = 1
    for i in 1:layer.in_dim
        len = layer.in_dim
        binary_D_tensor[start:start+len-1, 1] .= i
        binary_D_tensor[start:start+len-1, 2] = 1:layer.in_dim
        start += len
    end
    
    # 组合所有索引
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

# 添加打印方法
function Base.show(io::IO, psrn::PSRN)
    print(io, "PSRN(n_variables=$(psrn.n_variables), operators=$(psrn.operators), " *
              "n_layers=$(psrn.n_symbol_layers))\n")
    print(io, "Layer dimensions: ")
    print(io, join([layer.out_dim for layer in psrn.layers], " → "))
end

function to_device(psrn::PSRN, backend)
    # 创建新的PSRN实例并更新backend
    new_psrn = PSRN(
        n_variables=psrn.n_variables,
        operators=[op.name for op in psrn.operators],
        n_symbol_layers=psrn.n_symbol_layers,
        backend=backend,
        initial_expressions=psrn.current_exprs  # 传递当前表达式
    )
    return new_psrn
end

function get_square_indices(n::Int, backend)
    if backend isa KA.CPU
        indices = Tuple{Int,Int}[]
        for i in 1:n
            for j in 1:n
                push!(indices, (i,j))
            end
        end
        return indices
    else
        # GPU版本返回两个向量
        row_idx = Int[]
        col_idx = Int[]
        for i in 1:n
            for j in 1:n
                push!(row_idx, i)
                push!(col_idx, j)
            end
        end
        return (to_device(row_idx, backend), to_device(col_idx, backend))
    end
end

# 修改 get_preferred_backend 函数
function get_preferred_backend()
    if CUDA.functional()
        return CUDA.CUDABackend()
    elseif @isdefined(AMDGPU) && AMDGPU.functional()
        return AMDGPU.ROCBackend()
    elseif @isdefined(oneAPI) && oneAPI.functional()
        return oneAPI.oneBackend()
    end
    return KernelAbstractions.CPU()
end

# 修改 to_device 函数
function to_device(x::AbstractArray, backend::Union{Module,KA.Backend})
    if backend isa KA.GPU
        if CUDA.functional()
            return CuArray(x)
        elseif @isdefined(AMDGPU) && AMDGPU.functional()
            return ROCArray(x)
        elseif @isdefined(oneAPI) && oneAPI.functional()
            return oneArray(x)
        end
    end
    return Array(x)
end

# 添加find_best_indices函数
function find_best_indices(outputs::AbstractArray, y::AbstractArray; top_k::Int=100)
    # 确保y在正确的设备上
    backend = outputs isa CUDA.CuArray ? CUDA : CPU
    y_device = to_device(y, backend)
    
    # 计算每个输出与目标值的MSE
    n_samples = size(outputs, 1)
    n_expressions = size(outputs, 2)
    
    # 初始化误差累加器
    sum_squared_errors = CUDA.zeros(eltype(outputs), n_expressions)
    
    # 计算每个表达式的MSE
    for i in 1:n_samples
        diff = outputs[i, :] .- y_device[i]
        sum_squared_errors .+= diff .^ 2
    end
    mean_squared_errors = sum_squared_errors ./ n_samples
    @info "Mean squared errors before handling NaN/Inf" mean_squared_errors
    
    # 将数据移回CPU进行处理
    mean_squared_errors_cpu = Array(mean_squared_errors)
    
    # 在CPU上处理无效值
    mean_squared_errors_cpu[isnan.(mean_squared_errors_cpu)] .= Inf32
    mean_squared_errors_cpu[isinf.(mean_squared_errors_cpu)] .= Inf32
    
    @info "Mean squared errors after handling NaN/Inf" mean_squared_errors_cpu
    
    # 找到top_k个最小的MSE对应的索引
    sorted_indices = partialsortperm(mean_squared_errors_cpu, 1:min(top_k, length(mean_squared_errors_cpu)))
    
    # 返回索引和对应的MSE值
    return sorted_indices, mean_squared_errors_cpu[sorted_indices]
end

# 添加get_best_expressions函数
function get_best_expressions(psrn::PSRN, X::AbstractArray, y::AbstractArray; top_k::Int=100)
    # 确保X和y在正确的设备上
    backend = get_preferred_backend()
    X_device = to_device(X, backend)
    
    # 前向传播获取所有输出
    outputs = forward(psrn, X_device)
    
    # 找到最佳的索引
    best_indices, mse_values = find_best_indices(outputs, y; top_k=top_k)
    
    # 获取对应的表达式
    best_expressions = [get_expr(psrn, idx) for idx in best_indices]
    
    # 打印结果
    println("Best expressions:")
    println("-"^20)
    for (expr, mse) in zip(best_expressions, mse_values)
        println("MSE: ", mse, " | Expression: ", expr)
    end
    println("-"^20)
    
    return best_expressions, mse_values
end

export PSRN, forward, get_expr, to_device, find_best_indices, get_best_expressions

end