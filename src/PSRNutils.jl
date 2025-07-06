
using Symbolics: expand, flatten_fractions, quick_cancel
using ..CoreModule: Dataset, AbstractOptions, Options
using Random
using DynamicExpressions


function get_used_variables(node, var_names)
    used_vars = Set{String}()
    
    function traverse(n)
        if !isnothing(n)
            if n.constant == false && n.feature != 0x0000 && n.feature != 0xffff
                # feature从1开始索引
                if n.feature <= length(var_names)
                    push!(used_vars, var_names[n.feature])
                end
            end
            if isdefined(n, :l)
                traverse(n.l)
            end
            if isdefined(n, :r)
                traverse(n.r)
            end
        end
    end
    
    traverse(node)
    return used_vars
end

function select_top_subtrees(
    common_subtrees::Dict{Node,Float64},
    n::Int,
    options::AbstractOptions,
    n_variables::Int;
    ratio_subtrees::Float64=0.5,
    ratio_subtrees_crossover::Float64=0.4
)
    @assert ratio_subtrees + ratio_subtrees_crossover <= 1.0 "Ratios sum must be <= 1.0"

    # 先过滤掉复杂度过高或过低的子树
    filtered_subtrees = filter(pair -> begin
        node = pair.first
        comp = compute_complexity(node, options)
        1 <= comp <= 5
    end, common_subtrees)

    # 将字典转成 (node, ratio_score) 的元组数组
    filtered_pairs = collect(filtered_subtrees)

    # 如果过滤后还有可用子树
    scored_nodes = Node[]
    if !isempty(filtered_pairs)
        # 根据 ratio_score 降序排序
        sorted_pairs = sort(filtered_pairs, by = x -> x.second * (1.0 + 0.5*randn()), rev = true)
        scored_nodes = [p.first for p in sorted_pairs]
    end

    result = Node[]
    # 先用得分最高的子树填充一部分
    n_subtrees = min(floor(Int, n * ratio_subtrees), length(scored_nodes))
    for i in 1:n_subtrees
        push!(result, scored_nodes[i])
    end

    # 获取已经使用的变量
    variable_names = ["x$i" for i in 1:n_variables]
    used_variables = Set{String}()
    for node in result
        union!(used_variables, get_used_variables(node, variable_names))
    end
    
    # 获取还未使用的变量索引
    available_features = Int[]
    for i in 1:n_variables
        if !("x$i" in used_variables)
            push!(available_features, i)
        end
    end

    # 如果还没凑够，就用随机生成的树来填充
    while length(result) < n
        # if isempty(available_features)
            # 如果没有可用的feature了，就生成随机的树
            # push!(result, Node(FloatType; val=rand(-5:5)))
        tree = gen_random_tree(
            rand(1:5),                     # length
            options,              # options
            n_variables,          # nfeatures
            Float32;
            only_gen_bin_op=true,
            only_gen_int_const=true,
            feature_prob=0.7
        )
        push!(result, tree)
        # else
        #     # 随机选择一个未使用的feature
        #     feature = rand(available_features)
        #     tree = Node(FloatType; feature=feature)
            
        #     if !(tree in result)
        #         push!(result, tree)
        #         # 更新已使用的变量
        #         union!(used_variables, get_used_variables(tree, variable_names))
        #         # 从可用feature中移除已使用的
        #         filter!(f -> f != feature, available_features)
        #     end
        # end
    end

    return result
end

function evaluate_subtrees(
    subtrees::Vector{Node}, dataset::Dataset, options::AbstractOptions
)
    n_samples = size(dataset.X, 2)  # Use the number of columns as the number of samples
    n_subtrees = length(subtrees)

    # Create a result matrix - using the same type as dataset.X
    T = eltype(dataset.X)
    result = zeros(T, n_samples, n_subtrees)

    # @info "n_subtrees: $n_subtrees"
    # @info "n_samples: $n_samples"

    # Evaluate each subtree
    for (i, subtree) in enumerate(subtrees)
        if isnothing(subtree)
            result[:, i] .= one(T)
        else
            # Creates an Expression object, providing the necessary parameters
            # @info "Evaluating subtree: $subtree"  # Print the Node object first

            # Use operators in options when creating an Expression
            expr = Expression(
                subtree;
                operators=options.operators,  # Use operators in options
                variable_names=dataset.variable_names,  # Get variable_names from dataset
            )

            # Evaluate on data set X
            # @info "Starting eval_tree_array..."
            output, success = eval_tree_array(
                expr,
                dataset.X,  # Just use X, no transpose
            )
            # @info "eval_tree_array completed" success=success output_size=size(output)

            if success
                # If the output is one-dimensional, it is assigned directly to the corresponding column
                if length(output) == n_samples
                    result[:, i] = output
                    # @info "Successfully assigned output to result[:, $i]"
                else
                    @warn "Dimension mismatch: output length $(length(output)) doesn't match expected size ($n_samples). Using ones."
                    result[:, i] .= one(T)
                end
            else
                result[:, i] .= one(T)
                # @warn "eval_tree_array failed for subtree $i, using ones"
                # @warn "where the failed tree is:"
                # @warn "🔥 $(subtrees[i]) 🔥"
            end
        end
    end

    # @info "Evaluation complete" result_size=size(result)
    return result
end

"""
计算给定子树在所有表达式中的加权评分，即 sum( subtree_complexity / parent_complexity )。
返回的字典结构为：
    Dict{Node, Float64}
其中键是子树节点，值是该子树节点所对应的打分。
"""
function analyze_common_subtrees(trees::Any, options::Options)
    # 为每个子树同时记录：
    #   - 出现次数 count（若你还需要对出现次数进行筛选，可继续保留 count）
    #   - 累加的占比得分 ratio_score
    # 这里使用一个字典，值为 (count, ratio_score)
    subtree_stats = Dict{Node, Tuple{Int, Float64}}()  # Correct

    for expr in trees
        # 如果该表达式有树结构
        if !isnothing(expr.tree)
            parent_complexity = compute_complexity(expr.tree, options)
            # 获取该表达式的所有子树
            subtrees = get_subtrees(expr.tree)

            for st in subtrees
                st_comp = compute_complexity(st, options)
                # 子树对于该表达式的贡献
                contribution = st_comp / parent_complexity

                if haskey(subtree_stats, st)
                    old_count, old_ratio_score = subtree_stats[st]
                    subtree_stats[st] = (old_count + 1, old_ratio_score + contribution)
                else
                    subtree_stats[st] = (1, contribution)
                end
            end
        end
    end

    # 你所需的出现次数阈值（也可以只用 ratio_score 过滤）
    threshold = 1

    # 过滤掉出现次数太少或者复杂度过低的子树
    # 如果您不想用 count 做过滤，可以只用 ratio_score 做过滤；这里仅示例
    common_patterns = Dict{Node, Float64}()
    for (st, (count, rscore)) in subtree_stats
        if count >= threshold && compute_complexity(st, options) >= 1
            # 将 ratio_score 作为我们后续排序使用的“全局打分”
            common_patterns[st] = rscore
        end
    end

    return common_patterns
end


# Gets all the subtrees of an expression tree
# function get_subtrees(expr::Expression)
#     if isnothing(expr.tree)
#         return Node[]
#     end
#     return get_subtrees(expr.tree)
# end


function get_subtrees(expr::Expression)
    if isnothing(expr.tree)
        return Node[]
    end
    expanded = expand(expr.tree)
    flattend = flatten_fractions(expr.tree)
    canceled = quick_cancel(expr.tree)
    return vcat(
        get_subtrees(expr.tree),
        get_subtrees(expanded),
        get_subtrees(flattend),
        get_subtrees(canceled)
        ) 
end

function get_subtrees(node::Node)
    subtrees = Node[]
    if isnothing(node)
        return subtrees
    end

    push!(subtrees, node)

    # Recursive processing of left and right subtrees
    if isdefined(node, :l) && !isnothing(node.l)
        append!(subtrees, get_subtrees(node.l))
    end

    if isdefined(node, :r) && !isnothing(node.r)
        append!(subtrees, get_subtrees(node.r))
    end

    return subtrees
end

get_subtrees(x::Number) = Node[]
get_subtrees(x::Symbol) = Node[]




function psrn_preprocess(
    # dominating_trees::Vector{<:Expression},
    dominating_trees::Any,
    dataset::Dataset,
    options::AbstractOptions,
    N_PSRN_INPUT::Int,
    n_variables::Int,
    max_samples::Int,
)
    FloatType = Float32

    common_subtrees = analyze_common_subtrees(dominating_trees, options)
    top_subtrees = select_top_subtrees(common_subtrees, N_PSRN_INPUT, options, n_variables)
    shuffle!(top_subtrees)

    # # end 
    # @info "Selected subtrees: ================ "
    # @info "👇"
    # for expr in top_subtrees
    #     # expr type is node
    #     string = string_tree(expr, options)
    #     @info string
    # end 
    # @info "👆"

    X_mapped = evaluate_subtrees(top_subtrees, dataset, options)

    # add downsampling 
    n_samples = size(X_mapped, 1)
    if n_samples > max_samples
        # random sample
        sample_indices = randperm(n_samples)[1:(max_samples)]
        X_mapped_sampled = X_mapped[sample_indices, :]

        # check the dimension of dataset.y
        y_dims = size(dataset.y)
        if length(y_dims) == 1
            y_sampled = dataset.y[sample_indices]
        else
            y_sampled = dataset.y[:, sample_indices]
        end
    else
        X_mapped_sampled = X_mapped
        y_sampled = dataset.y
    end

    # add debug info
    # @info "Dimensions:" X_mapped_size=size(X_mapped_sampled) y_size=size(y_sampled)
    # to cuda 0
    X_mapped_sampled = FloatType.(X_mapped_sampled) # for saving memory
    y_sampled = FloatType.(y_sampled) # for saving memory

    n_variables = size(X_mapped_sampled, 2)
    variable_names = ["x$i" for i in 1:n_variables]
    current_expr_ls = if isnothing(top_subtrees)
        # Variable expressions are used by default
        [
            Expression(
                Node(FloatType; feature=i);
                operators=options.operators,
                variable_names=variable_names,
            ) for i in 1:n_variables
        ]
    elseif top_subtrees isa Vector{Node}
        # If it is a Node array, convert it to an Expression array
        [
            Expression(
                node; operators=options.operators, variable_names=variable_names
            ) for node in top_subtrees
        ]
    elseif top_subtrees isa Vector{Expression}
        # If it is already an Expression array, use it directly
        top_subtrees
    else
        throw(
            ArgumentError(
                "top_subtrees must be Nothing, Vector{Node}, or Vector{Expression}",
            ),
        )
    end

    # best_expressions = get_best_expr_and_MSE_topk(
    #     X_mapped_sampled, y_sampled
    # )
    return X_mapped_sampled, y_sampled, current_expr_ls
    # return best_expressions
end


# 全局变量来跟踪异步状态
mutable struct AsyncState
    pending_requests::Int64
    last_signal_check::Int64
end

# 全局状态实例
const ASYNC_STATE = AsyncState(0, 0)

"""
序列化并发送一个数组到指定的IO流。
协议:
1. 写入维度数 (Int64)
2. 依次写入每个维度的大小 (Int64)
3. 写入整个数组的原始数据
"""
function send_array(fifo_out::IO, arr::AbstractArray{<:AbstractFloat})
    # 确保数据类型是 Float64，与 Python 端匹配
    arr_f64 = convert(Array{Float64}, arr)

    # 1. 发送维度数量
    num_dims = Int64(ndims(arr_f64))
    write(fifo_out, num_dims)

    # 2. 发送每个维度的大小
    dims = Int64.(size(arr_f64))
    
    # ---- 这是修正的部分 ----
    # 错误行：write(fifo_out, dims) -> 不能直接写入元组
    # 修正：遍历元组，将每个维度的大小单独写入
    for d in dims
        write(fifo_out, d)
    end
    # -------------------------

    # 3. 发送扁平化的数组数据
    # Julia 的 write 函数可以直接处理数组，它会按列主序（column-major）写入
    write(fifo_out, arr_f64)
end

"""
从指定的IO流接收并反序列化字符串列表。
协议:
1. 读取字符串列表长度 (Int64)
2. 对于每个字符串：
   - 读取字符串长度 (Int64)
   - 读取字符串的UTF-8字节并转换为字符串
"""
function receive_string_list(fifo_in::IO)
    try
        # 1. 读取字符串列表长度
        list_length = read(fifo_in, Int64)
        println("Julia: Reading string list of length: $list_length")
        
        # 2. 读取每个字符串
        string_list = String[]
        for i in 1:list_length
            # 读取字符串长度
            str_length = read(fifo_in, Int64)
            # 读取字符串字节
            str_bytes = read(fifo_in, str_length)
            # 转换为字符串
            str_content = String(str_bytes)
            push!(string_list, str_content)
        end
        
        return string_list
        
    catch e
        @warn "Error in receive_string_list" exception=(e, catch_backtrace())
        return String[]
    end
end

"""
检查是否有新的结果可读
"""
function check_for_results(fifo_in::IO)
    signal_files = filter(x -> startswith(x, "python_result_ready_"), readdir("."))
    
    if isempty(signal_files)
        return nothing, nothing
    end
    
    # 按文件名排序，处理最早的信号
    sort!(signal_files)
    oldest_signal = signal_files[1]
    
    try
        # 从文件名中解析索引，这个索引现在就是我们发送的 global_index
        signal_index = tryparse(Int, replace(oldest_signal, "python_result_ready_" => ""))
        
        # 读取结果
        println("Julia: Found signal file $oldest_signal (for index $signal_index), reading result...")
        expr_list = receive_string_list(fifo_in)
        
        # 删除信号文件
        rm(oldest_signal)
        ASYNC_STATE.pending_requests = max(0, ASYNC_STATE.pending_requests - 1)
        
        return expr_list, signal_index
        
    catch e
        @warn "Error reading result after signal" exception=(e, catch_backtrace())
        # 即使读取失败也要删除信号文件，避免死循环
        try
            rm(oldest_signal)
        catch
        end
        return String[], nothing
    end
end

"""
尝试读取数据，使用阻塞读取而不是检查bytesavailable
"""
function safe_receive_string_list(fifo_in::IO, timeout_seconds::Float64 = 30.0)
    # 创建一个任务来执行读取操作
    read_task = @async begin
        try
            return receive_string_list(fifo_in)
        catch e
            @warn "Error in async read" exception=(e, catch_backtrace())
            return String[]
        end
    end
    
    # 等待任务完成或超时
    result = nothing
    elapsed = 0.0
    while elapsed < timeout_seconds
        if istaskdone(read_task)
            result = fetch(read_task)
            break
        end
        sleep(0.1)
        elapsed += 0.1
    end
    
    if result === nothing
        # 超时了，尝试取消任务
        println("Julia: Timeout occurred, cancelling read task")
        return String[]
    end
    
    return result
end

function communicate_with_python(
    fifo_out::Any,
    fifo_in::Any,
    X_mapped_sampled::Matrix{<:AbstractFloat},
    y_sampled::Vector{<:AbstractFloat},
    options::AbstractOptions,
    global_index::Int64,
    current_expr_ls::Any
)
    # FIX 1: Initialize a typed vector of Nodes, not a Vector{Any}
    expr_from_python = String[]

    try
        # First, check for any results that might be ready
        expr_list, received_index = check_for_results(fifo_in)

        if expr_list !== nothing && !isempty(expr_list)
            println("🔍Julia: Successfully received $(length(expr_list)) expressions from Python for index #$received_index")
            
            open("julia_get.log", "a") do f
                write(f, "🔍Received $(length(expr_list)) expressions from Python for index #$received_index\n")
                for (i, expr) in enumerate(expr_list)
                    write(f, "      [$i]: $expr\n")
                    push!(expr_from_python, expr)
                end
                write(f, "\n")
            end
        else
            open("julia_get.log", "a") do f
                write(f, "🔍no data\n")
            end
        end
        
        # ---- MODIFIED PART ----
        # Send new data to Python with the new protocol.
        # Protocol: global_index (Int64) -> X_array -> y_array
        
        # 1. Send the global_index
        write(fifo_out, global_index)
        
        # 2. Send the X array
        send_array(fifo_out, X_mapped_sampled)
        
        # 3. Send the y vector
        send_array(fifo_out, y_sampled)

        # Ensure data is sent immediately
        flush(fifo_out)
        # ---- END MODIFIED PART ----
        
        if global_index == 0
            empty!(history_subtrees_list)
            empty!(history_number_list)
        end
        push!(history_subtrees_list, deepcopy(current_expr_ls))
        push!(history_number_list, global_index)


        open("julia_send.log", "a") do f
            write(f, "writing global_index = $(global_index) \n")
            write(f, "now length(history_subtrees_list) = $(length(history_subtrees_list)) \n")
            write(f, "now length(history_number_list) = $(length(history_number_list)) \n")
            for (i, expr) in enumerate(current_expr_ls)
                write(f, "      [$i]: $expr\n")
            end
            write(f, "\n")
        end

        ASYNC_STATE.pending_requests += 1
        # println("🔍Julia: Data with index #$global_index sent to Python. Pending requests: $(ASYNC_STATE.pending_requests)")
            # This function now correctly returns a `Vector{Node}`
        return expr_from_python, received_index
    catch e
        @warn "🔍Communication error in Julia" exception=(e, catch_backtrace())
    end

    return nothing, nothing
end
"""
通用的表达式解析器 - 支持SR.jl的所有运算符
"""
struct ExpressionParser
    binary_operators::Vector
    unary_operators::Vector
    operator_symbols::Dict{String, Any}
    function_names::Vector{String}
    
    function ExpressionParser(options::AbstractOptions)
        binary_ops = collect(options.operators.ops[2])
        unary_ops = collect(options.operators.ops[1])
        
        # 自动构建运算符符号映射
        symbols = build_operator_symbol_map(binary_ops, unary_ops)
        
        # 提取所有函数名用于tokenizer
        func_names = collect(keys(filter(p -> !is_binary_symbol(p.first), symbols)))
        
        new(binary_ops, unary_ops, symbols, func_names)
    end
end

"""
自动构建运算符符号到函数的映射
"""
function build_operator_symbol_map(binary_operators, unary_operators)
    symbols = Dict{String, Any}()
    
    # 标准二元运算符映射
    standard_binary = Dict(
        "+" => (+),
        "-" => (-), 
        "*" => (*),
        "/" => (/),
        "^" => (^),
        "%" => (%),
        "&" => (&),
        "|" => (|),
        ">" => (>),
        "<" => (<),
        "==" => (==),
        "!=" => (!=),
        ">=" => (>=),
        "<=" => (<=)
    )
    
    # 添加找到的二元运算符
    for op in binary_operators
        for (symbol, func) in standard_binary
            if op === func
                symbols[symbol] = op
            end
        end
        
        # 特殊处理safe_pow -> ^
        op_str = string(op)
        if contains(op_str, "safe_pow")
            symbols["^"] = op
        end
    end
    
    # 自动检测一元运算符
    for op in unary_operators
        name = get_function_name(op)
        if !isnothing(name)
            symbols[name] = op
        end
    end
    
    return symbols
end

"""
获取函数的名称字符串 - 改进版本
"""
function get_function_name(func)
    func_str = string(func)
    
    # 特殊处理SymbolicRegression的safe函数
    if contains(func_str, "safe_log")
        return "log"
    elseif contains(func_str, "safe_sqrt")
        return "sqrt"
    elseif contains(func_str, "safe_pow")
        return "pow"
    elseif contains(func_str, "safe_exp")
        return "exp"
    end
    
    # 标准函数映射
    known_functions = Dict(
        sin => "sin",
        cos => "cos", 
        tan => "tan",
        exp => "exp",
        log => "log",
        sqrt => "sqrt",
        abs => "abs",
        floor => "floor",
        ceil => "ceil",
        round => "round",
        tanh => "tanh",
        sinh => "sinh",
        cosh => "cosh",
        atan => "atan",
        asin => "asin",
        acos => "acos"
    )
    
    # 直接匹配已知函数
    if haskey(known_functions, func)
        return known_functions[func]
    end
    
    # 正则表达式匹配函数名
    patterns = [
        r"^([a-zA-Z_][a-zA-Z0-9_]*)",
        r"\.([a-zA-Z_][a-zA-Z0-9_]*)$"
    ]
    
    for pattern in patterns
        m = match(pattern, func_str)
        if !isnothing(m) && length(m.captures) > 0
            name = m.captures[1]
            if name ∉ ["typeof", "Module", "Function", "Core", "Base", "Main"]
                return name
            end
        end
    end
    
    return nothing
end

"""
判断是否为二元运算符符号
"""
function is_binary_symbol(symbol::String)
    return symbol in ["+", "-", "*", "/", "^", "%", "&", "|", ">", "<", "==", "!=", ">=", "<="]
end

"""
改进的tokenizer - 修复括号和函数识别问题
"""
function tokenize_expression(expr_str::String, parser::ExpressionParser)
    tokens = String[]
    i = 1
    n = length(expr_str)
    
    while i <= n
        c = expr_str[i]
        
        # 跳过空白
        if isspace(c)
            i += 1
            continue
        end
        
        # 括号
        if c in ['(', ')']
            push!(tokens, string(c))
            i += 1
        # 多字符运算符（如 ==, !=, >=, <=）
        elseif i + 1 <= n
            two_char = expr_str[i:i+1]
            if two_char in keys(parser.operator_symbols)
                push!(tokens, two_char)
                i += 2
            # 单字符运算符
            elseif string(c) in keys(parser.operator_symbols)
                push!(tokens, string(c))
                i += 1
            # 字母开头的token（函数名或变量）
            elseif isletter(c) || c == '_'
                j = i
                while j <= n && (isletter(expr_str[j]) || isdigit(expr_str[j]) || expr_str[j] == '_')
                    j += 1
                end
                token = expr_str[i:j-1]
                push!(tokens, token)
                i = j
            # 数字
            elseif isdigit(c) || c == '.'
                j = i
                while j <= n && (isdigit(expr_str[j]) || expr_str[j] == '.')
                    j += 1
                end
                push!(tokens, expr_str[i:j-1])
                i = j
            else
                i += 1
            end
        # 单字符处理
        elseif string(c) in keys(parser.operator_symbols)
            push!(tokens, string(c))
            i += 1
        elseif isletter(c) || c == '_'
            j = i
            while j <= n && (isletter(expr_str[j]) || isdigit(expr_str[j]) || expr_str[j] == '_')
                j += 1
            end
            token = expr_str[i:j-1]
            push!(tokens, token)
            i = j
        elseif isdigit(c) || c == '.'
            j = i
            while j <= n && (isdigit(expr_str[j]) || expr_str[j] == '.')
                j += 1
            end
            push!(tokens, expr_str[i:j-1])
            i = j
        else
            i += 1
        end
    end
    
    return tokens
end

"""
通用表达式解析器 - 主函数
"""
function convert_python_tree_to_nodes(expr_str::String, options::AbstractOptions)
    expr_str = String(strip(expr_str))
    
    if isempty(expr_str) || contains(expr_str, "no data")
        return nothing
    end
    
    try
        parser = ExpressionParser(options)
        tokens = tokenize_expression(expr_str, parser)
        
        if isempty(tokens)
            return nothing
        end
        
        node, remaining_tokens = parse_expression_with_parser(tokens, parser)
        if !isempty(remaining_tokens)
            @warn "Unexpected remaining tokens: $remaining_tokens"
        end
        return node
    catch e
        @warn "Failed to parse expression: $expr_str" exception=e
        return nothing
    end

end

"""
使用解析器进行表达式解析
"""
function parse_expression_with_parser(tokens::Vector{String}, parser::ExpressionParser)
    if isempty(tokens)
        error("Empty token list")
    end
    return parse_additive_with_parser(tokens, parser)
end

"""
解析加法减法
"""
function parse_additive_with_parser(tokens::Vector{String}, parser::ExpressionParser)
    node, tokens = parse_multiplicative_with_parser(tokens, parser)
    
    while !isempty(tokens) && tokens[1] in ["+", "-"]
        op = popfirst!(tokens)
        right_node, tokens = parse_multiplicative_with_parser(tokens, parser)
        node = create_binary_node_with_parser(op, node, right_node, parser)
    end
    
    return node, tokens
end

"""
解析乘法除法
"""
function parse_multiplicative_with_parser(tokens::Vector{String}, parser::ExpressionParser)
    node, tokens = parse_power_with_parser(tokens, parser)
    
    while !isempty(tokens) && tokens[1] in ["*", "/", "%"]
        op = popfirst!(tokens)
        right_node, tokens = parse_power_with_parser(tokens, parser)
        node = create_binary_node_with_parser(op, node, right_node, parser)
    end
    
    return node, tokens
end

"""
解析幂运算
"""
function parse_power_with_parser(tokens::Vector{String}, parser::ExpressionParser)
    node, tokens = parse_unary_with_parser(tokens, parser)
    
    # 幂运算是右结合的
    if !isempty(tokens) && tokens[1] == "^"
        op = popfirst!(tokens)
        right_node, tokens = parse_power_with_parser(tokens, parser)  # 递归处理右结合
        node = create_binary_node_with_parser(op, node, right_node, parser)
    end
    
    return node, tokens
end

"""
解析一元运算符 - 修复函数识别
"""
function parse_unary_with_parser(tokens::Vector{String}, parser::ExpressionParser)
    if isempty(tokens)
        error("Unexpected end of expression")
    end
    
    token = tokens[1]
    
    # 检查是否为函数名（在parser的function_names中或直接在operator_symbols中）
    if token in parser.function_names || (haskey(parser.operator_symbols, token) && !is_binary_symbol(token))
        popfirst!(tokens)
        
        if isempty(tokens) || tokens[1] != "("
            error("Expected '(' after function $token")
        end
        popfirst!(tokens)
        
        arg_node, tokens = parse_additive_with_parser(tokens, parser)
        
        if isempty(tokens) || tokens[1] != ")"
            error("Expected ')' after function argument")
        end
        popfirst!(tokens)
        
        return create_unary_node_with_parser(token, arg_node, parser), tokens
    else
        return parse_primary_with_parser(tokens, parser)
    end
end

"""
解析原子表达式
"""
function parse_primary_with_parser(tokens::Vector{String}, parser::ExpressionParser)
    if isempty(tokens)
        error("Unexpected end of expression")
    end
    
    token = tokens[1]
    
    if token == "("
        popfirst!(tokens)
        node, tokens = parse_additive_with_parser(tokens, parser)
        
        if isempty(tokens) || tokens[1] != ")"
            error("Expected ')' to close parentheses")
        end
        popfirst!(tokens)
        
        return node, tokens
    elseif startswith(token, "v_")
        popfirst!(tokens)
        var_index_str = token[3:end]
        var_index = parse(Int, var_index_str)
        return create_variable_node(var_index), tokens
    else
        popfirst!(tokens)
        try
            val = parse(Float64, token)
            return create_constant_node(val), tokens
        catch
            error("Cannot parse token as number: $token")
        end
    end
end

"""
使用解析器创建二元节点
"""
function create_binary_node_with_parser(op::String, left::Node, right::Node, parser::ExpressionParser)
    if !haskey(parser.operator_symbols, op)
        error("Unknown operator: $op")
    end

    
    target_func = parser.operator_symbols[op]
    
    for (i, binary_op) in enumerate(parser.binary_operators)
        if binary_op === target_func
            return Node(i, left, right)
        end
    end
    
    error("Operator $op not found in binary operators list")
end

"""
使用解析器创建一元节点
"""
function create_unary_node_with_parser(op::String, arg::Node, parser::ExpressionParser)
    if !haskey(parser.operator_symbols, op)
        error("Unknown operator: $op")
    end
    
    target_func = parser.operator_symbols[op]
    
    for (i, unary_op) in enumerate(parser.unary_operators)
        if unary_op === target_func
            return Node(i, arg)
        end
    end
    
    error("Operator $op not found in unary operators list")
end

# 保留原有的辅助函数
function create_variable_node(feature_index::Int)
    return Node(Float32; feature=feature_index + 1)
end

function create_constant_node(value::Float64)
    return Node(Float32; val=Float32(value))
end

function convert_python_expressions_to_nodes(expr_strings::Vector{String}, options::AbstractOptions)
    nodes = Node[]
    for expr_str in expr_strings
        node = convert_python_tree_to_nodes(expr_str, options)
        if !isnothing(node)
            push!(nodes, node)
        end
    end
    return nodes
end

"""
测试函数 - 支持任意运算符配置
"""
function test_conversion(options::AbstractOptions)
    # 创建解析器并显示支持的运算符
    parser = ExpressionParser(options)
    
    println("Testing with operators:")
    println("  Binary: ", parser.binary_operators)
    println("  Unary:  ", parser.unary_operators)
    println("  Detected symbols: ", sort(collect(keys(parser.operator_symbols))))
    println()
    
    test_exprs = [
        "(((v_1)-(v_4))*((v_0)/(v_1)))-(((v_1)/(v_1))-(exp(v_0)))",
        "sin(v_0)",
        "((v_1)+(v_2))",
        "exp(((v_0)*(v_1)))",
        "log(v_3)",
        "sqrt(v_0)",  # 测试是否支持sqrt
        "v_0^v_1",    # 测试幂运算
        "no data"
    ]
    
    for expr in test_exprs
        println("Testing: $expr")
        node = convert_python_tree_to_nodes(expr, options)
        if isnothing(node)
            println("  Result: nothing")
        else
            println("  Result: Node created successfully")
            println("  Node: $node")
            
            # 创建Expression来验证结果
            try
                expression = Expression(
                    node;
                    operators=options.operators,
                    variable_names=["x0", "x1", "x2", "x3", "x4"]
                )
                println("  Expression: $expression")
            catch e
                println("  Expression creation failed: $e")
            end
        end
        println()
    end
end

function convert_node_type(node::AbstractNode, target_T::Type)
    # --- 关键修正：明确处理所有叶子节点情况 ---
    if node.degree == 0
        if node.constant
            # It's a constant leaf node
            return Node{target_T}(; val=convert(target_T, node.val))
        else
            # It's a variable leaf node
            return Node{target_T}(; feature=node.feature)
        end
    elseif node.degree == 1
        new_l = convert_node_type(node.l, target_T)
        return Node{target_T}(; op=node.op, l=new_l)
    elseif node.degree == 2
        new_l = convert_node_type(node.l, target_T)
        new_r = convert_node_type(node.r, target_T)
        return Node{target_T}(; op=node.op, l=new_l, r=new_r)
    else
        # This should not be reached with standard operators
        error("Unsupported node degree encountered in `convert_node_type`: $(node.degree)")
    end
end


"""
    _recursive_replace(node, base_expression_trees, target_T)

Recursively traverses a high-level node tree, replacing variable nodes with corresponding
base expression trees. It ensures that all nodes in the final tree have the
type `target_T`.
"""
function _recursive_replace(
    node::AbstractNode, 
    base_expression_trees::Vector{<:AbstractNode},
    target_T::Type
)
    # Base case for constants in the high-level expression (e.g., v1 + 5.0)
    if node.constant
        return Node{target_T}(; val=convert(target_T, node.val))
    end

    # Base case for variables (placeholders) in the high-level expression
    if node.degree == 0 && !node.constant
        feature_index = node.feature
        if 1 <= feature_index <= length(base_expression_trees)
            # Get the base tree to substitute
            base_tree_to_insert = base_expression_trees[feature_index]
            # Convert the entire base tree to the target type before inserting
            return convert_node_type(base_tree_to_insert, target_T)
        else
            @warn "Feature index $feature_index is out of bounds for base expressions (size=$(length(base_expression_trees))). Keeping original variable."
            return Node{target_T}(; feature=node.feature)
        end
    end

    # Recursive step for operators
    if node.degree == 1
        new_l = _recursive_replace(node.l, base_expression_trees, target_T)
        return Node{target_T}(; op=node.op, l=new_l)
    elseif node.degree == 2
        new_l = _recursive_replace(node.l, base_expression_trees, target_T)
        new_r = _recursive_replace(node.r, base_expression_trees, target_T)
        return Node{target_T}(; op=node.op, l=new_l, r=new_r)
    else
        @warn "Unsupported degree in `_recursive_replace`: $(node.degree). Attempting to convert node as is."
        return convert_node_type(node, target_T)
    end
end


"""
    replace_base_expressions(high_level_expressions, base_expressions)

Substitutes placeholder variables in high-level expressions with corresponding base expressions.
This function is robust to type mismatches (e.g., Float32 vs Float64) between the
expression sets.
"""
function replace_base_expressions(
    high_level_expressions::Vector{<:Expression},
    base_expressions::Vector{<:Expression}
)
    # @info "替换开始: 👇👇👇👇👇👇👇👇👇👇"
    # @debug "High-level expressions to be replaced:" high_level_expressions
    # @debug "Base expressions for substitution:" base_expressions

    if isempty(high_level_expressions) || isempty(base_expressions)
        @warn "Input expressions are empty, returning an empty result."
        return Expression[]
    end

    base_expression_trees = [get_contents(expr) for expr in base_expressions]
    final_expr_template = base_expressions[1]
    template_tree = get_contents(final_expr_template)
    target_T = eltype(template_tree)
    
    # @info "确定目标节点类型为: $target_T"

    ret = [
        begin
            high_level_tree = get_contents(high_level_expr)
            replaced_tree = _recursive_replace(high_level_tree, base_expression_trees, target_T)
            with_contents(final_expr_template, replaced_tree)
        end
        for high_level_expr in high_level_expressions
    ]
    
    # @debug "Final replaced expressions:" ret
    # @info "替换结束 👆👆👆👆👆👆👆👆👆👆"

    return ret
end



history_subtrees_list = []
history_number_list = []



function replace_v_indices(strings_list::Vector{String})
    result = String[]
    
    for str in strings_list
        new_str = str
        # 匹配 v_ 后面跟数字的模式（没有花括号）
        while true
            m = match(r"v_(\d+)", new_str)
            if m === nothing
                break
            end
            index = parse(Int, m.captures[1]) + 1
            replacement = "vexprs[$index]"
            new_str = replace(new_str, m.match => replacement, count=1)
        end
        push!(result, new_str)
    end
    
    return result
end