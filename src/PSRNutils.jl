
using Symbolics: expand, flatten_fractions, quick_cancel
using ..CoreModule: Dataset, AbstractOptions, Options
using Random

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
        1 <= comp <= 10
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

    # end 
    @info "Selected subtrees: ================ "
    @info "👇"
    for expr in top_subtrees
        # expr type is node
        string = string_tree(expr, options)
        @info string
    end 
    @info "👆"

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
        return nothing
    end
    
    # 按文件名排序，处理最早的信号
    sort!(signal_files)
    oldest_signal = signal_files[1]
    
    try
        # 读取结果
        println("Julia: Found signal file $oldest_signal, reading result...")
        expr_list = receive_string_list(fifo_in)
        
        # 删除信号文件
        rm(oldest_signal)
        ASYNC_STATE.pending_requests = max(0, ASYNC_STATE.pending_requests - 1)
        
        return expr_list
        
    catch e
        @warn "Error reading result after signal" exception=(e, catch_backtrace())
        # 即使读取失败也要删除信号文件，避免死循环
        try
            rm(oldest_signal)
        catch
        end
        return String[]
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
    y_sampled::Vector{<:AbstractFloat}
)
    try
        # 首先检查是否有之前的结果可读
        expr_list = check_for_results(fifo_in)

        if expr_list !== nothing && !isempty(expr_list)
            println("Julia: Successfully received $(length(expr_list)) expressions from Python")
            
            open("julia_get.log", "a") do f
                write(f, "Received $(length(expr_list)) expressions from Python\n")
                # 只记录前3个表达式
                for (i, expr) in enumerate(expr_list[1:min(3, length(expr_list))])
                    write(f, "  [$i]: $expr\n")
                end
                write(f, "\n")
            end
        else
            open("julia_get.log", "a") do f
                write(f, "no data\n")
            end
        end
        
        # 发送新的数据到 Python（Python会自动丢弃积压的旧数据）
        trigger_value = rand(Float64)
        write(fifo_out, trigger_value)
        send_array(fifo_out, X_mapped_sampled)
        send_array(fifo_out, y_sampled)
        
        # 不执行flush，避免阻塞
        
        ASYNC_STATE.pending_requests += 1
        println("Julia: Data sent to Python (#$(ASYNC_STATE.pending_requests)), Python will process latest data only")
        
    catch e
        @warn "Communication error in Julia" exception=(e, catch_backtrace())
    end
end