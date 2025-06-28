
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
