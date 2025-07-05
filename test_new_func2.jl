using SymbolicUtils
using SymbolicRegression
using DynamicExpressions

# We revert to using the explicit `Node(...)` constructor, which is
# exported by DynamicExpressions and avoids the `UndefVarError`.
function _recursive_replace(node::AbstractNode, base_expression_trees::Vector{<:AbstractNode})
    if node.constant
        return node
    end

    if node.degree == 0 && !node.constant
        feature_index = node.feature
        if 1 <= feature_index <= length(base_expression_trees)
            return copy_node(base_expression_trees[feature_index])
        else
            @warn "Feature index $feature_index is out of bounds. Keeping original variable."
            return node
        end
    end

    if node.degree == 1
        new_l = _recursive_replace(node.l, base_expression_trees)
        return Node(node.op, new_l) # <--- Reverted to Node(...)
    elseif node.degree == 2
        new_l = _recursive_replace(node.l, base_expression_trees)
        new_r = _recursive_replace(node.r, base_expression_trees)
        return Node(node.op, new_l, new_r) # <--- Reverted to Node(...)
    else
        @warn "Unsupported degree: $(node.degree). Returning as is."
        return node
    end
end


function replace_base_expressions(
    high_level_expressions::Vector{<:Expression},
    base_expressions::Vector{<:Expression}
)
    if isempty(high_level_expressions) || isempty(base_expressions)
        return Expression[]
    end

    base_expression_trees = [get_contents(expr) for expr in base_expressions]
    final_expr_template = base_expressions[1]

    return [
        begin
            high_level_tree = get_contents(high_level_expr)
            replaced_tree = _recursive_replace(high_level_tree, base_expression_trees)
            with_contents(final_expr_template, replaced_tree)
        end
        for high_level_expr in high_level_expressions
    ]
end


# --- 测试运行 (Test Run) ---

# 1. 定义基础环境
operators = OperatorEnum(1 => (sin, cos, exp, safe_log), 2 => (+, -, *, /))
variable_names = ["x1", "x2"]
x1 = Expression(Node{Float64}(feature=1); operators, variable_names)
x2 = Expression(Node{Float64}(feature=2); operators, variable_names)

# 2. 定义基础表达式 (b_i)
b1 = x1 + x2
b2 = x1 * sin(x2)
b3 = x2 - exp(x2)
b4 = x2 / x1
base_expressions = [b1, b2, b3, b4]

println("--- Base Expressions (b_i) ---")
for (i, expr) in enumerate(base_expressions)
    println("b$i = $expr")
end
println("-"^30)

# 3. 定义高阶表达式 (p_i)，它们使用占位符变量 v_i
python_variable_names = ["v1", "v2", "v3", "v4"]
v1 = Expression(Node{Float64}(feature=1); operators, variable_names=python_variable_names)
v2 = Expression(Node{Float64}(feature=2); operators, variable_names=python_variable_names)
v3 = Expression(Node{Float64}(feature=3); operators, variable_names=python_variable_names)
v4 = Expression(Node{Float64}(feature=4); operators, variable_names=python_variable_names)

p1 = v1 - v2
p2 = v3 - v4
python_expressions = [p1, p2]

println("--- High-Level Expressions (p_i) before replacement ---")
println("p1 = $p1")
println("p2 = $p2")
println("-"^30)

# 4. 执行替换
final_expressions = replace_base_expressions(python_expressions, base_expressions)

# 5. 打印结果并验证
println("--- Final Expressions after replacement ---")
println("Expected p1: (x1 + x2) - (x1 * sin(x2))")
println("Actual p1:   ", final_expressions[1])
println()
println("Expected p2: (x2 - exp(x2)) - (x2 / x1)")
println("Actual p2:   ", final_expressions[2])
println("-"^30)

# 验证表达式的正确性
@assert string(final_expressions[1]) == "((x1 + x2) - (x1 * sin(x2)))"
@assert string(final_expressions[2]) == "((x2 - exp(x2)) - (x2 / x1))"
println("All assertions passed. The replacement was successful!")

# 我们再测试一个更复杂的例子，其中一个基础表达式被多次使用
p3 = v1 / (v1 + v2)
final_p3_list = replace_base_expressions([p3], base_expressions)
final_p3 = final_p3_list[1]


println("\n--- More Complex Test ---")
println("High-level p3 = $p3")
println("Expected final p3: (x1 + x2) / ((x1 + x2) + (x1 * sin(x2)))")
println("Actual final p3:   ", final_p3)
@assert string(final_p3) == "((x1 + x2) / ((x1 + x2) + (x1 * sin(x2))))"
println("Complex test passed!")