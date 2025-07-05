using SymbolicUtils, SymbolicRegression
using DynamicExpressions


# tree = Node("x1") * Node("x1")* Node("x1")* Node("x1")
# eqn = convert(SymbolicUtils.Symbolic, tree, options)
# tree_copy = convert(Node, eqn, options)
# println(tree_copy)  # =(x1 * x1)


function replace_base_expressions(
    nodes_from_python::Any,
    current_expr_node_ls::Any
)
    nodes_from_python_replaced = Node[]

    for python_node in nodes_from_python
        # For each high-level tree, perform the recursive replacement
        replaced_node = _recursive_replace(python_node, current_expr_node_ls)
        push!(nodes_from_python_replaced, replaced_node)
    end

    return nodes_from_python_replaced
end


function _recursive_replace(node::Any, base_expressions::Any)
    # Base Case 1: If the node is a constant, return it as is.
    if node.constant
        return node
    end

    # Base Case 2: If the node is a variable (from Python, e.g., v_i),
    # this is where the replacement happens.
    # Your parser correctly converts v_i to feature=(i+1).
    if node.degree == 0 && !node.constant
        feature_index = node.feature

        # Check if the index is valid for our base expressions list.
        if 1 <= feature_index <= length(base_expressions)
            # Replace this variable leaf with the entire corresponding tree.
            # We return a copy to avoid aliasing issues if the same base
            # expression is used multiple times.
            return copy_node(base_expressions[feature_index])
        else
            # This case should ideally not happen if Python and Julia are in sync.
            # It means Python asked for a variable v_i for which we have no base expression.
            @warn "Feature index $feature_index from Python is out of bounds for the base expression list (size $(length(base_expressions))). The original variable node will be kept."
            return node
        end
    end

    # Recursive Step: If the node is an operator, process its children.
    if node.degree == 1
        # Unary operator
        new_l = _recursive_replace(node.l, base_expressions)
        return Node(node.op, new_l)
    elseif node.degree == 2
        # Binary operator
        new_l = _recursive_replace(node.l, base_expressions)
        new_r = _recursive_replace(node.r, base_expressions)
        return Node(node.op, new_l, new_r)
    else
        # Should not happen for standard operators.
        @warn "Encountered a node with unsupported degree: $(node.degree). Returning as is."
        return node
    end
end



operators = OperatorEnum(1 => (sin, cos, exp, safe_log), 2 => (+, -, *, /))
variable_names = ["x1", "x2"]
python_variable_names = ["v1", "v2", "v3", "v4"]



x1 = Expression(Node{Float64}(feature=1); operators, variable_names)
x2 = Expression(Node{Float64}(feature=2); operators, variable_names)

v1 = Expression(Node{Float64}(feature=1); operators, python_variable_names)
v2 = Expression(Node{Float64}(feature=2); operators, python_variable_names)
v3 = Expression(Node{Float64}(feature=3); operators, python_variable_names)
v4 = Expression(Node{Float64}(feature=4); operators, python_variable_names)

expression = x1 * cos(x2 - 3.2)

# X = randn(Float64, 2, 100);
# expression(X) # 100-element Vector{Float64}

b1 = x1 + x2
b2 = x1 * sin(x2)
b3 = x2 - exp(x2)
b4 = x2 / x1
base_expressions = [b1, b2, b3, b4]

length(base_expressions) == length(python_variable_names)

p1 = v1 - v2
p2 = v3 - v4

# 预期 想要得到： [p1, p2]
# [v1 - v2, v3 - v4]
# [b1 - b2, b3 - b4]
# [x1 + x2 - x1 * sin(x2), x2 - exp(x2) - x2 / x1]
