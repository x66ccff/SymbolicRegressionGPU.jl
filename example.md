using SymbolicUtils, SymbolicRegression
options = Options(binary_operators=(+, *), unary_operators=(square, cube))
tree = Node("x1") * Node("x1")* Node("x1")* Node("x1")
eqn = convert(SymbolicUtils.Symbolic, tree, options)
tree_copy = convert(Node, eqn, options)
println(tree_copy)  # =(x1 * x1)