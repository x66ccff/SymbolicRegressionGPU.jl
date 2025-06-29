using SymbolicRegression


# before: 2e+02 3e+02
# after: 6e+03 5e+03

X = randn(Float32, 5, 10000) # speed: ~2e+03 for default ~1e+04 for -t 16
# X = randn(Float32, 5, 100) # speed: ~1e+05 for default ~4e+05 for -t 16
y = 2 * cos.(X[4, :]) + X[1, :] .^ 2 .- 2

options = SymbolicRegression.Options(;
    binary_operators=[+, *, /, -], unary_operators=[cos, exp, sin, log], timeout_in_seconds=30
)

hall_of_fame = equation_search(X, y; options=options, parallelism=:multithreading)

dominating = calculate_pareto_frontier(hall_of_fame)

trees = [member.tree for member in dominating]

tree = trees[end]
output, did_succeed = eval_tree_array(tree, X, options)

println("Complexity\tMSE\tEquation")

for member in dominating
    complexity = compute_complexity(member, options)
    loss = member.loss
    string = string_tree(member.tree, options)

    println("$(complexity)\t$(loss)\t$(string)")
end
