# export JULIA_NUM_THREADS=4
# export JULIA_DEBUG=loading
# julia --project=. example.jl
# julia example.jl

using SymbolicRegressionGPU
using LoopVectorization

function main()
    X = randn(Float32, 5, 100)
    # X = randn(Float64, 5, 101) # test for Float64, passed
    # X = randn(Float64, 5, 10100) # test for PSRN downsampling, passed

    # y = 2 * cos.(X[4, :]) + X[1, :] .^ 2 .- 2
    y = 2 * cos.(X[4, :]) .^ 3 + X[1, :] .^ 2 .- 2 # harder problem

    options = SymbolicRegressionGPU.Options(;
    binary_operators=[+, *, /, -],
    unary_operators=[sin, cos, exp, log, sqrt],
    population_size=100,
    populations=15,
    batching=true,
    batch_size=100,
    adaptive_parsimony_scaling=1_000.0,
    parsimony=0.0,
    maxsize=30,
    maxdepth=20,
    turbo=true,
    early_stop_condition=(l, c) -> l < 1e-6 && c == 5,
    constraints = [
        sin => 9,
        cos => 9,
        exp => 9,
        log => 9,
        sqrt => 9
    ],
    nested_constraints = [
        sin => [
            sin => 0,
            cos => 0,
            exp => 1,
            log => 1,
            sqrt => 1
        ],
        cos => [
            sin => 0,
            cos => 0,
            exp => 1,
            log => 1,
            sqrt => 1
        ],
        exp => [
            exp => 0,
            log => 0
        ],
        log => [
            exp => 0,
            log => 0
        ],
        sqrt => [
            sqrt => 0
        ]
    ]
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
end

@time main()
