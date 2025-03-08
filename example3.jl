# export JULIA_NUM_THREADS=4
# export JULIA_DEBUG=loading
# julia --project=. example.jl
# julia example.jl

using SymbolicRegressionGPU
using LoopVectorization
using DelimitedFiles  # 用于读取TSV文件

function main()
    # 读取TSV文件
    data = readdlm("15_3t.tsv", '\t', Float64, skipstart=1)  # skipstart=1 跳过标题行
    
    # 分离特征和目标
    X = Float32.(transpose(data[:, 1:4]))  # 取前7列作为特征,转置使其符合要求的维度
    y = Float32.(data[:, 5])  # 最后一列是target

    options = SymbolicRegressionGPU.Options(;
        timeout_in_seconds=60000000,
        binary_operators=[+, *, /, -],
        unary_operators=[sin, cos, exp, log, sqrt, square, cube],
        population_size=100,
        populations=15,
        batching=true,
        batch_size=10,
        adaptive_parsimony_scaling=1_000.0,
        parsimony=0.0,
        maxsize=40,
        maxdepth=20,
        turbo=true,
        # complexity_of_constants=3,
        # should_optimize_constants=false,
        # optimizer_iterations=4,
        # optimizer_f_calls_limit=1000,
        # optimizer_probability=0.02,
        # complexity_of_constants=3,
        early_stop_condition=(l, c) -> (l < 1e-6 && c <= 5) || (l < 1e-10 && c <= 10),
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

    hall_of_fame = equation_search(X, y;
     options=options,
      parallelism=:multithreading,
       niterations=30000000)

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
