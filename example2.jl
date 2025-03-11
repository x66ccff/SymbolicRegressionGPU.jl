# export JULIA_NUM_THREADS=4
# export JULIA_DEBUG=loading
# julia --project=. example.jl
# julia example.jl

using SymbolicRegression
using LoopVectorization
using DelimitedFiles  # 用于读取TSV文件
using Random
function main()
    # 读取TSV文件
    data = readdlm("test_20.tsv", '\t', Float64, skipstart=1)  # skipstart=1 跳过标题行
    
    # 分离特征和目标
    X = Float32.(transpose(data[:, 1:7]))  # 取前7列作为特征,转置使其符合要求的维度
    y = Float32.(data[:, 8])  # 最后一列是target

    
    # n = 1000  # 目标样本数
    # indices = shuffle(1:size(X,2))[1:n]  # 随机抽取n个索引
    # X = X[:, indices]  # 获取对应的特征
    # y = y[indices]     # 获取对应的标签

    options = SymbolicRegression.Options(;
        timeout_in_seconds=60000000,
        binary_operators=[+, *, /, -],
        unary_operators=[sin, cos, exp, log, sqrt],
        # population_size=100,
        # populations=15,
        batching=true,
        batch_size=100,
        adaptive_parsimony_scaling=1_000.0,
        parsimony=0.0,
        maxsize=30,
        maxdepth=20,
        turbo=true,
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
