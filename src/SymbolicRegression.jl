module SymbolicRegression

# Types
export Population,
    PopMember,
    HallOfFame,
    Options,
    Dataset,
    MutationWeights,
    Node,
    GraphNode,
    ParametricNode,
    Expression,
    ParametricExpression,
    TemplateExpression,
    TemplateStructure,
    ValidVector,
    ComposableExpression,
    NodeSampler,
    AbstractExpression,
    AbstractExpressionNode,
    EvalOptions,
    SRRegressor,
    MultitargetSRRegressor,
    SRLogger,

    #Functions:
    equation_search,
    s_r_cycle,
    calculate_pareto_frontier,
    count_nodes,
    compute_complexity,
    @parse_expression,
    parse_expression,
    @declare_expression_operator,
    print_tree,
    string_tree,
    eval_tree_array,
    eval_diff_tree_array,
    eval_grad_tree_array,
    differentiable_eval_tree_array,
    set_node!,
    copy_node,
    node_to_symbolic,
    symbolic_to_node,
    simplify_tree!,
    tree_mapreduce,
    combine_operators,
    gen_random_tree,
    gen_random_tree_fixed_size,
    @extend_operators,
    get_tree,
    get_contents,
    get_metadata,
    with_contents,
    with_metadata,

    #Operators
    plus,
    sub,
    mult,
    square,
    cube,
    pow,
    safe_pow,
    safe_log,
    safe_log2,
    safe_log10,
    safe_log1p,
    safe_asin,
    safe_acos,
    safe_acosh,
    safe_atanh,
    safe_sqrt,
    neg,
    greater,
    cond,
    relu,
    logical_or,
    logical_and,

    # special operators
    gamma,
    erf,
    erfc,
    atanh_clip,
    PSRNManager,
    start_psrn_task,
    process_psrn_results!

using Distributed
using Printf: @printf, @sprintf
using Pkg: Pkg
using TOML: parsefile
using Random: seed!, shuffle!, randperm
using Reexport
using ProgressMeter: finish!
using DynamicExpressions:
    Node,
    GraphNode,
    ParametricNode,
    Expression,
    ParametricExpression,
    NodeSampler,
    AbstractExpression,
    AbstractExpressionNode,
    ExpressionInterface,
    @parse_expression,
    parse_expression,
    @declare_expression_operator,
    copy_node,
    set_node!,
    string_tree,
    print_tree,
    count_nodes,
    get_constants,
    get_scalar_constants,
    set_constants!,
    set_scalar_constants!,
    index_constants,
    NodeIndex,
    eval_tree_array,
    EvalOptions,
    differentiable_eval_tree_array,
    eval_diff_tree_array,
    eval_grad_tree_array,
    node_to_symbolic,
    symbolic_to_node,
    combine_operators,
    simplify_tree!,
    tree_mapreduce,
    set_default_variable_names!,
    node_type,
    get_tree,
    get_contents,
    get_metadata,
    with_contents,
    with_metadata
using DynamicExpressions: with_type_parameters
@reexport using LossFunctions:
    MarginLoss,
    DistanceLoss,
    SupervisedLoss,
    ZeroOneLoss,
    LogitMarginLoss,
    PerceptronLoss,
    HingeLoss,
    L1HingeLoss,
    L2HingeLoss,
    SmoothedL1HingeLoss,
    ModifiedHuberLoss,
    L2MarginLoss,
    ExpLoss,
    SigmoidLoss,
    DWDMarginLoss,
    LPDistLoss,
    L1DistLoss,
    L2DistLoss,
    PeriodicLoss,
    HuberLoss,
    EpsilonInsLoss,
    L1EpsilonInsLoss,
    L2EpsilonInsLoss,
    LogitDistLoss,
    QuantileLoss,
    LogCoshLoss
using DynamicDiff: D
using Compat: @compat, Fix
using CUDA
using LinearAlgebra # Needed for PSRN hcat
if !CUDA.functional()
    @warn """
    CUDA.jl is not functional.
    Ensure you have a NVIDIA GPU and compatible drivers installed.
    PSRN GPU acceleration will be disabled.
    Set `options.populations <= 3` (or adjust threshold) to avoid PSRN attempts.
    """
    # Define placeholder types/functions if CUDA is not functional
    # This prevents errors later if PSRN parts are called conditionally.
    struct PSRNManager end
    start_psrn_task(args...) = nothing
    process_psrn_results!(args...) = nothing
    struct PSRN end # Placeholder
    const T_GPU = Float32 # Define a default CPU type
end
#! format: off
@compat(
    public,
    (
        AbstractOptions, AbstractRuntimeOptions, RuntimeOptions,
        AbstractMutationWeights, mutate!, condition_mutation_weights!,
        sample_mutation, MutationResult, AbstractSearchState, SearchState,
        LOSS_TYPE, DATA_TYPE, node_type,
    )
)
#! format: on
# ^ We can add new functions here based on requests from users.
# However, I don't want to add many functions without knowing what
# users will actually want to overload.

# https://discourse.julialang.org/t/how-to-find-out-the-version-of-a-package-from-its-module/37755/15
const PACKAGE_VERSION = try
    root = pkgdir(@__MODULE__)
    if root == String
        let project = parsefile(joinpath(root, "Project.toml"))
            VersionNumber(project["version"])
        end
    else
        VersionNumber(0, 0, 0)
    end
catch
    VersionNumber(0, 0, 0)
end

using DispatchDoctor: @stable

@stable default_mode = "disable" begin
    include("Utils.jl")
    include("InterfaceDynamicQuantities.jl")
    include("Core.jl")
    include("InterfaceDynamicExpressions.jl")
    include("Recorder.jl")
    include("Complexity.jl")
    include("DimensionalAnalysis.jl")
    include("CheckConstraints.jl")
    include("AdaptiveParsimony.jl")
    include("MutationFunctions.jl")
    include("LossFunctions.jl")
    include("PopMember.jl")
    include("ConstantOptimization.jl")
    include("Population.jl")
    include("HallOfFame.jl")
    include("Mutate.jl")
    include("RegularizedEvolution.jl")
    include("SingleIteration.jl")
    include("ProgressBars.jl")
    include("Migration.jl")
    include("SearchUtils.jl")
    include("Logging.jl")
    include("ExpressionBuilder.jl")
    include("ComposableExpression.jl")
    include("TemplateExpression.jl")
    include("ParametricExpression.jl")
    include("PSRNfunctions.jl")
    include("PSRNmodel.jl")
end

using .CoreModule:
    DATA_TYPE,
    LOSS_TYPE,
    RecordType,
    Dataset,
    AbstractOptions,
    Options,
    ComplexityMapping,
    AbstractMutationWeights,
    MutationWeights,
    get_safe_op,
    max_features,
    is_weighted,
    sample_mutation,
    plus,
    sub,
    mult,
    square,
    cube,
    pow,
    safe_pow,
    safe_log,
    safe_log2,
    safe_log10,
    safe_log1p,
    safe_sqrt,
    safe_asin,
    safe_acos,
    safe_acosh,
    safe_atanh,
    neg,
    greater,
    cond,
    relu,
    logical_or,
    logical_and,
    gamma,
    erf,
    erfc,
    atanh_clip,
    create_expression,
    has_units
using .UtilsModule: is_anonymous_function, recursive_merge, json3_write, @ignore
using .ComplexityModule: compute_complexity
using .CheckConstraintsModule: check_constraints
using .AdaptiveParsimonyModule:
    RunningSearchStatistics, update_frequencies!, move_window!, normalize_frequencies!
using .MutationFunctionsModule:
    gen_random_tree,
    gen_random_tree_fixed_size,
    random_node,
    random_node_and_parent,
    crossover_trees
using .InterfaceDynamicExpressionsModule: @extend_operators
using .LossFunctionsModule: eval_loss, score_func, update_baseline_loss!
using .PopMemberModule: PopMember, reset_birth!
using .PopulationModule: Population, best_sub_pop, record_population, best_of_sample
using .HallOfFameModule:
    HallOfFame, calculate_pareto_frontier, string_dominating_pareto_curve
using .MutateModule: mutate!, condition_mutation_weights!, MutationResult
using .SingleIterationModule: s_r_cycle, optimize_and_simplify_population
using .ProgressBarsModule: WrappedProgressBar
using .RecorderModule: @recorder, find_iteration_from_record
using .MigrationModule: migrate!
using .SearchUtilsModule:
    AbstractSearchState,
    SearchState,
    AbstractRuntimeOptions,
    RuntimeOptions,
    WorkerAssignments,
    DefaultWorkerOutputType,
    assign_next_worker!,
    get_worker_output_type,
    extract_from_worker,
    @sr_spawner,
    StdinReader,
    watch_stream,
    close_reader!,
    check_for_user_quit,
    check_for_loss_threshold,
    check_for_timeout,
    check_max_evals,
    ResourceMonitor,
    record_channel_state!,
    estimate_work_fraction,
    update_progress_bar!,
    print_search_state,
    init_dummy_pops,
    load_saved_hall_of_fame,
    load_saved_population,
    construct_datasets,
    save_to_file,
    get_cur_maxsize,
    update_hall_of_fame!,
    logging_callback!
using .LoggingModule: AbstractSRLogger, SRLogger, get_logger
using .TemplateExpressionModule: TemplateExpression, TemplateStructure
using .TemplateExpressionModule: TemplateExpression, TemplateStructure, ValidVector
using .ComposableExpressionModule: ComposableExpression
using .ExpressionBuilderModule: embed_metadata, strip_metadata

import .PSRNmodel
# Import PSRN components only if CUDA is functional
if CUDA.functional()
    using .PSRNmodel: PSRN, forward, get_expr, get_best_expr_and_MSE_topk, T_GPU, OPERATORS as PSRN_OPERATORS # Alias PSRN's OPERATORS
end
import .PSRNmodel # To qualify PSRNManager constructor if needed

@stable default_mode = "disable" begin
    include("deprecates.jl")
    include("Configure.jl")
end

"""
    equation_search(X, y[; kws...])

Perform a distributed equation search for functions `f_i` which
describe the mapping `f_i(X[:, j]) ≈ y[i, j]`. Options are
configured using SymbolicRegression.Options(...),
which should be passed as a keyword argument to options.
One can turn off parallelism with `numprocs=0`,
which is useful for debugging and profiling.

# Arguments
- `X::AbstractMatrix{T}`:  The input dataset to predict `y` from.
    The first dimension is features, the second dimension is rows.
- `y::Union{AbstractMatrix{T}, AbstractVector{T}}`: The values to predict. The first dimension
    is the output feature to predict with each equation, and the
    second dimension is rows.
- `niterations::Int=100`: The number of iterations to perform the search.
    More iterations will improve the results.
- `weights::Union{AbstractMatrix{T}, AbstractVector{T}, Nothing}=nothing`: Optionally
    weight the loss for each `y` by this value (same shape as `y`).
- `options::AbstractOptions=Options()`: The options for the search, such as
    which operators to use, evolution hyperparameters, etc.
- `variable_names::Union{Vector{String}, Nothing}=nothing`: The names
    of each feature in `X`, which will be used during printing of equations.
- `display_variable_names::Union{Vector{String}, Nothing}=variable_names`: Names
    to use when printing expressions during the search, but not when saving
    to an equation file.
- `y_variable_names::Union{String,AbstractVector{String},Nothing}=nothing`: The
    names of each output feature in `y`, which will be used during printing
    of equations.
- `parallelism=:multithreading`: What parallelism mode to use.
    The options are `:multithreading`, `:multiprocessing`, and `:serial`.
    By default, multithreading will be used. Multithreading uses less memory,
    but multiprocessing can handle multi-node compute. If using `:multithreading`
    mode, the number of threads available to julia are used. If using
    `:multiprocessing`, `numprocs` processes will be created dynamically if
    `procs` is unset. If you have already allocated processes, pass them
    to the `procs` argument and they will be used.
    You may also pass a string instead of a symbol, like `"multithreading"`.
- `numprocs::Union{Int, Nothing}=nothing`:  The number of processes to use,
    if you want `equation_search` to set this up automatically. By default
    this will be `4`, but can be any number (you should pick a number <=
    the number of cores available).
- `procs::Union{Vector{Int}, Nothing}=nothing`: If you have set up
    a distributed run manually with `procs = addprocs()` and `@everywhere`,
    pass the `procs` to this keyword argument.
- `addprocs_function::Union{Function, Nothing}=nothing`: If using multiprocessing
    (`parallelism=:multiprocessing`), and are not passing `procs` manually,
    then they will be allocated dynamically using `addprocs`. However,
    you may also pass a custom function to use instead of `addprocs`.
    This function should take a single positional argument,
    which is the number of processes to use, as well as the `lazy` keyword argument.
    For example, if set up on a slurm cluster, you could pass
    `addprocs_function = addprocs_slurm`, which will set up slurm processes.
- `heap_size_hint_in_bytes::Union{Int,Nothing}=nothing`: On Julia 1.9+, you may set the `--heap-size-hint`
    flag on Julia processes, recommending garbage collection once a process
    is close to the recommended size. This is important for long-running distributed
    jobs where each process has an independent memory, and can help avoid
    out-of-memory errors. By default, this is set to `Sys.free_memory() / numprocs`.
- `worker_imports::Union{Vector{Symbol},Nothing}=nothing`: If you want to import
    additional modules on each worker, pass them here as a vector of symbols.
    By default some of the extensions will automatically be loaded when needed.
- `runtests::Bool=true`: Whether to run (quick) tests before starting the
    search, to see if there will be any problems during the equation search
    related to the host environment.
- `saved_state=nothing`: If you have already
    run `equation_search` and want to resume it, pass the state here.
    To get this to work, you need to have set return_state=true,
    which will cause `equation_search` to return the state. The second
    element of the state is the regular return value with the hall of fame.
    Note that you cannot change the operators or dataset, but most other options
    should be changeable.
- `return_state::Union{Bool, Nothing}=nothing`: Whether to return the
    state of the search for warm starts. By default this is false.
- `loss_type::Type=Nothing`: If you would like to use a different type
    for the loss than for the data you passed, specify the type here.
    Note that if you pass complex data `::Complex{L}`, then the loss
    type will automatically be set to `L`.
- `verbosity`: Whether to print debugging statements or not.
- `logger::Union{AbstractSRLogger,Nothing}=nothing`: An optional logger to record
    the progress of the search. You can use an `SRLogger` to wrap a custom logger,
    or pass `nothing` to disable logging.
- `progress`: Whether to use a progress bar output. Only available for
    single target output.
- `X_units::Union{AbstractVector,Nothing}=nothing`: The units of the dataset,
    to be used for dimensional constraints. For example, if `X_units=["kg", "m"]`,
    then the first feature will have units of kilograms, and the second will
    have units of meters.
- `y_units=nothing`: The units of the output, to be used for dimensional constraints.
    If `y` is a matrix, then this can be a vector of units, in which case
    each element corresponds to each output feature.

# Returns
- `hallOfFame::HallOfFame`: The best equations seen during the search.
    hallOfFame.members gives an array of `PopMember` objects, which
    have their tree (equation) stored in `.tree`. Their score (loss)
    is given in `.score`. The array of `PopMember` objects
    is enumerated by size from `1` to `options.maxsize`.
"""
function equation_search(
    X::AbstractMatrix{T},
    y::AbstractMatrix;
    niterations::Int=100,
    weights::Union{AbstractMatrix{T},AbstractVector{T},Nothing}=nothing,
    options::AbstractOptions=Options(),
    variable_names::Union{AbstractVector{String},Nothing}=nothing,
    display_variable_names::Union{AbstractVector{String},Nothing}=variable_names,
    y_variable_names::Union{String,AbstractVector{String},Nothing}=nothing,
    parallelism=:multithreading,
    numprocs::Union{Int,Nothing}=nothing,
    procs::Union{Vector{Int},Nothing}=nothing,
    addprocs_function::Union{Function,Nothing}=nothing,
    heap_size_hint_in_bytes::Union{Integer,Nothing}=nothing,
    worker_imports::Union{Vector{Symbol},Nothing}=nothing,
    runtests::Bool=true,
    saved_state=nothing,
    return_state::Union{Bool,Nothing,Val}=nothing,
    run_id::Union{String,Nothing}=nothing,
    loss_type::Type{L}=Nothing,
    verbosity::Union{Integer,Nothing}=nothing,
    logger::Union{AbstractSRLogger,Nothing}=nothing,
    progress::Union{Bool,Nothing}=nothing,
    X_units::Union{AbstractVector,Nothing}=nothing,
    y_units=nothing,
    enable_psrn::Bool = true, # Default PSRN on
    psrn_options::NamedTuple = NamedTuple(), # PSRN config
    extra::NamedTuple=NamedTuple(),
    v_dim_out::Val{DIM_OUT}=Val(nothing),
    # Deprecated:
    multithreaded=nothing,
) where {T<:DATA_TYPE,L,DIM_OUT}
    if multithreaded !== nothing
        error(
            "`multithreaded` is deprecated. Use the `parallelism` argument instead. " *
            "Choose one of :multithreaded, :multiprocessing, or :serial.",
        )
    end


    # --- PSRN Activation Check ---
    use_psrn = true
    psrn_min_pop_threshold = 4 # Example threshold
    if enable_psrn
        if !CUDA.functional()
            @warn "PSRN requested (`enable_psrn=true`) but CUDA is not functional. PSRN will be disabled."
        elseif options.populations < psrn_min_pop_threshold
            @warn "PSRN requested (`enable_psrn=true`) but number of populations ($(options.populations)) is less than threshold ($psrn_min_pop_threshold). PSRN will be disabled for efficiency."
        else
            @info "PSRN GPU acceleration is enabled."
            use_psrn = true
        end
    end
    # Pass use_psrn and psrn_options down
    # Combine with existing extra NamedTuple
    extended_extra = merge(extra, (use_psrn=use_psrn, psrn_options=psrn_options))
    # --- End PSRN Check ---


    if weights !== nothing
        @assert length(weights) == length(y)
        weights = reshape(weights, size(y))
    end

    datasets = construct_datasets(
        X,
        y,
        weights,
        variable_names,
        display_variable_names,
        y_variable_names,
        X_units,
        y_units,
        extended_extra,
        L,
    )

    return equation_search(
        datasets;
        niterations=niterations,
        options=options,
        parallelism=parallelism,
        numprocs=numprocs,
        procs=procs,
        addprocs_function=addprocs_function,
        heap_size_hint_in_bytes=heap_size_hint_in_bytes,
        worker_imports=worker_imports,
        runtests=runtests,
        saved_state=saved_state,
        return_state=return_state,
        run_id=run_id,
        verbosity=verbosity,
        logger=logger,
        progress=progress,
        v_dim_out=Val(DIM_OUT),
    )
end

function equation_search(
    X::AbstractMatrix{T}, y::AbstractVector; kw...
) where {T<:DATA_TYPE}
    return equation_search(X, reshape(y, (1, size(y, 1))); kw..., v_dim_out=Val(1))
end

function equation_search(dataset::Dataset; kws...)
    return equation_search([dataset]; kws..., v_dim_out=Val(1))
end

function equation_search(
    datasets::Vector{D};
    options::AbstractOptions=Options(),
    saved_state=nothing,
    runtime_options::Union{AbstractRuntimeOptions,Nothing}=nothing,
    runtime_options_kws...,
) where {T<:DATA_TYPE,L<:LOSS_TYPE,D<:Dataset{T,L}}
    _runtime_options = @something(
        runtime_options,
        RuntimeOptions(;
            options_return_state=options.return_state,
            options_verbosity=options.verbosity,
            options_progress=options.progress,
            nout=length(datasets),
            runtime_options_kws...,
        )
    )

    # Underscores here mean that we have mutated the variable
    return _equation_search(datasets, _runtime_options, options, saved_state)
end

# Core equation search implementation (internal)
@noinline function _equation_search(
    datasets::Vector{D},
    ropt::AbstractRuntimeOptions, # Now contains niterations
    options::AbstractOptions,
    saved_state
) where {D<:Dataset}
   # Extract PSRN flags from the first dataset (assume consistent across datasets)
   # These were added in the main equation_search call
   d1_extra = isempty(datasets) ? NamedTuple() : datasets[1].extra
   use_psrn = get(d1_extra, :use_psrn, false)
   psrn_opts_from_extra = get(d1_extra, :psrn_options, NamedTuple())
    _validate_options(datasets, ropt, options)
    state = _create_workers(datasets, ropt, options)

    # Initialize PSRN manager if enabled
    local psrn_manager::Union{PSRNManager, Nothing} = nothing # Type annotation
    if use_psrn
        psrn_manager = _initialize_psrn_manager(options, d1_extra, psrn_opts_from_extra)
    end

    _initialize_search!(state, datasets, ropt, options, saved_state)
    _warmup_search!(state, datasets, ropt, options)
    @info "Entering main loop: use_psrn = $use_psrn, psrn_manager type = $(typeof(psrn_manager))"
    _main_search_loop!(state, datasets, ropt, options, use_psrn, psrn_manager)
    _tear_down!(state, ropt, options)
    _info_dump(state, datasets, ropt, options)
    return _format_output(state, datasets, ropt, options)
end



# --- Helper Functions for Search ---

# (Keep _validate_options as is, or add PSRN validation inside)

# Function to initialize PSRN Manager (called conditionally)
function _initialize_psrn_manager(options::Options, extra::NamedTuple, psrn_options::NamedTuple)
    # This function encapsulates the logic for setting up PSRNManager
    # based on merged options from AbstractOptions and psrn_options NamedTuple.

    if !CUDA.functional()
        error("CUDA not functional, cannot initialize PSRN Manager.")
    end

    @info "Initializing PSRN Manager..."
    # --- Extract PSRN Configuration ---
    # Defaults can be set here or within PSRNManager constructor
    psrn_n_layers = get(psrn_options, :n_layers, 2)
    psrn_topk = get(psrn_options, :topk, 50)
    psrn_max_samples = get(psrn_options, :max_samples, 1000)
    psrn_device_id = get(psrn_options, :device_id, 0)
    n_psrn_input = get(psrn_options, :n_input_features, 15) # Default N_PSRN_INPUT

    # Determine Operators for PSRN
    default_psrn_ops = ["Add", "Mul", "Sub", "Div", "Identity", "Sqrt"]
    # Filter SR operators based on names supported by PSRN
    supported_psrn_names = Set(keys(PSRN_OPERATORS)) # From PSRNmodel
    derived_ops = String[]
    # Assumes options.operators.binops/unaops exists and elements can be stringified
    try
        for op in options.operators.binops
            op_str = string(op) # Fragile conversion
            if op_str in supported_psrn_names push!(derived_ops, op_str) end
        end
        for op in options.operators.unaops
            op_str = string(op)
            if op_str in supported_psrn_names push!(derived_ops, op_str) end
        end
    catch e
        @warn "Could not derive PSRN operators from SR options. Using defaults." error=e
        derived_ops = default_psrn_ops
    end

    psrn_op_names = get(psrn_options, :operators, isempty(derived_ops) ? default_psrn_ops : derived_ops)

    @info """
    PSRN Configuration:
      Input Features (Subtrees): $n_psrn_input
      Operators: $psrn_op_names
      Symbol Layers: $psrn_n_layers
      Top-K Return: $psrn_topk
      Max Samples per Call: $psrn_max_samples
      GPU Device ID: $psrn_device_id
    """

    # --- Create PSRNManager Instance ---
    manager = PSRNManager(;
        N_PSRN_INPUT=n_psrn_input,
        operators=psrn_op_names,
        n_symbol_layers=psrn_n_layers,
        options=options, # Pass SR options for context
        max_samples=psrn_max_samples,
        PSRN_topk=psrn_topk,
        device_id=psrn_device_id
    )
    # @info "PSRN Manager initialized successfully." # Already logged inside constructor
    return manager
end

function _validate_options(
    datasets::Vector{D}, ropt::AbstractRuntimeOptions, options::AbstractOptions
) where {T,L,D<:Dataset{T,L}}
    example_dataset = first(datasets)
    nout = length(datasets)
    @assert nout >= 1
    @assert (nout == 1 || ropt.dim_out == 2)
    @assert options.populations >= 1
    if ropt.progress
        @assert(nout == 1, "You cannot display a progress bar for multi-output searches.")
        @assert(ropt.verbosity > 0, "You cannot display a progress bar with `verbosity=0`.")
    end
    if options.node_type <: GraphNode && ropt.verbosity > 0
        @warn "The `GraphNode` interface and mutation operators are experimental and will change in future versions."
    end
    if ropt.runtests
        test_option_configuration(ropt.parallelism, datasets, options, ropt.verbosity)
        test_dataset_configuration(example_dataset, options, ropt.verbosity)
    end
    for dataset in datasets
        update_baseline_loss!(dataset, options)
    end
    if options.define_helper_functions
        set_default_variable_names!(first(datasets).variable_names)
    end
    if options.seed !== nothing
        seed!(options.seed)
    end
    return nothing
end
@stable default_mode = "disable" function _create_workers(
    datasets::Vector{D}, ropt::AbstractRuntimeOptions, options::AbstractOptions
) where {T,L,D<:Dataset{T,L}}
    stdin_reader = watch_stream(options.input_stream)

    record = RecordType()
    @recorder record["options"] = "$(options)"

    nout = length(datasets)
    example_dataset = first(datasets)
    example_ex = create_expression(zero(T), options, example_dataset)
    NT = typeof(example_ex)
    PopType = Population{T,L,NT}
    HallOfFameType = HallOfFame{T,L,NT}
    WorkerOutputType = get_worker_output_type(
        Val(ropt.parallelism), PopType, HallOfFameType
    )
    ChannelType = ropt.parallelism == :multiprocessing ? RemoteChannel : Channel

    # Pointers to populations on each worker:
    worker_output = Vector{WorkerOutputType}[WorkerOutputType[] for j in 1:nout]
    # Initialize storage for workers
    tasks = [Task[] for j in 1:nout]
    # Set up a channel to send finished populations back to head node
    channels = [[ChannelType(1) for i in 1:(options.populations)] for j in 1:nout]
    (procs, we_created_procs) = if ropt.parallelism == :multiprocessing
        configure_workers(;
            procs=ropt.init_procs,
            ropt.numprocs,
            ropt.addprocs_function,
            options,
            worker_imports=ropt.worker_imports,
            project_path=splitdir(Pkg.project().path)[1],
            file=@__FILE__,
            ropt.exeflags,
            ropt.verbosity,
            example_dataset,
            ropt.runtests,
        )
    else
        Int[], false
    end
    # Get the next worker process to give a job:
    worker_assignment = WorkerAssignments()
    # Randomly order which order to check populations:
    # This is done so that we do work on all nout equally.
    task_order = [(j, i) for j in 1:nout for i in 1:(options.populations)]
    shuffle!(task_order)

    # Persistent storage of last-saved population for final return:
    last_pops = init_dummy_pops(options.populations, datasets, options)
    # Best 10 members from each population for migration:
    best_sub_pops = init_dummy_pops(options.populations, datasets, options)
    # TODO: Should really be one per population too.
    all_running_search_statistics = [
        RunningSearchStatistics(; options=options) for j in 1:nout
    ]
    # Records the number of evaluations:
    # Real numbers indicate use of batching.
    num_evals = [[0.0 for i in 1:(options.populations)] for j in 1:nout]

    halls_of_fame = Vector{HallOfFameType}(undef, nout)

    total_cycles = ropt.niterations * options.populations
    cycles_remaining = [total_cycles for j in 1:nout]
    cur_maxsizes = [
        get_cur_maxsize(; options, total_cycles, cycles_remaining=cycles_remaining[j]) for
        j in 1:nout
    ]

    return SearchState{T,L,typeof(example_ex),WorkerOutputType,ChannelType}(;
        procs=procs,
        we_created_procs=we_created_procs,
        worker_output=worker_output,
        tasks=tasks,
        channels=channels,
        worker_assignment=worker_assignment,
        task_order=task_order,
        halls_of_fame=halls_of_fame,
        last_pops=last_pops,
        best_sub_pops=best_sub_pops,
        all_running_search_statistics=all_running_search_statistics,
        num_evals=num_evals,
        cycles_remaining=cycles_remaining,
        cur_maxsizes=cur_maxsizes,
        stdin_reader=stdin_reader,
        record=Ref(record),
    )
end
function _initialize_search!(
    state::AbstractSearchState{T,L,N},
    datasets,
    ropt::AbstractRuntimeOptions,
    options::AbstractOptions,
    saved_state,
) where {T,L,N}
    nout = length(datasets)

    init_hall_of_fame = load_saved_hall_of_fame(saved_state)
    if init_hall_of_fame === nothing
        for j in 1:nout
            state.halls_of_fame[j] = HallOfFame(options, datasets[j])
        end
    else
        # Recompute losses for the hall of fame, in
        # case the dataset changed:
        for j in eachindex(init_hall_of_fame, datasets, state.halls_of_fame)
            hof = strip_metadata(init_hall_of_fame[j], options, datasets[j])
            for member in hof.members[hof.exists]
                score, result_loss = score_func(datasets[j], member, options)
                member.score = score
                member.loss = result_loss
            end
            state.halls_of_fame[j] = hof
        end
    end

    for j in 1:nout, i in 1:(options.populations)
        worker_idx = assign_next_worker!(
            state.worker_assignment; out=j, pop=i, parallelism=ropt.parallelism, state.procs
        )
        saved_pop = load_saved_population(saved_state; out=j, pop=i)
        new_pop =
            if saved_pop !== nothing && length(saved_pop.members) == options.population_size
                _saved_pop = strip_metadata(saved_pop, options, datasets[j])
                ## Update losses:
                for member in _saved_pop.members
                    score, result_loss = score_func(datasets[j], member, options)
                    member.score = score
                    member.loss = result_loss
                end
                copy_pop = copy(_saved_pop)
                @sr_spawner(
                    begin
                        (copy_pop, HallOfFame(options, datasets[j]), RecordType(), 0.0)
                    end,
                    parallelism = ropt.parallelism,
                    worker_idx = worker_idx
                )
            else
                if saved_pop !== nothing && ropt.verbosity > 0
                    @warn "Recreating population (output=$(j), population=$(i)), as the saved one doesn't have the correct number of members."
                end
                @sr_spawner(
                    begin
                        (
                            Population(
                                datasets[j];
                                population_size=options.population_size,
                                nlength=3,
                                options=options,
                                nfeatures=max_features(datasets[j], options),
                            ),
                            HallOfFame(options, datasets[j]),
                            RecordType(),
                            Float64(options.population_size),
                        )
                    end,
                    parallelism = ropt.parallelism,
                    worker_idx = worker_idx
                )
                # This involves population_size evaluations, on the full dataset:
            end
        push!(state.worker_output[j], new_pop)
    end
    return nothing
end
function _warmup_search!(
    state::AbstractSearchState{T,L,N},
    datasets,
    ropt::AbstractRuntimeOptions,
    options::AbstractOptions,
) where {T,L,N}
    nout = length(datasets)
    for j in 1:nout, i in 1:(options.populations)
        dataset = datasets[j]
        running_search_statistics = state.all_running_search_statistics[j]
        cur_maxsize = state.cur_maxsizes[j]
        @recorder state.record[]["out$(j)_pop$(i)"] = RecordType()
        worker_idx = assign_next_worker!(
            state.worker_assignment; out=j, pop=i, parallelism=ropt.parallelism, state.procs
        )

        # TODO - why is this needed??
        # Multi-threaded doesn't like to fetch within a new task:
        c_rss = deepcopy(running_search_statistics)
        last_pop = state.worker_output[j][i]
        updated_pop = @sr_spawner(
            begin
                in_pop = first(
                    extract_from_worker(last_pop, Population{T,L,N}, HallOfFame{T,L,N})
                )
                _dispatch_s_r_cycle(
                    in_pop,
                    dataset,
                    options;
                    pop=i,
                    out=j,
                    iteration=0,
                    ropt.verbosity,
                    cur_maxsize,
                    running_search_statistics=c_rss,
                )::DefaultWorkerOutputType{Population{T,L,N},HallOfFame{T,L,N}}
            end,
            parallelism = ropt.parallelism,
            worker_idx = worker_idx
        )
        state.worker_output[j][i] = updated_pop
    end
    return nothing
end
# Define PSRNManager struct conditionally
if CUDA.functional()
    mutable struct PSRNManager
        channel::Channel{Vector{Expression}}
        current_task::Union{Task,Nothing}
        call_count::Int
        N_PSRN_INPUT::Int # Number of features PSRN expects (evaluated subtrees)
        net::PSRN # PSRN model instance from PSRNmodel.jl
        max_samples::Int # Max samples to use for PSRN training/evaluation per call
        PSRN_topk::Int # How many top expressions to request from PSRN
        device_id::Int # Which GPU device to use

        function PSRNManager(;
            N_PSRN_INPUT::Int,
            operators::Vector{String}, # Operator names PSRN should use
            n_symbol_layers::Int,
            options::Options, # Pass SR options for context
            max_samples::Int=1000,
            PSRN_topk::Int=50,
            device_id::Int = 0 # Default to GPU 0
        )
            # Check CUDA availability before initializing PSRN
            if !CUDA.functional()
                error("Cannot initialize PSRNManager: CUDA is not functional.")
            end

            # Set the device for initialization
            try
                CUDA.device!(device_id)
                current_dev = CUDA.device()
                @info "Initializing PSRNManager on CUDA device $(current_dev)..."
            catch e
                @error "Failed to set CUDA device to $device_id during PSRNManager initialization." exception=(e, catch_backtrace())
                rethrow(e)
            end

            # Initialize the PSRN network (defined in PSRNmodel.jl)
            psrn_net = PSRN(;
                n_variables=N_PSRN_INPUT, # Should match the number of evaluated subtrees
                operators=operators,      # Operators PSRN internal layers will use
                n_symbol_layers=n_symbol_layers,
                dr_mask=nothing,          # Dimensionality reduction mask (optional)
                options=options,          # Pass SR options to PSRN if needed
                PSRN_topk=PSRN_topk       # Pass top-k parameter
            )
            @info "PSRN network initialized."

            return new(
                Channel{Vector{Expression}}(1000), # Channel buffer size
                nothing, # No task running initially
                0,       # Call count starts at 0
                N_PSRN_INPUT,
                psrn_net,
                max_samples,
                PSRN_topk,
                device_id
            )
        end
    end
else
    # If CUDA not functional, PSRNManager is just a placeholder defined earlier
    # We still need a definition to avoid UndefVarError if used conditionally later
    # The placeholder struct PSRNManager was defined near the top in Step 1
    # Alternatively, define it here if not done earlier:
    # if !isdefined(@__MODULE__, :PSRNManager)
    #     struct PSRNManager end
    # end
end

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
            # push!(result, Node(Float32; val=rand(-5:5)))
        tree = gen_random_tree(
            rand(1:4),                     # length
            options,              # options
            n_variables,          # nfeatures
            Float32;
            only_gen_bin_op=true,
            only_gen_int_const=false,
            feature_prob=0.7
        )
        push!(result, tree)
        # else
        #     # 随机选择一个未使用的feature
        #     feature = rand(available_features)
        #     tree = Node(Float32; feature=feature)
            
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
function analyze_common_subtrees(trees::Vector{<:Expression}, options::Options)
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
            subtrees = get_subtrees(expr)

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

using Symbolics: expand, flatten_fractions, quick_cancel

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




function start_psrn_task(
    manager::PSRNManager,
    dominating_trees::Vector{<:Expression}, # Trees from Hall of Fame
    dataset::Dataset{T, L}, # Original dataset
    options::AbstractOptions,
    # N_PSRN_INPUT and n_variables passed explicitly for clarity
    N_PSRN_INPUT::Int, # How many features PSRN expects (from manager is fine too)
    n_variables::Int   # Number of original features in dataset.X
) where {T, L}

    # --- Pre-checks ---
    if !CUDA.functional() || !(manager isa PSRNManager)
         # @warn "CUDA not functional or PSRNManager not initialized, skipping PSRN task."
         return nothing # Silently skip if not functional
    end
    if manager.current_task !== nothing && !istaskdone(manager.current_task)
        # Task already running
        return nothing
    end
     if isempty(dominating_trees)
         @info "No dominating trees found, skipping PSRN task."
         return nothing
     end

    # --- Prepare Task ---
    device_id = manager.device_id # Use device specified in manager

    # Spawn a new thread/task for the PSRN computation
    return manager.current_task = Threads.@spawn begin
        @info "[PSRN Task START] Task spawned for manager call #$(manager.call_count)"
        task_start_time = time()
        final_status = "unknown"
        best_expressions = Expression[] # Default empty result

        try
            # --- Setup Environment in Task ---
            CUDA.device!(device_id) # Ensure this task uses the correct GPU
            manager.call_count += 1
            current_device = CUDA.device()
            task_id = Threads.threadid() # Get thread ID for logging
            @info "✅ [Dev:$current_device Task:$task_id] Starting PSRN computation #$(manager.call_count)"
            # Optional: Log initial GPU memory
            # @info "[Dev:$current_device] Initial GPU Memory: $(CUDA.memory_status())"

            # --- Feature Engineering (CPU) ---
            # 1. Analyze common subtrees from HoF
            common_subtrees = analyze_common_subtrees(dominating_trees, options)

            # 2. Select top N subtrees (potential features for PSRN)
            # This function MUST return exactly N_PSRN_INPUT nodes
            top_subtrees = select_top_subtrees(
                common_subtrees,
                N_PSRN_INPUT, # Target number of features for PSRN
                options,
                n_variables # Number of original variables
            )

            if length(top_subtrees) == 0
                 @warn "[Dev:$current_device Task:$task_id] No suitable subtrees selected for PSRN input. Skipping."
                 put!(manager.channel, Expression[]) # Put empty result
                 return # Exit task
            end
            if length(top_subtrees) != N_PSRN_INPUT
                 @warn "[Dev:$current_device Task:$task_id] Selected $(length(top_subtrees)) subtrees, but PSRN expected $N_PSRN_INPUT. PSRN input layer size might mismatch if not flexible."
                 # PSRN needs to handle this size mismatch or selection must guarantee N
                 # Adjusting N_PSRN_INPUT here might be complex if PSRN is pre-built
            end
            actual_n_psrn_inputs = length(top_subtrees) # The actual number selected

            # 3. Evaluate selected subtrees on the dataset (CPU)
            # @info "[Dev:$current_device Task:$task_id] Evaluating $(actual_n_psrn_inputs) subtrees..."
            X_mapped_cpu = evaluate_subtrees(top_subtrees, dataset, options)
            # @info "[Dev:$current_device Task:$task_id] Subtree evaluation complete. Size: $(size(X_mapped_cpu))"

            # 4. Check evaluated features for issues (CPU) BEFORE sampling/transfer
            nan_inf_x = .!isfinite.(X_mapped_cpu)
            if any(nan_inf_x)
                num_nan_inf = sum(nan_inf_x)
                @warn "[Dev:$current_device Task:$task_id] Non-finite values ($num_nan_inf) detected in evaluated subtrees (X_mapped_cpu). Replacing with 0."
                X_mapped_cpu[nan_inf_x] .= zero(T) # Use zero of dataset type
            end
            y_cpu = vec(dataset.y) # Ensure y is a vector
            nan_inf_y = .!isfinite.(y_cpu)
             if any(nan_inf_y)
                 num_nan_inf = sum(nan_inf_y)
                 @error "[Dev:$current_device Task:$task_id] Non-finite values ($num_nan_inf) detected in target variable (dataset.y). Cannot proceed with PSRN."
                 put!(manager.channel, Expression[]) # Signal failure/empty result
                 return # Exit the task
            end

            # --- Data Sampling (CPU) ---
            n_samples = size(X_mapped_cpu, 1)
            local X_sampled_cpu::Matrix{T}, y_sampled_cpu::Vector{T}

            if n_samples > manager.max_samples
                # @info "[Dev:$current_device Task:$task_id] Sampling $(manager.max_samples) out of $n_samples for PSRN."
                sample_indices = randperm(n_samples)[1:(manager.max_samples)]
                X_sampled_cpu = X_mapped_cpu[sample_indices, :]
                y_sampled_cpu = y_cpu[sample_indices] # y is already a vector
            else
                # Use all samples
                X_sampled_cpu = X_mapped_cpu
                y_sampled_cpu = y_cpu
            end

            # --- Data Transfer to GPU ---
            # @info "[Dev:$current_device Task:$task_id] Transferring data to GPU..."
            X_gpu = cu(T_GPU.(X_sampled_cpu)) # Convert to T_GPU and move
            y_gpu = cu(T_GPU.(y_sampled_cpu)) # Convert Vector to T_GPU and move
             # @info "[Dev:$current_device Task:$task_id] Data transfer complete. X size: $(size(X_gpu)), Y size: $(size(y_gpu))"

             # Verify dimensions match expected PSRN input size
             if size(X_gpu, 2) != actual_n_psrn_inputs
                  @error "[Dev:$current_device Task:$task_id] Dimension mismatch after transfer: X_gpu columns ($(size(X_gpu, 2))) != selected subtrees ($actual_n_psrn_inputs)."
                   # Clean up and exit task
                  X_gpu = nothing; y_gpu = nothing; GC.gc(true); CUDA.synchronize()
                  put!(manager.channel, Expression[])
                  return
             end

            # --- Prepare PSRN Input Expressions ---
            # PSRN's internal `_get_expr` uses `current_expr_ls` as its base cases.
            # This list should contain the Expression objects corresponding to the columns of X_gpu.
            variable_names_psrn = ["f$i" for i in 1:actual_n_psrn_inputs] # Names for PSRN input features
            manager.net.current_expr_ls = [
                Expression(node; operators=options.operators, variable_names=variable_names_psrn)
                for node in top_subtrees # Use the actual selected subtrees
            ]
            # @info "[Dev:$current_device Task:$task_id] Updated PSRN input expression list with $(length(manager.net.current_expr_ls)) expressions."

            # Verify PSRN input layer size matches actual inputs (important!)
             if manager.net.layers[1].in_dim != actual_n_psrn_inputs
                 @error "[Dev:$current_device Task:$task_id] PSRN n_variables ($(manager.net.layers[1].in_dim)) doesn't match actual input features ($actual_n_psrn_inputs). PSRN was likely initialized incorrectly or selection failed."
                  # Clean up and exit task
                 X_gpu = nothing; y_gpu = nothing; GC.gc(true); CUDA.synchronize()
                 put!(manager.channel, Expression[])
                 return
             end

            # --- Execute PSRN on GPU ---
            # @info "[Dev:$current_device Task:$task_id] Executing PSRN forward pass and top-k selection..."
            forward_start_time = time()
            @info "[PSRN Task RUN] Calling get_best_expr_and_MSE_topk on device $(CUDA.device()). X_gpu size=$(size(X_gpu)), Y_gpu size=$(size(y_gpu))"
            # CUDA.synchronize() # Force sync before call for timing/visibility
            best_expressions = get_best_expr_and_MSE_topk(
                manager.net, X_gpu, y_gpu # Pass GPU data
            )
            # CUDA.synchronize() # Force sync after call
            @info "[PSRN Task DONE] get_best_expr_and_MSE_topk finished. Found $(length(best_expressions)) expressions."
            forward_duration = time() - forward_start_time
            # @info "[Dev:$current_device Task:$task_id] PSRN execution complete in $(round(forward_duration, digits=3))s. Found $(length(best_expressions)) expressions."

            # --- Cleanup GPU Memory ---
            X_gpu = nothing; y_gpu = nothing # Allow GC to collect CuArrays
            # Trigger GC more aggressively if needed:
            # GC.gc(true); CUDA.synchronize()
            # @info "[Dev:$current_device Task:$task_id] GPU Memory after PSRN: $(CUDA.memory_status())"
            final_status = "success"

        catch e
            # --- Error Handling ---
            final_status = "error"
            bt = stacktrace(catch_backtrace())
            # Ensure manager is accessible for channel even in catch
            current_device_err = CUDA.isfunctional() ? CUDA.device() : -1 # Get current device for logging safely
            task_id_err = Threads.threadid()
            @error """
            [Dev:$current_device_err Task:$task_id_err] PSRN task execution error in computation #$(manager.call_count):
            Error type: $(typeof(e))
            Error message: $e
            Stacktrace:
            $(join(string.(bt), "\n"))
            """
             try
                 # Attempt to put an empty result to signal failure
                 put!(manager.channel, Expression[])
             catch chan_err
                 @error "[Dev:$current_device_err Task:$task_id_err] Failed to put error signal into PSRN channel: $chan_err"
             end
            # Cleanup attempt
            GC.gc(true); CUDA.isfunctional() && CUDA.synchronize()
            # Ensure best_expressions is empty on error
            best_expressions = Expression[]
        end

        # --- Finalize Task ---
        # Put the results (potentially empty on error) into the channel
        put!(manager.channel, best_expressions)
        task_duration = time() - task_start_time
        @info "✅ [Dev:$device_id Task:$(Threads.threadid())] PSRN task #$(manager.call_count) finished. Status: $final_status. Duration: $(round(task_duration, digits=3))s. Results: $(length(best_expressions)) expressions."

    end # end Threads.@spawn
end

function process_psrn_results!(
    manager::PSRNManager,
    hall_of_fame::HallOfFame{T, L, NT}, # Add type parameters for clarity
    dataset::Dataset{T, L},
    options::AbstractOptions,
) where {T, L, NT} # Match types
    # Check if manager is the placeholder or not
    if !CUDA.functional() || !(manager isa PSRNManager)
        return # Skip if CUDA not working or manager is placeholder
    end

    results_processed = 0
    newly_added_members = PopMember{T,L,NT}[] # Store members added in this call
    expressions_added_count = 0

    while isready(manager.channel)
        new_expressions = take!(manager.channel) # Vector{Expression}
        results_processed += 1
        if !isempty(new_expressions)
            # @info "Processing $(length(new_expressions)) expressions from PSRN..."
            current_add_batch = PopMember{T,L,NT}[] # Members from this specific channel read

            for psrn_expr in new_expressions
                # Create PopMember from the raw PSRN expression tree
                # Ensure type T matches the dataset/hall_of_fame
                try
                     # PopMember constructor handles type T for constants
                    mem = PopMember(dataset, psrn_expr.tree, options; T=T, deterministic=false)
                    push!(current_add_batch, mem)
                catch e
                    @warn "Failed to create PopMember from PSRN expression: $(string_tree(psrn_expr, options)). Error: $e"
                    # Skip this expression
                end
            end

            if !isempty(current_add_batch)
                 # Update HoF with the batch of new members
                num_added = update_hall_of_fame!(hall_of_fame, current_add_batch, options)
                expressions_added_count += num_added
            end
        else
             # @info "PSRN task returned an empty result list." # Reduce verbosity
        end
    end

    if expressions_added_count > 0
        @info "✅ Added $expressions_added_count new equations from PSRN results to Hall of Fame."
    elseif results_processed > 0 # Channel was read but list was empty or no improvement
         # @info "PSRN results processed, but no new equations improved the Hall of Fame." # Reduce verbosity
    end
    # No message if channel wasn't ready
end

function _main_search_loop!(
    state::AbstractSearchState{T,L,N},
    datasets::Vector{D}, # Add type
    ropt::AbstractRuntimeOptions,
    options::AbstractOptions,
    # Add PSRN related arguments:
    use_psrn::Bool,
    psrn_manager::Union{PSRNManager, Nothing} # Receive manager (or nothing)
) where {T,L,N, D<:Dataset} # Add D type parameter

    ropt.verbosity > 0 && @info "Started!"
    nout = length(datasets)
    start_time = time()
    progress_bar = if ropt.progress
        #TODO: need to iterate this on the max cycles remaining!
        sum_cycle_remaining = sum(state.cycles_remaining)
        WrappedProgressBar(
            sum_cycle_remaining, ropt.niterations; barlen=options.terminal_width
        )
    else
        nothing
    end

    last_print_time = time()
    last_speed_recording_time = time()
    num_evals_last = sum(sum, state.num_evals)
    num_evals_since_last = sum(sum, state.num_evals) - num_evals_last  # i.e., start at 0
    print_every_n_seconds = 5
    equation_speed = Float32[]

    println(options)


    if ropt.parallelism in (:multiprocessing, :multithreading)
        for j in 1:nout, i in 1:(options.populations)
            # Start listening for each population to finish:
            t = @async put!(state.channels[j][i], fetch(state.worker_output[j][i]))
            push!(state.tasks[j], t)
        end
    end
    kappa = 0
    resource_monitor = ResourceMonitor(;
        # Storing n times as many monitoring intervals as populations seems like it will
        # help get accurate resource estimates:
        max_recordings=options.populations * 100 * nout,
        start_reporting_at=options.populations * 3 * nout,
        window_size=options.populations * 2 * nout,
    )
    while sum(state.cycles_remaining) > 0
        kappa += 1
        if kappa > options.populations * nout
            kappa = 1
        end
        # nout, populations:
        j, i = state.task_order[kappa]

        # Check if error on population:
        if ropt.parallelism in (:multiprocessing, :multithreading)
            if istaskfailed(state.tasks[j][i])
                fetch(state.tasks[j][i])
                error("Task failed for population")
            end
        end
        # Non-blocking check if a population is ready:
        population_ready = if ropt.parallelism in (:multiprocessing, :multithreading)
            # TODO: Implement type assertions based on parallelism.
            isready(state.channels[j][i])
        else
            true
        end
        record_channel_state!(resource_monitor, population_ready)

        # Don't start more if this output has finished its cycles:
        # TODO - this might skip extra cycles?
        population_ready &= (state.cycles_remaining[j] > 0)
        if population_ready
            # @info "⏰state.cycles_remaining = $(state.cycles_remaining), j = $j, i = $i"
            # Take the fetch operation from the channel since its ready
            (cur_pop, best_seen, cur_record, cur_num_evals) = if ropt.parallelism in
                (
                :multiprocessing, :multithreading
            )
                take!(
                    state.channels[j][i]
                )
            else
                state.worker_output[j][i]
            end::DefaultWorkerOutputType{Population{T,L,N},HallOfFame{T,L,N}}
            state.last_pops[j][i] = copy(cur_pop)
            state.best_sub_pops[j][i] = best_sub_pop(cur_pop; topn=options.topn)
            @recorder state.record[] = recursive_merge(state.record[], cur_record)
            state.num_evals[j][i] += cur_num_evals
            dataset = datasets[j]
            cur_maxsize = state.cur_maxsizes[j]

            n_variables = size(dataset.X, 1)
            n_samples = size(dataset.X, 2)
            
            for member in cur_pop.members
                size = compute_complexity(member, options)
                update_frequencies!(state.all_running_search_statistics[j]; size)
            end
            #! format: off
            update_hall_of_fame!(state.halls_of_fame[j], cur_pop.members, options)
            update_hall_of_fame!(state.halls_of_fame[j], best_seen.members[best_seen.exists], options)
            #! format: on

            # @info "[Cycle Check] use_psrn=$use_psrn, psrn_manager !== nothing = $(psrn_manager !== nothing)"
            if use_psrn && psrn_manager !== nothing
                @info "[PSRN Check] Conditions met, attempting PSRN logic for j=$j, i=$i"
                # --- PSRN Logic ---
                if use_psrn && psrn_manager !== nothing
                    # If initialized successfully, run PSRN steps
            dominating = calculate_pareto_frontier(state.halls_of_fame[j])
            dominating = calculate_pareto_frontier(state.halls_of_fame[j])

                    dominating = calculate_pareto_frontier(state.halls_of_fame[j])

                    dominating_trees = [member.tree for member in dominating]

                    # Ensure N_PSRN_INPUT is correct from the initialized manager
                    if !isempty(dominating_trees) # Only run if HoF is not empty
                        start_psrn_task(
                            psrn_manager, dominating_trees, dataset, options, psrn_manager.N_PSRN_INPUT, n_variables
                        )
                        process_psrn_results!(
                            psrn_manager, state.halls_of_fame[j], dataset, options
                        )
                    end
                end
                # --- End PSRN Logic ---
            end



            dominating = calculate_pareto_frontier(state.halls_of_fame[j])

            dominating_trees = [member.tree for member in dominating]


            if options.save_to_file
                save_to_file(dominating, nout, j, dataset, options, ropt)
            end
            ###################################################################
            # Migration #######################################################
            if options.migration
                best_of_each = Population([
                    member for pop in state.best_sub_pops[j] for member in pop.members
                ])
                migrate!(
                    best_of_each.members => cur_pop, options; frac=options.fraction_replaced
                )
            end
            if options.hof_migration && length(dominating) > 0
                migrate!(dominating => cur_pop, options; frac=options.fraction_replaced_hof)
            end
            ###################################################################

            state.cycles_remaining[j] -= 1
            if state.cycles_remaining[j] == 0
                break
            end
            worker_idx = assign_next_worker!(
                state.worker_assignment;
                out=j,
                pop=i,
                parallelism=ropt.parallelism,
                state.procs,
            )
            iteration = if options.use_recorder
                key = "out$(j)_pop$(i)"
                find_iteration_from_record(key, state.record[]) + 1
            else
                0
            end

            c_rss = deepcopy(state.all_running_search_statistics[j])
            in_pop = copy(cur_pop::Population{T,L,N})
            state.worker_output[j][i] = @sr_spawner(
                begin
                    _dispatch_s_r_cycle(
                        in_pop,
                        dataset,
                        options;
                        pop=i,
                        out=j,
                        iteration,
                        ropt.verbosity,
                        cur_maxsize,
                        running_search_statistics=c_rss,
                    )
                end,
                parallelism = ropt.parallelism,
                worker_idx = worker_idx
            )
            if ropt.parallelism in (:multiprocessing, :multithreading)
                state.tasks[j][i] = @async put!(
                    state.channels[j][i], fetch(state.worker_output[j][i])
                )
            end

            total_cycles = ropt.niterations * options.populations
            state.cur_maxsizes[j] = get_cur_maxsize(;
                options, total_cycles, cycles_remaining=state.cycles_remaining[j]
            )
            move_window!(state.all_running_search_statistics[j])
            if !isnothing(progress_bar)
                head_node_occupation = estimate_work_fraction(resource_monitor)
                update_progress_bar!(
                    progress_bar,
                    only(state.halls_of_fame),
                    only(datasets),
                    options,
                    equation_speed,
                    head_node_occupation,
                    ropt.parallelism,
                )
            end
            if ropt.logger !== nothing
                logging_callback!(ropt.logger; state, datasets, ropt, options)
            end
        end
        yield()

        ################################################################
        ## Search statistics
        elapsed_since_speed_recording = time() - last_speed_recording_time
        if elapsed_since_speed_recording > 1.0
            num_evals_since_last, num_evals_last = let s = sum(sum, state.num_evals)
                s - num_evals_last, s
            end
            current_speed = num_evals_since_last / elapsed_since_speed_recording
            push!(equation_speed, current_speed)
            average_over_m_measurements = 20 # 20 second running average
            if length(equation_speed) > average_over_m_measurements
                deleteat!(equation_speed, 1)
            end
            last_speed_recording_time = time()
        end
        ################################################################

        ################################################################
        ## Printing code
        elapsed = time() - last_print_time
        # Update if time has passed
        if elapsed > print_every_n_seconds
            if ropt.verbosity > 0 && !ropt.progress && length(equation_speed) > 0

                # Dominating pareto curve - must be better than all simpler equations
                head_node_occupation = estimate_work_fraction(resource_monitor)
                total_cycles = ropt.niterations * options.populations
                print_search_state(
                    state.halls_of_fame,
                    datasets;
                    options,
                    equation_speed,
                    total_cycles,
                    state.cycles_remaining,
                    head_node_occupation,
                    parallelism=ropt.parallelism,
                    width=options.terminal_width,
                )
            end
            last_print_time = time()
        end
        ################################################################

        ################################################################
        ## Early stopping code
        if any((
            check_for_loss_threshold(state.halls_of_fame, options),
            check_for_user_quit(state.stdin_reader),
            check_for_timeout(start_time, options),
            check_max_evals(state.num_evals, options),
        ))
            @info "Stopping search due to early stopping condition."
            # Ensure PSRN task is cleaned up if stopping early
            if use_psrn && psrn_manager !== nothing
                if psrn_manager.current_task !== nothing && !istaskdone(psrn_manager.current_task)
                    @info "Waiting for final PSRN task to complete before exiting..."
                    # Optionally add a timeout here
                    try wait(psrn_manager.current_task) catch end
                end
                # Process any remaining results from all outputs
                for jj in 1:nout
                    process_psrn_results!(psrn_manager, state.halls_of_fame[jj], datasets[jj], options)
                end
            end

            break
        end
        ################################################################
    end
    if !isnothing(progress_bar)
        finish!(progress_bar)
    end
    return nothing
end

function _tear_down!(
    state::AbstractSearchState, ropt::AbstractRuntimeOptions, options::AbstractOptions
)
    close_reader!(state.stdin_reader)
    # Safely close all processes or threads
    if ropt.parallelism == :multiprocessing
        state.we_created_procs && rmprocs(state.procs)
    elseif ropt.parallelism == :multithreading
        nout = length(state.worker_output)
        for j in 1:nout, i in eachindex(state.worker_output[j])
            wait(state.worker_output[j][i])
        end
    end
    @recorder json3_write(state.record[], options.recorder_file)
    return nothing
end
function _format_output(
    state::AbstractSearchState,
    datasets,
    ropt::AbstractRuntimeOptions,
    options::AbstractOptions,
)
    nout = length(datasets)
    out_hof = if ropt.dim_out == 1
        embed_metadata(only(state.halls_of_fame), options, only(datasets))
    else
        map(Fix{2}(embed_metadata, options), state.halls_of_fame, datasets)
    end
    if ropt.return_state
        return (map(Fix{2}(embed_metadata, options), state.last_pops, datasets), out_hof)
    else
        return out_hof
    end
end

@stable default_mode = "disable" function _dispatch_s_r_cycle(
    in_pop::Population{T,L,N},
    dataset::Dataset,
    options::AbstractOptions;
    pop::Int,
    out::Int,
    iteration::Int,
    verbosity,
    cur_maxsize::Int,
    running_search_statistics,
) where {T,L,N}
    record = RecordType()
    @recorder record["out$(out)_pop$(pop)"] = RecordType(
        "iteration$(iteration)" => record_population(in_pop, options)
    )
    num_evals = 0.0
    normalize_frequencies!(running_search_statistics)
    out_pop, best_seen, evals_from_cycle = s_r_cycle(
        dataset,
        in_pop,
        options.ncycles_per_iteration,
        cur_maxsize,
        running_search_statistics;
        verbosity=verbosity,
        options=options,
        record=record,
    )
    num_evals += evals_from_cycle
    out_pop, evals_from_optimize = optimize_and_simplify_population(
        dataset, out_pop, options, cur_maxsize, record
    )
    num_evals += evals_from_optimize
    if options.batching
        for i_member in 1:(options.maxsize)
            score, result_loss = score_func(dataset, best_seen.members[i_member], options)
            best_seen.members[i_member].score = score
            best_seen.members[i_member].loss = result_loss
            num_evals += 1
        end
    end
    return (out_pop, best_seen, record, num_evals)
end
function _info_dump(
    state::AbstractSearchState,
    datasets::Vector{D},
    ropt::AbstractRuntimeOptions,
    options::AbstractOptions,
) where {D<:Dataset}
    ropt.verbosity <= 0 && return nothing

    nout = length(state.halls_of_fame)
    if nout > 1
        @info "Final populations:"
    else
        @info "Final population:"
    end
    for (j, (hall_of_fame, dataset)) in enumerate(zip(state.halls_of_fame, datasets))
        if nout > 1
            @info "Output $j:"
        end
        equation_strings = string_dominating_pareto_curve(
            hall_of_fame,
            dataset,
            options;
            width=@something(
                options.terminal_width,
                ropt.progress ? displaysize(stdout)[2] : nothing,
                Some(nothing)
            )
        )
        println(equation_strings)
    end

    if options.save_to_file
        output_directory = joinpath(
            something(options.output_directory, "outputs"), ropt.run_id
        )
        @info "Results saved to:"
        for j in 1:nout
            filename = nout > 1 ? "hall_of_fame_output$(j).csv" : "hall_of_fame.csv"
            output_file = joinpath(output_directory, filename)
            println("  - ", output_file)
        end
    end
    return nothing
end

include("MLJInterface.jl")
using .MLJInterfaceModule: SRRegressor, MultitargetSRRegressor

# Hack to get static analysis to work from within tests:
@ignore include("../test/runtests.jl")

# TODO: Hack to force ConstructionBase version
using ConstructionBase: ConstructionBase as _

include("precompile.jl")
redirect_stdout(devnull) do
    redirect_stderr(devnull) do
        do_precompilation(Val(:precompile))
    end
end

end #module SR
