# PSRNmodel.jl
module PSRNmodel

using ..PSRNfunctions # Import the refactored functions/types
import ..CoreModule.OperatorsModule
import ..CoreModule: Options, Dataset, safe_log, safe_sqrt

using Printf: @sprintf
using DynamicExpressions: Node, Expression
using CUDA
using LinearAlgebra: hcat
import Base: copy

# Import the logging helper from PSRNfunctions
using ..PSRNfunctions: log_gpu_mem # This now uses the corrected version

const T_GPU = PSRNfunctions.T_GPU
const T_IDX_GPU = PSRNfunctions.T_IDX_GPU

# --- Operator Types (no changes) ---
abstract type Operator end
mutable struct UnaryOperator <: Operator; name::String; base_func::Function; expr_gen::Function; end
abstract type BinaryOperator <: Operator end
mutable struct BinaryTriuOperator <: BinaryOperator; name::String; base_func::Function; expr_gen::Function; end
mutable struct BinarySquaredOperator <: BinaryOperator; name::String; base_func::Function; expr_gen::Function; end
function copy(op::UnaryOperator); UnaryOperator(op.name, op.base_func, op.expr_gen); end
function copy(op::BinaryTriuOperator); BinaryTriuOperator(op.name, op.base_func, op.expr_gen); end
function copy(op::BinarySquaredOperator); BinarySquaredOperator(op.name, op.base_func, op.expr_gen); end
function get_scale(op::Operator, in_dim::Int)
    if op isa UnaryOperator; return in_dim;
    elseif op isa BinaryTriuOperator; return T_IDX_GPU(in_dim) * (T_IDX_GPU(in_dim) + 1) ÷ 2;
    elseif op isa BinarySquaredOperator; return T_IDX_GPU(in_dim) * T_IDX_GPU(in_dim);
    else error("Unsupported operator type: $(typeof(op))"); end
end
const OPERATORS = Dict{String,Operator}(
    "Identity" => UnaryOperator("Identity", identity, identity), "Sin" => UnaryOperator("Sin", sin, sin),
    "Cos" => UnaryOperator("Cos", cos, cos), "Exp" => UnaryOperator("Exp", exp, exp),
    "Log" => UnaryOperator("Log", safe_log, safe_log), "Add" => BinaryTriuOperator("Add", +, +),
    "Mul" => BinaryTriuOperator("Mul", *, *), "Div" => BinarySquaredOperator("Div", /, /),
    "Sub" => BinarySquaredOperator("Sub", -, -), "SemiDiv" => BinaryTriuOperator("SemiDiv", /, /),
    "SemiSub" => BinaryTriuOperator("SemiSub", -, -), "Inv" => UnaryOperator("Inv", inv, x -> 1 / x),
    "Neg" => UnaryOperator("Neg", -, -), "Sqrt" => UnaryOperator("Sqrt", safe_sqrt, safe_sqrt)
)

# --- DRLayer (no changes) ---
mutable struct DRLayer; in_dim::Int; out_dim::Int; dr_indices::Vector{Int}; dr_mask::Vector{Bool};
    function DRLayer(in_dim::Int, dr_mask::Vector{Bool})
        out_dim = sum(dr_mask); dr_indices = findall(dr_mask); new(in_dim, out_dim, dr_indices, dr_mask)
    end
end
function forward(layer::DRLayer, x::CuArray)
     @info "[forward(DRLayer)] Applying mask. Input size: $(size(x)), Output dim: $(layer.out_dim)"
     out = x[:, layer.dr_mask]
     @info "[forward(DRLayer)] Mask applied. Output size: $(size(out))"
     return out
 end
function get_op_and_offset(layer::DRLayer, index::Int); return layer.dr_indices[index]; end


# --- SymbolLayer ---
mutable struct SymbolLayer
    in_dim::Int; out_dim::Int; operators::Vector{Operator}; operator_list::Vector{Operator}
    n_binary_U::Int; n_binary_D::Int; n_unary::Int; n_triu::Int; in_dim_square::Int
    out_dim_cum_ls::Union{Vector{Int},Nothing}; offset_tensor::Union{Matrix{Int},Nothing}
    triu_idx::Union{CuArray{T_IDX_GPU, 2}, Nothing}; squared_idx::Union{CuArray{T_IDX_GPU, 2}, Nothing}

    function SymbolLayer(in_dim::Int, operator_names::Vector{String})
        n_binary_U=0; n_binary_D=0; n_unary=0
        operators = [copy(OPERATORS[name]) for name in operator_names]
        operator_list = Operator[]
        n_triu = T_IDX_GPU(in_dim) * (T_IDX_GPU(in_dim) + 1) ÷ 2
        in_dim_square = T_IDX_GPU(in_dim) * T_IDX_GPU(in_dim)
        for op in operators
            if op isa UnaryOperator; n_unary += 1;
            elseif op isa BinaryTriuOperator; n_binary_U += 1;
            elseif op isa BinarySquaredOperator; n_binary_D += 1;
            else error("Invalid operator type"); end
        end
        append!(operator_list, filter(op -> op isa BinaryTriuOperator, operators))
        append!(operator_list, filter(op -> op isa BinarySquaredOperator, operators))
        append!(operator_list, filter(op -> op isa UnaryOperator, operators))
        out_dim = n_unary * in_dim + n_binary_U * n_triu + n_binary_D * in_dim_square
        triu_idx_gpu = (n_binary_U == 0) ? nothing : cu(get_triu_indices_cpu(in_dim))
        squared_idx_gpu = (n_binary_D == 0) ? nothing : cu(get_squared_indices_cpu(in_dim))
        layer = new(in_dim, out_dim, operators, operator_list, n_binary_U, n_binary_D, n_unary,
                    n_triu, in_dim_square, nothing, nothing, triu_idx_gpu, squared_idx_gpu)
        init_offset(layer)
        return layer
    end
end

# --- SymbolLayer CPU helpers (no changes) ---
function get_out_dim_cum_ls(layer::SymbolLayer)
    layer.out_dim_cum_ls !== nothing && return layer.out_dim_cum_ls
    out_dim_ls = [get_scale(op, layer.in_dim) for op in layer.operator_list]
    layer.out_dim_cum_ls = cumsum(out_dim_ls)
    return layer.out_dim_cum_ls
end
function get_offset_tensor(layer::SymbolLayer)
    layer.offset_tensor !== nothing && return layer.offset_tensor
    offset_tensor = zeros(Int, layer.out_dim, 2)
    arange_vector = 1:(layer.in_dim)
    unary_tensor = zeros(Int, layer.in_dim, 2); unary_tensor[:, 1] = arange_vector; unary_tensor[:, 2] .= layer.in_dim + 1
    binary_U_tensor = zeros(Int, layer.n_triu, 2); idx = 1; for i=1:layer.in_dim, j=i:layer.in_dim; binary_U_tensor[idx,:] .= (i,j); idx+=1; end
    binary_D_tensor = zeros(Int, layer.in_dim_square, 2); idx=1; for i=1:layer.in_dim, j=1:layer.in_dim; binary_D_tensor[idx,:] .= (i,j); idx+=1; end
    start = 1
    for func in layer.operator_list
        t, len = if func isa UnaryOperator; (unary_tensor, layer.in_dim);
                 elseif func isa BinaryTriuOperator; (binary_U_tensor, layer.n_triu);
                 elseif func isa BinarySquaredOperator; (binary_D_tensor, layer.in_dim_square);
                 else error("Unknown operator type"); end
        offset_tensor[start:(start+len-1), :] = t; start += len
    end
    layer.offset_tensor = offset_tensor; return offset_tensor
end
function init_offset(layer::SymbolLayer); get_out_dim_cum_ls(layer); get_offset_tensor(layer); end
function get_op_and_offset(layer::SymbolLayer, index::Int)
    out_dim_cum_ls = layer.out_dim_cum_ls; offset_tensor = layer.offset_tensor
    op_idx = findfirst(>=(index), out_dim_cum_ls)
    isnothing(op_idx) && error("Index $index out of bounds")
    return layer.operator_list[op_idx], offset_tensor[index, :]
end

# --- SymbolLayer Forward (Logging kept, should work now) ---
function forward(layer::SymbolLayer, x::CuArray{T, 2}) where T
    @info "[forward(SymbolLayer) START] Input x size: $(size(x)), Layer in_dim: $(layer.in_dim), Layer out_dim: $(layer.out_dim)"
    log_gpu_mem("SymbolLayer forward start")
    results = CuArray{T, 2}[] # Expecting 2D results now
    expected_total_cols = 0

    @info "[forward(SymbolLayer)] Entering operator loop ($(length(layer.operator_list)) operators)"
    for (op_idx, op) in enumerate(layer.operator_list)
        op_name = op.name
        op_type = typeof(op)
        @info "[forward(SymbolLayer) Op Loop $op_idx/$(length(layer.operator_list))] Processing Op: '$op_name' ($op_type)"
        local res::CuArray{T, 2}
        try
            if op isa UnaryOperator
                @info "[forward(SymbolLayer) Op '$op_name'] Applying unary broadcast..."
                res = op.base_func.(x)
                CUDA.synchronize()
                @info "[forward(SymbolLayer) Op '$op_name'] Unary broadcast complete. Result size: $(size(res))" # Should be 2D
                log_gpu_mem("SymbolLayer after unary '$op_name'")
            elseif op isa BinaryTriuOperator
                @info "[forward(SymbolLayer) Op '$op_name'] Applying binary triu..."
                res = apply_binary_op_triu(op.base_func, x, layer.triu_idx) # Calls corrected function
                # CUDA sync happens inside apply_binary_op_triu
                @info "[forward(SymbolLayer) Op '$op_name'] Binary triu complete. Result size: $(size(res))" # Should be 2D
                log_gpu_mem("SymbolLayer after triu '$op_name'")
            elseif op isa BinarySquaredOperator
                 @info "[forward(SymbolLayer) Op '$op_name'] Applying binary squared..."
                 res = apply_binary_op_squared(op.base_func, x, layer.squared_idx) # Calls corrected function
                 # CUDA sync happens inside apply_binary_op_squared
                 @info "[forward(SymbolLayer) Op '$op_name'] Binary squared complete. Result size: $(size(res))" # Should be 2D
                 log_gpu_mem("SymbolLayer after squared '$op_name'")
            else
                error("Unsupported operator type during forward pass: $op_type")
            end

            # Check for non-finite values immediately after computation
            if !all(isfinite, res)
                num_nonfinite = sum(.!isfinite.(res))
                 @warn "[forward(SymbolLayer) Op '$op_name'] Detected $num_nonfinite non-finite values in result size $(size(res)). Consider using safer operators or checking inputs."
            end

            # This push should now work as `res` is expected to be 2D
            push!(results, res)
            expected_total_cols += size(res, 2)

        catch e
            @error "[forward(SymbolLayer) Op '$op_name'] Error during processing: $e"
            bt = stacktrace(catch_backtrace())
            @error join(string.(bt), "\n")
            rethrow(e) # Propagate error
        end
    end
    @info "[forward(SymbolLayer)] Operator loop finished."

    if isempty(results)
        @warn "[forward(SymbolLayer)] No operator results generated."
        return CUDA.zeros(T, size(x, 1), 0)
    else
        @info "[forward(SymbolLayer)] Concatenating $(length(results)) results..."
        sizes_str = join([string(size(r)) for r in results], ", ")
        @info "[forward(SymbolLayer)] Sizes to concat: [$sizes_str]" # All should be 2D
        @info "[forward(SymbolLayer)] Expected total output columns: $expected_total_cols (should match layer.out_dim = $(layer.out_dim))"
        log_gpu_mem("SymbolLayer before hcat")

        final_result = hcat(results...)
        CUDA.synchronize() # Sync after potentially large hcat

        @info "[forward(SymbolLayer) END] Concatenation complete. Final output size: $(size(final_result))"
        log_gpu_mem("SymbolLayer after hcat")

        # Final check
        if size(final_result, 2) != layer.out_dim
             @warn "[forward(SymbolLayer) END] Mismatch! Final output columns $(size(final_result, 2)) != layer.out_dim $(layer.out_dim)"
        end
        if !all(isfinite, final_result)
             num_nonfinite = sum(.!isfinite.(final_result))
             @warn "[forward(SymbolLayer) END] Detected $num_nonfinite non-finite values in final concatenated output size $(size(final_result))."
        end
        return final_result
    end
end


# --- PSRN Model ---
mutable struct PSRN
    n_variables::Int; operators::Vector{String}; n_symbol_layers::Int
    layers::Vector{Union{SymbolLayer,DRLayer}}; out_dim::Int; use_dr_mask::Bool
    current_expr_ls::Vector{Expression}; options::Options; PSRN_topk::Int

    function PSRN(;n_variables::Int, operators::Vector{String}, n_symbol_layers::Int,
                  dr_mask::Union{Vector{Bool},Nothing}=nothing, options::Options=Options(), PSRN_topk::Int)
        @info "Initializing PSRN with CUDA.jl backend."
        layers = Union{SymbolLayer,DRLayer}[]
        current_out_dim = n_variables
        use_dr_mask = !isnothing(dr_mask)
        for i in 1:n_symbol_layers
            @info "Creating SymbolLayer $i / $n_symbol_layers (input dim: $current_out_dim)"
            log_gpu_mem("Before SymbolLayer $i create")
            layer = SymbolLayer(current_out_dim, operators)
            log_gpu_mem("After SymbolLayer $i create")
            push!(layers, layer)
            current_out_dim = layer.out_dim
            @info "  SymbolLayer $i created. Output dim: $current_out_dim"
            if use_dr_mask && i == n_symbol_layers
                 @info "Creating final DRLayer (input dim: $current_out_dim)"
                 dr_layer = DRLayer(current_out_dim, dr_mask)
                 push!(layers, dr_layer)
                 current_out_dim = dr_layer.out_dim
                 @info "  DRLayer created. Output dim after DR: $current_out_dim"
            end
        end
        @info "PSRN Initialization complete. Final output dim: $current_out_dim"
        return new(n_variables, operators, n_symbol_layers, layers, current_out_dim,
                   use_dr_mask, Expression[], options, PSRN_topk)
    end
end

# --- PSRN Forward (Logging kept) ---
function PSRN_forward(model::PSRN, x::CuArray{T, 2}) where T
    @info "[PSRN_forward START] Initial input size: $(size(x))"
    log_gpu_mem("PSRN_forward start")
    h = x
    @info "[PSRN_forward] Entering layer loop ($(length(model.layers)) layers)"
    for (i, layer) in enumerate(model.layers)
        layer_type = typeof(layer)
        @info "[PSRN_forward Loop $i/$(length(model.layers))] Processing Layer Type: $layer_type"
        @info "[PSRN_forward Loop $i/$(length(model.layers))] Input size: $(size(h))"
        log_gpu_mem("PSRN_forward before layer $i ($layer_type)")

        h_new = forward(layer, h) # forward functions now log internally
        CUDA.synchronize() # Ensure layer computation is done

        @info "[PSRN_forward Loop $i/$(length(model.layers))] Layer complete. Output size: $(size(h_new))"
        log_gpu_mem("PSRN_forward after layer $i ($layer_type)")

        # NaN/Inf check per layer
        if !all(isfinite, h_new)
            num_nonfinite = sum(.!isfinite.(h_new))
             @warn "[PSRN_forward Loop $i/$(length(model.layers))] Detected $num_nonfinite non-finite values after layer $i ($layer_type)."
        end
        h = h_new # Update h for the next layer
    end
    @info "[PSRN_forward END] Layer loop finished. Final output size: $(size(h))"
    log_gpu_mem("PSRN_forward end")
    return h
end

# --- get_best_expr_and_MSE_topk (Logging kept) ---
function get_best_expr_and_MSE_topk(
    model::PSRN,
    X_gpu::CuArray{T_GPU, 2},
    Y_gpu::CuArray{T_GPU, 1}
) where {T_GPU}

    @info "[get_best_expr_and_MSE_topk START]"
    log_gpu_mem("get_best_expr_and_MSE_topk start")
    @info "🤓1"
    batch_size = size(X_gpu, 1)
    Y_col_gpu = reshape(Y_gpu, batch_size, 1)
    @info "[get_best_expr_and_MSE_topk] Prepared Y_col_gpu size: $(size(Y_col_gpu))"

    @info "🤓2 >> Calling PSRN_forward..."
    log_gpu_mem("get_best_expr_and_MSE_topk before PSRN_forward")
    H_gpu = PSRN_forward(model, X_gpu) # This call now has internal logging
    CUDA.synchronize() # Ensure forward pass is complete
    @info "🤓2 << Returned from PSRN_forward. H_gpu size: $(size(H_gpu))" # Should be 2D
    log_gpu_mem("get_best_expr_and_MSE_topk after PSRN_forward")

    # Check H_gpu immediately
     if !all(isfinite.(H_gpu))
         num_nonfinite = sum(.!isfinite.(H_gpu))
         @warn "[get_best_expr_and_MSE_topk] Detected $num_nonfinite non-finite values in H_gpu (output of PSRN_forward)."
     end

    @info "[get_best_expr_and_MSE_topk] Calculating squared errors..."
    diffs_gpu = H_gpu .- Y_col_gpu
    squared_errors_gpu = diffs_gpu .^ 2
    CUDA.synchronize()
    @info "[get_best_expr_and_MSE_topk] Squared errors calculated."
    log_gpu_mem("get_best_expr_and_MSE_topk after sq error")

    @info "🤓3 >> Summing errors..."
    sum_squared_errors_gpu = sum(squared_errors_gpu, dims=1)
    CUDA.synchronize()
    @info "🤓3 << Errors summed. Size: $(size(sum_squared_errors_gpu))"
    log_gpu_mem("get_best_expr_and_MSE_topk after sum")

    @info "[get_best_expr_and_MSE_topk] Cleaning errors (handling non-finite)..."
    large_val = T_GPU(Inf)
    finite_mask = isfinite.(sum_squared_errors_gpu)
    cleaned_errors_gpu = ifelse.(finite_mask, sum_squared_errors_gpu, large_val)
    CUDA.synchronize()
    @info "🤓4 >> Finding top K indices..."

    num_to_select = min(model.PSRN_topk, length(cleaned_errors_gpu))
    if num_to_select <= 0
        @warn "[get_best_expr_and_MSE_topk] PSRN_topk resulted in 0 expressions to select. Returning empty list."
        H_gpu=nothing; diffs_gpu=nothing; squared_errors_gpu=nothing; sum_squared_errors_gpu=nothing; cleaned_errors_gpu=nothing; finite_mask=nothing
        CUDA.reclaim()
        return Expression[]
    end
    @info "[get_best_expr_and_MSE_topk] Selecting top $num_to_select indices."

    top_k_indices_gpu = CUDA.sortperm(vec(cleaned_errors_gpu))[1:num_to_select]
    CUDA.synchronize()
    @info "🤓5 >> Transferring indices to CPU..."
    indices_cpu = collect(top_k_indices_gpu)
    @info "🤓5 << Indices transferred: $(length(indices_cpu)) indices."
    log_gpu_mem("get_best_expr_and_MSE_topk after sortperm+collect")

    @info "[get_best_expr_and_MSE_topk] Releasing large GPU arrays (H_gpu, errors etc.)..."
    H_gpu=nothing; diffs_gpu=nothing; squared_errors_gpu=nothing; sum_squared_errors_gpu=nothing; cleaned_errors_gpu=nothing; finite_mask=nothing; top_k_indices_gpu = nothing
    CUDA.reclaim()
    log_gpu_mem("get_best_expr_and_MSE_topk after cleanup")

    @info "🤓6 >> Reconstructing expressions..."
    expr_best_ls = Expression[]
    reconstruction_error = false
    for (count, idx) in enumerate(indices_cpu)
         @info "[get_best_expr_and_MSE_topk Recon Loop $count/$(length(indices_cpu))] Reconstructing expr for index $idx..."
         try
             expr = get_expr(model, Int(idx))
             push!(expr_best_ls, expr)
         catch e
             @error "[get_best_expr_and_MSE_topk Recon Loop $count] Error reconstructing expression for index $idx: $e"
             reconstruction_error = true
         end
    end
    @info "🤓6 << Expression reconstruction complete. Found $(length(expr_best_ls)) expressions."
    if reconstruction_error
         @warn "[get_best_expr_and_MSE_topk] Errors occurred during expression reconstruction for one or more indices."
    end

    @info "[get_best_expr_and_MSE_topk END]"
    return expr_best_ls
end

# --- Base.show (no changes) ---
function Base.show(io::IO, model::PSRN)
    print(io,"PSRN(n_variables=$(model.n_variables), operators=$(model.operators), n_layers=$(model.n_symbol_layers))\n")
    layer_dims = [model.n_variables]; for layer in model.layers; push!(layer_dims, layer.out_dim); end
    print(io, "Layer dimensions: "); print(io, join(layer_dims, " → "))
end

# --- get_expr / _get_expr (no changes) ---
function get_expr(model::PSRN, index::Int); return _get_expr(model, index, length(model.layers)); end
function _get_expr(model::PSRN, index::Int, layer_idx::Int)
    if layer_idx < 1
        (index <= 0 || index > length(model.current_expr_ls)) && error("Invalid base index $index (layer $layer_idx), current_expr_ls size: $(length(model.current_expr_ls))")
        return model.current_expr_ls[index]
    end
    layer = model.layers[layer_idx]
    if layer isa DRLayer
        new_index = get_op_and_offset(layer, index)
        return _get_expr(model, new_index, layer_idx - 1)
    else # SymbolLayer
        op, offsets = get_op_and_offset(layer, index)
        expr1 = _get_expr(model, offsets[1], layer_idx - 1)
        if op isa UnaryOperator
            op.name == "Identity" && return expr1
            return op.expr_gen(expr1)
        else # Binary operator
            expr2 = _get_expr(model, offsets[2], layer_idx - 1)
            return op.expr_gen(expr1, expr2)
        end
    end
end

# --- Exports (no changes) ---
export PSRN, SymbolLayer, DRLayer, Operator, UnaryOperator, BinaryOperator
export get_best_expr_and_MSE_topk, get_expr, get_op_and_offset

end # module PSRNmodel