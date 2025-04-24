# PSRNfunctions.jl
module PSRNfunctions

import ..CoreModule: safe_log, safe_sqrt

using CUDA
using Printf # For formatted printing

const T_GPU = Float32
const T_IDX_GPU = Int32

# --- Helper for Memory Logging ---
# Formats bytes into KB/MB/GB
function format_bytes(bytes)
    if bytes < 1024
        return @sprintf("%d B", bytes)
    elseif bytes < 1024^2
        return @sprintf("%.2f KB", bytes / 1024)
    elseif bytes < 1024^3
        return @sprintf("%.2f MB", bytes / 1024^2)
    else
        return @sprintf("%.2f GB", bytes / 1024^3)
    end
end

function log_gpu_mem(label::String="")
    try
        # CORRECTED: Access fields directly
        info = CUDA.MemoryInfo()
        free_bytes = info.free
        total_bytes = info.total
        used_bytes = total_bytes - free_bytes # Calculate used
        used_pct = total_bytes > 0 ? (used_bytes / total_bytes) * 100 : 0.0 # Avoid NaN if total is 0
        @info @sprintf("[GPU Mem %s] Used: %s / %s (%.1f%%)",
                       label, format_bytes(used_bytes), format_bytes(total_bytes), used_pct)
    catch e
        @warn "Could not get CUDA memory info: $e"
    end
    return nothing
end


# --- Index Generation (Run on CPU, transfer result to GPU) ---
# (No changes needed here)
function get_triu_indices_cpu(n::Int)
    num_indices = n * (n + 1) ÷ 2
    indices_matrix = Matrix{T_IDX_GPU}(undef, 2, num_indices)
    idx = 1
    for i in 1:n
        for j in i:n
            indices_matrix[1, idx] = T_IDX_GPU(i)
            indices_matrix[2, idx] = T_IDX_GPU(j)
            idx += 1
        end
    end
    return indices_matrix
end

function get_squared_indices_cpu(n::Int)
    num_indices = n * n
    indices_matrix = Matrix{T_IDX_GPU}(undef, 2, num_indices)
    idx = 1
    for i in 1:n
        for j in 1:n
            indices_matrix[1, idx] = T_IDX_GPU(i)
            indices_matrix[2, idx] = T_IDX_GPU(j)
            idx += 1
        end
    end
    return indices_matrix
end


# --- Binary Kernels (Implemented via Broadcasting and Indexing) ---

function apply_binary_op_triu(op::Function, x::CuArray{T, 2}, indices::CuArray{Ti, 2}) where {T, Ti}
    op_name = nameof(op)
    @info "[apply_binary_op_triu START] Op: $op_name, Input x size: $(size(x))"
    log_gpu_mem("apply_binary_op_triu start")

    # CORRECTED: Index to get vectors (1D CuArrays)
    l_idx = indices[1, :]
    r_idx = indices[2, :]

    @info "[apply_binary_op_triu] Creating l_values..."
    l_values = x[:, l_idx] # Result should now be 2D: (batch_size, n_triu_indices)
    # CUDA.synchronize() # Sync after potential large allocation/gather
    @info "[apply_binary_op_triu] Created l_values size: $(size(l_values))" # Should be 2D
    log_gpu_mem("apply_binary_op_triu after l_values")

    @info "[apply_binary_op_triu] Creating r_values..."
    r_values = x[:, r_idx] # Result should now be 2D: (batch_size, n_triu_indices)
    # CUDA.synchronize() # Sync after potential large allocation/gather
    @info "[apply_binary_op_triu] Created r_values size: $(size(r_values))" # Should be 2D
    log_gpu_mem("apply_binary_op_triu after r_values")

    @info "[apply_binary_op_triu] Broadcasting op: $op_name..."
    result = op.(l_values, r_values) # Result should be 2D
    # CUDA.synchronize() # Sync after computation
    @info "[apply_binary_op_triu] Broadcasting complete. Result size: $(size(result))" # Should be 2D
    log_gpu_mem("apply_binary_op_triu after op broadcast")

    # Clean up intermediates explicitly if memory is tight (optional, GC usually handles it)
    # l_values = nothing
    # r_values = nothing
    # CUDA.reclaim() # Force memory cleanup

    @info "[apply_binary_op_triu END] Op: $op_name"
    return result # Should be 2D
end

function apply_binary_op_squared(op::Function, x::CuArray{T, 2}, indices::CuArray{Ti, 2}) where {T, Ti}
    op_name = nameof(op)
    @info "[apply_binary_op_squared START] Op: $op_name, Input x size: $(size(x))"
    log_gpu_mem("apply_binary_op_squared start")

    # CORRECTED: Index to get vectors (1D CuArrays)
    l_idx = indices[1, :]
    r_idx = indices[2, :]

    @info "[apply_binary_op_squared] Creating l_values..."
    l_values = x[:, l_idx] # Result should now be 2D: (batch_size, n_squared_indices)
    # CUDA.synchronize()
    @info "[apply_binary_op_squared] Created l_values size: $(size(l_values))" # Should be 2D
    log_gpu_mem("apply_binary_op_squared after l_values")

    @info "[apply_binary_op_squared] Creating r_values..."
    r_values = x[:, r_idx] # Result should now be 2D: (batch_size, n_squared_indices)
    # CUDA.synchronize()
    @info "[apply_binary_op_squared] Created r_values size: $(size(r_values))" # Should be 2D
    log_gpu_mem("apply_binary_op_squared after r_values")

    @info "[apply_binary_op_squared] Broadcasting op: $op_name..."
    result = op.(l_values, r_values) # Result should be 2D
    # CUDA.synchronize()
    @info "[apply_binary_op_squared] Broadcasting complete. Result size: $(size(result))" # Should be 2D
    log_gpu_mem("apply_binary_op_squared after op broadcast")

    @info "[apply_binary_op_squared END] Op: $op_name"
    return result # Should be 2D
end

# Example usage wrappers (no changes needed)
function add_kernel(x::CuArray{T, 2}, indices::CuArray{Ti, 2}) where {T, Ti}
    return apply_binary_op_triu(+, x, indices)
end

function mul_kernel(x::CuArray{T, 2}, indices::CuArray{Ti, 2}) where {T, Ti}
    return apply_binary_op_triu(*, x, indices)
end

function sub_kernel(x::CuArray{T, 2}, indices::CuArray{Ti, 2}) where {T, Ti}
    return apply_binary_op_squared(-, x, indices)
end

function div_kernel(x::CuArray{T, 2}, indices::CuArray{Ti, 2}) where {T, Ti}
    return apply_binary_op_squared(/, x, indices)
end

function semisub_kernel(x::CuArray{T, 2}, indices::CuArray{Ti, 2}) where {T, Ti}
     return apply_binary_op_triu(-, x, indices)
end

function semidiv_kernel(x::CuArray{T, 2}, indices::CuArray{Ti, 2}) where {T, Ti}
     return apply_binary_op_triu(/, x, indices)
end


# Export necessary functions
export T_GPU, T_IDX_GPU, log_gpu_mem # Export helper
export get_triu_indices_cpu, get_squared_indices_cpu
export apply_binary_op_triu, apply_binary_op_squared # Export core functions
export add_kernel, mul_kernel, sub_kernel, div_kernel, semisub_kernel, semidiv_kernel

end # module PSRNfunctions