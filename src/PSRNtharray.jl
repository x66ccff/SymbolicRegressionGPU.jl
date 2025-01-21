module PSRNtharray

include("THArrays/src/THArrays.jl")

using .THArrays

export TorchNumber, Tensor, Scalar, eltype_id,
    THC, THAD, TrackerAD, THJIT,
    Device, CPU, CUDA, to, on

end # module