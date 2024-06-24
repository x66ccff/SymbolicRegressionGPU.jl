using SymbolicRegression
using Random: MersenneTwister
using Zygote
using MLJBase: machine, fit!, predict

rng = MersenneTwister(0)
X = NamedTuple{(:x1, :x2, :x3, :x4, :x5)}(ntuple(_ -> randn(rng, Float32, 30), Val(5)))
classes = rand(rng, 1:2, 30)
p1 = rand(rng, Float32, 2)
p2 = rand(rng, Float32, 2)

y = [
    2 * cos(X.x4[i] + p1[classes[i]]) + X.x1[i]^2 - p2[classes[i]] for
    i in eachindex(classes)
]

y .+= classes

X = (; X..., classes)

model = SRRegressor(;
    niterations=40,
    binary_operators=[+, *, /, -],
    unary_operators=[cos, exp],
    populations=10,
    expression_type=ParametricExpression,
    expression_options=(; max_parameters=2),
    autodiff_backend=:Zygote,
    parallelism=:multithreading,
)

mach = machine(model, X, y)
fit!(mach)
ypred = predict(mach, X)
