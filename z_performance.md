commit: correct replacement

# Problem

X = randn(Float32, 5, 100) # speed: ~1e+05 for default ~4e+05 for -t 16
y = 2 * cos.(X[4, :]) .^ 3 + X[1, :] .^ 2 .- 2 # harder problem

options = SymbolicRegression.Options(;
    binary_operators=[+, *, /, -],
     unary_operators=[cos, exp, sin, log],
    timeout_in_seconds=30
)


# clear && julia -t 1 example.jl --project=.

## w/PSRN

-15
-5
-15
-15
-3

-15
-3
-3
-3
-15
-15

# w/o PSRN

-5
-5
-4
-4
-5
-5
-6
-3
-3
-11
-14
-5
0
-3
0
-3