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

-4
-4
-3
0
-15
0
0
-6
-4
-16
-14
-3
-14
-8
-3
-3
-3
0
-6
-5
0
-5
-3
-5
-5
0
-10
-12
-3
-11
-3
-14

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