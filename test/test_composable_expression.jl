@testitem "Integration Test with fit! and Performance Check" tags = [:part3] begin
    include("../examples/template_expression.jl")
end
@testitem "Test ComposableExpression" tags = [:part2] begin
    using SymbolicRegression: ComposableExpression, Node
    using DynamicExpressions: OperatorEnum

    operators = OperatorEnum(; binary_operators=(+, *, /, -), unary_operators=(sin, cos))
    variable_names = (i -> "x$i").(1:3)
    ex = ComposableExpression(Node(Float64; feature=1); operators, variable_names)
    x = randn(32)
    y = randn(32)

    @test ex(x, y) == x
end

@testitem "Test interface for ComposableExpression" tags = [:part2] begin
    using SymbolicRegression: ComposableExpression
    using DynamicExpressions.InterfacesModule: Interfaces, ExpressionInterface
    using DynamicExpressions: OperatorEnum

    operators = OperatorEnum(; binary_operators=(+, *, /, -), unary_operators=(sin, cos))
    variable_names = (i -> "x$i").(1:3)
    x1 = ComposableExpression(Node(Float64; feature=1); operators, variable_names)
    x2 = ComposableExpression(Node(Float64; feature=2); operators, variable_names)

    f = x1 * sin(x2)
    g = f(f, f)

    @test string_tree(f) == "x1 * sin(x2)"
    @test string_tree(g) == "(x1 * sin(x2)) * sin(x1 * sin(x2))"

    @test Interfaces.test(ExpressionInterface, ComposableExpression, [f, g])
end

@testitem "Test interface for TemplateExpression" tags = [:part2] begin
    using SymbolicRegression
    using SymbolicRegression: TemplateExpression
    using DynamicExpressions.InterfacesModule: Interfaces, ExpressionInterface
    using DynamicExpressions: OperatorEnum

    operators = OperatorEnum(; binary_operators=(+, *, /, -), unary_operators=(sin, cos))
    variable_names = (i -> "x$i").(1:3)
    x1 = ComposableExpression(Node(Float64; feature=1); operators, variable_names)
    x2 = ComposableExpression(Node(Float64; feature=2); operators, variable_names)

    structure = TemplateStructure{(:f, :g)}(
        ((; f, g), (x1, x2)) -> f(f(f(x1))) - f(g(x2, x1))
    )
    @test structure.num_features == (; f=1, g=2)

    expr = TemplateExpression((; f=x1, g=x2 * x2); structure, operators, variable_names)

    @test String(string_tree(expr)) == "f = #1; g = #2 * #2"
    @test String(string_tree(expr; pretty=true)) == "╭ f = #1\n╰ g = #2 * #2"
    @test string_tree(get_tree(expr), operators) == "x1 - (x1 * x1)"
    @test Interfaces.test(ExpressionInterface, TemplateExpression, [expr])
end

@testitem "Printing and evaluation of TemplateExpression" tags = [:part2] begin
    using SymbolicRegression

    structure = TemplateStructure{(:f, :g)}(
        ((; f, g), (x1, x2, x3)) -> sin(f(x1, x2)) + g(x3)^2
    )
    operators = Options().operators
    variable_names = ["x1", "x2", "x3"]

    x1, x2, x3 = [
        ComposableExpression(Node{Float64}(; feature=i); operators, variable_names) for
        i in 1:3
    ]
    f = x1 * x2
    g = x1
    expr = TemplateExpression((; f, g); structure, operators, variable_names)

    # Default printing strategy:
    @test String(string_tree(expr)) == "f = #1 * #2; g = #1"

    x1_val = randn(5)
    x2_val = randn(5)

    # The feature indicates the index passed as argument:
    @test x1(x1_val) ≈ x1_val
    @test x2(x1_val, x2_val) ≈ x2_val
    @test x1(x2_val) ≈ x2_val

    # Composing expressions and then calling:
    @test String(string_tree((x1 * x2)(x3, x3))) == "x3 * x3"

    # Can evaluate with `sin` even though it's not in the allowed operators!
    X = randn(3, 5)
    x1_val = X[1, :]
    x2_val = X[2, :]
    x3_val = X[3, :]
    @test expr(X) ≈ @. sin(x1_val * x2_val) + x3_val^2

    # This is even though `g` is defined on `x1` only:
    @test g(x3_val) ≈ x3_val
end

@testitem "Test error handling" tags = [:part2] begin
    using SymbolicRegression
    using SymbolicRegression: ComposableExpression, Node, ValidVector
    using DynamicExpressions: OperatorEnum

    operators = OperatorEnum(; binary_operators=(+, *, /, -), unary_operators=(sin, cos))
    variable_names = (i -> "x$i").(1:3)
    ex = ComposableExpression(Node{Float64}(; feature=1); operators, variable_names)

    # Test error for unsupported input type with specific message
    @test_throws "ComposableExpression does not support input of type String" ex(
        "invalid input"
    )

    # Test ValidVector operations with numbers
    x = ValidVector([1.0, 2.0, 3.0], true)

    # Test binary operations between ValidVector and Number
    @test (x + 2.0).x ≈ [3.0, 4.0, 5.0]
    @test (2.0 + x).x ≈ [3.0, 4.0, 5.0]
    @test (x * 2.0).x ≈ [2.0, 4.0, 6.0]
    @test (2.0 * x).x ≈ [2.0, 4.0, 6.0]

    # Test unary operations on ValidVector
    @test sin(x).x ≈ sin.([1.0, 2.0, 3.0])
    @test cos(x).x ≈ cos.([1.0, 2.0, 3.0])
    @test abs(x).x ≈ [1.0, 2.0, 3.0]
    @test (-x).x ≈ [-1.0, -2.0, -3.0]

    # Test propagation of invalid flag
    invalid_x = ValidVector([1.0, 2.0, 3.0], false)
    @test !((invalid_x + 2.0).valid)
    @test !((2.0 + invalid_x).valid)
    @test !(sin(invalid_x).valid)

    # Test that regular numbers are considered valid
    @test (x + 2).valid
    @test sin(x).valid
end
@testitem "Test validity propagation with NaN" tags = [:part2] begin
    using SymbolicRegression: ComposableExpression, Node, ValidVector
    using DynamicExpressions: OperatorEnum

    operators = OperatorEnum(; binary_operators=(+, *, /, -), unary_operators=(sin, cos))
    variable_names = (i -> "x$i").(1:3)
    x1 = ComposableExpression(Node{Float64}(; feature=1); operators, variable_names)
    x2 = ComposableExpression(Node{Float64}(; feature=2); operators, variable_names)
    x3 = ComposableExpression(Node{Float64}(; feature=3); operators, variable_names)

    ex = 1.0 + x2 / x1

    @test ex([1.0], [2.0]) ≈ [3.0]

    @test ex([1.0, 1.0], [2.0, 2.0]) |> Base.Fix1(count, isnan) == 0
    @test ex([1.0, 0.0], [2.0, 2.0]) |> Base.Fix1(count, isnan) == 2

    x1_val = ValidVector([1.0, 2.0], false)
    x2_val = ValidVector([1.0, 2.0], false)
    @test ex(x1_val, x2_val).valid == false
end

@testitem "Test nothing return and type inference for TemplateExpression" tags = [:part2] begin
    using SymbolicRegression
    using Test: @inferred

    # Create a template expression that divides by x1
    structure = TemplateStructure{(:f,)}(((; f), (x1, x2)) -> 1.0 + f(x1) / x1)
    operators = Options(; binary_operators=(+, -, *, /)).operators
    variable_names = ["x1", "x2"]

    x1 = ComposableExpression(Node{Float64}(; feature=1); operators, variable_names)
    x2 = ComposableExpression(Node{Float64}(; feature=2); operators, variable_names)
    expr = TemplateExpression((; f=x1); structure, operators, variable_names)

    # Test division by zero returns nothing
    X = [0.0 1.0]'
    @test expr(X) === nothing

    # Test type inference
    X_good = [1.0 2.0]'
    @test @inferred(Union{Nothing,Vector{Float64}}, expr(X_good)) ≈ [2.0]

    # Test type inference with ValidVector input
    x1_val = ValidVector([1.0], true)
    x2_val = ValidVector([2.0], true)
    @test @inferred(ValidVector{Vector{Float64}}, x1(x1_val, x2_val)).x ≈ [1.0]

    x2_val_false = ValidVector([2.0], false)
    @test @inferred(x1(x1_val, x2_val_false)).valid == false
end
@testitem "Test compatibility with power laws" tags = [:part3] begin
    using SymbolicRegression
    using DynamicExpressions: OperatorEnum

    operators = OperatorEnum(; binary_operators=(+, -, *, /, ^))
    variable_names = ["x1", "x2"]
    x1 = ComposableExpression(Node{Float64}(; feature=1); operators, variable_names)
    x2 = ComposableExpression(Node{Float64}(; feature=2); operators, variable_names)

    structure = TemplateStructure{(:f,)}(((; f), (x1, x2)) -> f(x1)^f(x2))
    expr = TemplateExpression((; f=x1); structure, operators, variable_names)

    # There shouldn't be an error when we evaluate with invalid
    # expressions, even though the source of the NaN comes from the structure
    # function itself:
    X = -rand(2, 32)
    @test expr(X) === nothing
end

@testitem "Test constraints checking in TemplateExpression" tags = [:part2] begin
    using SymbolicRegression
    using SymbolicRegression: CheckConstraintsModule as CC

    # Create a template expression with nested exponentials
    options = Options(;
        binary_operators=(+, -, *, /),
        unary_operators=(exp,),
        nested_constraints=[exp => [exp => 1]], # Only allow one nested exp
    )
    operators = options.operators
    variable_names = ["x1", "x2"]

    # Create a valid inner expression
    x1 = ComposableExpression(Node{Float64}(; feature=1); operators, variable_names)
    valid_expr = exp(x1)  # One exp is ok

    # Create an invalid inner expression with too many nested exp
    invalid_expr = exp(exp(exp(x1)))
    # Three nested exp's violates constraint

    @test CC.check_constraints(valid_expr, options, 20)
    @test !CC.check_constraints(invalid_expr, options, 20)
end

@testitem "Test feature constraints in TemplateExpression" tags = [:part1] begin
    using SymbolicRegression
    using DynamicExpressions: Node

    operators = Options(; binary_operators=(+, -, *, /)).operators
    variable_names = ["x1", "x2", "x3"]

    # Create a structure where f only gets access to x1, x2
    # and g only gets access to x3
    structure = TemplateStructure{(:f, :g)}(((; f, g), (x1, x2, x3)) -> f(x1, x2) + g(x3))

    x1 = ComposableExpression(Node{Float64}(; feature=1); operators, variable_names)
    x2 = ComposableExpression(Node{Float64}(; feature=2); operators, variable_names)
    x3 = ComposableExpression(Node{Float64}(; feature=3); operators, variable_names)

    # Test valid case - each function only uses allowed features
    valid_f = x1 + x2
    valid_g = x1
    valid_template = TemplateExpression(
        (; f=valid_f, g=valid_g); structure, operators, variable_names
    )
    @test valid_template([1.0 2.0 3.0]') ≈ [6.0]  # (1 + 2) + 3

    # Test invalid case - f tries to use x3 which it shouldn't have access to
    invalid_f = x1 + x3
    invalid_template = TemplateExpression(
        (; f=invalid_f, g=valid_g); structure, operators, variable_names
    )
    @test invalid_template([1.0 2.0 3.0]') === nothing

    # Test invalid case - g tries to use x2 which it shouldn't have access to
    invalid_g = x2
    invalid_template2 = TemplateExpression(
        (; f=valid_f, g=invalid_g); structure, operators, variable_names
    )
    @test invalid_template2([1.0 2.0 3.0]') === nothing
end
@testitem "Test invalid structure" tags = [:part3] begin
    using SymbolicRegression

    operators = Options(; binary_operators=(+, -, *, /)).operators
    variable_names = ["x1", "x2", "x3"]

    x1 = ComposableExpression(Node{Float64}(; feature=1); operators, variable_names)
    x2 = ComposableExpression(Node{Float64}(; feature=2); operators, variable_names)
    x3 = ComposableExpression(Node{Float64}(; feature=3); operators, variable_names)

    @test_throws ArgumentError TemplateStructure{(:f,)}(
        ((; f), (x1, x2)) -> f(x1) + f(x1, x2)
    )
    @test_throws "Inconsistent number of arguments passed to f" TemplateStructure{(:f,)}(
        ((; f), (x1, x2)) -> f(x1) + f(x1, x2)
    )

    @test_throws ArgumentError TemplateStructure{(:f, :g)}(((; f, g), (x1, x2)) -> f(x1))
    @test_throws "Failed to infer number of features used by (:g,)" TemplateStructure{(
        :f, :g
    )}(
        ((; f, g), (x1, x2)) -> f(x1)
    )
end

@testitem "Test argument-less template structure" tags = [:part2] begin
    using SymbolicRegression
    using DynamicExpressions: OperatorEnum

    operators = OperatorEnum(; binary_operators=(+, *, /, -), unary_operators=(sin, cos))
    variable_names = ["x1", "x2"]
    x1 = ComposableExpression(Node{Float64}(; feature=1); operators, variable_names)
    x2 = ComposableExpression(Node{Float64}(; feature=2); operators, variable_names)
    c1 = ComposableExpression(Node{Float64}(; val=3.0); operators, variable_names)

    # We can evaluate an expression with no arguments:
    @test c1() == 3.0
    @test typeof(c1()) === Float64

    # Create a structure where f takes no arguments and g takes two
    structure = TemplateStructure{(:f, :g)}(((; f, g), (x1, x2)) -> f() + g(x1, x2))

    @test structure.num_features == (; f=0, g=2)

    X = [1.0 2.0]'
    expr = TemplateExpression((; f=c1, g=x1 + x2); structure, operators, variable_names)
    @test expr(X) ≈ [6.0]  # 3 + (1 + 2)
end

@testitem "Test TemplateExpression with differential operator" tags = [:part3] begin
    using SymbolicRegression
    using SymbolicRegression: D
    using DynamicExpressions: OperatorEnum

    operators = OperatorEnum(; binary_operators=(+, -, *, /), unary_operators=(sin, cos))
    variable_names = ["x1", "x2"]
    x1 = ComposableExpression(Node{Float64}(; feature=1); operators, variable_names)
    x2 = ComposableExpression(Node{Float64}(; feature=2); operators, variable_names)
    x3 = ComposableExpression(Node{Float64}(; feature=3); operators, variable_names)

    structure = TemplateStructure{(:f, :g)}(
        ((; f, g), (x1, x2, x3)) -> f(x1) + D(g, 1)(x2, x3)
    )
    expr = TemplateExpression(
        (; f=x1, g=cos(x1 - x2) + 2.5 * x1); structure, operators, variable_names
    )
    # Truth: x1 - sin(x2 - x3) + 2.5
    X = stack(([1.0, 2.0], [3.0, 4.0], [5.0, 6.0]); dims=1)
    @test expr(X) ≈ [1.0, 2.0] .- sin.([3.0, 4.0] .- [5.0, 6.0]) .+ 2.5
end

@testitem "Test literal_pow with ValidVector" tags = [:part2] begin
    using SymbolicRegression: ValidVector

    # Test with valid data
    x = ValidVector([2.0, 3.0, 4.0], true)

    # Test literal_pow with different powers
    @test (x^2).x ≈ [4.0, 9.0, 16.0]
    @test (x^3).x ≈ [8.0, 27.0, 64.0]

    # And explicitly
    @test Base.literal_pow(^, x, Val(2)).x ≈ [4.0, 9.0, 16.0]
    @test Base.literal_pow(^, x, Val(3)).x ≈ [8.0, 27.0, 64.0]

    # Test with invalid data
    invalid_x = ValidVector([2.0, 3.0, 4.0], false)
    @test (invalid_x^2).valid == false
    @test Base.literal_pow(^, invalid_x, Val(2)).valid == false
end

@testitem "Test nan behavior with argument-less expressions" tags = [:part2] begin
    using SymbolicRegression
    using DynamicExpressions: OperatorEnum, Node

    operators = OperatorEnum(; binary_operators=(+, *, /, -), unary_operators=(sin, cos))
    variable_names = ["x1", "x2"]

    # Test with floating point
    c1 = ComposableExpression(Node{Float64}(; val=3.0); operators, variable_names)
    invalid_const = (c1 / c1 - 1) / (c1 / c1 - 1)  # Creates 0/0
    @test isnan(invalid_const())
    @test typeof(invalid_const()) === Float64

    # Test with integer constant
    c2 = ComposableExpression(Node{Int}(; val=0); operators, variable_names)
    @test c2() == 0
    @test typeof(c2()) === Int
end

@testitem "Test higher-order derivatives of safe_log with DynamicDiff" tags = [:part3] begin
    using SymbolicRegression
    using SymbolicRegression: D, safe_log, ValidVector
    using DynamicExpressions: OperatorEnum
    using ForwardDiff: DimensionMismatch

    operators = OperatorEnum(; binary_operators=(+, -, *, /), unary_operators=(safe_log,))
    variable_names = ["x"]
    x = ComposableExpression(Node{Float64}(; feature=1); operators, variable_names)

    # Test first and second derivatives of log(x)
    structure = TemplateStructure{(:f,)}(
        ((; f), (x,)) ->
            ValidVector([(f(x).x[1], D(f, 1)(x).x[1], D(D(f, 1), 1)(x).x[1])], true),
    )
    expr = TemplateExpression((; f=log(x)); structure, operators, variable_names)

    # Test at x = 2.0 where log(x) is well-defined
    X = [2.0]'
    result = only(expr(X))
    @test result !== nothing
    @test result[1] == log(2.0)  # function value
    @test result[2] == 1 / 2.0     # first derivative
    @test result[3] == -1 / 4.0    # second derivative

    # We handle invalid ranges gracefully:
    X_invalid = [-1.0]'
    result = only(expr(X_invalid))
    @test result !== nothing
    @test isnan(result[1])
    @test result[2] == 0.0
    @test result[3] == 0.0

    # Eventually we want to support complex numbers:
    X_complex = [-1.0 - 1.0im]'
    @test_throws DimensionMismatch expr(X_complex)
end
