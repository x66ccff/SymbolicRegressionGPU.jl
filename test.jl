using DynamicExpressions
using SymbolicRegression

"""
通用的表达式解析器 - 支持SR.jl的所有运算符
"""
struct ExpressionParser
    binary_operators::Vector
    unary_operators::Vector
    operator_symbols::Dict{String, Any}
    function_names::Vector{String}
    
    function ExpressionParser(options::SymbolicRegression.AbstractOptions)
        binary_ops = collect(options.operators.ops[2])
        unary_ops = collect(options.operators.ops[1])
        
        # 自动构建运算符符号映射
        symbols = build_operator_symbol_map(binary_ops, unary_ops)
        
        # 提取所有函数名用于tokenizer
        func_names = collect(keys(filter(p -> !is_binary_symbol(p.first), symbols)))
        
        new(binary_ops, unary_ops, symbols, func_names)
    end
end

"""
自动构建运算符符号到函数的映射
"""
function build_operator_symbol_map(binary_operators, unary_operators)
    symbols = Dict{String, Any}()
    
    # 标准二元运算符映射
    standard_binary = Dict(
        "+" => (+),
        "-" => (-), 
        "*" => (*),
        "/" => (/),
        "^" => (^),
        "%" => (%),
        "&" => (&),
        "|" => (|),
        ">" => (>),
        "<" => (<),
        "==" => (==),
        "!=" => (!=),
        ">=" => (>=),
        "<=" => (<=)
    )
    
    # 添加找到的二元运算符
    for op in binary_operators
        for (symbol, func) in standard_binary
            if op === func
                symbols[symbol] = op
            end
        end
        
        # 特殊处理safe_pow -> ^
        op_str = string(op)
        if contains(op_str, "safe_pow")
            symbols["^"] = op
        end
    end
    
    # 自动检测一元运算符
    for op in unary_operators
        name = get_function_name(op)
        if !isnothing(name)
            symbols[name] = op
        end
    end
    
    return symbols
end

"""
获取函数的名称字符串 - 改进版本
"""
function get_function_name(func)
    func_str = string(func)
    
    # 特殊处理SymbolicRegression的safe函数
    if contains(func_str, "safe_log")
        return "log"
    elseif contains(func_str, "safe_sqrt")
        return "sqrt"
    elseif contains(func_str, "safe_pow")
        return "pow"
    elseif contains(func_str, "safe_exp")
        return "exp"
    end
    
    # 标准函数映射
    known_functions = Dict(
        sin => "sin",
        cos => "cos", 
        tan => "tan",
        exp => "exp",
        log => "log",
        sqrt => "sqrt",
        abs => "abs",
        floor => "floor",
        ceil => "ceil",
        round => "round",
        tanh => "tanh",
        sinh => "sinh",
        cosh => "cosh",
        atan => "atan",
        asin => "asin",
        acos => "acos"
    )
    
    # 直接匹配已知函数
    if haskey(known_functions, func)
        return known_functions[func]
    end
    
    # 正则表达式匹配函数名
    patterns = [
        r"^([a-zA-Z_][a-zA-Z0-9_]*)",
        r"\.([a-zA-Z_][a-zA-Z0-9_]*)$"
    ]
    
    for pattern in patterns
        m = match(pattern, func_str)
        if !isnothing(m) && length(m.captures) > 0
            name = m.captures[1]
            if name ∉ ["typeof", "Module", "Function", "Core", "Base", "Main"]
                return name
            end
        end
    end
    
    return nothing
end

"""
判断是否为二元运算符符号
"""
function is_binary_symbol(symbol::String)
    return symbol in ["+", "-", "*", "/", "^", "%", "&", "|", ">", "<", "==", "!=", ">=", "<="]
end

"""
改进的tokenizer - 修复括号和函数识别问题
"""
function tokenize_expression(expr_str::String, parser::ExpressionParser)
    tokens = String[]
    i = 1
    n = length(expr_str)
    
    while i <= n
        c = expr_str[i]
        
        # 跳过空白
        if isspace(c)
            i += 1
            continue
        end
        
        # 括号
        if c in ['(', ')']
            push!(tokens, string(c))
            i += 1
        # 多字符运算符（如 ==, !=, >=, <=）
        elseif i + 1 <= n
            two_char = expr_str[i:i+1]
            if two_char in keys(parser.operator_symbols)
                push!(tokens, two_char)
                i += 2
            # 单字符运算符
            elseif string(c) in keys(parser.operator_symbols)
                push!(tokens, string(c))
                i += 1
            # 字母开头的token（函数名或变量）
            elseif isletter(c) || c == '_'
                j = i
                while j <= n && (isletter(expr_str[j]) || isdigit(expr_str[j]) || expr_str[j] == '_')
                    j += 1
                end
                token = expr_str[i:j-1]
                push!(tokens, token)
                i = j
            # 数字
            elseif isdigit(c) || c == '.'
                j = i
                while j <= n && (isdigit(expr_str[j]) || expr_str[j] == '.')
                    j += 1
                end
                push!(tokens, expr_str[i:j-1])
                i = j
            else
                i += 1
            end
        # 单字符处理
        elseif string(c) in keys(parser.operator_symbols)
            push!(tokens, string(c))
            i += 1
        elseif isletter(c) || c == '_'
            j = i
            while j <= n && (isletter(expr_str[j]) || isdigit(expr_str[j]) || expr_str[j] == '_')
                j += 1
            end
            token = expr_str[i:j-1]
            push!(tokens, token)
            i = j
        elseif isdigit(c) || c == '.'
            j = i
            while j <= n && (isdigit(expr_str[j]) || expr_str[j] == '.')
                j += 1
            end
            push!(tokens, expr_str[i:j-1])
            i = j
        else
            i += 1
        end
    end
    
    return tokens
end

"""
通用表达式解析器 - 主函数
"""
function convert_python_tree_to_nodes(expr_str::String, options::SymbolicRegression.AbstractOptions)
    expr_str = String(strip(expr_str))
    
    if isempty(expr_str) || contains(expr_str, "no data")
        return nothing
    end
    
    try
        parser = ExpressionParser(options)
        tokens = tokenize_expression(expr_str, parser)
        
        if isempty(tokens)
            return nothing
        end
        
        node, remaining_tokens = parse_expression_with_parser(tokens, parser)
        if !isempty(remaining_tokens)
            @warn "Unexpected remaining tokens: $remaining_tokens"
        end
        return node
    catch e
        @warn "Failed to parse expression: $expr_str" exception=e
        return nothing
    end
end

"""
使用解析器进行表达式解析
"""
function parse_expression_with_parser(tokens::Vector{String}, parser::ExpressionParser)
    if isempty(tokens)
        error("Empty token list")
    end
    return parse_additive_with_parser(tokens, parser)
end

"""
解析加法减法
"""
function parse_additive_with_parser(tokens::Vector{String}, parser::ExpressionParser)
    node, tokens = parse_multiplicative_with_parser(tokens, parser)
    
    while !isempty(tokens) && tokens[1] in ["+", "-"]
        op = popfirst!(tokens)
        right_node, tokens = parse_multiplicative_with_parser(tokens, parser)
        node = create_binary_node_with_parser(op, node, right_node, parser)
    end
    
    return node, tokens
end

"""
解析乘法除法
"""
function parse_multiplicative_with_parser(tokens::Vector{String}, parser::ExpressionParser)
    node, tokens = parse_power_with_parser(tokens, parser)
    
    while !isempty(tokens) && tokens[1] in ["*", "/", "%"]
        op = popfirst!(tokens)
        right_node, tokens = parse_power_with_parser(tokens, parser)
        node = create_binary_node_with_parser(op, node, right_node, parser)
    end
    
    return node, tokens
end

"""
解析幂运算
"""
function parse_power_with_parser(tokens::Vector{String}, parser::ExpressionParser)
    node, tokens = parse_unary_with_parser(tokens, parser)
    
    # 幂运算是右结合的
    if !isempty(tokens) && tokens[1] == "^"
        op = popfirst!(tokens)
        right_node, tokens = parse_power_with_parser(tokens, parser)  # 递归处理右结合
        node = create_binary_node_with_parser(op, node, right_node, parser)
    end
    
    return node, tokens
end

"""
解析一元运算符 - 修复函数识别
"""
function parse_unary_with_parser(tokens::Vector{String}, parser::ExpressionParser)
    if isempty(tokens)
        error("Unexpected end of expression")
    end
    
    token = tokens[1]
    
    # 检查是否为函数名（在parser的function_names中或直接在operator_symbols中）
    if token in parser.function_names || (haskey(parser.operator_symbols, token) && !is_binary_symbol(token))
        popfirst!(tokens)
        
        if isempty(tokens) || tokens[1] != "("
            error("Expected '(' after function $token")
        end
        popfirst!(tokens)
        
        arg_node, tokens = parse_additive_with_parser(tokens, parser)
        
        if isempty(tokens) || tokens[1] != ")"
            error("Expected ')' after function argument")
        end
        popfirst!(tokens)
        
        return create_unary_node_with_parser(token, arg_node, parser), tokens
    else
        return parse_primary_with_parser(tokens, parser)
    end
end

"""
解析原子表达式
"""
function parse_primary_with_parser(tokens::Vector{String}, parser::ExpressionParser)
    if isempty(tokens)
        error("Unexpected end of expression")
    end
    
    token = tokens[1]
    
    if token == "("
        popfirst!(tokens)
        node, tokens = parse_additive_with_parser(tokens, parser)
        
        if isempty(tokens) || tokens[1] != ")"
            error("Expected ')' to close parentheses")
        end
        popfirst!(tokens)
        
        return node, tokens
    elseif startswith(token, "x_")
        popfirst!(tokens)
        var_index_str = token[3:end]
        var_index = parse(Int, var_index_str)
        return create_variable_node(var_index), tokens
    else
        popfirst!(tokens)
        try
            val = parse(Float64, token)
            return create_constant_node(val), tokens
        catch
            error("Cannot parse token as number: $token")
        end
    end
end

"""
使用解析器创建二元节点
"""
function create_binary_node_with_parser(op::String, left::Node, right::Node, parser::ExpressionParser)
    if !haskey(parser.operator_symbols, op)
        error("Unknown operator: $op")
    end
    
    target_func = parser.operator_symbols[op]
    
    for (i, binary_op) in enumerate(parser.binary_operators)
        if binary_op === target_func
            return Node(i, left, right)
        end
    end
    
    error("Operator $op not found in binary operators list")
end

"""
使用解析器创建一元节点
"""
function create_unary_node_with_parser(op::String, arg::Node, parser::ExpressionParser)
    if !haskey(parser.operator_symbols, op)
        error("Unknown operator: $op")
    end
    
    target_func = parser.operator_symbols[op]
    
    for (i, unary_op) in enumerate(parser.unary_operators)
        if unary_op === target_func
            return Node(i, arg)
        end
    end
    
    error("Operator $op not found in unary operators list")
end

# 保留原有的辅助函数
function create_variable_node(feature_index::Int)
    return Node(Float32; feature=feature_index + 1)
end

function create_constant_node(value::Float64)
    return Node(Float32; val=Float32(value))
end

function convert_python_expressions_to_nodes(expr_strings::Vector{String}, options::SymbolicRegression.AbstractOptions)
    nodes = Node[]
    for expr_str in expr_strings
        node = convert_python_tree_to_nodes(expr_str, options)
        if !isnothing(node)
            push!(nodes, node)
        end
    end
    return nodes
end

"""
测试函数 - 支持任意运算符配置
"""
function test_conversion(options::SymbolicRegression.AbstractOptions)
    # 创建解析器并显示支持的运算符
    parser = ExpressionParser(options)
    
    println("Testing with operators:")
    println("  Binary: ", parser.binary_operators)
    println("  Unary:  ", parser.unary_operators)
    println("  Detected symbols: ", sort(collect(keys(parser.operator_symbols))))
    println()
    
    test_exprs = [
        "(((x_1)-(x_4))*((x_0)/(x_1)))-(((x_1)/(x_1))-(exp(x_0)))",
        "sin(x_0)",
        "((x_1)+(x_2))",
        "exp(((x_0)*(x_1)))",
        "log(x_3)",
        "sqrt(x_0)",  # 测试是否支持sqrt
        "x_0^x_1",    # 测试幂运算
        "no data"
    ]
    
    for expr in test_exprs
        println("Testing: $expr")
        node = convert_python_tree_to_nodes(expr, options)
        if isnothing(node)
            println("  Result: nothing")
        else
            println("  Result: Node created successfully")
            println("  Node: $node")
            
            # 创建Expression来验证结果
            try
                expression = Expression(
                    node;
                    operators=options.operators,
                    variable_names=["x0", "x1", "x2", "x3", "x4"]
                )
                println("  Expression: $expression")
            catch e
                println("  Expression creation failed: $e")
            end
        end
        println()
    end
end

"""
使用示例 - 支持更多运算符
"""
function example_usage()
    # 测试基本运算符
    options1 = SymbolicRegression.Options(
        binary_operators=[+, -, *, /],
        unary_operators=[sin, cos, exp, log],
        maxsize=30
    )
    
    println("=== Testing basic operators ===")
    test_conversion(options1)
    
    # 测试扩展运算符（如果支持的话）
    try
        options2 = SymbolicRegression.Options(
            binary_operators=[+, -, *, /, ^],
            unary_operators=[sin, cos, exp, log, sqrt, abs, tanh],
            maxsize=30
        )
        
        println("\n=== Testing extended operators ===")
        test_conversion(options2)
    catch e
        println("Extended operators not available: $e")
    end
end

# 运行示例
example_usage()