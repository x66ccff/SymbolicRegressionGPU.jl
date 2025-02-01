using PythonCall

# 导入PyTorch相关模块
torch = pyimport("torch")
nn = pyimport("torch.nn")
F = pyimport("torch.nn.functional")

# 一元运算符类

# Identity_op
Identity_op = pytype("Identity_op", (), [
    "__module__" => "__main__",
    pyfunc(
        name = "__init__",
        function (self)
            pybuiltins.super(Identity_op, self).__init__()
            self.is_unary = true
            return
        end
    ),
    pyfunc(
        name = "get_expr",
        (self, sub_expr) -> sub_expr
    ),
    pyfunc(
        name = "transform_inputs",
        (self, x) -> x
    )
])

# Sin_op
Sin_op = pytype("Sin_op", (), [
    "__module__" => "__main__",
    pyfunc(
        name = "__init__",
        function (self)
            pybuiltins.super(Sin_op, self).__init__()
            self.is_unary = true
            return
        end
    ),
    pyfunc(
        name = "get_expr",
        (self, sub_expr) -> "sin($sub_expr)"
    ),
    pyfunc(
        name = "transform_inputs",
        (self, x) -> torch.sin(x)
    )
])

# Cos_op
Cos_op = pytype("Cos_op", (), [
    "__module__" => "__main__",
    pyfunc(
        name = "__init__",
        function (self)
            pybuiltins.super(Cos_op, self).__init__()
            self.is_unary = true
            return
        end
    ),
    pyfunc(
        name = "get_expr",
        (self, sub_expr) -> "cos($sub_expr)"
    ),
    pyfunc(
        name = "transform_inputs",
        (self, x) -> torch.cos(x)
    )
])

# Exp_op
Exp_op = pytype("Exp_op", (), [
    "__module__" => "__main__",
    pyfunc(
        name = "__init__",
        function (self, threshold=10)
            pybuiltins.super(Exp_op, self).__init__()
            self.is_unary = true
            self.threshold = threshold
            return
        end
    ),
    pyfunc(
        name = "get_expr",
        (self, sub_expr) -> "exp($sub_expr)"
    ),
    pyfunc(
        name = "transform_inputs",
        (self, x) -> torch.exp(torch.clamp(x, max=self.threshold))
    )
])

# Log_op
Log_op = pytype("Log_op", (), [
    "__module__" => "__main__",
    pyfunc(
        name = "__init__",
        function (self, threshold=1e-10)
            pybuiltins.super(Log_op, self).__init__()
            self.is_unary = true
            self.threshold = threshold
            return
        end
    ),
    pyfunc(
        name = "get_expr",
        (self, sub_expr) -> "log($sub_expr)"
    ),
    pyfunc(
        name = "transform_inputs",
        (self, x) -> torch.log(x)
    )
])

# Neg_op
Neg_op = pytype("Neg_op", (), [
    "__module__" => "__main__",
    pyfunc(
        name = "__init__",
        function (self)
            pybuiltins.super(Neg_op, self).__init__()
            self.is_unary = true
            return
        end
    ),
    pyfunc(
        name = "get_expr",
        (self, sub_expr) -> "(-($sub_expr))"
    ),
    pyfunc(
        name = "transform_inputs",
        (self, x) -> -x
    )
])

# Inv_op
Inv_op = pytype("Inv_op", (), [
    "__module__" => "__main__",
    pyfunc(
        name = "__init__",
        function (self, threshold=1e-10)
            pybuiltins.super(Inv_op, self).__init__()
            self.is_unary = true
            self.threshold = threshold
            return
        end
    ),
    pyfunc(
        name = "get_expr",
        (self, sub_expr) -> "(1/($sub_expr))"
    ),
    pyfunc(
        name = "transform_inputs",
        function (self, x)
            x = torch.where(
                x < 0,
                torch.clamp(x, max=-self.threshold),
                torch.clamp(x, min=self.threshold)
            )
            return 1 / x
        end
    )
])

# 二元运算符类

# Mul_op
Mul_op = pytype("Mul_op", (), [
    "__module__" => "__main__",
    pyfunc(
        name = "__init__",
        function (self)
            pybuiltins.super(Mul_op, self).__init__()
            self.is_unary = false
            self.is_directed = false
            return
        end
    ),
    pyfunc(
        name = "get_expr",
        (self, sub_expr1, sub_expr2) -> "($sub_expr1)*($sub_expr2)"
    ),
    pyfunc(
        name = "transform_inputs",
        (self, x1, x2) -> x1 * x2
    )
])

# Add_op
Add_op = pytype("Add_op", (), [
    "__module__" => "__main__",
    pyfunc(
        name = "__init__",
        function (self)
            pybuiltins.super(Add_op, self).__init__()
            self.is_unary = false
            self.is_directed = false
            return
        end
    ),
    pyfunc(
        name = "get_expr",
        (self, sub_expr1, sub_expr2) -> "($sub_expr1)+($sub_expr2)"
    ),
    pyfunc(
        name = "transform_inputs",
        (self, x1, x2) -> x1 + x2
    )
])

# Div_op
Div_op = pytype("Div_op", (), [
    "__module__" => "__main__",
    pyfunc(
        name = "__init__",
        function (self, threshold=1e-10)
            pybuiltins.super(Div_op, self).__init__()
            self.is_unary = false
            self.is_directed = true
            self.threshold = threshold
            return
        end
    ),
    pyfunc(
        name = "get_expr",
        (self, sub_expr1, sub_expr2) -> "($sub_expr1)/($sub_expr2)"
    ),
    pyfunc(
        name = "transform_inputs",
        function (self, x1, x2)
            deno = torch.where(
                x2 < 0,
                torch.clamp(x2, max=-self.threshold),
                torch.clamp(x2, min=self.threshold)
            )
            num = x1
            return num / deno
        end
    )
])

# Sub_op
Sub_op = pytype("Sub_op", (), [
    "__module__" => "__main__",
    pyfunc(
        name = "__init__",
        function (self)
            pybuiltins.super(Sub_op, self).__init__()
            self.is_unary = false
            self.is_directed = true
            return
        end
    ),
    pyfunc(
        name = "get_expr",
        (self, sub_expr1, sub_expr2) -> "($sub_expr1)-($sub_expr2)"
    ),
    pyfunc(
        name = "transform_inputs",
        (self, x1, x2) -> x1 - x2
    )
])

# SemiDiv_op
SemiDiv_op = pytype("SemiDiv_op", (), [
    "__module__" => "__main__",
    pyfunc(
        name = "__init__",
        function (self, threshold=1e-10)
            pybuiltins.super(SemiDiv_op, self).__init__()
            self.is_unary = false
            self.is_directed = false
            self.threshold = threshold
            return
        end
    ),
    pyfunc(
        name = "get_expr",
        (self, sub_expr1, sub_expr2) -> "($sub_expr1)/($sub_expr2)"
    ),
    pyfunc(
        name = "transform_inputs",
        function (self, x1, x2)
            deno = torch.where(
                x2 < 0,
                torch.clamp(x2, max=-self.threshold),
                torch.clamp(x2, min=self.threshold)
            )
            num = x1
            return num / deno
        end
    )
])

# SemiSub_op
SemiSub_op = pytype("SemiSub_op", (), [
    "__module__" => "__main__",
    pyfunc(
        name = "__init__",
        function (self)
            pybuiltins.super(SemiSub_op, self).__init__()
            self.is_unary = false
            self.is_directed = false
            return
        end
    ),
    pyfunc(
        name = "get_expr",
        (self, sub_expr1, sub_expr2) -> "($sub_expr1)-($sub_expr2)"
    ),
    pyfunc(
        name = "transform_inputs",
        (self, x1, x2) -> x1 - x2
    )
])

# Sign_op
Sign_op = pytype("Sign_op", (), [
    "__module__" => "__main__",
    pyfunc(
        name = "__init__",
        function (self)
            pybuiltins.super(Sign_op, self).__init__()
            self.is_unary = true
            self.is_directed = true
            return
        end
    ),
    pyfunc(
        name = "get_expr",
        (self, sub_expr) -> "(sign($sub_expr))"
    ),
    pyfunc(
        name = "transform_inputs",
        (self, x) -> torch.sign(x)
    )
])

# Pow2_op
Pow2_op = pytype("Pow2_op", (), [
    "__module__" => "__main__",
    pyfunc(
        name = "__init__",
        function (self)
            pybuiltins.super(Pow2_op, self).__init__()
            self.is_unary = true
            self.is_directed = true
            return
        end
    ),
    pyfunc(
        name = "get_expr",
        (self, sub_expr) -> "(($sub_expr)**2)"
    ),
    pyfunc(
        name = "transform_inputs",
        (self, x) -> x^2
    )
])

# Pow3_op
Pow3_op = pytype("Pow3_op", (), [
    "__module__" => "__main__",
    pyfunc(
        name = "__init__",
        function (self)
            pybuiltins.super(Pow3_op, self).__init__()
            self.is_unary = true
            self.is_directed = true
            return
        end
    ),
    pyfunc(
        name = "get_expr",
        (self, sub_expr) -> "(($sub_expr)**3)"
    ),
    pyfunc(
        name = "transform_inputs",
        (self, x) -> x^3
    )
])

# Pow_op
Pow_op = pytype("Pow_op", (), [
    "__module__" => "__main__",
    pyfunc(
        name = "__init__",
        function (self)
            pybuiltins.super(Pow_op, self).__init__()
            self.is_unary = false
            self.is_directed = true
            return
        end
    ),
    pyfunc(
        name = "get_expr",
        (self, sub_expr1, sub_expr2) -> "(($sub_expr1)**($sub_expr2))"
    ),
    pyfunc(
        name = "transform_inputs",
        (self, x1, x2) -> x1^x2
    )
])

# Sigmoid_op
Sigmoid_op = pytype("Sigmoid_op", (), [
    "__module__" => "__main__",
    pyfunc(
        name = "__init__",
        function (self)
            pybuiltins.super(Sigmoid_op, self).__init__()
            self.is_unary = true
            self.is_directed = true
            return
        end
    ),
    pyfunc(
        name = "get_expr",
        (self, sub_expr) -> "(1/(1+exp(-($sub_expr))))"
    ),
    pyfunc(
        name = "transform_inputs",
        (self, x) -> 1 / (1 + torch.exp(-x))
    )
])

# Abs_op
Abs_op = pytype("Abs_op", (), [
    "__module__" => "__main__",
    pyfunc(
        name = "__init__",
        function (self)
            pybuiltins.super(Abs_op, self).__init__()
            self.is_unary = true
            self.is_directed = true
            return
        end
    ),
    pyfunc(
        name = "get_expr",
        (self, sub_expr) -> "Abs($sub_expr)"
    ),
    pyfunc(
        name = "transform_inputs",
        (self, x) -> torch.abs(x)
    )
])

# Cosh_op
Cosh_op = pytype("Cosh_op", (), [
    "__module__" => "__main__",
    pyfunc(
        name = "__init__",
        function (self)
            pybuiltins.super(Cosh_op, self).__init__()
            self.is_unary = true
            self.is_directed = true
            return
        end
    ),
    pyfunc(
        name = "get_expr",
        (self, sub_expr) -> "cosh($sub_expr)"
    ),
    pyfunc(
        name = "transform_inputs",
        (self, x) -> torch.cosh(x)
    )
])

# Tanh_op
Tanh_op = pytype("Tanh_op", (), [
    "__module__" => "__main__",
    pyfunc(
        name = "__init__",
        function (self)
            pybuiltins.super(Tanh_op, self).__init__()
            self.is_unary = true
            self.is_directed = true
            return
        end
    ),
    pyfunc(
        name = "get_expr",
        (self, sub_expr) -> "tanh($sub_expr)"
    ),
    pyfunc(
        name = "transform_inputs",
        (self, x) -> torch.tanh(x)
    )
])

# Sqrt_op
Sqrt_op = pytype("Sqrt_op", (), [
    "__module__" => "__main__",
    pyfunc(
        name = "__init__",
        function (self)
            pybuiltins.super(Sqrt_op, self).__init__()
            self.is_unary = true
            self.is_directed = true
            return
        end
    ),
    pyfunc(
        name = "get_expr",
        (self, sub_expr) -> "($sub_expr)**0.5"
    ),
    pyfunc(
        name = "transform_inputs",
        (self, x) -> torch.sqrt(x)
    )
])


# # 创建实例
# identity_op = Identity_op()
# sin_op = Sin_op()
# exp_op = Exp_op(10.0)
# mul_op = Mul_op()

# # 使用示例
# x = torch.randn(3)
# y = torch.randn(3)

# # 一元运算
# result1 = sin_op.transform_inputs(x)
# result2 = exp_op.transform_inputs(x)

# # 二元运算
# result3 = mul_op.transform_inputs(x, y)

# # 获取表达式
# expr1 = sin_op.get_expr("x")  # 返回 "sin(x)"
# expr2 = mul_op.get_expr("x", "y")  # 返回 "(x)*(y)"