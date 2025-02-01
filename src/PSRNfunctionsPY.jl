using PythonCall

# 导入PyTorch相关模块
torch = pyimport("torch")
nn = pyimport("torch.nn")
F = pyimport("torch.nn.functional")



# 简化的基类 CanCountLeaveOperator
CanCountLeaveOperator = pytype("CanCountLeaveOperator", (nn.Module,), [
    "__module__" => "__main__",
    pyfunc(
        name = "__init__",
        function (self)
            pybuiltins.super(CanCountLeaveOperator, self).__init__()
            return
        end
    )
])

# Identity类
Identity = pytype("Identity", (CanCountLeaveOperator,), [
    "__module__" => "__main__",
    pyfunc(
        name = "__init__",
        function (self, in_dim, device)
            pybuiltins.super(Identity, self).__init__()
            self.in_dim = in_dim
            self.out_dim = in_dim
            self.is_unary = true
            self.is_directed = true
            self.operator = operators.Identity_op()
            self.complexity = 0
            return
        end
    ),
    pyfunc(
        name = "forward",
        function (self, x, second_device=nothing)
            if second_device !== nothing
                x = x.to(second_device)
            end
            return x
        end
    )
])

# Sin类
Sin = pytype("Sin", (CanCountLeaveOperator,), [
    "__module__" => "__main__",
    pyfunc(
        name = "__init__",
        function (self, in_dim, device)
            pybuiltins.super(Sin, self).__init__()
            self.in_dim = in_dim
            self.out_dim = in_dim
            self.is_unary = true
            self.is_directed = true
            self.operator = operators.Sin_op()
            self.complexity = 4
            return
        end
    ),
    pyfunc(
        name = "forward",
        function (self, x, second_device=nothing)
            if second_device !== nothing
                x = x.to(second_device)
            end
            return torch.sin(x)
        end
    )
])

# Cos类
Cos = pytype("Cos", (CanCountLeaveOperator,), [
    "__module__" => "__main__",
    pyfunc(
        name = "__init__",
        function (self, in_dim, device)
            pybuiltins.super(Cos, self).__init__()
            self.in_dim = in_dim
            self.out_dim = in_dim
            self.is_unary = true
            self.is_directed = true
            self.operator = operators.Cos_op()
            self.complexity = 4
            return
        end
    ),
    pyfunc(
        name = "forward",
        function (self, x, second_device=nothing)
            if second_device !== nothing
                x = x.to(second_device)
            end
            return torch.cos(x)
        end
    )
])

# Exp类
Exp = pytype("Exp", (CanCountLeaveOperator,), [
    "__module__" => "__main__",
    pyfunc(
        name = "__init__",
        function (self, in_dim, device, threshold=10)
            pybuiltins.super(Exp, self).__init__()
            self.threshold = threshold
            self.in_dim = in_dim
            self.out_dim = in_dim
            self.is_unary = true
            self.is_directed = true
            self.operator = operators.Exp_op()
            self.complexity = 4
            return
        end
    ),
    pyfunc(
        name = "forward",
        function (self, x, second_device=nothing)
            if second_device !== nothing
                x = x.to(second_device)
            end
            return torch.exp(x)
        end
    )
])

# Log类
Log = pytype("Log", (CanCountLeaveOperator,), [
    "__module__" => "__main__",
    pyfunc(
        name = "__init__",
        function (self, in_dim, device, threshold=1e-10)
            pybuiltins.super(Log, self).__init__()
            self.threshold = threshold
            self.in_dim = in_dim
            self.out_dim = in_dim
            self.is_unary = true
            self.is_directed = true
            self.complexity = 4
            self.operator = operators.Log_op()
            return
        end
    ),
    pyfunc(
        name = "forward",
        function (self, x, second_device=nothing)
            if second_device !== nothing
                x = x.to(second_device)
            end
            log = torch.log(x)
            return log
        end
    )
])

# Mul类
Mul = pytype("Mul", (CanCountLeaveOperator,), [
    "__module__" => "__main__",
    pyfunc(
        name = "__init__",
        function (self, in_dim, device)
            pybuiltins.super(Mul, self).__init__()
            self.in_dim = in_dim
            self.out_dim = in_dim * (in_dim + 1) ÷ 2
            self.is_unary = false
            self.is_directed = false
            self.complexity = 1
            self.operator = operators.Mul_op()
            self.device = device
            return
        end
    ),
    pyfunc(
        name = "forward",
        function (self, x, second_device=nothing)
            if second_device !== nothing
                x = x.to(second_device)
            end
            indices = torch.triu_indices(
                self.in_dim, self.in_dim, offset=0, dtype=torch.int32, device=x.device
            )
            out = x[:, indices[1]] * x[:, indices[2]]
            return out
        end
    )
])

# Add类
Add = pytype("Add", (CanCountLeaveOperator,), [
    "__module__" => "__main__",
    pyfunc(
        name = "__init__",
        function (self, in_dim, device)
            pybuiltins.super(Add, self).__init__()
            self.in_dim = in_dim
            self.out_dim = in_dim * (in_dim + 1) ÷ 2
            self.is_unary = false
            self.is_directed = false
            self.complexity = 1
            self.operator = operators.Add_op()
            self.device = device
            return
        end
    ),
    pyfunc(
        name = "forward",
        function (self, x, second_device=nothing)
            if second_device !== nothing
                x = x.to(second_device)
            end
            indices = torch.triu_indices(
                self.in_dim, self.in_dim, offset=0, dtype=torch.int32, device=x.device
            )
            out = x[:, indices[1]] + x[:, indices[2]]
            return out
        end
    )
])

# Pow类
Pow = pytype("Pow", (CanCountLeaveOperator,), [
    "__module__" => "__main__",
    pyfunc(
        name = "__init__",
        function (self, in_dim, device)
            pybuiltins.super(Pow, self).__init__()
            self.in_dim = in_dim
            self.out_dim = in_dim * in_dim
            self.is_unary = false
            self.is_directed = true
            self.complexity = 4
            self.operator = operators.Pow_op()
            return
        end
    ),
    pyfunc(
        name = "forward",
        function (self, x, second_device=nothing)
            if second_device !== nothing
                x = x.to(second_device)
            end
            num = x.view(1, -1, 1)
            deno = x.view(1, 1, -1)
            out = num^deno
            out = out.view(1, -1)
            return out
        end
    )
])

# Sigmoid类
Sigmoid = pytype("Sigmoid", (CanCountLeaveOperator,), [
    "__module__" => "__main__",
    pyfunc(
        name = "__init__",
        function (self, in_dim, device)
            pybuiltins.super(Sigmoid, self).__init__()
            self.in_dim = in_dim
            self.out_dim = in_dim
            self.is_unary = true
            self.is_directed = true
            self.complexity = 6
            self.operator = operators.Sigmoid_op()
            return
        end
    ),
    pyfunc(
        name = "forward",
        function (self, x, second_device=nothing)
            if second_device !== nothing
                x = x.to(second_device)
            end
            return 1 / (1 + torch.exp(-x))
        end
    )
])


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# x = torch.randint(-1, 5, (3, 5))
# println(x)

# layer = Mul(5, device)
# out = layer(x)
# println(out.shape)
# println(out)
