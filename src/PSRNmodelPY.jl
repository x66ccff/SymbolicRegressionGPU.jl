# Duplicate Removal Layer
DRLayer = pytype("DRLayer", (nn.Module,), [
    "__module__" => "__main__",
    pyfunc(
        name = "__init__",
        function (self, in_dim, dr_mask, device=nothing)
            pybuiltins.super(DRLayer, self).__init__()
            
            self.in_dim = in_dim
            self.out_dim = round(torch.sum(dr_mask).item())
            arange_tensor = torch.arange(length(dr_mask), device=device)
            self.dr_indices = arange_tensor[dr_mask]  # (n,)
            self.dr_mask = dr_mask  # (n,)
            
            self.dr_indices = self.dr_indices.to(device)
            self.dr_mask = self.dr_mask.to(device)
            return
        end
    ),
    pyfunc(
        name = "forward",
        function (self, x)
            return x[:, self.dr_mask]
        end
    ),
    pyfunc(
        name = "get_op_and_offset",
        function (self, index)
            return self.dr_indices[index].item()
        end
    )
])

# Symbol Layer
SymbolLayer = pytype("SymbolLayer", (nn.Module,), [
    "__module__" => "__main__",
    pyfunc(
        name = "__init__",
        function (self, in_dim, operators=["Add", "Mul", "Identity", "Sin", "Exp", "Neg", "Inv"], device=nothing)
            pybuiltins.super(SymbolLayer, self).__init__()
            
            self.device = device
            self.in_dim = in_dim
            self.n_triu = in_dim * (in_dim + 1) ÷ 2
            self.in_dim_square = in_dim * in_dim
            self.operators = operators
            
            self.list = nn.ModuleList()
            self.n_binary_U = 0  # undirected * +
            self.n_binary_D = 0  # directed   / -
            self.n_unary = 0
            
            # 计算各类运算符数量
            for op in operators
                func = eval(op)(in_dim, device)
                if !func.is_unary
                    if func.is_directed
                        self.n_binary_D += 1
                    else
                        self.n_binary_U += 1
                    end
                else
                    self.n_unary += 1
                end
            end
            
            # 按顺序添加运算符
            # 先添加Add和Mul (三角形状)
            for op in operators
                func = eval(op)(in_dim, device)
                if !func.is_unary && !func.is_directed
                    self.list.append(func)
                end
            end
            
            # 再添加Sub和Div (方形)
            for op in operators
                func = eval(op)(in_dim, device)
                if !func.is_unary && func.is_directed
                    self.list.append(func)
                end
            end
            
            # 最后添加一元运算符
            for op in operators
                func = eval(op)(in_dim, device)
                if func.is_unary
                    self.list.append(func)
                end
            end
            
            self.out_dim = self.n_unary * self.in_dim + 
                          self.n_binary_U * self.n_triu + 
                          self.n_binary_D * self.in_dim_square
            
            self.out_dim_cum_ls = nothing
            self.init_offset(device)
            return
        end
    ),
    pyfunc(
        name = "forward",
        function (self, x)
            h = []
            for md in self.list
                push!(h, md(x))
            end
            h = torch.cat(h, dim=2)
            return h
        end
    ),
    pyfunc(
        name = "init_offset",
        function (self, device)
            self.offset_tensor = self.get_offset_tensor(device)
        end
    ),
    pyfunc(
        name = "get_offset_tensor",
        function (self, device)
            offset_tensor = torch.zeros((self.out_dim, 2), dtype=torch.int, device=device)
            arange_tensor = torch.arange(self.in_dim, dtype=torch.int, device=device)
            
            binary_U_tensor = torch.zeros((self.n_triu, 2), dtype=torch.int, device=device)
            binary_D_tensor = torch.zeros((self.in_dim_square, 2), dtype=torch.int, device=device)
            unary_tensor = torch.zeros((self.in_dim, 2), dtype=torch.int, device=device)
            
            unary_tensor[:, 1] = arange_tensor
            unary_tensor[:, 2] = self.in_dim
            
            start = 1
            for i in 1:self.in_dim
                len_ = self.in_dim - i + 1
                binary_U_tensor[start:start+len_-1, 1] .= i
                binary_U_tensor[start:start+len_-1, 2] = arange_tensor[i:end]
                start += len_
            end
            
            start = 1
            for i in 1:self.in_dim
                len_ = self.in_dim
                binary_D_tensor[start:start+len_-1, 1] .= i
                binary_D_tensor[start:start+len_-1, 2] = arange_tensor[1:end]
                start += len_
            end
            
            start = 1
            for func in self.list
                if !func.is_unary
                    if func.is_directed
                        t = binary_D_tensor
                    else
                        t = binary_U_tensor
                    end
                else
                    t = unary_tensor
                end
                len_ = size(t, 1)
                offset_tensor[start:start+len_-1, :] = t
                start += len_
            end
            
            return offset_tensor
        end
    ),
    pyfunc(
        name = "get_out_dim_cum_ls",
        function (self)
            if self.out_dim_cum_ls !== nothing
                return self.out_dim_cum_ls
            end
            
            out_dim_ls = []
            for func in self.list
                if !func.is_unary
                    if func.is_directed
                        push!(out_dim_ls, self.in_dim_square)
                    else
                        push!(out_dim_ls, self.n_triu)
                    end
                else
                    push!(out_dim_ls, self.in_dim)
                end
            end
            self.out_dim_cum_ls = [sum(out_dim_ls[1:i]) for i in 1:length(out_dim_ls)]
            return self.out_dim_cum_ls
        end
    ),
    pyfunc(
        name = "get_op_and_offset",
        function (self, index)
            out_dim_cum_ls = self.get_out_dim_cum_ls()
            i = 1
            for (idx, val) in enumerate(out_dim_cum_ls)
                if index < val
                    i = idx
                    break
                end
            end
            func = self.list[i]
            offset = self.offset_tensor[index+1].tolist()
            return func.operator, offset
        end
    )
])

# PSRN类
PSRN = pytype("PSRN", (nn.Module,), [
    "__module__" => "__main__",
    pyfunc(
        name = "__init__",
        function (self, n_variables=1, 
                operators=["Add", "Mul", "Identity", "Sin", "Exp", "Neg", "Inv"],
                n_symbol_layers=3,
                dr_mask=pybuiltins.None,  # 默认值改为pybuiltins.None
                device="cuda")
            pybuiltins.super(PSRN, self).__init__()
            
            # 设置设备
            if pyisinstance(device, pybuiltins.str)
                if pyeq(device, "cuda")
                    self.device = torch.device("cuda")
                elseif pyeq(device, "cpu")
                    self.device = torch.device("cpu")
                else
                    error("device must be cuda or cpu, got $(device)")
                end
            else
                self.device = device
            end
            
            self.n_variables = n_variables
            self.operators = operators
            self.n_symbol_layers = n_symbol_layers
            
            self.list = nn.ModuleList()
            
            # 处理dr_mask
            if pyis(dr_mask, pybuiltins.None)  # 使用pyis替代===
                self.use_dr_mask = pybool(false)
                dr_mask = pybuiltins.None
            else
                self.use_dr_mask = pybool(true)
                if pyisinstance(dr_mask, torch.Tensor)
                    if pyeq(dr_mask.dim(), Py(1))
                        dr_mask = dr_mask.to(self.device)
                    else
                        error("dr_mask should be 1-dim, got $(dr_mask.dim())")
                    end
                else
                    error("dr_mask must be a tensor")
                end
            end
            
            # 构建层
            for i in 1:pyconvert(Int, n_symbol_layers)
                if pyconvert(Bool, self.use_dr_mask) && pyeq(i, n_symbol_layers)
                    self.list.append(
                        DRLayer(self.list[end].out_dim, dr_mask=dr_mask, device=self.device)
                    )
                end
                
                if pyeq(i, 1)
                    self.list.append(
                        SymbolLayer(n_variables, operators, device=self.device)
                    )
                else
                    self.list.append(
                        SymbolLayer(self.list[end].out_dim, operators, device=self.device)
                    )
                end
            end
            
            self.current_expr_ls = []
            self.out_dim = self.list[end].out_dim
            
            return
        end
    ),
    pyfunc(
        name = "__repr__",
        function (self)
            base_repr = pybuiltins.super(PSRN, self).__repr__()
            info = "n_inputs: $(self.n_variables), operators: $(self.operators), n_layers: $(self.n_symbol_layers)"
            dims = join([string(layer.out_dim) for layer in self.list], "\n")
            return base_repr * "\n" * info * "\n dim:\n" * dims
        end
    ),
    pyfunc(
        name = "forward",
        function (self, x)
            # shape x: (batch_size, n_variables)
            h = x
            for layer in self.list
                h = layer(h)
            end
            return h  # shape: (batch_size, out_dim)
        end
    ),
    pyfunc(
        name = "get_expr",
        function (self, index)
            return self._get_expr(index, Py(-1))
        end
    ),
    pyfunc(
        name = "_get_expr",
        function (self, index, layer_idx)
            if pyconvert(Bool, pylt(pylen(self.list) + layer_idx, 0))
                return self.current_expr_ls[index + 1]
            end
            
            layer = self.list[pylen(self.list) + layer_idx + 1]
            
            if pyeq(layer._get_name(), "DRLayer")
                new_index = layer.get_op_and_offset(index)
                return self._get_expr(new_index, layer_idx - 1)
            else
                # SymbolLayer
                func_op, offset = layer.get_op_and_offset(index)
                
                if pyconvert(Bool, func_op.is_unary)
                    return func_op.get_expr(self._get_expr(offset[1], layer_idx - 1))
                else
                    return func_op.get_expr(
                        self._get_expr(offset[1], layer_idx - 1),
                        self._get_expr(offset[2], layer_idx - 1)
                    )
                end
            end
        end
    )
])



# 测试函数
function test_psrn()
    # 检查是否有可用的CUDA设备
    is_cuda_available = pyconvert(Bool, torch.cuda.is_available())
    device = torch.device(is_cuda_available ? "cuda" : "cpu")
    println("Using device: ", device)

    # 创建PSRN模型
    model = PSRN(
        Py(3),  # n_variables
        Py(["Add", "Mul", "Identity", "Sin", "Exp"]),  # operators
        Py(2),  # n_symbol_layers
        pybuiltins.None,  # dr_mask
        device  # device
    )
    
    # 将模型移到指定设备
    model = model.to(device)
    
    # 创建一些随机输入数据
    x = torch.randn((2, 3), device=device)
    println("\nInput shape: ", x.shape)
    println("Input data:\n", x)
    
    # 前向传播
    output = model(x)
    println("\nOutput shape: ", output.shape)
    println("Output snippet:\n", output[Py(0):Py(1), Py(0):Py(5)])
    
    # 获取一些表达式示例
    println("\nSome expression examples:")
    for i in 1:3
        expr = model.get_expr(Py(i-1))
        println("Expression $i: ", expr)
    end
    
    return model, x, output
end

model, x, output = test_psrn()