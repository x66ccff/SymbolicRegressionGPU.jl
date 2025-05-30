module PSRNmodel

import ..CoreModule.OperatorsModule:
    plus,
    sub,
    mult,
    square,
    cube,
    safe_pow,
    safe_log,
    safe_log2,
    safe_log10,
    safe_sqrt,
    safe_acosh,
    neg,
    greater,
    cond,
    relu,
    logical_or,
    logical_and,
    gamma

import ..CoreModule: Options, Dataset

using Printf: @sprintf
using DynamicExpressions: Node, Expression
using PythonCall
# 导入PyTorch相关模块
# torch = @pyconst(pyimport("torch"))

const torch = Ref{Py}()
const nn = Ref{Py}()


const numpy = Ref{Py}()

# const array = Ref{Py}()

const array_class_ref = Ref{Py}()
const torch_tensor_ref = Ref{Py}()


const CanCountLeaveOperator = Ref{Py}()
const DRLayer = Ref{Py}()
const SymbolLayer = Ref{Py}()
const PSRN = Ref{Py}()

const op_dict = Ref{Dict}()
const kernel_dict = Ref{Dict}()


########################## op #####################
const Identity_op = Ref{Py}()
const Add_op = Ref{Py}()
const Mul_op = Ref{Py}()
const Sub_op = Ref{Py}()
const Div_op = Ref{Py}()
const SemiSub_op = Ref{Py}()
const SemiDiv_op = Ref{Py}()
const Inv_op = Ref{Py}()
const Neg_op = Ref{Py}()
const Sin_op = Ref{Py}()
const Cos_op = Ref{Py}()
const Exp_op = Ref{Py}()
const Log_op = Ref{Py}()
const Pow2_op = Ref{Py}()
const Pow3_op = Ref{Py}()
const Sqrt_op = Ref{Py}()


########################### kernel ##############
const Identity = Ref{Py}()
const Add = Ref{Py}()
const Mul = Ref{Py}()
const Sub = Ref{Py}()
const Div = Ref{Py}()
const SemiSub = Ref{Py}()
const SemiDiv = Ref{Py}()
const Inv = Ref{Py}()
const Neg = Ref{Py}()
const Sin = Ref{Py}()
const Cos = Ref{Py}()
const Exp = Ref{Py}()
const Log = Ref{Py}()
const Pow2 = Ref{Py}()
const Pow3 = Ref{Py}()
const Sqrt = Ref{Py}()


const now_device = Ref{Py}()


function __init__()
    # 检查是否有可用的CUDA设备
    torch[] = pyimport("torch")
    nn[] = pyimport("torch.nn")
    numpy[] = pyimport("numpy")

    torch_tensor_ref[] = pyimport("torch").Tensor
    array_class_ref[] = pyimport("array").array # 注意这个 array 是类，所有要这样导入

    # 基础运算符类
    CanCountLeaveOperator[] = pytype("CanCountLeaveOperator", (nn[].Module,), [
        "__module__" => "__main__",
        pyfunc(
            name = "__init__",
            function (self)
                pybuiltins.super(CanCountLeaveOperator[], self).__init__()
                return
            end
        )
    ])

    # Duplicate Removal Layer
    DRLayer[] = pytype("DRLayer", (nn[].Module,), [
        "__module__" => "__main__",
        pyfunc(
            name = "__init__",
            # 同样添加kwargs参数
            function (self, in_dim, dr_mask=nothing; device=nothing)
                pybuiltins.super(DRLayer[], self).__init__()
                
                self.in_dim = in_dim

                self.device = device
                
                arange_tensor = torch[].arange(pylen(dr_mask), device=device)
                self.dr_indices = arange_tensor[dr_mask]  # (n,)
                self.dr_mask = dr_mask  # (n,)
        
                self.dr_indices = self.dr_indices.to(device)
                self.dr_mask = self.dr_mask.to(device)

                if !pyis(dr_mask, pybuiltins.None)
                    self.out_dim = pyconvert(Int, dr_mask.sum().item())
                else
                    self.out_dim = pyconvert(Int, in_dim)
                end
                
                return
            end
        ),
        pyfunc(
            name = "forward",
            function (self, x)
                return x[pyslice(nothing), self.dr_mask]
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
    SymbolLayer[] = pytype("SymbolLayer", (nn[].Module,), [
        "__module__" => "__main__",
        pyfunc(
            name = "__init__",
            # 这里添加kwargs参数来处理关键字参数
            function (self, in_dim, operators=["Add", "Mul", "Identity", "Sin", "Exp", "Neg", "Inv"]; device=nothing)
                pybuiltins.super(SymbolLayer[], self).__init__()
                
                self.in_dim = in_dim
                self.operators = operators
                self.device = device

                self.n_triu = pyfloordiv(in_dim * (in_dim + 1), 2)
                self.in_dim_square = in_dim * in_dim

                self.out_dim_cum_ls = Py(nothing)
                self.list = pylist([])
                self.offset_tensor = Py(nothing)
                
                for op_str in operators
                    self.list.append(kernel_dict[][op_str](in_dim))
                end

                # 计算输出维度
                self.out_dim = 0
                for op_str in operators
                    op = op_dict[][op_str]
                    if pyconvert(Bool, op.is_unary)
                        res = pyconvert(Int, in_dim)
                    else
                        res = pyconvert(Bool, op.is_directed) ? 
                            pyconvert(Int, in_dim)^2 : 
                            pyconvert(Int, in_dim) * (pyconvert(Int, in_dim) + 1) ÷ 2
                    end
                    @info "加了 $res ，因为 $op"
                    self.out_dim += res
                end

                @assert pylen(operators) == pylen(self.list)

                @info "计算的🎇🎇🎇🎇 self.out_dim"
                @info self.out_dim
                
                @info "开始init offset"
                self.init_offset()

                return
            end
        ),
        pyfunc(
            name = "forward",
            function (self, x)
                h = pylist([])
                for i in 1:pylen(self.list)
                    # 强制当前设备上所有流的所有GPU操作完成
                    md = self.list[pyindex(i-1)]
                    h.append(md(x))
                    # torch[].cuda.empty_cache() # TODO

                end
                res = torch[].cat(h, dim=1)
                PythonCall.pydel!(h)
                return res
            end
        ),
        pyfunc(
            name = "init_offset",
            function (self)
                self.offset_tensor = self.get_offset_tensor()
            end
        ),
        pyfunc(
            name = "get_offset_tensor",
            function (self)
                device = self.device
                offset_tensor = torch[].zeros((self.out_dim, 2), dtype=torch[].int, device=device)
                arange_tensor = torch[].arange(self.in_dim, dtype=torch[].int, device=device)
                
                binary_U_tensor = torch[].zeros((self.n_triu, 2), dtype=torch[].int, device=device)
                binary_D_tensor = torch[].zeros((self.in_dim_square, 2), dtype=torch[].int, device=device)
                unary_tensor = torch[].zeros((self.in_dim, 2), dtype=torch[].int, device=device)
                
                unary_tensor[pyslice(nothing), 0] = arange_tensor
                unary_tensor[pyslice(nothing), 1] = self.in_dim

                start = 0
                for i in 0:pyconvert(Int, self.in_dim) - 1
                    
                    len_ = self.in_dim - i
                    
                    binary_U_tensor[pyslice(start , start + len_), 0] = pyint(i)
                    
                    binary_U_tensor[pyslice(start , start + len_), 1] = arange_tensor[pyslice(i,nothing)]
                    
                    start += len_
                end
                
                start = 0
                for i in 0:pyconvert(Int, self.in_dim) - 1
                    len_ = self.in_dim
                    binary_D_tensor[pyslice(start , start + len_), 0] = pyint(i)
                    binary_D_tensor[pyslice(start , start + len_), 1] = arange_tensor[pyslice(0,nothing)]
                    start += len_    
                end

                start = 0
                for func in self.list
                    if !pyconvert(Bool, func.is_unary)
                        if pyconvert(Bool,func.is_directed)
                            t = binary_D_tensor
                        else
                            t = binary_U_tensor

                        end
                    else
                        t = unary_tensor
                    end
                    len_ = t.shape[0]
                    
                    if ((pyconvert(Int, start) + pyconvert(Int, len_)) <= pyconvert(Int, self.out_dim))
                    # if pyle(start + len_, self.out_dim)
                        offset_tensor[pyslice(start,start + len_,nothing)] = t
                    else
                        @info "pass"
                    end
                    start += len_
                end
                
                return offset_tensor
            end
        ),
        pyfunc(
            name = "get_out_dim_cum_ls",
            function (self)
                if !pyconvert(Bool, pyis(self.out_dim_cum_ls, Py(nothing)))
                    return self.out_dim_cum_ls
                end
                
                out_dim_ls = pylist([])
                for func in self.list
                    if !pyconvert(Bool, func.is_unary)
                        if pyconvert(Bool, func.is_directed)
                            out_dim_ls.append(self.in_dim_square)
                        else
                            out_dim_ls.append(self.n_triu)
                        end
                    else
                        out_dim_ls.append(self.in_dim)
                    end
                end
                self.out_dim_cum_ls = [sum(out_dim_ls[pyslice(nothing,i+1)]) for i in 0:length(out_dim_ls)-1]
                @info "self.out_dim_cum_ls"
                @info self.out_dim_cum_ls
                return self.out_dim_cum_ls
            end
        ),
        pyfunc(
            name = "get_op_and_offset",
            function (self, index)
                out_dim_cum_ls = self.get_out_dim_cum_ls()
                i = 0
                for (idx, val) in enumerate(out_dim_cum_ls)
                    if pyconvert(Bool, pylt(index,val))
                        i = idx - 1
                        break
                    end
                end
                func = self.list[i]
                offset = self.offset_tensor[index].tolist()
                return func.operator, offset
            end
        )
    ])

    # PSRN类
    PSRN[] = pytype("PSRN", (nn[].Module,), [
        "__module__" => "__main__",
        pyfunc(
            name = "__init__",
            function (self, n_variables=1, 
                    operators=["Add", "Mul", "Identity", "Sin", "Exp", "Neg", "Inv"],
                    n_symbol_layers=3,
                    dr_mask=pybuiltins.None,
                    device="cuda")
                pybuiltins.super(PSRN[], self).__init__()
                
                # 设置设备
                if pyconvert(Bool, pyisinstance(device, pybuiltins.str))
                    if pyconvert(Bool, pyeq(device, "cuda"))
                        self.device = torch[].device("cuda")
                    elseif pyconvert(Bool, pyeq(device, "cpu"))
                        self.device = torch[].device("cpu")
                    else
                        error("device must be cuda or cpu, got $(device)")
                    end
                else
                    self.device = device
                end
                
                self.n_variables = n_variables
                self.operators = operators
                self.n_symbol_layers = n_symbol_layers
                
                self.list = nn[].ModuleList()
                
                # 处理dr_mask
                if pyconvert(Bool, pyis(dr_mask, pybuiltins.None))
                    self.use_dr_mask = pybool(false)
                else
                    self.use_dr_mask = pybool(true)
                    if !pyconvert(Bool, pyisinstance(dr_mask, torch[].Tensor))
                        error("dr_mask must be a tensor")
                    end
                    if !pyconvert(Bool, pyeq(dr_mask.dim(), Py(1)))
                        error("dr_mask should be 1-dim, got $(dr_mask.dim())")
                    end
                    dr_mask = dr_mask.to(self.device)
                end
                
                # 构建层
                @info "构建层"
                for i in 1:pyconvert(Int, n_symbol_layers)
                    @info "构建第 $i 层"
                    if pyconvert(Bool, pygt(pylen(self.list), 0)) && pyconvert(Bool, self.use_dr_mask) && pyconvert(Bool, pyeq(i, n_symbol_layers))
                        last_layer = self.list[pylen(self.list) - 1]  # Python的索引从0开始
                        self.list.append(
                            DRLayer[](last_layer.out_dim, dr_mask, device=self.device)
                        )
                    end
                    
                    if pyconvert(Bool, pyeq(i, 1))
                        self.list.append(
                            SymbolLayer[](n_variables, operators, device=self.device)
                        )
                    else
                        last_layer = self.list[pylen(self.list) - 1]  # 获取最后一层
                        self.list.append(
                            SymbolLayer[](last_layer.out_dim, operators, device=self.device)
                        )
                    end
                    @info "长度 $(pylen(self.list))"
                end
                
                self.current_expr_ls = []
                last_layer = self.list[pylen(self.list) - 1]  # 获取最后一层
                self.out_dim = last_layer.out_dim

                @info "self.out_dim = $(self.out_dim)"
                
                return
            end
        ),
        pyfunc(
            name = "__repr__",
            function (self)
                base_repr = pybuiltins.super(PSRN[], self).__repr__()
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
                return h
            end
        ),
        pyfunc(
            name = "get_expr",
            function (self, index)
                return self._get_expr(index, -1)
            end
        ),
        pyfunc(
            name = "_get_expr",
            function (self, index, layer_idx)
                if pyconvert(Bool, pylt(pylen(self.list) + layer_idx, 0))
                    return self.current_expr_ls[index]
                end
                
                layer = self.list[layer_idx]
                
                if pyconvert(Bool, pyeq(layer._get_name(), Py("DRLayer")))
                    new_index = layer.get_op_and_offset(index)
                    res = self._get_expr(new_index, layer_idx - 1)
                    return res
                else
                    # SymbolLayer
                    func_op, offset = layer.get_op_and_offset(index)
                    # @info "func_op, offset <======== index"
                    # @info "$func_op, $offset <======== $index"
                    if pyconvert(Bool, func_op.is_unary)
                        return func_op.get_expr(
                            self._get_expr(offset[0], layer_idx - 1))
                    else
                        return func_op.get_expr(
                            self._get_expr(offset[0], layer_idx - 1),
                            self._get_expr(offset[1], layer_idx - 1)
                        )
                    end
                end
            end
        )
    ])

    Add_op[] = pytype("Add_op", (), [
        "__module__" => "__main__",
        pyfunc(
            name = "__init__",
            function (self)
                pybuiltins.super(Add_op[], self).__init__()
                self.is_unary = false
                self.is_directed = false
                return
            end
        ),
        pyfunc(
            name = "get_expr",
            (self, sub_expr1, sub_expr2) -> sub_expr1 + sub_expr2
        ),
        pyfunc(
            name = "transform_inputs",
            (self, x1, x2) -> x1 + x2
        )
    ])

    Identity_op[] = pytype("Identity_op", (), [
        "__module__" => "__main__",
        pyfunc(
            name = "__init__",
            function (self)
                pybuiltins.super(Identity_op[], self).__init__()
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

    Mul_op[] = pytype("Mul_op", (), [
        "__module__" => "__main__",
        pyfunc(
            name = "__init__",
            function (self)
                pybuiltins.super(Mul_op[], self).__init__()
                self.is_unary = false
                self.is_directed = false
                return
            end
        ),
        pyfunc(
            name = "get_expr",
            (self, sub_expr1, sub_expr2) -> sub_expr1 * sub_expr2
        ),
        pyfunc(
            name = "transform_inputs",
            (self, x1, x2) -> x1 * x2
        )
    ])
    
    Div_op[] = pytype("Div_op", (), [
        "__module__" => "__main__",
        pyfunc(
            name = "__init__",
            function (self, threshold=1e-10)
                pybuiltins.super(Div_op[], self).__init__()
                self.is_unary = false
                self.is_directed = true
                self.threshold = threshold
                return
            end
        ),
        pyfunc(
            name = "get_expr",
            (self, sub_expr1, sub_expr2) -> sub_expr1 / sub_expr2
        ),
        pyfunc(
            name = "transform_inputs",
            function (self, x1, x2)
                return x1 / x2
            end
        )
    ])
    
    # Sub_op
    Sub_op[] = pytype("Sub_op", (), [
        "__module__" => "__main__",
        pyfunc(
            name = "__init__",
            function (self)
                pybuiltins.super(Sub_op[], self).__init__()
                self.is_unary = false
                self.is_directed = true
                return
            end
        ),
        pyfunc(
            name = "get_expr",
            (self, sub_expr1, sub_expr2) -> sub_expr1 - sub_expr2
        ),
        pyfunc(
            name = "transform_inputs",
            (self, x1, x2) -> x1 - x2
        )
    ])


    # Sin_op
    Sin_op[] = pytype("Sin_op", (), [
        "__module__" => "__main__",
        pyfunc(
            name = "__init__",
            function (self)
                pybuiltins.super(Sin_op[], self).__init__()
                self.is_unary = true
                return
            end
        ),
        pyfunc(
            name = "get_expr",
            (self, sub_expr) -> sin(pyconvert(Expression,sub_expr))
        ),
        pyfunc(
            name = "transform_inputs",
            (self, x) -> torch[].sin(x)
        )
    ])

    # Cos_op
    Cos_op[] = pytype("Cos_op", (), [
        "__module__" => "__main__",
        pyfunc(
            name = "__init__",
            function (self)
                pybuiltins.super(Cos_op[], self).__init__()
                self.is_unary = true
                return
            end
        ),
        pyfunc(
            name = "get_expr",
            (self, sub_expr) -> cos(pyconvert(Expression,sub_expr))
        ),
        pyfunc(
            name = "transform_inputs",
            (self, x) -> torch[].cos(x)
        )
    ])

    # Neg_op
    Neg_op[] = pytype("Neg_op", (), [
        "__module__" => "__main__",
        pyfunc(
            name = "__init__",
            function (self)
                pybuiltins.super(Neg_op[], self).__init__()
                self.is_unary = true
                return
            end
        ),
        pyfunc(
            name = "get_expr",
            (self, sub_expr) -> sub_expr * Node(; val=Float32(-1)) # TODO error here
        ),
        pyfunc(
            name = "transform_inputs",
            (self, x) -> -x
        )
    ])

    # Inv_op
    Inv_op[] = pytype("Inv_op", (), [
        "__module__" => "__main__",
        pyfunc(
            name = "__init__",
            function (self, threshold=1e-10)
                pybuiltins.super(Inv_op[], self).__init__()
                self.is_unary = true
                self.threshold = threshold
                return
            end
        ),
        pyfunc(
            name = "get_expr",
            (self, sub_expr) -> 1/sub_expr
        ),
        pyfunc(
            name = "transform_inputs",
            function (self, x)
                return 1 / x
            end
        )
    ])

    # SemiDiv_op
    SemiDiv_op[] = pytype("SemiDiv_op", (), [
        "__module__" => "__main__",
        pyfunc(
            name = "__init__",
            function (self, threshold=1e-10)
                pybuiltins.super(SemiDiv_op[], self).__init__()
                self.is_unary = false
                self.is_directed = false
                self.threshold = threshold
                return
            end
        ),
        pyfunc(
            name = "get_expr",
            (self, sub_expr1, sub_expr2) -> sub_expr1 / sub_expr2
        ),
        pyfunc(
            name = "transform_inputs",
            function (self, x1, x2)
                return x1 / x2
            end
        )
    ])

    # SemiSub_op
    SemiSub_op[] = pytype("SemiSub_op", (), [
        "__module__" => "__main__",
        pyfunc(
            name = "__init__",
            function (self)
                pybuiltins.super(SemiSub_op[], self).__init__()
                self.is_unary = false
                self.is_directed = false
                return
            end
        ),
        pyfunc(
            name = "get_expr",
            (self, sub_expr1, sub_expr2) -> sub_expr1 - sub_expr2
        ),
        pyfunc(
            name = "transform_inputs",
            (self, x1, x2) -> x1 - x2
        )
    ])

    Sqrt_op[] = pytype("Sqrt_op", (), [
        "__module__" => "__main__",
        pyfunc(
            name = "__init__",
            function (self)
                pybuiltins.super(Sqrt_op[], self).__init__()
                self.is_unary = true
                self.is_directed = true
                return
            end
        ),
        pyfunc(
            name = "get_expr",
            (self, sub_expr) -> sqrt(pyconvert(Expression,sub_expr))
        ),
        pyfunc(
            name = "transform_inputs",
            (self, x) -> torch[].sqrt(x)
        )
    ])

    # Pow2_op
    Pow2_op[] = pytype("Pow2_op", (), [
        "__module__" => "__main__",
        pyfunc(
            name = "__init__",
            function (self)
                pybuiltins.super(Pow2_op[], self).__init__()
                self.is_unary = true
                self.is_directed = true
                return
            end
        ),
        pyfunc(
            name = "get_expr",
            (self, sub_expr) -> sub_expr * sub_expr
        ),
        pyfunc(
            name = "transform_inputs",
            (self, x) -> x^2
        )
    ])

    # Pow3_op
    Pow3_op[] = pytype("Pow3_op", (), [
        "__module__" => "__main__",
        pyfunc(
            name = "__init__",
            function (self)
                pybuiltins.super(Pow3_op[], self).__init__()
                self.is_unary = true
                self.is_directed = true
                return
            end
        ),
        pyfunc(
            name = "get_expr",
            (self, sub_expr) -> cube(pyconvert(Expression,sub_expr))
        ),
        pyfunc(
            name = "transform_inputs",
            (self, x) -> x^3
        )
    ])

    
    # Exp_op
    Exp_op[] = pytype("Exp_op", (), [
        "__module__" => "__main__",
        pyfunc(
            name = "__init__",
            function (self, threshold=10)
                pybuiltins.super(Exp_op[], self).__init__()
                self.is_unary = true
                self.threshold = threshold
                return
            end
        ),
        pyfunc(
            name = "get_expr",
            (self, sub_expr) -> exp(pyconvert(Expression,sub_expr))
        ),
        pyfunc(
            name = "transform_inputs",
            (self, x) -> torch[].exp(x)
        )
    ])

    # Log_op
    Log_op[] = pytype("Log_op", (), [
        "__module__" => "__main__",
        pyfunc(
            name = "__init__",
            function (self, threshold=1e-10)
                pybuiltins.super(Log_op[], self).__init__()
                self.is_unary = true
                self.threshold = threshold
                return
            end
        ),
        pyfunc(
            name = "get_expr",
            (self, sub_expr) -> log(pyconvert(Expression,sub_expr))
        ),
        pyfunc(
            name = "transform_inputs",
            (self, x) -> torch[].log(x)
        )
    ])

########################################## op 👆 kernel 👇 ####################


    # Pow2类
    Pow2[] = pytype("Pow2", (CanCountLeaveOperator[],), [
        "__module__" => "__main__",
        pyfunc(
            name = "__init__",
            function (self, in_dim=1, device=nothing)
                pybuiltins.super(Pow2[], self).__init__()
                self.in_dim = in_dim
                self.out_dim = in_dim
                self.is_unary = pybool(true)
                self.is_directed = pybool(true)
                self.complexity = 2
                self.operator = Pow2_op[]()
                self.device = device
                return
            end
        ),
        pyfunc(
            name = "forward",
            function (self, x)
                return x^2
            end
        )
    ])

    # Pow3类
    Pow3[] = pytype("Pow3", (CanCountLeaveOperator[],), [
        "__module__" => "__main__",
        pyfunc(
            name = "__init__",
            function (self, in_dim=1, device=nothing)
                pybuiltins.super(Pow3[], self).__init__()
                self.in_dim = in_dim
                self.out_dim = in_dim
                self.is_unary = pybool(true)
                self.is_directed = pybool(true)
                self.complexity = 3
                self.operator = Pow3_op[]()
                self.device = device
                return
            end
        ),
        pyfunc(
            name = "forward",
            function (self, x)
                return x^3
            end
        )
    ])

    # Sqrt类
    Sqrt[] = pytype("Sqrt", (CanCountLeaveOperator[],), [
        "__module__" => "__main__",
        pyfunc(
            name = "__init__",
            function (self, in_dim=1, device=nothing)
                pybuiltins.super(Sqrt[], self).__init__()
                self.in_dim = in_dim
                self.out_dim = in_dim
                self.is_unary = pybool(true)
                self.is_directed = pybool(true)
                self.complexity = 3
                self.operator = Sqrt_op[]()
                self.device = device
                return
            end
        ),
        pyfunc(
            name = "forward",
            function (self, x)
                return torch[].sqrt(x)
            end
        )
    ])


    # Identity类
    Identity[] = pytype("Identity", (CanCountLeaveOperator[],), [
        "__module__" => "__main__",
        pyfunc(
            name = "__init__",
            function (self, in_dim=1, device=nothing)
                pybuiltins.super(Identity[], self).__init__()
                self.in_dim = in_dim
                self.out_dim = in_dim
                self.is_unary = pybool(true)
                self.is_directed = pybool(true)
                self.operator = Identity_op[]()
                self.complexity = 0
                self.device = device
                return
            end
        ),
        pyfunc(
            name = "forward",
            function (self, x)
                return x
            end
        )
    ])


    # Add类
    Add[] = pytype("Add", (CanCountLeaveOperator[],), [
        "__module__" => "__main__",
        pyfunc(
            name = "__init__",
            function (self, in_dim=1, device=nothing)
                pybuiltins.super(Add[], self).__init__()
                self.in_dim = in_dim
                self.out_dim = in_dim * (in_dim + 1) ÷ 2
                self.is_unary = pybool(false)
                self.is_directed = pybool(false)
                self.complexity = 1
                self.operator = Add_op[]()
                self.device = device
                return
            end
        ),
        pyfunc(
            name = "forward",
            function (self, x)
                indices = torch[].triu_indices(
                    self.in_dim, self.in_dim, offset=0, dtype=torch[].int32, device=x.device
                )
                out = x[pyslice(nothing), indices[0]] + x[pyslice(nothing), indices[1]]
                PythonCall.pydel!(indices)
                return out
            end
        )
    ])

    Mul[] = pytype("Mul", (CanCountLeaveOperator[],), [
        "__module__" => "__main__",
        pyfunc(
            name = "__init__",
            function (self, in_dim=1, device=nothing)
                pybuiltins.super(Mul[], self).__init__()
                self.in_dim = in_dim
                self.out_dim = in_dim * (in_dim + 1) ÷ 2
                self.is_unary = pybool(false)
                self.is_directed = pybool(false)
                self.complexity = 1
                self.operator = Mul_op[]()
                self.device = device
                return
            end
        ),
        pyfunc(
            name = "forward",
            function (self, x)
                indices = torch[].triu_indices(
                    self.in_dim, self.in_dim, offset=0, dtype=torch[].int32, device=x.device
                )
                out = x[pyslice(nothing), indices[0]] * x[pyslice(nothing), indices[1]]
                PythonCall.pydel!(indices)
                return out
            end
        )
    ])



    # Sub类
    Sub[] = pytype("Sub", (CanCountLeaveOperator[],), [
        "__module__" => "__main__",
        pyfunc(
            name = "__init__",
            function (self, in_dim=1, device=nothing)
                pybuiltins.super(Sub[], self).__init__()
                self.in_dim = in_dim
                self.out_dim = in_dim * in_dim
                self.is_unary = pybool(false)
                self.is_directed = pybool(true)
                self.complexity = 1
                self.operator = Sub_op[]()
                self.device = device
                return
            end
        ),
        pyfunc(
            name = "forward",
            function (self, x)
                num = x.view(1, -1, 1)
                deno = x.view(1, 1, -1)
                out = (num - deno).view(1, -1)
                return out
            end
        )
    ])


    # Div
    Div[] = pytype("Div", (CanCountLeaveOperator[],), [
        "__module__" => "__main__",
        pyfunc(
            name = "__init__",
            function (self, in_dim=1, device=nothing)
                pybuiltins.super(Div[], self).__init__()
                self.in_dim = in_dim
                self.out_dim = in_dim * in_dim
                self.is_unary = pybool(false)
                self.is_directed = pybool(true)
                self.complexity = 1
                self.operator = Div_op[]()
                self.device = device
                return
            end
        ),
        pyfunc(
            name = "forward",
            function (self, x)
                num = x.view(1, -1, 1)
                deno = x.view(1, 1, -1)
                out = (num / deno).view(1, -1)
                return out
            end
        )
    ])


    # Neg类
    Neg[] = pytype("Neg", (CanCountLeaveOperator[],), [
        "__module__" => "__main__",
        pyfunc(
            name = "__init__",
            function (self, in_dim=1, device=nothing)
                pybuiltins.super(Neg[], self).__init__()
                self.in_dim = in_dim
                self.out_dim = in_dim
                self.is_unary = pybool(true)
                self.is_directed = pybool(true)
                self.operator = Neg_op[]()
                self.complexity = 1
                self.device = device
                return
            end
        ),
        pyfunc(
            name = "forward",
            function (self, x)
                return -x
            end
        )
    ])

    # Inv类
    Inv[] = pytype("Inv", (CanCountLeaveOperator[],), [
        "__module__" => "__main__",
        pyfunc(
            name = "__init__",
            function (self, in_dim=1, device=nothing)
                pybuiltins.super(Inv[], self).__init__()
                self.in_dim = in_dim
                self.out_dim = in_dim
                self.is_unary = pybool(true)
                self.is_directed = pybool(true)
                self.operator = Inv_op[]()
                self.complexity = 2
                self.device = device
                return
            end
        ),
        pyfunc(
            name = "forward",
            function (self, x)
                return 1 / x
            end
        )
    ])

    # SemiDiv类
    SemiDiv[] = pytype("SemiDiv", (CanCountLeaveOperator[],), [
        "__module__" => "__main__",
        pyfunc(
            name = "__init__",
            function (self, in_dim=1, device=nothing, threshold=1e-10)
                pybuiltins.super(SemiDiv[], self).__init__()
                self.threshold = threshold
                self.in_dim = in_dim
                self.out_dim = in_dim * (in_dim + 1) ÷ 2
                self.is_unary = pybool(false) 
                self.is_directed = pybool(false)
                self.complexity = 2
                self.operator = SemiDiv_op[]()
                self.device = device
                return
            end
        ),
        pyfunc(
            name = "forward",
            function (self, x)
                indices = torch[].triu_indices(
                    self.in_dim, self.in_dim, offset=0, dtype=torch[].int32, device=x.device
                )
                deno = x[pyslice(nothing), indices[1]]
                num = x[pyslice(nothing), indices[0]]
                PythonCall.pydel!(indices)
                return num / deno
            end
        )
    ])

    # SemiSub类
    SemiSub[] = pytype("SemiSub", (CanCountLeaveOperator[],), [
        "__module__" => "__main__",
        pyfunc(
            name = "__init__",
            function (self, in_dim=1, device=nothing)
                pybuiltins.super(SemiSub[], self).__init__()
                self.in_dim = in_dim
                self.out_dim = in_dim * (in_dim + 1) ÷ 2
                self.is_unary = pybool(false)
                self.is_directed = pybool(false)
                self.complexity = 1
                self.operator = SemiSub_op[]()
                self.device = device
                return
            end
        ),
        pyfunc(
            name = "forward",
            function (self, x)
                indices = torch[].triu_indices(
                    self.in_dim, self.in_dim, offset=0, dtype=torch[].int32, device=x.device
                )
                out = x[pyslice(nothing), indices[0]] - x[pyslice(nothing), indices[1]]
                PythonCall.pydel!(indices)
                return out
            end
        )
    ])

    # Sin类
    Sin[] = pytype("Sin", (CanCountLeaveOperator[],), [
        "__module__" => "__main__",
        pyfunc(
            name = "__init__",
            function (self, in_dim=1, device=nothing)
                pybuiltins.super(Sin[], self).__init__()
                self.in_dim = in_dim
                self.out_dim = in_dim
                self.is_unary = pybool(true)
                self.is_directed = pybool(true)
                self.operator = Sin_op[]()
                self.complexity = 4
                self.device = device
                return
            end
        ),
        pyfunc(
            name = "forward",
            function (self, x)
                return torch[].sin(x)
            end
        )
    ])

    # Cos类
    Cos[] = pytype("Cos", (CanCountLeaveOperator[],), [
        "__module__" => "__main__",
        pyfunc(
            name = "__init__",
            function (self, in_dim=1, device=nothing)
                pybuiltins.super(Cos[], self).__init__()
                self.in_dim = in_dim
                self.out_dim = in_dim
                self.is_unary = pybool(true)
                self.is_directed = pybool(true)
                self.operator = Cos_op[]()
                self.complexity = 4
                self.device = device
                return
            end
        ),
        pyfunc(
            name = "forward",
            function (self, x)
                return torch[].cos(x)
            end
        )
    ])


    # Exp类
    Exp[] = pytype("Exp", (CanCountLeaveOperator[],), [
        "__module__" => "__main__",
        pyfunc(
            name = "__init__",
            function (self, in_dim=1, device=nothing, threshold=10)
                pybuiltins.super(Exp[], self).__init__()
                self.threshold = threshold
                self.in_dim = in_dim
                self.out_dim = in_dim
                self.is_unary = pybool(true)
                self.is_directed = pybool(true)
                self.operator = Exp_op[]()
                self.complexity = 4
                self.device = device
                return
            end
        ),
        pyfunc(
            name = "forward",
            function (self, x)
                return torch[].exp(x)
            end
        )
    ])

    # Log类
    Log[] = pytype("Log", (CanCountLeaveOperator[],), [
        "__module__" => "__main__",
        pyfunc(
            name = "__init__",
            function (self, in_dim=1, device=nothing, threshold=1e-10)
                pybuiltins.super(Log[], self).__init__()
                self.threshold = threshold
                self.in_dim = in_dim
                self.out_dim = in_dim
                self.is_unary = pybool(true)
                self.is_directed = pybool(true)
                self.complexity = 4
                self.operator = Log_op[]()
                self.device = device
                return
            end
        ),
        pyfunc(
            name = "forward",
            function (self, x)
                return torch[].log(x)
            end
        )
    ])


    # 初始化运算符字典
    op_dict[] = Dict(
        pystr("Add") => Add_op[](),
        pystr("Mul") => Mul_op[](),
        pystr("Identity") => Identity_op[](),
        pystr("Sin") => Sin_op[](),
        pystr("Cos") => Cos_op[](),
        pystr("Exp") => Exp_op[](),
        pystr("Log") => Log_op[](),
        pystr("Neg") => Neg_op[](),
        pystr("Inv") => Inv_op[](),
        pystr("Div") => Div_op[](),
        pystr("Sub") => Sub_op[](),
        pystr("SemiDiv") => SemiDiv_op[](),
        pystr("SemiSub") => SemiSub_op[](),
        # pystr("Sign") => Sign_op(),
        pystr("Pow2") => Pow2_op[](),
        pystr("Pow3") => Pow3_op[](),
        # pystr("Pow") => Pow_op(),
        # pystr("Sigmoid") => Sigmoid_op(),
        # pystr("Abs") => Abs_op(),
        # pystr("Cosh") => Cosh_op(),
        # pystr("Tanh") => Tanh_op(),
        pystr("Sqrt") => Sqrt_op[]()
    )

    kernel_dict[] = Dict(
        pystr("Add") => Add[],
        pystr("Mul") => Mul[],
        pystr("Identity") => Identity[],
        pystr("Sin") => Sin[],
        pystr("Cos") => Cos[],
        pystr("Exp") => Exp[],
        pystr("Log") => Log[],
        pystr("Neg") => Neg[],
        pystr("Inv") => Inv[],
        pystr("Div") => Div[],
        pystr("Sub") => Sub[],
        pystr("SemiDiv") => SemiDiv[],
        pystr("SemiSub") => SemiSub[],
        # pystr("Sign") => Sign,
        pystr("Pow2") => Pow2[],
        pystr("Pow3") => Pow3[],
        # pystr("Pow") => Pow,
        # pystr("Sigmoid") => Sigmoid,
        # pystr("Abs") => Abs,
        # pystr("Cosh") => Cosh,
        # pystr("Tanh") => Tanh,
        pystr("Sqrt") => Sqrt[]
    )


    is_cuda_available = pyconvert(Bool, torch[].cuda.is_available())
    if is_cuda_available
        @info "Yes, cuda is available 😀"
    else 
        @info "No, cuda is not available 😭"
    end
    now_device[] = torch[].device(is_cuda_available ? "cuda" : "cpu")

    # # device = torch_device(is_cuda_available ? "cpu" : "cpu")
    println("Using device: ", now_device[])
end

export PSRN, now_device, torch, torch_tensor_ref, array_class_ref

end
