import os
import struct
import sys
sys.path.append(".")
import traceback
import numpy as np
import torch

gpu_index = 0 
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

from PSRNmodels import PSRN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

operators = ['Add','Mul','SemiSub','SemiDiv','Sin','Cos','Exp','Log']
n_psrn_input = 5
print(operators)

cnt_success = 0
sum_time = 0

variables_name = [f"x_{i}" for i in range(n_psrn_input)]
target_name = ["y"]

n_symbol_layers = 3
use_constant = False
psrn = PSRN(
            n_variables=n_psrn_input,
            operators=operators,
            n_symbol_layers=n_symbol_layers,
            dr_mask=None,
            device=device,
        )

JULIA_TO_PYTHON_PIPE = 'julia_to_python_pipe'
PYTHON_TO_JULIA_PIPE = 'python_to_julia_pipe'
ERROR_LOG_FILE = 'python_errors.log'
STDOUT_LOG_FILE = 'python_stdout.log'

def read_array_from_pipe(fifo_read):
    try:
        num_dims_data = fifo_read.read(8); num_dims = struct.unpack('q', num_dims_data)[0]
        shape_data = fifo_read.read(8 * num_dims); shape = struct.unpack(f'{num_dims}q', shape_data)
        num_elements = np.prod(shape) if shape else 0; data_bytes = int(num_elements * 8)
        flat_data = fifo_read.read(data_bytes)
        array = np.frombuffer(flat_data, dtype=np.float64).reshape(shape, order='F')
        
        # 修复1: 确保数组是可写的
        if not array.flags.writeable:
            array = array.copy()
        
        return array
    except Exception as e:
        sys.stderr.write(f"CRITICAL ERROR in read_array_from_pipe: {e}\n"); traceback.print_exc(file=sys.stderr)
        return None

def send_string_list(fifo_write, string_list):
    """
    序列化并发送字符串列表到指定的IO流。
    """
    try:
        # 1. 发送字符串列表长度
        list_length = len(string_list)
        packed_length = struct.pack('q', list_length)
        fifo_write.write(packed_length)
        sys.stdout.write(f"Sent list length: {list_length}\n")
        sys.stdout.flush()
        
        # 2. 发送每个字符串
        for i, s in enumerate(string_list):
            # 将字符串转换为UTF-8字节
            s_bytes = s.encode('utf-8')
            # 发送字符串长度
            str_length = len(s_bytes)
            packed_str_length = struct.pack('q', str_length)
            fifo_write.write(packed_str_length)
            # 发送字符串内容
            fifo_write.write(s_bytes)
            sys.stdout.write(f"Sent string {i+1}/{list_length}, length: {str_length}\n")
            sys.stdout.flush()
            
        sys.stdout.write(f"Sent string list with {list_length} strings\n")
        sys.stdout.flush()
        
        # 修复2: 改进管道刷新机制，添加错误处理
        try:
            fifo_write.flush()
            # 检查文件描述符是否有效
            fd = fifo_write.fileno()
            if fd >= 0:
                os.fsync(fd)
                sys.stdout.write("Successfully flushed and synced data to Julia\n")
            else:
                sys.stdout.write("Warning: Invalid file descriptor, skipping fsync\n")
        except (OSError, ValueError) as sync_error:
            sys.stdout.write(f"Warning: Could not sync to disk: {sync_error}. Data may still be sent successfully.\n")
        
        sys.stdout.flush()
        
    except Exception as e:
        sys.stderr.write(f"Error in send_string_list: {e}\n")
        traceback.print_exc(file=sys.stderr)
        raise  # 重新抛出异常，让调用者知道发送失败

def main():
    sys.stdout = open(STDOUT_LOG_FILE, 'w')
    sys.stderr = open(ERROR_LOG_FILE, 'w')
    if not torch.cuda.is_available(): 
        sys.stdout.write("CUDA not available.\n")
        return
    try:
        if not os.path.exists(JULIA_TO_PYTHON_PIPE): 
            os.mkfifo(JULIA_TO_PYTHON_PIPE)
        if not os.path.exists(PYTHON_TO_JULIA_PIPE): 
            os.mkfifo(PYTHON_TO_JULIA_PIPE)
        with open(JULIA_TO_PYTHON_PIPE, 'rb') as fifo_read, \
             open(PYTHON_TO_JULIA_PIPE, 'wb') as fifo_write:
            sys.stdout.write("Python ROBUST process started and listening...\n")
            sys.stdout.flush()
            while True:
                sys.stdout.write("\nWaiting for new job...\n")
                sys.stdout.flush()
                
                try:
                    # 1. 接收数据
                    trigger_data = fifo_read.read(8)
                    if len(trigger_data) != 8:
                        sys.stdout.write("Received incomplete trigger data, breaking loop\n")
                        break
                        
                    trigger_value = struct.unpack('d', trigger_data)[0]
                    X_np = read_array_from_pipe(fifo_read)
                    y_np = read_array_from_pipe(fifo_read)
                    if X_np is None or y_np is None: 
                        break
                    
                    # 2. 记录接收信息并转换为torch张量
                    sys.stdout.write(f"Received X shape {X_np.shape}, y shape {y_np.shape}\n")
                    
                    # 修复3: 确保NumPy数组可写后再转换为torch张量
                    if not X_np.flags.writeable:
                        X_np = X_np.copy()
                    if not y_np.flags.writeable:
                        y_np = y_np.copy()
                    
                    X_torch = torch.from_numpy(X_np).to('cuda')
                    y_torch = torch.from_numpy(y_np).to('cuda')
                    sys.stdout.write(f"Successfully converted to CUDA tensors.\n")
                    sys.stdout.flush()
                    
                    psrn.current_expr_ls = variables_name
                    n_top = 10
                    expr_best_ls, MSE_min_ls = psrn.get_best_expr_and_MSE_topk(X_torch, y_torch, n_top)
                    sys.stdout.write(f"Received expr_best_ls {expr_best_ls}, MSE_min_ls {MSE_min_ls}\n")
                    
                    # 4. 发送字符串列表结果
                    sys.stdout.write("About to send string list to Julia...\n")
                    sys.stdout.flush()
                    send_string_list(fifo_write, expr_best_ls)
                    sys.stdout.write("Finished sending string list to Julia\n")
                    sys.stdout.flush()
                    
                except BrokenPipeError:
                    sys.stdout.write("Broken pipe detected, Julia process may have terminated\n")
                    break
                except Exception as loop_error:
                    sys.stderr.write(f"Error in processing loop: {loop_error}\n")
                    traceback.print_exc(file=sys.stderr)
                    # 继续下一次循环而不是退出
                    continue
                
    except Exception as e:
        sys.stderr.write(f"Python script crashed in main loop: {e}\n")
        traceback.print_exc(file=sys.stderr)
    finally:
        sys.stdout.close()
        sys.stderr.close()

if __name__ == "__main__":
    main()