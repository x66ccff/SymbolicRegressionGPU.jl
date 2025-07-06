import os
import struct
import sys
sys.path.append(".")
import traceback
import numpy as np
import torch
import time
import gc
import select # Added for check_pipe_data_available

gpu_index = 0 
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

from PSRNmodels import PSRN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

operators = ['Add','Mul','Identity','Neg','Inv','Sin','Cos','Exp','Log']
n_top = 50
n_psrn_input = 8
# n_psrn_input = 4
print(operators)

cnt_success = 0
sum_time = 0

variables_name = [f"v_{i}" for i in range(n_psrn_input)]
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

def check_pipe_data_available(fifo_read):
    """
    检查管道中是否有数据可读，非阻塞方式
    """
    import select
    import os
    
    try:
        # 获取文件描述符
        fd = fifo_read.fileno()
        # 使用select检查是否有数据可读，超时时间为0（非阻塞）
        ready, _, _ = select.select([fd], [], [], 0)
        return len(ready) > 0
    except:
        return False

def read_array_from_pipe(fifo_read):
    try:
        num_dims_data = fifo_read.read(8); num_dims = struct.unpack('q', num_dims_data)[0]
        shape_data = fifo_read.read(8 * num_dims); shape = struct.unpack(f'{num_dims}q', shape_data)
        num_elements = np.prod(shape) if shape else 0; data_bytes = int(num_elements * 8)
        flat_data = fifo_read.read(data_bytes)
        array = np.frombuffer(flat_data, dtype=np.float64).reshape(shape, order='F')
        
        if not array.flags.writeable:
            array = array.copy()
        
        return array
    except Exception as e:
        sys.stderr.write(f"CRITICAL ERROR in read_array_from_pipe: {e}\n"); traceback.print_exc(file=sys.stderr)
        return None

def read_latest_data_only(fifo_read):
    """
    读取管道中的所有数据，但只返回最新的一组
    这样可以确保总是处理最新的数据，丢弃积压的旧数据
    MODIFIED: Protocol is now index (Int64) -> X_array -> y_array
    """
    datasets = []  # 存储所有读取到的数据组
    
    sys.stdout.write("Python: Checking for available data in pipe...\n")
    sys.stdout.flush()
    
    # 持续读取直到管道为空
    while check_pipe_data_available(fifo_read):
        try:
            # 读取一组完整的数据：global_index + X + y
            sys.stdout.write("Python: Reading one data set from pipe...\n")
            sys.stdout.flush()
            
            # 1. 读取 global_index
            index_data = fifo_read.read(8)
            if len(index_data) != 8:
                sys.stdout.write("Python: Incomplete index data, stopping read.\n")
                break
            global_index = struct.unpack('q', index_data)[0]
            
            # 2. 读取X矩阵
            X_np = read_array_from_pipe(fifo_read)
            if X_np is None:
                sys.stdout.write("Python: Failed to read X matrix, stopping read.\n")
                break
                
            # 3. 读取y向量
            y_np = read_array_from_pipe(fifo_read)
            if y_np is None:
                sys.stdout.write("Python: Failed to read y vector, stopping read.\n")
                break
            
            # 存储这组数据
            datasets.append({
                'index': global_index, # <-- Storing the index
                'X': X_np,
                'y': y_np
            })
            
            sys.stdout.write(f"Python: Successfully read dataset with index #{global_index} (X: {X_np.shape}, y: {y_np.shape})\n")
            sys.stdout.flush()
            
        except Exception as e:
            sys.stdout.write(f"Python: Error reading data set: {e}, stopping read\n") 
            sys.stdout.flush()
            break
    
    if len(datasets) == 0:
        sys.stdout.write("Python: No complete data available in pipe.\n")
        sys.stdout.flush()
        return None
    elif len(datasets) == 1:
        latest_data = datasets[0]
        sys.stdout.write(f"Python: Found 1 dataset, processing it (index #{latest_data['index']}).\n")
        sys.stdout.flush()
        return latest_data
    else:
        # 有多组数据，只返回最新的，丢弃旧的
        latest_data = datasets[-1]
        sys.stdout.write(f"Python: Found {len(datasets)} datasets, DISCARDING {len(datasets)-1} old datasets, processing only the latest one (index #{latest_data['index']}).\n")
        sys.stdout.flush()
        
        # 手动清理丢弃的数据，释放内存
        # for i in range(len(datasets) - 1):
        #     del datasets[i]
        
        return latest_data


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
            
        sys.stdout.write(f"Sent string list with {list_length} strings\n")
        sys.stdout.flush()
        
        # 刷新数据
        try:
            fifo_write.flush()
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
        raise

def signal_result_ready(request_id):
    """创建信号文件告知Julia结果已准备好"""
    # Using a zero-padded format for the index
    signal_filename = f"python_result_ready_{request_id:06d}"
    try:
        with open(signal_filename, 'w') as f:
            f.write(str(time.time()))
        sys.stdout.write(f"Created signal file: {signal_filename}\n")
        sys.stdout.flush()
    except Exception as e:
        sys.stderr.write(f"Error creating signal file: {e}\n")

def main():
    sys.stdout = open(STDOUT_LOG_FILE, 'w', buffering=1) # Use line buffering
    sys.stderr = open(ERROR_LOG_FILE, 'w', buffering=1) # Use line buffering
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
            # request_count is no longer needed, as we use the index from Julia
            
            while True:
                sys.stdout.write("\n" + "="*50 + "\n")
                sys.stdout.write("Python: Waiting for new job...\n")
                sys.stdout.flush()
                
                try:
                    # 关键改变：读取最新数据，丢弃积压数据
                    latest_data = read_latest_data_only(fifo_read)
                    
                    if latest_data is None:
                        # 没有数据，等待一小段时间再检查
                        time.sleep(0.1)
                        continue
                    
                    # 提取数据和索引
                    global_index = latest_data['index'] 
                    X_np = latest_data['X']
                    y_np = latest_data['y']
                    
                    # 转换为torch张量并处理
                    sys.stdout.write(f"Processing job with index #{global_index}: X shape {X_np.shape}, y shape {y_np.shape}\n")
                    
                    if not X_np.flags.writeable:
                        X_np = X_np.copy()
                    if not y_np.flags.writeable:
                        y_np = y_np.copy()
                    
                    X_torch = torch.from_numpy(X_np).to('cuda')
                    y_torch = torch.from_numpy(y_np).to('cuda')
                    sys.stdout.write(f"Successfully converted to CUDA tensors for job #{global_index}.\n")
                    sys.stdout.flush()
                    
                    # 进行PSRN处理
                    psrn.current_expr_ls = variables_name
                    
                    expr_best_ls, MSE_min_ls = psrn.get_best_expr_and_MSE_topk(X_torch, y_torch, n_top)
                    sys.stdout.write(f"Job #{global_index} completed. First expression: {expr_best_ls[0] if expr_best_ls else 'None'}\n")
                    
                    # 手动清理torch张量和数据字典
                    del X_torch, y_torch
                    del latest_data
                    gc.collect() # Trigger garbage collection
                    
                    # 发送结果
                    sys.stdout.write(f"About to send string list to Julia for job #{global_index}...\n")
                    sys.stdout.flush()
                    send_string_list(fifo_write, expr_best_ls)
                    sys.stdout.write(f"Finished sending string list to Julia for job #{global_index}\n")
                    sys.stdout.flush()
                    
                    # 创建信号文件通知Julia结果已准备好
                    signal_result_ready(global_index)
                    
                except BrokenPipeError:
                    sys.stdout.write("Broken pipe detected, Julia process may have terminated\n")
                    break
                except Exception as loop_error:
                    sys.stderr.write(f"Error in processing loop: {loop_error}\n")
                    traceback.print_exc(file=sys.stderr)
                    continue
                
    except Exception as e:
        sys.stderr.write(f"Python script crashed in main loop: {e}\n")
        traceback.print_exc(file=sys.stderr)
    finally:
        sys.stdout.close()
        sys.stderr.close()

if __name__ == "__main__":
    main()