#
# ===== FINAL, ROBUST PYTHON SCRIPT =====
# Re-introduces the calculation with better stability and logging.
#
import os
import struct
import sys
import traceback
import numpy as np
import torch

# ... (常量和 read_array_from_pipe 函数保持不变) ...
JULIA_TO_PYTHON_PIPE = 'julia_to_python_pipe'
PYTHON_TO_JULIA_PIPE = 'python_to_julia_pipe'
ERROR_LOG_FILE = 'python_errors.log'
STDOUT_LOG_FILE = 'python_stdout.log'

def read_array_from_pipe(fifo_read):
    # (This function is correct, no changes needed)
    try:
        num_dims_data = fifo_read.read(8); num_dims = struct.unpack('q', num_dims_data)[0]
        shape_data = fifo_read.read(8 * num_dims); shape = struct.unpack(f'{num_dims}q', shape_data)
        num_elements = np.prod(shape) if shape else 0; data_bytes = int(num_elements * 8)
        flat_data = fifo_read.read(data_bytes)
        array = np.frombuffer(flat_data, dtype=np.float64).reshape(shape, order='F')
        return array
    except Exception as e:
        sys.stderr.write(f"CRITICAL ERROR in read_array_from_pipe: {e}\n"); traceback.print_exc(file=sys.stderr)
        return None

def robust_gpu_calculation(value, tensor_size=1_000_000, iterations=100):
    """
    一个更健壮的 GPU 密集型任务版本。
    - tensor_size: 每个随机张量中的元素数。
    - iterations: 计算循环的次数。
    """
    sys.stdout.write(
        f"Python: Starting robust calculation with trigger={value}, "
        f"tensor_size={tensor_size}, iterations={iterations}\n"
    )
    # 打印初始显存
    if torch.cuda.is_available():
        sys.stdout.write(f"  Initial VRAM Used: {torch.cuda.memory_allocated() / 1e6:.2f} MB\n")
    sys.stdout.flush()

    result = torch.tensor(value, device='cuda', dtype=torch.float64)
    
    for i in range(iterations):
        try:
            # 创建随机张量并执行操作
            noise = torch.randn(tensor_size, device='cuda', dtype=torch.float64)
            # 使用加法，而不是原地操作，因为 result 是一个标量，noise是一个向量
            result += torch.mean(noise) # 使用 mean 更稳定
            
            # 清理中间变量
            del noise

            if (i + 1) % 20 == 0:
                # 每20次迭代打印一次进度并清理缓存
                sys.stdout.write(f"  Calculation progress: {i+1}/{iterations}\n")
                sys.stdout.flush()
                # 强制PyTorch释放未被引用的缓存，有助于防止显存碎片化
                torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            sys.stderr.write(f"FATAL: CUDA Out of Memory during iteration {i+1}. "
                             f"Attempted to allocate for a tensor of size {tensor_size}.\n")
            # 发生内存溢出时，无法继续，返回一个错误码
            return float('-inf') 
            
    final_result = result.item()
    
    # 打印最终显存
    if torch.cuda.is_available():
        sys.stdout.write(f"  Final VRAM Used: {torch.cuda.memory_allocated() / 1e6:.2f} MB\n")
    sys.stdout.write(f"Python: Calculation finished. Sending back {final_result}\n")
    sys.stdout.flush()
    return final_result

def main():
    # ... (stdout/stderr redirection, pipe creation is the same) ...
    sys.stdout = open(STDOUT_LOG_FILE, 'w'); sys.stderr = open(ERROR_LOG_FILE, 'w')
    if not torch.cuda.is_available(): sys.stdout.write("CUDA not available.\n"); return
    try:
        if not os.path.exists(JULIA_TO_PYTHON_PIPE): os.mkfifo(JULIA_TO_PYTHON_PIPE)
        if not os.path.exists(PYTHON_TO_JULIA_PIPE): os.mkfifo(PYTHON_TO_JULIA_PIPE)
        with open(JULIA_TO_PYTHON_PIPE, 'rb') as fifo_read, \
             open(PYTHON_TO_JULIA_PIPE, 'wb') as fifo_write:
            sys.stdout.write("Python ROBUST process started and listening...\n"); sys.stdout.flush()
            while True:
                sys.stdout.write("\nWaiting for new job...\n"); sys.stdout.flush()
                # 1. 接收数据
                trigger_data = fifo_read.read(8); trigger_value = struct.unpack('d', trigger_data)[0]
                X_np = read_array_from_pipe(fifo_read)
                y_np = read_array_from_pipe(fifo_read)
                if X_np is None or y_np is None: break
                
                # 2. 记录接收信息
                sys.stdout.write(f"Received X shape {X_np.shape}, y shape {y_np.shape}\n")
                X_torch = torch.from_numpy(X_np).to('cuda')
                y_torch = torch.from_numpy(y_np).to('cuda')
                sys.stdout.write(f"Successfully converted to CUDA tensors.\n"); sys.stdout.flush()

                # 3. 执行健壮的计算任务
                # !!! 从一个较小的值开始测试 !!!
                # 如果 100万 仍然崩溃, 尝试 100_000 或更小
                result = robust_gpu_calculation(trigger_value, tensor_size=1_000_000, iterations=100)
                
                # 4. 发送结果
                packed_result = struct.pack('d', result)
                fifo_write.write(packed_result)
                fifo_write.flush()
    except Exception as e:
        sys.stderr.write(f"Python script crashed in main loop: {e}\n")
        traceback.print_exc(file=sys.stderr)
    finally:
        sys.stdout.close(); sys.stderr.close()

if __name__ == "__main__":
    main()