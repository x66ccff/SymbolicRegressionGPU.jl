import os
import struct
import sys
import traceback

import torch

JULIA_TO_PYTHON_PIPE = 'julia_to_python_pipe'
PYTHON_TO_JULIA_PIPE = 'python_to_julia_pipe'
ERROR_LOG_FILE = 'python_errors.log'

def cpu_intensive_calculation(value):
    """Simulates a CPU-intensive task for a few seconds."""
    sys.stdout.write(f"Python: Received {value}, starting calculation...\n")
    sys.stdout.flush()
    # time.sleep(2) # Simulate work
    # result = value + 1.0
    for i in range(100):
        result = value + torch.randn((1, 10_0000_0000), device='cuda')
    result = result[0, 0].item()
    sys.stdout.write(f"Python: Calculation finished. Sending back {result}\n")
    sys.stdout.flush()
    return result

def main():
    # Use standard I/O for pipes
    sys.stdout = open('python_stdout.log', 'w')
    sys.stderr = open(ERROR_LOG_FILE, 'w')

    # Open pipes in binary mode
    try:
        with open(JULIA_TO_PYTHON_PIPE, 'rb') as fifo_read, \
             open(PYTHON_TO_JULIA_PIPE, 'wb') as fifo_write:
            
            sys.stdout.write("Python process started and listening...\n")
            sys.stdout.flush()
            
            while True:
                # Blocking read for one double (8 bytes)
                data = fifo_read.read(8)
                if not data:
                    sys.stdout.write("Pipe closed from Julia side. Exiting.\n")
                    sys.stdout.flush()
                    break
                
                value = struct.unpack('d', data)[0]
                
                result = cpu_intensive_calculation(value)
                
                packed_result = struct.pack('d', result)
                fifo_write.write(packed_result)
                fifo_write.flush()

    except Exception as e:
        sys.stderr.write(f"Python script crashed: {e}\n")
        sys.stderr.write(traceback.format_exc())
        sys.stderr.flush()

if __name__ == "__main__":
    import time
    main()
