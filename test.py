import sys
print("Hello world!")

import torch
import time

if torch.cuda.is_available():
    print("CUDA is available")
    print("Number of GPUs available:", torch.cuda.device_count())
    print("GPU name:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available")

size = 1024
device = torch.device("cuda")

x = torch.rand(size, size).to(device)
y = torch.rand(size, size).to(device)

start = time.time()
result = torch.mm(x, y)
end = time.time()

print("Time taken for mm operation on GPU:", end - start, "seconds")