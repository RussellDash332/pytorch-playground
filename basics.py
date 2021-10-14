import torch
import numpy as np

# Creating a tensor
arr = [[1, 2], [3, 4]]
tensor = torch.Tensor(arr)
print(tensor)

# Predefined values
print(torch.ones((2, 2)))
print(torch.randn(2, 2))

# Random seed
torch.manual_seed(0)
print(torch.randn(2, 2))

# From NumPy
np_array = np.ones((2, 2))
torch_tensor = torch.from_numpy(np_array)
print(torch_tensor)

# CPU-GPU
tensor_cpu = torch.ones(2, 2)
if torch.cuda.is_available():
    tensor_cpu.cuda()
print(tensor_cpu.cpu())

# Viewing
print(tensor.view(4, 1))
print(tensor.view(-1, 4))
print(tensor.view(2,-1))
print(tensor.size())

# Arithmetic Operations
a = torch.ones((2, 2))
b = torch.ones((2, 2))
c = a + b
c = a.add(b)
c = torch.add(a, b)
c.add_(b) # in-place addition
# replace add with sub, mul, div

# Mean
tensor = torch.Tensor([[1,2,3,4,5,6,7,8,9,10], [1,2,3,4,5,6,7,8,9,10]])
print(tensor.mean(dim=0))
print(tensor.mean(dim=1))
