import torch
from torch.autograd import Variable

a = Variable(torch.ones((2, 2)), requires_grad=True)
b = Variable(torch.ones((2, 2)), requires_grad=True)
print(a + b)
print(a.add(b))

x = Variable(torch.ones(2), requires_grad=True)
y = 5 * (x + 1) ** 2
o = 1/2 * torch.sum(y)

o.backward()
print(x.grad)

p = torch.tensor([10, 10], requires_grad=True)
q = torch.tensor([20, 10], requires_grad=True)
R = p * q
R.backward(gradient=torch.tensor([1, 1]))
print(p.grad) # q
print(q.grad) # p
