import torch

# 0D scalar tensor , size empty torch.Size[]
s = torch.tensor(5)
print(s)
print(s.shape)

# 1D vector tensor 
v = torch.tensor([3,5])
print(v)
print(v.shape)

# 2D Matrix tensor
m = torch.tensor( [[1,2],[3,4]] , dtype=torch.long )
print(m)
print(m.shape)

# temel math

a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

c = a + b
print(c)

a = torch.tensor([1, 2, 3])
b = torch.tensor([2, 2, 2])

print(a * b)

a = torch.tensor([1., 2., 3.])
b = torch.tensor([4., 5., 6.])

print(torch.dot(a, b))


# indexing

#reshape

x = torch.arange(8)

y = x.reshape(4, 2)
print(y)


#squeze / unsqeueeze

x = torch.tensor([1, 2, 3])
print(x)
x = x.unsqueeze(0)   # batch dimension ekler
print(x)


x = x.squeeze()
print(x)

print("-"*80)
# Autograd

x = torch.tensor(2.0, requires_grad=True)

print(x)

y = x**2
print(y)

y.backward()

print(x.grad)


x = torch.tensor(3.0, requires_grad=True)

y = 2*x + 1
z = y**2

z.backward()

print(x.grad)
print(y.grad)

# mini linear regression

# y = w*x + b

w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

x = torch.tensor(2.0)
target = torch.tensor(5.0)

y = w*x + b
loss = (y - target)**2

loss.backward()

print(w.grad)
print(b.grad)

#-------------------------------

torch.manual_seed(42)

# Data
N = 100
x = torch.linspace(-3, 3, N).reshape(-1, 1)

noise = torch.randn(N, 1) * 2
y = 3*x**2 + 2*x + 1 + noise

print(x)

X = torch.cat([x**2, x], dim=1)
print(X)

w = torch.randn(2, 1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

lr = 0.01
epochs = 1000

for epoch in range(epochs):
    
    # Forward
    y_pred = X @ w + b
    
    # MSE Loss
    loss = torch.mean((y_pred - y)**2)
    
    # Backward
    loss.backward()
    
    # Update
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
    
    # Zero grad
    w.grad.zero_()
    b.grad.zero_()
    
    if epoch % 5 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")


print(w)
print(b)

#