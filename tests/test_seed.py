import torch
import torch.nn as nn
from nbr.utils.seed import seed_everything

# Test 1: Check that seeding works
seed_everything(42)
x1 = torch.rand(3, 3)
y1 = torch.rand(3, 3)
z1 = torch.randint(0, 10, (3, 3))

seed_everything(42)
x2 = torch.rand(3, 3)
y2 = torch.rand(3, 3)
z2 = torch.randint(0, 10, (3, 3))

assert torch.equal(x1, x2), "Random tensors should be equal"
assert torch.equal(y1, y2), "Random tensors should be equal"
assert torch.equal(z1, z2), "Random integers should be equal"

# Test 2: Check model initialization consistency
seed_everything(123)
model1 = nn.Linear(10, 5)
w1 = model1.weight.clone()
b1 = model1.bias.clone()

seed_everything(123)
model2 = nn.Linear(10, 5)
w2 = model2.weight.clone()
b2 = model2.bias.clone()

assert torch.equal(w1, w2), "Model weights should be equal"
assert torch.equal(b1, b2), "Model biases should be equal"

print("All seed tests passed!")
