import torch
import numpy as np

batch_size = 4
dm_length = 10
t = np.random.randint(dm_length - 1, size=batch_size)
print(t)
index = [[j for j in range(i)] for i in t]
index_next = [[j for j in range(i)] for i in t+1]
summation_vec = np.zeros([batch_size, dm_length - 1])
summation_vec_next = np.zeros([batch_size, dm_length - 1])
for i in range(batch_size):
    summation_vec[i, index[i]] = 1
    summation_vec_next[i, index_next[i]] = 1
print(summation_vec)
print(summation_vec_next)
input_tensor = torch.stack((torch.from_numpy(summation_vec), torch.from_numpy(summation_vec_next)), dim=1)
print(input_tensor)
sum_p = torch.randn(dm_length - 1, dtype=torch.float64)

print('-'*57)

print(sum_p)
out = torch.einsum('i,bji->bj', sum_p, input_tensor)
print(out.shape)
s_0, s_1 = out.chunk(2, dim=1)
print(s_0)
for i in t:
    print(sum_p[:i].sum())
print(s_1)
for i in t:
    print(sum_p[:i+1].sum())