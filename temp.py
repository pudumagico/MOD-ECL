import torch
import numpy as np

const_path="./constraints/constraints.npy"
const = np.load(const_path)
print(const.shape)
#np.save(const_path, const)
#print(np.load(const_path))
for i in range(const.shape[0]):
    if const[i].sum() == 0:
        print(i)
exit()
constraints = torch.from_numpy(np.load(const_path)).to_sparse()

for req_id in range(constraints.shape[0]):
    print(constraints.indices()[1][constraints.indices()[0]==req_id])