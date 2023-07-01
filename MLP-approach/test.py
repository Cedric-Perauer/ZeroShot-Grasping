import torch 

def check_and_remove_tensors(a, b):
    # Calculate element-wise inequality
    unequal_tensors = torch.all(torch.ne(a[:, None], b), dim=2)

    # Find indices where no tensor in a matches with any tensor in b
    indices_to_keep = torch.all(unequal_tensors, dim=0)

    # Filter tensors in b based on the indices
    filtered_b = b[indices_to_keep]

    return filtered_b

def check_and_remove_tensors2(a, b):
    for tensor_a in a:
        for i, tensor_b in enumerate(b):
            if torch.all(torch.eq(tensor_a, tensor_b)):
                b = torch.cat((b[:i], b[i+1:]), dim=0)
                break

    return b

a = torch.Tensor([[1,2],[3,4],[5,6]])

b = torch.Tensor([[1,2],[3,4],[5,6],[7,8],[9,10]])

print(a.shape,b.shape)
print(check_and_remove_tensors2(a,b))