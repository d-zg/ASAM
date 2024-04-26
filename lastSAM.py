import torch
import torch.nn as nn
import quadprog
import numpy as np

def get_parameters(module):
    return (param for param in module.parameters() if param.requires_grad)

def get_grad_dims(module):
    grad_dims = []
    for param in get_parameters(module):
        grad_dims.append(param.data.numel())
    return grad_dims

def store_grad(module, grads, grad_dims):
    grads.fill_(0.0)
    cnt = 0
    for param in get_parameters(module):
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg:en].copy_(param.grad.data.view(-1))
        cnt += 1

def get_grads_in_p_vector(module):
    grad_dims = get_grad_dims(module)
    grads = torch.zeros(sum(grad_dims))
    store_grad(module, grads, grad_dims)
    return grads

def project2cone2(gradient, memories, margin=0.5, eps=1e-3):
    """
        Solves the GEM dual QP described in the GEM  paper given a proposed
        gradient "gradient", and a memory of task gradients "memories".
        Overwrites "gradient" with the final projected update.

        input:  gradient, p-vector
        input:  memories, (t * p)-vector
        output: x, p-vector
    """
    memories_np = memories.cpu().t().double().numpy()
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
    t = memories_np.shape[0]
    P = np.dot(memories_np, memories_np.transpose())
    P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
    q = np.dot(memories_np, gradient_np) * -1
    G = np.eye(t)
    h = np.zeros(t) + margin
    v = quadprog.solve_qp(P, q, G, h)[0]
    x = np.dot(v, memories_np) + gradient_np
    gradient.copy_(torch.Tensor(x).view(-1, 1))

def orthogonalize_grads(model, grads2):
    """
    Args:
        model: The PyTorch model whose gradients will be orthogonalized.
        grads2: the grads we will orthogonalize with respect to
    
    Steps:
    1. Compute the gradients of the model parameters with respect to the given loss.
    2. Perform a second backward pass to compute the gradients of the model with respect to the normal loss.
    3. Store the gradients of the model with respect to the normal loss.
    4. Compute the projection of the original gradients onto the gradients of the normal loss.
    5. Subtract the projection from the original gradients to make them orthogonal to the gradients of the normal loss.
    6. Update the model gradients with the orthogonal gradients.
    
    Note: The model gradients are modified in-place.
    """
    grads1 = torch.cat([param.grad.view(-1) for param in model.parameters() if param.grad is not None])

   
    # compute the projection of the original grads onto grads2 
    projection = torch.dot(grads1, grads2) / torch.dot(grads2, grads2)
    projected_grads1 = projection * grads2

    # subtract the projection from grads1 to make it orthogonal to grads2
    orthogonal_grads1 = grads1 - projected_grads1

    # update the model gradients with the orthogonal gradients
    index = 0
    for param in model.parameters():
        if param.grad is not None:
            num_params = param.numel()
            param.grad.copy_(orthogonal_grads1[index:index+num_params].view_as(param.grad))
            index += num_params
