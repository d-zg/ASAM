import torch
import torch.nn as nn

def compute_top_eigenvalues(model, input_data, target_data, loss_fn, num_eigenvalues):
    # Set the model to evaluation mode
    model.eval()

    # Move the input data and target data to the appropriate device (CPU or GPU)
    input_data = input_data.to(next(model.parameters()).device)
    target_data = target_data.to(next(model.parameters()).device)

    # Define the loss function
    def loss_fn_wrapper(input_data):
        output = model(input_data)
        loss = loss_fn(output, target_data)
        return loss

    # Compute the Hessian matrix
    hessian = torch.autograd.functional.hessian(loss_fn_wrapper, input_data)
    # Compute the eigenvalues and eigenvectors of the Hessian matrix
    eigenvalues, _ = torch.linalg.eigh(hessian)

    # Get the top eigenvalues
    top_eigenvalues = eigenvalues[-num_eigenvalues:]

    return top_eigenvalues

