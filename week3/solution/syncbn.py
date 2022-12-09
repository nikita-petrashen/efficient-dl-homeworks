import torch
import torch.distributed as dist
from torch.autograd import Function
from torch.nn.modules.batchnorm import _BatchNorm


class sync_batch_norm(Function):
    """
    A version of batch normalization that aggregates the activation statistics across all processes.

    This needs to be a custom autograd.Function, because you also need to communicate between processes
    on the backward pass (each activation affects all examples, so loss gradients from all examples affect
    the gradient for each activation).

    For a quick tutorial on torch.autograd.function, see
    https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
    """

    @staticmethod
    def forward(ctx, input, running_mean, running_std, eps: float, momentum: float):
        # Compute statistics, sync statistics, apply them to the input
        # Also, store relevant quantities to be used on the backward pass with `ctx.save_for_backward`
        batch_sum = torch.sum(input, dim=0)
        batch_squared_sum = torch.sum(input**2, dim=0)
        batch_num_elem = torch.tensor(input.shape[0], dtype=input.dtype, device=input.device)
        combined = torch.cat([batch_sum, batch_squared_sum, batch_num_elem], dim=0)
        dist.all_reduce(combined, op=dist.ReduceOp.SUM)
        all_sum, all_squared_sum, all_num_elem = torch.split(combined, [input.shape[1], input.shape[1], 1], dim=0)
        mean = all_sum / all_num_elem
        var = all_squared_sum / all_num_elem - mean ** 2 + eps
        std = torch.sqrt(var)
        centered = input - mean
        normalized = (input - mean) / std

        ctx.save_for_backward(var, centered)
        running_mean.data = (running_mean * momentum + mean * (1 - momentum)).data
        running_std.data = (running_std * momentum + std * (1 - momentum)).data

        return normalized

    @staticmethod
    def backward(ctx, grad_output):
        # don't forget to return a tuple of gradients wrt all arguments of `forward`!
        var, centered = ctx.saved_tensors
        batch_grad_sum = grad_output.sum(dim=0)
        batch_grad_cent_sum = (grad_output * centered).sum(dim=0)
        batch_num_elem = torch.tensor(grad_output.shape[0], dtype=grad_output.dtype, device=grad_output.device)
        combined = torch.cat([batch_grad_sum, batch_grad_cent_sum, batch_num_elem], dim=0)
        dist.all_reduce(combined, op=dist.ReduceOp.SUM)
        all_grad_sum, all_grad_cent_sum, all_num_elem = torch.split(combined, [grad_output.shape[1], grad_output.shape[1], 1])
        grad_input = (grad_output - (all_grad_cent_sum / var * centered + all_grad_sum) / all_num_elem) / torch.sqrt(var)

        return grad_input, None, None, None, None


class SyncBatchNorm(_BatchNorm):
    """
    Applies Batch Normalization to the input (over the 0 axis), aggregating the activation statistics
    across all processes. You can assume that there are no affine operations in this layer.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__(
            num_features,
            eps,
            momentum,
            affine=False,
            track_running_stats=True,
            device=None,
            dtype=None,
        )
        self.running_mean = torch.zeros((num_features,))
        self.running_std = torch.ones((num_features,))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.running_mean = self.running_mean.to(input.device)
        self.running_std = self.running_std.to(input.device)
        if self.training:
            out = sync_batch_norm.apply(input, self.running_mean, self.running_std, self.eps, self.momentum)
        else:
            out = (input - self.running_mean) / self.running_std

        return out
