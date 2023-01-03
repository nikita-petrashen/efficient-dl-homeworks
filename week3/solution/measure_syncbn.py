import os
import time
import csv
import torch
import torch.distributed as dist
from torch.nn import SyncBatchNorm
from syncbn import SyncBatchNorm as MySyncBatchNorm


def dummy_loss(x):
    return x[:x.shape[0] // 2].sum()


def syncbn_forward_backward(bn, x, batch_size, rank, size):
    slice_size = (batch_size + size - 1) // size
    batch_slice = slice(rank * slice_size, min((rank + 1) * slice_size, batch_size))
    out = bn(x[batch_slice])
    loss = dummy_loss(out)
    loss.backward()


def syncbn_forward_backward(rank, bn, size, num_features, batch_size, device, prefix, num_iters=100):
    backend = "gloo"
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(29512)
    dist.init_process_group(backend, rank=rank, world_size=size)

    if rank == 0:
        times = []

    for _ in range(num_iters):
        x = torch.randn(batch_size, num_features, device=device, requires_grad=True)
        if rank == 0:
            start = time.perf_counter()
        syncbn_forward_backward(bn, x, batch_size, rank, size)
        if device == "cuda":
            torch.cuda.synchronize()
        dist.barrier()
        if rank == 0:
            times.append(time.perf_counter()-start)

    if rank == 0:
        median_time = torch.median(torch.tensor(times)).item()
        print(f"median time of {prefix} = {median_time}")


def run_bn_multiproc(bn, size, batch_size, num_features, device, prefix):
    torch.multiprocessing.spawn(
        syncbn_forward_backward,
        args=(
        bn, size, num_features, batch_size, device, prefix),
        join=True,
        nprocs=size
    )


if __name__ == "__main__":
    device = "cuda"
    for size in [1, 2]:
        print(size)
        for batch_size in [32, 64]:
            print("\t", batch_size)
            for num_features in [128, 256]:
                print("\t\t", num_features)
                my_bn = MySyncBatchNorm(num_features)
                torch_bn = SyncBatchNorm(num_features, affine=False, eps=1e-5, momentum=0.1, device=device)
                run_bn_multiproc(my_bn, size, batch_size, num_features, device, "my bn")
                run_bn_multiproc(torch_bn, size, batch_size, num_features, device, "torch bn")


