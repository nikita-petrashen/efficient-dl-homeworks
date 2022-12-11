import pytest
import os
import torch
import torch.distributed as dist
from syncbn import SyncBatchNorm
from torch.multiprocessing import Process, Queue
from torch.nn import BatchNorm1d


def dummy_loss(x):
    return x[:x.shape[0] // 2].sum()


def _run_sync_bn(sync_bn, x, q, rank, size, master_port, backend="gloo"):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(master_port)
    dist.init_process_group(backend, rank=rank, world_size=size)
    x.requires_grad_(True)

    out_sync = sync_bn(x)
    loss = dummy_loss(out_sync)
    loss.backward()
    grad_sync = x.grad

    # make sure elements are enqueued in correct order
    for _ in range(rank):
        dist.barrier()

    q.put((out_sync.detach().numpy(), grad_sync.detach().numpy()))
    for _ in range(size - rank):
        dist.barrier()


def run_sync_bn(sync_bn, x, size):
    x_list = torch.chunk(x, size, dim=0)
    processes = []
    q = Queue()
    port = 29500
    for rank, x_mini in enumerate(x_list):
        p = Process(target=_run_sync_bn, args=(sync_bn, x_mini, q, rank, size, port))
        p.start()
        processes.append(p)

    out = []
    in_grad = []
    for _ in range(size):
        elem = q.get()
        out.append(torch.tensor(elem[0]))
        in_grad.append(torch.tensor(elem[1]))

    for p in processes:
        p.join()

    return torch.cat(out, dim=0), torch.cat(in_grad, dim=0)


def collect_bn_out_and_grad(bn, x, size, batch_size):
    x.requires_grad_(True)
    out = bn(x)
    loss = 0
    slice_size = (batch_size + size - 1) // size
    # compute loss as it is computed in distributed case
    for batch_index in range(size):
        batch_slice = slice(batch_index * slice_size, min((batch_index + 1) * slice_size, batch_size))
        loss += dummy_loss(out[batch_slice])
    loss.backward()
    in_grad = x.grad

    return out, in_grad


@pytest.mark.parametrize("size", [1, 2, 3, 4])
@pytest.mark.parametrize("batch_size", [32, 64])
@pytest.mark.parametrize("num_features", [128, 256, 512, 1024])
def test_sync_bn(size, num_features, batch_size):
    torch.multiprocessing.set_start_method("spawn", force=True)
    bn = BatchNorm1d(num_features, affine=False, eps=1e-5, momentum=0.1)
    sync_bn = SyncBatchNorm(num_features)
    x = torch.randn(batch_size, num_features)

    sync_out, sync_in_grad = run_sync_bn(sync_bn, x, size)
    out, in_grad = collect_bn_out_and_grad(bn, x, size, batch_size)

    torch.testing.assert_close(sync_out, out, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(sync_in_grad, in_grad, atol=1e-3, rtol=1e-3)
