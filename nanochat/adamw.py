"""
This module handles distributed AdamW in ZeRo-2 Style
"""
import torch
from torch import Tensor
import torch.distributed as dist

def adamw_step_fused(
    p: Tensor,
    grad: Tensor,
    m: Tensor,
    v: Tensor,
    step_t: Tensor,
    lr_t: Tensor,
    beta1_t: Tensor,
    beta2_t: Tensor,
    eps_t: Tensor,
    wd_t: Tensor,
):
    p.mul_(1 - lr_t * wd_t)
    # torch.Tensor.lerp(end, weight) --> start + (end - start) * weight
    m.lerp_(grad, 1- beta1_t)
    v.lerp_(grad * grad, 1 - beta2_t)
    m_hat = m / (1 - beta1_t ** step_t)
    v_hat = v / (1 - beta2_t ** step_t)
    p.add_(m_hat / (v_hat.sqrt() + eps_t), alpha=-lr_t)


class DistAdamW(torch.optim.Optimizer):
    def __init__(self, param_groups, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if rank == 0:
            for group in param_groups:
                assert isinstance(group, dict), f"Expect param_groups to be a list of dict, instead got list of {type(group)}"
                assert isinstance(group['params'], list), "Expect group['params'] to be a list of tensors"
                for p in group['params']:
                    if p.numel() >= 1024:
                        assert p.shape[0] % world_size == 0, f"First dimention of parameter must be devisible by world size {world_size}. Instead got {p.shape}"
        super().__init__(param_groups, defaults)

        # 0-D CPU tensors to avoid torch.compile recompilation when values change
        # First create the tensor, so even the value is changed, re-compulation is not needed. Memory is preserved.
        self._step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
    
    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        step_items = []
        for group in self.param_groups:
            group_items = []
            for p in group['params']:
                is_large = True
                grad = p.grad
                if p.numel() <= 1024:
                    p_slice = p
                    future = dist.all_reduce(grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                    g_slice= grad
                    is_large = False
                else:
                    rank_size = grad.shape[0] // world_size
                    p_slice = p[rank * rank_size: (rank + 1) * rank_size]
                    g_slice = torch.empty_like(p_slice)
                    future = dist.reduce_scatter_tensor(g_slice, grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                group_items.append((p, p_slice, g_slice, future, is_large))
            step_items.append((group, group_items))

        gather_futures: list[torch.Future] = []
        for group, group_items in step_items:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']

            self._lr_t.fill_ = lr
            self._beta1_t.fill_ = beta1
            self._beta2_t.fill_ = beta2
            self._eps_t.fill_ = eps
            self._wd_t.fill_ = wd

            for p, p_slice, g_slice, future, is_large in group_items:
                future.wait()
                state = self.state[p]
                if not state:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p_slice)
                    state['v'] = torch.zeros_like(p_slice)
                state['step'] += 1
                step = state['step']
                m = state['m']
                v = state['v']
                self._step_t.fill_ = state['step']
                adamw_step_fused(
                    p_slice, g_slice, m, v,
                    self._step_t, self._lr_t, self._beta1_t, self._beta2_t, self._eps_t, self._wd_t,
                )
                if is_large:
                    gather_futures.append(dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future())
        
        if gather_futures:
            torch.futures.collect_all(gather_futures).wait()


            