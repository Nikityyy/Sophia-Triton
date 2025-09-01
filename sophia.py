# Rewritten sophia.py to match the API of sophia_triton.py

import torch
from torch.optim.optimizer import Optimizer

class SophiaG(Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.965, 0.99), rho=0.04,
                 weight_decay=1e-1, *, maximize: bool = False,
                 capturable: bool = False, eps: float = 1e-15, bs: int):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= rho:
            raise ValueError(f"Invalid rho parameter: {rho}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not bs > 0:
            raise ValueError(f"Invalid batch size (bs): {bs}")

        defaults = dict(lr=lr, betas=betas, rho=rho, weight_decay=weight_decay,
                        maximize=maximize, eps=eps, bs=bs)
        super(SophiaG, self).__init__(params, defaults)

    def _init_state(self, p):
        """Initializes optimizer state for a parameter."""
        state = self.state[p]
        if len(state) == 0:
            state['step'] = 0
            state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
            state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)

    @torch.no_grad()
    def update_hessian(self):
        """
        Initializes the optimizer state. This should be called before the
        backward pass of the first training step. In subsequent steps,
        it ensures state is ready and handles cases where parameter shapes might change.
        """
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]

                if len(state) > 0 and state['exp_avg'].shape != p.shape:
                    print(f"SophiaG: Detected shape mismatch for a parameter (state: {state['exp_avg'].shape}, param: {p.shape}). Re-initializing state.")
                    state.clear()
                    
                self._init_state(p)

    @torch.no_grad()
    def schedule_hessian_update(self):
        """
        Updates the Hessian EMA. This method should be called after `loss.backward()`
        and before `optimizer.step()`. It computes the diagonal Hessian estimate
        using the current gradients.
        """
        for group in self.param_groups:
            _, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('SophiaG does not support sparse gradients')

                state = self.state[p]
                
                if len(state) == 0:
                    raise RuntimeError(f"SophiaG: State not initialized for parameter with shape {p.shape}, but it has a gradient. Ensure `optimizer.update_hessian()` is called before `backward()`.")

                state['hessian'].mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that re-evaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, _ = group['betas']
            lr = group['lr']
            weight_decay = group['weight_decay']
            rho = group['rho']
            eps = group['eps']
            bs = group['bs']
            maximize = group['maximize']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                if maximize:
                    grad = -grad
                
                if grad.is_sparse:
                    raise RuntimeError('SophiaG does not support sparse gradients')

                state = self.state[p]
                if len(state) == 0:
                    raise RuntimeError("Optimizer state not initialized. Call update_hessian() before the first step().")

                state['step'] += 1
                
                exp_avg = state['exp_avg']
                hessian = state['hessian']

                if weight_decay != 0:
                    p.mul_(1.0 - lr * weight_decay)

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                
                denominator = (rho * bs * hessian).clamp_(min=eps)
                
                ratio = (exp_avg.abs() / denominator).clamp_(max=1.0)
                
                p.addcmul_(exp_avg.sign(), ratio, value=-lr)
                
        return loss
