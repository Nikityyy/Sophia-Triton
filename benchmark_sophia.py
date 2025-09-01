import torch
from torch.optim import AdamW

import sophia as sophia_ref
import sophia_triton

try:
    import matplotlib.pyplot as plt
    import numpy as np
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False

# --- Configuration ---

NUM_PARAMS = 10
DIMS = (4096, 4096)
LR = 2e-4
RHO = 0.04
BS = 4096
STEPS = 100
WARMUP_STEPS = 10
CORRECTNESS_STEPS = 5
TOLERANCES = {
    torch.float32: 3e-6,
    torch.bfloat16: 1e-2,
    torch.float16: 1e-3,
}

# --- Utility Functions ---

def create_model_params(num_params, dim1, dim2, dtype, device):
    """Creates a list of tensors to simulate model parameters."""
    return [torch.randn(dim1, dim2, dtype=dtype, device=device, requires_grad=True) for _ in range(num_params)]

def get_total_params(params):
    """Calculates the total number of elements in a list of parameters."""
    return sum(p.numel() for p in params)

# --- Core Benchmark Logic ---

def check_correctness(params_ref, params_triton, optimizer_ref, optimizer_triton, dtype, device, tolerance):
    """
    Compares the Triton SophiaG implementation against the reference for numerical correctness.
    """
    print("\n[1] Correctness Check (Reference vs. Triton)...")
    print(f"  Using correctness tolerance: {tolerance}")
    diff_history = {'param': [], 'exp_avg': [], 'hessian': []}

    for step in range(CORRECTNESS_STEPS):
        torch.manual_seed(step + 1)
        grads_fp32 = [torch.randn_like(p, dtype=torch.float32) for p in params_ref]
        grads_ref = [g.to(dtype) for g in grads_fp32]
        grads_triton = [g.clone().to(dtype) for g in grads_fp32]

        for i in range(NUM_PARAMS):
            params_ref[i].grad = grads_ref[i]
            params_triton[i].grad = grads_triton[i]

        optimizer_ref.schedule_hessian_update()
        optimizer_ref.step()

        optimizer_triton.schedule_hessian_update()
        optimizer_triton.update_hessian()
        optimizer_triton.step()

        max_param_diff, max_exp_avg_diff, max_hessian_diff = 0, 0, 0
        for i in range(NUM_PARAMS):
            p_ref, p_tri = params_ref[i], params_triton[i]
            s_ref, s_tri = optimizer_ref.state[p_ref], optimizer_triton.state[p_tri]

            max_param_diff = max(max_param_diff, torch.max(torch.abs(p_ref.float() - p_tri.float())).item())
            max_exp_avg_diff = max(max_exp_avg_diff, torch.max(torch.abs(s_ref['exp_avg'].float() - s_tri['exp_avg'].float())).item())
            max_hessian_diff = max(max_hessian_diff, torch.max(torch.abs(s_ref['hessian'].float() - s_tri['hessian'].float())).item())

        diff_history['param'].append(max_param_diff)
        diff_history['exp_avg'].append(max_exp_avg_diff)
        diff_history['hessian'].append(max_hessian_diff)
        print(f"  Step {step+1}: Max Diffs | Param: {max_param_diff:.2e} | ExpAvg: {max_exp_avg_diff:.2e} | Hessian: {max_hessian_diff:.2e}")

    if diff_history['param'][-1] > tolerance:
        print(f"  [!] Correctness check FAILED: Final parameter difference ({diff_history['param'][-1]:.2e}) exceeds tolerance ({tolerance})")
    else:
        print("  [✓] Correctness check PASSED.")

def run_performance_step(optimizer, params, name):
    """
    Benchmarks a single optimizer's performance over several steps.
    """
    for _ in range(WARMUP_STEPS):
        grads = [torch.randn_like(p) for p in params]
        for i in range(len(params)): params[i].grad = grads[i]
        
        if "AdamW" in name:
            optimizer.step()
        elif "Reference" in name:
            optimizer.schedule_hessian_update()
            optimizer.step()
        elif "Triton" in name:
            optimizer.schedule_hessian_update()
            optimizer.update_hessian()
            optimizer.step()

    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    for _ in range(STEPS):
        grads = [torch.randn_like(p) for p in params]
        for i in range(len(params)): params[i].grad = grads[i]
        
        if "AdamW" in name:
            optimizer.step()
        elif "Reference" in name:
            optimizer.schedule_hessian_update()
            optimizer.step()
        elif "Triton" in name:
            optimizer.schedule_hessian_update()
            optimizer.update_hessian()
            optimizer.step()
    
    end_event.record()
    torch.cuda.synchronize()
    
    elapsed_time_ms = start_event.elapsed_time(end_event)
    time_per_step = elapsed_time_ms / STEPS
    return time_per_step
    
def plot_performance_results(results, filename):
    """Saves a bar chart of the performance results."""
    if not PLOT_AVAILABLE:
        print("\n[!] Matplotlib not found. Skipping plot generation.")
        return

    labels = list(results.keys())
    adamw_times = [res['adamw'] for res in results.values()]
    ref_times = [res['ref'] for res in results.values()]
    triton_times = [res['triton'] for res in results.values()]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 8))
    rects1 = ax.bar(x - width, adamw_times, width, label='AdamW (fused)')
    rects2 = ax.bar(x, ref_times, width, label='SophiaG (Reference)')
    rects3 = ax.bar(x + width, triton_times, width, label='SophiaG (Triton)')

    ax.set_ylabel('Time per Step (ms)')
    ax.set_title('Optimizer Performance Comparison (Lower is Better)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    fig.tight_layout()
    plt.savefig(filename)
    print(f"\n[✓] Performance benchmark graph saved to '{filename}'")


# --- Test Suite Orchestrator ---

def run_benchmark_for_dtype(dtype, device):
    """
    Runs the full correctness and performance benchmark suite for a given dtype.
    """
    print("-" * 80)
    print(f"Running Test for dtype={dtype} on {device}")
    print("-" * 80)

    torch.manual_seed(0)
    params_ref = create_model_params(NUM_PARAMS, DIMS[0], DIMS[1], dtype, device)
    params_triton = [p.clone().detach().requires_grad_(True) for p in params_ref]
    params_adamw = [p.clone().detach().requires_grad_(True) for p in params_ref]

    total_params = get_total_params(params_ref)
    print(f"Model parameters: {NUM_PARAMS} tensors of size {DIMS}, Total: {total_params / 1e9:.2f}B elements")

    optimizer_ref = sophia_ref.SophiaG(params_ref, lr=LR, rho=RHO, eps=1e-15, bs=BS)
    optimizer_triton = sophia_triton.SophiaG(params_triton, lr=LR, rho=RHO, eps=1e-15, bs=BS)
    
    optimizer_ref.update_hessian()
    optimizer_triton.update_hessian()

    try:
        optimizer_adamw = AdamW(params_adamw, lr=LR, fused=True)
        print("Using AdamW with fused=True")
    except RuntimeError:
        print("Fused AdamW not available. Using AdamW with fused=False.")
        optimizer_adamw = AdamW(params_adamw, lr=LR, fused=False)

    
    tolerance = TOLERANCES.get(dtype, 1e-5)
    check_correctness(params_ref, params_triton, optimizer_ref, optimizer_triton, dtype, device, tolerance)

    print("\n[2] Performance Benchmark...")
    time_ref = run_performance_step(optimizer_ref, params_ref, "SophiaG (Reference)")
    time_triton = run_performance_step(optimizer_triton, params_triton, "SophiaG (Triton)")
    time_adamw = run_performance_step(optimizer_adamw, params_adamw, "AdamW")

    print(f"  {'SophiaG (Reference)':<30}: {time_ref:.4f} ms/step")
    print(f"  {'SophiaG (Triton)':<30}: {time_triton:.4f} ms/step")
    print(f"  {'AdamW (Fused)':<30}: {time_adamw:.4f} ms/step")

    speedup_vs_ref = time_ref / time_triton if time_triton > 0 else float('inf')
    speedup_vs_adamw = time_adamw / time_triton if time_triton > 0 else float('inf')
    print(f"\nTriton SophiaG Speedup vs SophiaG Reference: {speedup_vs_ref:.2f}x")
    print(f"Triton SophiaG Speedup vs AdamW: {speedup_vs_adamw:.2f}x" if speedup_vs_adamw > 0 else f"Triton SophiaG is slower than AdamW by {time_triton / time_adamw:.2f}x")
    print("-" * 80)
    
    return time_adamw, time_ref, time_triton

# --- Main Execution ---

def main():
    """Main entry point for the benchmark script."""
    if not torch.cuda.is_available():
        print("CUDA is not available. Skipping benchmark.")
        return

    device = "cuda"
    performance_results = {}
    
    adamw_time, ref_time, triton_time = run_benchmark_for_dtype(torch.float32, device)
    performance_results['Float32'] = {'adamw': adamw_time, 'ref': ref_time, 'triton': triton_time}
    
    if torch.cuda.is_bf16_supported():
        adamw_time, ref_time, triton_time = run_benchmark_for_dtype(torch.bfloat16, device)
        performance_results['BFloat16'] = {'adamw': adamw_time, 'ref': ref_time, 'triton': triton_time}
    
    adamw_time, ref_time, triton_time = run_benchmark_for_dtype(torch.float16, device)
    performance_results['Float16'] = {'adamw': adamw_time, 'ref': ref_time, 'triton': triton_time}
    
    if performance_results:
        plot_performance_results(performance_results, "sophia_performance_benchmark.png")

if __name__ == "__main__":
    main()
