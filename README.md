# fa4-compile-wrapper

**Make Flash Attention 4 work with `torch.compile` on NVIDIA B200 (Blackwell) GPUs.**

## Problem

[modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) uses Flash Attention 3 via the [`kernels`](https://github.com/pytorch-labs/kernels) package. FA3 is **Hopper-only** (sm_90) and crashes on B200 (sm_100) with:

```
CUDA error (flash-attention/hopper/flash_fwd_launch_template.h:164): no kernel image is available for execution on the device
```

[Flash Attention 4](https://github.com/Dao-AILab/flash-attention) (`flash-attn-4`) supports B200 natively, but **FA4 4.0.0b4 is not `torch.compile` compatible**. The modded-nanogpt training code requires `torch.compile` for performance.

### Failure Mode 1: Without `allow_in_graph`

`torch.compile`'s dynamo tracer tries to trace into FA4's internals. FA4's `_flash_attn_fwd` ([source](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/interface.py#L278)) calls `cuda.CUstream(torch.cuda.current_stream().cuda_stream)`, which dynamo can't handle:

```
torch._dynamo.exc.Unsupported: Attempted to call function marked as skipped
  Explanation: Dynamo does not know how to trace the builtin `<unknown module>.CUstream.__new__`
```

### Failure Mode 2: With `torch.compiler.allow_in_graph`

Dynamo tries to run the function with FakeTensors to infer output shapes. FA4 calls `from_dlpack()` on the FakeTensor, which tries to access `.data_ptr()`:

```
torch._dynamo.exc.TorchRuntimeError: Dynamo failed to run FX node with fake tensors
  RuntimeError: Cannot access data pointer of Tensor (e.g. FakeTensor, FunctionalTensor).
  If you're using torch.compile/export/fx, it is likely that we are erroneously tracing
  into a custom kernel. To fix this, please wrap the custom kernel into an opaque custom op.
```

## Solution

Wrap FA4's `flash_attn_varlen_func` as a [`torch.library.custom_op`](https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html) with a registered fake/meta implementation. This tells dynamo:

1. **Don't trace into this function** — it's opaque
2. **Here's how to compute output shapes** without running the real kernel (fake impl returns `torch.empty_like(q)`)
3. **Here's how autograd flows through it** (must support backward pass for training)

## What to Deliver

A single Python file `fa4_compile_wrapper.py` that provides a drop-in replacement for FA3's `flash_attn_varlen_func`.

### Import Stubs Required

FA4's package also installs FA2 bindings that are ABI-incompatible with torch 2.9.1. These must be stubbed before any `flash_attn` import:

```python
import sys, types
sys.modules["flash_attn_2_cuda"] = types.ModuleType("flash_attn_2_cuda")
sys.modules["flash_attn_3_cuda"] = types.ModuleType("flash_attn_3_cuda")
```

### FA4 Function to Wrap

**Source**: [`flash_attn.cute.interface.flash_attn_varlen_func`](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/interface.py#L1559)

Full signature:

```python
flash_attn_varlen_func(
    q: torch.Tensor,                           # (total_q, num_heads, head_dim), bfloat16
    k: torch.Tensor,                           # (total_k, num_heads_k, head_dim), bfloat16
    v: torch.Tensor,                           # (total_k, num_heads_k, head_dim), bfloat16
    cu_seqlens_q: Optional[torch.Tensor],      # (batch+1,), int32
    cu_seqlens_k: Optional[torch.Tensor],      # (batch+1,), int32
    max_seqlen_q: Optional[int],
    max_seqlen_k: Optional[int],
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    learnable_sink: Optional[torch.Tensor] = None,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    deterministic: bool = False,
    score_mod: Optional[Callable] = None,
    aux_tensors: Optional[list] = None,
    return_lse: bool = False,
) -> torch.Tensor  # same shape as q
```

**Only these args are used by modded-nanogpt** (call site at [train_gpt.py line 1135](https://github.com/KellerJordan/modded-nanogpt/blob/master/train_gpt.py#L1135)):

```python
y = flash_attn_interface.flash_attn_varlen_func(
    q[0], k[0], v[0],
    cu_seqlens_q=seqlens, cu_seqlens_k=seqlens,
    max_seqlen_q=max_len, max_seqlen_k=max_len,
    causal=True, softmax_scale=yarn.attn_scale,
    window_size=(bm_size, 0)
)
```

### Custom Op Requirements

1. **Forward pass**: Delegate to `flash_attn.cute.interface.flash_attn_varlen_func`
2. **Fake/meta impl**: Return `torch.empty_like(q)` (output has same shape/dtype as q)
3. **Autograd support**: This is training — backward pass is required. FA4's `flash_attn_varlen_func` internally uses `FlashAttnVarlenFunc(torch.autograd.Function)` which has its own backward. The wrapper must preserve this.
4. **`window_size` handling**: The real API takes a `Tuple[int, int]`, but `torch.library.custom_op` may not support tuple args. Split into `window_size_left: int` and `window_size_right: int`.
5. **Exported API**: Must expose a function with the same signature as FA3's `flash_attn_varlen_func` so the integration patch is minimal.

### Integration Patch for modded-nanogpt

Add to the **very top** of `train_gpt.py` (before any other imports):
```python
import sys, types
sys.modules["flash_attn_2_cuda"] = types.ModuleType("flash_attn_2_cuda")
sys.modules["flash_attn_3_cuda"] = types.ModuleType("flash_attn_3_cuda")
```

Replace [line 1062](https://github.com/KellerJordan/modded-nanogpt/blob/master/train_gpt.py#L1062):
```python
# OLD:
flash_attn_interface = get_kernel('varunneal/flash-attention-3').flash_attn_interface

# NEW:
import types as _types
from fa4_compile_wrapper import flash_attn_varlen_func as _fa4_fn
flash_attn_interface = _types.SimpleNamespace()
flash_attn_interface.flash_attn_varlen_func = _fa4_fn
```

### Reference: How FA3 Was Made torch.compile Compatible

modded-nanogpt loads FA3 via `get_kernel('varunneal/flash-attention-3')` from the [`kernels`](https://github.com/pytorch-labs/kernels) package. This package pre-compiles FA3 as a C extension with proper torch custom op registration, which is why it works with `torch.compile` out of the box. The same pattern needs to be applied to FA4, but in pure Python using `torch.library`.

## Test Environment

| Component | Value |
|-----------|-------|
| GPU | 8× NVIDIA B200 (sm_100, 180GB each) |
| Machine | GCP `a4-highgpu-8g` (`volta-b200-0`) |
| SSH | `ssh volta-b200-0` from kurtz |
| Torch | 2.9.1+cu128 |
| FA4 | flash-attn-4==4.0.0b4 |
| CUDA | 12.8 |
| Python | 3.12 |
| Docker image | `modded-nanogpt-fa4` on volta-b200-0 |
| Repo | `/mnt/data/modded-nanogpt/` (cloned at HEAD = record #77) |
| Data | `/mnt/data/modded-nanogpt/data/fineweb10B/` (already downloaded) |

### Verified Working

```bash
# FA4 imports and runs standalone on B200:
docker run --rm --gpus all modded-nanogpt-fa4 python -c "
import sys, types
sys.modules['flash_attn_2_cuda'] = types.ModuleType('flash_attn_2_cuda')
sys.modules['flash_attn_3_cuda'] = types.ModuleType('flash_attn_3_cuda')
from flash_attn.cute.interface import flash_attn_varlen_func
import torch
q = torch.randn(1024, 8, 128, device='cuda', dtype=torch.bfloat16)
k = torch.randn(1024, 8, 128, device='cuda', dtype=torch.bfloat16)
v = torch.randn(1024, 8, 128, device='cuda', dtype=torch.bfloat16)
cu = torch.tensor([0, 512, 1024], dtype=torch.int32, device='cuda')
out = flash_attn_varlen_func(q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu,
    max_seqlen_q=512, max_seqlen_k=512, causal=True, softmax_scale=0.125,
    window_size=(128, 0))
print('OK:', out.shape)
"
# Output: OK: torch.Size([1024, 8, 128])
```

## Test Steps

1. `from fa4_compile_wrapper import flash_attn_varlen_func` — import test
2. Forward pass **without** torch.compile — sanity check
3. Forward pass **with** `torch.compile` — **the critical test**
4. Forward + backward with `torch.compile` — training requires gradients
5. Patch `train_gpt.py` as described above
6. `sh run.sh` — full modded-nanogpt training completes on 8×B200

## Links

- modded-nanogpt: https://github.com/KellerJordan/modded-nanogpt
- Flash Attention 4 repo: https://github.com/Dao-AILab/flash-attention
- FA4 cute interface source: https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/interface.py
- FA4 on PyPI: `flash-attn-4==4.0.0b4` (`pip install --pre flash-attn-4`)
- torch custom ops tutorial: https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html
- `kernels` package (how FA3 is wrapped): https://github.com/pytorch-labs/kernels
- Relevant FA issues: https://github.com/Dao-AILab/flash-attention/issues/2003 (FA3 on GB200), https://github.com/Dao-AILab/flash-attention/issues/2307 (FA4 on SM120)
