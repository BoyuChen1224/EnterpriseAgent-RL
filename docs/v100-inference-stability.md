# V100 inference stability

## Problem

On the pinned V100-SM70 stack (`PyTorch 2.2.2+cu121`, `Transformers 4.46.3`), Qwen FP16 inference through the eager attention path is numerically unstable. FP32 eager inference is finite, while FP16 eager inference can produce non-finite attention scores before generation makes a token decision.

The production path is therefore:

`FP16 weights + SDPA math backend`

FP32 remains a diagnostic/reference path, not the production default.

## Evidence

The root-cause experiment was completed by the unique GPU Job `20385898` on a V100S-PCIE-32GB (compute capability 7.0):

| Path | Result |
|---|---|
| FP32 eager | finite/reference path |
| FP16 eager | non-finite QK scores and final logits |
| FP16 + SDPA math | finite ratio `1.0` |
| FP16 + SDPA math throughput | `44.40 tok/s` |
| FP32 throughput | `36.88 tok/s` |
| Peak reserved VRAM | approximately `13.15 GB` |

## Root cause

The first non-finite tensor was:

`decoder_layer[0].self_attn.attention_scores_qk`

The Q and K tensors were finite, but their FP16 QKᵀ product overflowed:

- `+Inf`: `858`
- `-Inf`: `195`

The overflow propagated into softmax, producing NaNs, and then into the final logits. The FP32 attention-score accumulation path is retained in the diagnostic code as a research/debug reference only; it is not wired into the production Runtime.

## Solution

The model configuration uses the explicit setting:

`attention_backend = "sdpa_math"`

The Runtime maps this setting to the official Transformers API:

`attn_implementation="sdpa"`

For each model generation call, it enters a local PyTorch SDPA context with:

```text
flash = disabled
math = enabled
mem_efficient = disabled
```

On the pinned PyTorch 2.2.2 environment this uses `torch.backends.cuda.sdp_kernel`. The newer `torch.nn.attention.sdpa_kernel` form is only a fallback for an already-installed environment; this change does not upgrade PyTorch or Transformers. CPU execution uses a no-op context.

The Runtime also passes `output_attentions=False` when loading and generating, and verifies the effective Transformers implementation. This prevents Qwen2 SDPA from being silently downgraded to eager attention because attention weights were requested.

The checkpoint, checkpoint revision, and `generation_config` are not modified.

## Finite-logit guard

The Runtime installs a Transformers-compatible `LogitsProcessor` for every autoregressive generation. It checks `torch.isfinite(logits).all()` immediately before the sampling/greedy decision at generation step 0 and at every later step.

On failure it raises `non_finite_logits` and records:

- generation step;
- logits shape, dtype, device, and finite ratio;
- calculable finite min/max values and NaN/+Inf/-Inf counts;
- attention backend, model revision, and GPU name;
- the full diagnostic context in the generation artifact and a dedicated JSON diagnostic file.

There is no NaN/Inf replacement, token-0 fallback, sampling-parameter adjustment, or prompt modification. The event telemetry sets `non_finite_logits_detected=true`; normal runs set it to `false`.

## Runtime telemetry

Each generation artifact records model revision, model dtype, device, GPU name, compute capability, attention backend, input/output token counts, load time, generation latency, throughput, peak allocated/reserved memory, the selected SDPA control API, and the non-finite-logits flag. Existing latency and memory aliases remain for downstream trajectory/evaluation readers.

GPU acceptance for this production integration must still be run explicitly on the V100. This implementation stage submits no GPU job and does not run Protocol, Agent, Base, SFT, or GRPO workloads.
