# Controlled SFT Pilot — Corrected Run Report

## Final status

`SFT_PILOT_COMPLETED`

The one explicitly authorized corrected training run completed on Slurm, saved
the final adapter and resume state, and completed the approved 20-task
validation benchmark. No further training job, hyperparameter search, scale-up,
or GRPO job was started. The historical failure report remains unchanged at
`docs/experiments/sft-pilot-report.md`; Job `20389950` and its artifacts were
not overwritten.

## Frozen inputs and verification

- Branch: `dev/qwen-v100-stable-inference`
- Training commit: `408590d64a3f1d8d53c46dbe54c267c0636543d4`
- Fixed commit `d2d2fcb` is an ancestor of the training commit.
- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Model revision: `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- Dataset: `artifacts/sft-data-train-teacher-v2/dataset.jsonl`
- Dataset SHA-256: `e3a583be6fd7b399e9088f098f317d7589bf564dcce506b0d8a5de825ea40e3c`
- Dataset: 20 train trajectories; validation/test used for training: `false/false`
- Provenance and format gates: 20/20 accepted; no duplicate or validation/test leakage
- Token preflight: sequence range 771–1,732; maximum 1,732; truncation 0; empty targets 0
- CPU verification after the audit change: `90 passed, 3 skipped`
- Ruff, Python compilation, and `git diff --check`: passed

The two pre-existing untracked runtime sanity files were preserved:
`scripts/inference/runtime_performance_sanity.py` and
`scripts/slurm/runtime_performance_sanity.sbatch`.

## Training Job `20390015`

- Slurm: account `pi_daipeishan`, partition `gpu2Q`, QOS `gpuq`
- Allocation: 1 GPU, 2 CPU, 16 GB RAM, 30 minutes
- Node/GPU: `gpu207`, Tesla V100S-PCIE-32GB, compute capability 7.0
- Environment: Python 3.12.7, PyTorch `2.2.2+cu121`, CUDA 12.1,
  Transformers `4.46.3`, PEFT `not-installed` (manual LoRA implementation)
- Attention/dtype: V100-compatible SDPA math, FP16
- Training config: seed 42, 3 epochs, batch size 1, gradient accumulation 4,
  learning rate 1e-4, warmup ratio .05, weight decay 0, max grad norm 1.0,
  gradient checkpointing enabled, FP16 enabled, BF16 disabled
- LoRA config: rank 8, alpha 16, dropout .05

The runtime LoRA audit passed on the real loaded model before the training loop:

| Check | Result |
|---|---:|
| `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` | 28 each |
| Injected target modules | 196 |
| Trainable LoRA parameters | 9,232,384 |
| Total model parameters | 1,552,946,688 |
| Trainable ratio | 0.00594507 (0.5945%) |
| Base-model trainable parameters | 0 |
| Optimizer groups / parameters | 1 / 392 |
| All trainable parameters are expected LoRA parameters | true |

Training reached 15/15 optimizer steps and all 3 epochs. Per-epoch mean loss
was 1.064962, 0.688628, and 0.550508. Losses, logits, gradients, and gradient
norms were finite; no OOM or CUDA error occurred.

Final artifacts:

- Adapter: `artifacts/sft-pilot-20390015/adapter/`
- Final checkpoint: `artifacts/sft-pilot-20390015/checkpoints/checkpoint-epoch-3/`
- Adapter weights: 36,981,040 bytes on disk, 392 safetensors keys, all finite
- Optimizer, scheduler, scaler, RNG, and trainer state were saved for all 3 checkpoints

The exact adapter weight-file size and hash are recorded in the artifact
directory; the final weight file SHA-256 was
`a7d252f782a541dfc697055d43c7a25768d99633b6d5c3335131d00a6247bd77`.

## Post-training Runtime and validation Job `20390127`

The originally queued `gpu2Q` validation attempt was blocked by Slurm priority;
with the user's authorization, the same validation command was run on `gpu8Q`.
Job `20390127` completed with exit code 0 on `gpu801` (Tesla V100-SXM2-32GB).

Runtime checks passed:

- Base model loaded at the fixed revision in FP16 on `cuda:0` with SDPA math.
- Adapter merge succeeded for all 196 modules; merged delta norm was finite.
- HTTP model server became ready; server cleanup passed.
- Generation logits were finite for all scored tasks; no OOM, CUDA, tokenizer,
  runtime, verifier, or infrastructure errors occurred.
- Runtime parsed and scored model actions for all 20 validation tasks.
- `test_split_accessed=false`; `train_split_accessed=false`.

### Base vs SFT — validation split, 20 tasks

The prompt, tool schema, generation configuration, action budget, environment,
verifier, reward, and statistics were held constant. Only the model was changed
to the SFT adapter.

| Metric | Base | SFT | SFT − Base |
|---|---:|---:|---:|
| Task success rate | 0.0000 | 0.0000 | 0.0000 |
| Verifier success rate | 1.0000 | 1.0000 | 0.0000 |
| Executable SQL rate | 0.069565 | 0.000000 | -0.069565 |
| Average tool calls | 11.70 | 3.75 | -7.95 |
| Average invalid tool calls | 10.70 | 3.75 | -6.95 |
| Average SQL attempts | 11.50 | 3.75 | -7.75 |
| Average recovery count | 0.10 | 0.00 | -0.10 |
| Tool error rate | 0.914530 | 1.000000 | +0.085470 |
| Average latency (s) | 43.2186 | 19.3827 | -23.8359 |
| Average input tokens | 17,264.75 | 5,708.10 | -11,556.65 |
| Average output tokens | 1,061.40 | 625.70 | -435.70 |
| Average total tokens | 18,326.15 | 6,333.80 | -11,992.35 |
| Average reward | 0.0000 | 0.0000 | 0.0000 |

Counts and failure categories:

- Base: 230 SQL calls, 16 executable SQL calls, 214 invalid tool calls;
  4 invalid-action failures and 16 action-budget-exhausted failures.
- SFT: 75 SQL calls, 0 executable SQL calls, 75 invalid tool calls;
  15 invalid-action failures and 5 action-budget-exhausted failures.
- Both runs completed all 20 scored episodes with zero infrastructure failures.

Artifacts:

- Training: `artifacts/sft-pilot-20390015/`
- Validation: `artifacts/sft-agent-benchmark-20390127/`
- Comparison: `artifacts/sft-agent-benchmark-20390127/comparison.json`
- Preserved old failure: `artifacts/sft-pilot-20389950/`
