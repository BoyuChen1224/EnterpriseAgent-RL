# Controlled SFT Pilot — Training Failure Report

## Status

`SFT_TRAINING_FAILED`

The single authorized pilot job started on a V100 and loaded the fixed base
model, but exited before the first optimizer step. No validation evaluation was
submitted after the failure, and no SFT capability claim is made.

## Frozen inputs

- Branch: `dev/qwen-v100-stable-inference`
- Training code commit used by Job `20389950`: `fb3dc863b2d0c0508154e8ee7a75c5849d56e2b6`
- Model: `Qwen/Qwen2.5-1.5B-Instruct`, revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- Dataset: `artifacts/sft-data-train-teacher-v2/dataset.jsonl`
- Dataset SHA-256: `e3a583be6fd7b399e9088f098f317d7589bf564dcce506b0d8a5de825ea40e3c`
- Samples: 20 train-only trajectories; 6 deterministic teacher and 14 reference-SQL teacher
- Validation/test used for training: false/false

The tokenizer-only preflight found complete sequences from 771 to 1,732 tokens,
with zero truncation, zero empty targets, and 125 to 289 supervised assistant
tokens per example. The assistant-only spans were checked against the Qwen chat
template before training.

## Job evidence

- Job: `20389950`
- Account/partition/QOS: `pi_daipeishan / gpu2Q / gpuq`
- Allocation: 1 Tesla V100S-PCIE-32GB, 2 CPU, 16 GB, 30 minutes
- Node: `gpu207`
- Runtime: 2m38s; Slurm exit `1:0`
- Environment: PyTorch `2.2.2+cu121`, Transformers `4.46.3`, FP16, SDPA math, CC 7.0
- Artifact root: `artifacts/sft-pilot-20389950/`

The raw failure is preserved in `raw-slurm-output`. The exception was
`ValueError: no_trainable_lora_parameters`; the manual LoRA parameters were
registered as `...lora_A`/`...lora_B`, while the selector incorrectly searched
for `...lora_A.weight`/`...lora_B.weight`. Therefore no loss, optimizer step,
checkpoint, adapter, or SFT-vs-Base validation result exists for this job.

## Correction and verification

The selector was corrected in commit
`d2d2fcb fix: register manual LoRA parameters for training`. CPU verification
after the correction: `89 passed, 3 skipped`; Ruff, syntax, and diff checks
also pass. This correction was not used to silently restart Job `20389950`.

The pilot remains failed. A second GPU job requires explicit authorization.
