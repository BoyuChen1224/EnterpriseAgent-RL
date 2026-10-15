# M2A preparation evidence

Date: 2026-10-15, Asia/Shanghai. This directory records CPU engineering validation, **not Qwen capability results**.

- Original 48 CPU tests passed before new regression tests were added (11.25 seconds).
- Expanded suite: 69 passed in 20.40 seconds on the login node; no model was loaded.
- Ruff check / format and shell syntax checks passed.
- Independent `.venv-gpu`: pip check passed; Torch `2.2.2+cu121`, compiled CUDA `12.1`, CPU tensor `[1.0]`, and Transformers AutoTokenizer/AutoModelForCausalLM class imports succeeded. No CUDA device probe was performed on the login node.
- Complete pinned inference dependencies: `requirements/gpu-py312-linux.lock`; nvJitLink explicitly pinned to `12.1.105` alongside the CUDA 12.1 runtime libraries.
- 60/60 synthetic reference queries executed through SQLTool and matched independent Python row-based calculations. Three splits of 20, six distinct database fingerprints; frozen manifest `bbc638387323b1be946be6d761d5dcd973de087d6ab94145e97389f471db6d56`.
- GPU compatibility, model loading, real Agent trajectories, Base metrics, GPU peak memory and throughput: **Pending — no paid GPU job authorized or submitted**.

CPU protocol tests use explicit test doubles; they are not saved or presented as Base model evaluation reports. No SFT or GRPO was executed. Model weights/cache were not committed.
