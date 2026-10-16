# Base Agent Benchmark

This is the formal Base Agent baseline for the frozen M2A benchmark.

- Model: `Qwen2.5-1.5B-Instruct`
- Revision: `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- Split: `validation`
- Tasks: `20`
- Policy: `local_llm`
- `deterministic_fixture_expert`: `false`
- Runtime: FP16, SDPA math, finite-logit guard
- Slurm job: `20387111` on `gpu2Q` / `gpuq`, one Tesla V100, 2 CPUs, 16 GB

The benchmark completed all 20 tasks. The independent verifier scored every
trajectory; task success was `0/20`, which is a business result rather than an
infrastructure failure. The final status is `READY_FOR_SFT`.

Headline results:

- Task Success Rate: `0.0`
- Verifier Success Rate: `1.0`
- Executable SQL Rate: `0.0695652174`
- Tool Error Rate: `0.9145299145`
- Average Tool Calls: `11.7`
- Average SQL Attempts: `11.5`
- Average Recovery Count: `0.1`
- Average Latency: `43.2186 s`
- Average Input Tokens: `17264.75`
- Average Output Tokens: `1061.4`
- Average Reward: `0.0`

Failure analysis classified 4 tasks as Invalid Action and 16 as Action Budget
Exhausted. Recovery analysis found 2 successful recoveries and 14 repeated
recovery loops. No non-finite logits, OOM, CUDA, tokenizer, verifier, or other
runtime infrastructure failure was observed.

`test_split_accessed = false`. The 20 validation tasks measure this Base Agent
baseline and do not represent final generalization performance on the unseen
test split.

The complete machine-readable artifact is under
`artifacts/base-agent-benchmark-20387111/`.
