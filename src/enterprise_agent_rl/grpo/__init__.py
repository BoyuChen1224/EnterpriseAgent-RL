"""Reusable online GRPO orchestration for EnterpriseAgent-RL.

The package intentionally contains orchestration and backend adapters only.  It
does not replace the existing Runtime, LoRA, Reward, or task contracts.
"""

from .config import GrpoConfig, RunContext, load_config
from .orchestration import run_online_training

__all__ = ["GrpoConfig", "RunContext", "load_config", "run_online_training"]
