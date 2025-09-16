from dataclasses import dataclass, field
from typing import Optional
import trl

@dataclass
class GRPOConfig(trl.GRPOConfig):
    """
    args for callbacks, benchmarks etc
    """
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The optional system prompt to use."},
    )