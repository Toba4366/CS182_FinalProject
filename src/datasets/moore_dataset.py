"""
Dataset builder for Moore machine trajectories.

The dataset stores both the generated FSM and the sampled trajectories to allow
post-hoc analysis of the induced behaviour.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, cast
import torch # type: ignore
from torch.utils.data import Dataset # type: ignore

from src.fsm.trajectory_sampler import TrajectorySampler, TrajectorySamplerConfig
from src.fsm import MAX_STATES, MAX_ACTIONS


@dataclass
class ICLDatasetConfig:
    """Configuration for the ICL-friendly dataset."""

    num_samples: int = 10_000
    traj_sampler_config: TrajectorySamplerConfig = field(
        default_factory=TrajectorySamplerConfig
    )
    max_seq_len: int = 512
    cache_path: Path = Path("data/icl_dataset.pt")
    # Optional explicit demo and query lengths (if None, computed as base_len * log(base_len))
    demo_length: Optional[int] = None
    query_length: Optional[int] = None
    # Add randomness to lengths
    length_variation: float = 0.2  # ±20% variation

class MooreICLDataset(Dataset):
    """
    Dataset that stores trajectories suitable for in-context learning.

    Vocabulary layout (fixed):
        - 0 … MAX_STATES-1 : state IDs (always 0-7, even if fewer states used)
        - MAX_STATES … MAX_STATES+MAX_ACTIONS-1 : action IDs (always 8-17, even if fewer actions used)
        - `eos_token` : end-of-sequence marker between segments
        - `query_token` : marks the start of the query segment
        - `pad_token` : padding value for batching

    Each sample contains:
        - Multiple demo trajectories with variable lengths
        - One query trajectory with variable length whose states must be predicted from actions
    """

    def __init__(
        self,
        sample_indices: List[int],
        all_samples: List[Dict[str, object]],
        traj_sampler_config: TrajectorySamplerConfig,
        max_seq_len: int,
    ):
        self.sample_ids = sample_indices
        self.samples = all_samples
        self.max_seq_len = max_seq_len

        self.sampler = TrajectorySampler(traj_sampler_config)
        generator_cfg = self.sampler.generator.config
        self.num_states = generator_cfg.num_states
        self.max_actions = generator_cfg.max_actions

        # Fixed vocabulary layout (always use MAX_STATES and MAX_ACTIONS)
        self.action_offset = MAX_STATES
        self.eos_token = MAX_STATES + MAX_ACTIONS
        self.query_token = self.eos_token + 1
        self.pad_token = self.eos_token + 2
        self.vocab_size = self.pad_token + 1

        self.rng = random.Random(traj_sampler_config.seed)

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Takes one FSM exampthe tale and compiles its demo and query trajectories into a single token sequence.
        Builds the shifted inputs, targets, and loss mask for that token sequence and returns them as PyTorch tensors. 

        Ex) 
            sequence_tokens = [0, 8, 1, 8, 2, <eos>, 1, 9, 2, 8, 0, <eos>, <query>, 2, 9, 0, 8, 1, <eos>]
            sequence_mask   = [F, F, F, F, F,  F,   F, F, F, F, F,  F,   F,  F, F, T, F, T, F]

            {
                "input_ids": torch.tensor(
                    [0, 0, 8, 1, 8, 2, <eos>, 1, 9, 2, 8, 0, <eos>, <query>, 2, 9, 0, 8, 1, <eos>],
                    dtype=torch.long
                ),

                "target_ids": torch.tensor(
                    [0, 8, 1, 8, 2, <eos>, 1, 9, 2, 8, 0, <eos>, <query>, 2, 9, 0, 8, 1, <eos>],
                    dtype=torch.long
                ),

                "loss_mask": torch.tensor(
                    [False, False, False, False, False,
                    False, False, False, False, False,
                    False, False, False, False, False,
                    True,  False, True,  False],
                    dtype=torch.bool
                )           
            }
        
        """
        
        sample = self.samples[self.sample_ids[idx]]
        demos = cast(List[Dict[str, List[int]]], sample["demos"])
        query = cast(Dict[str, List[int]], sample["query"])
        demo_tokens = self._select_and_encode_demos(demos, num_samples=3)
        sequence_tokens: List[int] = []
        sequence_mask: List[bool] = []

        for tokens in demo_tokens:
            for token in tokens:
                sequence_tokens.append(token)
                sequence_mask.append(False)
            sequence_tokens.append(self.eos_token)
            sequence_mask.append(False)

        sequence_tokens.append(self.query_token)
        sequence_mask.append(False)

        query_tokens, query_mask = self._encode_query(query)
        sequence_tokens.extend(query_tokens)
        sequence_mask.extend(query_mask)

        sequence_tokens.append(self.eos_token)
        sequence_mask.append(False)

        if len(sequence_tokens) > self.max_seq_len:
            raise ValueError(
                f"Sequence length {len(sequence_tokens)} exceeds max_seq_len "
                f"{self.max_seq_len}. Increase max_seq_len."
            )

        input_tokens = [sequence_tokens[0]] + sequence_tokens[:-1]
        target_tokens = sequence_tokens
        loss_mask = sequence_mask[:]
        loss_mask[0] = False

        return {
            "input_ids": torch.tensor(input_tokens, dtype=torch.long),
            "target_ids": torch.tensor(target_tokens, dtype=torch.long),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.bool),
        }

    # --------------------------------------------------------------------- #
    # Helper methods
    # --------------------------------------------------------------------- #
    def _select_and_encode_demos(
        self, demos: List[Dict[str, List[int]]], num_samples: int
    ) -> List[List[int]]:
        """Turn each demo trajectory into a token sequence using _trajectory_to_tokens"""
        # Sample without replacement, but take all if num_samples >= len(demos)
        if num_samples > len(demos):
            num_samples = len(demos)
        choices = self.rng.sample(demos, num_samples)
        return [self._trajectory_to_tokens(traj) for traj in choices]

    def _trajectory_to_tokens(self, traj: Dict[str, List[int]]) -> List[int]:
        """
        Turns a trajectory into a token sequence: [state, action, state]
        Ex)
            trajectory = {
                "states":  [0, 1, 2],
                "actions": [8, 9],
            }

            tokens = [0, 8, 1, 9, 2]
        """
        tokens: List[int] = []
        states = traj["states"]
        actions = traj["actions"]

        for idx, action in enumerate(actions):
            tokens.append(states[idx])
            tokens.append(action)
        tokens.append(states[-1])
        return tokens

    def _encode_query(
        self, traj: Dict[str, List[int]]
    ) -> Tuple[List[int], List[bool]]:
        """
        Converts the query into a token sequence and builds a parallel mask array

        Mask array:
            False: Starting state and all actions
            True: Each next state

        Ex)
            tokens = [2, 9, 0, 8, 1]
            mask   = [F, F, T, F, T]
        """
        tokens: List[int] = []
        mask: List[bool] = []
        states = traj["states"]
        actions = traj["actions"]

        tokens.append(states[0])
        mask.append(False)

        for idx, action in enumerate(actions):
            tokens.append(action)
            mask.append(False)

            next_state = states[idx + 1]
            tokens.append(next_state)
            mask.append(True)

        return tokens, mask
    


def _generate_icl_sample(
    sampler: TrajectorySampler,
    config: ICLDatasetConfig,
    rng: random.Random,
) -> Dict[str, object]:
    """Generate a single ICL sample with variable-length demos and query."""
    fsm = sampler.regenerate_fsm()
    
    # Compute base length from num_states * max_actions
    generator_cfg = sampler.generator.config
    base_length = generator_cfg.num_states * generator_cfg.max_actions
    
    # Compute demo and query lengths: use explicit if provided, otherwise base_len * log2(base_len)
    if config.demo_length is not None:
        demo_base = config.demo_length
    else:
        demo_base = int(base_length * math.log(base_length))
    
    if config.query_length is not None:
        query_base = config.query_length
    else:
        query_base = int(base_length * math.log(base_length))
    
    # Add random variation to each demo
    demo_starts = list(fsm.keys())
    demos = []
    for start_state in demo_starts:
        variation = rng.uniform(-config.length_variation, config.length_variation)
        demo_length = max(1, int(demo_base * (1 + variation)))
        demo = sampler.rollout(demo_length, start_state=start_state)
        demos.append(demo)
    
    # Add random variation to query
    query_state = rng.choice(list(fsm.keys()))
    variation = rng.uniform(-config.length_variation, config.length_variation)
    query_length = max(1, int(query_base * (1 + variation)))
    query = sampler.rollout(query_length, start_state=query_state)

    return {"fsm": fsm, "demos": demos, "query": query}

    """
    return value = {
        "fsm":   <FSM dictionary>,
        "demos": <List of trajectories>,
        "query": <One trajectory>
    }
    """

def load_or_create_icl_samples(config: ICLDatasetConfig) -> List[Dict[str, object]]:
    """Generate multiple ICL samples"""
    if config.cache_path.exists():
        with config.cache_path.open("rb") as f:
            samples = torch.load(f)
        return samples

    sampler = TrajectorySampler(config.sampler_config)
    rng = random.Random(config.sampler_config.seed)
    samples = [
        _generate_icl_sample(sampler, config, rng)
        for _ in range(config.num_samples)
    ]

    config.cache_path.parent.mkdir(parents=True, exist_ok=True)
    with config.cache_path.open("wb") as f:
        torch.save(samples, f)

    return samples



