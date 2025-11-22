"""
Dataset builder for Moore machine trajectories.

The dataset stores both the generated FSM and the sampled trajectories to allow
post-hoc analysis of the induced behaviour.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, cast
import torch
from torch.utils.data import Dataset

from src.fsm.trajectory_sampler import TrajectorySampler, TrajectorySamplerConfig


@dataclass
class ICLDatasetConfig:
    """Configuration for the ICL-friendly dataset."""

    num_samples: int = 256
    demo_length: int = 64
    query_length: int = 32
    sampler_config: TrajectorySamplerConfig = field(
        default_factory=TrajectorySamplerConfig
    )
    max_seq_len: int = 512
    cache_path: Path = Path("data/icl_dataset.pt")

class MooreICLDataset(Dataset):
    """
    Dataset that stores trajectories suitable for in-context learning.

    Vocabulary layout (per sample):
        - 0 … num_states-1 : state IDs
        - num_states … num_states+max_actions-1 : action IDs
        - `eos_token` : end-of-sequence marker between segments
        - `query_token` : marks the start of the query segment
        - `pad_token` : padding value for batching

    Each sample contains:
        - One demo trajectory of length 64 for every start state in the FSM
        - One query trajectory of length 32 whose states must be predicted from actions
    """

    def __init__(
        self,
        sample_indices: List[int],
        all_samples: List[Dict[str, object]],
        sampler_config: TrajectorySamplerConfig,
        max_seq_len: int,
    ):
        self.sample_ids = sample_indices
        self.samples = all_samples
        self.max_seq_len = max_seq_len

        self.sampler = TrajectorySampler(sampler_config)
        generator_cfg = self.sampler.generator.config
        self.num_states = generator_cfg.num_states
        self.max_actions = generator_cfg.max_actions_per_state

        # Vocabulary layout
        self.action_offset = self.num_states
        self.eos_token = self.action_offset + self.max_actions
        self.query_token = self.eos_token + 1
        self.unk_token = self.eos_token + 2
        self.pad_token = self.eos_token + 3
        self.vocab_size = self.pad_token + 1

        self.rng = random.Random(sampler_config.seed)

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[self.sample_ids[idx]]
        demos = cast(List[Dict[str, List[int]]], sample["demos"])
        query = cast(Dict[str, List[int]], sample["query"])
        demo_tokens = self._select_and_encode_demos(demos)
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
        self, demos: List[Dict[str, List[int]]], num_samples: int = 3
    ) -> List[List[int]]:
        choices = self.rng.sample(demos, num_samples)
        return [self._trajectory_to_tokens(traj) for traj in choices]

    def _trajectory_to_tokens(self, traj: Dict[str, List[int]]) -> List[int]:
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
    fsm = sampler.regenerate_fsm()
    demo_starts = list(fsm.keys())
    demos = sampler.sample(
        start_states=demo_starts,
        trajectory_length=config.demo_length,
    )

    query_state = rng.choice(list(fsm.keys()))
    query = sampler.sample(
        start_states=[query_state],
        trajectory_length=config.query_length,
    )[0]

    return {"fsm": fsm, "demos": demos, "query": query}


def load_or_create_icl_samples(config: ICLDatasetConfig) -> List[Dict[str, object]]:
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



