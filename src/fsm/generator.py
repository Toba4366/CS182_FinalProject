"""
Finite State Machine (FSM) generator utilities.

The generator creates random connected Moore machines (without outputs for now).
Each state has a set of outgoing actions that transition to other states.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Optional


FSM = Dict[int, Dict[int, int]]


@dataclass
class FSMGeneratorConfig:
    """Configuration object for the FSM generator."""
    num_states: int = 5
    min_actions_per_state: int = 3
    max_actions_per_state: int = 8
    seed: Optional[int] = None


class FSMGenerator:
    """
    Random fully-connected FSM generator.

    The resulting FSM is guaranteed to be (undirected) connected, meaning there
    exists a path between every pair of states. This is sufficient for
    generating long trajectories for training data.
    """

    def __init__(self, config: FSMGeneratorConfig):
        assert (
            config.num_states >= 3
        ), "An FSM requires at least three states to be interesting."
        assert (
            config.min_actions_per_state >= 1
        ), "Each state must have at least one outgoing action."
        assert (
            config.max_actions_per_state >= config.min_actions_per_state
        ), "max_actions_per_state must be >= min_actions_per_state."

        self.config = config
        self.rng = random.Random(config.seed)
        vocab_size = max(8, config.max_actions_per_state)
        start_id = config.num_states
        self.action_ids = list(range(start_id, start_id + vocab_size))

    def generate(self) -> FSM:
        """
        Generate a random connected FSM.

        Returns:
            A dictionary mapping state_id -> {action: next_state}
        """
        num_states = self.config.num_states
        fsm: FSM = {state: {} for state in range(num_states)}

        # Step 1: ensure connectivity with a random spanning tree
        remaining_states = list(range(1, num_states))
        connected_states = [0]

        while remaining_states:
            src = self.rng.choice(connected_states)
            dst = remaining_states.pop(self.rng.randrange(len(remaining_states)))
            action_id = self._sample_action_id(fsm[src])

            fsm[src][action_id] = dst

            # Add reverse edge to strengthen connectivity (optional)
            reverse_action = self._sample_action_id(fsm[dst])
            fsm[dst][reverse_action] = src

            connected_states.append(dst)

        # Step 2: add random transitions until action budgets are met
        for state in range(num_states):
            target_num_actions = self.rng.randint(
                self.config.min_actions_per_state, self.config.max_actions_per_state
            )

            while len(fsm[state]) < target_num_actions:
                next_state = self.rng.randrange(num_states)
                action_id = self._sample_action_id(fsm[state])
                fsm[state][action_id] = next_state

        return fsm

    def _sample_action_id(self, state_actions: Dict[int, int]) -> int:
        """Sample an unused action ID for the given state's dictionary."""
        available = [a for a in self.action_ids if a not in state_actions]
        if not available:
            raise ValueError(
                "Action vocabulary exhausted. Increase max_actions_per_state or vocab_size."
            )
        return self.rng.choice(available)


