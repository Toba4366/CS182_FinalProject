"""
Finite State Machine (FSM) generator utilities.

The generator creates random connected Moore machines (without outputs for now).
Each state has a set of outgoing actions that transition to other states.

The generator is vocabulary-aware and uses fixed token ranges:
- States: 0 to num_states-1 (always in range [0, MAX_STATES-1])
- Actions: MAX_STATES to MAX_STATES+max_actions-1 (uses lowest available action indices)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Optional

from . import MAX_STATES, MAX_ACTIONS


FSM = Dict[int, Dict[int, int]]


@dataclass
class FSMGeneratorConfig:
    """Configuration object for the FSM generator."""
    num_states: int = 5
    min_actions: int = 3
    max_actions: int = 8
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
            config.num_states >= 2
        ), "An FSM requires at least two states to be interesting."
        assert (
            config.num_states <= MAX_STATES
        ), f"num_states ({config.num_states}) exceeds MAX_STATES ({MAX_STATES})"
        assert (
            config.min_actions >= 1
        ), "Each state must have at least one outgoing action."
        assert (
            config.max_actions >= config.min_actions
        ), "max_actions must be >= min_actions."
        assert (
            config.max_actions <= MAX_ACTIONS
        ), f"max_actions ({config.max_actions}) exceeds MAX_ACTIONS ({MAX_ACTIONS})"

        self.config = config
        self.rng = random.Random(config.seed)
        self.action_ids = list(range(MAX_STATES, MAX_STATES + config.max_actions))

    def generate(self) -> FSM:
        """
        Generate a random connected DFA.

        Returns:
            A dictionary mapping state_id -> {action: next_state},
            where every state has exactly one outgoing transition for
            each action in self.action_ids (i.e., a total, deterministic
            transition function over a shared action alphabet).
        """
        num_states = self.config.num_states

        # Initialize empty transition dict for each state
        fsm: FSM = {state: {} for state in range(num_states)}

        # ------------------------------------------------------------------
        # Step 1: ensure (undirected) connectivity with a random spanning tree
        # ------------------------------------------------------------------
        remaining_states = list(range(1, num_states))
        connected_states = [0]

        while remaining_states:
            src = self.rng.choice(connected_states)
            dst = remaining_states.pop(self.rng.randrange(len(remaining_states)))

            # Use an unused action for src so we don't overwrite an existing edge
            action_id = self._sample_action_id(fsm[src])
            fsm[src][action_id] = dst

            # Optional reverse edge to strengthen connectivity. NO NEED FOR THIS.
            # reverse_action = self._sample_action_id(fsm[dst])
            # fsm[dst][reverse_action] = src

            connected_states.append(dst)

        # ------------------------------------------------------------------
        # Step 2: fill in missing (state, action) transitions to make a DFA
        # ------------------------------------------------------------------
        for state in range(num_states):
            for action_id in self.action_ids:
                if action_id not in fsm[state]:
                    next_state = self.rng.randrange(num_states)
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

    def generate_with_absorbing_state(self) -> FSM:
        fsm = self.generate()  # generate a DFA
        num_states = self.config.num_states
        absorbing_state = self.rng.randrange(num_states)  # pick one at random
        # Overwrite all outgoing transitions from this state to point to itself
        for action_id in self.action_ids:
            fsm[absorbing_state][action_id] = absorbing_state
        return fsm