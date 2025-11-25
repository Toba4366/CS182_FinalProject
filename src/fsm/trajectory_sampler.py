"""
Trajectory sampler built on top of the FSM generator.

The sampler produces a fixed number of trajectories by performing random walks
over the generated FSM. Each trajectory records both states and actions so that
we can reconstruct the underlying dynamics later.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

from .generator import FSM, FSMGenerator, FSMGeneratorConfig


Trajectory = Dict[str, Sequence[int]]
"""
Dict with two keys:
    "states" -> list of state IDs
    "actions" -> list of action IDs

Ex) Output
[
    {
        "states": [...], => Length 65
        "actions": [...] => Length 64
    },

    {"states": [...], "actions": [...]},
    {"states": [...], "actions": [...]},
]
"""


@dataclass
class TrajectorySamplerConfig:
    """Configuration for the trajectory sampler."""

    num_states: int = 5
    min_actions_per_state: int = 3
    max_actions_per_state: int = 8
    num_trajectories: int = 3
    trajectory_length: int = 64
    seed: Optional[int] = None

    def to_generator_config(self) -> FSMGeneratorConfig:
        """Convert sampler hyperparameters to generator settings."""
        min_actions = max(1, self.min_actions_per_state)
        max_actions = max(min_actions, self.max_actions_per_state)

        return FSMGeneratorConfig(
            num_states=max(2, self.num_states),
            min_actions=min_actions,
            max_actions=max_actions,
            seed=self.seed,
        )

class TrajectorySampler:
    """Sampler that uses the FSM generator to create rollouts."""

    def __init__(self, config: TrajectorySamplerConfig):
        """Generate FSM."""
        self.config = config
        self.rng = random.Random(config.seed)
        self.generator = FSMGenerator(config.to_generator_config())
        self.fsm: FSM = self.generator.generate()

    def regenerate_fsm(self) -> FSM:
        """Create and store a new FSM."""
        self.fsm = self.generator.generate()
        return self.fsm

    def get_fsm(self) -> FSM:
        """Return the currently stored FSM."""
        return self.fsm

    def sample(
        self,
        start_states: Optional[Sequence[Optional[int]]] = None,
        trajectory_length: Optional[int] = None,
        regenerate: bool = False,
    ) -> Sequence[Trajectory]:
        """
        Roll out trajectories from the internally stored FSM.

        Args:
            start_states: optional list of starting states (None -> random).
            trajectory_length: optional override for rollout length (default: 64).
            regenerate: whether to sample a fresh FSM before rolling out.
        """
        if regenerate or self.fsm is None:
            self.regenerate_fsm()

        traj_length = trajectory_length or self.config.trajectory_length
        num_traj = len(start_states) if start_states is not None else self.config.num_trajectories
        start_states = start_states or [None] * num_traj
        return [
            self.rollout(traj_length, start_state=state) 
            for state in start_states
            # Generate one random walk for each start state
        ]
    

    def rollout(
        self, traj_length: int, start_state: Optional[int] = None
    ) -> Trajectory:
        """Perform one random walk over the FSM."""
        state = start_state if start_state is not None else self.rng.choice(list(self.fsm.keys()))
        states = [state]
        actions = []

        for _ in range(traj_length):
            transitions = self.fsm[state]

            state_actions = list(transitions.keys())
            action = self.rng.choice(state_actions)
            next_state = transitions[action]

            actions.append(action)
            states.append(next_state)
            state = next_state

        return {"states": states, "actions": actions}


