"""
Moore Machine implementation for in-context learning experiments.

A Moore machine is a finite state automaton where outputs depend only on 
the current state, not on the input. This implementation supports:
- 5 states exactly  
- 5-8 actions (input alphabet)
- 4-8 state transitions per state (including self-transitions)
"""

import random
import numpy as np
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass


@dataclass
class MooreMachine:
    """
    Moore Machine implementation with constrained parameters for ICL experiments.
    
    Attributes:
        num_states: Number of states (fixed at 5)
        num_actions: Number of possible input actions (5-8)
        states: Set of state identifiers  
        actions: Set of action identifiers
        transitions: Dict mapping (state, action) -> next_state
        outputs: Dict mapping state -> output_symbol
        initial_state: Starting state
    """
    
    num_states: int = 5
    num_actions: int = 8  # Default, but can vary 5-8
    states: Set[int] = None
    actions: Set[int] = None
    transitions: Dict[Tuple[int, int], int] = None
    outputs: Dict[int, str] = None
    initial_state: int = 0
    
    def __post_init__(self):
        """Initialize default values if not provided."""
        if self.states is None:
            self.states = set(range(self.num_states))
        if self.actions is None:
            self.actions = set(range(self.num_actions))
        if self.transitions is None:
            self.transitions = {}
        if self.outputs is None:
            self.outputs = {}
    
    def add_transition(self, from_state: int, action: int, to_state: int):
        """Add a state transition."""
        if from_state not in self.states:
            raise ValueError(f"Invalid from_state: {from_state}")
        if to_state not in self.states:
            raise ValueError(f"Invalid to_state: {to_state}")
        if action not in self.actions:
            raise ValueError(f"Invalid action: {action}")
        
        self.transitions[(from_state, action)] = to_state
    
    def set_output(self, state: int, output: str):
        """Set the output for a given state."""
        if state not in self.states:
            raise ValueError(f"Invalid state: {state}")
        self.outputs[state] = output
    
    def step(self, current_state: int, action: int) -> Tuple[int, str]:
        """
        Execute one step of the Moore machine.
        
        Args:
            current_state: Current state
            action: Input action
            
        Returns:
            Tuple of (next_state, output)
        """
        if (current_state, action) not in self.transitions:
            raise ValueError(f"No transition defined for state {current_state}, action {action}")
        
        next_state = self.transitions[(current_state, action)]
        output = self.outputs.get(next_state, "")
        
        return next_state, output
    
    def run_sequence(self, action_sequence: List[int]) -> Tuple[List[int], List[str]]:
        """
        Run a sequence of actions through the Moore machine.
        
        Args:
            action_sequence: List of input actions
            
        Returns:
            Tuple of (state_sequence, output_sequence)
        """
        states = [self.initial_state]
        outputs = [self.outputs.get(self.initial_state, "")]
        
        current_state = self.initial_state
        
        for action in action_sequence:
            current_state, output = self.step(current_state, action)
            states.append(current_state)
            outputs.append(output)
        
        return states, outputs
    
    def is_valid(self) -> bool:
        """
        Check if the Moore machine satisfies the constraints:
        - Exactly 5 states
        - 5-8 actions  
        - Each state has 4-8 outgoing transitions
        - All states have outputs defined
        """
        # Check state count
        if len(self.states) != 5:
            return False
        
        # Check action count
        if not (5 <= len(self.actions) <= 8):
            return False
        
        # Check that all states have outputs
        if len(self.outputs) != len(self.states):
            return False
        
        # Check transition counts per state
        for state in self.states:
            outgoing_transitions = sum(1 for (s, a) in self.transitions.keys() if s == state)
            if not (4 <= outgoing_transitions <= 8):
                return False
        
        return True
    
    def to_dict(self) -> Dict:
        """Convert Moore machine to dictionary representation."""
        return {
            "num_states": self.num_states,
            "num_actions": self.num_actions,
            "states": list(self.states),
            "actions": list(self.actions),
            "transitions": {f"{s},{a}": next_s for (s, a), next_s in self.transitions.items()},
            "outputs": self.outputs,
            "initial_state": self.initial_state
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "MooreMachine":
        """Create Moore machine from dictionary representation."""
        fsm = cls(
            num_states=data["num_states"],
            num_actions=data["num_actions"],
            states=set(data["states"]),
            actions=set(data["actions"]),
            initial_state=data["initial_state"]
        )
        
        # Reconstruct transitions
        for key, next_state in data["transitions"].items():
            state, action = map(int, key.split(","))
            fsm.transitions[(state, action)] = next_state
        
        fsm.outputs = data["outputs"]
        return fsm


class MooreMachineGenerator:
    """Generator for random Moore machines with specified constraints."""
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
    
    def generate(self, 
                 num_actions: Optional[int] = None,
                 output_alphabet_size: int = 3) -> MooreMachine:
        """
        Generate a random Moore machine satisfying the constraints.
        
        Args:
            num_actions: Number of actions (5-8, random if None)
            output_alphabet_size: Size of output alphabet
            
        Returns:
            Random Moore machine
        """
        # Determine number of actions
        if num_actions is None:
            num_actions = self.rng.randint(5, 8)
        elif not (5 <= num_actions <= 8):
            raise ValueError("num_actions must be between 5 and 8")
        
        # Create machine
        fsm = MooreMachine(
            num_states=5,
            num_actions=num_actions,
            states=set(range(5)),
            actions=set(range(num_actions)),
            initial_state=0
        )
        
        # Generate outputs for each state
        output_symbols = [f"out_{i}" for i in range(output_alphabet_size)]
        for state in fsm.states:
            fsm.outputs[state] = self.rng.choice(output_symbols)
        
        # Generate transitions for each state
        for state in fsm.states:
            # Number of outgoing transitions (4-8)
            num_transitions = self.rng.randint(4, min(8, num_actions))
            
            # Randomly select which actions have transitions
            available_actions = list(fsm.actions)
            selected_actions = self.rng.sample(available_actions, num_transitions)
            
            # Assign random target states (including self-transitions)
            for action in selected_actions:
                target_state = self.rng.choice(list(fsm.states))
                fsm.add_transition(state, action, target_state)
        
        # Ensure the machine is valid
        assert fsm.is_valid(), "Generated machine violates constraints"
        
        return fsm
    
    def generate_batch(self, batch_size: int, **kwargs) -> List[MooreMachine]:
        """Generate a batch of random Moore machines."""
        return [self.generate(**kwargs) for _ in range(batch_size)]