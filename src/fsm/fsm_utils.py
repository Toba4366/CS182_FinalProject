"""
Utility functions for FSM operations and sequence generation.
"""

import random
import numpy as np
from typing import List, Tuple, Optional
from .moore_machine import MooreMachine


def generate_random_sequence(fsm: MooreMachine, 
                           length: int, 
                           seed: Optional[int] = None) -> Tuple[List[int], List[int], List[str]]:
    """
    Generate a random sequence of actions and corresponding states/outputs.
    
    Args:
        fsm: Moore machine to run
        length: Length of action sequence to generate
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (action_sequence, state_sequence, output_sequence)
    """
    if seed is not None:
        random.seed(seed)
    
    # Generate random action sequence
    action_sequence = [random.choice(list(fsm.actions)) for _ in range(length)]
    
    # Run through FSM
    state_sequence, output_sequence = fsm.run_sequence(action_sequence)
    
    return action_sequence, state_sequence, output_sequence


def validate_fsm_constraints(fsm: MooreMachine) -> Tuple[bool, List[str]]:
    """
    Validate that an FSM meets all constraints and return detailed feedback.
    
    Args:
        fsm: Moore machine to validate
        
    Returns:
        Tuple of (is_valid, list_of_constraint_violations)
    """
    violations = []
    
    # Check state count
    if len(fsm.states) != 5:
        violations.append(f"Must have exactly 5 states, got {len(fsm.states)}")
    
    # Check action count
    if not (5 <= len(fsm.actions) <= 8):
        violations.append(f"Must have 5-8 actions, got {len(fsm.actions)}")
    
    # Check that all states have outputs
    missing_outputs = fsm.states - set(fsm.outputs.keys())
    if missing_outputs:
        violations.append(f"States missing outputs: {missing_outputs}")
    
    # Check transition counts per state
    for state in fsm.states:
        outgoing_transitions = [(s, a) for (s, a) in fsm.transitions.keys() if s == state]
        num_transitions = len(outgoing_transitions)
        if not (4 <= num_transitions <= 8):
            violations.append(f"State {state} has {num_transitions} transitions, must have 4-8")
    
    # Check for unreachable states (except initial state)
    reachable_states = {fsm.initial_state}
    changed = True
    while changed:
        changed = False
        for (from_state, action), to_state in fsm.transitions.items():
            if from_state in reachable_states and to_state not in reachable_states:
                reachable_states.add(to_state)
                changed = True
    
    unreachable = fsm.states - reachable_states
    if unreachable:
        violations.append(f"Unreachable states detected: {unreachable}")
    
    return len(violations) == 0, violations


def create_input_output_examples(fsm: MooreMachine,
                                num_examples: int = 5,
                                sequence_length: int = 10,
                                seed: Optional[int] = None) -> List[Tuple[List[int], List[str]]]:
    """
    Create input-output examples for in-context learning.
    
    Args:
        fsm: Moore machine to generate examples from
        num_examples: Number of input-output pairs to generate
        sequence_length: Length of each sequence
        seed: Random seed for reproducibility
        
    Returns:
        List of (input_sequence, output_sequence) tuples
    """
    if seed is not None:
        random.seed(seed)
    
    examples = []
    for i in range(num_examples):
        # Use different seed for each example to ensure diversity
        example_seed = None if seed is None else seed + i
        actions, states, outputs = generate_random_sequence(fsm, sequence_length, example_seed)
        
        # Store input actions and corresponding outputs
        examples.append((actions, outputs[1:]))  # Skip initial state output
    
    return examples


def format_sequence_for_model(action_seq: List[int], 
                            output_seq: List[str],
                            action_token_prefix: str = "A",
                            output_token_prefix: str = "O") -> str:
    """
    Format action and output sequences for language model consumption.
    
    Args:
        action_seq: Sequence of action integers
        output_seq: Sequence of output strings
        action_token_prefix: Prefix for action tokens
        output_token_prefix: Prefix for output tokens
        
    Returns:
        Formatted string sequence
    """
    formatted_parts = []
    
    for action, output in zip(action_seq, output_seq):
        action_token = f"{action_token_prefix}{action}"
        output_token = f"{output_token_prefix}{output}"
        formatted_parts.extend([action_token, output_token])
    
    return " ".join(formatted_parts)


def compute_sequence_accuracy(predicted_outputs: List[str], 
                            ground_truth_outputs: List[str]) -> float:
    """
    Compute accuracy of predicted output sequence.
    
    Args:
        predicted_outputs: Model-predicted output sequence
        ground_truth_outputs: Correct output sequence
        
    Returns:
        Accuracy as fraction of correct predictions
    """
    if len(predicted_outputs) != len(ground_truth_outputs):
        min_len = min(len(predicted_outputs), len(ground_truth_outputs))
        predicted_outputs = predicted_outputs[:min_len]
        ground_truth_outputs = ground_truth_outputs[:min_len]
    
    if len(ground_truth_outputs) == 0:
        return 1.0
    
    correct = sum(p == gt for p, gt in zip(predicted_outputs, ground_truth_outputs))
    return correct / len(ground_truth_outputs)