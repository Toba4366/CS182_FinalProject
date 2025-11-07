"""
Finite State Machine (FSM) module for in-context learning experiments.

This module contains implementations of Moore machines and related utilities
for generating training data and evaluating model performance on ICL tasks.
"""

from .moore_machine import MooreMachine, MooreMachineGenerator
from .fsm_utils import generate_random_sequence, validate_fsm_constraints

__all__ = [
    "MooreMachine",
    "MooreMachineGenerator", 
    "generate_random_sequence",
    "validate_fsm_constraints"
]