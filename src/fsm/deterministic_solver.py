"""
Deterministic solver for Moore machines.

This solver learns the FSM transition function from demo trajectories
and uses it to predict states in query trajectories.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from .generator import FSM
from .trajectory_sampler import Trajectory


class DeterministicSolver:
    """
    A deterministic solver that learns FSM transitions from demos
    and predicts states in queries.
    
    The solver builds a transition table from demo trajectories:
    - For each (state, action) pair seen in demos, record the next state
    - If multiple transitions are seen for the same (state, action), use majority vote
    - For unseen (state, action) pairs, predict the most common next state overall
    """
    
    def __init__(self):
        self.transitions: Dict[Tuple[int, int], int] = {}
        self.transition_counts: Dict[Tuple[int, int], Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.fallback_state: Optional[int] = None
    
    def learn_from_demos(self, demos: List[Trajectory]) -> None:
        """
        Learn the FSM transition function from demo trajectories.
        
        Args:
            demos: List of demo trajectories, each with "states" and "actions" keys
        """
        # Reset state
        self.transitions = {}
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        all_states = []
        
        for demo in demos:
            states = demo["states"]
            actions = demo["actions"]
            
            # Record all states for fallback
            all_states.extend(states)
            
            # Learn transitions: (state, action) -> next_state
            for i, action in enumerate(actions):
                current_state = states[i]
                next_state = states[i + 1]
                key = (current_state, action)
                self.transition_counts[key][next_state] += 1
        
        # Build transition table using majority vote
        for (state, action), next_state_counts in self.transition_counts.items():
            # Get the most common next state for this (state, action) pair
            most_common_next = max(next_state_counts.items(), key=lambda x: x[1])[0]
            self.transitions[(state, action)] = most_common_next
        
        # Set fallback state (most common state overall)
        if all_states:
            from collections import Counter
            state_counter = Counter(all_states)
            self.fallback_state = state_counter.most_common(1)[0][0]
        else:
            self.fallback_state = 0
    
    def predict_query(self, query: Trajectory) -> Tuple[List[int], float]:
        """
        Predict states in a query trajectory given its actions.
        
        Args:
            query: Query trajectory with "states" (ground truth) and "actions"
        
        Returns:
            Tuple of (predicted_states, accuracy)
            - predicted_states: List of predicted state IDs
            - accuracy: Fraction of correctly predicted states (excluding first state)
        """
        states = query["states"]
        actions = query["actions"]
        
        predicted_states = [states[0]]  # First state is given
        correct = 0
        total = 0
        
        current_state = states[0]
        
        for i, action in enumerate(actions):
            # Predict next state
            key = (current_state, action)
            if key in self.transitions:
                predicted_next = self.transitions[key]
            else:
                # Fallback: use most common state or current state
                predicted_next = self.fallback_state if self.fallback_state is not None else current_state
            
            predicted_states.append(predicted_next)
            
            # Check accuracy (compare with ground truth)
            true_next = states[i + 1]
            if predicted_next == true_next:
                correct += 1
            total += 1
            
            # Update current state for next prediction
            current_state = predicted_next
        
        accuracy = correct / total if total > 0 else 0.0
        return predicted_states, accuracy
    
    def evaluate_trajectory(self, demos: List[Trajectory], query: Trajectory) -> float:
        """
        Evaluate accuracy on a single trajectory (demos + query).
        
        Args:
            demos: List of demo trajectories
            query: Query trajectory to evaluate
        
        Returns:
            Accuracy (float between 0 and 1)
        """
        self.learn_from_demos(demos)
        _, accuracy = self.predict_query(query)
        return accuracy
    
    def get_learned_fsm(self) -> Optional[FSM]:
        """
        Return the learned FSM as a dictionary structure.
        
        Returns:
            FSM dictionary or None if no demos have been learned
        """
        if not self.transitions:
            return None
        
        # Build FSM structure: state -> {action: next_state}
        fsm: FSM = {}
        
        for (state, action), next_state in self.transitions.items():
            if state not in fsm:
                fsm[state] = {}
            fsm[state][action] = next_state
        
        return fsm

