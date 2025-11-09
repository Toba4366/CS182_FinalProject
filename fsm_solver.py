"""
FSM solver and sequence generator with truncation capabilities.

Works with fsm_generator.py to create sequences that start from different points
in the FSM execution, simulating partial observation scenarios.
"""

import random
from typing import List, Tuple, Dict, Optional, Union
from fsm_generator import FSM, State, Action, TransitionTriples


def solve_fsm(fsm: FSM, action_sequence: List[Action]) -> List[Tuple[State, Action, State]]:
    """
    Solve an FSM by executing a sequence of actions and return the state transitions.
    
    Args:
        fsm: Tuple of (transition_triples, start_state)
        action_sequence: List of actions to execute
        
    Returns:
        List of (current_state, action, next_state) tuples showing the execution path
    """
    triples, start_state = fsm
    
    # Build transition dictionary for fast lookup
    transitions: Dict[Tuple[State, Action], State] = {}
    for s, a, s_next in triples:
        transitions[(s, a)] = s_next
    
    # Execute the sequence
    current_state = start_state
    execution_path = []
    
    for action in action_sequence:
        if (current_state, action) not in transitions:
            raise ValueError(f"No transition defined for state {current_state}, action {action}")
        
        next_state = transitions[(current_state, action)]
        execution_path.append((current_state, action, next_state))
        current_state = next_state
    
    return execution_path


def generate_full_sequence(fsm: FSM, 
                          sequence_length: int, 
                          seed: Optional[int] = None) -> Tuple[List[Action], List[Tuple[State, Action, State]]]:
    """
    Generate a random action sequence and solve it through the FSM.
    
    Args:
        fsm: Tuple of (transition_triples, start_state)
        sequence_length: Length of action sequence to generate
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (action_sequence, execution_path)
    """
    if seed is not None:
        random.seed(seed)
    
    triples, start_state = fsm
    
    # Extract available actions from the FSM
    actions = set()
    states = set()
    transitions: Dict[Tuple[State, Action], State] = {}
    
    for s, a, s_next in triples:
        actions.add(a)
        states.add(s)
        states.add(s_next)
        transitions[(s, a)] = s_next
    
    actions = list(actions)
    states = list(states)
    
    # Generate random action sequence, ensuring each action is valid from some state
    action_sequence = []
    current_state = start_state
    
    for _ in range(sequence_length):
        # Find valid actions from current state
        valid_actions = [a for a in actions if (current_state, a) in transitions]
        
        if not valid_actions:
            # Fallback: pick any action (shouldn't happen with well-formed FSMs)
            action = random.choice(actions)
        else:
            action = random.choice(valid_actions)
        
        action_sequence.append(action)
        current_state = transitions.get((current_state, action), current_state)
    
    # Solve the sequence
    execution_path = solve_fsm(fsm, action_sequence)
    
    return action_sequence, execution_path


def truncate_sequence(execution_path: List[Tuple[State, Action, State]], 
                     truncate_mode: str = "random",
                     truncate_amount: Optional[int] = None,
                     seed: Optional[int] = None) -> Tuple[List[Tuple[State, Action, State]], str]:
    """
    Truncate a sequence according to the specified strategy.
    
    Args:
        execution_path: Full execution path from solve_fsm
        truncate_mode: "random" (default), "start_state", "action", "non_start_state"
        truncate_amount: Number of elements to remove from start (0-3 if None)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (truncated_path, truncation_info)
        
    Truncation Strategy:
    - 25% start with start state (remove 0-3 state-action-state triples)
    - 50% start with action (remove partial first triple, start from action)
    - 25% start with non-start state (start from arbitrary state)
    """
    if seed is not None:
        random.seed(seed)
    
    if not execution_path:
        return execution_path, "empty_sequence"
    
    # Determine truncation amount
    if truncate_amount is None:
        truncate_amount = random.randint(0, 3)
    else:
        truncate_amount = max(0, min(3, truncate_amount))
    
    # Determine truncation mode if random
    if truncate_mode == "random":
        mode_choice = random.random()
        if mode_choice < 0.25:
            truncate_mode = "start_state"
        elif mode_choice < 0.75:
            truncate_mode = "action"
        else:
            truncate_mode = "non_start_state"
    
    truncation_info = f"mode={truncate_mode}, amount={truncate_amount}"
    
    if truncate_mode == "start_state":
        # 25% of the time: start with start state (remove n complete triples)
        truncated = execution_path[truncate_amount:]
        return truncated, f"start_state_removal({truncation_info})"
    
    elif truncate_mode == "action":
        # 50% of the time: start with an action (may or may not branch from start)
        if truncate_amount >= len(execution_path):
            # If truncating more than available, return last action
            if execution_path:
                last_triple = execution_path[-1]
                return [(last_triple[0], last_triple[1], last_triple[2])], f"action_start_fallback({truncation_info})"
            return [], f"action_start_empty({truncation_info})"
        
        # Start from the action of the truncate_amount-th triple
        truncated = execution_path[truncate_amount:]
        return truncated, f"action_start({truncation_info})"
    
    elif truncate_mode == "non_start_state":
        # 25% of the time: start with a state that isn't necessarily the start state
        if truncate_amount >= len(execution_path):
            return [], f"non_start_empty({truncation_info})"
        
        # Start from the end state of the truncate_amount-th triple
        if truncate_amount < len(execution_path):
            truncated = execution_path[truncate_amount:]
            return truncated, f"non_start_state({truncation_info})"
        
        return execution_path, f"non_start_fallback({truncation_info})"
    
    else:
        raise ValueError(f"Unknown truncate_mode: {truncate_mode}")


def generate_truncated_sequence(fsm: FSM,
                               sequence_length: int,
                               truncate_mode: str = "random",
                               truncate_amount: Optional[int] = None,
                               seed: Optional[int] = None) -> Tuple[List[Action], 
                                                                  List[Tuple[State, Action, State]], 
                                                                  List[Tuple[State, Action, State]], 
                                                                  str]:
    """
    Generate a full sequence and then truncate it according to the strategy.
    
    Args:
        fsm: FSM to execute on
        sequence_length: Length of initial sequence to generate
        truncate_mode: Truncation strategy
        truncate_amount: Amount to truncate (0-3 if None)
        seed: Random seed
        
    Returns:
        Tuple of (original_actions, full_execution_path, truncated_path, truncation_info)
    """
    # Generate full sequence
    actions, full_path = generate_full_sequence(fsm, sequence_length, seed)
    
    # Truncate it
    truncated_path, truncation_info = truncate_sequence(
        full_path, truncate_mode, truncate_amount, seed
    )
    
    return actions, full_path, truncated_path, truncation_info


def format_sequence_for_training(truncated_path: List[Tuple[State, Action, State]], 
                                format_type: str = "state_action_state") -> List[str]:
    """
    Format a truncated sequence for training data.
    
    Args:
        truncated_path: Sequence from truncate_sequence
        format_type: "state_action_state", "action_only", "state_only"
        
    Returns:
        List of formatted tokens
    """
    if not truncated_path:
        return []
    
    tokens = []
    
    if format_type == "state_action_state":
        for state, action, next_state in truncated_path:
            tokens.extend([f"S{state}", f"A{action}", f"S{next_state}"])
    
    elif format_type == "action_only":
        for state, action, next_state in truncated_path:
            tokens.append(f"A{action}")
    
    elif format_type == "state_only":
        # Include all states in the path
        if truncated_path:
            # Start with initial state of first transition
            tokens.append(f"S{truncated_path[0][0]}")
            for state, action, next_state in truncated_path:
                tokens.append(f"S{next_state}")
    
    else:
        raise ValueError(f"Unknown format_type: {format_type}")
    
    return tokens


def batch_generate_sequences(fsms: List[FSM],
                           sequences_per_fsm: int = 5,
                           sequence_length: int = 10,
                           truncate_mode: str = "random",
                           seed: Optional[int] = None) -> List[Dict]:
    """
    Generate multiple truncated sequences for a batch of FSMs.
    
    Args:
        fsms: List of FSMs to generate sequences for
        sequences_per_fsm: Number of sequences per FSM
        sequence_length: Length of each sequence
        truncate_mode: Truncation strategy
        seed: Random seed
        
    Returns:
        List of dictionaries containing sequence data
    """
    if seed is not None:
        random.seed(seed)
    
    results = []
    
    for fsm_idx, fsm in enumerate(fsms):
        for seq_idx in range(sequences_per_fsm):
            # Use different seed for each sequence
            seq_seed = None if seed is None else seed + fsm_idx * 1000 + seq_idx
            
            actions, full_path, truncated_path, truncation_info = generate_truncated_sequence(
                fsm, sequence_length, truncate_mode, seed=seq_seed
            )
            
            # Format for different training purposes
            formatted_sas = format_sequence_for_training(truncated_path, "state_action_state")
            formatted_actions = format_sequence_for_training(truncated_path, "action_only")
            formatted_states = format_sequence_for_training(truncated_path, "state_only")
            
            results.append({
                'fsm_idx': fsm_idx,
                'sequence_idx': seq_idx,
                'original_actions': actions,
                'full_path': full_path,
                'truncated_path': truncated_path,
                'truncation_info': truncation_info,
                'formatted_sas': formatted_sas,
                'formatted_actions': formatted_actions,
                'formatted_states': formatted_states,
                'sequence_length': len(truncated_path)
            })
    
    return results


# Example usage and testing
if __name__ == "__main__":
    from fsm_generator import generate_random_fsms
    
    # Generate a few test FSMs
    test_fsms = generate_random_fsms(2, seed=42)
    
    print("Testing FSM solver and sequence truncation:")
    print("=" * 50)
    
    for i, fsm in enumerate(test_fsms):
        print(f"\nFSM {i}:")
        triples, start = fsm
        print(f"Start state: {start}")
        print(f"Transitions: {triples[:3]}...")  # Show first few transitions
        
        # Generate and truncate sequences with different modes
        modes = ["start_state", "action", "non_start_state"]
        
        for mode in modes:
            actions, full_path, truncated_path, info = generate_truncated_sequence(
                fsm, sequence_length=8, truncate_mode=mode, seed=42+i
            )
            
            formatted = format_sequence_for_training(truncated_path, "state_action_state")
            
            print(f"\n  Mode: {mode}")
            print(f"    {info}")
            print(f"    Original length: {len(full_path)}, Truncated: {len(truncated_path)}")
            print(f"    Formatted: {' '.join(formatted[:10])}{'...' if len(formatted) > 10 else ''}")