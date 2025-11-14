"""
FSM solver and sequence generator with truncation capabilities.

Works with fsm_generator.py to create sequences that start from different points
in the FSM execution, simulating partial observation scenarios.

Updated to work with both the original tuple format and the new FSM class,
and aligned with project plan specifications (256 context, 64 sequence length).
"""

import random
from typing import List, Tuple, Dict, Optional, Union, Any
from fsm_generator import generate_random_fsms, generate_random_fsms_no_absorption

# Type aliases for backward compatibility
State = str
Action = int
TransitionTriples = List[Tuple[State, Action, State]]
FSMTuple = Tuple[TransitionTriples, State]  # Original tuple format
FSMUniversal = Union[Any, FSMTuple]  # Either FSM class or tuple format


def solve_fsm_universal(fsm: FSMUniversal, action_sequence: List[Action]) -> List[Tuple[State, Action, State]]:
    """
    Universal FSM solver that works with both tuple format and FSM class.
    
    Args:
        fsm: Either (transition_triples, start_state) tuple or FSM class instance
        action_sequence: List of actions to execute
        
    Returns:
        List of (current_state, action, next_state) tuples showing the execution path
    """
    if isinstance(fsm, tuple):
        # Original tuple format: (triples, start_state)
        return solve_fsm_tuple(fsm, action_sequence)
    elif hasattr(fsm, 'solve'):
        # New FSM class format
        return solve_fsm_class(fsm, action_sequence)
    else:
        raise ValueError(f"Unknown FSM format: {type(fsm)}")


def solve_fsm_tuple(fsm: FSMTuple, action_sequence: List[Action]) -> List[Tuple[State, Action, State]]:
    """
    Solve an FSM in tuple format by executing a sequence of actions.
    
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


def solve_fsm_class(fsm: Any, action_sequence: List[Action]) -> List[Tuple[State, Action, State]]:
    """
    Solve an FSM class instance by executing a sequence of actions and building full path.
    
    Args:
        fsm: FSM class instance with .step() and .start_state
        action_sequence: List of actions to execute
        
    Returns:
        List of (current_state, action, next_state) tuples showing the execution path
    """
    current_state = fsm.start_state
    execution_path = []
    
    for action in action_sequence:
        if current_state not in fsm.fsm or action not in fsm.fsm[current_state]:
            raise ValueError(f"No transition defined for state {current_state}, action {action}")
        
        next_state = fsm.step(current_state, action)
        execution_path.append((current_state, action, next_state))
        current_state = next_state
    
    return execution_path


# Backward compatibility alias
def solve_fsm(fsm: FSMTuple, action_sequence: List[Action]) -> List[Tuple[State, Action, State]]:
    """Legacy function for backward compatibility - use solve_fsm_universal instead."""
    return solve_fsm_tuple(fsm, action_sequence)


def generate_full_sequence(fsm: FSMUniversal, 
                          sequence_length: int = 64,  # Updated to match plan specification 
                          seed: Optional[int] = None) -> Tuple[List[Action], List[Tuple[State, Action, State]]]:
    """
    Generate a random action sequence and solve it through the FSM.
    
    Args:
        fsm: Either tuple (transition_triples, start_state) or FSM class instance
        sequence_length: Length of action sequence to generate (default 64 per plan)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (action_sequence, execution_path)
    """
    if seed is not None:
        random.seed(seed)
    
    # Handle both FSM formats
    if isinstance(fsm, tuple):
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
        current_state = start_state
    else:
        # FSM class instance
        actions = list(fsm.actions)
        current_state = fsm.start_state
        transitions = {}
        for state in fsm.states:
            for action in fsm.actions:
                if state in fsm.fsm and action in fsm.fsm[state]:
                    transitions[(state, action)] = fsm.fsm[state][action]
    
    # Generate random action sequence, ensuring each action is valid from current state
    action_sequence = []
    
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
    
    # Solve the sequence using universal solver
    execution_path = solve_fsm_universal(fsm, action_sequence)
    
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
    - 25% start with start state (ensure sequence begins from s0)
    - 50% start with action (begin mid-triple from an action)
    - 25% start with non-start state (begin from state that isn't s0)
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
        # 25% of the time: ensure we start with start state (s0)
        # Find a position where we can start from s0
        start_state = execution_path[0][0] if execution_path else None
        
        # Look for s0 after truncate_amount position
        for i in range(truncate_amount, len(execution_path)):
            if execution_path[i][0] == start_state:  # Found s0 as current state
                truncated = execution_path[i:]
                return truncated, f"start_state_s0({truncation_info})"
        
        # Fallback: if s0 not found, use original truncation
        truncated = execution_path[truncate_amount:] if truncate_amount < len(execution_path) else []
        return truncated, f"start_state_fallback({truncation_info})"
    
    elif truncate_mode == "action":
        # 50% of the time: start with an action (mid-triple)
        if truncate_amount >= len(execution_path):
            # If truncating more than available, return last action
            if execution_path:
                last_triple = execution_path[-1]
                return [(last_triple[0], last_triple[1], last_triple[2])], f"action_start_fallback({truncation_info})"
            return [], f"action_start_empty({truncation_info})"
        
        # Start from the specified position, but format to begin with action
        start_idx = truncate_amount
        if start_idx < len(execution_path):
            # Create sequence that starts with action from the chosen triple
            chosen_triple = execution_path[start_idx]
            current_state, action, next_state = chosen_triple
            
            # Build new sequence starting with this action
            truncated = [(current_state, action, next_state)]
            # Add remaining triples
            truncated.extend(execution_path[start_idx + 1:])
            
            return truncated, f"action_start({truncation_info})"
        
        return [], f"action_start_empty({truncation_info})"
    
    elif truncate_mode == "non_start_state":
        # 25% of the time: start with a state that isn't the start state
        start_state = execution_path[0][0] if execution_path else None
        
        # Look for first occurrence of non-start state after truncate_amount
        for i in range(truncate_amount, len(execution_path)):
            if execution_path[i][0] != start_state:  # Found non-s0 state
                truncated = execution_path[i:]
                return truncated, f"non_start_state({truncation_info})"
        
        # If all remaining states are s0, look at next_state in transitions
        for i in range(truncate_amount, len(execution_path)):
            if execution_path[i][2] != start_state:  # next_state is not s0
                # Start from this transition but focus on the non-s0 result
                truncated = execution_path[i:]
                return truncated, f"non_start_via_transition({truncation_info})"
        
        # Fallback: use regular truncation if no non-s0 state found
        truncated = execution_path[truncate_amount:] if truncate_amount < len(execution_path) else []
        return truncated, f"non_start_fallback({truncation_info})"
    
    else:
        raise ValueError(f"Unknown truncate_mode: {truncate_mode}")


def generate_truncated_sequence(fsm: FSMUniversal,
                               sequence_length: int = 64,  # Updated to match plan
                               truncate_mode: str = "random",
                               truncate_amount: Optional[int] = None,
                               seed: Optional[int] = None) -> Tuple[List[Action], 
                                                                  List[Tuple[State, Action, State]], 
                                                                  List[Tuple[State, Action, State]], 
                                                                  str]:
    """
    Generate a full sequence and then truncate it according to the strategy.
    
    Args:
        fsm: Either FSM tuple or class instance
        sequence_length: Length of initial sequence to generate (default 64 per plan)
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
                                format_type: str = "action_state") -> List[str]:
    """
    Format a truncated sequence for training data.
    Updated to match plan specification: (A, S) pairs format.
    
    Args:
        truncated_path: Sequence from truncate_sequence
        format_type: "action_state" (A,S pairs per plan), "state_action_state", "action_only", "state_only"
        
    Returns:
        List of formatted tokens
    """
    if not truncated_path:
        return []
    
    tokens = []
    
    if format_type == "action_state":
        # Plan specification: (Action, State) pairs - this is the new default
        for state, action, next_state in truncated_path:
            tokens.extend([f"A{action}", f"S{next_state}"])
    
    elif format_type == "state_action_state":
        # Original format: State-Action-State triples
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


def batch_generate_sequences(fsms: List[FSMUniversal],
                           sequences_per_fsm: int = 5,
                           sequence_length: int = 64,  # Updated to match plan
                           truncate_mode: str = "random",
                           max_context_length: int = 256,  # Added context limit per plan
                           seed: Optional[int] = None) -> List[Dict]:
    """
    Generate multiple truncated sequences for a batch of FSMs.
    Updated to align with plan specifications.
    
    Args:
        fsms: List of FSMs (either tuple or class format)
        sequences_per_fsm: Number of sequences per FSM
        sequence_length: Length of each sequence (default 64 per plan)
        truncate_mode: Truncation strategy
        max_context_length: Maximum context window (default 256 per plan)
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
            formatted_as = format_sequence_for_training(truncated_path, "action_state")     # Plan default
            formatted_sas = format_sequence_for_training(truncated_path, "state_action_state")
            formatted_actions = format_sequence_for_training(truncated_path, "action_only")
            formatted_states = format_sequence_for_training(truncated_path, "state_only")
            
            # Check context length compliance
            context_length_check = len(formatted_as) <= max_context_length
            
            results.append({
                'fsm_idx': fsm_idx,
                'sequence_idx': seq_idx,
                'original_actions': actions,
                'full_path': full_path,
                'truncated_path': truncated_path,
                'truncation_info': truncation_info,
                'formatted_action_state': formatted_as,      # Plan default format
                'formatted_sas': formatted_sas,
                'formatted_actions': formatted_actions,
                'formatted_states': formatted_states,
                'sequence_length': len(truncated_path),
                'context_length': len(formatted_as),
                'context_fits': context_length_check
            })
    
    return results


# Example usage and testing
if __name__ == "__main__":
    from fsm_generator import generate_random_fsms, FSM
    
    print("Testing updated FSM solver with both tuple and class formats:")
    print("=" * 60)
    
    # Test with original tuple format
    print("\n1. Testing with tuple format (original):")
    test_fsms_tuple = generate_random_fsms(1, seed=42)
    fsm_tuple = test_fsms_tuple[0]
    
    actions, full_path, truncated_path, info = generate_truncated_sequence(
        fsm_tuple, sequence_length=8, truncate_mode="random", seed=42
    )
    
    formatted = format_sequence_for_training(truncated_path, "action_state")
    print(f"Tuple format result: {' '.join(formatted[:10])}{'...' if len(formatted) > 10 else ''}")
    print(f"Context length: {len(formatted)}/256")
    
    # Test with new FSM class format
    print("\n2. Testing with FSM class format (new):")
    fsm_class = FSM.from_random(avoid_absorption=False, seed=42)
    
    actions, full_path, truncated_path, info = generate_truncated_sequence(
        fsm_class, sequence_length=8, truncate_mode="random", seed=42
    )
    
    formatted = format_sequence_for_training(truncated_path, "action_state")
    print(f"Class format result: {' '.join(formatted[:10])}{'...' if len(formatted) > 10 else ''}")
    print(f"Context length: {len(formatted)}/256")
    
    # Test batch generation with plan specifications
    print("\n3. Testing batch generation with plan specifications:")
    mixed_fsms = [fsm_tuple, fsm_class]  # Mix of formats
    batch_results = batch_generate_sequences(
        mixed_fsms, 
        sequences_per_fsm=2, 
        sequence_length=64,      # Plan specification
        max_context_length=256,  # Plan specification
        seed=42
    )
    
    print(f"Generated {len(batch_results)} sequences")
    for i, result in enumerate(batch_results[:2]):  # Show first 2
        print(f"  Sequence {i}: length={result['context_length']}, fits_context={result['context_fits']}")
        tokens = result['formatted_action_state'][:6]  # Show first 6 tokens
        print(f"    Tokens: {' '.join(tokens)}...")
    
    print(f"\n✅ Updated FSM solver ready for {len([r for r in batch_results if r['context_fits']])} " +
          f"/ {len(batch_results)} sequences that fit in 256-token context")