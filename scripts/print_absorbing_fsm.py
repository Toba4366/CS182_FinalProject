#!/usr/bin/env python3
"""
Print a sample 5-state, 5-action Moore machine with an absorbing state.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.fsm.generator import FSMGenerator, FSMGeneratorConfig
from src.fsm import MAX_STATES, MAX_ACTIONS

def print_fsm(fsm, absorbing_state_id=None):
    """Print an FSM in a readable format."""
    print("\n" + "="*60)
    print("Finite State Machine (FSM)")
    if absorbing_state_id is not None:
        print(f"Absorbing State: {absorbing_state_id}")
    print("="*60)
    print("\nFormat: State -> {Action: NextState, ...}")
    print("-"*60)
    
    for state_id in sorted(fsm.keys()):
        transitions = fsm[state_id]
        marker = " [ABSORBING]" if state_id == absorbing_state_id else ""
        print(f"\nState {state_id}{marker}:")
        print(f"  Transitions:")
        for action_id in sorted(transitions.keys()):
            next_state = transitions[action_id]
            arrow = "→" if next_state == state_id == absorbing_state_id else "→"
            print(f"    Action {action_id} {arrow} State {next_state}")

def find_absorbing_state(fsm):
    """Find the absorbing state in an FSM (if any)."""
    for state_id, transitions in fsm.items():
        # Check if all transitions from this state lead back to itself
        if all(next_state == state_id for next_state in transitions.values()):
            return state_id
    return None

def main():
    # Create generator config for 5 states, 5 actions
    config = FSMGeneratorConfig(
        num_states=5,
        min_actions=5,
        max_actions=5,
        seed=42,  # Fixed seed for reproducibility
    )
    
    generator = FSMGenerator(config)
    
    # Generate FSM with absorbing state
    fsm = generator.generate_with_absorbing_state()
    
    # Find which state is absorbing
    absorbing_state_id = find_absorbing_state(fsm)
    
    # Print the FSM
    print_fsm(fsm, absorbing_state_id)
    
    # Print some statistics
    print("\n" + "="*60)
    print("Statistics:")
    print("="*60)
    print(f"Total states: {len(fsm)}")
    print(f"Total actions per state: {len(list(fsm.values())[0])}")
    print(f"Absorbing state: {absorbing_state_id}")
    
    # Show example trajectory
    print("\n" + "="*60)
    print("Example Trajectory (starting from state 0, length 10):")
    print("="*60)
    state = 0
    trajectory = [state]
    actions = []
    
    import random
    rng = random.Random(42)
    
    for _ in range(10):
        transitions = fsm[state]
        action = rng.choice(list(transitions.keys()))
        next_state = transitions[action]
        actions.append(action)
        trajectory.append(next_state)
        state = next_state
    
    print("States:", trajectory)
    print("Actions:", actions)
    print("\nNote: Once the trajectory reaches the absorbing state, it stays there.")

if __name__ == "__main__":
    main()

