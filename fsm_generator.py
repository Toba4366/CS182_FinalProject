import random
from collections import deque
from typing import List, Tuple, Dict, Optional, Any

State = str
Action = int
TransitionTriples = List[Tuple[State, Action, State]]
FSM = Tuple[TransitionTriples, State]  # (triples, start_state)

def _is_all_reachable(transitions: Dict[Tuple[State, Action], State],
                      states: List[State],
                      actions: List[Action],
                      start: State) -> bool:
    """BFS to ensure every state is reachable from start."""
    seen = {start}
    q = deque([start])
    while q:
        s = q.popleft()
        for a in actions:
            t = transitions[(s, a)]
            if t not in seen:
                seen.add(t)
                q.append(t)
    return len(seen) == len(states)

def _has_absorption_state(transitions: Dict[Tuple[State, Action], State],
                          states: List[State],
                          actions: List[Action]) -> bool:
    """True if any state is absorbing (all actions lead to itself)."""
    for s in states:
        if all(transitions[(s, a)] == s for a in actions):
            return True
    return False

def _ensure_no_absorbing(transitions: Dict[Tuple[State, Action], State],
                         states: List[State],
                         actions: List[Action],
                         rng: random.Random) -> None:
    """Modify transitions in-place so that no state is absorbing."""
    for s in states:
        if all(transitions[(s, a)] == s for a in actions):
            # Pick one action and redirect to a different state
            a = rng.choice(actions)
            t = rng.choice([x for x in states if x != s])
            transitions[(s, a)] = t

def _build_one_fsm(avoid_absorbing: bool, rng: random.Random) -> FSM:
    """
    Build one DFA with:
      - number of states in {4, 5}
      - number of actions k in [2, 8]
      - deterministic (every (state, action) has exactly one successor)
      - every state reachable from the start
      - if avoid_absorbing=True, no absorption states
    Returns (triples, start_state).
    """
    # Choose states and action set
    n_states = rng.choice([4, 5])
    k_actions = rng.randint(2, 8)  # interpret "number of transitions varies 2-8" as |A|
    states = [f"s{i}" for i in range(n_states)]
    # Use actions labeled 1..k for readability
    actions = list(range(1, k_actions + 1))
    start = states[0]

    # We'll attempt until constraints are satisfied (rarely loops more than once)
    while True:
        # Initialize empty transition function
        transitions: Dict[Tuple[State, Action], State] = {}

        # Step 1: Create a random spanning arborescence (ensures reachability).
        # For each non-start state, pick a parent from the connected set and an action
        # that isn't used yet by that parent (if all are used, we can reuse an action;
        # determinism is preserved since we set exactly one successor per (s,a)).
        connected = [start]
        remaining = states[1:].copy()
        rng.shuffle(remaining)

        for child in remaining:
            parent = rng.choice(connected)
            # Prefer an unused action from parent to make structure diverse
            unused_actions = [a for a in actions if (parent, a) not in transitions]
            if unused_actions:
                a = rng.choice(unused_actions)
            else:
                a = rng.choice(actions)
            transitions[(parent, a)] = child
            connected.append(child)

        # Step 2: Fill in any missing (state, action) entries with random targets.
        for s in states:
            for a in actions:
                if (s, a) not in transitions:
                    transitions[(s, a)] = rng.choice(states)

        # Step 3: If no-absorption requested, fix any absorbing states.
        if avoid_absorbing:
            _ensure_no_absorbing(transitions, states, actions, rng)

        # Step 4: Verify reachability (and no-absorption if requested).
        if not _is_all_reachable(transitions, states, actions, start):
            continue
        if avoid_absorbing and _has_absorption_state(transitions, states, actions):
            continue

        # Build triples in a stable order
        triples: TransitionTriples = []
        for s in states:
            for a in actions:
                triples.append((s, a, transitions[(s, a)]))
        return triples, start

def generate_random_fsms(n: int, seed: Optional[int] = None) -> List[FSM]:
    """
    Generate n deterministic FSMs.
    Each FSM is returned as (triples, start_state), where:
      - triples is a list of (S, A, S') for every state-action pair
      - start_state is the start state's label (e.g., 's0')
    Notes / alignment with your spec:
      - Number of states is 4 or 5 (chosen at random per FSM).
      - Number of actions varies from 2 to 8 (chosen at random per FSM).
      - "All states have all transitions" → the DFA is total over its action set.
      - Absorbing states are allowed in this version.
      - Every state is reachable from the start state.
    """
    rng = random.Random(seed)
    return [_build_one_fsm(avoid_absorbing=False, rng=rng) for _ in range(n)]

def generate_random_fsms_no_absorption(n: int, seed: Optional[int] = None) -> List[FSM]:
    """
    Same as generate_random_fsms, but guarantees there are NO absorption states
    (i.e., for every state s, not all actions send s back to itself).
    """
    rng = random.Random(seed)
    return [_build_one_fsm(avoid_absorbing=True, rng=rng) for _ in range(n)]


from typing import Dict, List, Tuple, Any

class FSM:
    """
    FSM represented as:
    {
        state1: { action1: next_state, action2: next_state, ... },
        state2: { ... },
        ...
    }
    """

    def __init__(self, triples: List[Tuple[str, int, str]], start_state: str):
        """
        triples: list of (S, A, S') tuples
        start_state: starting state label
        """
        self.start_state = start_state
        self.states = set()
        self.actions = set()

        # Nested dictionary: state -> {action: next_state}
        self.fsm: Dict[str, Dict[int, str]] = {}

        for s, a, s_next in triples:
            self.states.add(s)
            self.states.add(s_next)
            self.actions.add(a)

            if s not in self.fsm:
                self.fsm[s] = {}
            self.fsm[s][a] = s_next

    def step(self, state: str, action: int) -> str:
        """Return next state given a state and action."""
        return self.fsm[state][action]

    def solve(self, actions: List[int]) -> str:
        """
        Execute a sequence of actions starting from start_state.
        Returns final state.
        """
        current = self.start_state
        for a in actions:
            current = self.step(current, a)
        return current

    @classmethod
    def from_random(cls, avoid_absorption: bool = False, seed=None):
        """
        Build a random FSM using the previously defined generator.
        """
        from_random_list = generate_random_fsms_no_absorption if avoid_absorption else generate_random_fsms
        triples, start = from_random_list(1, seed=seed)[0]
        return cls(triples, start)

    def __repr__(self):
        return f"FSM(start={self.start_state}, transitions={self.fsm})"
