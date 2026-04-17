"""Blocks World environment for backtracking search experiments."""


class BlocksWorldState:
    """Immutable state of a blocks world configuration.

    Represented as a tuple of stacks, where each stack is a tuple of block IDs
    (bottom to top). Block IDs are integers 0..num_blocks-1.

    Example: 3 stacks, 4 blocks
    stacks = ((0, 1), (2,), (3,))  # Stack 0 has blocks 0,1; Stack 1 has block 2; Stack 2 has block 3
    """

    def __init__(self, stacks: tuple):
        # stacks is a tuple of tuples
        self.stacks = tuple(tuple(s) for s in stacks)
        self.num_stacks = len(stacks)
        self.num_blocks = sum(len(s) for s in stacks)

    def top_blocks(self) -> list:
        """Return list of (block_id, stack_idx) for blocks on top of each non-empty stack."""
        return [(s[-1], i) for i, s in enumerate(self.stacks) if s]

    def is_clear(self, block: int) -> bool:
        """Check if a block has nothing on top of it."""
        for s in self.stacks:
            if s and s[-1] == block:
                return True
        return False

    def find_block(self, block: int) -> tuple:
        """Return (stack_idx, position_in_stack) for a block."""
        for i, s in enumerate(self.stacks):
            for j, b in enumerate(s):
                if b == block:
                    return (i, j)
        raise ValueError(f"Block {block} not found")

    def legal_moves(self) -> list:
        """Return list of (block, from_stack, to_stack) for all legal moves.

        A legal move picks up the top block of any non-empty stack and places it
        on top of any OTHER stack (including empty stacks).
        """
        moves = []
        for i, s in enumerate(self.stacks):
            if not s:
                continue
            block = s[-1]
            for j in range(self.num_stacks):
                if j != i:
                    moves.append((block, i, j))
        return moves

    def apply_move(
        self, block: int, from_stack: int, to_stack: int
    ) -> "BlocksWorldState":
        """Apply a move and return new state."""
        new_stacks = list(list(s) for s in self.stacks)
        assert new_stacks[from_stack] and new_stacks[from_stack][-1] == block
        new_stacks[from_stack].pop()
        new_stacks[to_stack].append(block)
        return BlocksWorldState(tuple(tuple(s) for s in new_stacks))

    def matches(self, goal: "BlocksWorldState") -> bool:
        """Check if this state matches the goal."""
        return self.canonical() == goal.canonical()

    def canonical(self) -> tuple:
        """Canonical representation (sorted stacks for comparison)."""
        return tuple(sorted(self.stacks))

    def __eq__(self, other):
        return self.canonical() == other.canonical()

    def __hash__(self):
        return hash(self.canonical())

    def __repr__(self):
        return f"BlocksWorldState({self.stacks})"


class BlocksWorldEnv:
    """Environment for blocks world planning with backtracking."""

    def __init__(self, num_stacks: int = 3, num_blocks: int = 5):
        self.num_stacks = num_stacks
        self.num_blocks = num_blocks

    def generate_instance(self, rng, min_scramble: int = 3, max_scramble: int = 10):
        """Generate a (start, goal) pair by random scrambling.

        1. Generate a random goal state
        2. Apply min_scramble..max_scramble random moves to get start state
        3. Return (start_state, goal_state, optimal_distance_estimate)
        """
        # Generate random goal: distribute blocks randomly across stacks
        blocks = list(range(self.num_blocks))
        rng.shuffle(blocks)

        # Random partition into stacks
        stacks = [[] for _ in range(self.num_stacks)]
        for b in blocks:
            stacks[rng.randint(0, self.num_stacks - 1)].append(b)
        goal = BlocksWorldState(tuple(tuple(s) for s in stacks))

        # Scramble to get start state
        num_moves = rng.randint(min_scramble, max_scramble)
        current = goal
        for _ in range(num_moves):
            moves = current.legal_moves()
            if moves:
                move = moves[rng.randint(0, len(moves) - 1)]
                current = current.apply_move(*move)

        return current, goal


def dfs_solve(
    start: BlocksWorldState, goal: BlocksWorldState, max_steps: int = 1000
) -> dict:
    """DFS solver with backtracking. Returns trace of (state, action, result).

    Uses deterministic DFS: at each state, try moves in canonical order.
    Tracks visited states to avoid cycles.

    Returns dict with:
        'solved': bool
        'trace': list of dicts with keys:
            'state': BlocksWorldState
            'action': (block, from_stack, to_stack) or None (backtrack)
            'result': 'advance' | 'backtrack' | 'solved'
            'tried_actions': list of actions tried at this state before
        'steps': int
        'backtracks': int
    """
    # Stack-based DFS
    visited = set()
    visited.add(start.canonical())

    # Each entry: (state, move_index, tried_moves)
    stack = [(start, 0, [])]
    trace = []
    steps = 0
    backtracks = 0

    while stack and steps < max_steps:
        state, move_idx, tried = stack[-1]

        if state.matches(goal):
            trace.append(
                {
                    "state": state,
                    "action": None,
                    "result": "solved",
                    "tried_actions": list(tried),
                }
            )
            return {
                "solved": True,
                "trace": trace,
                "steps": steps,
                "backtracks": backtracks,
            }

        moves = state.legal_moves()

        # Find next untried move that leads to unvisited state
        found = False
        while move_idx < len(moves):
            move = moves[move_idx]
            next_state = state.apply_move(*move)
            move_idx += 1

            if next_state.canonical() not in visited:
                # Record advance
                trace.append(
                    {
                        "state": state,
                        "action": move,
                        "result": "advance",
                        "tried_actions": list(tried),
                    }
                )
                tried.append(move)
                stack[-1] = (state, move_idx, tried)

                visited.add(next_state.canonical())
                stack.append((next_state, 0, []))
                steps += 1
                found = True
                break
            else:
                tried.append(move)
                stack[-1] = (state, move_idx, tried)

        if not found:
            # Backtrack
            trace.append(
                {
                    "state": state,
                    "action": None,
                    "result": "backtrack",
                    "tried_actions": list(tried),
                }
            )
            stack.pop()
            backtracks += 1
            steps += 1

    return {
        "solved": False,
        "trace": trace,
        "steps": steps,
        "backtracks": backtracks,
    }
