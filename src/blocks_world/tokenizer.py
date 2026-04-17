"""Tokenizer for Blocks World search traces."""


class BlocksWorldTokenizer:
    """Encodes Blocks World states and actions as token sequences.

    Token layout:
        0: PAD
        1: BOS
        2: EOS
        3: SEP
        4: STATE (start of state description)
        5: GOAL (start of goal description)
        6: MOVE (action marker)
        7: BACKTRACK
        8: SOLVED
        9: FAILED
        10: TRIED (marks tried actions)
        11: END_TRIED
        12: STACK (stack separator within state)
        13: EMPTY_STACK
        14-19: reserved
        20-29: BLOCK_0 through BLOCK_9 (block IDs)
        30-39: STACK_0 through STACK_9 (stack indices)
        40-49: reserved for future
    """

    PAD = 0
    BOS = 1
    EOS = 2
    SEP = 3
    STATE = 4
    GOAL = 5
    MOVE = 6
    BACKTRACK = 7
    SOLVED = 8
    FAILED = 9
    TRIED = 10
    END_TRIED = 11
    STACK = 12
    EMPTY_STACK = 13

    BLOCK_OFFSET = 20
    STACK_OFFSET = 30

    VOCAB_SIZE = 50  # plenty of room

    def __init__(self, num_blocks: int = 6, num_stacks: int = 3):
        self.num_blocks = num_blocks
        self.num_stacks = num_stacks

    def block_token(self, block_id: int) -> int:
        return self.BLOCK_OFFSET + block_id

    def stack_token(self, stack_id: int) -> int:
        return self.STACK_OFFSET + stack_id

    def decode_token(self, token_id: int) -> str:
        names = {
            0: "PAD",
            1: "BOS",
            2: "EOS",
            3: "SEP",
            4: "STATE",
            5: "GOAL",
            6: "MOVE",
            7: "BT",
            8: "SOLVED",
            9: "FAILED",
            10: "TRIED",
            11: "END_TRIED",
            12: "STACK",
            13: "EMPTY",
        }
        if token_id in names:
            return names[token_id]
        if self.BLOCK_OFFSET <= token_id < self.BLOCK_OFFSET + 10:
            return f"B{token_id - self.BLOCK_OFFSET}"
        if self.STACK_OFFSET <= token_id < self.STACK_OFFSET + 10:
            return f"S{token_id - self.STACK_OFFSET}"
        return f"?{token_id}"

    def encode_state(self, state) -> list:
        """Encode a BlocksWorldState as tokens.

        Format: [STATE] [STACK_0 block block ... | STACK_1 block block ... | ...]
        Empty stacks: [STACK_i EMPTY_STACK]
        """
        tokens = [self.STATE]
        for i, s in enumerate(state.stacks):
            tokens.append(self.stack_token(i))
            if not s:
                tokens.append(self.EMPTY_STACK)
            else:
                for b in s:
                    tokens.append(self.block_token(b))
        tokens.append(self.SEP)
        return tokens

    def encode_goal(self, goal) -> list:
        """Encode goal state. Same as encode_state but with GOAL marker."""
        tokens = [self.GOAL]
        for i, s in enumerate(goal.stacks):
            tokens.append(self.stack_token(i))
            if not s:
                tokens.append(self.EMPTY_STACK)
            else:
                for b in s:
                    tokens.append(self.block_token(b))
        tokens.append(self.SEP)
        return tokens

    def encode_move(self, block: int, from_stack: int, to_stack: int) -> list:
        """Encode a move action."""
        return [
            self.MOVE,
            self.block_token(block),
            self.stack_token(from_stack),
            self.stack_token(to_stack),
        ]

    def encode_tried(self, tried_actions: list) -> list:
        """Encode tried actions at revisited state."""
        if not tried_actions:
            return []
        tokens = [self.TRIED]
        for block, from_s, to_s in tried_actions:
            tokens.extend(
                [
                    self.block_token(block),
                    self.stack_token(from_s),
                    self.stack_token(to_s),
                ]
            )
        tokens.append(self.END_TRIED)
        return tokens

    def build_prefix(self, goal) -> list:
        """Build the fixed prefix: [BOS] [GOAL encoding]"""
        return [self.BOS] + self.encode_goal(goal)

    def trace_to_sequence(self, goal, trace: list) -> tuple:
        """Convert a DFS trace to a token sequence with loss mask.

        Returns: (sequence: list[int], loss_mask: list[bool])

        The sequence format for each step:
        [TRIED actions END_TRIED]  (if state was revisited / has tried actions)
        [STATE encoding]
        [MOVE block from to] or [BACKTRACK] or [SOLVED]

        Loss mask is True only for MOVE tokens (the model should predict the action).
        """
        prefix = self.build_prefix(goal)
        sequence = list(prefix)
        loss_mask = [False] * len(prefix)

        for entry in trace:
            state = entry["state"]
            action = entry["action"]
            result = entry["result"]
            tried = entry.get("tried_actions", [])

            # Encode tried actions (if any)
            if tried:
                tried_tokens = self.encode_tried(tried)
                sequence.extend(tried_tokens)
                loss_mask.extend([False] * len(tried_tokens))

            # Encode current state
            state_tokens = self.encode_state(state)
            sequence.extend(state_tokens)
            loss_mask.extend([False] * len(state_tokens))

            # Encode action/result
            if result == "advance" and action is not None:
                move_tokens = self.encode_move(*action)
                sequence.extend(move_tokens)
                # Loss mask: predict MOVE token + block + from + to
                loss_mask.extend([True] * len(move_tokens))
            elif result == "backtrack":
                sequence.append(self.BACKTRACK)
                loss_mask.append(True)  # predict backtrack decision
            elif result == "solved":
                sequence.append(self.SOLVED)
                loss_mask.append(True)

        sequence.append(self.EOS)
        loss_mask.append(False)

        return sequence, loss_mask
