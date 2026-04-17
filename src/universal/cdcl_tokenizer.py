"""Tokenizer for CDCL-style conflict analysis sequences.

This tokenizer maps graph coloring conflict states into a small, fixed
vocabulary suitable for decoder-only transformers.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class CDCLTokenizer:
    """Tokenizer for serialized CDCL conflict states and scratchpads."""

    VOCAB_SIZE = 239
    COMPRESSED_VOCAB_SIZE = 540
    SLIM_VOCAB_SIZE = 556
    SORTED_SLIM_VOCAB_SIZE = 562
    RLE_BUCKET_VOCAB_SIZE = 562
    DS_VOCAB_SIZE = 562
    WITNESS_VOCAB_SIZE = 565
    PDL_VOCAB_SIZE = 569
    PROP_VOCAB_SIZE = 572
    TRIED_VOCAB_SIZE = 574

    # Special tokens
    PAD = 0
    BOS = 1
    EOS = 2
    SEP = 3

    # Section markers
    GRAPH_START = 4
    STACK_START = 5
    CONFLICT_START = 6
    THINK_START = 7
    TARGET_START = 8

    # Operators / keywords
    COLON = 9
    ARROW = 10
    BEST = 11
    DEPTHS = 12
    SECOND = 13

    # Search action tokens
    ASSIGN = 14
    BACKJUMP = 15
    SOLVED = 16
    FAILED = 17
    SEARCH_START = 18
    CHECK = 19

    # Interleaved search tokens
    CANDIDATES = 233
    DECIDE = 234
    DOMAIN = 235
    PICK = 236
    RESULT = 237
    OK = 238

    # Combined ASSIGN tokens
    ASSIGN_OFFSET = 240

    # Slim domain mask tokens
    MASK_OFFSET = 540

    # Sorted slim tokens
    STATE = 556
    DS0 = 557
    DS1 = 558
    DS2 = 559
    DS3 = 560
    DS4 = 561

    # Witness tokens
    WITNESS = 562
    WITNESS_NONE = 563
    CF = 564
    LEDGER = 567
    ACT_TOKEN = 568

    # Propagation scratchpad tokens
    PROP = 569
    ENDPROP = 570
    FREE = 571
    TRIED = 572
    END_TRIED = 573

    # Offsets for structured tokens
    NODE_OFFSET = 20
    COLOR_OFFSET = 120
    DEPTH_OFFSET = 130

    # NONE / EMPTY / NO_CONFLICT tokens
    NONE_TOKEN = 230
    EMPTY_DOMAIN = 231
    NO_CONFLICT = 232

    MAX_NODES = 75
    MAX_COLORS = 4
    MAX_DEPTH = 100

    _SPECIAL_TOKEN_STRINGS: Dict[int, str] = {
        PAD: "[PAD]",
        BOS: "[BOS]",
        EOS: "[EOS]",
        SEP: "SEP",
        GRAPH_START: "[GRAPH]",
        STACK_START: "[STACK]",
        CONFLICT_START: "[CONFLICT]",
        THINK_START: "[THINK]",
        TARGET_START: "[TARGET]",
        COLON: ":",
        ARROW: "->",
        BEST: "best",
        DEPTHS: "depths",
        SECOND: "second",
        ASSIGN: "ASSIGN",
        BACKJUMP: "BACKJUMP",
        SOLVED: "SOLVED",
        FAILED: "FAILED",
        SEARCH_START: "[SEARCH]",
        CHECK: "[CHECK]",
        CANDIDATES: "CANDIDATES",
        DECIDE: "DECIDE",
        DOMAIN: "DOMAIN",
        PICK: "PICK",
        RESULT: "RESULT",
        OK: "OK",
        STATE: "STATE",
        DS0: "DS0",
        DS1: "DS1",
        DS2: "DS2",
        DS3: "DS3",
        DS4: "DS4",
        WITNESS: "W",
        WITNESS_NONE: "NONE",
        CF: "CF",
        LEDGER: "[D]",
        ACT_TOKEN: "[ACT]",
        PROP: "[PROP]",
        ENDPROP: "[/PROP]",
        FREE: "FREE",
        TRIED: "TRIED",
        END_TRIED: "END_TRIED",
        NONE_TOKEN: "NONE",
        EMPTY_DOMAIN: "EMPTY",
        NO_CONFLICT: "NO_CONFLICT",
    }

    @staticmethod
    def node_token(node_id: int) -> int:
        """Convert node ID to token."""
        node_id = int(node_id)
        if node_id < 0 or node_id >= CDCLTokenizer.MAX_NODES:
            raise ValueError(f"node_id out of range: {node_id}")
        return CDCLTokenizer.NODE_OFFSET + node_id

    @staticmethod
    def color_token(color: int) -> int:
        """Convert color (1-based) to token."""
        color = int(color)
        if color < 1 or color > CDCLTokenizer.MAX_COLORS:
            raise ValueError(f"color out of range: {color}")
        return CDCLTokenizer.COLOR_OFFSET + color

    @staticmethod
    def depth_token(depth: int) -> int:
        """Convert stack depth to token."""
        depth = int(depth)
        if depth < 0 or depth >= CDCLTokenizer.MAX_DEPTH:
            raise ValueError(f"depth out of range: {depth}")
        return CDCLTokenizer.DEPTH_OFFSET + depth

    def mask_token(self, domain: set[int]) -> int:
        """Encode domain as 4-bit mask token. domain is set of colors (1-4)."""
        mask = 0
        for color in domain:
            color_id = int(color)
            if color_id < 1 or color_id > self.MAX_COLORS:
                raise ValueError(f"color out of range in domain: {color_id}")
        for color in range(1, self.MAX_COLORS + 1):
            if color in domain:
                mask |= 1 << (self.MAX_COLORS - color)
        return self.MASK_OFFSET + mask

    def decode_mask_token(self, token: int) -> set[int]:
        """Decode mask token to set of available colors."""
        token = int(token)
        if token < self.MASK_OFFSET or token >= self.MASK_OFFSET + 16:
            raise ValueError(f"mask token out of range: {token}")
        mask = token - self.MASK_OFFSET
        domain: set[int] = set()
        for color in range(1, self.MAX_COLORS + 1):
            if mask & (1 << (self.MAX_COLORS - color)):
                domain.add(int(color))
        return domain

    def domain_size_token(self, size: int) -> int:
        """Get domain size token for a given size (0-4)."""
        size = int(size)
        assert 0 <= size <= 4
        return int(self.DS0) + size

    def assign_token(self, node: int, color: int) -> int:
        """Get combined ASSIGN token for (node, color) pair."""
        assert 0 <= int(node) < self.MAX_NODES
        assert 1 <= int(color) <= self.MAX_COLORS
        return self.ASSIGN_OFFSET + int(node) * self.MAX_COLORS + (int(color) - 1)

    def decode_assign_token(self, token: int) -> Tuple[int, int]:
        """Decode combined ASSIGN token to (node, color)."""
        idx = int(token) - self.ASSIGN_OFFSET
        node = int(idx // self.MAX_COLORS)
        color = int(idx % self.MAX_COLORS + 1)
        return node, color

    def is_assign_token(self, token: int) -> bool:
        return (
            self.ASSIGN_OFFSET
            <= int(token)
            < self.ASSIGN_OFFSET + self.MAX_NODES * self.MAX_COLORS
        )

    def decode_token(self, token_id: int) -> str:
        """Convert token ID to human-readable string."""
        token_id = int(token_id)
        if token_id in self._SPECIAL_TOKEN_STRINGS:
            return self._SPECIAL_TOKEN_STRINGS[token_id]
        if self.NODE_OFFSET <= token_id < self.COLOR_OFFSET:
            return f"N{token_id - self.NODE_OFFSET}"
        if self.COLOR_OFFSET + 1 <= token_id <= self.COLOR_OFFSET + self.MAX_COLORS:
            return f"C{token_id - self.COLOR_OFFSET}"
        if self.DEPTH_OFFSET <= token_id < self.DEPTH_OFFSET + self.MAX_DEPTH:
            return f"D{token_id - self.DEPTH_OFFSET}"
        if self.is_assign_token(token_id):
            node, color = self.decode_assign_token(token_id)
            return f"A_N{node}_C{color}"
        if self.MASK_OFFSET <= token_id < self.MASK_OFFSET + 16:
            mask = int(token_id) - int(self.MASK_OFFSET)
            return f"M{mask:04b}"
        raise ValueError(f"Unknown token id: {token_id}")

    @staticmethod
    def is_search_token(token_id: int) -> bool:
        token_id = int(token_id)
        return token_id in (
            CDCLTokenizer.ASSIGN,
            CDCLTokenizer.BACKJUMP,
            CDCLTokenizer.SOLVED,
            CDCLTokenizer.FAILED,
            CDCLTokenizer.SEARCH_START,
            CDCLTokenizer.CHECK,
        )

    def _validate_inputs(
        self,
        adjacency,
        stack: List[Tuple[int, int]],
        contradiction_node: int,
        num_nodes: int,
        num_colors: int,
    ) -> np.ndarray:
        num_nodes = int(num_nodes)
        num_colors = int(num_colors)
        contradiction_node = int(contradiction_node)

        if num_nodes <= 0 or num_nodes > self.MAX_NODES:
            raise ValueError(f"num_nodes out of range: {num_nodes}")
        if num_colors <= 0 or num_colors > self.MAX_COLORS:
            raise ValueError(f"num_colors out of range: {num_colors}")
        if contradiction_node < 0 or contradiction_node >= num_nodes:
            raise ValueError(f"contradiction_node out of range: {contradiction_node}")
        if len(stack) > self.MAX_DEPTH:
            raise ValueError(f"stack depth out of range: {len(stack)}")

        adjacency_arr = np.asarray(adjacency)
        if adjacency_arr.ndim != 2:
            raise ValueError("adjacency must be a 2D matrix")
        if adjacency_arr.shape[0] != num_nodes or adjacency_arr.shape[1] != num_nodes:
            raise ValueError(
                f"adjacency shape mismatch: expected ({num_nodes}, {num_nodes}), "
                f"got {adjacency_arr.shape}"
            )

        for depth, (node, color) in enumerate(stack):
            node_id = int(node)
            color_id = int(color)
            if node_id < 0 or node_id >= num_nodes:
                raise ValueError(f"stack node out of range: {node_id}")
            if color_id < 1 or color_id > num_colors:
                raise ValueError(f"stack color out of range: {color_id}")
            if depth >= self.MAX_DEPTH:
                raise ValueError(f"depth out of range: {depth}")

        return adjacency_arr

    @staticmethod
    def _neighbors(adjacency: np.ndarray, node_id: int) -> List[int]:
        row = adjacency[int(node_id)]
        neighbors = np.nonzero(row)[0].astype(int).tolist()
        neighbors = [n for n in neighbors if n != int(node_id)]
        neighbors.sort()
        return neighbors

    def _build_graph_section(self, adjacency: np.ndarray, num_nodes: int) -> List[int]:
        num_nodes = int(num_nodes)
        if num_nodes <= 0 or num_nodes > self.MAX_NODES:
            raise ValueError(f"num_nodes out of range: {num_nodes}")
        adj = np.asarray(adjacency)
        if adj.ndim != 2 or adj.shape[0] != num_nodes or adj.shape[1] != num_nodes:
            raise ValueError(
                f"adjacency shape mismatch: expected ({num_nodes}, {num_nodes}), got {adj.shape}"
            )
        tokens: List[int] = [self.BOS, self.GRAPH_START]
        for i in range(num_nodes):
            neighbors = sorted(np.where(adj[i])[0])
            tokens.append(self.node_token(int(i)))
            tokens.append(self.COLON)
            for nb in neighbors:
                tokens.append(self.node_token(int(nb)))
            tokens.append(self.SEP)
        return tokens

    def build_graph_prefix(self, adjacency: np.ndarray, num_nodes: int) -> List[int]:
        tokens = self._build_graph_section(adjacency, num_nodes)
        tokens.append(self.SEARCH_START)
        return tokens

    def serialize_state(
        self,
        adjacency,
        stack: List[Tuple[int, int]],
        contradiction_node: int,
        num_nodes: int,
        num_colors: int,
        mode: str = "flat",
    ) -> List[int]:
        """Serialize a conflict state into a token sequence.

        Args:
            adjacency: [N, N] adjacency matrix (numpy array or list of lists)
            stack: list of (node, color) tuples, ordered by assignment depth
            contradiction_node: int, the node with empty domain
            num_nodes: int
            num_colors: int
            mode: "flat" for just state+target marker, "scratchpad" for full CoT

        Returns:
            List of token IDs (without target — target appended during training)
        """
        mode = str(mode)
        if mode not in {"flat", "scratchpad"}:
            raise ValueError(f"Unknown mode: {mode}")

        adjacency_arr = self._validate_inputs(
            adjacency, stack, contradiction_node, num_nodes, num_colors
        )

        tokens: List[int] = [self.GRAPH_START]

        # Graph section: adjacency lists per node.
        for node_id in range(int(num_nodes)):
            neighbors = self._neighbors(adjacency_arr, int(node_id))
            tokens.append(self.node_token(node_id))
            tokens.append(self.COLON)
            for nbr in neighbors:
                tokens.append(self.node_token(nbr))
            tokens.append(self.SEP)

        # Stack section: depth, node, color.
        tokens.append(self.STACK_START)
        for depth, (node, color) in enumerate(stack):
            tokens.append(self.depth_token(depth))
            tokens.append(self.node_token(node))
            tokens.append(self.color_token(color))
            tokens.append(self.SEP)

        # Conflict node section.
        tokens.append(self.CONFLICT_START)
        tokens.append(self.node_token(contradiction_node))

        if mode == "flat":
            tokens.append(self.TARGET_START)
        else:
            tokens.append(self.THINK_START)

        return tokens

    def generate_scratchpad(
        self,
        adjacency,
        stack: List[Tuple[int, int]],
        contradiction_node: int,
        num_nodes: int,
        num_colors: int,
    ) -> Tuple[List[int], int]:
        """Generate the chain-of-thought scratchpad tokens AND the target.

        The scratchpad implements the greedy witness algorithm:
        1. List neighbors of contradiction node
        2. For each color, find shallowest witness among neighbors
        3. Collect witness depths
        4. Compute 2nd-deepest

        Returns:
            (scratchpad_tokens, target_depth)
        """
        adjacency_arr = self._validate_inputs(
            adjacency, stack, contradiction_node, num_nodes, num_colors
        )

        # Build stack index for quick lookup of (depth, color) per node.
        stack_index: Dict[int, int] = {}
        stack_color: Dict[int, int] = {}
        for depth, (node, color) in enumerate(stack):
            node_id = int(node)
            if node_id not in stack_index:
                stack_index[node_id] = int(depth)
                stack_color[node_id] = int(color)

        tokens: List[int] = []

        neighbors = self._neighbors(adjacency_arr, int(contradiction_node))
        tokens.append(self.node_token(contradiction_node))
        tokens.append(self.COLON)
        for nbr in neighbors:
            tokens.append(self.node_token(nbr))
        tokens.append(self.SEP)

        best_depths: List[Optional[int]] = []

        for color in range(1, int(num_colors) + 1):
            witnesses: List[Tuple[int, int]] = []
            for nbr in neighbors:
                if int(nbr) not in stack_index:
                    continue
                if int(stack_color[int(nbr)]) != int(color):
                    continue
                witnesses.append((int(nbr), int(stack_index[int(nbr)])))

            # Sort by depth so the shallowest witness comes first.
            witnesses.sort(key=lambda pair: pair[1])

            tokens.append(self.color_token(color))
            tokens.append(self.COLON)
            if witnesses:
                for node_id, depth in witnesses:
                    tokens.append(self.node_token(node_id))
                    tokens.append(self.depth_token(depth))
            else:
                tokens.append(self.NONE_TOKEN)

            tokens.append(self.ARROW)
            tokens.append(self.BEST)
            if witnesses:
                best_node, best_depth = witnesses[0]
                tokens.append(self.node_token(best_node))
                tokens.append(self.depth_token(best_depth))
                best_depths.append(int(best_depth))
            else:
                tokens.append(self.NONE_TOKEN)
                best_depths.append(None)
            tokens.append(self.SEP)

        tokens.append(self.DEPTHS)
        for depth in best_depths:
            if depth is None:
                tokens.append(self.NONE_TOKEN)
            else:
                tokens.append(self.depth_token(depth))
        tokens.append(self.SEP)

        valid_depths = [d for d in best_depths if d is not None]
        if not valid_depths:
            # Graceful fallback: no witnesses found (should be rare).
            target_depth = 0
        else:
            sorted_depths = sorted(int(d) for d in valid_depths)
            if len(sorted_depths) >= 2:
                target_depth = int(sorted_depths[-2])
            else:
                target_depth = int(sorted_depths[-1])

        tokens.append(self.SECOND)
        tokens.append(self.depth_token(target_depth))
        tokens.append(self.SEP)

        return tokens, int(target_depth)

    def build_training_sequence(
        self,
        adjacency,
        stack: List[Tuple[int, int]],
        contradiction_node: int,
        num_nodes: int,
        num_colors: int,
        mode: str = "scratchpad",
    ) -> Tuple[List[int], int]:
        """Build complete training sequence.

        For mode="flat": [BOS] [GRAPH] ... [STACK] ... [CONFLICT] Nv [TARGET] Dt [EOS]
        For mode="scratchpad": [BOS] [GRAPH] ... [STACK] ... [CONFLICT] Nv [THINK] ... [TARGET] Dt [EOS]

        Returns:
            (token_ids, target_depth)
        """
        mode = str(mode)
        if mode not in {"flat", "scratchpad"}:
            raise ValueError(f"Unknown mode: {mode}")

        state_tokens = self.serialize_state(
            adjacency,
            stack,
            contradiction_node,
            num_nodes,
            num_colors,
            mode=mode,
        )

        if mode == "flat":
            _, target_depth = self.generate_scratchpad(
                adjacency, stack, contradiction_node, num_nodes, num_colors
            )
            sequence = [
                self.BOS,
                *state_tokens,
                self.depth_token(target_depth),
                self.EOS,
            ]
            return sequence, int(target_depth)

        scratchpad_tokens, target_depth = self.generate_scratchpad(
            adjacency, stack, contradiction_node, num_nodes, num_colors
        )

        sequence = [
            self.BOS,
            *state_tokens,
            *scratchpad_tokens,
            self.TARGET_START,
            self.depth_token(target_depth),
            self.EOS,
        ]
        return sequence, int(target_depth)

    def serialize_state_augmented(
        self,
        adjacency,
        stack: List[Tuple[int, int]],
        contradiction_node: int,
        num_nodes: int,
        num_colors: int,
        mode: str = "scratchpad",
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[List[int], int]:
        """Same as build_training_sequence but with random node ID permutation.

        Randomly permutes all node IDs to teach the model that node identity
        doesn't matter — only structure does.
        """
        if rng is None:
            rng = np.random.default_rng()

        adjacency_arr = self._validate_inputs(
            adjacency, stack, contradiction_node, num_nodes, num_colors
        )

        perm = np.array(rng.permutation(int(num_nodes)), dtype=np.int64)

        # Reindex adjacency to the new node order.
        adjacency_perm = adjacency_arr[np.ix_(perm, perm)]

        # Map old node IDs to new node IDs.
        old_to_new = np.empty(int(num_nodes), dtype=np.int64)
        for new_id, old_id in enumerate(perm):
            old_to_new[int(old_id)] = int(new_id)

        permuted_stack = [
            (int(old_to_new[int(node)]), int(color)) for node, color in stack
        ]
        permuted_contradiction = int(old_to_new[int(contradiction_node)])

        return self.build_training_sequence(
            adjacency_perm,
            permuted_stack,
            permuted_contradiction,
            num_nodes,
            num_colors,
            mode=mode,
        )

    def build_search_trace(
        self,
        adjacency,
        search_events,
        num_nodes,
        num_colors,
        include_cot: bool = False,
    ) -> List[int]:
        """Build a complete autoregressive search trace.

        Args:
            adjacency: [N, N] adjacency matrix
            search_events: list of event dicts
            num_nodes: int
            num_colors: int
            include_cot: if True, add scratchpad reasoning at conflict points

        Returns:
            List of token IDs: [BOS] [GRAPH] ... [SEARCH] ASSIGN N C SEP ...
        """
        num_nodes = int(num_nodes)
        num_colors = int(num_colors)
        if num_nodes <= 0 or num_nodes > self.MAX_NODES:
            raise ValueError(f"num_nodes out of range: {num_nodes}")
        if num_colors <= 0 or num_colors > self.MAX_COLORS:
            raise ValueError(f"num_colors out of range: {num_colors}")

        tokens = self.build_graph_prefix(adjacency, num_nodes)

        for event in search_events:
            event_type = str(event.get("type"))
            if event_type == "assign":
                tokens.append(self.ASSIGN)
                tokens.append(self.node_token(event["node"]))
                tokens.append(self.color_token(event["color"]))
                tokens.append(self.SEP)
                continue

            if event_type == "conflict":
                tokens.append(self.CONFLICT_START)
                tokens.append(self.node_token(event["node"]))

                if include_cot and event.get("witnesses"):
                    tokens.append(self.THINK_START)
                    witness_depths = event.get("witness_depths")
                    if witness_depths:
                        tokens.append(self.DEPTHS)
                        for depth in witness_depths:
                            tokens.append(self.depth_token(depth))
                        tokens.append(self.SEP)
                        sorted_depths = sorted(witness_depths, reverse=True)
                        if len(sorted_depths) >= 2:
                            tokens.append(self.SECOND)
                            tokens.append(self.depth_token(sorted_depths[1]))
                            tokens.append(self.SEP)

                tokens.append(self.BACKJUMP)
                tokens.append(self.depth_token(event["backjump_target"]))
                tokens.append(self.SEP)
                continue

            if event_type == "solved":
                tokens.append(self.SOLVED)
                continue

            if event_type == "failed":
                tokens.append(self.FAILED)
                continue

            raise ValueError(f"Unknown search event type: {event_type}")

        tokens.append(self.EOS)
        return tokens

    def build_search_trace_with_checks(
        self,
        adjacency,
        search_events,
        num_nodes,
        num_colors,
    ) -> List[int]:
        """Build a search trace where each assignment is followed by a domain check.

        Format:
            [BOS] [GRAPH] ... [SEARCH]
            ASSIGN N3 C1 SEP
            [CHECK] N5 C2 C3 C4 SEP N7 C1 C2 C3 SEP N22 C2 C3 C4 SEP SEP
            ASSIGN N7 C2 SEP
            [CHECK] N5 C3 C4 SEP N22 C3 C4 SEP SEP
            ...
            ASSIGN N35 C4 SEP
            [CHECK] N22 EMPTY SEP SEP
            CONFLICT N22 BACKJUMP D14 SEP
            ...
            SOLVED [EOS]

        Each CHECK section lists affected neighbors (those whose domain changed)
        with their remaining domain colors. Double SEP ends the CHECK section.
        EMPTY_DOMAIN token (231) means the domain is empty → signals conflict.
        """
        num_nodes = int(num_nodes)
        num_colors = int(num_colors)
        if num_nodes <= 0 or num_nodes > self.MAX_NODES:
            raise ValueError(f"num_nodes out of range: {num_nodes}")
        if num_colors <= 0 or num_colors > self.MAX_COLORS:
            raise ValueError(f"num_colors out of range: {num_colors}")

        tokens = self.build_graph_prefix(adjacency, num_nodes)

        for event in search_events:
            event_type = str(event.get("type"))
            if event_type == "assign":
                tokens.append(self.ASSIGN)
                tokens.append(self.node_token(event["node"]))
                tokens.append(self.color_token(event["color"]))
                tokens.append(self.SEP)

                if "affected_domains" in event:
                    tokens.append(self.CHECK)
                    affected = event.get("affected_domains") or {}
                    for nb_node, remaining_colors in sorted(affected.items()):
                        tokens.append(self.node_token(nb_node))
                        if len(remaining_colors) == 0:
                            tokens.append(self.EMPTY_DOMAIN)
                        else:
                            for c in sorted(remaining_colors):
                                tokens.append(self.color_token(c))
                        tokens.append(self.SEP)
                    tokens.append(self.SEP)
                continue

            if event_type == "conflict":
                tokens.append(self.CONFLICT_START)
                tokens.append(self.node_token(event["node"]))
                tokens.append(self.BACKJUMP)
                tokens.append(self.depth_token(event["backjump_target"]))
                tokens.append(self.SEP)
                continue

            if event_type == "solved":
                tokens.append(self.SOLVED)
                continue

            if event_type == "failed":
                tokens.append(self.FAILED)
                continue

            raise ValueError(f"Unknown search event type: {event_type}")

        tokens.append(self.EOS)
        return tokens

    def build_interleaved_trace(
        self,
        adjacency,
        search_events,
        num_nodes,
        num_colors,
    ) -> List[int]:
        """Build an interleaved model/environment search trace.

        Format:
            [BOS] [GRAPH] ... [SEARCH]
            CANDIDATES N0 N1 ... SEP
            DECIDE N0 SEP
            DOMAIN C1 C2 ... SEP
            PICK C1 SEP
            RESULT OK SEP
            ...
            SOLVED [EOS]
        """
        num_nodes = int(num_nodes)
        num_colors = int(num_colors)
        if num_nodes <= 0 or num_nodes > self.MAX_NODES:
            raise ValueError(f"num_nodes out of range: {num_nodes}")
        if num_colors <= 0 or num_colors > self.MAX_COLORS:
            raise ValueError(f"num_colors out of range: {num_colors}")

        adj = np.array(adjacency)
        if adj.ndim != 2 or adj.shape[0] != num_nodes or adj.shape[1] != num_nodes:
            raise ValueError(
                f"adjacency shape mismatch: expected ({num_nodes}, {num_nodes}), got {adj.shape}"
            )

        tokens: List[int] = self.build_graph_prefix(adjacency, num_nodes)

        for event in search_events:
            event_type = str(event.get("type"))
            if event_type == "assign":
                unassigned = event.get("unassigned_before")
                if unassigned is None:
                    raise ValueError("assign event missing unassigned_before")

                tokens.append(self.CANDIDATES)
                for node in sorted(int(n) for n in unassigned):
                    tokens.append(self.node_token(int(node)))
                tokens.append(self.SEP)

                tokens.append(self.DECIDE)
                tokens.append(self.node_token(event["node"]))
                tokens.append(self.SEP)

                domain = event.get("domain")
                if domain is None:
                    raise ValueError("assign event missing domain")
                domain_list = sorted(int(c) for c in domain)

                tokens.append(self.DOMAIN)
                if len(domain_list) == 0:
                    tokens.append(self.EMPTY_DOMAIN)
                else:
                    for c in domain_list:
                        tokens.append(self.color_token(int(c)))
                tokens.append(self.SEP)

                tokens.append(self.PICK)
                tokens.append(self.color_token(event["color"]))
                tokens.append(self.SEP)

                tokens.append(self.RESULT)
                tokens.append(self.OK)
                tokens.append(self.SEP)
                continue

            if event_type == "conflict":
                unassigned = event.get("unassigned_before")
                if unassigned is None:
                    raise ValueError("conflict event missing unassigned_before")

                tokens.append(self.CANDIDATES)
                for node in sorted(int(n) for n in unassigned):
                    tokens.append(self.node_token(int(node)))
                tokens.append(self.SEP)

                tokens.append(self.DECIDE)
                tokens.append(self.node_token(event["node"]))
                tokens.append(self.SEP)

                tokens.append(self.DOMAIN)
                tokens.append(self.EMPTY_DOMAIN)
                tokens.append(self.SEP)

                tokens.append(self.BACKJUMP)
                tokens.append(self.depth_token(event["backjump_target"]))
                tokens.append(self.SEP)
                continue

            if event_type == "solved":
                tokens.append(self.SOLVED)
                continue

            if event_type == "failed":
                tokens.append(self.FAILED)
                continue

            raise ValueError(f"Unknown search event type: {event_type}")

        tokens.append(self.EOS)
        return tokens

    def build_compressed_trace(
        self,
        adjacency: np.ndarray,
        events: List[dict],
        num_nodes: int,
        num_colors: int,
    ) -> List[int]:
        """Build compressed trace: graph prefix + assign/conflict tokens."""
        num_nodes = int(num_nodes)
        num_colors = int(num_colors)
        if num_nodes <= 0 or num_nodes > self.MAX_NODES:
            raise ValueError(f"num_nodes out of range: {num_nodes}")
        if num_colors <= 0 or num_colors > self.MAX_COLORS:
            raise ValueError(f"num_colors out of range: {num_colors}")

        tokens = self._build_graph_section(adjacency, num_nodes)
        tokens.append(self.SEARCH_START)

        for event in events:
            event_type = str(event.get("type"))
            if event_type == "assign":
                tokens.append(self.assign_token(event["node"], event["color"]))
                tokens.append(self.OK)
                continue

            if event_type == "conflict":
                if "last_assign_node" in event and "last_assign_color" in event:
                    tokens.append(
                        self.assign_token(
                            event["last_assign_node"], event["last_assign_color"]
                        )
                    )
                tokens.append(self.CONFLICT_START)
                tokens.append(self.node_token(event["node"]))
                tokens.append(self.depth_token(event["backjump_target"]))
                continue

            if event_type == "solved":
                tokens.append(self.SOLVED)
                continue

            if event_type == "failed":
                tokens.append(self.FAILED)
                continue

            raise ValueError(f"Unknown search event type: {event_type}")

        tokens.append(self.EOS)
        return tokens

    def build_slim_trace(
        self,
        adjacency: np.ndarray,
        events: List[dict],
        num_nodes: int,
        num_colors: int,
    ) -> List[int]:
        """Build slim interleaved trace with domain mask tokens."""
        num_nodes = int(num_nodes)
        num_colors = int(num_colors)
        if num_nodes <= 0 or num_nodes > self.MAX_NODES:
            raise ValueError(f"num_nodes out of range: {num_nodes}")
        if num_colors <= 0 or num_colors > self.MAX_COLORS:
            raise ValueError(f"num_colors out of range: {num_colors}")

        tokens = self._build_graph_section(adjacency, num_nodes)
        tokens.append(self.SEARCH_START)

        for event in events:
            event_type = str(event.get("type"))
            if event_type == "assign":
                domain = event.get("domain")
                if domain is None:
                    raise ValueError("assign event missing domain")
                tokens.append(self.node_token(event["node"]))
                tokens.append(self.mask_token(set(domain)))
                tokens.append(self.color_token(event["color"]))
                tokens.append(self.OK)
                continue

            if event_type == "conflict":
                tokens.append(self.node_token(event["node"]))
                tokens.append(self.mask_token(set()))
                tokens.append(self.depth_token(event["backjump_target"]))
                continue

            if event_type == "solved":
                tokens.append(self.SOLVED)
                continue

            if event_type == "failed":
                tokens.append(self.FAILED)
                continue

            raise ValueError(f"Unknown search event type: {event_type}")

        tokens.append(self.EOS)
        return tokens

    def build_sorted_slim_trace(
        self,
        adjacency: np.ndarray,
        events: List[dict],
        num_nodes: int,
        num_colors: int,
    ) -> List[int]:
        """Build slim trace with domain-sized STATE sections before each step."""
        num_nodes = int(num_nodes)
        num_colors = int(num_colors)
        if num_nodes <= 0 or num_nodes > self.MAX_NODES:
            raise ValueError(f"num_nodes out of range: {num_nodes}")
        if num_colors <= 0 or num_colors > self.MAX_COLORS:
            raise ValueError(f"num_colors out of range: {num_colors}")

        tokens = self._build_graph_section(adjacency, num_nodes)
        tokens.append(self.SEARCH_START)

        def _append_state(
            sorted_candidates: List[int],
            domain_sizes: Dict[int, int],
        ) -> None:
            tokens.append(self.STATE)
            for node_id in sorted_candidates:
                ds = domain_sizes.get(int(node_id))
                if ds is None:
                    raise ValueError(f"domain_sizes missing node {node_id}")
                tokens.append(self.domain_size_token(min(int(ds), self.MAX_COLORS)))
                tokens.append(self.node_token(node_id))
            tokens.append(self.SEP)

        for event in events:
            event_type = str(event.get("type"))
            if event_type == "assign":
                sorted_candidates = event.get("sorted_candidates")
                if sorted_candidates is None:
                    raise ValueError("assign event missing sorted_candidates")
                domain_sizes = event.get("domain_sizes")
                if domain_sizes is None:
                    raise ValueError("assign event missing domain_sizes")
                _append_state(sorted_candidates, domain_sizes)

                domain = event.get("domain")
                if domain is None:
                    raise ValueError("assign event missing domain")
                tokens.append(self.node_token(event["node"]))
                tokens.append(self.mask_token(set(domain)))
                tokens.append(self.color_token(event["color"]))
                tokens.append(self.OK)
                continue

            if event_type == "conflict":
                sorted_candidates = event.get("sorted_candidates")
                if sorted_candidates is None:
                    raise ValueError("conflict event missing sorted_candidates")
                domain_sizes = event.get("domain_sizes")
                if domain_sizes is None:
                    raise ValueError("conflict event missing domain_sizes")
                _append_state(sorted_candidates, domain_sizes)

                tokens.append(self.node_token(event["node"]))
                tokens.append(self.mask_token(set()))
                tokens.append(self.depth_token(event["backjump_target"]))
                continue

            if event_type == "solved":
                tokens.append(self.SOLVED)
                continue

            if event_type == "failed":
                tokens.append(self.FAILED)
                continue

            raise ValueError(f"Unknown search event type: {event_type}")

        tokens.append(self.EOS)
        return tokens

    def build_rle_bucket_trace(
        self,
        adjacency: np.ndarray,
        events: List[dict],
        num_nodes: int,
        num_colors: int,
    ) -> List[int]:
        """Build slim trace with RLE bucketed STATE sections before each step."""
        num_nodes = int(num_nodes)
        num_colors = int(num_colors)
        if num_nodes <= 0 or num_nodes > self.MAX_NODES:
            raise ValueError(f"num_nodes out of range: {num_nodes}")
        if num_colors <= 0 or num_colors > self.MAX_COLORS:
            raise ValueError(f"num_colors out of range: {num_colors}")

        tokens = self._build_graph_section(adjacency, num_nodes)
        tokens.append(self.SEARCH_START)

        def _append_state(
            sorted_candidates: List[int],
            domain_sizes: Dict[int, int],
        ) -> None:
            tokens.append(self.STATE)
            buckets: Dict[int, List[int]] = {
                size: [] for size in range(int(self.MAX_COLORS) + 1)
            }
            for node_id in sorted_candidates:
                ds = min(
                    int(domain_sizes.get(int(node_id), int(self.MAX_COLORS))),
                    int(self.MAX_COLORS),
                )
                buckets[int(ds)].append(int(node_id))
            for size in range(int(self.MAX_COLORS) + 1):
                tokens.append(self.domain_size_token(int(size)))
                for node_id in buckets[int(size)]:
                    tokens.append(self.node_token(node_id))
            tokens.append(self.SEP)

        for event in events:
            event_type = str(event.get("type"))
            if event_type == "assign":
                sorted_candidates = event.get("sorted_candidates")
                if sorted_candidates is None:
                    raise ValueError("assign event missing sorted_candidates")
                domain_sizes = event.get("domain_sizes")
                if domain_sizes is None:
                    raise ValueError("assign event missing domain_sizes")
                _append_state(sorted_candidates, domain_sizes)

                domain = event.get("domain")
                if domain is None:
                    raise ValueError("assign event missing domain")
                tokens.append(self.node_token(event["node"]))
                tokens.append(self.mask_token(set(domain)))
                tokens.append(self.color_token(event["color"]))
                tokens.append(self.OK)
                continue

            if event_type == "conflict":
                sorted_candidates = event.get("sorted_candidates")
                if sorted_candidates is None:
                    raise ValueError("conflict event missing sorted_candidates")
                domain_sizes = event.get("domain_sizes")
                if domain_sizes is None:
                    raise ValueError("conflict event missing domain_sizes")
                _append_state(sorted_candidates, domain_sizes)

                tokens.append(self.node_token(event["node"]))
                tokens.append(self.mask_token(set()))
                tokens.append(self.depth_token(event["backjump_target"]))
                continue

            if event_type == "solved":
                tokens.append(self.SOLVED)
                continue

            if event_type == "failed":
                tokens.append(self.FAILED)
                continue

            raise ValueError(f"Unknown search event type: {event_type}")

        tokens.append(self.EOS)
        return tokens

    def build_prop_evidence(
        self,
        checked_node: int,
        adjacency: np.ndarray,
        assignment: np.ndarray,
        num_colors: int,
    ) -> Tuple[List[int], bool]:
        """Build propagation evidence with neighbor copy + color copy.

        Format: node : neighbors SEP neighbor_colors SEP per_color_evidence verdict

        Phase 1: N9 : N3 N7 N12 N20 SEP
        Phase 2: N3 C2 N7 C1 N12 C3 N20 FREE SEP
        Phase 3: C1 N7 C2 N3 C3 N12 C4 FREE verdict
        """
        checked_node = int(checked_node)
        num_colors = int(num_colors)
        adj = np.asarray(adjacency)
        asgn = np.asarray(assignment)
        neighbors = sorted(
            int(n) for n in np.nonzero(adj[checked_node])[0] if int(n) != checked_node
        )

        tokens: List[int] = []

        # Phase 1: Node + neighbor copy
        tokens.append(self.node_token(checked_node))
        tokens.append(int(self.COLON))
        for nb in neighbors:
            tokens.append(self.node_token(nb))
        tokens.append(int(self.SEP))

        # Phase 2: Neighbor-color pairs
        neighbor_color_assigned = 0
        neighbor_color_free = 0
        for nb in neighbors:
            tokens.append(self.node_token(nb))
            color = int(asgn[int(nb)])
            if color >= 1 and color <= num_colors:
                tokens.append(self.color_token(color))
                neighbor_color_assigned += 1
            else:
                tokens.append(int(self.FREE))
                neighbor_color_free += 1
        tokens.append(int(self.SEP))

        # Phase 3: Per-color evidence (now derivable from Phase 2 alone)
        all_blocked = True
        free_count = 0

        for color in range(1, num_colors + 1):
            tokens.append(self.color_token(color))
            blocker: Optional[int] = None
            for nb in neighbors:
                if int(asgn[int(nb)]) == color:
                    blocker = int(nb)
                    break
            if blocker is not None:
                tokens.append(self.node_token(blocker))
            else:
                tokens.append(int(self.FREE))
                all_blocked = False
                free_count += 1

        if all_blocked:
            tokens.append(int(self.CF))
        else:
            tokens.append(int(self.OK))

        logger.debug(
            "build_prop_evidence: checked_node=%d neighbors=%d num_colors=%d neighbor_color_assigned=%d neighbor_color_free=%d free_colors=%d all_blocked=%s tokens_len=%d",
            checked_node,
            int(len(neighbors)),
            num_colors,
            int(neighbor_color_assigned),
            int(neighbor_color_free),
            int(free_count),
            bool(all_blocked),
            int(len(tokens)),
        )

        return tokens, bool(all_blocked)

    def build_prop_trace(
        self,
        adjacency: np.ndarray,
        events: List[dict],
        num_nodes: int,
        num_colors: int,
    ) -> List[int]:
        """Build trace with propagation scratchpad blocks (View A).

        Format per step:
            STATE N9 N15 ... SEP
            [PROP] N9 : N3 N7 N12 N20 SEP N3 C2 N7 C1 N12 C3 N20 FREE SEP C1 N7 C2 N3 C3 N12 C4 FREE OK [/PROP]
            N9 M0000 D14   (or N5 M0110 C2 OK)
        """
        num_nodes = int(num_nodes)
        num_colors = int(num_colors)
        adj = np.asarray(adjacency)
        tokens = self._build_graph_section(adj, num_nodes)
        tokens.append(self.SEARCH_START)

        assign_steps = 0
        conflict_steps = 0
        logger.debug(
            "build_prop_trace: num_nodes=%d num_colors=%d events=%d",
            num_nodes,
            num_colors,
            int(len(events)),
        )

        for event in events:
            event_type = str(event.get("type"))

            if event_type in ("assign", "conflict"):
                sorted_candidates = event["sorted_candidates"]
                # STATE section (plain sorted nodes, no DS tokens)
                tokens.append(self.STATE)
                for nd in sorted_candidates:
                    tokens.append(self.node_token(int(nd)))
                tokens.append(self.SEP)

                # PROP block
                asgn = np.asarray(event["current_assignment"])
                checked = int(sorted_candidates[0])
                prop_inner, _is_cf = self.build_prop_evidence(
                    checked,
                    adj,
                    asgn,
                    num_colors,
                )
                tokens.append(int(self.PROP))
                tokens.extend(prop_inner)
                tokens.append(int(self.ENDPROP))

                if event_type == "assign":
                    assign_steps += 1
                    tokens.append(self.node_token(event["node"]))
                    tokens.append(self.mask_token(set(event["domain"])))
                    tokens.append(self.color_token(event["color"]))
                    tokens.append(int(self.OK))
                    # Shadow assignment anchor for easy lookup
                    tokens.append(self.node_token(event["node"]))
                    tokens.append(self.color_token(event["color"]))
                else:
                    conflict_steps += 1
                    tokens.append(self.node_token(event["node"]))
                    tokens.append(self.mask_token(set()))
                    tokens.append(self.depth_token(event["backjump_target"]))
                continue

            if event_type == "solved":
                tokens.append(int(self.SOLVED))
                continue
            if event_type == "failed":
                tokens.append(int(self.FAILED))
                continue
            raise ValueError(f"Unknown event type: {event_type}")

        tokens.append(int(self.EOS))

        logger.debug(
            "build_prop_trace done: tokens_len=%d assign_steps=%d conflict_steps=%d",
            int(len(tokens)),
            int(assign_steps),
            int(conflict_steps),
        )
        return tokens

    def build_prop_verdict_only_trace(
        self,
        adjacency: np.ndarray,
        events: List[dict],
        num_nodes: int,
        num_colors: int,
    ) -> List[int]:
        """Build trace with verdict only, no PROP reasoning (View B)."""
        num_nodes = int(num_nodes)
        num_colors = int(num_colors)
        adj = np.asarray(adjacency)
        tokens = self._build_graph_section(adj, num_nodes)
        tokens.append(self.SEARCH_START)

        assign_steps = 0
        conflict_steps = 0
        logger.debug(
            "build_prop_verdict_only_trace: num_nodes=%d num_colors=%d events=%d",
            num_nodes,
            num_colors,
            int(len(events)),
        )

        for event in events:
            event_type = str(event.get("type"))

            if event_type in ("assign", "conflict"):
                sorted_candidates = event["sorted_candidates"]
                tokens.append(self.STATE)
                for nd in sorted_candidates:
                    tokens.append(self.node_token(int(nd)))
                tokens.append(self.SEP)

                # Verdict only
                asgn = np.asarray(event["current_assignment"])
                checked = int(sorted_candidates[0])
                _prop_inner, is_cf = self.build_prop_evidence(
                    checked,
                    adj,
                    asgn,
                    num_colors,
                )
                tokens.append(int(self.CF) if is_cf else int(self.OK))

                if event_type == "assign":
                    assign_steps += 1
                    tokens.append(self.node_token(event["node"]))
                    tokens.append(self.mask_token(set(event["domain"])))
                    tokens.append(self.color_token(event["color"]))
                    tokens.append(int(self.OK))
                    # Shadow assignment anchor for easy lookup
                    tokens.append(self.node_token(event["node"]))
                    tokens.append(self.color_token(event["color"]))
                else:
                    conflict_steps += 1
                    tokens.append(self.node_token(event["node"]))
                    tokens.append(self.mask_token(set()))
                    tokens.append(self.depth_token(event["backjump_target"]))
                continue

            if event_type == "solved":
                tokens.append(int(self.SOLVED))
                continue
            if event_type == "failed":
                tokens.append(int(self.FAILED))
                continue
            raise ValueError(f"Unknown event type: {event_type}")

        tokens.append(int(self.EOS))

        logger.debug(
            "build_prop_verdict_only_trace done: tokens_len=%d assign_steps=%d conflict_steps=%d",
            int(len(tokens)),
            int(assign_steps),
            int(conflict_steps),
        )
        return tokens

    def build_witness_trace(
        self,
        adjacency: np.ndarray,
        events: List[dict],
        num_nodes: int,
        num_colors: int,
    ) -> List[int]:
        """Build sorted slim trace with witness certificate tokens."""
        num_nodes = int(num_nodes)
        num_colors = int(num_colors)
        if num_nodes <= 0 or num_nodes > self.MAX_NODES:
            raise ValueError(f"num_nodes out of range: {num_nodes}")
        if num_colors <= 0 or num_colors > self.MAX_COLORS:
            raise ValueError(f"num_colors out of range: {num_colors}")

        tokens = self.build_graph_prefix(adjacency, num_nodes)

        for event in events:
            event_type = str(event.get("type"))
            if event_type == "assign":
                sorted_candidates = event.get("sorted_candidates")
                if sorted_candidates is None:
                    raise ValueError("assign event missing sorted_candidates")
                tokens.append(self.STATE)
                for node_id in sorted_candidates:
                    tokens.append(self.node_token(node_id))
                tokens.append(self.SEP)

                domain = event.get("domain")
                if domain is None:
                    raise ValueError("assign event missing domain")
                tokens.append(self.node_token(event["node"]))
                tokens.append(self.mask_token(set(domain)))
                tokens.append(self.color_token(event["color"]))

                if "has_conflict" not in event:
                    raise ValueError("assign event missing has_conflict")
                if "witness_node" not in event:
                    raise ValueError("assign event missing witness_node")

                witness_node = event.get("witness_node")
                has_conflict = bool(event.get("has_conflict"))
                if has_conflict:
                    if witness_node is None:
                        raise ValueError("conflict assign event missing witness_node")
                    tokens.append(self.CF)
                    tokens.append(self.WITNESS)
                    tokens.append(self.node_token(witness_node))
                    backjump_target = event.get("backjump_target")
                    if backjump_target is None:
                        raise ValueError(
                            "conflict assign event missing backjump_target"
                        )
                    tokens.append(self.depth_token(backjump_target))
                else:
                    if witness_node is not None:
                        raise ValueError(
                            "non-conflict assign event should have witness_node=None"
                        )
                    tokens.append(self.OK)
                continue

            if event_type == "conflict":
                raise ValueError("conflict events are not used in witness traces")

            if event_type == "solved":
                tokens.append(self.SOLVED)
                continue

            if event_type == "failed":
                tokens.append(self.FAILED)
                continue

            raise ValueError(f"Unknown search event type: {event_type}")

        tokens.append(self.EOS)
        return tokens

    def build_pdl_trace(
        self,
        adjacency: np.ndarray,
        events: List[dict],
        num_nodes: int,
        num_colors: int,
    ) -> List[int]:
        """Build PDL trace: graph prefix + [D] ledger + [ACT] actions."""
        num_nodes = int(num_nodes)
        num_colors = int(num_colors)
        if num_nodes <= 0 or num_nodes > self.MAX_NODES:
            raise ValueError(f"num_nodes out of range: {num_nodes}")
        if num_colors <= 0 or num_colors > self.MAX_COLORS:
            raise ValueError(f"num_colors out of range: {num_colors}")

        tokens = self.build_graph_prefix(adjacency, num_nodes)

        for event in events:
            event_type = str(event.get("type"))
            if event_type in {"assign", "backjump"}:
                domains = event.get("domains")
                if domains is None:
                    raise ValueError("PDL event missing domains")
                domain_map = {
                    int(node): {int(c) for c in colors}
                    for node, colors in domains.items()
                }

                tokens.append(self.LEDGER)
                for node_id in range(num_nodes):
                    if int(node_id) not in domain_map:
                        raise ValueError(f"domains missing node {node_id}")
                    tokens.append(self.mask_token(set(domain_map[int(node_id)])))

                tokens.append(self.ACT_TOKEN)
                if event_type == "assign":
                    tokens.append(self.node_token(event["node"]))
                    tokens.append(self.color_token(event["color"]))
                else:
                    tokens.append(self.BACKJUMP)
                    tokens.append(self.depth_token(event["backjump_target"]))
                continue

            if event_type == "solved":
                tokens.append(self.SOLVED)
                continue

            if event_type == "failed":
                tokens.append(self.FAILED)
                continue

            raise ValueError(f"Unknown search event type: {event_type}")

        tokens.append(self.EOS)
        return tokens

    def build_loss_mask_pdl(self, token_ids: List[int]) -> List[bool]:
        """Return True for model-generated tokens in PDL traces."""
        mask = [False] * int(len(token_ids))
        if not token_ids:
            return mask

        search_idx = None
        for idx, token_id in enumerate(token_ids):
            if int(token_id) == int(self.SEARCH_START):
                search_idx = int(idx)
                break
        if search_idx is None:
            raise ValueError("SEARCH_START token not found in PDL trace")

        i = int(search_idx + 1)
        while i < len(token_ids):
            tok = int(token_ids[i])
            if tok == int(self.EOS):
                break
            mask[i] = True
            i += 1

        if int(token_ids[0]) == int(self.BOS):
            mask[0] = False
        if int(token_ids[-1]) == int(self.EOS):
            mask[-1] = False

        return mask

    def build_loss_mask_prop(self, token_ids: List[int]) -> List[bool]:
        """Loss mask for View A (full PROP) traces.

        Mask pattern per step:
            STATE ... SEP       -> all False (env)
            PROP                -> False (structural)
            evidence+verdict    -> all True (model generates)
            ENDPROP             -> False (structural)
            NODE                -> True (model picks)
            MASK                -> False (env)
            COLOR OK            -> True, False
            shadow NODE COLOR   -> False, False
            DEPTH               -> True
        """
        mask = [False] * len(token_ids)
        search_idx: Optional[int] = None
        for idx, tok in enumerate(token_ids):
            if int(tok) == int(self.SEARCH_START):
                search_idx = int(idx)
                break
        if search_idx is None:
            return mask

        i = int(search_idx) + 1
        while i < len(token_ids):
            tok = int(token_ids[i])
            if tok in (int(self.SOLVED), int(self.FAILED)):
                mask[i] = True
                i += 1
                continue
            if tok == int(self.EOS):
                i += 1
                continue

            if tok == int(self.STATE):
                # Skip STATE...SEP (all False)
                i += 1
                while i < len(token_ids) and int(token_ids[i]) != int(self.SEP):
                    i += 1
                i += 1  # skip SEP
                if i >= len(token_ids):
                    break

                # PROP block
                if int(token_ids[i]) == int(self.PROP):
                    mask[i] = False  # PROP marker
                    i += 1
                    # Evidence + verdict: all True until ENDPROP
                    while i < len(token_ids) and int(token_ids[i]) != int(self.ENDPROP):
                        mask[i] = True
                        i += 1
                    if i < len(token_ids):
                        mask[i] = False  # ENDPROP marker
                        i += 1
                elif int(token_ids[i]) in (int(self.CF), int(self.OK)):
                    # View B: verdict only
                    mask[i] = True  # verdict
                    i += 1
                else:
                    pass  # unexpected, skip

                if i >= len(token_ids):
                    break

                # ACTION tokens: NODE(True) MASK(False) then COLOR(True) OK(False)
                # + shadow NODE/COLOR(False) or DEPTH(True)
                mask[i] = True  # NODE
                i += 1
                if i >= len(token_ids):
                    break
                mask[i] = False  # MASK
                i += 1
                if i >= len(token_ids):
                    break
                mask_tok = int(token_ids[i - 1])
                if mask_tok == int(self.MASK_OFFSET):  # M0000 = conflict
                    mask[i] = True  # DEPTH
                    i += 1
                else:  # valid mask = assign
                    mask[i] = True  # COLOR
                    i += 1
                    if i < len(token_ids):
                        mask[i] = False  # OK
                        i += 1
                    # Shadow assignment tokens (env-provided, always False)
                    if i < len(token_ids):
                        shadow_tok = int(token_ids[i])
                        if (
                            self.NODE_OFFSET
                            <= shadow_tok
                            < self.NODE_OFFSET + self.MAX_NODES
                        ):
                            mask[i] = False  # shadow NODE
                            i += 1
                            if i < len(token_ids):
                                mask[i] = False  # shadow COLOR
                                i += 1
                continue

            i += 1

        if token_ids and int(token_ids[0]) == int(self.BOS):
            mask[0] = False
        if token_ids and int(token_ids[-1]) == int(self.EOS):
            mask[-1] = False

        true_count = int(sum(1 for value in mask if value))
        logger.debug(
            "build_loss_mask_prop: tokens_len=%d true_tokens=%d",
            int(len(token_ids)),
            int(true_count),
        )
        return mask

    def build_loss_weight_prop(self, token_ids: List[int]) -> List[float]:
        """Return per-token loss weights for PROP traces.

        Blocker/evidence tokens (positions within PROP block after the second SEP)
        get 3x weight. All other supervised tokens get 1x weight.
        Non-supervised tokens get 0.0 weight.
        """
        bool_mask = self.build_loss_mask_prop(token_ids)
        weights = [1.0 if m else 0.0 for m in bool_mask]

        i = 0
        while i < len(token_ids):
            if int(token_ids[i]) == int(self.PROP):
                sep_count = 0
                j = i + 1
                while j < len(token_ids) and int(token_ids[j]) != int(self.ENDPROP):
                    if int(token_ids[j]) == int(self.SEP):
                        sep_count += 1
                        j += 1
                        continue
                    if sep_count >= 2 and bool_mask[j]:
                        weights[j] = 3.0
                    j += 1
                i = j + 1
            else:
                i += 1

        weighted_tokens = int(sum(1 for w in weights if w > 1.0))
        logger.debug(
            "build_loss_weight_prop: tokens_len=%d weighted_tokens=%d",
            int(len(token_ids)),
            int(weighted_tokens),
        )
        return weights

    def build_loss_mask_slim(self, token_ids: List[int]) -> List[bool]:
        """Return True for model-generated tokens in slim traces."""
        mask = [False] * int(len(token_ids))
        if not token_ids:
            return mask

        search_start_idx = None
        for idx, token_id in enumerate(token_ids):
            if int(token_id) == int(self.SEARCH_START):
                search_start_idx = int(idx)
                break
        if search_start_idx is None:
            raise ValueError("SEARCH_START token not found in slim trace")

        i = int(search_start_idx + 1)
        while i < len(token_ids):
            tok = int(token_ids[i])
            if tok in {int(self.SOLVED), int(self.FAILED)}:
                mask[i] = True
                i += 1
                continue
            if tok == int(self.EOS):
                if i != len(token_ids) - 1:
                    raise ValueError("EOS must appear at end of slim trace")
                mask[i] = False
                break

            if i + 1 >= len(token_ids):
                raise ValueError("Slim trace truncated before mask token")
            mask_tok = int(token_ids[i + 1])
            if not (self.MASK_OFFSET <= mask_tok < self.MASK_OFFSET + 16):
                raise ValueError("Expected mask token after node in slim trace")

            is_conflict = mask_tok == int(self.MASK_OFFSET)
            if is_conflict:
                if i + 2 >= len(token_ids):
                    raise ValueError("Conflict step missing depth token")
                mask[i] = False
                mask[i + 1] = False
                mask[i + 2] = True
                i += 3
                continue

            if i + 3 >= len(token_ids):
                raise ValueError("Assign step missing color/OK token")
            mask[i] = True
            mask[i + 1] = False
            mask[i + 2] = True
            mask[i + 3] = False
            i += 4

        if int(token_ids[0]) == int(self.BOS):
            mask[0] = False
        if int(token_ids[-1]) == int(self.EOS):
            mask[-1] = False

        return mask

    def build_loss_mask_sorted_slim(self, token_ids: List[int]) -> List[bool]:
        """Return True for model-generated tokens in sorted slim traces."""
        mask = [False] * int(len(token_ids))
        if not token_ids:
            return mask

        search_idx = None
        for idx, token_id in enumerate(token_ids):
            if int(token_id) == int(self.SEARCH_START):
                search_idx = int(idx)
                break
        if search_idx is None:
            return mask

        i = int(search_idx + 1)
        while i < len(token_ids):
            tok = int(token_ids[i])
            if tok in {int(self.SOLVED), int(self.FAILED)}:
                mask[i] = True
                i += 1
                continue
            if tok == int(self.EOS):
                i += 1
                continue

            if tok == int(self.STATE):
                i += 1
                while i < len(token_ids) and int(token_ids[i]) != int(self.SEP):
                    i += 1
                i += 1

                if i >= len(token_ids):
                    break

                mask[i] = True
                i += 1
                if i >= len(token_ids):
                    break

                mask[i] = False
                i += 1
                if i >= len(token_ids):
                    break

                mask[i] = True
                i += 1
                if i >= len(token_ids):
                    break

                mask[i] = True
                i += 1
                if i >= len(token_ids):
                    break

                mask[i] = True
                i += 1
                if i >= len(token_ids):
                    break

                mask[i] = True
                i += 1
                if i >= len(token_ids):
                    break

                mask[i] = True
                i += 1
                if i >= len(token_ids):
                    break

                mask[i] = True
                i += 1
                if i >= len(token_ids):
                    break

                mask[i] = True
                i += 1
                if i >= len(token_ids):
                    break

                mask[i] = True
                i += 1
                if i >= len(token_ids):
                    break

                mask[i] = True
                i += 1
                if i >= len(token_ids):
                    break

                verdict = int(token_ids[i])
                mask[i] = True
                i += 1
                if verdict == int(self.CF):
                    if i < len(token_ids):
                        mask[i] = True
                        i += 1
                continue

            i += 1

        if int(token_ids[0]) == int(self.BOS):
            mask[0] = False
        if int(token_ids[-1]) == int(self.EOS):
            mask[-1] = False

        return mask

    def build_loss_mask_rle_bucket(self, token_ids: List[int]) -> List[bool]:
        """Return True for model-generated tokens in RLE bucket traces."""
        mask = [False] * int(len(token_ids))
        if not token_ids:
            return mask

        search_idx = None
        for idx, tok in enumerate(token_ids):
            if int(tok) == int(self.SEARCH_START):
                search_idx = int(idx)
                break
        if search_idx is None:
            return mask

        i = int(search_idx + 1)
        while i < len(token_ids):
            tok = int(token_ids[i])

            if tok in (int(self.SOLVED), int(self.FAILED)):
                mask[i] = True
                i += 1
                continue

            if tok == int(self.EOS):
                i += 1
                continue

            if tok == int(self.STATE):
                i += 1
                while i < len(token_ids) and int(token_ids[i]) != int(self.SEP):
                    i += 1
                i += 1

                if i >= len(token_ids):
                    break

                mask[i] = True
                i += 1
                if i >= len(token_ids):
                    break

                mask[i] = False
                i += 1
                if i >= len(token_ids):
                    break

                mask_tok = int(token_ids[i - 1])
                if mask_tok == int(self.MASK_OFFSET):
                    mask[i] = True
                    i += 1
                else:
                    mask[i] = True
                    i += 1
                    if i >= len(token_ids):
                        break
                    mask[i] = False
                    i += 1
                continue

            i += 1

        if token_ids and int(token_ids[0]) == int(self.BOS):
            mask[0] = False
        if token_ids and int(token_ids[-1]) == int(self.EOS):
            mask[-1] = False

        return mask

    def build_loss_mask_witness(self, token_ids: List[int]) -> List[bool]:
        """Return True for model-generated tokens in witness traces."""
        mask = [False] * int(len(token_ids))
        if not token_ids:
            return mask

        search_idx = None
        for idx, token_id in enumerate(token_ids):
            if int(token_id) == int(self.SEARCH_START):
                search_idx = int(idx)
                break
        if search_idx is None:
            return mask

        i = int(search_idx + 1)
        while i < len(token_ids):
            tok = int(token_ids[i])
            if tok in {int(self.SOLVED), int(self.FAILED)}:
                mask[i] = True
                i += 1
                continue
            if tok == int(self.EOS):
                i += 1
                continue

            if tok == int(self.STATE):
                i += 1
                while i < len(token_ids) and int(token_ids[i]) != int(self.SEP):
                    i += 1
                i += 1

                if i >= len(token_ids):
                    break

                mask[i] = True
                i += 1
                if i >= len(token_ids):
                    break

                mask[i] = False
                i += 1
                if i >= len(token_ids):
                    break

                mask[i] = True
                i += 1
                if i >= len(token_ids):
                    break

                mask[i] = True
                i += 1
                if i >= len(token_ids):
                    break

                mask[i] = True
                i += 1
                if i >= len(token_ids):
                    break

                mask[i] = True
                i += 1
                if i >= len(token_ids):
                    break

                mask[i] = True
                i += 1
                if i >= len(token_ids):
                    break

                mask[i] = True
                i += 1
                if i >= len(token_ids):
                    break

                mask[i] = True
                i += 1
                if i >= len(token_ids):
                    break

                mask[i] = True
                i += 1
                if i >= len(token_ids):
                    break

                mask[i] = True
                i += 1
                if i >= len(token_ids):
                    break

                verdict = int(token_ids[i])
                mask[i] = True
                i += 1
                if verdict == int(self.CF):
                    for _ in range(3):
                        if i >= len(token_ids):
                            break
                        mask[i] = True
                        i += 1
                continue

            i += 1

        if int(token_ids[0]) == int(self.BOS):
            mask[0] = False
        if int(token_ids[-1]) == int(self.EOS):
            mask[-1] = False

        return mask

    def build_loss_mask_compressed(self, token_ids: List[int]) -> List[bool]:
        """Return True for model-generated tokens in compressed traces."""
        mask = [False] * int(len(token_ids))
        if not token_ids:
            return mask

        for idx, token_id in enumerate(token_ids):
            tok = int(token_id)
            if self.is_assign_token(tok):
                mask[idx] = True
                continue
            if self.DEPTH_OFFSET <= tok < self.DEPTH_OFFSET + self.MAX_DEPTH:
                mask[idx] = True
            if tok in {int(self.SOLVED), int(self.FAILED)}:
                mask[idx] = True

        if int(token_ids[0]) == int(self.BOS):
            mask[0] = False
        if int(token_ids[-1]) == int(self.EOS):
            mask[-1] = False

        return mask

    def build_loss_mask_interleaved(self, token_ids: List[int]) -> List[bool]:
        """Return True for model-generated tokens in interleaved traces."""
        mask = [False] * int(len(token_ids))
        if not token_ids:
            return mask

        for idx, token_id in enumerate(token_ids):
            tok = int(token_id)
            if tok == int(self.DECIDE):
                if idx + 2 >= len(token_ids):
                    raise ValueError("DECIDE missing node or SEP")
                mask[idx] = True
                mask[idx + 1] = True
                if int(token_ids[idx + 2]) != int(self.SEP):
                    raise ValueError("DECIDE must be followed by node and SEP")
                mask[idx + 2] = True
                continue

            if tok == int(self.PICK):
                if idx + 2 >= len(token_ids):
                    raise ValueError("PICK missing color or SEP")
                mask[idx] = True
                mask[idx + 1] = True
                if int(token_ids[idx + 2]) != int(self.SEP):
                    raise ValueError("PICK must be followed by color and SEP")
                mask[idx + 2] = True
                continue

            if tok == int(self.BACKJUMP):
                if idx + 2 >= len(token_ids):
                    raise ValueError("BACKJUMP missing depth or SEP")
                mask[idx] = True
                mask[idx + 1] = True
                if int(token_ids[idx + 2]) != int(self.SEP):
                    raise ValueError("BACKJUMP must be followed by depth and SEP")
                mask[idx + 2] = True
                continue

            if tok in {int(self.SOLVED), int(self.FAILED)}:
                mask[idx] = True

        if int(token_ids[0]) == int(self.BOS):
            mask[0] = False
        if int(token_ids[-1]) == int(self.EOS):
            mask[-1] = False

        return mask

    def decode_sequence(self, token_ids: List[int]) -> str:
        """Convert token sequence to human-readable string for debugging."""
        return " ".join(self.decode_token(token_id) for token_id in token_ids)
